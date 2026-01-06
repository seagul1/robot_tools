import os
import h5py
import numpy as np
from typing import List, Dict, Any, Optional, Union
import sys

# 处理导入问题
try:
    from .base import DatasetAdapter
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from base import DatasetAdapter

try:
    from ..schema_loader import load_schema, extract_visualization_fields
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from schema_loader import load_schema, extract_visualization_fields
    except ImportError:
        # 定义空实现以避免导入失败
        def load_schema(path):
            import yaml
            with open(path, "r") as f:
                return yaml.safe_load(f)
        
        def extract_visualization_fields(schema):
            return {}


class HDF5Adapter(DatasetAdapter):
    """更灵活的 HDF5 适配器，支持常见的三类文件组织：

    1) 多轨迹在顶层：文件内顶层每个 group 为一个 episode（默认旧行为）。
    2) 单文件单轨迹：root 下直接包含 `timestamps`/`images`/`robot_state`/`actions` 等 dataset。
    3) 容器式：例如 LIBERO 的 `data/demo_0...`，此时在 container group 下的子 group 为 episode。

    适配器会自动检测布局并在 `list_episodes` / `read_sequence` 中做相应处理。
    """

    def __init__(self, path: str):
        self.path = path
        self._h5 = h5py.File(path, "r")
        self._layout = self._detect_layout()
        # single-file episode id (basename without ext)
        if self._layout.get("mode") == "single":
            self._single_episode_id = os.path.splitext(os.path.basename(path))[0]
        # optional schema (dict) used to map keys
        self._schema: Optional[Dict[str, Any]] = None
        self._schema_fields: Dict[str, List[str]] = {}

    def set_schema(self, schema: Union[str, Dict[str, Any]]):
        """Provide a schema to guide which keys to load.

        Args:
            schema: either a path to a yaml schema file or a parsed dict.
        """
        if isinstance(schema, str):
            s = load_schema(schema)
        else:
            s = schema
        self._schema = s
        self._schema_fields = extract_visualization_fields(s)

    def _detect_layout(self) -> Dict[str, Any]:
        """Inspect the HDF5 file to determine its layout mode."""
        keys = list(self._h5.keys())

        # Heuristic: if root contains common dataset names -> single-file episode
        root_names = set(keys)
        common_ds = {"timestamps", "images", "robot_state", "actions", "joint_positions", "joint_angles"}
        if root_names & common_ds:
            return {"mode": "single"}

        # Container-style: a top-level group like 'data' holding demos
        if "data" in keys and isinstance(self._h5["data"], h5py.Group):
            # check if children of data are groups
            children = list(self._h5["data"].keys())
            if children and any(isinstance(self._h5["data"][c], h5py.Group) for c in children):
                return {"mode": "container", "container": "data"}

        # Default: treat top-level groups as episodes (multi-episode)
        # If top-level keys are groups, we assume multi-episode layout
        group_keys = [k for k in keys if isinstance(self._h5[k], h5py.Group)]
        if group_keys:
            return {"mode": "multi"}

        # Fallback to single
        return {"mode": "single"}

    def list_episodes(self) -> List[str]:
        mode = self._layout.get("mode")
        if mode == "single":
            return [getattr(self, "_single_episode_id", "episode_0")]
        if mode == "container":
            container = self._layout.get("container")
            return list(self._h5[container].keys())
        # multi
        return [k for k in list(self._h5.keys()) if isinstance(self._h5[k], h5py.Group)]

    def _resolve_episode_group(self, episode_id: Optional[str]):
        """Return the h5py Group corresponding to episode_id based on detected layout.

        If layout is single and episode_id is None or matches the implied id, return the file root.
        """
        mode = self._layout.get("mode")
        if mode == "single":
            # root-level dataset organization
            if episode_id is None or episode_id == getattr(self, "_single_episode_id", None):
                return self._h5
            raise KeyError(f"Single-episode HDF5; unknown episode id: {episode_id}")
        if mode == "container":
            container = self._layout.get("container")
            if episode_id is None:
                raise KeyError("Must provide episode_id for container-style HDF5")
            return self._h5[container][episode_id]
        # multi
        if episode_id is None:
            raise KeyError("Must provide episode_id for multi-episode HDF5")
        return self._h5[episode_id]

    def get_episode_meta(self, episode_id: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        grp = self._resolve_episode_group(episode_id)
        if "metadata" in grp:
            md = grp["metadata"]
            meta.update({k: md.attrs[k] for k in md.attrs.keys()})
        # also try group attrs
        for k in getattr(grp, "attrs", {}).keys():
            meta[k] = grp.attrs[k]
        return meta

    def _read_dataset(self, dataset, start: Optional[int], end: Optional[int]):
        if start is None and end is None:
            return dataset[()]
        s = slice(start, end)
        return dataset[s]

    def _try_get_group(self, grp, name: str):
        """Try several common name variants under grp to find a subgroup or dataset.

        Returns the found object or None.
        """
        # direct
        if name in grp:
            return grp[name]
        # plural/singular variants
        variants = [name + 's', name.rstrip('s')]
        for v in variants:
            if v in grp:
                return grp[v]
        # common observation containers
        for obs in ('observations', 'obs'):
            if obs in grp and name in grp[obs]:
                return grp[obs][name]
            if obs in grp and (name + 's') in grp[obs]:
                return grp[obs][name + 's']
        return None

    def _collect_images(self, grp, schema_img_keys: List[str], start, end) -> Dict[str, np.ndarray]:
        images: Dict[str, np.ndarray] = {}
        # look under several possible containers
        candidates = []
        if 'images' in grp:
            candidates.append(grp['images'])
        for obs in ('observations', 'obs'):
            if obs in grp and 'images' in grp[obs]:
                candidates.append(grp[obs]['images'])

        # if schema keys provided, try to fetch those first from candidates or top-level
        if schema_img_keys:
            for view in schema_img_keys:
                for c in candidates:
                    if view in c:
                        ds = c[view]
                        try:
                            images[view] = self._read_dataset(ds, start, end)
                        except Exception:
                            images[view] = ds[()]
                        break
                else:
                    # try top-level
                    if view in grp:
                        ds = grp[view]
                        try:
                            images[view] = self._read_dataset(ds, start, end)
                        except Exception:
                            images[view] = ds[()]

        # if still empty, load any available images from candidates
        if not images:
            for c in candidates:
                # c may be a Group or a Dataset
                if isinstance(c, h5py.Group):
                    for view in c.keys():
                        ds = c[view]
                        try:
                            images[view] = self._read_dataset(ds, start, end)
                        except Exception:
                            images[view] = ds[()]
                else:
                    # if candidate is a dataset, use its basename as view name
                    name = os.path.basename(c.name)
                    try:
                        images[name] = self._read_dataset(c, start, end)
                    except Exception:
                        images[name] = c[()]
            # fallback top-level image_* datasets
            if not images:
                for k in grp.keys():
                    if k.startswith('image'):
                        images[k] = grp[k][()]

        return images

    def read_sequence(
        self,
        episode_id: Optional[str] = None,
        fields: Optional[List[str]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Dict[str, Any]:
        grp = self._resolve_episode_group(episode_id)
        out: Dict[str, Any] = {}

        # timestamps
        if "timestamps" in grp:
            out["timestamps"] = self._read_dataset(grp["timestamps"], start, end)

        # images
        schema_img_keys = self._schema_fields.get("images") if self._schema_fields else []
        images = self._collect_images(grp, schema_img_keys, start, end)
        if images:
            out["images"] = images

        # robot_state
        robot_state: Dict[str, Any] = {}
        schema_rs_keys = self._schema_fields.get("robot_state") if self._schema_fields else []
        # try common robot_state containers
        rs_grp = None
        for candidate in ('robot_state', 'robot_states', 'states'):
            if candidate in grp:
                rs_grp = grp[candidate]
                break
        if rs_grp is None:
            # maybe under observations/obs
            for obs in ('observations', 'obs'):
                if obs in grp:
                    for candidate in ('robot_state', 'robot_states', 'states'):
                        if candidate in grp[obs]:
                            rs_grp = grp[obs][candidate]
                            break
                    if rs_grp is not None:
                        break

        if schema_rs_keys:
            for key in schema_rs_keys:
                # prefer rs_grp then top-level
                if rs_grp is not None and isinstance(rs_grp, h5py.Group) and key in rs_grp:
                    robot_state[key] = self._read_dataset(rs_grp[key], start, end)
                elif key in grp:
                    robot_state[key] = self._read_dataset(grp[key], start, end)
                else:
                    # try alternative names present in rs_grp
                    if rs_grp is not None and isinstance(rs_grp, h5py.Group):
                        # find any dataset in rs_grp that is likely equivalent (e.g., qpos ~ joint_positions)
                        for cand in rs_grp.keys():
                            if key in cand or cand in key:
                                robot_state[cand] = self._read_dataset(rs_grp[cand], start, end)
                                break
                    elif rs_grp is not None and not isinstance(rs_grp, h5py.Group):
                        # rs_grp is a dataset, read it as a candidate robot_state value
                        name = os.path.basename(rs_grp.name)
                        robot_state[name] = self._read_dataset(rs_grp, start, end)
        else:
            if rs_grp is not None:
                for key in rs_grp.keys():
                    robot_state[key] = self._read_dataset(rs_grp[key], start, end)
            else:
                for name in ['joint_positions', 'joint_angles']:
                    if name in grp:
                        robot_state[name] = self._read_dataset(grp[name], start, end)

        if robot_state:
            out['robot_state'] = robot_state

        # actions (top-level in episode)
        # actions (top-level in episode)
        actions = None
        schema_act_keys = self._schema_fields.get("actions") if self._schema_fields else []
        if schema_act_keys:
            # pick the first matching action key present in robot_state or top-level
            for key in schema_act_keys:
                if "robot_state" in grp and key in grp["robot_state"]:
                    actions = self._read_dataset(grp["robot_state"][key], start, end)
                    break
                if key in grp:
                    actions = self._read_dataset(grp[key], start, end)
                    break
        else:
            if "actions" in grp:
                actions = self._read_dataset(grp["actions"], start, end)

        if actions is not None:
            out["actions"] = actions

        return out

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass
