"""Schema loader utilities for visualizer.

This module reads YAML schema files (in `toolkits/visualizer/schema/`) and
extracts keys/fields that should be visualized (vision sensors, proprioception,
actions, tactile, timestamps, etc.).

Functions provided:
- `load_schema(path)` -> dict: parse a YAML file into a Python dict.
- `load_schemas_from_dir(dir_path)` -> Dict[str, dict]: load all .yaml files in a dir.
- `extract_visualization_fields(schema_dict)` -> dict: extract lists of keys for
  images, robot_state and actions to be used by `HDF5Adapter`/viewer.
"""
from typing import Dict, Any, List
import os
import yaml


def load_schema(path: str) -> Dict[str, Any]:
    """Load a YAML schema file and return it as a dict.

    Args:
        path: Path to a .yaml file.
    Returns:
        Parsed YAML as a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schemas_from_dir(dir_path: str) -> Dict[str, Dict[str, Any]]:
    """Load all .yaml files in a directory.

    Returns a mapping from filename -> parsed schema.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(dir_path):
        return out
    for name in os.listdir(dir_path):
        if not name.endswith(".yaml") and not name.endswith(".yml"):
            continue
        full = os.path.join(dir_path, name)
        try:
            out[name] = load_schema(full)
        except Exception:
            # ignore parse errors but continue
            out[name] = {}
    return out


def _list_or_none(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str) and v.lower() in ("none", "null"):
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    # if provided as comma-separated string
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


def extract_visualization_fields(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """Given a parsed schema dict, extract keys for visualization.

    The returned dict will contain:
      - `images`: list of image view keys (e.g. camera names)
      - `robot_state`: list of robot state keys (e.g. joint_positions, end_effector_pose)
      - `actions`: list of action keys
      - `timestamps`: mapping keys that act as timestamps (if listed)

    This function uses common keys found in the project's example YAML format.
    """
    out: Dict[str, List[str]] = {"images": [], "robot_state": [], "actions": [], "timestamps": []}

    handled_sensors = set()

    # vision sensors
    vs = schema.get("vision_sensor") or {}
    out_images = _list_or_none(vs.get("key"))
    if out_images:
        out["images"] = out_images
    vts = vs.get("timestamp")
    if vts not in (None, "none"):
        out["timestamps"].append(str(vts))
    handled_sensors.add("vision_sensor")

    # proprioception -> robot_state (extend)
    prop = schema.get("proprioception_sensor") or {}
    pkeys = _list_or_none(prop.get("key"))
    if pkeys:
        out["robot_state"].extend(pkeys)
    pts = prop.get("timestamp")
    if pts not in (None, "none"):
        out["timestamps"].append(str(pts))
    handled_sensors.add("proprioception_sensor")

    # actions
    act = schema.get("action") or {}
    akeys = _list_or_none(act.get("key"))
    if akeys:
        out["actions"] = akeys
    ats = act.get("timestamp")
    if ats not in (None, "none"):
        out["timestamps"].append(str(ats))

    # dynamic: handle any other sensor entries like force_sensor, imu_sensor, etc.
    for name, val in schema.items():
        if not isinstance(name, str):
            continue
        if name.endswith("_sensor") and name not in handled_sensors:
            sensor_keys = _list_or_none((val or {}).get("key"))
            if sensor_keys:
                sensor_short = name[:-7]  # strip '_sensor'
                out[sensor_short] = sensor_keys
            sensor_ts = (val or {}).get("timestamp")
            if sensor_ts not in (None, "none"):
                out["timestamps"].append(str(sensor_ts))

    # dedupe lists while preserving order, and remove empty entries
    cleaned: Dict[str, List[str]] = {}
    for k, lst in out.items():
        if not lst:
            continue
        seen = set()
        deduped = []
        for item in lst:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        cleaned[k] = deduped

    return cleaned


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("schema", help="path to schema yaml (or directory)")
    args = p.parse_args()
    if os.path.isdir(args.schema):
        schemas = load_schemas_from_dir(args.schema)
        for name, s in schemas.items():
            print("==", name)
            print(extract_visualization_fields(s))
    else:
        s = load_schema(args.schema)
        print(extract_visualization_fields(s))
