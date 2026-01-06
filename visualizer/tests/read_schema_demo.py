"""Demo for schema loader: reads the bundled hdf5_example.yaml and prints fields."""
import os
import sys
# ensure repo root is on sys.path so absolute imports work when running script directly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from toolkits.visualizer.schema_loader import load_schema, extract_visualization_fields


def main():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(pkg_dir, "schema", "hdf5_example.yaml")
    if not os.path.exists(schema_path):
        print("Schema example not found:", schema_path)
        return
    s = load_schema(schema_path)
    fields = extract_visualization_fields(s)
    print("Loaded schema:", schema_path)
    print("Visualization fields:")
    for k, v in fields.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
