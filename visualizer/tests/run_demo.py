"""Demo runner for the visualizer package.

This script locates the bundled example HDF5 and launches the simple viewer.
"""
import os
import sys

from .simple_viewer import viewer_main


def main():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    example_path = os.path.join(pkg_dir, "test_example", "episode_05.hdf5")
    if not os.path.exists(example_path):
        print("Bundled example HDF5 not found:", example_path)
        print("You can run the viewer against your own file: python toolkits/visualizer/simple_viewer.py --file /path/to/data.h5")
        sys.exit(2)

    print("Running visualizer demo using:", example_path)
    viewer_main(example_path)


if __name__ == "__main__":
    main()
