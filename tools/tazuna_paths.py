"""Where the parts of TAZUNA are.

The harness modules live in tools/<feature>/ and keep their plain names (`import robot`, `import kinematics`).
Importing this module puts every feature folder on sys.path, so a module can be started from anywhere:

    import os, sys
    sys.path.insert(0, os.path.join(<project folder>, "tools"))
    import tazuna_paths

It also says where the files of one rig and the files that the harness writes are:

    ROOT          the project folder (fd, CLAUDE.md, README.md)
    CONFIG_DIR    config/   the measurements of one rig: config.json, camera_model.json, calib_points.json, probes.json
    ROOT/runs/, ROOT/backups/, ROOT/tool_active.json   written by the harness, never in the repository
"""
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
CONFIG_DIR = os.path.join(ROOT, "config")
# feature folders that hold importable modules (tools/prototypes/ holds records and is not on the path)
FEATURES = ("cli", "motion", "ik", "vision", "pick", "takeout", "draw", "sim", "calib", "publish")

for _name in FEATURES:
    _path = os.path.join(TOOLS, _name)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _name, _path
