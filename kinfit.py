"""Joint fit of camera pose and kinematic zero offsets (lift, elbow, wrist_flex) from the
on-axis fingertip correspondences plus the table-probe heights. Analysis tool, no hardware."""
import json
import math
import sys

import numpy as np

import camera as C
import kinematics as K
import vision as V
from fd import AXIS_TOOL
from robot import load_config

np.set_printoptions(precision=2, suppress=True)
cfg = load_config()
pts = json.load(open("calib_points.json"))
probes = json.load(open("probes.json"))
axis_tool = K.tool_offset(*AXIS_TOOL)
tip_tool = K.tool_offset(*cfg["tip_offset"])
wide = {j: (-10, 10) for j in K.ARM_JOINTS}


def q_of(xyz, tool, pitch=90.0):
    q, info = K.ik(xyz, pitch, 0.0, tool=tool, limits=wide)
    assert info["ok"], info
    return q


Q = [np.array(p["q"][0]) if "q" in p else q_of(p["xyz"], axis_tool, p.get("pitch") or 90.0) for p in pts]
UV = np.array([p["uv"] for p in pts])
QP = [q_of([p["xy"][0], p["xy"][1], p["z"]], tip_tool) for p in probes]


def points(delta):
    d = np.array([0.0, delta[0], delta[1], delta[2], 0.0])
    xyz = np.array([K.fk(q + d, axis_tool)[:3, 3] for q in Q])
    zt = np.array([K.fk(q + d, tip_tool)[2, 3] for q in QP])
    return xyz, zt


def cost(delta, f=1400.0, w_table=3000.0, full=False):
    xyz, zt = points(delta)
    try:
        cam = C.fit_camera(xyz, UV, f_range=(f, f))
    except Exception:
        return 1e9
    res = cam.project(xyz) - UV
    flat = (zt - zt.mean()) * w_table  # px per metre equivalent
    c = float(np.sum(res ** 2) + np.sum(flat ** 2))
    if full:
        return c, cam, res, zt
    return c


if __name__ == "__main__":
    f = float(sys.argv[1]) if len(sys.argv) > 1 else 1400.0
    c0, cam0, res0, zt0 = cost(np.zeros(3), f, full=True)
    print(f"f={f}  delta=0: rms {math.sqrt(np.mean(np.sum(res0**2, axis=1))):.2f} px, table z (mm) {(zt0 * 1000).round(1)}, res {np.linalg.norm(res0, axis=1).round(1)}")
    best, bc = None, 1e18
    for start in ([0, 0, 0], [0.05, 0, 0], [-0.05, 0, 0], [0, 0.05, 0], [0, -0.05, 0], [0, 0, 0.08], [0, 0, -0.08]):
        d, c = V.nelder_mead(lambda x: cost(x, f), np.array(start, dtype=float), [0.03, 0.03, 0.03], iters=400, tol=1e-6)
        if c < bc:
            best, bc = d, c
    c, cam, res, zt = cost(best, f, full=True)
    print(f"best delta (deg) lift/elbow/wrist: {np.degrees(best).round(2)}  rms {math.sqrt(np.mean(np.sum(res**2, axis=1))):.2f} px")
    print("table z after (mm):", (zt * 1000).round(1), " camera centre:", cam.center.round(3))
    print("residuals px:", np.linalg.norm(res, axis=1).round(1))
