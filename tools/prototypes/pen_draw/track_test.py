#!/home/jetson/miniconda/envs/lerobot/bin/python
"""How well does the pen tip follow a planned path?  In the AIR (3 cm above the paper): a circle and a square
at a few speeds, joint angles logged every control cycle, pen-tip path computed from them."""
import json
import math
import sys

import penlib as P
from penlib import K, np

sim = "--sim" in sys.argv
arm = P.make_arm(sim).open()
zt = float(arm.cfg["table_z"])
Z = zt + 0.03
cx, cy = P.XC, P.YC


def shape(kind):
    if kind == "circle":
        return [(cx + 0.008 * math.cos(a), cy + 0.008 * math.sin(a), Z) for a in np.linspace(0, 2 * math.pi, 41)]
    s = 0.012
    return [(cx - s, cy - s, Z), (cx - s, cy + s, Z), (cx + s, cy + s, Z), (cx + s, cy - s, Z), (cx - s, cy - s, Z)]


out = []
try:
    if not arm.is_torqued():
        arm.torque_on()
    tool = arm.tip_tool()
    for kind, v, freeze in (("circle", 6.0, False), ("circle", 10.0, False), ("circle", 15.0, False), ("square", 10.0, False),
                            ("square", 15.0, False)):
        pts = shape(kind)
        P.go_tip(arm, pts[0], P.PITCH, 0.0, 30.0, min_tip_z=0.02, settle=True)
        q0 = arm.read_q()
        wps = arm.plan_tip_path([(np.array(p), P.PITCH, 0.0) for p in pts[1:]], q_start=q0, step_m=0.0015)
        length = sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))
        dq = float(np.degrees(np.abs(np.diff(np.array(wps), axis=0)).sum(axis=0).max()))
        speed = max(1.5, 1.3 * dq * (v / 1000.0) / length)
        log = []

        def mon(t, q_plan, q_meas, *rest):
            log.append((t, K.fk(q_plan, tool)[:3, 3].copy(), K.fk(q_meas, tool)[:3, 3].copy()))
            return False

        ki = arm.cfg.get("ki_per_s", 2.0)
        if freeze:
            arm.cfg["ki_per_s"] = 0.0
        try:
            r = arm.run_path(wps, speed, allow_contact=True, settle=0.8, monitor=mon)
        finally:
            arm.cfg["ki_per_s"] = ki
        T = np.array([l[0] for l in log]); Pp = np.array([l[1] for l in log]); Pm = np.array([l[2] for l in log])
        # distance of each measured point to the planned polyline (dense plan samples)
        d = np.array([np.min(np.linalg.norm(Pp[:, :2] - m[:2], axis=1)) for m in Pm])
        # lag: time shift that best aligns measured with planned
        best = (1e9, 0.0)
        for lag in np.arange(0.0, 1.6, 0.05):
            idx = np.searchsorted(T, T - lag)
            e = np.linalg.norm(Pm[:, :2] - Pp[np.clip(idx, 0, len(T) - 1), :2], axis=1)[int(len(T) * 0.2):int(len(T) * 0.9)].mean()
            if e < best[0]:
                best = (e, lag)
        rec = {"kind": kind, "v_mm_s": v, "freeze": freeze, "duration_s": round(float(T[-1]), 1), "n": len(log),
               "path_err_mm_mean": round(float(d.mean() * 1000), 2), "path_err_mm_p95": round(float(np.percentile(d, 95) * 1000), 2),
               "path_err_mm_max": round(float(d.max() * 1000), 2), "lag_s": round(float(best[1]), 2),
               "err_after_lag_mm": round(float(best[0] * 1000), 2),
               "z_err_mm_range": [round(float((Pm[:, 2] - Pp[:, 2]).min() * 1000), 2), round(float((Pm[:, 2] - Pp[:, 2]).max() * 1000), 2)],
               "end_err_mm": round(float(np.linalg.norm(Pm[-1] - Pp[-1]) * 1000), 2)}
        print(json.dumps(rec), flush=True)
        out.append(rec)
        np.save("/tmp/tazuna_scratch/pen/track_%s_%d_%d.npy" % (kind, int(v), int(freeze)),
                np.hstack([T[:, None], Pp, Pm]))
    P.go_aside(arm)
finally:
    arm.close()
