#!/home/jetson/miniconda/envs/lerobot/bin/python
"""Roughly where is the pen tip in the gripper frame?  (real arm unless --sim)

The gripper is held at about ten poses well above the paper and photographed (short exposure); the lowest
point of the dark pen below the gripper is taken as its tip, and the pen-tip point in gripper_frame_link axes
is the one whose projections fit those pixels best. This is only a ROUGH value (the camera model is calibrated
on the table plane and is off by up to a centimetre for points held in the air): cal_hover.py refines it on
the paper. Writes config_pen.json. The arm ends at the side pose, torque ON.
"""
import argparse
import json
import math
import sys
import time

import penlib as P
from penlib import K, SafetyError, camera, cv2, np

POSES = [  # (name, gripper_frame_link origin x, y, z [m], gripper pitch, wrist roll [deg])
    ("a", 0.29, 0.00, 0.120, 14, 0),
    ("b", 0.27, 0.09, 0.120, 10, 0),
    ("c", 0.27, -0.09, 0.120, 10, 0),
    ("d", 0.31, 0.00, 0.120, 3, 0),
    ("e", 0.29, 0.00, 0.120, 30, 0),
    ("f", 0.29, 0.00, 0.120, 14, 26),
    ("g", 0.29, 0.00, 0.120, 18, -26),
    ("h", 0.31, 0.06, 0.125, 20, 12),
    ("i", 0.31, -0.06, 0.125, 20, -12),
    ("j", 0.29, 0.03, 0.120, 45, 0),
    ("k", 0.28, -0.03, 0.120, 55, 0),
]


def fit(obs, cam, p0):
    """obs: [(q, uv)] -> pen tip p (gripper_frame_link axes) minimising the reprojection error."""
    p = np.array(p0, float)
    Ts = [K.fk(q) for q, _ in obs]
    uv = np.array([o[1] for o in obs], float)

    def res(pp):
        pts = [(T @ np.array([*pp, 1.0]))[:3] for T in Ts]
        return (cam.project(pts) - uv).ravel()

    J = None
    for _ in range(40):
        r = res(p)
        J = np.zeros((len(r), 3))
        for k in range(3):
            d = np.zeros(3)
            d[k] = 1e-5
            J[:, k] = (res(p + d) - r) / 1e-5
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        p = p + step
        if np.linalg.norm(step) < 1e-7:
            break
    r = res(p).reshape(-1, 2)
    cov = np.linalg.inv(J.T @ J) * (float((r ** 2).sum()) / max(1, r.size - 3))
    return p, r, np.sqrt(np.diag(cov))


def frame_min_z(q_from, q_to, n: int = 40) -> float:
    return float(min(K.fk(q_from + (q_to - q_from) * k / n)[2, 3] for k in range(n + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--speed", type=float, default=20.0)
    a = ap.parse_args()
    st = P.load_state()
    p0 = st.get("tip_offset", [0.03, 0.01, 0.03])
    cam = camera.CameraModel.load()
    arm = P.make_arm(a.sim).open()
    obs, log = [], []
    try:
        if not arm.is_torqued():
            arm.torque_on()
        if hasattr(arm, "apply_profile"):
            arm.apply_profile(None)
        q_side = np.radians(st.get("aside_q_deg", [43.15, -20.31, 10.46, 63.28, -0.05]))
        for name, x, y, z, pitch, roll in POSES:
            q_now = arm.read_q()
            try:
                q_t = arm.solve(np.array([x, y, z]), pitch, roll, q_seed=q_now, use_tip=False)
                if frame_min_z(q_now, q_t) < 0.095:
                    raise SafetyError("the gripper would come below 9.5 cm on the way")
                arm.run_path([q_now, q_t], a.speed, settle=1.0)
            except SafetyError as e:
                print("skip %s: %s" % (name, str(e)[:160]), flush=True)
                continue
            time.sleep(0.0 if a.sim else 0.4)
            q_m = arm.read_q()
            anchor = cam.project([K.fk(q_m)[:3, 3]])[0]
            rec = {"name": name, "q_deg": np.degrees(q_m).round(3).tolist(), "anchor_uv": anchor.round(1).tolist()}
            if not a.sim:
                path, img = P.snap_paper("tip_" + name)
                uv = P.find_pen_generic(img, anchor)
                rec.update(photo=path, uv=None if uv is None else [round(uv[0], 1), round(uv[1], 1)])
                if uv is not None:
                    obs.append((q_m, uv))
            print(json.dumps(rec), flush=True)
            log.append(rec)
        q_now = arm.read_q()
        if frame_min_z(q_now, q_side) < 0.095:
            print("NOT going to the side pose: the gripper would come low on the way")
        else:
            arm.run_path([q_now, q_side], a.speed, settle=0.6)
    finally:
        arm.close()
    if a.sim:
        print("sim: %d poses reachable" % len(log))
        return 0
    P.save_state(tip_obs=log, aside_q_deg=np.degrees(q_side).round(3).tolist())
    if len(obs) < 5:
        print(json.dumps({"ok": False, "message": "pen tip found in only %d photos" % len(obs)}))
        return 1
    p, r, sd = fit(obs, cam, p0)
    e = np.hypot(r[:, 0], r[:, 1])
    keep = e < max(25.0, 3.0 * float(np.median(e)))
    if keep.sum() >= 5 and not keep.all():
        p, r, sd = fit([o for o, k in zip(obs, keep) if k], cam, p)
        e = np.hypot(r[:, 0], r[:, 1])
    res = {"ok": True, "tip_offset": p.round(5).tolist(), "sd_mm": (sd * 1000).round(2).tolist(), "n": int(len(e)),
           "residual_px": e.round(1).tolist(), "rms_px": round(float(np.sqrt((e ** 2).mean())), 2)}
    P.write_pen_config(p)
    P.save_state(tip_offset=p.round(5).tolist(), tip_fit=res)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
