#!/home/jetson/miniconda/envs/lerobot/bin/python
"""At which commanded height does the pen tip touch the paper?  (real arm unless --sim)

The pen draws short strokes while it comes down slowly ("ramp strokes"); the camera then shows where along
each stroke the ink begins, which is the height of first contact at that place. One coarse search at the top
centre, then seven more strokes around the drawing box; a plane is fitted through the eight heights and saved
in pen_state.json. The strokes lie just outside the drawing box. The arm ends at the side, torque ON.
"""
import argparse
import json
import sys
import time

import penlib as P
from penlib import K, SafetyError, camera, cv2, np

LEN = 0.016      # stroke length
LINE_W = 0.9     # mm: the ink line is about this wide, which lengthens what the camera sees


def ramp(arm, x, ya, yb, z_hi, z_lo, v=14.0):
    """Pen tip from (x, ya, z_hi) down to (x, yb, z_lo) in one straight stroke, then straight up."""
    P.tip_path(arm, [(x, ya, z_hi + 0.012)], 45.0, settle=0.15)
    P.tip_path(arm, [(x, ya, z_hi)], 20.0, settle=0.3)
    P.tip_path(arm, [(x, yb, z_lo)], v, settle=0.3, step=0.002)
    P.tip_path(arm, [(x, yb, z_lo + 0.012)], 25.0, settle=0.1)


def contact_z(m, z_hi, z_lo):
    """Height of first contact from the inked share of a ramp stroke."""
    share = min(1.0, max(0.0, (m["ink_mm"] - LINE_W) / (LEN * 1000.0)))
    return z_lo + share * (z_hi - z_lo), share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--z0", type=float, default=None, help="skip the coarse search: contact height at the top centre (m)")
    a = ap.parse_args()
    cam = camera.CameraModel.load()
    arm = P.make_arm(a.sim).open()
    zt = float(arm.cfg["table_z"])
    out = {"strokes": []}

    def photo(name):
        return None if a.sim else P.snap_paper(name)[1]

    try:
        if not arm.is_torqued():
            arm.torque_on()
        if hasattr(arm, "apply_profile"):
            arm.apply_profile(None)
        P.go_aside(arm)
        before = photo("z_before")
        z0 = 0.0015 if a.z0 is None else a.z0
        s = P.HALF
        places = [("top_left", P.X_TOP, P.YC + 0.045), ("top_right", P.X_TOP, P.YC - 0.045),
                  ("mid_left", P.XC, P.YC + s + 0.012), ("mid_right", P.XC, P.YC - s - 0.012),
                  ("bot_left", P.X_BOT, P.YC + 0.045), ("bot_mid", P.X_BOT, P.YC), ("bot_right", P.X_BOT, P.YC - 0.045)]
        todo = [(n, x, y, z0 + 0.002, z0 - 0.002) for n, x, y in places]
        for rnd in range(3):
            if not todo:
                break
            for n, x, y, z_hi, z_lo in todo:
                ramp(arm, x, y - LEN / 2, y + LEN / 2, z_hi, z_lo)
            P.go_aside(arm)
            after = photo("z_fine%d" % rnd)
            again = []
            for n, x, y, z_hi, z_lo in todo:
                if a.sim:
                    m = {"found": True, "ink_mm": 8.0 + LINE_W, "n": 0}
                else:
                    m = P.ink_on_segment(before, after, cam, (x, y - LEN / 2), (x, y + LEN / 2), zt)
                rec = {"name": n, "round": rnd, "x": x, "y": y, "z_hi": z_hi, "z_lo": z_lo, **m}
                if not m["found"]:
                    again.append((n, x, y, z_lo + 0.0003, z_lo - 0.0027))          # deeper, same line (nothing was drawn)
                else:
                    z, share = contact_z(m, z_hi, z_lo)
                    rec.update(z_contact=z, share=share)
                    if share > 0.93:                                                 # inked from the start: higher, on a fresh line
                        again.append((n, x + (0.0035 if x > P.XC + 0.01 else -0.0035), y, z_hi + 0.0027, z_hi - 0.0003))
                        rec["redo"] = "higher"
                print(json.dumps(rec), flush=True)
                out["strokes"].append(rec)
            if after is not None:
                before = after
            todo = again
        # ---- plane through the contact heights
        pts = {}
        for r in out["strokes"]:
            if r.get("z_contact") is not None and not r.get("redo"):
                pts["top_mid" if r["name"].startswith("coarse") else r["name"]] = (r["x"], r["y"], r["z_contact"])
        A = np.array([[1.0, x - P.XC, y - P.YC] for x, y, _ in pts.values()])
        zz = np.array([z for _, _, z in pts.values()])
        coef = np.linalg.lstsq(A, zz, rcond=None)[0]
        res = zz - A @ coef
        out.update(ok=len(pts) >= 4, plane=coef.tolist(), n=len(pts),
                   residual_mm={k: round(float(v) * 1000, 2) for k, v in zip(pts.keys(), res)},
                   z_centre_mm=round(float(coef[0]) * 1000, 2), slope_x_mm_per_100=round(float(coef[1]) * 100000, 2),
                   slope_y_mm_per_100=round(float(coef[2]) * 100000, 2))
        if not a.sim and out["ok"]:
            P.save_state(plane=coef.tolist(), plane_pts={k: list(v) for k, v in pts.items()}, pitch=P.PITCH, z_strokes=out["strokes"],
                         centre=[P.XC, P.YC], half=P.HALF)
        print(json.dumps({k: v for k, v in out.items() if k != "strokes"}))
        return 0 if out["ok"] else 1
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
