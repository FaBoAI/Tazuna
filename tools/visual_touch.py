#!/home/jetson/miniconda/envs/lerobot/bin/python
"""Height self-check: at which kinematic height does the fingertip FIRST TOUCH the table?

    tools/visual_touch.py 0.18,-0.06 0.22,-0.07 0.26,-0.07 0.20,0.0        (about 30 s per place, real robot)

Why this exists (2026-11-11): `fd probe` cannot find the first touch on this arm. When the fingertip
meets a surface the measured joints keep following the plan and the fingertip slides radially outward,
so the probe reported heights 3-18 mm too deep. The camera can see the touch instead:

  the closed fingertip is stepped down 2 mm at a time (`fd tip --straight --contact`), photographed after
  each step (`fd snap`), the white gripper is tracked between photos, and its image motion is split, with
  the camera model, into "straight down" and "radially outward". While the fingertip is in the air the
  photos show the same descent as the joints; from the touch on they show no descent (only sliding).

A healthy calibration gives the same height (config table_z, within about +-1.5 mm) at every place.
A height that changes with the distance from the base means a joint zero is off (it was the shoulder_lift
zero: see config.json "calibration_note").

Limits: the gripper must be seen against something that is not white (not above white objects), the
places must be free of objects and reachable straight down (r < about 0.28 m), and the camera model must
be roughly right (it only supplies the image directions of "down" and "outward").
The arm is parked at the end unless --no-park is given.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import camera  # noqa: E402

FD = os.path.join(HERE, "fd")


def fd(*a):
    p = subprocess.run([FD, *a], capture_output=True, text=True, timeout=120)
    try:
        return json.loads(p.stdout)
    except Exception:
        return {"ok": False, "message": (p.stdout + p.stderr)[:300]}


def gripper_mask(path, c):
    """Soft mask (0..1) of the white blob nearest to pixel c (the expected fingertip), in a crop that
    follows c. None when no white blob is there."""
    mn = cv2.imread(path).astype(np.float32).min(axis=2)
    x0, y0 = max(0, int(c[0]) - 110), max(0, int(c[1]) - 190)
    roi = mn[y0:int(c[1]) + 70, x0:int(c[0]) + 110]
    m = np.clip((roi - 110.0) / 60.0, 0, 1)
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0.5).astype(np.uint8), 8)
    best, bd = None, 1e9
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 400:
            continue
        ys, xs = np.where(lab == k)
        d = np.min(np.hypot(xs - (c[0] - x0), ys - (c[1] - y0)))
        if d < bd:
            best, bd = k, d
    if best is None:
        return None
    return m * (cv2.dilate((lab == best).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)


def image_shift(a, b):
    """Sub-pixel displacement (dx, dy) of the lowest 70 rows of the gripper mask from a to b."""
    ys = np.where(a.max(axis=1) > 0.5)[0]
    yb = ys.max()
    t0, t1 = max(0, yb - 70), min(a.shape[0], yb + 8)
    xs = np.where(a[t0:t1].max(axis=0) > 0.5)[0]
    xa, xb = max(0, xs.min() - 6), min(a.shape[1], xs.max() + 6)
    res = cv2.matchTemplate(b, a[t0:t1, xa:xb], cv2.TM_CCORR_NORMED)
    m = np.full(res.shape, -1.0, np.float32)
    ya, yb2 = max(0, t0 - 14), min(res.shape[0], t0 + 15)
    xa2, xb2 = max(0, xa - 14), min(res.shape[1], xa + 15)
    m[ya:yb2, xa2:xb2] = res[ya:yb2, xa2:xb2]
    _, _, _, (px, py) = cv2.minMaxLoc(m)

    def sub(v, i):
        if 0 < i < len(v) - 1 and v[i - 1] > -1 and v[i + 1] > -1:
            den = v[i - 1] - 2 * v[i] + v[i + 1]
            return 0.5 * (v[i - 1] - v[i + 1]) / den if den < 0 else 0.0
        return 0.0

    return px + sub(m[py, :], px) - xa, py + sub(m[:, px], py) - t0


def touch_at(cam, x, y, z_start, z_min, tag, log=print):
    """Step down at (x, y) from z_start; returns (first-touch height in m or None, records)."""
    recs = []
    r = fd("tip", str(x), str(y), "%.4f" % (z_start + 0.012), "--pitch", "90", "--roll", "0", "--speed", "20")
    if not r.get("ok"):
        log("  cannot go above (%.3f, %.3f): %s" % (x, y, (r.get("message") or "")[:140]))
        return None, recs
    rad = np.array([x, y, 0.0]) / np.hypot(x, y)
    p0 = cam.project([[x, y, 0.0]])[0]
    A = np.array([cam.project([[x, y, -0.001]])[0] - p0, cam.project([np.array([x, y, 0.0]) + 0.001 * rad])[0] - p0]).T
    z, prev, hits, zc = z_start, None, 0, None
    while z >= z_min - 1e-9:
        r = fd("tip", str(x), str(y), "%.4f" % z, "--pitch", "90", "--roll", "0", "--straight", "--speed", "6", "--contact")
        s = fd("snap", "--name", "%s_%03d_%s%03d_%s%02d" % (tag, round(x * 1000), "m" if y < 0 else "p", round(abs(y) * 1000),
                                                         "m" if z < 0 else "p", round(abs(z) * 1000)), "--plain")
        tip = s.get("tip_xyz")
        im = gripper_mask(s["raw"], cam.project([tip])[0]) if (s.get("raw") and tip) else None
        if im is not None and prev is not None and im.shape != prev[0].shape:
            im = None
        rec = {"x": x, "y": y, "z_cmd": z, "ok": bool(r.get("ok")), "tip": tip, "q_deg": s.get("q_deg"), "raw": s.get("raw"),
               "load": r.get("load"), "err": r.get("joint_err_deg"), "msg": r.get("message")}
        if prev is not None and im is not None and tip:
            dx, dy = image_shift(prev[0], im)
            c0, c1 = cam.project([prev[2]])[0], cam.project([tip])[0]  # the crop follows the expected pixel
            dx += int(c1[0]) - int(c0[0])
            dy += int(c1[1]) - int(c0[1])
            down, outw = np.linalg.solve(A, [dx, dy])
            dz = (prev[1] - tip[2]) * 1000
            touch = (dz > 0.8 and down < 0.4 * dz) or (dz <= 0.8 and abs(outw) > 0.8 and down < 0.3)
            if touch and zc is None:
                zc = prev[1] - max(float(down), 0.0) / 1000.0
            if not touch:
                zc = None
            hits = hits + 1 if touch else 0
            rec.update(image_px=[float(dx), float(dy)], down_mm=float(down), outward_mm=float(outw), joints_down_mm=float(dz),
                       touch=bool(touch))
            log("  z %+6.1f mm: joints went down %4.1f, photo shows down %+5.2f / outward %+5.2f mm %s" % (
                tip[2] * 1000, dz, down, outw, "TOUCH" if touch else ""))
        recs.append(rec)
        if im is not None and tip:
            prev = (im, tip[2], tip)
        if not r.get("ok") or hits >= 2:
            break
        z -= 0.002
    fd("tip", str(x), str(y), "%.4f" % (z_start + 0.022), "--pitch", "90", "--roll", "0", "--straight", "--speed", "10", "--contact")
    return (zc if hits >= 2 else None), recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("points", nargs="+", help="x,y in metres (free table, reachable straight down)")
    ap.add_argument("--above", type=float, default=0.010, help="start this far above config table_z (m)")
    ap.add_argument("--below", type=float, default=0.005, help="give up this far below config table_z (m)")
    ap.add_argument("--no-park", action="store_true")
    ap.add_argument("--out", help="write the records to this JSON file")
    a = ap.parse_args()
    cfg = fd("config")
    table_z = float(cfg.get("table_z"))
    cam = camera.CameraModel.load()
    pts = [tuple(float(v) for v in p.split(",")) for p in a.points]
    out, found = [], []
    try:
        r = fd("ready")
        if not r.get("ok"):
            print(json.dumps({"ok": False, "message": "ready failed: %s" % (r.get("message") or r)}))
            return 2
        fd("grip", "0")
        for (x, y) in pts:
            print("(%.3f, %+.3f):" % (x, y), flush=True)
            zc, recs = touch_at(cam, x, y, table_z + a.above, table_z - a.below, "vt", log=lambda s: print(s, flush=True))
            out.append({"x": x, "y": y, "first_touch_z": zc, "steps": recs})
            if zc is not None:
                found.append(zc)
            print("  => first touch at z = %s" % ("not found" if zc is None else "%+.1f mm" % (zc * 1000)), flush=True)
    finally:
        fd("ready")
        if not a.no_park:
            fd("park")
    res = {"ok": bool(found), "table_z_config_mm": round(table_z * 1000, 1),
           "first_touch_mm": [None if o["first_touch_z"] is None else round(o["first_touch_z"] * 1000, 1) for o in out],
           "mean_mm": None if not found else round(float(np.mean(found)) * 1000, 1),
           "spread_mm": None if len(found) < 2 else round(float(np.ptp(found)) * 1000, 1)}
    if found:
        dev = float(np.mean(found)) - table_z
        res["verdict"] = ("height calibration looks right" if abs(dev) < 0.002 and (len(found) < 2 or np.ptp(found) < 0.003)
                          else "height calibration is off: mean differs from table_z by %+.1f mm, spread %.1f mm" % (
                              dev * 1000, (np.ptp(found) * 1000 if len(found) > 1 else 0.0)))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"result": res, "places": out, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
    print(json.dumps(res))
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
