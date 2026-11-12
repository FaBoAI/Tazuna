#!/home/jetson/miniconda/envs/lerobot/bin/python
"""Measured map "commanded pen position -> where the ink really lands" over the drawing area.

A 3 x 3 grid of small marks (3 mm dashes) is drawn at commanded places, photographed, each mark is found on the
table plane with the camera, and a smooth map (affine, or quadratic when that fits clearly better) is fitted.
draw.py inverts it. Saved in pen_state.json ("map").   grid_cal.py [--sim] [--z PLAN_Z] [--analyse PHOTO]
"""
import argparse
import json
import sys
import time

import penlib as P
from penlib import K, camera, cv2, np

GX = [P.XC - 0.0575, P.XC, P.XC + 0.0575]
GY = [P.YC - 0.060, P.YC, P.YC + 0.060]
DASH = 0.003


def feats(xy, quad):
    x, y = xy[:, 0] - P.XC, xy[:, 1] - P.YC
    cols = [np.ones_like(x), x, y]
    if quad:
        cols += [x * x, x * y, y * y]
    return np.stack(cols, axis=1)


def new_blobs(before, after, cam, zt, roi):
    x0, y0, x1, y1 = roi
    a = (P.ink_map(after)[y0:y1, x0:x1] > 28).astype(np.uint8)
    if before is not None:
        b = (P.ink_map(before)[y0:y1, x0:x1] > 14).astype(np.uint8)
        a[cv2.dilate(b, np.ones((9, 9), np.uint8)) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(a, 8)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 10:
            continue
        ys, xs = np.where(lab == k)
        pts = np.array([cam.to_plane(np.array([float(u + x0), float(v + y0)]), zt)[:2] for u, v in zip(xs, ys)])
        out.append({"area": int(st[k, cv2.CC_STAT_AREA]), "xy": pts.mean(axis=0), "px": [float(xs.mean() + x0), float(ys.mean() + y0)]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--z", type=float, default=-0.0030, help="planned height of the marks (m)")
    ap.add_argument("--premap", action="store_true", help="aim the marks with the map already stored (first guess)")
    ap.add_argument("--skip", action="store_true", help="leave out the centre and the bottom-centre mark (ink is there already)")
    a = ap.parse_args()
    cam = camera.CameraModel.load()
    arm = P.make_arm(a.sim).open()
    zt = float(arm.cfg["table_z"])
    marks = []
    try:
        if not arm.is_torqued():
            arm.torque_on()
        P.go_aside(arm)
        before = None if a.sim else P.snap_paper("grid_before")[1]
        import draw
        prev = P.load_state().get("map") if a.premap else None
        want = [(x, y) for x in GX for y in GY if not (a.skip and (round(x, 4), round(y, 4)) in
                                                       {(round(GX[1], 4), round(GY[1], 4)), (round(GX[2], 4), round(GY[1], 4))})]
        cmdl = draw.inverse_map(np.array(want), prev) if prev else np.array(want)
        for (x, y) in cmdl:
            if True:
                P.tip_path(arm, [(x, y - DASH / 2, a.z + P.LIFT)], 50.0, settle=True)
                P.tip_path(arm, [(x, y - DASH / 2, a.z)], 12.0, settle=False)
                time.sleep(0.0 if a.sim else 0.35)
                q1 = arm.read_q()
                P.tip_path(arm, [(x, y + DASH / 2, a.z)], 5.0, settle=False, step=0.0015)
                time.sleep(0.0 if a.sim else 0.25)
                q2 = arm.read_q()
                P.tip_path(arm, [(x, y + DASH / 2, a.z + P.LIFT + 0.004)], 30.0, settle=0.3)
                tool = arm.tip_tool()
                m = (K.fk(q1, tool)[:3, 3] + K.fk(q2, tool)[:3, 3]) / 2
                marks.append({"cmd": [float(x), float(y)], "model": m.round(4).tolist()})
                print(json.dumps(marks[-1]), flush=True)
        P.go_aside(arm)
    finally:
        arm.close()
    if a.sim:
        print("sim ok")
        return 0
    path, after = P.snap_paper("grid_after")
    pa = cam.project([[GX[0] - 0.04, GY[0] - 0.03, zt], [GX[0] - 0.04, GY[2] + 0.03, zt], [GX[2] + 0.02, GY[0] - 0.03, zt], [GX[2] + 0.02, GY[2] + 0.03, zt]])
    roi = (max(0, int(pa[:, 0].min())), max(0, int(pa[:, 1].min())), min(1920, int(pa[:, 0].max())), min(1080, int(pa[:, 1].max())))
    blobs = new_blobs(before, after, cam, zt, roi)
    print("new ink blobs:", [(b["area"], b["xy"].round(4).tolist()) for b in blobs])
    P.save_state(grid={"marks": marks, "photo": path, "z": a.z, "blobs": [{"area": b["area"], "xy": b["xy"].tolist(), "px": b["px"]} for b in blobs]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
