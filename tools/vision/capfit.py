"""Locate an upright cylinder (bottle cap, peg) standing on the table (single calibrated view).

Kept for compatibility: `fit_cap` now uses the scene pipeline (scene.perceive): the table-colour
segmentation is relative to the table brightness of the frame (no fixed white threshold) and the
cylinder is measured from its outline (two straight sides -> diameter, lower arc -> distance,
upper arc -> height) instead of a Nelder-Mead search over an IoU that took several seconds.

    python tools/vision/capfit.py frame.jpg view.jpg [x0,y0,x1,y1]
"""
import json
import math
import sys

import cv2
import numpy as np

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # tools/
import tazuna_paths  # noqa: E402,F401  (puts every tools/<feature>/ folder on sys.path)

import scene as S
import vision as V


def cyl_points(x, y, r, h, z0, n=40):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)
    return np.vstack([np.column_stack([ring, np.full(n, z0)]), np.column_stack([ring, np.full(n, z0 + h)])])


def fit_cap(img, cam, z0, roi=None, colors=("white", "ivory")):
    """Largest cylinder of one of the given colours (None = any) whose image centre lies in
    roi = (x0, y0, x1, y1). Returns None or a dict: center_xy, radius, diameter, height, iou
    (the fit confidence, 0..1), bbox, mask (uint8, the projected model outline filled), notes."""
    sc = S.perceive(img, cam, {"table_z": float(z0)})
    best = None
    for o in sc.graspable():
        if o.kind != "cylinder" or (colors is not None and o.color not in colors):
            continue
        x, y, w, h = o.pixel_bbox
        if roi is not None and not (roi[0] <= x + w / 2 <= roi[2] and roi[1] <= y + h / 2 <= roi[3]):
            continue
        if best is None or o.size[0] > best.size[0]:
            best = o
    if best is None:
        return None
    r = 0.5 * best.size[0]
    mask = V.project_hull_mask(cam, cyl_points(best.center[0], best.center[1], r, best.size[2], z0), img.shape)
    return {"center_xy": [round(float(best.center[0]), 4), round(float(best.center[1]), 4)], "radius": round(r, 4),
            "diameter": round(2 * r, 4), "height": round(float(best.size[2]), 4), "iou": round(float(best.confidence), 3),
            "bbox": [int(v) for v in best.pixel_bbox], "mask": mask, "color": best.color, "notes": list(best.notes)}


if __name__ == "__main__":
    import camera as C
    from robot import load_config
    if len(sys.argv) < 3:
        sys.exit("usage: python tools/vision/capfit.py frame.jpg view.jpg [x0,y0,x1,y1]")
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit(f"cannot read {sys.argv[1]}")
    roi = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    cam = C.CameraModel.load()
    cfg = load_config()
    r = fit_cap(img, cam, cfg["table_z"], roi)
    if r is None:
        sys.exit("no cylinder found")
    r.pop("mask")
    print(json.dumps(r))
    vis = img.copy()
    pts = cam.project(cyl_points(r["center_xy"][0], r["center_xy"][1], r["radius"], r["height"], cfg["table_z"])).astype(np.int32)
    n = len(pts) // 2
    cv2.polylines(vis, [pts[:n]], True, (0, 255, 255), 1)
    cv2.polylines(vis, [pts[n:]], True, (0, 255, 0), 1)
    V.draw_points(vis, {"c": cam.project([[r["center_xy"][0], r["center_xy"][1], cfg["table_z"] + r["height"]]])[0]},
                  color=(0, 0, 255), size=14)
    x, y, w, h = r["bbox"]
    V.save_view(vis, sys.argv[2], crop=(x - 120, y - 110, x + w + 120, y + h + 80), scale=2.0)
