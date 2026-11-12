"""Measure a box-shaped block standing on the table (single calibrated view).

Kept for compatibility: `measure_block` now uses the scene pipeline (scene.perceive), which
measures the top face and the height from the vertical offset between the top-face edges and the
ground-contact edges of ALL columns. The previous implementation split the faces with one Otsu
threshold and averaged two corner heights that could differ by 11 mm (review finding VM-2:
17.5 and 28.5 mm on the real block), and returned `length` / `yaw_deg` for whichever edge was
lowest in the image. Here length >= width and yaw_deg is the direction of the long axis.

    python blockfit.py frame.jpg view.jpg [color]
"""
import math

import numpy as np

import scene as S
import vision as V


def measure_block(img, cam, z_table, roi=None, color="red"):
    """Largest box of the given colour (None = any colour) inside roi = (x0, y0, x1, y1) pixels.

    Returns None when there is none, else a dict: center_xy, yaw_deg (long axis, 0..180), length,
    width, height, top_z, top_corners (4 x [x, y, z]), bbox (x, y, w, h), confidence, notes,
    height_samples (kept for old callers: [height, height])."""
    sc = S.perceive(img, cam, {"table_z": float(z_table)})
    best = None
    for o in sc.graspable():
        if o.kind != "box" or (color is not None and o.color != color):
            continue
        x, y, w, h = o.pixel_bbox
        if roi is not None and not (roi[0] <= x + w / 2 <= roi[2] and roi[1] <= y + h / 2 <= roi[3]):
            continue
        if best is None or o.size[0] * o.size[1] > best.size[0] * best.size[1]:
            best = o
    if best is None:
        return None
    length, width, height = best.size
    corners = S.box_corners(best.center[0], best.center[1], best.yaw, length, width, height, best.z_bottom)
    return {
        "center_xy": [float(best.center[0]), float(best.center[1])], "yaw_deg": math.degrees(best.yaw) % 180.0,
        "length": float(length), "width": float(width), "height": float(height), "height_samples": [float(height)] * 2,
        "top_z": float(best.z_top), "top_corners": corners[4:].tolist(), "bbox": tuple(best.pixel_bbox),
        "confidence": float(best.confidence), "notes": list(best.notes), "scene_warnings": list(sc.warnings),
    }


if __name__ == "__main__":
    import json
    import sys

    import cv2

    import camera as C
    from robot import load_config
    if len(sys.argv) < 3:
        sys.exit("usage: python blockfit.py frame.jpg view.jpg [color]")
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit(f"cannot read {sys.argv[1]}")
    cam = C.CameraModel.load()
    cfg = load_config()
    r = measure_block(img, cam, cfg["table_z"], color=sys.argv[3] if len(sys.argv) > 3 else "red")
    if r is None:
        sys.exit("no box of that colour found")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "top_corners"}))
    vis = img.copy()
    corners = V.box_corners(r["center_xy"][0], r["center_xy"][1], math.radians(r["yaw_deg"]), r["length"], r["width"],
                            r["height"], cfg["table_z"])
    V.draw_box(vis, cam, np.asarray(corners), (0, 255, 255), 1)
    x, y, w, h = r["bbox"]
    V.save_view(vis, sys.argv[2], crop=(x - 120, y - 110, x + w + 120, y + h + 90), scale=2.2)
