"""What a painting plan will look like: a simulation of the brush on paper, calibrated with the strokes of
2026-11-13 on the real arm. Planning tools score their strokes with it; `fd paint preview` shows it.

Which brush. A plan (and the painting set-up) carries a "brush" block; `brush_of` fills in what is missing:
  {"kind": "flat", "wide_mm": 9.0, "thin_mm": 2.5, "spread_mm": 1.75, "load_mm": 180.0}    the flat brush (below)
  {"kind": "round", "wide_mm": 2.7, "broad_mm": 5.0, "load_mm": 140.0}                      the thin round brush
(a set-up names a round brush's two widths under "modes": {"fine": {"press", "width_mm"}, "broad": {...}}; that
is read as well). No block = the flat brush of the morning, so every plan made before the block existed renders
as it did (but for the spread, see BRUSH_SPREAD_MM); the executor reads it the same way and refuses such a plan
while the set-up says a round brush is on (paintskill.plan_problems). A round plan is handed to roundbrush.py
(render, plan_stats), whose module text says what that brush does; everything else in this module is the flat
brush.

The flat brush (bristles about 9 mm wide, measured in a photo, taped along the gripper; the wrist does not roll,
the handle would hit the arm):
- its wide side always faces the robot: the contact line of the bristles is PERPENDICULAR to the line from
  the robot's pan axis to the brush (the "radial" direction). So a stroke that travels radially (towards or
  away from the robot) paints a band about 9 mm wide, and one that travels sideways (tangentially) drags the
  flat edge-on and leaves a thin line of 2 to 3 mm; anything between is between. That is the brush on scrap
  paper, lightly pressed (8.5 to 10 mm). On the sheet of the Akabeko the band was 12.5 mm: BRUSH_SPREAD_MM.
- one load of paint lasts about 180 mm of travel and gets paler towards the end.
- watercolour is transparent: a second coat darkens the first (the paints multiply).

A plan (JSON, see docs in paintskill.py):
  {"canvas_mm": [W, H], "pan_axis_mm": [u0, v0], "brush": {...} (optional),
   "paints": {name: {"rgb": [r, g, b], "opacity": a}},
   "layers": [{"name": ..., "paint": name, "strokes": [{"pts": [[u, v], ...], "press": 1.0}, ...]}, ...]}
Canvas frame: millimetres, u to the right, v DOWN, origin at the top left corner of the picture as the camera
and the viewer in front of the table see it. The robot stands beyond the TOP edge: pan_axis_mm has a negative v.

What was painted against what was planned: `fit_to_photo` lays a layer of a plan over a top view of the sheet
(paintskill.canvas_view) and says where the marks landed and how much wider than the model they are. On the
red of the Akabeko (2026-11-13, 14:11): 1.75 mm wider on every side, landed at (-4.25, -8.5) mm.

Units: mm on the canvas, px = mm * px_per_mm in images. Nothing here touches the arm or the camera.
Tests: tests/test_paint_model.py.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

BRUSH_WIDE_MM = 9.0        # band of a radial stroke on scrap paper, lightly pressed (measured: 8.5 .. 10)
BRUSH_THIN_MM = 2.5        # thickness of the contact line = width of a sideways stroke
# How much further than that the paint reached on the real sheet, on EVERY side of a stroke: the red of the
# Akabeko matches its photo best when the modelled marks are grown by 1.5 to 2.0 mm (fit_to_photo: IoU 0.840 at
# 1.75 against 0.794 without; the morning's own script said 0.834), which makes the band 12.5 mm. It was first
# read as wet paint spreading. The evening found the larger part: with a tool tip that had been measured in
# the air, the bristles first touched the paper 3 to 9 mm ABOVE the height that had been taken for "first
# touch" that morning. The flat brush, run at "contact 0.0, press 2.3 mm", was in fact pressed about a
# centimetre into the paper. So 12.5 mm is the band AT THAT PRESS, with that juicy red, and not a property of
# the brush: on a contact plane that has been measured with ladders (paintskill "contact") the same brush at
# 2.3 mm has to be measured again, and a plan made for it says so in its brush block ("spread_mm").
BRUSH_SPREAD_MM = 1.75
LOAD_MM = 180.0            # travel on one load of paint (the executor reloads before a stroke that would exceed it)
LOAD_FADE = 0.55           # opacity at the end of a load, relative to its start
FLAT_BRUSH = {"kind": "flat", "wide_mm": BRUSH_WIDE_MM, "thin_mm": BRUSH_THIN_MM, "spread_mm": BRUSH_SPREAD_MM,
              "load_mm": LOAD_MM}
PAPER_RGB = (244, 242, 236)
DEFAULT_PAINTS = {
    "sky": {"rgb": [232, 214, 178], "opacity": 0.85},        # thin brown + yellow wash
    "grey": {"rgb": [150, 150, 158], "opacity": 0.60},       # thin black wash (dark sky at the horizon)
    "cerulean": {"rgb": [60, 150, 205], "opacity": 0.85},
    "blue": {"rgb": [35, 80, 170], "opacity": 0.90},
    "navy": {"rgb": [22, 30, 85], "opacity": 0.92},
    "ochre": {"rgb": [200, 160, 90], "opacity": 0.85},
}


def brush_of(plan: dict) -> dict:
    """The brush a plan (or a painting set-up) is for, every number filled in: its "brush" block, the flat brush
    of the morning when it has none. kind "round": the block as roundbrush.brush_of completes it."""
    b = (plan or {}).get("brush") or {}
    kind = str(b.get("kind", "flat"))
    if kind == "round":
        import roundbrush
        return roundbrush.brush_of(plan)
    if kind != "flat":
        raise ValueError(f"unknown brush kind {kind!r}: 'flat' or 'round'")
    return {"kind": "flat", **{k: float(b.get(k, FLAT_BRUSH[k])) for k in ("wide_mm", "thin_mm", "spread_mm", "load_mm")}}


def radial(p, pan_axis) -> np.ndarray:
    """Unit vector at canvas point p pointing AWAY from the robot's pan axis."""
    d = np.asarray(p, dtype=float) - np.asarray(pan_axis, dtype=float)
    return d / max(float(np.linalg.norm(d)), 1e-9)


def wide_axis(p, pan_axis) -> np.ndarray:
    """Unit vector along the brush's contact line at p (perpendicular to radial)."""
    r = radial(p, pan_axis)
    return np.array([-r[1], r[0]])


def resample(pts, step_mm: float = 0.5) -> np.ndarray:
    """Points every step_mm along a polyline [(u, v), ...] (both ends kept)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step_mm)))
        out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return np.array(out)


def length_mm(pts) -> float:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def stroke_mask(pts, pan_axis, shape, px_per_mm: float, wide_mm: float = BRUSH_WIDE_MM,
                thin_mm: float = BRUSH_THIN_MM, press: float = 1.0, spread_mm: float = 0.0) -> np.ndarray:
    """uint8 mask (1 = painted) of one stroke of the flat brush on an image of `shape` (rows, cols): the contact
    line swept along the path. press scales the width of the contact line a little (0.5 -> 0.8 x, 1.5 -> 1.1 x).
    spread_mm: how much further the paint reaches on every side (0: the brush itself, which is what a planner
    keeps inside a shape; BRUSH_SPREAD_MM: the sheet of 2026-11-13)."""
    mask = np.zeros(shape[:2], np.uint8)
    w = wide_mm * float(np.clip(0.6 + 0.4 * press, 0.7, 1.15))
    half = max(0.2, (w - thin_mm) / 2.0)      # the dilation below adds half the thickness at both ends: w in all
    s = resample(pts, 0.5)
    quads = []
    for a, b in zip(s[:-1], s[1:]):
        wa, wb = wide_axis(a, pan_axis) * half, wide_axis(b, pan_axis) * half
        quads.append(np.array([a - wa, a + wa, b + wb, b - wb]))
    for q in quads:
        cv2.fillConvexPoly(mask, np.round(q * px_per_mm).astype(np.int32), 1)
    # the contact line has a thickness: grow the swept band by half of it (and by the spread)
    k = max(1, int(round((thin_mm + 2.0 * max(0.0, float(spread_mm))) * px_per_mm)) | 1)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def render(plan: dict, px_per_mm: float = 4.0, upto_layer: int | None = None, paper_rgb=PAPER_RGB,
           return_coverage: bool = False):
    """The painted picture as a BGR uint8 image. Strokes are laid in the order of the plan; the load of paint is
    refilled at the start of each layer and whenever the next stroke would pass the load, and fades linearly
    along the travel since the refill. return_coverage: also {paint: float32 map of how much of that paint lies
    where (sum of opacities)}.

    The image is the canvas: the point (u, v) of the plan is at pixel (u * px_per_mm, v * px_per_mm). A flat
    plan (no brush block, or kind "flat"): the marks are wide_mm / thin_mm grown by the block's spread_mm. A
    round plan: roundbrush.render, the brush as it really paints (its text says how; with a margin it also
    shows where the marks land on the sheet)."""
    brush = brush_of(plan)
    if brush["kind"] == "round":
        import roundbrush
        return roundbrush.render(plan, px_per_mm, upto_layer=upto_layer, return_coverage=return_coverage)
    load = brush["load_mm"]
    w_mm, h_mm = plan["canvas_mm"]
    pan = plan["pan_axis_mm"]
    hh, ww = int(round(h_mm * px_per_mm)), int(round(w_mm * px_per_mm))
    img = np.empty((hh, ww, 3), np.float32)
    img[:] = np.array(paper_rgb[::-1], np.float32) / 255.0
    paints = {**DEFAULT_PAINTS, **(plan.get("paints") or {})}
    cover = {}
    for li, layer in enumerate(plan["layers"]):
        if upto_layer is not None and li > upto_layer:
            break
        paint = paints[layer["paint"]]
        if not isinstance(paint, dict):                 # a plan of the round brush's kit writes [r, g, b]
            paint = {"rgb": list(paint)}
        col = np.array(paint["rgb"][::-1], np.float32) / 255.0
        base = float(paint.get("opacity", 0.85))
        used = 0.0
        for st in layer["strokes"]:
            ln = length_mm(st["pts"])
            if used + ln > load and used > 0.0:
                used = 0.0
            m = stroke_mask(st["pts"], pan, (hh, ww), px_per_mm, brush["wide_mm"], brush["thin_mm"],
                            press=float(st.get("press", 1.0)), spread_mm=brush["spread_mm"])
            a = base * (1.0 - (1.0 - LOAD_FADE) * min(1.0, (used + 0.5 * ln) / load))
            sel = m > 0
            img[sel] *= (1.0 - a * (1.0 - col))
            if return_coverage:
                c = cover.setdefault(layer["paint"], np.zeros((hh, ww), np.float32))
                c[sel] += a
            used += ln
    out = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return (out, cover) if return_coverage else out


def plan_stats(plan: dict, setup: dict | None = None) -> dict:
    """Numbers the executor cares about: strokes, travel, reloads, an estimate of the painting time, and what
    breaks the rules (strokes longer than one load or shorter than 1 mm, points outside the canvas, points
    that are not pairs). setup: the
    painting set-up the plan is meant for (paintskill.load_setup()): then also a press that is more than brush
    work allows with its press_mm, which the executor refuses (roundbrush.press_problem). A round plan:
    roundbrush.stats (these keys and more; it flags a press over the kit's own cap also without a set-up, and
    its minutes are fitted to the calls of 2026-11-13). The executor's own word on a plan is
    paintskill.plan_problems(plan, setup)."""
    brush = brush_of(plan)
    if brush["kind"] == "round":
        import roundbrush
        return roundbrush.stats(plan, setup=setup)
    load = brush["load_mm"]
    w_mm, h_mm = plan["canvas_mm"]
    n = 0
    total = 0.0
    reloads = 0
    bad = []
    per_layer = []
    for layer in plan["layers"]:
        used, ln_layer, n_layer, reloads_layer = 0.0, 0.0, 0, 1
        for i, st in enumerate(layer["strokes"]):
            pts = np.asarray(st["pts"], dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 1:      # ([[u, v, z], ...] read as pairs would be other points)
                bad.append(f"{layer['name']}[{i}]: its points are not pairs [u, v] of canvas millimetres")
                continue
            ln = length_mm(pts)
            if ln > load + 1e-6:
                bad.append(f"{layer['name']}[{i}]: {ln:.0f} mm is longer than one load ({load:.0f} mm)")
            if len(pts) < 2 or ln < 1.0:
                bad.append(f"{layer['name']}[{i}]: shorter than 1 mm")
            if pts[:, 0].min() < 0 or pts[:, 1].min() < 0 or pts[:, 0].max() > w_mm or pts[:, 1].max() > h_mm:
                bad.append(f"{layer['name']}[{i}]: leaves the canvas")
            if setup is not None:
                import roundbrush       # (the press limit of a set-up is not the round brush's alone: it lives with the kit's cap)
                why = roundbrush.press_problem(float(st.get("press", 1.0)), setup)
                if why:
                    bad.append(f"{layer['name']}[{i}]: {why}")
            if used + ln > load and used > 0.0:
                used = 0.0
                reloads_layer += 1
            used += ln
            ln_layer += ln
            n_layer += 1
        per_layer.append({"name": layer["name"], "paint": layer["paint"], "strokes": n_layer,
                          "travel_mm": round(ln_layer), "loads": reloads_layer})
        n += n_layer
        total += ln_layer
        reloads += reloads_layer
    est = 6.0 * n + total / 20.0 + 22.0 * reloads + 60.0 * len(plan["layers"])
    return {"strokes": n, "travel_mm": round(total), "loads": reloads, "layers": per_layer,
            "minutes": round(est / 60.0, 1), "problems": bad[:20], "n_problems": len(bad)}


def score(img_bgr, target_bgr) -> dict:
    """How close a rendered picture is to the target (same size): mean colour difference in CIELAB of the two
    pictures blurred to brush scale (lower is better), plus the same on a coarse 16 x 11 grid (composition)."""
    a = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (0, 0), 3.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(target_bgr, (0, 0), 3.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.linalg.norm(a - b, axis=2)
    small = cv2.resize(d, (16, 11), interpolation=cv2.INTER_AREA)
    return {"lab_mean": round(float(d.mean()), 2), "lab_p90": round(float(np.percentile(d, 90)), 2),
            "coarse_max": round(float(small.max()), 2)}


# ----------------------------------------------------------------------------- the plan against a photo of the sheet
def layer_mask(plan: dict, layer: int = 0, px_per_mm: float = 4.0, margin_mm: float = 0.0, spread_mm: float = 0.0) -> np.ndarray:
    """What one layer of a plan covers, as the brush is MODELLED (uint8, 1 = painted), on an image with margin_mm
    of paper round the canvas (the frame of paintskill.canvas_view). Flat: the swept contact line, grown by
    spread_mm; round: every stroke as wide as it was designed (roundbrush.width_of), grown by spread_mm."""
    brush = brush_of(plan)
    w_mm, h_mm = plan["canvas_mm"]
    m = float(margin_mm)
    shape = (int(round((h_mm + 2.0 * m) * px_per_mm)), int(round((w_mm + 2.0 * m) * px_per_mm)))
    out = np.zeros(shape, np.uint8)
    strokes = plan["layers"][layer]["strokes"]
    if brush["kind"] == "round":
        import roundbrush
        for st in strokes:
            w = roundbrush.width_of(float(st.get("press", roundbrush.P_FINE)), brush["wide_mm"], brush["broad_mm"])
            p = np.round((np.asarray(st["pts"], dtype=float) + m) * px_per_mm).astype(np.int32)
            cv2.polylines(out, [p.reshape(-1, 1, 2)], False, 1, max(1, int(round((w + 2.0 * spread_mm) * px_per_mm))))
        return out
    pan = [float(plan["pan_axis_mm"][0]) + m, float(plan["pan_axis_mm"][1]) + m]
    for st in strokes:
        out |= stroke_mask(np.asarray(st["pts"], dtype=float) + m, pan, shape, px_per_mm, brush["wide_mm"],
                           brush["thin_mm"], press=float(st.get("press", 1.0)), spread_mm=spread_mm)
    return out


def _paint_rgb(plan: dict, name: str):
    p = {**DEFAULT_PAINTS, **(plan.get("paints") or {})}.get(name)
    if p is None and brush_of(plan)["kind"] == "round":
        import roundbrush
        p = roundbrush.PAINTS.get(roundbrush.tube_of(name))
    if p is None:
        return None
    return tuple(p["rgb"]) if isinstance(p, dict) else tuple(p)


def photo_mask(topview_bgr, paint_rgb=None, other_rgb=()) -> np.ndarray:
    """Where a top view of the sheet is painted (uint8, 1 = paint): the pixels whose colour is far from the
    paper's (the median of the picture) in CIELAB, the threshold by Otsu. With paint_rgb and other_rgb (colours of
    the other paints on the sheet): only the pixels that are nearer to this paint than to any of the others, by
    their hue more than by their lightness (a pool of red is dark, and still red and not black)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(topview_bgr, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    paper = np.median(lab.reshape(-1, 3), axis=0)
    d = np.linalg.norm(lab - paper, axis=2)
    thr, _ = cv2.threshold(np.clip(d, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = d > float(thr)
    others = [c for c in other_rgb if tuple(c) != tuple(paint_rgb or ())]
    if paint_rgb is not None and others:
        weight = np.array([0.25, 1.0, 1.0], np.float32)             # L counts a quarter of a, b

        def far(rgb):
            px = np.array([[list(rgb)[::-1]]], np.uint8)
            return np.linalg.norm((lab - cv2.cvtColor(px, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]) * weight, axis=2)
        mine = far(paint_rgb)
        for c in others:
            mask &= mine <= far(c)
    return mask.astype(np.uint8)


def _best_shift(pred, photo, reach_px: int):
    """Intersection over union of two masks for every whole-pixel shift of `pred` up to reach_px, by one
    correlation. -> (iou map [dy + reach, dx + reach])."""
    big = cv2.copyMakeBorder(photo.astype(np.float32), reach_px, reach_px, reach_px, reach_px, cv2.BORDER_CONSTANT, value=0)
    inter = cv2.matchTemplate(big, pred.astype(np.float32), cv2.TM_CCORR)
    return inter / np.maximum(float(pred.sum()) + float(photo.sum()) - inter, 1.0)


def fit_to_photo(plan: dict, topview_bgr, layer: int = 0, margin_mm: float = 12.0, max_shift_mm: float = 15.0,
                 spreads_mm=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0), paint_only: bool = False) -> dict:
    """Where a layer was really painted and how much wider than the model: the layer's modelled marks
    (layer_mask, the brush itself) against the painted pixels of a top view of the sheet (paintskill.canvas_view
    with the same margin_mm; its px per mm is taken from the picture's width). For every spread of the list the
    marks are grown by it on every side and slid over the photo; the pair with the largest intersection over
    union wins. The photo is best taken when only this layer is on the sheet; paint_only=True takes of a sheet
    with more paints on it the pixels that look like this layer's paint (photo_mask), which loses the darkest
    pools of a paint to black.

    -> {"shift_mm": [du, dv]   where the marks landed, from where the plan puts them (to the right, down),
        "spread_mm":           how much further than the modelled brush the paint reached, on every side,
        "iou":                 intersection over union of that fit,
        "iou_unfitted":        the same with the brush as it is modelled (no spread) at ITS best shift: what the
                               fit of the width has gained,
        "iou_as_planned":      no spread and no shift: the plan as it stands,
        "px_per_mm", "painted_mm2", "scan": [[spread, iou, du, dv], ...]}
    The shift is not the brush: it is where the arm paints against where the camera model says it would, and it
    was the same for every stroke of a sheet. The red of the Akabeko (stage photo of 2026-11-13 14:11):
    shift (-4.25, -8.5) mm, spread 1.75 mm, IoU 0.840 against 0.794 unfitted."""
    w_mm, h_mm = plan["canvas_mm"]
    ppm = topview_bgr.shape[1] / (float(w_mm) + 2.0 * float(margin_mm))
    want_h = (float(h_mm) + 2.0 * float(margin_mm)) * ppm
    if abs(topview_bgr.shape[0] - want_h) > 2.0:
        raise ValueError(f"the top view is {topview_bgr.shape[1]} x {topview_bgr.shape[0]} px: not a canvas of "
                         f"{w_mm} x {h_mm} mm with {margin_mm} mm of paper round it")
    pred = layer_mask(plan, layer, ppm, margin_mm)[:topview_bgr.shape[0]]
    if pred.shape[0] < topview_bgr.shape[0]:
        pred = np.vstack([pred, np.zeros((topview_bgr.shape[0] - pred.shape[0], pred.shape[1]), np.uint8)])
    if paint_only:
        names = [ly["paint"] for ly in plan["layers"]]
        photo = photo_mask(topview_bgr, _paint_rgb(plan, names[layer]),
                           [c for c in (_paint_rgb(plan, n) for n in names) if c is not None])
    else:
        photo = photo_mask(topview_bgr)
    reach = int(round(float(max_shift_mm) * ppm))
    scan, best = [], None
    for s in spreads_mm:
        k = int(round(float(s) * ppm))
        grown = pred if k == 0 else cv2.dilate(pred, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        iou = _best_shift(grown, photo, reach)
        iy, ix = np.unravel_index(int(np.argmax(iou)), iou.shape)
        row = [float(s), round(float(iou[iy, ix]), 4), float(ix - reach) / ppm, float(iy - reach) / ppm]
        scan.append(row)
        if float(s) == 0.0:
            unfitted, as_planned = row[1], round(float(iou[reach, reach]), 4)
        if best is None or row[1] > best[1]:
            best = row
    if 0.0 not in [float(s) for s in spreads_mm]:
        iou = _best_shift(pred, photo, reach)
        unfitted, as_planned = round(float(iou.max()), 4), round(float(iou[reach, reach]), 4)
    return {"shift_mm": [round(float(best[2]), 2), round(float(best[3]), 2)], "spread_mm": best[0], "iou": best[1],
            "iou_unfitted": unfitted, "iou_as_planned": as_planned, "px_per_mm": round(ppm, 3),
            "painted_mm2": round(float(photo.sum()) / (ppm * ppm), 1), "scan": scan}


def load_plan(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    plan = load_plan(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "preview.png"
    cv2.imwrite(out, render(plan))
    print(json.dumps(plan_stats(plan), indent=1, ensure_ascii=False))
