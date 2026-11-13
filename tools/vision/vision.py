"""Image analysis helpers: colour segmentation of the red block / black box / white gripper,
fingertip localisation, 3D box fitting through the camera model, and annotation.

These are the v1 building blocks, kept for the low-level commands in fd.py. The fixed colour
thresholds in red_mask / black_mask / white_mask have no margin against the mirror reflections
on the glossy table (review finding VM-1: +5 % brightness moves a block by 8 mm); scene.perceive()
segments relative to the table brightness of the same frame and should be used for locating
objects. scene.detect_jaws() replaces find_tip() for checking where the jaws are."""

from __future__ import annotations

import math

import cv2
import numpy as np


# --------------------------------------------------------------------------- masks
def red_mask(img: np.ndarray, v_min: int = 140, s_min: int = 158) -> np.ndarray:
    """Saturated bright red (the block). The table is dark reddish wood, so both
    saturation and brightness must be high.

    v_min / s_min are absolute thresholds: the reflection of a red block on the table reaches
    S 190 / V 150, its shaded front face starts at S 174 / V 153, so the defaults sit inside the
    overlap. Running the mask at v_min 140 and 165 and comparing the lowest rows shows whether a
    reflection is included (finding VM-1); scene.perceive() avoids the problem altogether."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h <= 9) | (h >= 171)) & (s >= s_min) & (v >= v_min)
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m


def black_mask(img: np.ndarray, v_dark: int = 85, v_gray: int = 130, s_gray: int = 80) -> np.ndarray:
    """Black box: truly dark pixels, plus its front-lit faces which look dull gray
    (dimmer than the table and far less saturated than the wood)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    m = ((v <= v_dark) | ((v <= v_gray) & (s <= s_gray))).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return m


def white_mask(img: np.ndarray, v_min: int = 170, s_max: int = 26) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def components(mask: np.ndarray, min_area: int = 200) -> list[dict]:
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append({"label": i, "bbox": (int(x), int(y), int(w), int(h)), "area": int(area),
                    "centroid": (float(cent[i][0]), float(cent[i][1]))})
    out.sort(key=lambda c: -c["area"])
    for c in out:
        c["mask"] = (lab == c["label"]).astype(np.uint8) * 255
    return out


def largest_component(mask: np.ndarray, min_area: int = 200, roi=None) -> dict | None:
    """roi = (x0, y0, x1, y1) restricts the search.
    The result has "roi_cut": True when the component touches an edge of the roi that is not also
    the lower image border, i.e. when the window cut the object (pass the same roi to fit_box)."""
    m = mask
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = 0, 0, w, h
    if roi is not None:
        m = np.zeros_like(mask)
        x0, y0, x1, y1 = max(0, int(roi[0])), max(0, int(roi[1])), min(w, int(roi[2])), min(h, int(roi[3]))
        m[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    comps = components(m, min_area)
    if not comps:
        return None
    best = comps[0]
    bx, by, bw, bh = best["bbox"]
    best["roi_cut"] = bool(roi is not None and (bx <= x0 or by <= y0 or bx + bw >= x1 or (by + bh >= y1 and y1 < h)))
    return best


def polygon(mask: np.ndarray, eps_frac: float = 0.012) -> np.ndarray:
    """Convex-hull polygon (N x 2, float) of a blob, simplified."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    ap = cv2.approxPolyDP(hull, eps_frac * cv2.arcLength(hull, True), True)
    return ap.reshape(-1, 2).astype(float)


# --------------------------------------------------------------------------- fingertip
def gripper_mask(img: np.ndarray, ref: np.ndarray | None = None, diff_thresh: int = 28) -> np.ndarray:
    """Pixels that look like the (white, possibly shaded) gripper plastic: low saturation
    and not dark. The wood table is strongly saturated even in shadow, so shadows drop out.
    With a reference frame (same camera settings, arm away from the table) only pixels
    that CHANGED are kept, which removes static low-saturation areas such as the glossy
    reflection on the table, the robot base and cables."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] <= 62) & (hsv[..., 2] >= 95)).astype(np.uint8) * 255
    if ref is not None:
        d = cv2.absdiff(img, ref).max(axis=2)
        m = cv2.bitwise_and(m, (d >= diff_thresh).astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return m


def find_tip(img: np.ndarray, ref: np.ndarray | None = None, roi=None, min_area: int = 600,
             band_px: int = 5, exclude=(), expect=None, max_dist: float = 120.0,
             info: dict | None = None, static_frac_max: float = 0.15) -> dict | None:
    """Locate the lowest point (largest image row) of the gripper silhouette inside roi.

    exclude: rectangles (x0, y0, x1, y1) to ignore (e.g. the robot base).
    expect:  approximate pixel of the tip. It only decides WHICH silhouette is meant (the one
             whose lowest point is nearest, within max_dist); it never chooses between the two
             jaws of one gripper: blobs whose lowest points are within 60 px of each other are
             one gripper and its lowest point is returned (review finding VM-5).
    info:    optional dict; receives info["reason"] when None is returned.
    Returns the pixel of the silhouette's bottom extremity (median column of the lowest
    `band_px` rows) or None. With a reference frame the result is refused (None) when the
    reference already shows pale pixels around / below the tip (glare, a white object): there
    the difference image loses the shaded underside of the fingertip and the "tip" would be
    several pixels off.

    For measuring where the jaws are relative to the kinematics use scene.detect_jaws(), which
    treats the two jaws separately; this function is kept for the calibration commands."""
    def _fail(reason):
        if info is not None:
            info["reason"] = reason
        return None

    m = gripper_mask(img, ref)
    if roi is not None:
        x0, y0, x1, y1 = roi
        keep = np.zeros_like(m)
        keep[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        m = keep
    for (x0, y0, x1, y1) in exclude:
        m[y0:y1, x0:x1] = 0
    # two jaws a few millimetres apart are one gripper: bridge the gap sideways
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 21), np.uint8))
    comps = components(m, min_area)
    cands = []
    for c in comps:
        ys, xs = np.nonzero(c["mask"])
        y_max = int(ys.max())
        sel = ys >= y_max - band_px
        u = float(np.median(xs[sel]))
        cands.append({"uv": (u, float(y_max)), "area": c["area"], "bbox": c["bbox"],
                      "width_at_tip": int(xs[sel].max() - xs[sel].min() + 1), "mask": c["mask"]})
    if not cands:
        return _fail("no gripper-coloured region found")
    # group blobs whose lowest points are close: the lowest point of the group is the tip
    groups = []
    for c in sorted(cands, key=lambda c: -c["uv"][1]):
        for g in groups:
            if any(math.hypot(c["uv"][0] - o["uv"][0], c["uv"][1] - o["uv"][1]) <= 60.0 for o in g):
                g.append(c)
                break
        else:
            groups.append([c])
    tips = [g[0] for g in groups]          # each group is sorted lowest first
    if expect is not None:
        best = min(tips, key=lambda c: math.hypot(c["uv"][0] - expect[0], c["uv"][1] - expect[1]))
        if math.hypot(best["uv"][0] - expect[0], best["uv"][1] - expect[1]) > max_dist:
            return _fail(f"no gripper region within {max_dist:.0f} px of the expected tip")
    else:
        best = max(tips, key=lambda c: c["uv"][1])
    if ref is not None:
        u, v = int(round(best["uv"][0])), int(round(best["uv"][1]))
        h, w = ref.shape[:2]
        x0, x1 = max(0, u - 40), min(w, u + 41)
        y0, y1 = max(0, v - 12), min(h, v + 26)
        if x1 > x0 and y1 > y0:
            hsv = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            frac = float(np.mean((hsv[..., 1] <= 62) & (hsv[..., 2] >= 95)))
            best["static_pale_fraction"] = frac
            if frac > static_frac_max:
                return _fail(f"tip over static low-saturation background ({frac:.0%} of the reference "
                             "around the tip is pale): the fingertip cannot be separated reliably")
    return best


# --------------------------------------------------------------------------- 3D boxes
def box_corners(x: float, y: float, yaw: float, length: float, width: float, height: float, z0: float) -> np.ndarray:
    """8 corners of a box whose base centre is (x, y, z0); length along the yaw direction."""
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    pts = []
    for dz in (0.0, height):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pts.append(np.array([x, y, z0 + dz]) + sx * length / 2 * ex + sy * width / 2 * ey)
    return np.array(pts)


def project_hull_mask(cam, corners: np.ndarray, shape) -> np.ndarray:
    uv = cam.project(corners)
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(cv2.bitwise_and(a, b))
    union = np.count_nonzero(cv2.bitwise_or(a, b))
    return inter / union if union else 0.0


def nelder_mead(fn, x0, steps, iters: int = 250, tol: float = 1e-5):
    """Minimal Nelder-Mead minimiser (no scipy on this machine)."""
    n = len(x0)
    sim = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        p[i] += steps[i]
        sim.append(p)
    vals = [fn(p) for p in sim]
    for _ in range(iters):
        order = np.argsort(vals)
        sim = [sim[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = np.mean(sim[:-1], axis=0)
        xr = cen + (cen - sim[-1])
        fr = fn(xr)
        if fr < vals[0]:
            xe = cen + 2.0 * (cen - sim[-1])
            fe = fn(xe)
            sim[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            sim[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (sim[-1] - cen)
            fc = fn(xc)
            if fc < vals[-1]:
                sim[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    sim[i] = sim[0] + 0.5 * (sim[i] - sim[0])
                    vals[i] = fn(sim[i])
    i = int(np.argmin(vals))
    return sim[i], vals[i]


def fit_box(cam, mask: np.ndarray, z0: float, init, steps=(0.01, 0.01, 0.15, 0.01, 0.01, 0.01),
            min_dims=(0.01, 0.01, 0.005), clip_to_image: bool = True, roi=None):
    """Fit (x, y, yaw, length, width, height) of a box standing on the plane z0 so that the
    projected silhouette matches `mask`. Returns (params, iou).

    roi = (x0, y0, x1, y1): the window the mask was taken from (e.g. the ROI given to
    largest_component). The model silhouette is clipped to the same window before it is compared,
    so a mask cut by the window does not shrink or shift the fitted box (review finding VM-7).
    clip_to_image=True (default) clips the model to the image as well, like the mask is.

    NOTE (finding VM-2): with all six parameters free the width and the height trade off against
    each other and the result is biased; fix what is known (step 0) or use scene.perceive(), which
    measures the faces instead. A step of 0 keeps that parameter exactly fixed."""
    h_img, w_img = mask.shape[:2]
    x0, y0, x1, y1 = 0, 0, w_img, h_img
    if roi is not None:
        x0, y0 = max(x0, int(roi[0])), max(y0, int(roi[1]))
        x1, y1 = min(x1, int(roi[2])), min(y1, int(roi[3]))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"fit_box: empty roi {roi}")
    elif not clip_to_image:
        x0, y0, x1, y1 = -10 ** 6, -10 ** 6, 10 ** 6, 10 ** 6
    # only the part of the mask inside the window counts; work on its bounding box for speed
    win = np.zeros_like(mask)
    win[max(y0, 0):min(y1, h_img), max(x0, 0):min(x1, w_img)] = mask[max(y0, 0):min(y1, h_img), max(x0, 0):min(x1, w_img)]
    ys, xs = np.nonzero(win)
    if len(xs) == 0:
        raise ValueError("fit_box: the mask is empty inside the window")
    bx0, by0 = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
    bx1, by1 = min(w_img, int(xs.max()) + 3), min(h_img, int(ys.max()) + 3)
    sub = np.ascontiguousarray(win[by0:by1, bx0:bx1])
    area_mask = cv2.countNonZero(sub)
    clip_rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    buf = np.zeros_like(sub)

    def cost(p):
        x, y, yaw, length, width, height = p
        if length < min_dims[0] or width < min_dims[1] or height < min_dims[2]:
            return 1.0
        try:
            uv = cam.project(box_corners(x, y, yaw, length, width, height, z0))
            if not np.all(np.isfinite(uv)) or np.abs(uv).max() > 1e6:
                return 1.0
            hull = cv2.convexHull(uv.astype(np.float32))
            # area of the model silhouette inside the window (exact polygon clipping) ...
            area_model, _ = cv2.intersectConvexConvex(hull, clip_rect)
            # ... and its overlap with the mask (the mask lies entirely inside `sub`)
            buf.fill(0)
            cv2.fillConvexPoly(buf, np.round(hull.reshape(-1, 2) - (bx0, by0)).astype(np.int32), 255)
            inter = cv2.countNonZero(cv2.bitwise_and(buf, sub))
        except Exception:
            return 1.0
        union = area_mask + max(float(area_model), float(inter)) - inter
        return 1.0 - inter / union if union > 0 else 1.0

    best_p, best_c = None, 2.0
    for restart in range(3):
        p0 = np.array(init, dtype=float) if best_p is None else best_p
        st = np.array(steps, dtype=float) * (0.5 ** restart)
        p, c = nelder_mead(cost, p0, st)
        if c < best_c:
            best_p, best_c = p, c
    return best_p, 1.0 - best_c


# --------------------------------------------------------------------------- drawing
def draw_grid(img: np.ndarray, step: int = 120, color=(255, 255, 0), offset=(0, 0)) -> np.ndarray:
    """Pixel grid with coordinate labels.

    Every line is labelled again every second cell, so that a crop taken afterwards (save_view
    with crop=...) still shows which line is which (review finding VM-8). offset = (x0, y0) is
    the position of img inside the full frame when the grid is drawn on a crop: lines and labels
    then show full-frame coordinates."""
    step = int(step)
    if step < 8:
        raise ValueError(f"draw_grid: step {step} px is too small")
    out = img.copy()
    h, w = out.shape[:2]
    ox, oy = int(offset[0]), int(offset[1])
    xs = [x - ox for x in range((ox // step) * step, ox + w, step) if 0 <= x - ox < w]
    ys = [y - oy for y in range((oy // step) * step, oy + h, step) if 0 <= y - oy < h]
    for x in xs:
        cv2.line(out, (x, 0), (x, h - 1), color, 1)
    for y in ys:
        cv2.line(out, (0, y), (w - 1, y), color, 1)

    def label(text, org):
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    rows = [16] + [y + 16 for y in ys[1::2]]        # label rows: top edge and under every 2nd line
    cols = [3] + [x + 3 for x in xs[1::2]]          # label columns: left edge and right of every 2nd line
    for x in xs:
        for r in rows:
            if r < h:
                label(str(x + ox), (x + 3, r))
    for y in ys:
        for c in cols:
            if c < w and y - 4 > 0:
                label(str(y + oy), (c, y - 4))
    return out


def draw_points(img: np.ndarray, pts: dict, color=(0, 255, 0), size: int = 14) -> np.ndarray:
    """pts: label -> (u, v)."""
    out = img
    for label, (u, v) in pts.items():
        u, v = int(round(u)), int(round(v))
        cv2.drawMarker(out, (u, v), color, cv2.MARKER_CROSS, size, 2)
        if label:
            cv2.putText(out, str(label), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(out, str(label), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_box(img: np.ndarray, cam, corners: np.ndarray, color=(0, 255, 255), thick: int = 2) -> np.ndarray:
    uv = cam.project(corners).astype(int)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, thick)
    return img


def draw_axes(img: np.ndarray, cam, origin=(0.0, 0.0, 0.0), length: float = 0.05) -> np.ndarray:
    o = np.asarray(origin, dtype=float)
    pts = cam.project([o, o + [length, 0, 0], o + [0, length, 0], o + [0, 0, length]]).astype(int)
    for i, col in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):  # x red, y green, z blue
        cv2.line(img, tuple(pts[0]), tuple(pts[i]), col, 3)
    return img


def save_view(img: np.ndarray, path: str, crop=None, scale: float = 1.0, quality: int = 90) -> str:
    """Save (optionally a crop, optionally rescaled) for viewing.
    Raises ValueError with the reason when the crop is empty or the file cannot be written
    (it used to fail inside OpenCV, or silently)."""
    if img is None or getattr(img, "size", 0) == 0:
        raise ValueError("save_view: no image")
    out = img
    if crop is not None:
        x0, y0, x1, y1 = (int(v) for v in crop)
        h, w = img.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"save_view: crop {tuple(crop)} is empty or outside the {w}x{h} image")
        out = out[y0:y1, x0:x1]
    if not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0):
        raise ValueError(f"save_view: bad scale {scale!r}")
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if not cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
        raise ValueError(f"save_view: could not write {path}")
    return path
