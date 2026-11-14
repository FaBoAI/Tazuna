"""From shapes to brush strokes: a picture made of filled shapes (a red cow, a sun, letters) is turned into the
zig-zag strokes the flat brush can really make.

The brush's contact line is perpendicular to the line from the robot's pan axis to the brush (render.py). So:
- an area is filled by RADIAL passes (along rays from the pan axis), stepping sideways between them, as one
  continuous zig-zag; the area is cut into rings around the pan axis so that the passes of one zig-zag are
  short and their spacing fits the ring's radius (on rays the spacing grows with the distance from the robot);
- the middle of the brush may only go where the WHOLE contact line stays inside the shape
  (`valid_centres`): paint then never crosses the shape's edge, and what the brush cannot reach without
  crossing it (sharp corners, slivers narrower than the brush) stays white.

Shapes are drawn into a mask (uint8 image of the canvas, `px_per_mm` pixels per millimetre): `Canvas` has
polygon / ellipse / rounded box, each either added or cut out. Canvas frame as in render.py: millimetres,
u right, v down, the robot beyond the top edge.

What is left of the white. A shape's unpainted holes and notches (an eye, a ring, a stripe, the gap between two
parts) come out smaller than they are drawn, and narrow ones close: `design_report` lists them with the size
they will have and flags those that close, BEFORE the paint is on the paper. Its numbers are the Akabeko of
2026-11-13 (flat brush, pressed about a centimetre, see render.BRUSH_SPREAD_MM): the 26 mm disc on the rump came
out about 23 x 16 mm, the 17 x 13 mm eye closed almost completely.

Everything here is the flat brush; the thin round brush has its own kit (roundbrush.py).
Nothing here touches the arm. Tests: tests/test_paint_model.py.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import render as RD


class Canvas:
    """A mask of the canvas that shapes are added to and cut out of."""

    def __init__(self, size_mm, px_per_mm: float = 4.0):
        self.w_mm, self.h_mm = float(size_mm[0]), float(size_mm[1])
        self.ppm = float(px_per_mm)
        self.mask = np.zeros((int(round(self.h_mm * self.ppm)), int(round(self.w_mm * self.ppm))), np.uint8)

    def _px(self, pts):
        return np.round(np.asarray(pts, dtype=float).reshape(-1, 2) * self.ppm).astype(np.int32)

    def polygon(self, pts, on: bool = True):
        cv2.fillPoly(self.mask, [self._px(pts)], 1 if on else 0)
        return self

    def ellipse(self, centre, radii, angle_deg: float = 0.0, on: bool = True):
        c = self._px([centre])[0]
        ax = (int(round(radii[0] * self.ppm)), int(round(radii[1] * self.ppm)))
        cv2.ellipse(self.mask, (int(c[0]), int(c[1])), ax, float(angle_deg), 0, 360, 1 if on else 0, -1)
        return self

    def circle(self, centre, r, on: bool = True):
        return self.ellipse(centre, (r, r), 0.0, on)

    def rounded_box(self, u0, v0, u1, v1, r, on: bool = True):
        r = min(float(r), (u1 - u0) / 2.0, (v1 - v0) / 2.0)
        self.polygon([(u0 + r, v0), (u1 - r, v0), (u1 - r, v1), (u0 + r, v1)], on)
        self.polygon([(u0, v0 + r), (u1, v0 + r), (u1, v1 - r), (u0, v1 - r)], on)
        for c in ((u0 + r, v0 + r), (u1 - r, v0 + r), (u1 - r, v1 - r), (u0 + r, v1 - r)):
            self.circle(c, r, on)
        return self

    def band(self, pts, width_mm: float, on: bool = True):
        """A line of the given width along a polyline (a gap between two parts, a stripe)."""
        cv2.polylines(self.mask, [self._px(pts)], False, 1 if on else 0, max(1, int(round(width_mm * self.ppm))), cv2.LINE_8)
        return self

    def copy(self) -> "Canvas":
        c = Canvas((self.w_mm, self.h_mm), self.ppm)
        c.mask = self.mask.copy()
        return c


def _grids(shape, ppm, pan):
    """Per pixel: radius from the pan axis (mm), angle of the ray (rad, 0 = straight down the picture, positive
    towards +u), and the unit vectors along the ray / along the brush's contact line (in pixels of u, v)."""
    hh, ww = shape
    u = (np.arange(ww, dtype=np.float32) + 0.5) / ppm - float(pan[0])
    v = (np.arange(hh, dtype=np.float32) + 0.5) / ppm - float(pan[1])
    du, dv = np.meshgrid(u, v)
    r = np.hypot(du, dv)
    th = np.arctan2(du, dv)
    return r, th, du / np.maximum(r, 1e-6), dv / np.maximum(r, 1e-6)


def valid_centres(mask, pan, ppm: float, wide_mm: float = RD.BRUSH_WIDE_MM, thin_mm: float = RD.BRUSH_THIN_MM,
                  margin_mm: float = 0.5) -> np.ndarray:
    """Where the middle of the brush may be so that its whole contact line (wide_mm across the ray, thin_mm along
    it, plus margin_mm all round) lies inside the mask. Bool array of the mask's shape."""
    hh, ww = mask.shape
    _r, _th, ru, rv = _grids(mask.shape, ppm, pan)
    xs, ys = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
    ok = mask > 0
    half_w = (wide_mm / 2.0 + margin_mm) * ppm
    half_t = (thin_mm / 2.0 + margin_mm) * ppm
    m = mask.astype(np.uint8)
    # contact line: along (-rv, ru) (perpendicular to the ray); thickness: along the ray
    def sample(du, dv):
        return cv2.remap(m, (xs + du).astype(np.float32), (ys + dv).astype(np.float32), cv2.INTER_NEAREST, borderValue=0) > 0

    for s in np.linspace(-1.0, 1.0, 9):
        ok &= sample(-rv * float(s * half_w), ru * float(s * half_w))
    for s in (-1.0, 1.0):
        for t in (-1.0, 0.0, 1.0):
            ok &= sample(ru * float(s * half_t) - rv * float(t * half_w), rv * float(s * half_t) + ru * float(t * half_w))
    return ok


def _runs(flags) -> list:
    """[(i0, i1), ...] index ranges (inclusive) of consecutive True values."""
    f = np.concatenate([[False], np.asarray(flags, dtype=bool), [False]])
    d = np.diff(f.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def fill_strokes(mask, pan, ppm: float, step_mm: float = 6.5, ring_mm: float = 55.0, overlap_mm: float = 2.0,
                 max_len_mm: float = RD.LOAD_MM - 5.0, min_pass_mm: float = 3.0, press: float = 1.0,
                 wide_mm: float = RD.BRUSH_WIDE_MM, margin_mm: float = 0.5) -> list:
    """Zig-zag strokes [{"pts": [[u, v], ...], "press": p}, ...] that fill the mask (see the module text).

    step_mm: sideways distance between neighbouring passes (the band is about 9 mm wide: 6.5 leaves no gaps).
    ring_mm: radial depth of one ring (= longest pass). max_len_mm: a zig-zag is cut between two passes before
    it gets longer than this (one load of paint)."""
    valid = valid_centres(mask, pan, ppm, wide_mm=wide_mm, margin_mm=margin_mm)
    if not valid.any():
        return []
    r, th, _ru, _rv = _grids(mask.shape, ppm, pan)
    hh, ww = mask.shape
    r_lo, r_hi = float(r[valid].min()), float(r[valid].max())
    n_ring = max(1, int(math.ceil((r_hi - r_lo) / ring_mm)))
    edges = np.linspace(r_lo, r_hi, n_ring + 1)
    strokes = []

    def at(rr, t):
        """pixel indices of the points at radii rr on the ray of angle t"""
        uu = (float(pan[0]) + rr * math.sin(t)) * ppm
        vv = (float(pan[1]) + rr * math.cos(t)) * ppm
        return np.round(uu - 0.5).astype(int), np.round(vv - 0.5).astype(int)

    for k in range(n_ring):
        ra = edges[k] - (overlap_mm if k > 0 else 0.0)
        rb = edges[k + 1]
        sel = valid & (r >= ra) & (r <= rb)
        if not sel.any():
            continue
        rm = 0.5 * (ra + rb)
        t_lo, t_hi = float(th[sel].min()), float(th[sel].max())
        n = max(1, int(math.ceil((t_hi - t_lo) * rm / step_mm)) + 1)
        thetas = [t_lo + (t_hi - t_lo) * (i / max(1, n - 1)) for i in range(n)] if n > 1 else [0.5 * (t_lo + t_hi)]
        rr = np.arange(ra, rb + 1e-6, 0.5)
        chains = []                       # each: {"passes": [(theta, r0, r1)], "open": True}
        for t in thetas:
            ui, vi = at(rr, t)
            inside = (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh)
            flags = np.zeros(len(rr), dtype=bool)
            flags[inside] = sel[vi[inside], ui[inside]]
            runs = [(float(rr[a]), float(rr[b])) for a, b in _runs(flags) if rr[b] - rr[a] >= min_pass_mm]
            used = set()
            for ch in chains:
                if not ch["open"]:
                    continue
                _t0, a0, b0 = ch["passes"][-1]
                best, best_ov = None, 0.0
                for j, (a1, b1) in enumerate(runs):
                    if j in used:
                        continue
                    ov = min(b0, b1) - max(a0, a1)
                    if ov > best_ov:
                        best, best_ov = j, ov
                if best is None or best_ov < 2.0:
                    ch["open"] = False
                else:
                    used.add(best)
                    ch["passes"].append((t, runs[best][0], runs[best][1]))
            for j, (a1, b1) in enumerate(runs):
                if j not in used:
                    chains.append({"passes": [(t, a1, b1)], "open": True})
        for ch in chains:
            pts, length = [], 0.0
            pull = True                      # first pass towards the robot (the bristles trail)
            for (t, a, b) in ch["passes"]:
                p_far = [float(pan[0]) + b * math.sin(t), float(pan[1]) + b * math.cos(t)]
                p_near = [float(pan[0]) + a * math.sin(t), float(pan[1]) + a * math.cos(t)]
                seg = [p_far, p_near] if pull else [p_near, p_far]
                add = (b - a) + (math.dist(pts[-1], seg[0]) if pts else 0.0)
                if pts and length + add > max_len_mm:
                    strokes.append({"pts": [[round(x, 1), round(y, 1)] for x, y in pts], "press": press})
                    pts, length, add = [], 0.0, (b - a)
                pts += seg
                length += add
                pull = not pull
            if pts:
                strokes.append({"pts": [[round(x, 1), round(y, 1)] for x, y in pts], "press": press})
    return strokes


def contour_strokes(mask, pan, ppm: float, max_len_mm: float = RD.LOAD_MM - 5.0, min_len_mm: float = 12.0,
                    press: float = 1.0, wide_mm: float = RD.BRUSH_WIDE_MM, margin_mm: float = 0.5,
                    smooth_mm: float = 0.6) -> list:
    """Strokes that run once round every edge of the shape (outer edges and the edges of its holes), with the
    brush just inside: the path is the border of `valid_centres`, so the end of the contact line follows the
    shape's edge. A stroke ALONG an edge leaves a clean edge whatever its direction (the end of the contact line
    sweeps a line parallel to the travel), whereas the ends of the fill's radial passes leave slanted teeth
    wherever an edge is not square to the rays. Painted before the fill, while it can still blend."""
    valid = valid_centres(mask, pan, ppm, wide_mm=wide_mm, margin_mm=margin_mm).astype(np.uint8)
    cnts, _ = cv2.findContours(valid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    strokes = []
    for c in cnts:
        c = cv2.approxPolyDP(c, smooth_mm * ppm, True).reshape(-1, 2).astype(float)
        if len(c) < 3:
            continue
        pts = (c + 0.5) / ppm
        pts = np.vstack([pts, pts[:1]])                       # closed
        if RD.length_mm(pts) < min_len_mm:
            continue
        # start where the path is nearest to the robot's side, so that the first move is a pull along a ray
        cur, length = [pts[0]], 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            d = float(np.linalg.norm(b - a))
            if length + d > max_len_mm and len(cur) > 1:
                strokes.append(polyline(cur, press))
                cur, length = [a], 0.0
            cur.append(b)
            length += d
        if len(cur) > 1 and RD.length_mm(cur) >= 3.0:
            strokes.append(polyline(cur, press))
    return strokes


def radial_dab(u: float, v: float, pan, length_mm: float = 4.0, press: float = 1.0) -> dict:
    """A short pull stroke towards the robot that ends at (u, v): a mark about 9 mm wide and length_mm + 3 mm
    long (a pupil, a nostril, a spot)."""
    rad = RD.radial((u, v), pan)
    a = np.array([u, v]) + rad * length_mm
    return {"pts": [[round(float(a[0]), 1), round(float(a[1]), 1)], [round(float(u), 1), round(float(v), 1)]], "press": press}


def polyline(pts, press: float = 1.0) -> dict:
    return {"pts": [[round(float(u), 1), round(float(v), 1)] for u, v in pts], "press": press}


def painted_mask(strokes, pan, size_mm, ppm: float = 4.0, spread_mm: float = 0.0) -> np.ndarray:
    """What a list of strokes covers (uint8 mask of the canvas): the brush itself, or with spread_mm
    (render.BRUSH_SPREAD_MM) what the paint covered on the real sheet."""
    shape = (int(round(size_mm[1] * ppm)), int(round(size_mm[0] * ppm)))
    out = np.zeros(shape, np.uint8)
    for st in strokes:
        out |= RD.stroke_mask(st["pts"], pan, shape, ppm, press=float(st.get("press", 1.0)), spread_mm=spread_mm)
    return out


# ----------------------------------------------------------------------------- what is left of the white
# How much of an unpainted feature the paint takes on each of its sides, measured on the one feature that was big
# enough to measure: the 26 mm white disc of the Akabeko came out about 23 x 16 mm (half its area). ACROSS the
# rays (where the passes of the fill run past it) it lost 1.5 mm a side; ALONG the rays (where passes start and
# end, the brush dips by itself and the bristles, pressed about a centimetre, lie along the ray) 5 mm a side.
HOLE_LOSS_ACROSS_MM = 1.5
HOLE_LOSS_ALONG_MM = 5.0
# What is left narrower than this counts as closed. On that sheet the eye (17 x 13 mm drawn, 3 to 4 mm left by
# the numbers above) closed almost completely and the 7.5 mm stripes on the back (4.5 left) were slivers; the
# 18 mm disc on the flank (8 mm left) stayed open, smaller than that (about 10 x 6 mm), and so did the 11 mm gap
# between body and head (8 left). The 7 mm crescent on the cheek, which lies across the rays, is flagged; a thin
# white line of it survived over a little more than half its length.
MIN_LEFT_MM = 5.0
NOTCH_MM = 20.0            # an unpainted gap or cut narrower than this, open to the outside, is a notch
NOTCH_MIN_MM2 = 40.0       # ... and at least this large (the rounding of an inner corner is not one)
PART_MIN_MM2 = 15.0        # a part of a feature that closes is listed from this size on


def _ray_frame(points_px, at_mm, pan, ppm: float):
    """Pixel coordinates in the frame of the ray through `at_mm`: (across the ray, along it), in millimetres."""
    r = RD.radial(at_mm, pan)
    t = np.array([-r[1], r[0]])
    p = np.asarray(points_px, dtype=float) / ppm
    return p @ t, p @ r


def _loss_kernel(pan, at_mm, ppm: float, across_mm: float, along_mm: float) -> np.ndarray:
    """The paint that enters a feature at `at_mm`, as a structuring element: an ellipse with the half axes
    across_mm (across the ray) and along_mm (along it)."""
    r = RD.radial(at_mm, pan)
    t = np.array([-r[1], r[0]])
    n = int(math.ceil(max(across_mm, along_mm) * ppm))
    ys, xs = np.mgrid[-n:n + 1, -n:n + 1].astype(float) / ppm
    a, b = xs * t[0] + ys * t[1], xs * r[0] + ys * r[1]
    return (((a / max(across_mm, 1e-6)) ** 2 + (b / max(along_mm, 1e-6)) ** 2) <= 1.0 + 1e-9).astype(np.uint8)


def _extent(sel, origin, at_mm, pan, ppm: float) -> list:
    """[across, along] extent in mm of the set pixels of `sel` (an image whose top left pixel is `origin`)."""
    ys, xs = np.nonzero(sel)
    if not len(xs):
        return [0.0, 0.0]
    a, b = _ray_frame(np.column_stack([xs + origin[0] + 0.5, ys + origin[1] + 0.5]), at_mm, pan, ppm)
    return [round(float(a.max() - a.min()) + 1.0 / ppm, 1), round(float(b.max() - b.min()) + 1.0 / ppm, 1)]


def _narrow(sel, ppm: float) -> float:
    """Width in mm of the widest place of a set of pixels: the diameter of the largest disc that fits in it."""
    if not sel.any():
        return 0.0
    return 2.0 * float(cv2.distanceTransform(sel.astype(np.uint8), cv2.DIST_L2, 5).max()) / ppm


def design_report(mask, pan, ppm: float, across_mm: float = HOLE_LOSS_ACROSS_MM, along_mm: float = HOLE_LOSS_ALONG_MM,
                  min_left_mm: float = MIN_LEFT_MM) -> dict:
    """The unpainted holes and notches of a shape (mask: uint8 image of the canvas, non-zero = painted) and what
    will be left of each once the flat brush has been there.

    A hole is unpainted paper with paint all round; a notch is an unpainted gap or cut narrower than NOTCH_MM
    that is open to the outside (a stripe cut into an edge, the gap between two parts). The paint takes
    across_mm from each side of a feature across the ray from the robot's pan axis and along_mm along it (the
    numbers of the Akabeko, see above). Where what is left is at least min_left_mm wide the feature stays open;
    a feature, or a part of one (the narrow arm of a gap), where it is not is flagged: it will close, or be a
    sliver.

    -> {"ok": nothing closes, "closing": how many features or parts do,
        "features": [{"kind": "hole" | "notch", "closes": bool, "at_mm": [u, v] (its middle),
                      "size_mm": [across, along] as drawn (its extent across the ray and along it),
                      "narrow_mm": the width of its widest place as drawn,
                      "left_mm": [across, along] after the brush ([0, 0]: nothing),
                      "left_narrow_mm": the width of the widest place of what is left,
                      "left_part": the share of its area that is left}, ...]}
    Holes first, then notches, each from the largest down. Made for the flat brush as it was run on 2026-11-13;
    another brush or press: pass its own losses."""
    painted = (np.asarray(mask) > 0).astype(np.uint8)
    hh, ww = painted.shape
    white = 1 - painted
    # holes: unpainted pieces that do not reach the edge of the canvas
    n, lab = cv2.connectedComponents(white, connectivity=4)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    holes = np.isin(lab, [k for k in range(1, n) if k not in edge]).astype(np.uint8)
    # notches: the unpainted paper outside the holes where a disc of NOTCH_MM does not fit
    # (the canvas edge is not paint: the paper goes on beyond it)
    outside = white & (1 - holes)
    k = int(round(NOTCH_MM * ppm)) | 1
    big = cv2.copyMakeBorder(outside, k, k, k, k, cv2.BORDER_CONSTANT, value=1)
    opened = cv2.morphologyEx(big, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))[k:-k, k:-k]
    notches = outside & (1 - opened)
    km = max(1, int(round(min_left_mm * ppm)))
    stay_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (km, km))
    g = int(math.ceil(max(across_mm, along_mm) * ppm)) + km + 2
    features = []
    for kind, region in (("hole", holes), ("notch", notches)):
        m, lb, stats, cent = cv2.connectedComponentsWithStats(region, connectivity=8)
        rows = []
        for j in range(1, m):
            if float(stats[j, cv2.CC_STAT_AREA]) / (ppm * ppm) < (NOTCH_MIN_MM2 if kind == "notch" else 1.0):
                continue
            x0, y0 = int(stats[j, cv2.CC_STAT_LEFT]), int(stats[j, cv2.CC_STAT_TOP])
            X0, Y0 = max(0, x0 - g), max(0, y0 - g)
            X1, Y1 = min(ww, x0 + int(stats[j, cv2.CC_STAT_WIDTH]) + g), min(hh, y0 + int(stats[j, cv2.CC_STAT_HEIGHT]) + g)
            piece = (lb[Y0:Y1, X0:X1] == j).astype(np.uint8)
            paper = white[Y0:Y1, X0:X1]
            mid = ((float(cent[j][0]) + 0.5) / ppm, (float(cent[j][1]) + 0.5) / ppm)
            loss = _loss_kernel(pan, mid, ppm, across_mm, along_mm)
            # the paint comes in from the painted sides only: erode the unpainted paper, keep what lies in the feature
            left = cv2.erode(paper, loss, borderType=cv2.BORDER_CONSTANT, borderValue=1) & piece
            # what is left stays open where a disc of min_left_mm fits into it; that part of the drawn feature is
            # the feature that survives, the rest of it closes
            stay = cv2.morphologyEx(left, cv2.MORPH_OPEN, stay_disc)
            alive = cv2.dilate(stay, loss) & piece
            for closes, part in ((False, alive), (True, piece & (1 - alive))):
                c, cl, cs, cc = cv2.connectedComponentsWithStats(part, connectivity=8)
                for i in range(1, c):
                    area = float(cs[i, cv2.CC_STAT_AREA]) / (ppm * ppm)
                    sel = cl == i
                    narrow = _narrow(sel, ppm)
                    if closes and (area < PART_MIN_MM2 or (alive.any() and narrow < 2.0 * across_mm)):
                        continue                      # (the tip of a corner; a rim or wedge beside a part that stays open)
                    at = ((float(cc[i][0]) + X0 + 0.5) / ppm, (float(cc[i][1]) + Y0 + 0.5) / ppm)
                    rest = left.astype(bool) & sel
                    rows.append((area, {
                        "kind": kind, "closes": closes, "at_mm": [round(at[0], 1), round(at[1], 1)],
                        "size_mm": _extent(sel, (X0, Y0), mid, pan, ppm),
                        "narrow_mm": round(narrow, 1),
                        "left_mm": _extent(rest, (X0, Y0), mid, pan, ppm),
                        "left_narrow_mm": round(_narrow(rest, ppm), 1),
                        "left_part": round(float(rest.sum()) / float(sel.sum()), 2)}))
        features += [row for _area, row in sorted(rows, key=lambda r: -r[0])]
    closing = sum(1 for f in features if f["closes"])
    return {"ok": closing == 0, "closing": closing, "features": features}
