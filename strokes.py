"""Stroke planning for a tool that draws on the table (a pen on the gripper): pure geometry.

Ported and generalised from the scripts that drew the cat on 2026-11-11 (tools/pen_draw/draw.py,
map_fit.py). Nothing here knows the robot, the camera or `fd`: numpy only, so every rule can be
tested without hardware and penskill.py (stage 2) decides what to do with the result.

What the evening taught, and where it lives here:
- The drawing is laid out on the sheet as the CAMERA sees it (u right, v up) and turned with a
  sheet that lies askew: layout(), fit_box().
- Where the ink lands differs from where the model puts the tool tip by up to 16.6 mm, differently
  left and right; a measured map commanded -> actual fixes it: fit_map(), apply_map(), invert_map().
- Which way of travel loses ink is NOT known. (This file first said "a pen pushed along its lean
  chatters"; the one real drawing, measured, says the opposite: 4-5 % of the ink missing on lines
  drawn away from the robot, 11-12 % on lines drawn towards it. One drawing, one pen: no rule.)
  pull_pieces() can cut a stroke so that no piece travels more than a few millimetres along a given
  direction field ("lean"); whether that is used, and with which direction, comes from the tool's
  own direction test (penskill.stroke_policy), never from an assumption. The lean is a function of
  the place (a pen that leans in the arm's vertical plane points radially away from the pan axis:
  radial_lean()), not "+x".
- The arm rounds corners; it must stop at sharp ones: split_corners().

Units: metres on the table (robot base frame: x away from the robot towards the camera, y to the
robot's left), millimetres inside a drawing file. Errors are ValueError (this module must not import
robot.SafetyError); the caller turns them into refusals.

Drawing file: {"box_mm": [w, h], "strokes": [[[u, v], ...], ...]}, origin at the box centre.
Tests: test_strokes.py.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRAWINGS_DIR = os.path.join(HERE, "drawings")
# The pan axis of the SO-101 in the base frame (kinematics._ORIGINS[0]): a tool that leans in the
# arm's vertical plane points away from (or towards) this axis wherever the arm stands.
PAN_AXIS_XY = (0.0388, 0.0)
# Where fit_box looks when it is given no sheet: the part of the table in front of the robot.
DEFAULT_REGION = (0.10, -0.20, 0.40, 0.20)


# --------------------------------------------------------------------------- small helpers
def _poly(poly) -> np.ndarray:
    """(N, 2) float copy of a polyline with repeated points removed (a zero-length segment has no
    direction, which every rule below needs)."""
    p = np.asarray(poly, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or len(p) == 0:
        raise ValueError(f"a polyline must be an (N, 2) array of points, got shape {p.shape}")
    p = p[:, :2]
    if not np.all(np.isfinite(p)):
        raise ValueError("a polyline must hold finite numbers only")
    if len(p) > 1:
        keep = np.concatenate([[True], np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-12])
        p = p[keep]
    return p.copy()


def length(poly) -> float:
    """Path length of a polyline."""
    p = np.asarray(poly, dtype=float)
    return float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def _turns(p: np.ndarray) -> np.ndarray:
    """Direction change (degrees, 0..180) at every interior vertex of a polyline without repeats."""
    if len(p) < 3:
        return np.zeros(0)
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1)
    c = np.einsum("ij,ij->i", d[:-1], d[1:]) / np.maximum(n[:-1] * n[1:], 1e-300)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _axes(rot_deg: float):
    """Table directions of the drawing's u (right in the photo) and v (up in the photo).

    The camera faces the robot, so u is +y and v is -x; rot_deg turns both counter-clockwise seen
    from above (the sheet of 2026-11-11 lay 6 degrees askew)."""
    th = math.radians(float(rot_deg))
    return (np.array([-math.sin(th), math.cos(th)]), np.array([-math.cos(th), -math.sin(th)]))


# --------------------------------------------------------------------------- drawing files
def names() -> list:
    """Names of the drawings that ship with the harness (drawings/*.json)."""
    try:
        return sorted(f[:-5] for f in os.listdir(DRAWINGS_DIR) if f.endswith(".json"))
    except OSError:
        return []


def load(path_or_name) -> dict:
    """A drawing: {"name", "path", "box_mm": [w, h], "strokes": [np.ndarray (N, 2) mm, ...]}.

    path_or_name: a file, a name in drawings/ ("cat"), or an already parsed dict. The file is
    checked here, before anything is planned: a stroke outside its own box would be laid out
    outside the area that was checked for sheet, reach and view. ValueError says what is wrong."""
    if isinstance(path_or_name, dict):
        job, path, name = path_or_name, path_or_name.get("path"), path_or_name.get("name")
    else:
        s = os.fspath(path_or_name)
        cands = [s, os.path.join(DRAWINGS_DIR, s), os.path.join(DRAWINGS_DIR, s + ".json")]
        path = next((c for c in cands if os.path.isfile(c)), None)
        if path is None:
            raise ValueError(f"drawing {s!r} not found: neither a file nor one of {names()} in {DRAWINGS_DIR}")
        with open(path) as fh:
            job = json.load(fh)
        name = os.path.splitext(os.path.basename(path))[0]
    if not isinstance(job, dict) or "strokes" not in job:
        raise ValueError('a drawing is {"box_mm": [w, h], "strokes": [[[u, v], ...], ...]}')
    box = job.get("box_mm", [120, 120])
    try:
        w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        raise ValueError(f"box_mm must be two numbers, got {box!r}") from None
    if not (math.isfinite(w) and math.isfinite(h) and w > 0 and h > 0):
        raise ValueError(f"box_mm must be two positive numbers, got {box!r}")
    strokes = []
    for i, st in enumerate(job["strokes"]):
        try:
            a = _poly(st)
        except (TypeError, ValueError) as e:
            raise ValueError(f"stroke {i}: {e}") from None
        if np.any(np.abs(a[:, 0]) > w / 2 + 0.5) or np.any(np.abs(a[:, 1]) > h / 2 + 0.5):
            raise ValueError(f"stroke {i} leaves the box of {w:g} x {h:g} mm (origin at the box centre)")
        strokes.append(a)
    if not strokes:
        raise ValueError("the drawing has no strokes")
    return {"name": job.get("name", name), "path": path, "box_mm": [w, h], "strokes": strokes}


# --------------------------------------------------------------------------- layout on the table
def layout(drawing, centre_xy, half_m, rot_deg=0.0) -> list:
    """The strokes in table coordinates (metres): the longer side of the drawing's box becomes
    2 * half_m, u goes to +y and v (up in the photo) to -x, then everything is turned by rot_deg
    about the box centre (counter-clockwise seen from above) to follow a sheet that lies askew."""
    w, h = (float(v) for v in drawing["box_mm"])
    s = 2.0 * float(half_m) / max(w, h)
    u_hat, v_hat = _axes(rot_deg)
    c = np.asarray(centre_xy, dtype=float)[:2]
    out = []
    for st in drawing["strokes"]:
        a = np.asarray(st, dtype=float).reshape(-1, 2)
        out.append(c + s * (a[:, :1] * u_hat + a[:, 1:2] * v_hat))
    return out


def box_corners(centre_xy, half_m, rot_deg=0.0, box_mm=None) -> np.ndarray:
    """Corners of the drawing box on the table, in the order top-left, top-right, bottom-right,
    bottom-left as the camera sees the sheet. box_mm: [w, h] of a box that is not square (the
    longer side is 2 * half_m, as in layout)."""
    ru, rv = _aspect(box_mm)
    u_hat, v_hat = _axes(rot_deg)
    c = np.asarray(centre_xy, dtype=float)[:2]
    hu, hv = float(half_m) * ru, float(half_m) * rv
    return np.array([c - hu * u_hat + hv * v_hat, c + hu * u_hat + hv * v_hat,
                     c + hu * u_hat - hv * v_hat, c - hu * u_hat - hv * v_hat])


def _aspect(box_mm):
    if box_mm is None:
        return 1.0, 1.0
    w, h = (float(v) for v in box_mm)
    return w / max(w, h), h / max(w, h)


def sheet_rot_deg(sheet) -> float:
    """How far a sheet is turned on the table, in (-45, 45] degrees: the direction of its edges
    (all four, weighted by length; a rectangle's edges agree modulo 90 degrees)."""
    p = np.asarray(sheet, dtype=float).reshape(-1, 2)
    e = np.roll(p, -1, axis=0) - p
    ang = np.arctan2(e[:, 1], e[:, 0])
    w = np.linalg.norm(e, axis=1)
    z = np.sum(w * np.exp(4j * ang))
    if abs(z) < 1e-12:
        return 0.0
    r = math.degrees(math.atan2(z.imag, z.real)) / 4.0
    return 45.0 if r <= -45.0 else r


def _in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test of points (N, 2) against a polygon (M, 2)."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    a, b = poly, np.roll(poly, -1, axis=0)
    for (x0, y0), (x1, y1) in zip(a, b):
        if y0 == y1:
            continue
        cross = ((y0 > y) != (y1 > y)) & (x < x0 + (y - y0) * (x1 - x0) / (y1 - y0))
        inside ^= cross
    return inside


def _edge_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance of points (N, 2) to the nearest edge of a polygon (M, 2)."""
    best = np.full(len(pts), np.inf)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-300), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
    return best


def _window_all(mask: np.ndarray, ku: int, kv: int) -> np.ndarray:
    """centres[i, j] = True where the (2 ku + 1) x (2 kv + 1) window of the mask is all True."""
    out = np.zeros_like(mask, dtype=bool)
    n, m = mask.shape
    if n < 2 * ku + 1 or m < 2 * kv + 1:
        return out
    s = np.zeros((n + 1, m + 1), dtype=np.int64)
    s[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    tot = (s[2 * ku + 1:, 2 * kv + 1:] - s[:n - 2 * ku, 2 * kv + 1:] - s[2 * ku + 1:, :m - 2 * kv]
           + s[:n - 2 * ku, :m - 2 * kv])
    out[ku:n - ku, kv:m - kv] = tot == (2 * ku + 1) * (2 * kv + 1)
    return out


def fit_box(want_half_m, sheet=None, ok_xy=None, visible_xy=None, margin_m=0.01, rot_deg=None, *,
            box_mm=None, region=None, prefer_xy=None, grid_m=0.004, min_half_m=0.015) -> dict:
    """The largest drawing box, up to want_half_m, that lies on the sheet, in reach and in view.

    sheet: polygon of the sheet on the table ((N, 2), metres) or None (then `region`
        (x0, y0, x1, y1), default DEFAULT_REGION, is searched and no margin applies).
    ok_xy(x, y) -> bool: the tool can work at this place (reach, joint limits, keep-outs);
    visible_xy(x, y) -> bool: the camera sees this place. None = no such limit.
    margin_m: the box stays this far inside the sheet's edge (tape, curl, and the 1-2 cm by which
        ink landed beside its aim before a tool was calibrated).
    rot_deg: turn of the box; None = follow the sheet (sheet_rot_deg), 0 without a sheet.
    box_mm: [w, h] when the drawing is not square. prefer_xy: where the box should be when
        several places allow the same size (default: the middle of the sheet / region).

    The three conditions are tested on a grid of about grid_m over the box (and on its outline),
    so that the WHOLE box holds them, not only its corners: a drawing box of 2026-11-11 reached
    its lower left corner only just. Returns {"ok", "centre": [x, y] | None, "half_m", "rot_deg",
    "reason", "limited_by": [...], "want_half_m", "corners": 4 x [x, y] | None}; ok False (with the
    reason) when not even min_half_m fits. A box smaller than wanted is ok True and says why."""
    want = float(want_half_m)
    if not (math.isfinite(want) and want > 0.0):
        raise ValueError(f"want_half_m must be a positive number, got {want_half_m!r}")
    poly = None if sheet is None else np.asarray(sheet, dtype=float).reshape(-1, 2)
    if poly is not None and len(poly) < 3:
        raise ValueError("sheet must be a polygon of at least 3 points")
    rot = float(rot_deg) if rot_deg is not None else (sheet_rot_deg(poly) if poly is not None else 0.0)
    u_hat, v_hat = _axes(rot)
    ru, rv = _aspect(box_mm)
    if poly is not None:
        outline = poly
    else:
        x0, y0, x1, y1 = region if region is not None else DEFAULT_REGION
        outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    origin = outline.mean(axis=0)
    prefer = origin if prefer_xy is None else np.asarray(prefer_xy, dtype=float)[:2]
    # grid in the box's own axes; the step divides want_half_m so that the wanted size is a grid size
    k_max = max(1, int(math.ceil(want / float(grid_m) - 1e-9)))
    g = want / k_max
    k_min = max(1, int(math.ceil(float(min_half_m) / g - 1e-9)))
    a = (outline - origin) @ u_hat
    b = (outline - origin) @ v_hat
    ga = np.arange(math.floor(a.min() / g), math.ceil(a.max() / g) + 1) * g
    gb = np.arange(math.floor(b.min() / g), math.ceil(b.max() / g) + 1) * g
    aa, bb = np.meshgrid(ga, gb, indexing="ij")
    pts = origin + aa.reshape(-1, 1) * u_hat + bb.reshape(-1, 1) * v_hat

    def holds(fn, p):
        return fn is None or bool(fn(float(p[0]), float(p[1])))

    if poly is not None:
        on_sheet = _in_polygon(pts, poly) & (_edge_distance(pts, poly) >= float(margin_m) - 1e-9)
    else:
        on_sheet = _in_polygon(pts, outline) | (_edge_distance(pts, outline) < 1e-9)
    reach = np.zeros(len(pts), dtype=bool)
    view = np.zeros(len(pts), dtype=bool)
    for i in np.flatnonzero(on_sheet):                     # the costly tests only where the sheet is
        reach[i] = holds(ok_xy, pts[i])
        view[i] = holds(visible_xy, pts[i])
    masks = {"sheet": on_sheet.reshape(aa.shape), "reach": (on_sheet & reach).reshape(aa.shape),
             "view": (on_sheet & view).reshape(aa.shape)}
    good = masks["sheet"] & masks["reach"] & masks["view"]

    def verified(c, half):
        """The outline of the box itself, sampled finer than the grid."""
        cs = box_corners(c, half, rot, box_mm)
        for p0, p1 in zip(cs, np.roll(cs, -1, axis=0)):
            n = max(2, int(math.ceil(float(np.linalg.norm(p1 - p0)) / (g / 2.0))))
            line = p0 + (p1 - p0) * (np.arange(n) / n)[:, None]
            if poly is not None and not (np.all(_in_polygon(line, poly))
                                         and np.all(_edge_distance(line, poly) >= float(margin_m) - 1e-6)):
                return False
            if not all(holds(ok_xy, p) and holds(visible_xy, p) for p in line):
                return False
        return True

    def ks(k):                  # half window in grid nodes; rounded up, so the window covers the whole box
        return max(1, int(math.ceil(k * ru - 1e-9))), max(1, int(math.ceil(k * rv - 1e-9)))

    found = None
    for k in range(k_max, k_min - 1, -1):
        centres = _window_all(good, *ks(k))
        if not centres.any():
            continue
        idx = np.argwhere(centres)
        cand = origin + ga[idx[:, 0], None] * u_hat + gb[idx[:, 1], None] * v_hat
        for j in np.argsort(np.linalg.norm(cand - prefer, axis=1))[:40]:
            if verified(cand[j], k * g):
                found = (k, cand[j])
                break
        if found:
            break
    share = {n: float(m.sum()) / max(1, int(masks["sheet"].sum())) for n, m in masks.items()}
    if found is None:
        why = [n for n in ("sheet", "reach", "view") if not _window_all(masks[n], *ks(k_min)).any()]
        if not masks["sheet"].any():
            reason = f"the sheet is too small for a margin of {float(margin_m) * 1000:.0f} mm"
        else:
            reason = (f"no place for a drawing box of at least {2 * k_min * g * 1000:.0f} mm: "
                      + (" and ".join({"sheet": "the sheet is too small", "reach": "too little of the sheet is in reach",
                                       "view": "too little of the sheet is in the camera's view"}[n] for n in why)
                         or "sheet, reach and view do not overlap enough")
                      + f" (in reach {share['reach'] * 100:.0f} %, in view {share['view'] * 100:.0f} % of the sheet)")
        return {"ok": False, "centre": None, "half_m": 0.0, "rot_deg": rot, "reason": reason, "limited_by": why,
                "want_half_m": want, "corners": None}
    k, c = found
    half = want if k == k_max else k * g
    limited = []
    if k < k_max:
        limited = [n for n in ("sheet", "reach", "view") if not _window_all(masks[n], *ks(k + 1)).any()] or ["overlap"]
    words = {"sheet": "the sheet", "reach": "reach", "view": "the camera's view", "overlap": "sheet, reach and view together"}
    reason = "" if not limited else (f"box {2 * half * 1000:.0f} mm instead of {2 * want * 1000:.0f} mm: limited by "
                                     + ", ".join(words[n] for n in limited))
    return {"ok": True, "centre": [float(c[0]), float(c[1])], "half_m": float(half), "rot_deg": rot, "reason": reason,
            "limited_by": limited, "want_half_m": want, "corners": box_corners(c, half, rot, box_mm).tolist()}


# --------------------------------------------------------------------------- polylines
def resample(poly, step_m, corner_deg: float = 25.0) -> np.ndarray:
    """The polyline with points about step_m apart (never further), corners kept.

    Vertices where the direction turns by more than corner_deg stay exactly where they are; between
    them the points are spread evenly ALONG the original line, so a dense curve is thinned and a
    long straight segment gets points (the lean and the correction map change along it). The night's
    version only thinned."""
    p = _poly(poly)
    step = float(step_m)
    if len(p) < 2 or not step > 0.0:
        return p
    keys = [0] + [int(i) + 1 for i in np.flatnonzero(_turns(p) > float(corner_deg))] + [len(p) - 1]
    out = [p[0]]
    for a, b in zip(keys[:-1], keys[1:]):
        seg = p[a:b + 1]
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
        n = max(1, int(math.ceil(cum[-1] / step - 1e-9)))
        s = cum[-1] * np.arange(1, n) / n
        out.extend(np.stack([np.interp(s, cum, seg[:, 0]), np.interp(s, cum, seg[:, 1])], axis=1))
        out.append(seg[-1])
    return np.array(out)


def split_corners(poly, deg: float = 55.0) -> list:
    """Sub-polylines that end at corners sharper than deg. One motion of the arm rounds a corner
    (the reference is blended over 0.2 s); stopping there keeps it sharp. The parts share the corner
    point, so drawn one after the other without lifting they are the same line."""
    p = _poly(poly)
    if len(p) < 3:
        return [p]
    cuts = [0] + [int(i) + 1 for i in np.flatnonzero(_turns(p) > float(deg))] + [len(p) - 1]
    return [p[a:b + 1] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


# --------------------------------------------------------------------------- lean and pull strokes
def radial_lean(axis_xy=PAN_AXIS_XY, outward: bool = True):
    """lean(p) of a tool that leans in the arm's vertical plane: at p its tip points away from the
    pan axis (outward, the pen of 2026-11-11 at gripper pitch 75: 15 degrees) or towards it.
    tooling.lean_xy supplies the real thing for an attached tool; this is the model for planning
    and tests."""
    ax = np.asarray(axis_xy, dtype=float)[:2]
    sgn = 1.0 if outward else -1.0

    def lean(p):
        v = np.asarray(p, dtype=float)[:2] - ax
        n = float(np.hypot(v[0], v[1]))
        return sgn * v / n if n > 1e-9 else np.zeros(2)

    return lean


def constant_lean(vec):
    """lean(p) that is the same everywhere (the night's assumption was (1, 0): "+x")."""
    v = np.asarray(vec, dtype=float)[:2]
    n = float(np.hypot(v[0], v[1]))
    v = v / n if n > 1e-12 else np.zeros(2)
    return lambda p: v


def direction_lean(prefer, axis_xy=PAN_AXIS_XY):
    """The `lean` to hand to pull_pieces / plan_pieces for a stroke-direction policy (SPEC 9.1: which
    direction of travel keeps its ink is MEASURED per tool, tool.calibrated["direction"]["prefer"], not a
    built-in rule). "toward": no piece travels more than min_push_m away from the robot (from the pan
    axis); "away": none travels more than that towards it; "as-is" / "any" / None: no cutting at all
    (returns None, which plan_pieces takes as "draw the strokes as the file has them")."""
    if prefer in (None, "as-is", "any"):
        return None
    if prefer == "toward":
        return radial_lean(axis_xy, outward=True)
    if prefer == "away":
        return radial_lean(axis_xy, outward=False)
    raise ValueError(f"stroke direction {prefer!r}: one of as-is, any, away, toward")


def advance(poly, lean) -> np.ndarray:
    """Travel of every segment along the lean (metres; positive = the tip is pushed ahead of the
    tool). The lean is taken at the middle of each segment, so resample() long segments first."""
    p = np.asarray(poly, dtype=float)[:, :2]
    if len(p) < 2:
        return np.zeros(0)
    mid = 0.5 * (p[1:] + p[:-1])
    ln = np.array([np.asarray(lean(m), dtype=float)[:2] for m in mid]).reshape(-1, 2)
    return np.einsum("ij,ij->i", p[1:] - p[:-1], ln)


def push_travel(poly, lean) -> float:
    """How far the tool is pushed along its lean while it follows the polyline: the sum of the
    positive advances (275 mm for the cat of 2026-11-11 as it was drawn)."""
    return float(np.clip(advance(poly, lean), 0.0, None).sum())


def push_excursion(poly, lean) -> float:
    """The longest net travel along the lean of any stretch of the polyline (what pull_pieces bounds:
    a wiggle that goes 2 mm forth and back counts 2 mm, a steady push counts its whole length)."""
    s = np.concatenate([[0.0], np.cumsum(advance(poly, lean))])
    return float(np.max(s - np.minimum.accumulate(s)))


def _turning_points(s: np.ndarray, h: float) -> list:
    """Indices where the sequence s really turns (it comes back by more than h from an extreme),
    with 0 and len - 1. Between two of them s rises or falls with at most h of back-travel; the
    stretch before the first turn is part of the first leg (it never spans more than h)."""
    cuts, direction, ext = [0], 0, 0
    i_hi = i_lo = 0
    for i in range(1, len(s)):
        if direction == 0:
            if s[i] > s[i_hi]:
                i_hi = i
            if s[i] < s[i_lo]:
                i_lo = i
            if s[i_hi] - s[i] > h:
                direction, ext = -1, i
            elif s[i] - s[i_lo] > h:
                direction, ext = 1, i
        elif direction > 0:
            if s[i] >= s[ext]:
                ext = i
            elif s[ext] - s[i] > h:
                cuts.append(ext)
                direction, ext = -1, i
        else:
            if s[i] <= s[ext]:
                ext = i
            elif s[i] - s[ext] > h:
                cuts.append(ext)
                direction, ext = 1, i
    cuts.append(len(s) - 1)
    return cuts


def _walk(p: np.ndarray, i: int, dist: float, step: int) -> list:
    """Points met when walking `dist` along the polyline from vertex i (step +1: onwards, -1: back):
    the vertices passed and the point where the distance is used up. Stops at the polyline's end."""
    out = []
    while dist > 1e-12 and 0 <= i + step < len(p):
        d = p[i + step] - p[i]
        n = float(np.linalg.norm(d))
        if n >= dist:
            out.append(p[i] + d * (dist / n))
            break
        out.append(p[i + step].copy())
        dist -= n
        i += step
    return out


# A piece is turned round only when that saves more push than this (a sideways line keeps its direction).
_SAME_PUSH_M = 0.0005


def _legs_to_pieces(p, s, lean, h, min_push, e, merge, depth=0) -> list:
    cuts = _turning_points(s, h)
    legs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    if merge and len(legs) > 1:
        # a short push is not worth lifting the pen: neighbours are joined while the joined piece,
        # drawn from its better end and with its overlaps, still obeys min_push
        room = min_push - 2.0 * e
        joined = [legs[0]]
        for a, b in legs[1:]:
            ss = s[joined[-1][0]:b + 1]
            up = float(np.max(ss - np.minimum.accumulate(ss)))
            down = float(np.max(np.maximum.accumulate(ss) - ss))
            if min(up, down) <= room:
                joined[-1] = (joined[-1][0], b)
            else:
                joined.append((a, b))
        legs = joined
    out = []
    for a, b in legs:
        core = p[a:b + 1]
        head = _walk(p, a, e, -1)[::-1] if a > 0 else []
        tail = _walk(p, b, e, +1) if b < len(p) - 1 else []
        done = None
        for piece in ((np.array([*head, *core, *tail]), core) if (head or tail) else (core,)):
            fwd, rev = push_excursion(piece, lean), push_excursion(piece[::-1], lean)
            if fwd <= min_push + 1e-12 and fwd <= rev + _SAME_PUSH_M:      # the drawing's own direction is as good
                done = piece.copy()
            elif rev <= min_push + 1e-12:
                done = piece[::-1].copy()
            elif fwd <= min_push + 1e-12:
                done = piece.copy()
            if done is not None:
                break
        if done is not None:
            out.append(done)
        elif len(core) > 2 and depth < 40:
            # neither end works (only with a hysteresis wider than min_push allows): cut finer
            out.extend(_legs_to_pieces(core, s[a:b + 1] - s[a], lean, h / 2.0 if h > 1e-6 else 0.0, min_push, 0.0,
                                       False, depth + 1))
        else:
            out.append(core[::-1].copy() if push_excursion(core, lean) > push_excursion(core[::-1], lean) else core.copy())
    return out


def pull_pieces(poly, lean, hyst_m=0.003, min_push_m=0.006, overlap_m=0.001, merge: bool = False) -> list:
    """Pieces of a polyline, each to be drawn from its first point to its last, so that the tool
    is pulled or moved sideways: NO PIECE TRAVELS MORE THAN min_push_m ALONG THE LEAN
    (push_excursion(piece, lean) <= min_push_m, for every polyline and every lean).

    lean(p) -> unit 2-vector on the table: where the tool tip points to, seen from the tool's
    body, at place p (tooling.lean_xy; radial_lean() for a pen in the arm's plane). Moved that
    way the tip leads ("pushed"), moved the other way it trails ("pulled"). Which of the two loses
    ink is NOT known (the one real drawing lost LESS when pushed; see the module text): give this
    function the direction that the tool's direction test found worse, or do not use it.

    The polyline is cut where its travel along the lean really turns (hyst_m of slack, so that a
    wiggle or a stair-stepped sideways line is not chopped up), and every piece is drawn from the
    end that pushes it less (the drawing's own direction when that makes no difference). Each piece
    is a pen-down of its own anyway, so turning one round costs nothing: what is left of pushing
    is the back-travel inside the slack, a few millimetres at most (the night's version left pushes
    of up to min_push_m as they were: 30 mm in total for the cat; this one leaves 11 mm). At every
    cut the two pieces share overlap_m of line (each reaches overlap_m / 2 past the cut), so that
    the join closes although the pen does not land exactly where it lifted. merge=True joins
    neighbouring pieces as long as the invariant holds: fewer pen lifts (about 28 instead of 34 for
    the cat) for more of the allowed pushing.

    Give it points about a millimetre apart (resample): the lean is read at the middle of each
    segment. Cut in INK coordinates and pre-distort the pieces afterwards (invert_map): the lean
    belongs to the place where the pen really is. A lean of (0, 0) (an upright tool) cuts nothing."""
    p = _poly(poly)
    if len(p) < 2:
        return [p]
    min_push = max(0.0, float(min_push_m))
    h = min(max(0.0, float(hyst_m)), min_push)
    e = min(max(0.0, float(overlap_m)), min_push) / 2.0
    s = np.concatenate([[0.0], np.cumsum(advance(p, lean))])
    return _legs_to_pieces(p, s, lean, h, min_push, e, bool(merge))


def stats(pieces, lean=None) -> dict:
    """What a list of pieces costs: {"count", "points", "ink_m" (line drawn), "penup_m" (travel in
    the air between the pieces, straight from each end to the next start), "push_m" (push travel
    left) and "push_max_m" (the worst piece's push_excursion); the last two are None without lean}."""
    pcs = [np.asarray(q, dtype=float)[:, :2] for q in pieces]
    ink = sum(length(q) for q in pcs)
    penup = sum(float(np.linalg.norm(b[0] - a[-1])) for a, b in zip(pcs[:-1], pcs[1:]))
    out = {"count": len(pcs), "points": int(sum(len(q) for q in pcs)), "ink_m": float(ink), "penup_m": float(penup),
           "push_m": None, "push_max_m": None}
    if lean is not None:
        out["push_m"] = float(sum(push_travel(q, lean) for q in pcs))
        out["push_max_m"] = float(max((push_excursion(q, lean) for q in pcs), default=0.0))
    return out


def order_pieces(pieces, start_xy=None) -> list:
    """Indices of the pieces in an order that keeps the travel in the air short: always the piece
    that STARTS nearest to where the tool is (start_xy, default: the first piece's start). No piece
    is turned round: its direction is what makes it a pull. It saves little (cat: 820 -> 754 mm):
    most of the travel is the way back out that pulling costs, whatever the order."""
    pcs = [np.asarray(q, dtype=float)[:, :2] for q in pieces]
    if not pcs:
        return []
    here = pcs[0][0] if start_xy is None else np.asarray(start_xy, dtype=float)[:2]
    left, out = list(range(len(pcs))), []
    while left:
        i = min(left, key=lambda k: (float(np.linalg.norm(pcs[k][0] - here)), k))
        left.remove(i)
        out.append(i)
        here = pcs[i][-1]
    return out


# --------------------------------------------------------------------------- correction map
def _map_parts(corr_map):
    """(centre, coef (3 or 6, 2), quad) of a map in tooling.Tool.corr_map form
    ({"centre", "kind", "coef"}) or as the night's scripts stored it ({"centre", "use", "affine":
    {"coef"}, "quad": {"coef"}})."""
    if "kind" in corr_map:
        kind, coef = corr_map["kind"], corr_map["coef"]
    elif "use" in corr_map:
        kind, coef = corr_map["use"], corr_map[corr_map["use"]]["coef"]
    else:
        raise ValueError('a correction map needs "kind" ("affine" | "quad") and "coef"')
    coef = np.asarray(coef, dtype=float)
    if kind not in ("affine", "quad") or coef.shape != ((6, 2) if kind == "quad" else (3, 2)):
        raise ValueError(f"correction map of kind {kind!r} with coefficients of shape {coef.shape}")
    return np.asarray(corr_map["centre"], dtype=float)[:2], coef, kind == "quad"


def _feats(c: np.ndarray, centre: np.ndarray, quad: bool) -> np.ndarray:
    x, y = c[:, 0] - centre[0], c[:, 1] - centre[1]
    cols = [np.ones_like(x), x, y] + ([x * x, x * y, y * y] if quad else [])
    return np.stack(cols, axis=1)


def apply_map(points, corr_map) -> np.ndarray:
    """Where the ink lands when the tool tip is commanded to `points` ((N, 2), metres): the measured
    map commanded -> actual. corr_map None = no map (a copy of the points)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if corr_map is None:
        return pts.copy()
    centre, coef, quad = _map_parts(corr_map)
    return _feats(pts, centre, quad) @ coef


def invert_map(points, corr_map) -> np.ndarray:
    """The commanded positions that make the ink land on `points`: apply_map(invert_map(p)) == p.
    Affine maps are solved exactly, quadratic ones by Newton's method (a few steps: the map is close
    to a shift). ValueError when the map cannot be inverted there (it folds: not a usable calibration)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if corr_map is None:
        return pts.copy()
    centre, coef, quad = _map_parts(corr_map)
    m = coef[1:3].T                                    # d(actual) / d(commanded) at the centre
    if abs(float(np.linalg.det(m))) < 1e-6:
        raise ValueError("the correction map is singular (its linear part has no inverse)")
    c = centre + np.linalg.solve(m, (pts - coef[0]).T).T
    if not quad:
        return c
    for _ in range(30):
        r = pts - _feats(c, centre, True) @ coef
        if float(np.max(np.abs(r))) < 1e-11:
            return c
        x, y = c[:, 0] - centre[0], c[:, 1] - centre[1]
        jx = coef[1] + 2.0 * x[:, None] * coef[3] + y[:, None] * coef[4]       # d(actual)/dx, (N, 2)
        jy = coef[2] + x[:, None] * coef[4] + 2.0 * y[:, None] * coef[5]
        det = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
        if np.any(np.abs(det) < 1e-6):
            raise ValueError("the correction map folds inside the drawing (cannot be inverted there)")
        c = c + np.stack([(r[:, 0] * jy[:, 1] - r[:, 1] * jy[:, 0]) / det,
                          (jx[:, 0] * r[:, 1] - jx[:, 1] * r[:, 0]) / det], axis=1)
    if float(np.max(np.abs(pts - _feats(c, centre, True) @ coef))) > 1e-7:
        raise ValueError("the correction map could not be inverted (no convergence)")
    return c


def fit_map(commanded, actual, centre=None, kind=None) -> dict:
    """The map commanded -> actual from marks: where the tool was sent and where its ink was found.

    Returns tooling.Tool.corr_map: {"centre", "kind", "coef", "rms_mm", "max_mm", "n", "fits":
    {"affine": {...}, "quad": {...}}}. kind None chooses as on 2026-11-11: the quadratic map only
    with nine or more marks and only when it is clearly better (affine rms >= 1.3 x quad rms + 0.8
    mm); six coefficients per axis follow the noise of a few marks all too willingly."""
    c = np.asarray(commanded, dtype=float).reshape(-1, 2)
    a = np.asarray(actual, dtype=float).reshape(-1, 2)
    if len(c) != len(a):
        raise ValueError(f"{len(c)} commanded places but {len(a)} measured ones")
    if len(c) < 3:
        raise ValueError("a correction map needs at least 3 marks")
    c0 = c.mean(axis=0) if centre is None else np.asarray(centre, dtype=float)[:2]
    if np.linalg.matrix_rank(_feats(c, c0, False), tol=1e-9) < 3:
        raise ValueError("the marks lie on one line: they do not fix a map of the plane")
    fits = {}
    for name, quad in (("affine", False), ("quad", True)):
        if quad and len(c) < 6:
            continue
        f = _feats(c, c0, quad)
        coef, *_ = np.linalg.lstsq(f, a, rcond=None)
        res = np.linalg.norm(a - f @ coef, axis=1)
        fits[name] = {"coef": coef.tolist(), "rms_mm": round(float(np.sqrt(np.mean(res ** 2))) * 1000.0, 2),
                      "max_mm": round(float(res.max()) * 1000.0, 2)}
    if kind is None:
        kind = "affine"
        if "quad" in fits and len(c) >= 9 and not fits["affine"]["rms_mm"] < 1.3 * fits["quad"]["rms_mm"] + 0.8:
            kind = "quad"
    if kind not in fits:
        raise ValueError(f"a {kind!r} map cannot be fitted to {len(c)} marks")
    return {"centre": [float(c0[0]), float(c0[1])], "kind": kind, "coef": fits[kind]["coef"],
            "rms_mm": fits[kind]["rms_mm"], "max_mm": fits[kind]["max_mm"], "n": int(len(c)),
            "fits": {k: {"rms_mm": v["rms_mm"], "max_mm": v["max_mm"]} for k, v in fits.items()}}


# --------------------------------------------------------------------------- the whole plan
def plan_pieces(polys_xy, lean=None, corr_map=None, step_m: float = 0.0012, corner_deg: float = 55.0,
                order: bool = False, start_xy=None, **pull_kw) -> dict:
    """From laid-out strokes (where the INK shall be) to what the tool is sent to do.

    Each stroke is resampled, cut into pull pieces (lean given) in ink coordinates and only then
    pre-distorted with the correction map. Returns {"pieces": [{"stroke": i, "ink": (N, 2),
    "cmd": (N, 2), "parts": [cmd sub-polylines that end at sharp corners]}, ...] in drawing order,
    "stats": stats() of the ink pieces, "strokes": number of strokes}. The caller lowers the tool at
    cmd[0], runs the parts one after the other without lifting, and lifts after the last.
    order=True sorts the pieces for short travel in the air (order_pieces, from start_xy); the
    default keeps the drawing's own order of strokes."""
    pieces = []
    for i, st in enumerate(polys_xy):
        ink = resample(st, step_m)
        for piece in (pull_pieces(ink, lean, **pull_kw) if lean is not None else [ink]):
            cmd = invert_map(piece, corr_map)
            pieces.append({"stroke": i, "ink": piece, "cmd": cmd, "parts": split_corners(cmd, corner_deg)})
    if order:
        pieces = [pieces[i] for i in order_pieces([q["ink"] for q in pieces], start_xy)]
    return {"pieces": pieces, "stats": stats([q["ink"] for q in pieces], lean), "strokes": len(polys_xy)}
