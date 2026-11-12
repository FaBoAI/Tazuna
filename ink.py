"""Ink on paper, read from photos of the fixed camera. Pure functions on images and a camera model:
no camera device, no robot (camera.CameraModel is used through its attributes only, never imported).

Why this module exists (2026-11-11): a detector answered "no ink" on photos in which the white paper
was clipped, the calibration believed it and pressed the pen 3 mm deeper per round. So every answer
given here is tri-state:

    True   ink was seen
    False  no ink, AND the photo could have shown it (paper not clipped, level sensible, the place in
           view, not hidden, not already inked)
    None   cannot judge, with the reason in words

and nothing here turns "I saw nothing" into "there is nothing" without that check (can_show_ink).

What is measured: a thin dark line on bright paper, with a black-hat filter (closing minus image), so
that broad shadows give no response. New ink = response in the `after` photo minus the (slightly
dilated) response in the `before` photo, which keeps new ink next to old ink. Thresholds follow the
noise of the paper in the photo at hand (with a floor), not a fixed number, and a faint pixel counts when
the pixels along a line through it are dark too: the extra-fine pen of 2026-11-11 (0.4 mm) answers with
4-8 grey levels, far below the old fixed threshold of 28.

Limits (see the tests): breaks in a line shorter than about 2 pixels (0.5 mm) are blurred shut by the
camera; without a `before` photo the edge of an overlapping sheet reads as a line; what counts as a
"speck" was set between a handful of marks of one evening.

Geometry: pixels are mapped to the table plane at height z with the camera model (a homography for the
pinhole part). The model is only good ON the table plane, which is where ink lies.

Units: metres on the table (base frame: x away from the robot, y to its left), pixels in the image.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

# ----------------------------------------------------------------------------- what a photo must offer
CLIP_LEVEL = 250          # grey level from which a pixel counts as clipped (JPEG leaves 253-255 on blown-out paper)
MAX_CLIPPED = 0.02        # share of clipped pixels in a region above which "no ink" cannot be earned
PAPER_LEVEL_MIN = 90.0    # paper darker than this: under-exposed, or not paper at all (the dark desk)
PAPER_LEVEL_MAX = 242.0   # paper brighter than this sits on the shoulder of the sensor curve: thin lines fade
MIN_PAPER_SHARE = 0.90    # share of a region that must be visible paper (the rest: arm, pen, tape, off the sheet)
MAX_CHROMA = 45           # max - min channel above which a pixel is not white paper (tape, skin, wood)
PAPER_REL = 0.72          # a pixel darker than this share of the paper level (thin lines removed) is not paper
THIN_LINE_RESPONSE = 8.0  # the 0.4 mm pen's better lines (4-8 grey levels measured, paper at 160-175): a photo whose
                          # noise puts the strong threshold above this cannot show such a line

# ----------------------------------------------------------------------------- detection
LINE_PX = 6               # widest line (pixels) that still gives the full response (kernel 2 * LINE_PX + 1)
# Thresholds on the line response (grey levels): never below the floors, and at least K times the pixel noise of
# the paper in the photo at hand. With a `before` photo the response is a difference whose noise on blank paper
# stayed below 3 levels in every photo of 2026-11-11 (the dilated old response is subtracted); without one the
# paper's own grain and the JPEG blocks reach 4-5 levels.
LOW_FLOOR, HIGH_FLOOR = 3.0, 5.0               # with a before photo
LOW_FLOOR_ABS, HIGH_FLOOR_ABS = 5.0, 8.0       # a single photo
K_LOW, K_HIGH = 3.0, 5.0
# The line filter proper: a pixel that is itself only a little dark (PIX) still counts as ink when the mean
# response along the best of 8 directions through it (LINE_LEN pixels) is above LINE_LOW; it is "strong" above
# LINE_HIGH. This is what finds the 0.4 mm pen (4-8 grey levels deep, 1-2 pixels wide, tens of pixels long): on
# blank paper that mean never exceeded 1.8 (difference of two photos) in the photos of 2026-11-11.
LINE_LEN = 11
PIX_MIN, LINE_LOW, LINE_HIGH = 2.0, 1.6, 2.5              # with a before photo
PIX_MIN_ABS, LINE_LOW_ABS, LINE_HIGH_ABS = 3.0, 2.5, 4.0  # a single photo
OLD_INK_GAIN = 1.25       # old ink is subtracted with this gain (it never has exactly the same value twice)
OLD_INK_DILATE = 5        # ... after a grey dilation of this size (the two photos differ by a pixel or two)
EDGE_PX = 7               # ink is not looked for closer than this to something that is not paper
# JPEG rings around a bold line: strongly right next to it, weakly through the whole 8 x 8 block. A pixel must
# reach RING_SHARE of the strongest response within RING_PX pixels and RING_FAR_SHARE of it within RING_FAR_PX,
# or the ringing joins neighbouring marks and grows faint "lines" beside every bold one.
RING_PX, RING_SHARE = 4, 0.28
RING_FAR_PX, RING_FAR_SHARE = 9, 0.07
COVER_SHARE = 0.5         # stroke_coverage: ink weaker than this share of the strongest ink nearby is a gap between dots
MAX_OFFSET_M = 0.025      # ink of an uncalibrated tool lay up to 25 mm from where it was aimed (2026-11-11)
# A rung of a ladder counts as inked when its ink is a LINE (a piece at least SPECK_SHARE of the commanded dash
# long, at most SPECK_MAX_M asked) or a FIRM TOUCH (a compact piece as dark as real ink). Anything less is a
# speck: the pen grazed the paper. Measured 2026-11-11: specks 1.2-2.1 mm long, darkness 0.10-0.21; the shortest
# real rung marks 3.2 mm (darkness 0.48) and a 1.9 mm dot of darkness 0.64.
SPECK_SHARE, SPECK_MAX_M = 0.45, 0.003
TOUCH_DARKNESS, TOUCH_AREA_PX = 0.35, 12
# "As dark as real ink" needs a reference: the darkest piece of the same ladder when that is at least
# REF_DARKNESS (a bold pen), or the caller's ink_darkness. A line must reach GRAZE_REL of the reference, a touch
# TOUCH_REL of it. Without a reference (a faint pen, or nothing but specks) a speck is neither yes nor no.
REF_DARKNESS, GRAZE_REL, TOUCH_REL = 0.30, 0.33, 0.50


@dataclass
class Mark:
    """One connected piece of ink on the paper (or several pieces merged into one mark: pieces > 1)."""
    xy: tuple                 # centre on the table (m)
    length_m: float           # extent along its long axis
    width_m: float            # extent across
    angle_deg: float          # direction of the long axis on the table, atan2(dy, dx), in (-90, 90]
    area_px: int
    bbox_px: tuple            # (u0, v0, u1, v1), u1 / v1 exclusive
    darkness: float           # 0..1 against the paper around it
    confidence: float         # 0..1: how far above the noise and how large
    pieces: int = 1
    extent_xy: tuple = ()     # (x_min, y_min, x_max, y_max) on the table

    def to_dict(self) -> dict:
        d = asdict(self)
        d["xy"] = [round(float(v), 5) for v in self.xy]
        d["extent_xy"] = [round(float(v), 5) for v in self.extent_xy]
        d["bbox_px"] = [int(v) for v in self.bbox_px]
        for k in ("length_m", "width_m"):
            d[k] = round(float(d[k]), 5)
        for k in ("angle_deg", "darkness", "confidence"):
            d[k] = round(float(d[k]), 3)
        return d


# ----------------------------------------------------------------------------- geometry on the table plane
def _homography(cam, z: float) -> np.ndarray:
    """3 x 3 matrix H with pixel ~ H (x, y, 1) for points on the horizontal plane at height z
    (the pinhole part of the model; a distortion term is handled by the callers)."""
    K = np.array([[cam.f, 0.0, cam.cx], [0.0, cam.f, cam.cy], [0.0, 0.0, 1.0]])
    R = np.asarray(cam.R, dtype=float)
    t = np.asarray(cam.tvec, dtype=float).reshape(3)
    return K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * float(z) + t])


def table_xy(cam, uv, z: float) -> np.ndarray:
    """Pixels (N, 2) -> points on the table plane at height z (N, 2). The vectorised form of
    CameraModel.to_plane (which takes one pixel per call: far too slow for every pixel of a mark)."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    if len(uv) == 0:
        return np.zeros((0, 2))
    if getattr(cam, "k1", 0.0):
        uv = cv2.undistortPoints(uv.reshape(-1, 1, 2), cam.K, cam.dist, P=cam.K).reshape(-1, 2)
    p = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(_homography(cam, z)).T
    return p[:, :2] / p[:, 2:3]


def pixels(cam, xy, z: float) -> np.ndarray:
    """Points on the table plane (N, 2) -> pixels (N, 2)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        return np.zeros((0, 2))
    return np.asarray(cam.project(np.column_stack([xy, np.full(len(xy), float(z))])), dtype=float).reshape(-1, 2)


def _norm_region(region):
    x0, y0, x1, y1 = (float(v) for v in region)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _roi(shape, cam, z, region, margin_px: int = 0):
    """Pixel window of a table rectangle: ((u0, v0, u1, v1), polygon mask of the window or None, share of
    the rectangle that is inside the picture). region None = the whole picture."""
    h, w = shape[:2]
    if region is None:
        return (0, 0, w, h), None, 1.0
    x0, y0, x1, y1 = _norm_region(region)
    corners = pixels(cam, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 13), np.linspace(y0, y1, 13))
    g = pixels(cam, np.column_stack([gx.ravel(), gy.ravel()]), z)
    in_view = float(np.mean((g[:, 0] >= 0) & (g[:, 0] <= w - 1) & (g[:, 1] >= 0) & (g[:, 1] <= h - 1)))
    u0 = int(max(0, math.floor(corners[:, 0].min()) - margin_px))
    v0 = int(max(0, math.floor(corners[:, 1].min()) - margin_px))
    u1 = int(min(w, math.ceil(corners[:, 0].max()) + 1 + margin_px))
    v1 = int(min(h, math.ceil(corners[:, 1].max()) + 1 + margin_px))
    if u1 - u0 < 2 or v1 - v0 < 2:
        return (0, 0, 0, 0), None, 0.0
    mask = np.zeros((v1 - v0, u1 - u0), np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners - (u0, v0)).astype(np.int32), 1)
    return (u0, v0, u1, v1), mask, in_view


# ----------------------------------------------------------------------------- the line filter
def _kernel(line_px):
    k = 2 * int(line_px if line_px else LINE_PX) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def ink_map(img, line_px=None) -> np.ndarray:
    """Thin dark lines as a positive image (int16 grey levels, same size as img): closing minus image
    (black-hat). A line up to line_px wide (default 6 pixels, about 2 mm on the table) answers with its
    full depth; anything broad (a shadow, the arm, the desk) answers with nothing."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, _kernel(line_px)).astype(np.int16)


def _analyse(crop, line_px=None) -> dict:
    """Everything the functions below need from one picture window: grey image, line response, the
    image with thin lines removed, what is paper, its level and its noise."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    resp = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, _kernel(line_px))
    closed = cv2.add(g, resp)                                   # = the closing: thin lines filled with paper
    if crop.ndim == 3:
        b, gr, r = cv2.split(crop)
        chroma = cv2.subtract(cv2.max(cv2.max(b, gr), r), cv2.min(cv2.min(b, gr), r))
    else:
        chroma = np.zeros_like(g)
    return {"gray": g, "resp": resp, "closed": closed, "chroma": chroma}


def _paper(a: dict, mask=None) -> dict:
    """Which pixels of an analysed window are paper, the paper's grey level and the noise of the line
    response on it. mask: uint8 window mask (1 = look here) or None."""
    closed, chroma = a["closed"], a["chroma"]
    sel = closed if mask is None else closed[mask > 0]
    if sel.size == 0:
        return {"paper": np.zeros(closed.shape, bool), "level": 0.0, "noise": 0.0, "share": 0.0, "n": 0}
    bright = float(np.percentile(sel, 80))                      # paper is the bright part of what is looked at
    paper = (closed >= PAPER_REL * bright) & (chroma <= MAX_CHROMA)
    inside = paper if mask is None else (paper & (mask > 0))
    n_in = int(np.count_nonzero(inside))
    n_all = int(closed.size if mask is None else np.count_nonzero(mask))
    if n_in == 0:
        return {"paper": paper, "level": bright, "noise": 0.0, "share": 0.0, "n": n_all}
    level = float(np.median(a["gray"][inside]))
    # pixel noise: what a pixel differs from the mean of its 3 x 3 neighbourhood, the largest 30 % left out
    # (ink edges: a finished drawing has them on a fifth of its area). 1.8 makes up for the part left out
    # (normal noise). A median would only ever be 0 or 1 on an 8-bit image.
    g = a["gray"]
    hp = cv2.absdiff(g, cv2.blur(g, (3, 3)))[inside].astype(np.float32)
    cut = float(np.percentile(hp, 70)) if hp.size else 0.0
    kept = hp[hp <= cut]
    noise = max(0.5, 1.8 * float(np.sqrt(np.mean(kept * kept)))) if kept.size else 0.5
    return {"paper": paper, "level": level, "noise": noise, "share": n_in / max(n_all, 1), "n": n_all}


def can_show_ink(img, cam, z, region=None) -> dict:
    """Could this photo show a 0.4 mm ink line in `region` (x0, y0, x1, y1 on the table, metres; None = the
    whole picture)?

    {"ok", "clipped", "paper_level", "reason", "in_view", "paper_share", "noise"}. ok is False when the
    region is not (fully) in the picture, the paper is clipped (the locked "desk" exposure does that), its
    level is too dark or too close to white, too little of the region is visible paper (the arm in front of
    it, the edge of the sheet), or the paper is too noisy. When ok is False the caller must treat every
    "no ink" there as UNKNOWN, never as "the pen did not touch"."""
    (u0, v0, u1, v1), mask, in_view = _roi(img.shape, cam, z, region)
    out = {"ok": False, "clipped": 0.0, "paper_level": 0.0, "reason": "", "in_view": round(in_view, 3),
           "paper_share": 0.0, "noise": 0.0}
    if u1 - u0 < 2 or v1 - v0 < 2 or in_view < 0.98:
        out["reason"] = f"the region is not in the picture (only {in_view:.0%} of it is in view)"
        return out
    a = _analyse(img[v0:v1, u0:u1])
    p = _paper(a, mask)
    if region is None:
        # the whole picture: the question is about the paper that is in it (desk and arm are not "hidden paper")
        g = a["gray"][p["paper"]]
        if g.size < 0.03 * a["gray"].size:
            out["reason"] = "no paper in the picture (nothing bright and white)"
            return out
        p["share"] = 1.0
    else:
        g = a["gray"][mask > 0]
    out["clipped"] = round(float(np.mean(g >= CLIP_LEVEL)), 4)
    out.update(paper_level=round(p["level"], 1), paper_share=round(p["share"], 3), noise=round(p["noise"], 2))
    if out["clipped"] > MAX_CLIPPED:
        out["reason"] = (f"the paper is clipped to white ({out['clipped']:.0%} of the region at {CLIP_LEVEL} or above): "
                         "a thin line cannot show; take the photo with the paper exposure")
    elif p["level"] < PAPER_LEVEL_MIN:
        out["reason"] = (f"too dark for paper (level {p['level']:.0f}, at least {PAPER_LEVEL_MIN:.0f} is needed): "
                         "no sheet here, or under-exposed")
    elif p["level"] > PAPER_LEVEL_MAX:
        out["reason"] = (f"the paper is almost white (level {p['level']:.0f}, at most {PAPER_LEVEL_MAX:.0f}): a thin line "
                         "fades; use the paper exposure")
    elif p["share"] < MIN_PAPER_SHARE:
        out["reason"] = (f"only {p['share']:.0%} of the region is visible paper (hidden by the arm or the pen, tape, "
                         "or off the sheet)")
    elif K_HIGH * p["noise"] > THIN_LINE_RESPONSE:
        out["reason"] = f"the paper is too noisy for a thin line (noise {p['noise']:.1f} grey levels)"
    else:
        out["ok"] = True
        out["reason"] = f"paper level {p['level']:.0f}, not clipped, {p['share']:.0%} of the region is visible paper"
    return out


TABLE_AREA = (0.09, -0.24, 0.45, 0.24)    # x0, y0, x1, y1: the work area on the table (scene.DEFAULTS["workspace"])


def clipped_share(img, cam=None, z: float = 0.0, region=TABLE_AREA) -> float:
    """Share of the pixels of the table area (the part of `region` that is in the picture) at CLIP_LEVEL or
    above. cam None: of the whole picture. With the locked desk exposure a sheet of white paper gives about
    0.4-0.8 here, with the paper exposure 0.0."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if cam is None or region is None:
        return float(np.mean(g >= CLIP_LEVEL))
    (u0, v0, u1, v1), mask, _ = _roi(img.shape, cam, z, region)
    if mask is None or not mask.any():
        return float(np.mean(g >= CLIP_LEVEL))
    return float(np.mean(g[v0:v1, u0:u1][mask > 0] >= CLIP_LEVEL))


# ----------------------------------------------------------------------------- finding ink

def _detect(after, cam, z, before=None, region=None, line_px=None) -> dict:
    """The working picture of one question: window, new-ink response, where ink could be seen at all.

    {"win": (u0, v0, u1, v1), "new": float32 response of NEW ink, "weak" / "strong": bool masks,
     "visible": bool (paper in both photos, away from edges), "old": bool (ink already in `before`),
     "level", "noise", "lo", "hi", "in_view"}"""
    (u0, v0, u1, v1), mask, in_view = _roi(after.shape, cam, z, region, margin_px=2 * LINE_PX + 2)
    if u1 - u0 < 2 or v1 - v0 < 2:
        return {"win": (0, 0, 0, 0), "empty": True, "in_view": in_view}
    a = _analyse(after[v0:v1, u0:u1], line_px)
    pa = _paper(a, mask)
    paper = pa["paper"]
    new = a["resp"].astype(np.float32)
    old = np.zeros(paper.shape, bool)
    level = pa["level"]
    if before is not None:
        b = _analyse(before[v0:v1, u0:u1], line_px)
        pb = _paper(b, mask)
        gain = min(1.25, max(0.8, level / pb["level"])) if pb["level"] > 1 else 1.0   # small exposure drift
        k = np.ones((OLD_INK_DILATE, OLD_INK_DILATE), np.uint8)
        was = cv2.dilate(b["resp"], k).astype(np.float32) * gain
        new = new - OLD_INK_GAIN * was
        paper = paper & pb["paper"]
        old = was > max(LOW_FLOOR, K_LOW * pb["noise"])
    # ink is looked for on paper only, a few pixels away from whatever is not paper (the arm's outline and
    # the edge of the sheet answer the line filter as well)
    k = 2 * EDGE_PX + 1
    visible = cv2.erode(paper.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    if mask is not None:
        visible &= mask > 0
    noise = pa["noise"]
    if before is not None:
        noise = max(noise, pb["noise"])
        floors = (LOW_FLOOR, HIGH_FLOOR, PIX_MIN, LINE_LOW, LINE_HIGH)
    else:
        floors = (LOW_FLOOR_ABS, HIGH_FLOOR_ABS, PIX_MIN_ABS, LINE_LOW_ABS, LINE_HIGH_ABS)
    scale = max(1.0, noise / 0.8)             # the floors hold up to a pixel noise of 0.8 grey levels
    lo, hi = max(floors[0], K_LOW * noise), max(floors[1], K_HIGH * noise)
    pix, l_lo, l_hi = floors[2] * scale, floors[3] * scale, floors[4] * scale
    # JPEG rings around a bold line with about a tenth of its depth: next to strong ink a pixel must reach a
    # share of the strongest response nearby, or the ringing would join neighbouring marks into one
    pos = np.clip(new, 0, None)
    near = cv2.dilate(pos, np.ones((2 * RING_PX + 1, 2 * RING_PX + 1), np.uint8))
    far = cv2.dilate(near, np.ones((2 * (RING_FAR_PX - RING_PX) + 1, 2 * (RING_FAR_PX - RING_PX) + 1), np.uint8))
    real = (new > RING_SHARE * near) & (new > RING_FAR_SHARE * far) & visible
    cand = real & (new >= pix) & (new <= hi)
    support = _line_support(new, cand)
    weak = real & ((new > lo) | (cand & (support > l_lo)))
    strong = real & ((new > hi) | (cand & (support > l_hi)))
    return {"win": (u0, v0, u1, v1), "new": new, "weak": weak, "strong": strong, "visible": visible,
            "old": old, "level": level, "noise": noise, "lo": lo, "hi": hi, "in_view": in_view, "closed": a["closed"],
            "mask": mask if mask is not None else np.ones(paper.shape, np.uint8), "near": near, "empty": False}


_LINE_OFFSETS = None


def _line_support(new, cand) -> np.ndarray:
    """For the candidate pixels only (anything else stays 0): the mean response along the best of 8
    directions through the pixel. Evaluated sparsely, because a filter over the whole window in 8
    directions costs 0.2 s on the Jetson and almost every pixel is blank paper."""
    global _LINE_OFFSETS
    if _LINE_OFFSETS is None:
        c = LINE_LEN // 2
        j = np.arange(-c, c + 1)
        _LINE_OFFSETS = [(np.round(j * math.sin(math.pi * k / 8)).astype(int), np.round(j * math.cos(math.pi * k / 8)).astype(int))
                         for k in range(8)]
    out = np.zeros(new.shape, np.float32)
    vs, us = np.nonzero(cand)
    if len(vs) == 0:
        return out
    c = LINE_LEN // 2
    x = np.pad(np.clip(new, -2.0, None), c, mode="edge")
    best = np.full(len(vs), -1e9, np.float32)
    for dv, du in _LINE_OFFSETS:
        best = np.maximum(best, x[vs[:, None] + c + dv[None, :], us[:, None] + c + du[None, :]].mean(axis=1))
    out[vs, us] = best
    return out


def _mark_from(us, vs, resp, d, cam, z) -> Mark:
    pts = table_xy(cam, np.column_stack([us, vs]), z)
    c = pts.mean(axis=0)
    if len(pts) >= 3:
        w, v = np.linalg.eigh(np.cov((pts - c).T))
        axis = v[:, 1]
    else:
        axis = np.array([0.0, 1.0])
    if axis[0] < 0 or (abs(axis[0]) < 1e-12 and axis[1] < 0):
        axis = -axis
    s = (pts - c) @ axis
    t = (pts - c) @ np.array([-axis[1], axis[0]])
    # one pixel is about 0.3 mm wide on the table: a single pixel still has that extent
    two = table_xy(cam, [[us[0], vs[0]], [us[0] + 1.0, vs[0]]], z)
    px_m = float(np.linalg.norm(two[1] - two[0]))
    top = float(np.percentile(resp, 90))
    level = max(d["level"], 1.0)
    conf = min(1.0, top / (3.0 * d["hi"])) * min(1.0, len(us) / 16.0)
    ang = math.degrees(math.atan2(axis[1], axis[0]))
    ang = ang - 180.0 if ang > 90.0 else ang
    return Mark(xy=(float(c[0]), float(c[1])), length_m=float(s.max() - s.min() + px_m),
                width_m=float(t.max() - t.min() + px_m), angle_deg=float(ang), area_px=int(len(us)),
                bbox_px=(int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1),
                darkness=float(min(1.0, top / level)), confidence=float(conf), pieces=1,
                extent_xy=(float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())))


def _pieces(d, min_area_px=8) -> list:
    """Connected pieces of new ink in a _detect() result as (us, vs, response) in WINDOW pixels: weak pixels
    that hang together (one-pixel breaks bridged), kept when the piece holds strong pixels and is large enough."""
    if d.get("empty"):
        return []
    joined = cv2.morphologyEx(d["weak"].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(joined, connectivity=8)
    if n <= 1:
        return []
    lab[~d["weak"]] = 0                                   # bridges join pieces but are not ink themselves
    vs, us = np.nonzero(lab)
    if len(us) == 0:
        return []
    ls = lab[vs, us]
    order = np.argsort(ls, kind="stable")
    us, vs, ls = us[order], vs[order], ls[order]
    cuts = np.flatnonzero(np.diff(ls)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(ls)]])
    strong = d["strong"][vs, us]
    resp = d["new"][vs, us]
    out = []
    for s, e in zip(starts, ends):
        if e - s < min_area_px or np.count_nonzero(strong[s:e]) < 2:
            continue
        out.append((us[s:e], vs[s:e], resp[s:e]))
    return out


def _marks(d, cam, z, min_area_px=8) -> list:
    if d.get("empty"):
        return []
    u0, v0 = d["win"][:2]
    return [_mark_from(us + u0, vs + v0, resp, d, cam, z) for us, vs, resp in _pieces(d, min_area_px)]


def find_ink(after, cam, z, before=None, region=None, min_area_px=8) -> list:
    """Marks (connected pieces of ink) on the paper, largest first. before given: only NEW ink (what the
    `after` photo shows more than the `before` photo; both taken with the arm at the same place).
    region: (x0, y0, x1, y1) on the table, None = the whole picture.

    An empty list is NOT "no ink" by itself: ask can_show_ink() for the same region first (or use
    ladder_readout / marks_readout / stroke_coverage, which do)."""
    d = _detect(after, cam, z, before, region)
    marks = _marks(d, cam, z, min_area_px)
    if region is not None:
        x0, y0, x1, y1 = _norm_region(region)
        marks = [m for m in marks if x0 <= m.xy[0] <= x1 and y0 <= m.xy[1] <= y1]
    return sorted(marks, key=lambda m: -m.area_px)


def merge_marks(marks) -> Mark | None:
    """Several pieces as ONE mark (a mark made with a chattering pen is often two or three blobs a few
    millimetres apart): centre = area-weighted mean of the pieces."""
    marks = list(marks)
    if not marks:
        return None
    if len(marks) == 1:
        return marks[0]
    w = np.array([m.area_px for m in marks], float)
    c = np.sum([np.array(m.xy) * a for m, a in zip(marks, w)], axis=0) / w.sum()
    ext = np.array([m.extent_xy for m in marks])
    x0, y0, x1, y1 = ext[:, 0].min(), ext[:, 1].min(), ext[:, 2].max(), ext[:, 3].max()
    bb = np.array([m.bbox_px for m in marks])
    dx, dy = x1 - x0, y1 - y0
    return Mark(xy=(float(c[0]), float(c[1])), length_m=float(max(dx, dy)), width_m=float(min(dx, dy)),
                angle_deg=float(0.0 if dx >= dy else 90.0), area_px=int(w.sum()),
                bbox_px=(int(bb[:, 0].min()), int(bb[:, 1].min()), int(bb[:, 2].max()), int(bb[:, 3].max())),
                darkness=float(max(m.darkness for m in marks)), confidence=float(max(m.confidence for m in marks)),
                pieces=int(sum(m.pieces for m in marks)), extent_xy=(float(x0), float(y0), float(x1), float(y1)))


def match_marks(expected_xy, marks, max_dist_m=0.03, merge_m=None) -> list:
    """One mark (or None) per expected place, one-to-one, nearest pairs first.

    Pieces closer to each other than merge_m count as one mark (default: 7 mm, but never more than a third
    of the smallest distance between two expected places, so that neighbouring marks stay apart).
    A None here only says "nothing matched": whether that is "no ink" or "cannot judge" is answered by
    marks_readout()."""
    exp = np.asarray(expected_xy, dtype=float).reshape(-1, 2)
    marks = list(marks)
    if merge_m is None:
        merge_m = 0.007
        if len(exp) > 1:
            dd = np.linalg.norm(exp[:, None, :] - exp[None, :, :], axis=2)
            merge_m = min(merge_m, float(dd[dd > 0].min()) / 3.0) if np.any(dd > 0) else merge_m
    groups = []                                           # single linkage on the piece centres
    for m in sorted(marks, key=lambda m: -m.area_px):
        hit = [g for g in groups if any(math.dist(m.xy, o.xy) < merge_m for o in g)]
        for g in hit:
            groups.remove(g)
        groups.append([m] + [o for g in hit for o in g])
    merged = [merge_marks(g) for g in groups]
    pairs = sorted((math.dist(e, m.xy), i, j) for i, e in enumerate(exp) for j, m in enumerate(merged))
    out, used = [None] * len(exp), set()
    for dist, i, j in pairs:
        if dist > max_dist_m:
            break
        if out[i] is None and j not in used:
            out[i] = merged[j]
            used.add(j)
    return out


def _place_state(d, cam, z, xy, radius_m) -> dict:
    """What the photo(s) offer around one place (a disc of radius_m on the table): the share of it that is
    visible paper and the share that was already inked in `before`."""
    u0, v0, u1, v1 = d["win"]
    c = pixels(cam, [xy], z)[0]
    e = pixels(cam, [(xy[0] + radius_m, xy[1]), (xy[0], xy[1] + radius_m)], z)
    ax = (max(1, int(round(abs(e[1][0] - c[0])))), max(1, int(round(abs(e[0][1] - c[1])))))   # y -> u, x -> v
    disc = np.zeros(d["weak"].shape, np.uint8)
    cv2.ellipse(disc, (int(round(c[0] - u0)), int(round(c[1] - v0))), ax, 0, 0, 360, 1, -1)
    n = int(np.count_nonzero(disc))
    full = math.pi * ax[0] * ax[1]
    if n == 0:
        return {"visible": 0.0, "old": 0.0, "in_window": 0.0}
    sel = disc > 0
    return {"visible": float(np.count_nonzero(d["visible"][sel])) / max(full, 1.0),
            "old": float(np.count_nonzero(d["old"][sel])) / max(full, 1.0), "in_window": n / max(full, 1.0)}


def marks_readout(after, cam, z, expected_xy, before=None, max_dist_m=0.03, min_len_m=0.0) -> dict:
    """Marks made at known commanded places (a calibration grid), tri-state per place.

    {"marks": [{"inked": True | False | None, "mark": Mark | None, "why": str, "xy": [x, y] | None,
                "faint": bool, "old_ink_here": bool}, ...],
     "found": n, "missing": [indices with inked False], "unknown": [indices with None], "judgeable": bool}
    inked False needs: the photo can show ink around the place (can_show_ink within max_dist_m) and nothing
    new there. None: the surroundings are clipped / hidden / out of the picture.
    "faint": the mark is much lighter than the other marks of this grid (the pen grazed the paper there: the
    place is good, the height was marginal). "old_ink_here": the `before` photo has ink within 3 mm of the
    commanded place; a mark landing exactly on it would not show, so with a tool whose ink lands where it is
    aimed such a "no" is not earned (with the offsets of 2026-11-11, 5-17 mm, the mark lands elsewhere: the
    two missing marks of that night's second grid are of this kind, hence a flag and not a verdict)."""
    exp = np.asarray(expected_xy, dtype=float).reshape(-1, 2)
    if len(exp) == 0:
        return {"marks": [], "found": 0, "missing": [], "unknown": [], "judgeable": True}
    region = (exp[:, 0].min() - max_dist_m, exp[:, 1].min() - max_dist_m, exp[:, 0].max() + max_dist_m,
              exp[:, 1].max() + max_dist_m)
    d = _detect(after, cam, z, before, region)
    marks = [m for m in _marks(d, cam, z) if m.length_m >= min_len_m]
    got = match_marks(exp, marks, max_dist_m)
    ref = max((m.darkness for m in got if m is not None), default=0.0)
    out = []
    for e, m in zip(exp, got):
        if m is not None:
            faint = bool(ref >= REF_DARKNESS and m.darkness < GRAZE_REL * ref)
            out.append({"inked": True, "mark": m, "xy": [round(float(m.xy[0]), 5), round(float(m.xy[1]), 5)], "faint": faint,
                        "old_ink_here": False,
                        "why": f"new ink {math.dist(e, m.xy) * 1000:.1f} mm from the commanded place ({m.area_px} px, "
                               f"{m.pieces} piece(s), darkness {m.darkness:.2f})"})
            continue
        r = (e[0] - max_dist_m, e[1] - max_dist_m, e[0] + max_dist_m, e[1] + max_dist_m)
        can = can_show_ink(after, cam, z, r)
        here = bool(before is not None and not d.get("empty") and _place_state(d, cam, z, e, 0.003)["old"] > 0.08)
        if not can["ok"]:
            out.append({"inked": None, "mark": None, "xy": None, "faint": False, "old_ink_here": here,
                        "why": "cannot judge: " + can["reason"]})
        else:
            out.append({"inked": False, "mark": None, "xy": None, "faint": False, "old_ink_here": here,
                        "why": f"no new ink within {max_dist_m * 1000:.0f} mm, and the photo could have shown it ({can['reason']})"
                               + ("; old ink lies on the commanded place itself: a mark exactly there would not show" if here else "")})
    return {"marks": out, "found": sum(1 for o in out if o["inked"]), "missing": [i for i, o in enumerate(out) if o["inked"] is False],
            "unknown": [i for i, o in enumerate(out) if o["inked"] is None], "judgeable": all(o["inked"] is not None for o in out)}


# ----------------------------------------------------------------------------- a ladder of dashes
def _bands(t, w, gap):
    """Cut ink into bands along the rung axis: [(t_lo, t_hi), ...] in rising order. Peaks of the ink profile
    (0.25 mm bins, smoothed over about 1 mm) at least half a rung gap apart; two peaks without a real valley
    between them are one band (a wavy dash), a valley is where to cut (ink of two rungs that hangs together)."""
    step = 0.00025
    lo = float(t.min()) - 4 * step
    nb = int((float(t.max()) - lo) / step) + 5
    h = np.bincount(np.clip(((t - lo) / step).astype(int), 0, nb - 1), weights=w, minlength=nb)
    h = np.convolve(h, np.array([1, 2, 3, 4, 3, 2, 1], float) / 16.0, mode="same")
    sep = max(2, int(round(0.5 * gap / step)))
    peaks, work = [], h.copy()
    top = float(h.max())
    while True:
        i = int(np.argmax(work))
        if work[i] <= max(0.01 * top, 1e-6):
            break
        peaks.append(i)
        work[max(0, i - sep):i + sep + 1] = 0.0
    peaks.sort()
    if not peaks:
        return [(-np.inf, np.inf)]
    cuts = []
    keep = [peaks[0]]
    for a, b in zip(peaks[:-1], peaks[1:]):
        j = a + int(np.argmin(h[a:b + 1]))
        if h[j] < 0.7 * min(h[keep[-1]], h[b]):
            cuts.append(lo + (j + 0.5) * step)
            keep.append(b)
        elif h[b] > h[keep[-1]]:
            keep[-1] = b
    edges = [-np.inf] + cuts + [np.inf]
    return list(zip(edges[:-1], edges[1:]))


def _assign(tb, tc, gap, proper, n, max_offset_m):
    """Rung index for every band (bands and rungs in rising order along the rung axis), from the ORDER and the
    SPACING of the bands only: t_band = a + b * t_commanded with the spacing b between 0.7 and 1.6 of the
    commanded one (measured 2026-11-11: 0.95 to 1.5) and any offset a up to max_offset_m. When fewer bands
    than rungs leave the offset open by whole rungs, the deeper rungs are taken (they are drawn later and
    lower: ink starts there). -> (indices, rms_m, a, b, ambiguous) or None when nothing fits."""
    from itertools import combinations
    k = len(tb)
    best, second = None, None
    for idx in combinations(range(n), k):
        x = np.array([tc[i] for i in idx])
        y = np.asarray(tb, float)
        if k >= 2 and np.ptp(x) > 0:
            b = float(np.clip(np.polyfit(x, y, 1)[0], 0.7, 1.6))
        else:
            b = 1.0
        a = float(np.mean(y - b * x))
        if abs(a + (b - 1.0) * float(np.mean(tc))) > max_offset_m + gap:
            continue
        rms = float(np.sqrt(np.mean((y - a - b * x) ** 2)))
        # deeper rungs first: every (inked rung, deeper rung without proper ink) pair costs a little
        inked = {i for i, pr in zip(idx, proper) if pr}
        inv = sum(1 for i in inked for j in range(i + 1, n) if j not in inked)
        cost = rms + 0.0003 * inv + 1e-6 * sum(n - 1 - i for i in idx)
        cand = (cost, idx, rms, a, b)
        if best is None or cost < best[0]:
            best, second = cand, best
        elif second is None or cost < second[0]:
            second = cand
    if best is None:
        return None
    ambiguous = second is not None and abs(second[2] - best[2]) < 0.0004
    return best[1], best[2], best[3], best[4], ambiguous


def ladder_readout(after, cam, z, rungs, before=None, max_offset_m=MAX_OFFSET_M, ink_darkness=None) -> dict:
    """Which rungs of a ladder of dashes left ink?  rungs: [((x, ya), (x, yb)), ...] as commanded, in the
    order they were drawn (deeper = later).

    {"rungs": [{"inked": True | False | None, "mark": Mark | None, "why": str, "speck": bool}, ...],
     "first_inked": i | None, "judgeable": bool, "notes": [...], "untouched": bool | None,
     "offset_mm": [dx, dy] | None, "can": can_show_ink() of the search window}

    inked None = cannot judge (the reason is in "why"). The ink of an uncalibrated tool lies up to
    max_offset_m from where it was aimed, so rungs are told apart by their ORDER and SPACING, never by their
    absolute place. A rung is inked when its ink is a line or a firm touch; a speck (the pen grazed the
    paper) is "no" when the ladder shows what real ink of this pen looks like, and "cannot judge" when it
    does not (ink_darkness: the darkness of this pen's line when the caller knows it).
    "untouched" False: the `before` photo already has ink inside the search window. A dash drawn onto old ink
    cannot be seen, so a caller that goes deeper on "no ink" must use paper where "untouched" is True."""
    R = np.asarray(rungs, dtype=float).reshape(-1, 2, 2)
    n = len(R)
    c = R.mean(axis=1)
    dvec = (R[:, 1] - R[:, 0]).mean(axis=0)
    dash = float(np.linalg.norm(dvec))
    e_d = dvec / dash if dash > 1e-9 else np.array([0.0, 1.0])
    if n >= 2 and np.linalg.norm(c[-1] - c[0]) > 1e-9:
        e_r = (c[-1] - c[0]) / np.linalg.norm(c[-1] - c[0])
        gap = float(np.linalg.norm(c[-1] - c[0])) / (n - 1)
    else:
        e_r, gap = np.array([-e_d[1], e_d[0]]), max(dash, 0.004)
    tc = (c - c[0]) @ e_r
    pts_all = R.reshape(-1, 2)
    region = (pts_all[:, 0].min() - max_offset_m, pts_all[:, 1].min() - max_offset_m,
              pts_all[:, 0].max() + max_offset_m, pts_all[:, 1].max() + max_offset_m)
    can = can_show_ink(after, cam, z, region)
    out = {"rungs": [], "first_inked": None, "judgeable": False, "notes": [], "untouched": None, "offset_mm": None,
           "can": can}

    def give(i, inked, why, mark=None, speck=False):
        out["rungs"].append({"i": i, "inked": inked, "mark": mark, "why": why, "speck": speck})

    def finish():
        out["rungs"].sort(key=lambda r: r["i"])
        hit = [r["i"] for r in out["rungs"] if r["inked"]]
        out["first_inked"] = min(hit) if hit else None
        out["judgeable"] = all(r["inked"] is not None for r in out["rungs"])
        return out

    if not can["ok"]:
        # nothing seen in such a photo means nothing; ink that IS seen is still ink, so the search goes on
        out["notes"].append("the photo cannot show thin ink here: " + can["reason"])
    d = _detect(after, cam, z, before, region)
    if d.get("empty"):
        for i in range(n):
            give(i, None, "cannot judge: " + can["reason"])
        return finish()
    u0, v0 = d["win"][:2]
    l_min = min(dash * SPECK_SHARE, SPECK_MAX_M) if dash > 0 else SPECK_MAX_M
    hidden_rungs = 0
    if before is not None:
        # Ink that was there before: a dash drawn onto an old line cannot be seen. Count the old lines that run
        # the way the dashes do, as rows along the rung axis: that many rungs could be hidden under them.
        was = [m for m in _marks(_detect(before, cam, z, None, region), cam, z, 6)]
        out["untouched"] = not was
        rows = sorted(float((np.array(m.xy) - c[0]) @ e_r) for m in was
                      if (m.extent_xy[2] - m.extent_xy[0]) * abs(e_d[0]) + (m.extent_xy[3] - m.extent_xy[1]) * abs(e_d[1]) >= l_min)
        last = None
        for tv in rows:
            if last is None or tv - last >= 0.5 * gap:
                hidden_rungs += 1
                last = tv
        if was:
            out["notes"].append(f"old ink inside the search window ({len(was)} piece(s), lines along the dashes in "
                                f"{hidden_rungs} row(s)): a dash drawn onto old ink would not show")
    pieces = _pieces(d, min_area_px=6)

    def nothing():
        for i in range(n):
            if not can["ok"]:
                give(i, None, "cannot judge: " + can["reason"])
            elif hidden_rungs >= n:
                give(i, None, f"cannot judge: no new ink, but the paper already had {hidden_rungs} line(s) along the dashes "
                              f"within {max_offset_m * 1000:.0f} mm: ink drawn onto them would not show (use untouched paper)")
            else:
                give(i, False, f"no new ink within {max_offset_m * 1000:.0f} mm of the ladder, and the photo could have "
                               f"shown it ({can['reason']})")
        return finish()

    if not pieces:
        return nothing()

    # every ink pixel on the ladder's own axes: t along the rungs' order, s along the dashes
    us = np.concatenate([p[0] for p in pieces]) + u0
    vs = np.concatenate([p[1] for p in pieces]) + v0
    resp = np.concatenate([p[2] for p in pieces])
    pid = np.concatenate([np.full(len(p[0]), k) for k, p in enumerate(pieces)])
    xy = table_xy(cam, np.column_stack([us, vs]), z)
    t = (xy - c[0]) @ e_r
    level = max(d["level"], 1.0)
    w = np.clip(resp, 0.5, None) / level

    def marks_in(sel):
        """The pieces of one band as marks, largest first (a piece cut by a band edge counts with its part)."""
        ms = []
        for k in np.unique(pid[sel]):
            m = sel & (pid == k)
            if np.count_nonzero(m) >= 4:
                ms.append(_mark_from(us[m], vs[m], resp[m], d, cam, z))
        return sorted(ms, key=lambda m: -m.area_px)

    whole = marks_in(np.ones(len(t), bool))
    ref = ink_darkness if ink_darkness else max((m.darkness for m in whole), default=0.0)
    has_ref = bool(ink_darkness) or ref >= REF_DARKNESS

    def proper(m):
        if has_ref:
            return ((m.length_m >= l_min and m.darkness >= GRAZE_REL * ref)
                    or (m.area_px >= TOUCH_AREA_PX and m.darkness >= max(TOUCH_DARKNESS, TOUCH_REL * ref)))
        return m.length_m >= l_min

    def verdict(i, ms, where):
        good = [m for m in ms if proper(m)]
        if good:
            m = merge_marks(good)
            off = math.hypot(m.xy[0] - c[i][0], m.xy[1] - c[i][1])
            give(i, True, f"ink {where}: {m.length_m * 1000:.1f} mm long, darkness {m.darkness:.2f}, "
                          f"{off * 1000:.0f} mm from where it was aimed", m)
        elif not can["ok"]:
            m = merge_marks(ms)
            give(i, None, f"cannot judge: a faint trace {where} ({m.area_px} px) in a photo that cannot show thin ink: "
                          + can["reason"], m, speck=True)
        elif has_ref:
            m = merge_marks(ms)
            give(i, False, f"only a speck {where} ({m.area_px} px, longest piece {max(q.length_m for q in ms) * 1000:.1f} mm, "
                           f"darkness {m.darkness:.2f} against {ref:.2f} for this pen's ink): the pen grazed the paper, it "
                           "did not draw", m, speck=True)
        else:
            m = merge_marks(ms)
            give(i, None, f"cannot judge: a faint speck {where} ({m.area_px} px, {max(q.length_m for q in ms) * 1000:.1f} mm, "
                          f"darkness {m.darkness:.2f}) and nothing in the photo shows what this pen's ink looks like: "
                          "it may be a graze or a faint line (look at the photo)", m, speck=True)

    if n == 1:
        verdict(0, whole, f"within {max_offset_m * 1000:.0f} mm")
        m = out["rungs"][0]["mark"]
        out["offset_mm"] = [round(float(m.xy[0] - c[0][0]) * 1000, 1), round(float(m.xy[1] - c[0][1]) * 1000, 1)]
        return finish()

    bands = _bands(t, w, gap)
    info = []
    for lo, hi in bands:
        sel = (t >= lo) & (t < hi)
        ms = marks_in(sel)
        if not ms:
            continue
        ws = w[sel]
        info.append({"t": float(np.sum(t[sel] * ws) / ws.sum()), "mass": float(ws.sum()), "marks": ms,
                     "proper": any(proper(m) for m in ms), "span": float(np.percentile(t[sel], 95) - np.percentile(t[sel], 5)),
                     "xy": (xy[sel] * ws[:, None]).sum(axis=0) / ws.sum()})
    if not info:
        return nothing()
    smeared = [b for b in info if b["span"] > 1.5 * gap]
    if smeared:
        out["notes"].append(f"ink runs {max(b['span'] for b in smeared) * 1000:.0f} mm along the ladder without a break "
                            "(the pen was dragged from rung to rung): the rungs cannot be told apart")
        for i in range(n):
            give(i, None, "cannot judge: the ink of several rungs hangs together (dragged pen)")
        return finish()
    dropped = 0
    while len(info) > n or (len(info) > 1 and _assign([b["t"] for b in info], tc, gap, [b["proper"] for b in info], n,
                                                       max_offset_m) is None) \
            or (len(info) > 1 and _assign([b["t"] for b in info], tc, gap, [b["proper"] for b in info], n,
                                          max_offset_m)[1] > 0.35 * gap):
        weak = [b for b in info if not b["proper"]]
        if not weak:
            break
        info.remove(min(weak, key=lambda b: b["mass"]))            # a stray speck must not decide the order
        dropped += 1
    fit = _assign([b["t"] for b in info], tc, gap, [b["proper"] for b in info], n, max_offset_m) if len(info) <= n else None
    if fit is None or fit[1] > 0.35 * gap:
        out["notes"].append(f"{len(info)} separate marks do not fit a ladder of {n} rungs {gap * 1000:.1f} mm apart")
        for i in range(n):
            give(i, None, "cannot judge: the marks in the photo do not look like this ladder")
        return finish()
    idx, rms, a, b, ambiguous = fit
    if dropped:
        out["notes"].append(f"{dropped} stray speck(s) left out of the rung order")
    if ambiguous and len(info) < n:
        out["notes"].append("fewer marks than rungs: taken as the DEEPEST rungs (ink starts low); which rung is which is "
                            "known only by that rule")
    s_off = float(np.mean([(bd["xy"] - c[i]) @ e_d for bd, i in zip(info, idx)]))
    t_off = a + (b - 1.0) * float(np.mean(tc))
    off = t_off * e_r + s_off * e_d
    out["offset_mm"] = [round(float(off[0]) * 1000, 1), round(float(off[1]) * 1000, 1)]
    by_rung = dict(zip(idx, info))
    for i in range(n):
        if i in by_rung:
            verdict(i, by_rung[i]["marks"], f"at rung {i}'s place in the row")
            continue
        place = c[0] + (a + b * tc[i]) * e_r + s_off * e_d
        st = _place_state(d, cam, z, place, max(0.5 * dash, 0.0015))
        if not can["ok"]:
            give(i, None, "cannot judge: " + can["reason"])
        elif st["visible"] < 0.8:
            give(i, None, f"cannot judge: where rung {i} should be ({place[0]:.3f}, {place[1]:.3f}) only {st['visible']:.0%} "
                          "is visible paper")
        elif st["old"] > 0.08:
            give(i, None, f"cannot judge: where rung {i} should be ({place[0]:.3f}, {place[1]:.3f}) there was ink before; "
                          "a dash on top of it would not show")
        else:
            give(i, False, f"no ink where rung {i} should be ({place[0]:.3f}, {place[1]:.3f}), {abs(b) * gap * 1000:.1f} mm "
                           "per rung from the rungs that did leave ink")
    return finish()


# ----------------------------------------------------------------------------- a drawing
def stroke_coverage(after, cam, z, polylines_xy, before=None, tol_m=0.004, step_m=0.0005, min_gap_m=0.0015,
                    detail=False) -> dict:
    """How much of the expected ink is on the paper?  polylines_xy: the strokes where the INK should be
    (table, metres), each in the direction it was drawn.

    Every stroke is sampled each step_m. A sample is DRAWN when (new) ink lies across the stroke there, within
    tol_m to either side; the test is a thin strip across the stroke, not a disc, so that the gaps of a dashed
    line are seen and not bridged by the dashes before and after. A sample is UNKNOWN where the photo cannot
    show ink (not visible paper) or where old ink lies on the line and nothing new was found.

    {"drawn_fraction": drawn / (drawn + missing), "gaps": [{"stroke": i, "from_m": s0, "to_m": s1}, ...] (missing
     runs of at least min_gap_m, s = path length from the stroke's start), "offset_mm": {"mean", "p95"} (how far
     the ink lies from the expected line, over the drawn samples), "judgeable": bool, "unknown_fraction",
     "ink_m", "missing_m", "can": can_show_ink() of the drawing's area, "per_stroke": [fraction or None, ...]}
    judgeable False: the area cannot show ink or more than 15 % of the samples are unknown; then
    drawn_fraction says nothing about gaps. detail=True adds "samples": per stroke {"s_m": [...], "state":
    [1 drawn | 0 missing | -1 unknown, ...]} (for redrawing what is missing)."""
    polys = [np.asarray(p, dtype=float).reshape(-1, 2) for p in polylines_xy]
    polys = [p for p in polys if len(p) >= 2]
    empty = {"drawn_fraction": 0.0, "gaps": [], "offset_mm": {"mean": None, "p95": None}, "judgeable": False,
             "unknown_fraction": 1.0, "ink_m": 0.0, "missing_m": 0.0, "can": None, "per_stroke": [],
             **({"samples": []} if detail else {})}
    if not polys:
        return empty
    allp = np.vstack(polys)
    region = (allp[:, 0].min() - tol_m, allp[:, 1].min() - tol_m, allp[:, 0].max() + tol_m, allp[:, 1].max() + tol_m)
    can = can_show_ink(after, cam, z, region)
    d = _detect(after, cam, z, before, region)
    if d.get("empty"):
        return {**empty, "can": can}
    u0, v0, u1, v1 = d["win"]
    ink = np.zeros(d["weak"].shape, bool)
    for us, vs, _ in _pieces(d, min_area_px=4):
        ink[vs, us] = True
    # A pen that chatters leaves a row of dots which the camera blurs into a beaded line: between the dots the
    # response is still well above the noise. For coverage only SOLID ink counts: at least COVER_SHARE of the
    # strongest response within RING_PX pixels.
    ink &= d["new"] >= COVER_SHARE * d["near"]
    old = cv2.dilate(d["old"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    hgt, wid = ink.shape
    lat_step = min(step_m, 0.00025)                                # at most a pixel between strip positions
    ks = np.arange(-int(round(tol_m / lat_step)), int(round(tol_m / lat_step)) + 1)
    lat = ks * lat_step
    order = np.argsort(np.abs(ks), kind="stable")
    n_drawn = n_miss = n_unknown = 0
    offs, gaps, per, samples = [], [], [], []
    for i, p in enumerate(polys):
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] < 1e-9:
            per.append(None)
            samples.append({"s_m": [], "state": []})
            continue
        sv = np.arange(0.0, cum[-1] + 1e-12, step_m)
        j = np.clip(np.searchsorted(cum, sv, side="right") - 1, 0, len(seg) - 1)
        f = (sv - cum[j]) / np.maximum(seg[j], 1e-12)
        q = p[j] + (p[j + 1] - p[j]) * f[:, None]
        tang = (p[j + 1] - p[j]) / np.maximum(seg[j], 1e-12)[:, None]
        norm = np.column_stack([-tang[:, 1], tang[:, 0]])
        grid = q[:, None, :] + lat[None, :, None] * norm[:, None, :]              # samples x strip x 2
        uv = pixels(cam, grid.reshape(-1, 2), z).reshape(len(sv), len(ks), 2)
        uu = np.round(uv[:, :, 0]).astype(int) - u0
        vv = np.round(uv[:, :, 1]).astype(int) - v0
        inside = (uu >= 0) & (uu < wid) & (vv >= 0) & (vv < hgt)
        uc, vc = np.clip(uu, 0, wid - 1), np.clip(vv, 0, hgt - 1)
        hit = ink[vc, uc] & inside
        mid = len(ks) // 2
        seen = d["visible"][vc[:, mid], uc[:, mid]] & inside[:, mid]
        near_old = (old[vc, uc] & inside)[:, np.abs(lat) <= 0.001].any(axis=1)
        drawn = hit.any(axis=1)
        # offset: to the MIDDLE of the nearest line across the strip (to its edge it would read half a line
        # width too small): the run of ink that holds the nearest inked strip position
        first = order[np.argmax(hit[:, order], axis=1)]
        runs = np.cumsum(~hit, axis=1)
        same = hit & (runs == runs[np.arange(len(sv)), first][:, None])
        centre = (same * lat[None, :]).sum(axis=1) / np.maximum(same.sum(axis=1), 1)
        unknown = ~drawn & (~seen | near_old)
        missing = ~drawn & ~unknown
        offs.append(np.abs(centre)[drawn])
        n_drawn += int(drawn.sum())
        n_miss += int(missing.sum())
        n_unknown += int(unknown.sum())
        if detail:
            samples.append({"s_m": sv.round(5).tolist(), "state": np.where(drawn, 1, np.where(unknown, -1, 0)).tolist()})
        per.append(round(float(drawn.sum()) / max(int(drawn.sum() + missing.sum()), 1), 3) if (drawn.sum() + missing.sum()) else None)
        run = None
        for k, m in enumerate(np.append(missing, False)):
            if m and run is None:
                run = k
            elif not m and run is not None:
                if (k - run) * step_m >= min_gap_m:
                    gaps.append({"stroke": i, "from_m": round(float(sv[run]), 5), "to_m": round(float(sv[k - 1]), 5)})
                run = None
    total = n_drawn + n_miss + n_unknown
    offs = np.concatenate(offs) if offs else np.zeros(0)
    unknown_fraction = n_unknown / max(total, 1)
    extra = {"samples": samples} if detail else {}
    return {**extra, "drawn_fraction": round(n_drawn / max(n_drawn + n_miss, 1), 4), "gaps": gaps,
            "offset_mm": {"mean": round(float(offs.mean()) * 1000, 2) if offs.size else None,
                          "p95": round(float(np.percentile(offs, 95)) * 1000, 2) if offs.size else None},
            "judgeable": bool(can["ok"] and unknown_fraction <= 0.15 and total > 0),
            "unknown_fraction": round(unknown_fraction, 4), "ink_m": round(n_drawn * step_m, 4),
            "missing_m": round(n_miss * step_m, 4), "can": can, "per_stroke": per}


def as_json(obj):
    """A result of this module with every Mark turned into a plain dict and numpy numbers into Python
    ones (for the JSON that `fd` prints and logs)."""
    if isinstance(obj, Mark):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: as_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj
