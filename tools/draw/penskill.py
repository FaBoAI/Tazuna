"""The pen skill: calibrate a tool that draws (a pen taped to the gripper) on the paper, and draw with it.

Why this exists (2026-11-11): the cat of that evening was drawn with ad-hoc scripts (tools/prototypes/pen_draw/) in 75
minutes. What went wrong then must be impossible here:
- a detector said "no ink" on a clipped photo and the script went 3 mm deeper per round: here every descent
  whose result is read from a photo is preceded by ink.can_show_ink at that place, a search goes deeper ONLY
  on a judgeable "no ink" for every rung, and "cannot judge" stops everything (stage "visual");
- marks were drawn on top of older ink: here every test mark is put on paper that a fresh photo shows to be
  untouched, 3 cm from older ink, and (by default) outside the box `fd draw` would use;
- a lift that failed left the pen 8 mm above the paper and the script died: here every exit path lifts the
  pen (tooling.lift, which tries other pitches), goes aside when it can, and reports where the pen is;
- the height wandered by millimetres: the contact height is measured at several places (a plane), the
  drawing presses `press` below it, and the pressing depth is capped (SPEC section 6).

Order of a calibration (each stage reads and writes the tool file, so one stage can be repeated):
  tip        one ladder search; the first inked rung gives the tool tip (its height: the length along the
             tool, its place: sideways and radial). Changes tip_offset, so contact and map are dropped.
  contact    ladders at 3 or more places -> plane "plan z of first ink" (tool.contact)
  map        marks (3 mm dashes) -> affine map commanded -> actual (tool.corr_map)
  direction  short lines away from / towards the robot and across -> which direction of travel keeps its
             ink (tool.calibrated["direction"]); nothing is assumed about pushing or pulling (SPEC 9.1)

Everything here was run on the simulated arm and the simulated paper (simpaper.SimPaper) only.
NOTHING IN THIS FILE HAS RUN ON THE REAL ARM.

Units: metres, base frame; "plan z" is the commanded height of the MODELLED tool tip. Results are plain
dicts with ok, stage, reason, next, say (Japanese), timings, photos, pen (where the pen was left).
Tests: test_penskill.py.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

import kinematics as K
import strokes as ST
import tooling as TL
import tracking as TR
from robot import SafetyError
from skills import MOTION_ERRORS, Timer, arm_clock, arm_sleep

STAGES = ("tip", "contact", "map", "direction")
STROKE_POLICIES = ("as-is", "away", "toward", "auto")
# Hard limits of the stop rule (SPEC section 6): options may be stricter, never looser.
MAX_RUNGS = 4                 # dashes per ladder
MAX_RUNG_DZ_M = 0.0015        # height step from dash to dash
MAX_SEARCH_BELOW_M = 0.006    # a search never goes further under the height where the model expects the paper
MAX_AFTER_INK_M = 0.003       # once ink was seen, calibration never presses further below that height
MAX_PRESS_M = 0.004           # a drawing never presses further below the contact plane
ASIDE_Q_DEG = (43.15, -20.31, 10.46, 63.28, -0.05)   # the side pose the photos of 2026-11-11 were taken from
ABOVE_M = 0.03                # a far hop ends this high above the table, over the next start
# Noise figure of ink.can_show_ink up to which a lone faint trace counts as a touch. The figure saturates: it
# reads 0.5 on every photo of 2026-11-11, 1.02 / 1.13 for simulated white noise of 0.6 / 0.8 grey levels (up
# to which ink.py's thresholds hold) and 1.22 / 1.36 for 1.0 / 1.6 (where blank paper starts to show specks).
GRAZE_MAX_NOISE = 1.17
UNVERIFIED_JA = "（この機能は実機ではまだ一度も動かしていません。）"
SIM_JA = "（シミュレーションです。実機の結果ではありません。）"


INTERRUPTED_NEXT = ("the command was interrupted; the pen was lifted when that was possible (see \"pen\": where / height_mm). "
                    "Do not start it again blindly: `fd tool rest` folds the arm away (it lifts first), then look at the photos")


class Stop(Exception):
    """A stage could not go on. stage / reason / next as in every result of the harness."""

    def __init__(self, stage: str, reason: str, next_: str = "", **detail):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason, self.next, self.detail = stage, reason, next_, detail


@dataclass
class PenOptions:
    """Everything `calibrate` and `draw` can be told. Lengths in metres unless the name says mm."""
    # the stop rule (SPEC section 6); check() refuses anything looser than the hard limits above
    rungs: int = MAX_RUNGS
    rung_dz_m: float = MAX_RUNG_DZ_M
    max_search_below_m: float = MAX_SEARCH_BELOW_M
    after_ink_max_m: float = MAX_AFTER_INK_M
    start_above_m: float = 0.012     # first rung of a first search: this far above where the model has the paper
    known_above_m: float = 0.003     # ... and of a search at a place where the contact height is roughly known
    max_ladders: int = 5             # ladders per search (4 reach from +12 mm down to -6 mm)
    max_up: int = 2                  # ladders a search may go UP when the top rung already left ink
    press_m: float = 0.0023          # drawing: below the contact plane (what drew the cat: -3.8 against -1.5 mm)
    mark_press_m: float = 0.003      # calibration marks: below the contact plane (the most rule 4 allows: at 2 mm
                                     # below a contact height of that evening two of seven marks left no ink)
    # test marks
    dash_m: float = 0.006            # a rung of a ladder
    rung_gap_m: float = 0.004
    mark_dash_m: float = 0.003
    places: int = 4                  # contact ladders (3 fix a plane, the 4th shows its residual)
    grid: int = 3                    # marks per side for the map
    marks_on: str = "outside"        # "outside" the box `fd draw` would use | "scrap": anywhere in reach
    clear_m: float = 0.03            # new marks keep this far from older ink
    box_margin_m: float = 0.008
    sheet_margin_m: float = 0.010
    at: tuple | None = None          # x, y: centre of the drawing box / where to calibrate
    # motion
    lift_m: float = 0.008            # pen-up height above the contact height
    v_mm_s: float = 6.0
    down_mm_s: float = 12.0
    up_mm_s: float = 30.0
    hop_mm_s: float = 40.0
    hop_max_m: float = 0.05          # further than this the pen travels by a free move, 3 cm up
    # drawing
    size_mm: float = 105.0           # longer side of the drawing box
    rot_deg: float | None = None     # None = follow the sheet
    strokes: str = "auto"
    end: str = "aside"               # "aside" (torque on, beside the sheet) | "rest" (tool rest pose, torque off)
    min_coverage: float = 0.85
    max_failed: int = 2              # pieces that may fail (pen lifted each time) before the drawing stops
    dry: bool = False
    # acceptance of the measurements
    max_plane_rms_mm: float = 1.5
    max_map_rms_mm: float = 3.0
    min_marks: int = 6
    max_tip_shift_m: float = 0.03
    direction_len_m: float = 0.02
    direction_min_diff: float = 0.10

    def check(self) -> "PenOptions":
        def num(name, lo, hi):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and lo <= v <= hi):
                raise SafetyError(f"pen option {name} = {v!r} is outside {lo} .. {hi}")

        num("rungs", 1, MAX_RUNGS)
        num("rung_dz_m", 0.0005, MAX_RUNG_DZ_M)
        num("max_search_below_m", 0.0, MAX_SEARCH_BELOW_M)
        num("after_ink_max_m", 0.0, MAX_AFTER_INK_M)
        num("start_above_m", 0.0, 0.03)
        num("known_above_m", 0.0, 0.01)
        num("max_ladders", 1, 8)
        num("press_m", 0.0, MAX_PRESS_M)
        num("mark_press_m", 0.0, MAX_AFTER_INK_M)
        num("places", 3, 8)
        num("grid", 3, 4)
        num("size_mm", 30.0, 200.0)
        num("v_mm_s", 1.0, 30.0)
        num("lift_m", 0.004, 0.02)
        num("min_coverage", 0.0, 1.0)
        if self.marks_on not in ("outside", "scrap"):
            raise SafetyError(f"marks_on {self.marks_on!r}: 'outside' (the box fd draw would use stays clean) or 'scrap'")
        if self.strokes not in STROKE_POLICIES:
            raise SafetyError(f"strokes {self.strokes!r}: one of {', '.join(STROKE_POLICIES)}")
        if self.end not in ("aside", "rest"):
            raise SafetyError(f"end {self.end!r}: 'aside' or 'rest'")
        if self.at is not None:
            try:
                self.at = (float(self.at[0]), float(self.at[1]))
            except (TypeError, ValueError, IndexError):
                raise SafetyError(f"at {self.at!r}: x,y in metres") from None
        return self


class PenSession:
    """Owns the arm and the photo source for one command (as skills.Session does for pick / take).

    photo(exposure) -> BGR image of the fixed camera ("paper" exposure for ink). On the real arm that is
    camera.grab(exposure=...); in simulation SimPaper.render. image_dir: where photos are written (None: not
    saved), or a function that returns it (called when the first photo is saved, so that a command that is
    refused before it takes a photo creates no directory). save_tool(tool): writes the tool file (default
    tooling.save(tool, sim))."""

    def __init__(self, arm, cam_model, photo, image_dir: str | None = None, sim: bool = False, paper=None,
                 save_tool=None, log=None):
        self.arm, self.cam, self._photo = arm, cam_model, photo
        self.image_dir, self.sim, self.paper = image_dir, bool(sim), paper
        self.save_tool = save_tool or (lambda tool: TL.save(tool, self.sim))
        self.log = log or (lambda *a, **k: None)
        self.images = {}
        self._n = 0

    def photo(self, exposure: str = "paper"):
        return self._photo(exposure)

    def save(self, tag: str, img) -> str | None:
        if self.image_dir is None or img is None:
            return None
        if callable(self.image_dir):
            self.image_dir = self.image_dir()
        os.makedirs(self.image_dir, exist_ok=True)
        self._n += 1
        path = os.path.join(self.image_dir, f"{time.strftime('%H%M%S')}_pen{self._n:02d}_{tag}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.images[tag] = path
        return path


# ----------------------------------------------------------------------------- simulation
def load_truth(path: str) -> dict:
    """The hidden truths of a simulated world from a small JSON file (`fd --sim ... --truth FILE`):
    {"night": true} = simpaper.night_truths(); any of "tip_error", "true_tip_offset", "pos_error", "contact",
    "wander_mm", "seed", "skips" ("none" | "tip-leading" | "tip-trailing"), "pen_width_m", "noise",
    "exposure" ("desk": every photo is taken with the desk exposure, to rehearse the stop), "sheet"
    ({"centre", "rot_deg"}) overrides it."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        raise SafetyError(f"cannot read the truth file {path}: {e}") from e
    if not isinstance(d, dict):
        raise SafetyError(f"the truth file {path} must hold one JSON object")
    known = {"night", "tip_error", "true_tip_offset", "pos_error", "contact", "wander_mm", "seed", "skips",
             "pen_width_m", "noise", "exposure", "sheet", "note"}
    bad = sorted(set(d) - known)
    if bad:
        raise SafetyError(f"the truth file {path} has unknown keys {bad} (known: {sorted(known)})")
    return d


def sim_world(arm, cam, truth: dict | None = None, state: dict | None = None):
    """(SimPaper, photo function) for a simulated arm: the sheet, the pen and the camera of `fd --sim draw` /
    `fd --sim tool calibrate` and of the tests. truth: see load_truth (None: the model is right).
    state: what sim_state() returned in an earlier run, so that separate `fd --sim` commands share one sheet
    and one REAL pen (the true tip offset is fixed when the world is made; a later correction of the tool file
    must not move it)."""
    import simpaper as SP
    truth = dict(truth or {})
    kw = dict(SP.night_truths()) if truth.get("night") else {}
    for k in ("tip_error", "true_tip_offset", "pos_error", "contact", "wander_mm", "seed", "skips", "pen_width_m"):
        if k in truth:
            kw[k] = truth[k]
    if "true_tip_offset" in kw:
        kw.pop("tip_error", None)
    if state and state.get("true_tip_offset") is not None:
        kw.pop("tip_error", None)
        kw["true_tip_offset"] = state["true_tip_offset"]
    if isinstance(truth.get("sheet"), dict):
        kw["sheet"] = SP.sheet_polygon(truth["sheet"].get("centre", (0.225, 0.0)), rot_deg=float(truth["sheet"].get("rot_deg", 6.0)))
    tool = getattr(arm, "tool", None)
    if tool is not None:
        kw.setdefault("pen_axis", TL.tool_axis(arm.cfg, tool))   # the pen leans as tooling says (see the notes)
    paper = SP.SimPaper(arm, cam, **kw)
    if state and state.get("paper"):
        paper.load_state(state["paper"])
    noise = float(truth.get("noise", 0.6))
    forced = truth.get("exposure")

    def photo(exposure: str = "paper"):
        return paper.render(forced or exposure, noise=noise)

    return paper, photo


def sim_state(paper, truth: dict | None = None) -> dict:
    """What a later `fd --sim` run needs to continue on the same sheet with the same real pen."""
    return {"true_tip_offset": [float(v) for v in paper.true_tip_offset], "paper": paper.state(), "truth": truth or {}}


# ----------------------------------------------------------------------------- what a photo shows of the sheet
def _arm_mask(cam, q, tip_offset, shape) -> np.ndarray:
    """Pixels the arm and the tool hide at joint pose q (uint8, 1 = hidden), from the kinematics: nothing
    is placed and nothing is judged behind it, whatever colour the arm has in the photo."""
    mask = np.zeros(shape[:2], np.uint8)
    fr = K.link_frames(q)
    pts = [f[:3, 3] for f in fr] + [K.fk(q)[:3, 3], K.fk(q, K.tool_offset(*tip_offset))[:3, 3]]
    for a, b in zip(pts[:-1], pts[1:]):
        uv = cam.project([a, b])
        depth = float((np.asarray(cam.R) @ ((np.asarray(a) + np.asarray(b)) / 2.0) + np.asarray(cam.tvec).reshape(3))[2])
        if depth < 0.05 or not np.all(np.isfinite(uv)) or np.abs(uv).max() > 20000:
            continue
        cv2.line(mask, (int(round(uv[0, 0])), int(round(uv[0, 1]))), (int(round(uv[1, 0])), int(round(uv[1, 1]))), 1,
                 thickness=max(30, int(round(cam.f * 0.07 / depth))))
    return mask


def _paper_mask(img, hidden=None):
    """(mask of white paper with thin lines closed, its grey level) or (None, 0) when nothing in the photo
    is paper. Ink does not cut the mask (a closing wider than any line)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    if img.ndim == 3:
        b, gr, r = cv2.split(img)
        chroma = cv2.subtract(cv2.max(cv2.max(b, gr), r), cv2.min(cv2.min(b, gr), r))
    else:
        chroma = np.zeros_like(g)
    cand = closed[(closed >= 90) & (chroma <= 45)]
    if cand.size < 0.01 * closed.size:
        return None, 0.0
    level = float(np.percentile(cand, 80))
    mask = ((closed >= 0.72 * level) & (chroma <= 45)).astype(np.uint8)
    if hidden is not None:
        mask[hidden > 0] = 0
    return mask, level


def find_sheet(img, cam, z, INK, hidden=None) -> dict | None:
    """The sheet of paper in a photo taken with the paper exposure: {"polygon": (N, 2) on the table, "rect":
    bool (four clean corners), "size_mm": [long, short], "rot_deg"} or None. Finding it is optional: without
    it the drawing box is laid out on whatever the photo shows as paper."""
    mask, _level = _paper_mask(img, hidden)
    if mask is None:
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 0.02 * mask.size:
        return None
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea)).reshape(-1, 2).astype(float)
    h, w = mask.shape
    cut = bool(np.any(hull[:, 0] <= 1) or np.any(hull[:, 0] >= w - 2) or np.any(hull[:, 1] <= 1) or np.any(hull[:, 1] >= h - 2))
    xy = INK.table_xy(cam, hull, z)
    if not np.all(np.isfinite(xy)) or np.abs(xy).max() > 2.0:
        return None
    rect = cv2.minAreaRect((xy * 1000.0).astype(np.float32))
    box = cv2.boxPoints(rect).astype(float) / 1000.0
    sides = sorted((float(rect[1][0]), float(rect[1][1])), reverse=True)
    if not (60.0 <= sides[1] and sides[0] <= 600.0):
        return None
    fill = float(cv2.contourArea((xy * 1000.0).astype(np.float32))) / max(sides[0] * sides[1], 1.0)
    clean = fill >= 0.93 and not cut
    poly = box if clean else xy
    return {"polygon": poly, "rect": bool(clean), "size_mm": [round(sides[0], 0), round(sides[1], 0)],
            "rot_deg": round(ST.sheet_rot_deg(box), 1), "cut_by_picture": cut}


class _Workspace:
    """What one photo (arm aside) says about where the pen may work: a 4 mm grid over the region with, per
    cell, how far inside visible paper it lies, plus the ink that is on the sheet already. Reach comes from
    the kinematics and is cached for the whole command."""

    G = 0.004

    def __init__(self, run, img, q_photo):
        self.run = run
        arm, tool, cam, INK = run.arm, run.tool, run.cam, run.INK
        self.z = run.tz
        x0, y0, x1, y1 = ST.DEFAULT_REGION
        self.xs = np.arange(x0, x1 + 1e-9, self.G)
        self.ys = np.arange(y0, y1 + 1e-9, self.G)
        hidden = _arm_mask(cam, q_photo, tool.tip_offset, img.shape)
        mask, self.level = _paper_mask(img, hidden)
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.dist_m = np.zeros(gx.shape)
        self.sheet = None
        self.rects = []
        if mask is not None:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            uv = INK.pixels(cam, np.column_stack([gx.ravel(), gy.ravel()]), self.z)
            h, w = mask.shape
            ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            c = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            e = INK.pixels(cam, [c, c + (0.01, 0.0), c + (0.0, 0.01)], self.z)
            ppm = 1.2 * max(float(np.linalg.norm(e[1] - e[0])), float(np.linalg.norm(e[2] - e[0]))) / 0.01
            d = np.zeros(len(uv))
            d[inside] = dist[vi[inside], ui[inside]] / ppm
            self.dist_m = d.reshape(gx.shape)
            self.sheet = find_sheet(img, cam, self.z, INK, hidden)
            for m in INK.find_ink(img, cam, self.z, before=None, region=(x0, y0, x1, y1), min_area_px=6):
                u, v = np.round(INK.pixels(cam, [m.xy], self.z)[0]).astype(int)
                if not (8 <= u < w - 8 and 8 <= v < h - 8) or hidden[v, u]:
                    continue        # the pen itself (it stands where the kinematics say), or the edge of the
                                    # picture cutting the sheet: the line filter answers to both, neither is ink
                ex = m.extent_xy or (m.xy[0], m.xy[1], m.xy[0], m.xy[1])
                self.rects.append(tuple(float(v) for v in ex))
        self.old_ink = len(self.rects)
        self.rects += [tuple(r) for r in run.reserved]

    # -- lookups
    def _ij(self, x, y):
        i, j = int(round((x - self.xs[0]) / self.G)), int(round((y - self.ys[0]) / self.G))
        return (i, j) if 0 <= i < len(self.xs) and 0 <= j < len(self.ys) else None

    def paper(self, x, y, margin: float = 0.0) -> bool:
        ij = self._ij(x, y)
        return ij is not None and self.dist_m[ij] > max(margin, 0.001)

    def reach(self, x, y) -> bool:
        ij = self._ij(x, y)
        if ij is None:
            return False
        cache = self.run.reach_cache
        if ij not in cache:
            cache[ij] = self.run.reach_ok(float(self.xs[ij[0]]), float(self.ys[ij[1]]))
        return cache[ij]

    def usable_centre(self, margin: float):
        """Middle of the paper the tool can work on (every second cell is asked), or None. A sheet that the
        picture cuts, or with tape and other things on it, has no useful "middle of its outline"."""
        pts = [(float(self.xs[i]), float(self.ys[j])) for i, j in np.argwhere(self.dist_m[::2, ::2] > margin) * 2
               if self.reach(float(self.xs[i]), float(self.ys[j]))]
        return tuple(np.mean(np.array(pts), axis=0)) if len(pts) >= 20 else None

    def ink_free(self, rect, clear: float) -> bool:
        """No older ink inside the rectangle grown by `clear` on every side (a rectangle, not a rounded one:
        the readouts search rectangular windows)."""
        ax0, ay0, ax1, ay1 = rect
        for bx0, by0, bx1, by1 in self.rects:
            if bx0 - ax1 < clear and ax0 - bx1 < clear and by0 - ay1 < clear and ay0 - by1 < clear:
                return False
        return True

    def add_rect(self, rect):
        self.rects.append(tuple(float(v) for v in rect))

    def place(self, size, prefer, clear: float, slack: float, avoid=None, max_dist: float = 0.20):
        """Centre (x, y) of the free spot nearest to `prefer` for a footprint of size (sx, sy): the footprint
        with `slack` around it lies on visible paper, the footprint itself is in reach, no older ink within
        `clear` of it, and it does not touch the rectangle `avoid`. None when there is no such spot."""
        sx, sy = float(size[0]), float(size[1])
        ku = int(math.ceil((sx / 2.0 + slack) / self.G))
        kv = int(math.ceil((sy / 2.0 + slack) / self.G))
        ok = ST._window_all(self.dist_m > 0.002, ku, kv)
        idx = np.argwhere(ok)
        if not len(idx):
            return None
        cand = np.column_stack([self.xs[idx[:, 0]], self.ys[idx[:, 1]]])
        dist = np.linalg.norm(cand - np.asarray(prefer, dtype=float)[:2], axis=1)
        for k in np.argsort(dist, kind="stable"):
            if dist[k] > max_dist:
                break
            cx, cy = float(cand[k][0]), float(cand[k][1])
            foot = (cx - sx / 2.0, cy - sy / 2.0, cx + sx / 2.0, cy + sy / 2.0)
            if not self.ink_free(foot, clear):
                continue
            if avoid is not None and not (foot[2] < avoid[0] or foot[0] > avoid[2] or foot[3] < avoid[1] or foot[1] > avoid[3]):
                continue
            if all(self.reach(x, y) for x, y in ((foot[0], foot[1]), (foot[0], foot[3]), (foot[2], foot[1]),
                                                 (foot[2], foot[3]), (cx, cy))):
                return cx, cy
        return None


# ----------------------------------------------------------------------------- one command
def _r(v, n=4):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, n) for x in v]
    return round(float(v), n)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class _Run:
    """The machinery calibrate() and draw() share: power, the side pose, photos, checked pen moves, the
    stop rule's depth guard, and the exit path that always lifts the pen and says where it is."""

    def __init__(self, sess: PenSession, opts: PenOptions, INK):
        self.s, self.o, self.INK = sess, opts, INK
        self.arm, self.cam = sess.arm, sess.cam
        self.tool = self.arm.tool
        self.tz = TL.table_z(self.arm)
        self.pitch = float(self.tool.pitch_deg)
        self.roll = float(self.arm._c("ready")["roll"])
        self.tm = Timer(arm_clock(self.arm))
        self.photos = []
        self.reach_cache = {}
        self.moved = False
        self.lowest_z = None          # lowest plan height a pen-down move was sent to
        self.pen_down_m = 0.0
        self.track = []
        self.fine = bool(TR.fine_accepted(self.arm.cfg))
        self.q_photo = None
        self.notes = []
        self.reserved = []            # rectangles of paper kept free for a later stage (the direction test)
        self._jaw_asked = False       # the jaw was compared with the tool file (power(): once per command)

    # -- kinematics
    def tip(self, q=None):
        q = self.arm.read_q() if q is None else q
        return K.fk(q, self.arm.tip_tool())[:3, 3].copy()

    def reach_ok(self, x, y) -> bool:
        """The tool can work at (x, y): its tip on the paper at the working pitch, inside the joint limits,
        the floor lines and the keep-outs (also for the tool's body), AND it can be lifted straight up from
        there to above tooling.LIFT_HEIGHT at that pitch. The second half is what was missing on 2026-11-11
        (a lift that was out of reach higher up left the pen 8 mm above the paper); in simulation it left
        the pen 14 mm up near the robot, where the wrist runs into its limit."""
        try:
            q = self.arm.solve(np.array([x, y, self.tz + 0.002]), self.pitch, self.roll)
            TL.check_pose(self.arm, self.tool, q, what="work pose")
            for h in (0.02, TL.LIFT_HEIGHT + 0.003):
                q = self.arm.solve(np.array([x, y, self.tz + h]), self.pitch, self.roll, q_seed=q)
            return True
        except SafetyError:
            return False

    def aside_q(self):
        cand = [np.radians((self.arm.cfg.get("pen") or {}).get("aside_q_deg", ASIDE_Q_DEG))]
        try:
            cand.append(TL.work_q(self.arm, self.tool, (0.215, -0.150), 0.075)[0])
        except SafetyError:
            pass
        for q in cand:
            try:
                TL.check_pose(self.arm, self.tool, np.asarray(q, dtype=float), min_height=0.03, what="side pose")
                return np.asarray(q, dtype=float)
            except SafetyError:
                continue
        return TL.ready_q(self.arm, self.tool)

    # -- motion
    def power(self):
        """Torque on (the arm holds where it is) and the speed tier's servo profile. Before the first time, and
        before the torque comes on, the jaw is compared with the tool file (tooling.jaw_in_place: reads only): the
        tip offset of a tool holds for the jaw position the file records, and a jaw that stands elsewhere is a
        refusal (SafetyError) with the arm exactly as it was found. `fd ready` and the painting commands asked
        that since the review of 2026-11-13; `fd draw` and `fd tool calibrate` switched the arm on without it. A
        tool without a recorded jaw position, or recorded on the fixed jaw, is not compared (as everywhere)."""
        if not self._jaw_asked:
            TL.jaw_in_place(self.arm, self.tool)      # (SafetyError: nothing was switched on or written)
            self._jaw_asked = True
        if not self.arm.is_torqued():
            self.arm.torque_on()
            arm_sleep(self.arm, 0.3)
        self.sp = self.arm.apply_profile(None)
        self.moved = True

    def go_aside(self):
        self.power()
        q_to = self.aside_q()
        if float(np.max(np.abs(self.arm.read_q() - q_to))) > math.radians(0.7):
            speed = min(float(self.sp["transit"]), 30.0)
            try:
                TL.free_move(self.arm, q_to, speed, settle=0.5)
            except SafetyError:
                # no checked way from here (e.g. from the folded rest pose): through the tool's stand-by pose
                TL.go_ready(self.arm)
                TL.free_move(self.arm, q_to, speed, settle=0.5)
        arm_sleep(self.arm, 0.3)

    def look(self, tag: str, move: bool = True):
        """(photo, path): the arm goes to the side pose first (move=False: a dry run, the photo is taken
        with the arm where it stands)."""
        if move:
            self.go_aside()
        self.q_photo = self.arm.read_q()
        if not move and float(np.max(np.abs(self.q_photo - self.aside_q()))) > math.radians(3.0):
            self.notes.append("dry run: the photo was taken with the arm where it stands, not from the side pose; the "
                              "arm may hide part of the sheet, and the real run lays everything out again from its own "
                              "photos (after `fd tool calibrate` the arm stands aside: a dry run then shows the real layout)")
        img = self.s.photo("paper")
        path = self.s.save(tag, img)
        if path:
            self.photos.append(path)
        self.tm.mark("photo")
        return img, path

    def guard(self, z: float, limit_z: float, what: str):
        """The stop rule in numbers: no pen-down height below limit_z, whatever the code above computed."""
        if not (math.isfinite(z) and math.isfinite(limit_z)) or z < limit_z - 1e-6:
            raise SafetyError(f"{what}: plan height {z * 1000:.2f} mm is below the depth limit {limit_z * 1000:.2f} mm "
                              "of the stop rule (internal guard); nothing was drawn at this height")

    def piece(self, p_now, parts, hover_z: float, v: float, record: bool = False) -> list:
        """Moves of one pen-down piece: settled hover above its start, down, the parts one after the other
        without lifting (contact allowed, no settle), up. parts: [(N, 3) plan points]. A far start is
        reached by a free move 3 cm up, a near one by a hop at the pen-up height."""
        o = self.o
        start, end = np.asarray(parts[0][0], dtype=float), np.asarray(parts[-1][-1], dtype=float)
        hover = np.array([start[0], start[1], hover_z])
        far = (float(np.hypot(p_now[0] - start[0], p_now[1] - start[1])) > o.hop_max_m or p_now[2] < hover_z - 0.0015
               or p_now[2] - self.tz > ABOVE_M + 0.005)
        mv = []
        if far:
            mv.append({"kind": "free", "xy": (float(start[0]), float(start[1]))})
        mv.append({"kind": "tip", "pts": [hover], "v": 25.0 if far else o.hop_mm_s, "settle": True, "what": "hover"})
        mv.append({"kind": "tip", "pts": [start], "v": o.down_mm_s, "settle": False, "contact": True, "what": "down",
                   "pause": 0.3})
        for part in parts:
            pts = [np.asarray(p, dtype=float) for p in part[1:]]
            if pts:
                mv.append({"kind": "tip", "pts": pts, "v": v, "settle": False, "contact": True, "step": 0.0015,
                           "what": "stroke", "pause": 0.12, "record": record})
        mv.append({"kind": "tip", "pts": [np.array([end[0], end[1], hover_z + 0.004])], "v": o.up_mm_s, "settle": 0.3,
                   "contact": True, "what": "up"})
        return mv

    def _tip_plan(self, q, mv):
        pts = [np.asarray(p, dtype=float) for p in mv["pts"]]
        wps = self.arm.plan_tip_path([(p, self.pitch, self.roll) for p in pts], q_start=q, step_m=mv.get("step", 0.003))
        p0 = self.tip(q)
        length = float(sum(np.linalg.norm(b - a) for a, b in zip([p0, *pts[:-1]], pts)))
        speed = TR.joint_speed_for(wps, length, mv["v"] / 1000.0)
        kw = {"settle": mv.get("settle", False)}
        if mv.get("contact"):
            base_floor = float(self.arm.cfg["floor_z"]) + float(self.arm.cfg["floor_clearance"])
            z_low = min(float(p[2]) for p in pts)
            kw.update(allow_contact=True, contact_depth=max(0.0, base_floor - z_low) + 0.0005)
        return wps, speed, kw, length

    def check(self, q, moves) -> np.ndarray:
        """Every waypoint of the moves, chained from joint pose q, through the arm's own checks. No motion."""
        q = np.asarray(q, dtype=float)
        for mv in moves:
            if mv["kind"] == "free":
                q_to = TL.work_q(self.arm, self.tool, mv["xy"], ABOVE_M)[0]
                TL.plan_free_move(self.arm, self.tool, q, q_to)
                q = q_to
            else:
                wps, speed, kw, _ = self._tip_plan(q, mv)
                self.arm.check_path(wps, speed, allow_contact=kw.get("allow_contact", False),
                                    contact_depth=kw.get("contact_depth"))
                q = np.asarray(wps[-1], dtype=float)
        return q

    def run(self, moves):
        for mv in moves:
            if mv["kind"] == "free":
                q_to = TL.work_q(self.arm, self.tool, mv["xy"], ABOVE_M)[0]
                TL.free_move(self.arm, q_to, min(float(self.sp["transit"]), 30.0), settle=0.3)
                continue
            wps, speed, kw, length = self._tip_plan(self.arm.read_q(), mv)
            if mv.get("contact") and mv["what"] != "up":
                z_low = min(float(p[2]) for p in mv["pts"])
                self.lowest_z = z_low if self.lowest_z is None else min(self.lowest_z, z_low)
            if mv["what"] == "stroke":
                if self.fine:
                    kw["fine"] = True
                self.pen_down_m += length
                if mv.get("record"):
                    tr = TR.record(self.arm, wps, speed, **kw)
                    try:
                        self.track.append(tr.analyse(self.tool))
                    except Exception:
                        pass
                else:
                    self.arm.run_path(wps, speed, **kw)
            else:
                self.arm.run_path(wps, speed, **kw)
            if mv.get("pause"):
                arm_sleep(self.arm, mv["pause"])

    # -- the exit path
    def retreat(self, end: str = "aside") -> dict:
        """Pen up, then aside (or the tool rest pose). Never raises: what did not work is in the report."""
        rep = {}
        if not self.moved or not self.arm.is_torqued():
            return self.pen(rep)
        try:
            TL.lift(self.arm)
        except MOTION_ERRORS as e:
            rep["lift_error"] = f"{type(e).__name__}: {e}"
            return self.pen(rep)
        try:
            if end == "rest":
                r = TL.rest(self.arm)
                rep["rest"] = {k: r.get(k) for k in ("ok", "torque", "tip_height_mm", "reason") if k in r}
            else:
                self.go_aside()
        except MOTION_ERRORS as e:
            rep["retreat_error"] = f"{type(e).__name__}: {e}"
        return self.pen(rep)

    def pen(self, rep: dict | None = None) -> dict:
        rep = dict(rep or {})
        p = self.tip()
        h = float(p[2]) - self.tz
        torque = bool(self.arm.is_torqued())
        if not self.moved:
            where = "not moved"
        elif not torque:
            where = "rest" if (rep.get("rest") or {}).get("ok") else "limp"
        elif "lift_error" in rep:
            where = "stopped"
        elif "retreat_error" in rep:
            where = "lifted"
        else:
            where = "aside"
        rep.update(tip_xyz=_r(p), height_mm=round(h * 1000.0, 1), lifted=bool(h >= TL.MIN_FREE_HEIGHT), where=where,
                   torque="on" if torque else "off")
        if self.lowest_z is not None:
            rep["lowest_plan_z_mm"] = round(self.lowest_z * 1000.0, 2)
        return rep

    def pen_ja(self, pen: dict) -> str:
        where = {"aside": "紙の横の待機姿勢（トルクON）", "rest": "ツール収納姿勢（トルクOFF）", "lifted": "持ち上げた位置（トルクON）",
                 "stopped": "その場（トルクON、持ち上げに失敗）", "limp": "トルクOFF（収納姿勢とは限りません: fd state で確認）",
                 "not moved": "動かしていません"}[pen["where"]]
        if pen["where"] == "not moved":
            return "アームは動かしていません。"
        return f"ペン先は机の {pen['height_mm']:.0f} mm 上、アームは{where}です。"

    # -- layout
    def workspace(self, img) -> _Workspace:
        ws = _Workspace(self, img, self.q_photo if self.q_photo is not None else self.arm.read_q())
        self.tm.mark("layout")
        return ws

    def draw_box(self, ws: _Workspace, box_mm=None, size_mm=None, avoid_ink: bool = True) -> dict:
        """The box `fd draw` uses on this sheet: strokes.fit_box with reach (Arm.solve at the tool's working
        pitch), what the photo shows as paper, the sheet when one was found, and away from older ink."""
        o = self.o
        half = (o.size_mm if size_mm is None else size_mm) / 2000.0

        def ok(x, y):
            return ws.reach(x, y) and (not avoid_ink or ws.ink_free((x, y, x, y), 0.004))

        sheet = ws.sheet["polygon"] if ws.sheet else None
        rot = o.rot_deg if o.rot_deg is not None else (ws.sheet["rot_deg"] if ws.sheet else 0.0)
        prefer = o.at if o.at is not None else ws.usable_centre(o.sheet_margin_m)
        box = ST.fit_box(half, sheet=sheet, ok_xy=ok, visible_xy=lambda x, y: ws.paper(x, y, o.sheet_margin_m),
                         margin_m=o.sheet_margin_m, rot_deg=rot, box_mm=box_mm, region=ST.DEFAULT_REGION, prefer_xy=prefer)
        box["sheet_found"] = bool(ws.sheet)
        self.tm.mark("layout")
        return box

    def preview(self, img, tag: str, lines=(), box=None, marks=()) -> str | None:
        vis = img.copy()
        z = self.tz

        def px(xy):
            return np.round(self.INK.pixels(self.cam, np.asarray(xy, dtype=float).reshape(-1, 2), z)).astype(np.int32)

        if box is not None and box.get("corners"):
            cv2.polylines(vis, [px(box["corners"])], True, (0, 160, 0), 2, cv2.LINE_AA)
        for ln in lines:
            cv2.polylines(vis, [px(ln)], False, (0, 0, 255), 1, cv2.LINE_AA)
        for m in marks:
            c = px([m])[0]
            cv2.circle(vis, (int(c[0]), int(c[1])), 7, (255, 80, 0), 2, cv2.LINE_AA)
        path = self.s.save(tag, vis)
        if path:
            self.photos.append(path)
        return path

    # -- a ladder of dashes and the search for the contact height
    def ladder(self, prefer, zs, tag: str, limit_z: float, avoid) -> dict:
        """One ladder: fresh photo, a spot of untouched paper, can_show_ink there, the dashes (deeper =
        later), photo, readout. Raises Stop("visual") when the photo cannot answer, Stop("layout") when
        there is no untouched paper."""
        o, INK = self.o, self.INK
        before, bpath = self.look(f"{tag}_before")
        ws = self.workspace(before)
        can0 = INK.can_show_ink(before, self.cam, self.tz)
        if not can0["ok"]:
            raise Stop("visual", f"the photo cannot show ink: {can0['reason']}", "check light and exposure "
                       "(`fd snap --exposure paper`), then repeat; nothing was drawn", photo=bpath)
        n = len(zs)
        slack = float(INK.MAX_OFFSET_M) + 0.002
        spot = ws.place(((n - 1) * o.rung_gap_m, o.dash_m), prefer, clear=max(o.clear_m, slack), slack=slack, avoid=avoid)
        if spot is None:
            raise Stop("layout", f"no untouched paper in reach for a ladder near ({prefer[0]:.3f}, {prefer[1]:.3f}): it "
                       f"needs {(2 * slack + (n - 1) * o.rung_gap_m) * 1000:.0f} x {(2 * slack + o.dash_m) * 1000:.0f} mm "
                       f"without older ink ({ws.old_ink} piece(s) of ink on the sheet)",
                       "put a fresh sheet under the pen (or `--marks-on scrap` on a sheet that will be thrown away), "
                       "then repeat this stage", photo=bpath)
        x0 = spot[0] - (n - 1) * o.rung_gap_m / 2.0
        rungs = [((x0 + o.rung_gap_m * i, spot[1] - o.dash_m / 2.0), (x0 + o.rung_gap_m * i, spot[1] + o.dash_m / 2.0))
                 for i in range(n)]
        pts = np.array(rungs).reshape(-1, 2)
        window = (pts[:, 0].min() - INK.MAX_OFFSET_M, pts[:, 1].min() - INK.MAX_OFFSET_M,
                  pts[:, 0].max() + INK.MAX_OFFSET_M, pts[:, 1].max() + INK.MAX_OFFSET_M)
        can = INK.can_show_ink(before, self.cam, self.tz, window)
        if not can["ok"]:
            raise Stop("visual", f"the photo cannot show ink where the ladder would go: {can['reason']}",
                       "check light, exposure and what lies on the sheet, then repeat; nothing was drawn", photo=bpath)
        q = self.arm.read_q()
        p, moves = self.tip(q), []
        for (a, b), z in zip(rungs, zs):
            self.guard(z, limit_z, f"ladder {tag}")
            moves += self.piece(p, [np.array([[a[0], a[1], z], [b[0], b[1], z]])], z + o.lift_m, o.v_mm_s)
            p = np.array([b[0], b[1], z + o.lift_m + 0.004])
        try:
            self.check(q, moves)
        except SafetyError as e:
            raise Stop("plan", f"the ladder at ({spot[0]:.3f}, {spot[1]:.3f}) did not pass the safety checks: {e}",
                       "nothing was drawn; `fd tool status` shows the tool, `fd tool set` corrects it") from e
        self.run(moves)
        self.tm.mark("ladder")
        after, apath = self.look(f"{tag}_after")
        rd = INK.ladder_readout(after, self.cam, self.tz, rungs, before=before)
        got = [r["inked"] for r in rd["rungs"]]
        out = {"tag": tag, "at": _r(spot), "plan_z_mm": _r([z * 1000.0 for z in zs], 2), "inked": got,
               "first_inked": rd["first_inked"], "offset_mm": rd.get("offset_mm"), "photos": [bpath, apath],
               "rungs": rungs, "zs": list(zs), "rd": rd}
        unknown = [r for r in rd["rungs"] if r["inked"] is None]
        if (unknown and rd["first_inked"] is None and can["ok"] and rd.get("untouched") is not False
                and float(can.get("noise", 9.9)) <= GRAZE_MAX_NOISE
                and [r["i"] for r in unknown] == list(range(unknown[0]["i"], n))
                and all(r.get("speck") and r.get("mark") is not None for r in unknown)):
            # A faint trace at the DEEPEST rung(s), nothing above it, and no proper ink in this ladder to compare
            # it with: the readout cannot say "graze or faint line", but either way the pen TOUCHED the paper
            # at that height, which is what this search asks. It ends here (nothing goes deeper on it); the
            # height is that rung's. Only in a photo quiet enough for ink.py's thresholds (GRAZE_MAX_NOISE): in
            # a noisier one such a speck may be noise, and the search stops at "visual" as for any "cannot judge".
            out["first_inked"] = min(r["i"] for r in unknown)
            out["graze"] = True
            self.notes.append(f"ladder {tag}: only a faint trace of ink (rung {out['first_inked']}): taken as the height "
                              "where the pen first touched")
            return out
        if not rd["judgeable"] or (rd["first_inked"] is None and rd.get("untouched") is False):
            why = next((r["why"] for r in rd["rungs"] if r["inked"] is None), "the paper under the ladder was not untouched")
            raise Stop("visual", f"the ladder cannot be read: {why}", "look at the photo; nothing more was drawn and "
                       "nothing went deeper. Repeat the stage when the photo can show the ink", photo=apath,
                       ladder={k: out[k] for k in ("tag", "at", "plan_z_mm", "inked")})
        return out

    def find_contact(self, prefer, z_ref: float, known: bool, tag: str, avoid, stage: str = "contact") -> dict:
        """Plan height of first ink near `prefer` (SPEC section 6, rules 2 to 4).

        known False: nothing was seen yet; the first ladder starts start_above_m over z_ref (where the model
        expects the paper) and the search never goes below z_ref - max_search_below_m. known True: z_ref is a
        contact height seen in a photo; the search starts known_above_m over it and never presses more than
        after_ink_max_m below it. It goes lower only on a judgeable "no ink" for every rung, each ladder on
        untouched paper; when the TOP rung already left ink it looks higher (never deeper).
        -> {"found", "z", "aim_xy", "ink_xy", "ladders": [...], "why"}"""
        o = self.o
        dz, n = o.rung_dz_m, int(o.rungs)
        top = z_ref + (o.known_above_m if known else o.start_above_m)
        floor = z_ref - (o.after_ink_max_m if known else o.max_search_below_m)
        ladders, cand, ups, no_min = [], None, 0, None

        def hit(lad, i):
            m = lad["rd"]["rungs"][i].get("mark")
            aim = np.mean(np.array(lad["rungs"][i]), axis=0)
            return {"found": True, "z": float(lad["zs"][i]), "aim_xy": _r(aim, 5), "ink_xy": _r(m.xy, 5) if m is not None else None,
                    "ladders": brief(), "why": f"first ink at plan z {lad['zs'][i] * 1000:+.1f} mm"}

        def brief():
            return [{k: ld[k] for k in ("tag", "at", "plan_z_mm", "inked", "offset_mm", "photos", "graze") if k in ld} for ld in ladders]

        for k in range(int(o.max_ladders)):
            zs = [top - i * dz for i in range(n) if top - i * dz >= floor - 1e-9]
            if not zs:
                break
            lad = self.ladder(prefer, zs, f"{tag}{k}", floor, avoid)
            ladders.append(lad)
            first = lad["first_inked"]
            if first is None:                                   # an earned "no ink" on every rung
                if cand is not None:
                    return {**cand, "ladders": brief()}
                no_min = zs[-1]
                top = zs[-1] - dz
                continue
            if first == 0 and not (no_min is not None and no_min <= zs[0] + dz + 1e-9):
                # ink already at the highest dash: the contact is at or above it. Look higher, never deeper.
                cand = hit(lad, 0)
                if ups >= int(o.max_up):
                    raise Stop(stage,
                               f"the highest dash (plan z {zs[0] * 1000:+.1f} mm) still left ink after {ups} ladder(s) "
                               "upwards: the tool is much longer than recorded",
                               "measure the tool and correct it (`fd tool set --along-fingers LEN` or `--tip x,y,z`), "
                               "then calibrate again", ladders=brief())
                ups += 1
                top = zs[0] + n * dz
                continue
            return hit(lad, first)
        if cand is not None:
            return {**cand, "ladders": brief()}
        return {"found": False, "z": None, "aim_xy": None, "ink_xy": None, "ladders": brief(),
                "why": f"no ink down to plan z {floor * 1000:+.1f} mm ({'3 mm below the height where ink was seen' if known else f'{o.max_search_below_m * 1000:.0f} mm below where the model expects the paper'}): "
                       "the search does not go deeper"}


def _result(skill: str, run: _Run | None, tool) -> dict:
    return {"ok": False, "skill": skill, "stage": "start", "reason": "", "next": "", "say": "",
            "tool": getattr(tool, "name", None), "photos": [] if run is None else run.photos, "timings": {}}


def _tool_gate(sess: PenSession, skill: str, need=()):
    """The refusal of a command that has no (calibrated) tool to work with, or None."""
    tool = getattr(sess.arm, "tool", None)
    res = _result(skill, None, tool)
    if tool is None:
        res.update(stage="tool", reason="no tool is attached",
                   next="`fd tool attach --name pen --along-fingers LEN` (LEN: metres from the fingertip to the pen tip)",
                   say="ツールが登録されていません。先に fd tool attach を実行してください。")
        return res
    missing = [k for k in need if not TL.calibration_state(tool).get(k)]
    if missing:
        res.update(stage="tool", reason=f"tool '{tool.name}' is not calibrated ({', '.join(missing)} missing)",
                   next="`fd tool calibrate` (all stages), or `fd tool calibrate --stage " + missing[0] + "`",
                   say=f"ツール「{tool.name}」は未校正です（{', '.join(missing)}）。先に fd tool calibrate を実行してください。")
        return res
    stale = [k for k in need if _stale(tool, k)]
    if stale:
        # (reviewer finding, 2026-11-12: `fd tool set --keep-calibration` changed the tip and `fd draw` then pressed
        # 15 mm into the simulated paper with a contact plane measured for another tip)
        res.update(stage="tool", reason=f"the {' and '.join(stale)} of tool '{tool.name}' were measured with another tip offset "
                                        "or pitch than the tool has now (or carry no record of it): they do not hold any more",
                   next="`fd tool calibrate` (all stages) with the tool as it is now",
                   say=f"ツール「{tool.name}」の校正（{', '.join(stale)}）は、いまの先端位置・角度とは別の値で測ったものです。"
                       "fd tool calibrate をやり直してください。")
        return res
    return None


def _measured_with(tool) -> dict:
    """What a contact plane or a map was measured with: kept with the measurement, compared by _stale."""
    return {"tip_offset": [round(float(v), 5) for v in tool.tip_offset], "pitch_deg": round(float(tool.pitch_deg), 2)}


def _stale(tool, key: str) -> bool:
    """True when the measurement `key` ("contact" | "map") was not made with the tool's present tip offset and
    pitch (0.5 mm / 0.5 deg), or does not say what it was made with."""
    rec = (tool.calibrated or {}).get(key) or {}
    tip, pitch = rec.get("tip_offset"), rec.get("pitch_deg")
    if tip is None or pitch is None or len(tip) != 3:
        return True
    return (max(abs(float(a) - float(b)) for a, b in zip(tip, tool.tip_offset)) > 0.0005
            or abs(float(pitch) - float(tool.pitch_deg)) > 0.5)


# A pen command that was refused because the jaw stands elsewhere than the tool file says (_jaw_refused)
JAW_REFUSED_JA = ("爪が、ツールを登録したときの位置からずれているため、{what}を始めませんでした（ツールの先端は、記録の位置には"
                  "ありません）。fd tool jaw status で爪の位置を確かめ、next のとおりに直してください。")


def _jaw_refused(run: _Run, e) -> str | None:
    """The `next` of a pen command that was refused because the jaw stands elsewhere than the tool file says
    (tooling.jaw_in_place, asked by _Run.power before the torque comes on), or None for any other stop.

    The refusal names the way out for THAT tool in its last sentence ("... Next: `fd tool jaw status` ...":
    tooling._jaw_elsewhere), and `fd ready` hands that sentence on as its next since round 2. Here next and say
    were the general ones of a stop on the way ("look at the reason and at `fd state`", 「校正の途中で止まりました」):
    nothing in either named the jaw (round 3, 2026-11-14). Only for a refusal before anything moved."""
    why = str(e)
    if run.moved or not isinstance(e, SafetyError) or "fd tool jaw" not in why:
        return None
    way = why.partition("Next: ")[2].strip()
    return "nothing moved. " + (way[0].upper() + way[1:] if way else
                                "`fd tool jaw status` shows where the jaw stands and what the tool file records")


def _finish(res: dict, run: _Run, end: str = "aside"):
    res["pen"] = run.retreat(end)
    res["timings"] = {**run.tm.marks, "total": run.tm.total()}
    if run.notes:
        res["notes"] = run.notes
    tail = SIM_JA if run.s.sim else UNVERIFIED_JA
    res["say"] = (res.get("say") or "") + run.pen_ja(res["pen"]) + tail
    return res


# ----------------------------------------------------------------------------- calibrate
def calibrate(sess: PenSession, stages=STAGES, opts: PenOptions | None = None, modules=None) -> dict:
    """`fd tool calibrate`: measure the attached tool on the paper, stage by stage (see the module text).

    stages: any of STAGES, run in that order. Each stage reads the tool file's state (sess.arm.tool) and
    writes its result with sess.save_tool, so one stage can be repeated. opts.dry: a photo from where the arm
    stands and the plan (where the marks would go); nothing moves. modules: (ink,) replacement for tests.
    Returns one dict: ok, stage (the last stage reached, "done" when all passed), reason, next, say,
    stages: {name: {...}}, photos, timings, pen."""
    INK = modules[0] if modules else __import__("ink")
    opts = (opts or PenOptions()).check()
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SafetyError(f"unknown calibration stage(s) {bad}: {', '.join(STAGES)} or all")
    stages = [s for s in STAGES if s in stages]
    gate = _tool_gate(sess, "calibrate")
    if gate:
        return gate
    run = _Run(sess, opts, INK)
    tool = run.tool
    res = _result("calibrate", run, tool)
    res["stages"] = {}
    stage = "start"
    try:
        run.arm.speeds(None)
        img, path = run.look("cal_start", move=not opts.dry)
        stage = "visual"
        can = INK.can_show_ink(img, run.cam, run.tz)
        if not can["ok"]:
            raise Stop("visual", f"the photo cannot show ink: {can['reason']}",
                       "nothing touched the paper. Check light and exposure (`fd snap --exposure paper`), then repeat",
                       photo=path)
        ws = run.workspace(img)
        box = run.draw_box(ws, avoid_ink=False)
        if not box["ok"]:
            raise Stop("layout", f"no place for the drawing box on what the photo shows: {box['reason']}",
                       "move the sheet into reach and view (`fd snap --exposure paper`), then repeat", photo=path)
        res["box"] = {"centre": _r(box["centre"]), "size_mm": round(2000.0 * box["half_m"], 1), "rot_deg": _r(box["rot_deg"], 1),
                      "sheet_found": box["sheet_found"], "note": box["reason"]}
        res["marks_on"] = opts.marks_on
        corners = np.array(box["corners"])
        m = opts.box_margin_m
        avoid = None if opts.marks_on == "scrap" else (corners[:, 0].min() - m, corners[:, 1].min() - m,
                                                       corners[:, 0].max() + m, corners[:, 1].max() + m)
        ctx = {"box": box, "avoid": avoid, "ws": ws, "img": img}
        if "direction" in stages and len(stages) > 1 and not opts.dry:
            # the direction test comes last and needs the largest piece of untouched paper: keep it free
            size = opts.direction_len_m + 0.02
            spot = ws.place((size, size), _spots(run, ctx, "direction")[0], opts.clear_m, 0.006, avoid)
            if spot is not None:
                run.reserved.append((spot[0] - size / 2, spot[1] - size / 2, spot[0] + size / 2, spot[1] + size / 2))
        if opts.dry:
            plan = _calibration_plan(run, ctx, stages)
            res["plan"] = plan
            res["preview"] = run.preview(img, "cal_preview", box=box, marks=[p for v in plan.values() for p in v["at"]])
            res.update(ok=True, stage="planned", next="run it without --dry",
                       say=f"校正の計画ができました（{'・'.join(stages)}）。試し書きは描画枠の" +
                           ("外" if avoid else "中も含む範囲") + "に置きます。")
            return _finish(res, run)
        for name in stages:
            stage = name
            try:
                out = _STAGE_FN[name](run, ctx)
            except Stop as e:
                if not (name == "direction" and e.stage == "layout" and len(stages) > 1):
                    raise
                # a full calibration whose last, optional stage found no paper: the tool can draw (the stroke
                # direction then stays "not measured" and `fd draw` says so)
                out = {"ok": True, "skipped": True, "reason": e.reason, "next": e.next}
                run.notes.append("the direction test was NOT done: " + e.reason + ". " + e.next)
            res["stages"][name] = out
            run.tm.mark(name)
            if not out.get("ok"):
                res.update(ok=False, stage=name, reason=out.get("reason", ""), next=out.get("next", ""))
                res["say"] = f"校正は {name} の段階で止まりました（{out.get('reason', '')}）。"
                return _finish(res, run)
        cal = TL.calibration_state(run.tool)
        ready = cal["contact"] and cal["map"]
        res.update(ok=True, stage="done",
                   next=("`fd draw NAME --dry` shows the layout, `fd draw NAME` draws" if ready else
                         "`fd tool calibrate --stage " + ("contact" if not cal["contact"] else "map") + "` is still needed before `fd draw`"))
        res["say"] = f"ツール「{run.tool.name}」の校正（{'・'.join(stages)}）が終わりました。" + _cal_ja(res["stages"])
    except Stop as e:
        res.update(ok=False, stage=e.stage, reason=e.reason, next=e.next, **{k: v for k, v in e.detail.items()})
        res["say"] = (f"校正を止めました（{e.stage}: {e.reason}）。" +
                      ("写真では線の有無を判定できないため、これ以上深くは下ろしていません。" if e.stage == "visual" else ""))
    except MOTION_ERRORS as e:
        jaw = _jaw_refused(run, e)
        res.update(ok=False, stage=stage, reason=f"{type(e).__name__}: {e}",
                   next=jaw or "look at the reason and at `fd state`; the pen was lifted when that was possible (see \"pen\")")
        res["say"] = JAW_REFUSED_JA.format(what="校正") if jaw else f"校正の途中で止まりました（{stage}: {type(e).__name__}）。"
    except (KeyboardInterrupt, SystemExit) as e:
        # Ctrl-C, a cancelled or timed-out tool call (fd turns SIGTERM into this): the pen must not stay pressed on
        # the paper with the torque on. Fall through to _finish, which lifts it and reports where it is.
        res.update(ok=False, stage=stage, reason=f"interrupted ({type(e).__name__}) during the stage {stage}", interrupted=True,
                   next=INTERRUPTED_NEXT)
        res["say"] = f"校正が途中で中断されました（{stage}）。"
    except Exception as e:  # a bug here: report it, lift, do not go on
        import traceback
        res.update(ok=False, stage=stage, reason=f"internal error: {type(e).__name__}: {e}",
                   trace=traceback.format_exc().strip().splitlines()[-6:])
        res["say"] = f"校正の途中で内部エラーが起きました（{stage}）。"
    return _finish(res, run)


def _cal_ja(st: dict) -> str:
    out = []
    if st.get("tip", {}).get("ok"):
        out.append(f"先端位置を {st['tip']['shift_mm']:.1f} mm 修正")
    if st.get("contact", {}).get("ok"):
        out.append(f"接触高さ {st['contact']['n']} か所（平面からのずれ RMS {st['contact']['rms_mm']} mm）")
    if st.get("map", {}).get("ok"):
        out.append(f"位置補正 印 {st['map']['n']} 個（RMS {st['map']['rms_mm']} mm）")
    if st.get("direction", {}).get("skipped"):
        out.append("線の向きのテストは紙の空きがなく未実施")
    elif st.get("direction", {}).get("ok"):
        ja = {"away": "ロボットから遠ざかる向き", "toward": "ロボットへ向かう向き", "any": "向きによる差なし"}[st["direction"]["prefer"]]
        out.append(f"線の向き: {ja}")
    return "、".join(out) + "。" if out else ""


def _spots(run: _Run, ctx: dict, name: str) -> list:
    """Preferred places (before the search for untouched paper) of a stage's test marks."""
    o, box = run.o, ctx["box"]
    c, h = np.asarray(box["centre"], dtype=float), float(box["half_m"])
    u_hat, v_hat = ST._axes(box["rot_deg"])
    out = h + 0.045
    if o.marks_on == "scrap":
        out = 0.6 * h
    if name == "tip":
        # (beside the box on the robot's right; the strip on its left is kept for the direction test)
        return [tuple(c - u_hat * out)] if o.at is None or o.marks_on != "scrap" else [tuple(o.at)]
    if name == "contact":
        # the four corners first: a plane fitted to heights that scatter by a millimetre must not be
        # extrapolated (places 6 cm apart in x gave a slope of 3 mm per 10 cm from noise alone)
        k = h + 0.012
        pts = [c - u_hat * out - v_hat * k, c - u_hat * out + v_hat * k, c + u_hat * out - v_hat * k, c + u_hat * out + v_hat * k,
               c - u_hat * out, c + u_hat * out, c + v_hat * out, c - v_hat * out]
        if o.marks_on == "scrap":
            pts = [c - u_hat * k - v_hat * k, c - u_hat * k + v_hat * k, c + u_hat * k - v_hat * k, c + u_hat * k + v_hat * k,
                   c, c - u_hat * k, c + u_hat * k, c + v_hat * k]
        return [tuple(p) for p in pts[:int(o.places)]]
    if name == "map":
        n = int(o.grid)
        d = (h + 0.02) if o.marks_on != "scrap" else 0.8 * h      # (ink may land 1.5 cm beside its aim)
        ks = np.linspace(-1.0, 1.0, n)
        pts = [c + u_hat * a * d + v_hat * b * d for b in ks for a in ks]
        if o.marks_on != "scrap":
            pts = [p for p in pts if max(abs(float((p - c) @ u_hat)), abs(float((p - c) @ v_hat))) > h]
        return [tuple(p) for p in pts]
    return [tuple(c + u_hat * (h + 0.058))] if o.marks_on != "scrap" else [tuple(c)]


def _calibration_plan(run: _Run, ctx: dict, stages) -> dict:
    """Dry run: where each stage would start looking for paper (the real places follow from fresh photos)."""
    ws, o, plan = ctx["ws"], run.o, {}
    slack = float(run.INK.MAX_OFFSET_M) + 0.002
    for name in (["direction"] if "direction" in stages else []) + [s for s in stages if s != "direction"]:
        at = []
        for p in _spots(run, ctx, name):
            if name in ("tip", "contact"):
                s = ws.place(((o.rungs - 1) * o.rung_gap_m, o.dash_m), p, max(o.clear_m, slack), slack, ctx["avoid"])
            elif name == "map":
                s = ws.place((0.001, o.mark_dash_m), p, o.clear_m, 0.015, ctx["avoid"])
            else:
                s = ws.place((0.04, 0.04), p, o.clear_m, 0.006, ctx["avoid"])
            if s is not None:
                at.append(_r(s))
                half = (0.02, 0.02) if name == "direction" else (0.001, 0.0015) if name == "map" else (0.006, 0.003)
                ws.add_rect((s[0] - half[0], s[1] - half[1], s[0] + half[0], s[1] + half[1]))
        plan[name] = {"at": at, "wanted": len(_spots(run, ctx, name))}
    return {name: plan[name] for name in stages}


def _save(run: _Run, tool):
    run.s.save_tool(tool)
    run.arm.tool = tool
    run.tool = tool


def _stage_tip(run: _Run, ctx: dict) -> dict:
    """The tool tip from ink: the first inked rung of a ladder search. Its plan height against the table
    gives the error along the tool, its place against the aim the sideways and radial errors. The corrected
    offset replaces tool.tip_offset; contact plane and map (measured with the old offset) are dropped."""
    tool, o = run.tool, run.o
    old = [float(v) for v in tool.tip_offset]
    prev = (tool.calibrated or {}).get("tip") or {}
    if tool.contact:
        known, z_ref = True, None
    elif prev.get("contact_z") is not None:
        known, z_ref = True, float(prev["contact_z"])
    else:
        known, z_ref = False, run.tz
    prefer = _spots(run, ctx, "tip")[0]
    if z_ref is None:
        z_ref = TL.contact_z(tool, prefer)
    found = run.find_contact(prefer, z_ref, known, "tip", ctx["avoid"], stage="tip")
    out = {"ok": False, "ladders": found["ladders"], "tip_offset_before": _r(old, 5)}
    if not found["found"] or found["ink_xy"] is None:
        out.update(reason=found["why"], next="check that the pen writes and that the recorded tip is about right "
                   "(`fd tool status`, `fd tool set`); then repeat `fd tool calibrate --stage tip`")
        return out
    aim, ink_xy, z_c = np.array(found["aim_xy"]), np.array(found["ink_xy"]), float(found["z"])
    err = np.array([ink_xy[0] - aim[0], ink_xy[1] - aim[1], run.tz - z_c])      # real tip minus modelled tip
    q = run.arm.solve(np.array([aim[0], aim[1], z_c]), run.pitch, run.roll)
    d_loc = K.fk(q)[:3, :3].T @ err
    shift = float(np.linalg.norm(d_loc))
    out.update(first_ink_plan_z_mm=round(z_c * 1000.0, 2), shift_base_mm=_r(err * 1000.0, 2), shift_mm=round(shift * 1000.0, 2),
               delta_local_mm=_r(d_loc * 1000.0, 2))
    if shift > o.max_tip_shift_m:
        out.update(reason=f"the ink lies {shift * 1000:.0f} mm from where the model has the tip (limit "
                          f"{o.max_tip_shift_m * 1000:.0f} mm): the recorded tip offset is not even roughly right",
                   next="measure the tool by hand and record it (`fd tool set --along-fingers LEN` or `--tip x,y,z`), "
                        "then repeat")
        return out
    new = [round(float(v), 5) for v in np.array(old) + d_loc]
    cal = {k: v for k, v in (tool.calibrated or {}).items() if k not in ("contact", "map", "direction")}
    dropped = [k for k in ("contact", "map") if getattr(tool, "contact" if k == "contact" else "corr_map")]
    cal["tip"] = {"at": _now(), "from": "ink", "place": _r(aim, 4), "ink_xy": _r(ink_xy, 4), "shift_mm": out["shift_mm"],
                  "delta_local_mm": out["delta_local_mm"], "tip_offset_before": _r(old, 5), "contact_z": run.tz,
                  "resolution_mm": round(o.rung_dz_m * 1000.0, 2)}
    tool.tip_offset, tool.contact, tool.corr_map, tool.calibrated = new, None, None, cal
    _save(run, tool)
    run.reach_cache.clear()
    out.update(ok=True, tip_offset=new, dropped=dropped,
               note="the tip offset changed: the contact plane and the map have to be measured with it (next stages)")
    return out


def _stage_contact(run: _Run, ctx: dict) -> dict:
    """Ladders at three or more places -> the plane "plan z of first ink". A plane whose residual exceeds
    max_plane_rms_mm is refused (not stored)."""
    tool, o = run.tool, run.o
    prev = (tool.calibrated or {}).get("tip") or {}
    pts, places = [], []
    for i, prefer in enumerate(_spots(run, ctx, "contact")):
        if tool.contact:
            known, z_ref = True, TL.contact_z(tool, prefer)
        elif pts:
            j = int(np.argmin([math.dist(p[:2], prefer) for p in pts]))
            known, z_ref = True, pts[j][2]
        elif prev.get("contact_z") is not None:
            known, z_ref = True, float(prev["contact_z"])
        else:
            known, z_ref = False, run.tz
        try:
            found = run.find_contact(prefer, z_ref, known, f"contact{chr(97 + i)}", ctx["avoid"])
        except Stop as e:
            if e.stage != "layout":
                raise
            # no untouched paper near this place: the other places may still fix the plane
            found = {"found": False, "z": None, "aim_xy": None, "ink_xy": None, "ladders": [], "why": e.reason}
        places.append({"wanted": _r(prefer), "found": found["found"], "plan_z_mm": None if found["z"] is None else round(found["z"] * 1000.0, 2),
                       "at": found["aim_xy"], "why": found["why"], "ladders": found["ladders"]})
        if found["found"]:
            pts.append([float(found["aim_xy"][0]), float(found["aim_xy"][1]), float(found["z"])])
    out = {"ok": False, "places": places, "n": len(pts)}
    if len(pts) < 3:
        out.update(reason=f"first ink was found at {len(pts)} of {len(places)} places; a plane needs 3",
                   next="look at the photos of the places without ink (the pen may not reach the paper there: the "
                        "sheet is lower, or the tip offset is off); then repeat `fd tool calibrate --stage contact`")
        return out
    p = np.array(pts)
    x0, y0 = float(p[:, 0].mean()), float(p[:, 1].mean())
    a_mat = np.column_stack([np.ones(len(p)), p[:, 0] - x0, p[:, 1] - y0])
    if np.linalg.matrix_rank(a_mat, tol=1e-6) < 3:
        out.update(reason="the places lie on one line: they do not fix a plane", next="repeat with `--marks-on scrap` or a larger sheet")
        return out
    coef, *_ = np.linalg.lstsq(a_mat, p[:, 2], rcond=None)
    res = p[:, 2] - a_mat @ coef
    rms, mx = float(np.sqrt(np.mean(res ** 2))) * 1000.0, float(np.abs(res).max()) * 1000.0
    out.update(rms_mm=round(rms, 2), max_mm=round(mx, 2), a_mm=round(float(coef[0]) * 1000.0, 2),
               tilt_mm_per_100mm=_r([coef[1] * 100.0, coef[2] * 100.0], 2))
    if len(pts) == 3:
        out["note"] = "three places fix a plane exactly: its residual says nothing (use 4 places to see it)"
    if max(abs(float(coef[1])), abs(float(coef[2]))) > 0.06:
        out.update(reason=f"the fitted contact plane is tilted by {max(abs(float(coef[1])), abs(float(coef[2]))) * 100:.1f} mm "
                          f"per 10 cm: not believable for a sheet on the table ({[round(z * 1000.0, 1) for z in p[:, 2]]} mm)",
                   next="look at the photos of the ladders; flatten / tape the sheet and repeat `fd tool calibrate --stage contact`")
        return out
    if rms > o.max_plane_rms_mm:
        out.update(reason=f"the contact heights do not lie on a plane (residual rms {rms:.2f} mm, limit "
                          f"{o.max_plane_rms_mm} mm): {[round(z * 1000.0, 1) for z in p[:, 2]]} mm",
                   next="the height wanders too much or the sheet is not flat: flatten / tape the sheet and repeat "
                        "`fd tool calibrate --stage contact`")
        return out
    corners = np.array(ctx["box"]["corners"])
    allxy = np.vstack([p[:, :2], corners])
    tool.contact = {"x0": round(x0, 5), "y0": round(y0, 5), "a": round(float(coef[0]), 6), "bx": round(float(coef[1]), 6),
                    "by": round(float(coef[2]), 6), "rms_mm": round(rms, 2), "max_mm": round(mx, 2), "n": len(pts),
                    "points": _r(pts, 5)}
    tool.region = _r([allxy[:, 0].min(), allxy[:, 1].min(), allxy[:, 0].max(), allxy[:, 1].max()], 4)
    cal = dict(tool.calibrated or {})
    cal["contact"] = {"at": _now(), "rms_mm": round(rms, 2), "max_mm": round(mx, 2), "n": len(pts),
                      "resolution_mm": round(o.rung_dz_m * 1000.0, 2), **_measured_with(tool)}
    tool.calibrated = cal
    _save(run, tool)
    out["ok"] = True
    return out


def _marks_z(run: _Run, xy, press: float) -> float:
    """Plan height for a calibration mark at xy: `press` below the contact plane, never more than the stop
    rule allows below it."""
    zc = TL.contact_z(run.tool, xy)
    z = zc - min(float(press), run.o.after_ink_max_m)
    run.guard(z, zc - run.o.after_ink_max_m, "calibration mark")
    return z


def _stage_map(run: _Run, ctx: dict) -> dict:
    """Marks at commanded places -> the affine map commanded -> actual. A mark that left no ink is
    reported, never guessed; fewer than min_marks found = no map; an RMS above max_map_rms_mm = no map."""
    tool, o, INK = run.tool, run.o, run.INK
    if not tool.contact:
        return {"ok": False, "reason": "the contact plane is not measured yet",
                "next": "`fd tool calibrate --stage contact` first"}
    # After the tip was corrected at ONE place the ink still lands up to 1.5 cm beside its aim elsewhere
    # (2026-11-11: 0.1 to 16.5 mm): paper is asked for 15 mm around a mark, the readout looks 25 mm around it.
    slack, look = 0.015, float(INK.MAX_OFFSET_M)
    before, bpath = run.look("map_before")
    ws = run.workspace(before)
    cmd, near_ink = [], 0
    for prefer in _spots(run, ctx, "map"):
        s = ws.place((0.001, o.mark_dash_m), prefer, o.clear_m, slack, ctx["avoid"], max_dist=0.06)
        if s is None:
            # no spot 3 cm from older ink near this place: 1.5 cm will do for a mark (the readout takes NEW ink
            # only, against the photo taken just before, and never further than 25 mm from the aim)
            s = ws.place((0.001, o.mark_dash_m), prefer, 0.015, slack, ctx["avoid"], max_dist=0.04)
            near_ink += s is not None
        if s is None:
            continue
        cmd.append(s)
        ws.add_rect((s[0] - 0.001, s[1] - o.mark_dash_m / 2.0, s[0] + 0.001, s[1] + o.mark_dash_m / 2.0))
    out = {"ok": False, "commanded": _r(cmd), "photos": [bpath]}
    if near_ink:
        out["near_older_ink"] = near_ink
    if len(cmd) < o.min_marks:
        raise Stop("layout", f"untouched paper in reach for only {len(cmd)} of the marks ({o.min_marks} are needed)",
                   "put a fresh sheet under the pen (or `--marks-on scrap`), then repeat `fd tool calibrate --stage map`",
                   photo=bpath)
    for s in cmd:
        can = INK.can_show_ink(before, run.cam, run.tz, (s[0] - slack, s[1] - slack, s[0] + slack, s[1] + slack))
        if not can["ok"]:
            raise Stop("visual", f"the photo cannot show ink around the mark at ({s[0]:.3f}, {s[1]:.3f}): {can['reason']}",
                       "nothing was drawn; check light, exposure and what lies on the sheet, then repeat", photo=bpath)
    q = run.arm.read_q()
    p, moves = run.tip(q), []
    for s in cmd:
        z = _marks_z(run, s, o.mark_press_m)
        a, b = (s[0], s[1] - o.mark_dash_m / 2.0, z), (s[0], s[1] + o.mark_dash_m / 2.0, z)
        moves += run.piece(p, [np.array([a, b])], z + min(o.mark_press_m, o.after_ink_max_m) + o.lift_m, o.v_mm_s)
        p = np.array([b[0], b[1], z + o.lift_m + 0.006])
    try:
        run.check(q, moves)
    except SafetyError as e:
        raise Stop("plan", f"the marks did not pass the safety checks: {e}", "nothing was drawn") from e
    run.run(moves)
    run.tm.mark("marks")
    after, apath = run.look("map_after")
    out["photos"].append(apath)
    rd = INK.marks_readout(after, run.cam, run.tz, cmd, before=before, max_dist_m=look)
    marks = [{"commanded": _r(c), "inked": m["inked"], "ink_xy": m["xy"], "faint": m.get("faint", False), "why": m["why"]}
             for c, m in zip(cmd, rd["marks"])]
    ok_i = [i for i, m in enumerate(rd["marks"]) if m["inked"]]
    out.update(marks=marks, found=len(ok_i), missing=rd["missing"], unknown=rd["unknown"])
    if len(ok_i) < o.min_marks:
        out.update(reason=f"only {len(ok_i)} of {len(cmd)} marks left ink that the photo shows ({len(rd['missing'])} left "
                          f"none, {len(rd['unknown'])} cannot be judged); a map needs {o.min_marks}",
                   next="where a mark left no ink the pen did not reach the paper: repeat `fd tool calibrate --stage "
                        "contact`, then `--stage map`")
        return out
    c = np.array([cmd[i] for i in ok_i])
    act = np.array([rd["marks"][i]["xy"] for i in ok_i])
    fit = ST.fit_map(c, act, centre=ctx["box"]["centre"], kind="affine")
    err = np.linalg.norm(act - c, axis=1) * 1000.0
    out.update(n=fit["n"], rms_mm=fit["rms_mm"], max_mm=fit["max_mm"], offset_mm={"mean": round(float(err.mean()), 2),
                                                                                   "max": round(float(err.max()), 2)})
    if fit["rms_mm"] > o.max_map_rms_mm:
        out.update(reason=f"the marks do not fit an affine map (rms {fit['rms_mm']} mm, limit {o.max_map_rms_mm} mm)",
                   next="look at the photo (a mark read at the wrong place?), then repeat `fd tool calibrate --stage map`")
        return out
    tool.corr_map = {k: fit[k] for k in ("centre", "kind", "coef", "rms_mm", "max_mm", "n")}
    cal = dict(tool.calibrated or {})
    cal["map"] = {"at": _now(), "rms_mm": fit["rms_mm"], "max_mm": fit["max_mm"], "n": fit["n"],
                  "missing": len(rd["missing"]), "unknown": len(rd["unknown"]), **_measured_with(tool)}
    tool.calibrated = cal
    _save(run, tool)
    out["ok"] = True
    if rd["missing"] or rd["unknown"]:
        out["note"] = (f"{len(rd['missing'])} mark(s) left no ink and {len(rd['unknown'])} could not be judged: they are "
                       "not part of the map")
    return out


def _stage_direction(run: _Run, ctx: dict) -> dict:
    """Which direction of travel keeps its ink (SPEC 9.1): two short lines away from the robot, two towards
    it and two across, read with ink.stroke_coverage. Stored as tool.calibrated["direction"]; nothing is
    assumed about pushing or pulling."""
    tool, o, INK = run.tool, run.o, run.INK
    if not (tool.contact and tool.corr_map):
        return {"ok": False, "reason": "the direction test needs the contact plane and the map",
                "next": "`fd tool calibrate --stage contact`, then `--stage map`, first"}
    before, bpath = run.look("dir_before")
    run.reserved = []                                  # (its own reservation)
    ws = run.workspace(before)
    ln = float(o.direction_len_m)
    size = ln + 0.02
    spot = ws.place((size, size), _spots(run, ctx, "direction")[0], o.clear_m, 0.006, ctx["avoid"])
    near_ink = False
    if spot is None:
        # These lines are read with a 2.5 mm strip against the photo taken just before: ink 1.5 cm away
        # cannot be taken for them (the 3 cm rule is there for the ladder readout, which takes any new ink
        # within 25 mm for rungs). Marks whose ink landed beside its aim may have eaten the 3 cm.
        spot = ws.place((size, size), _spots(run, ctx, "direction")[0], 0.015, 0.006, ctx["avoid"])
        near_ink = spot is not None
    if spot is None:
        raise Stop("layout", f"no untouched paper in reach for the direction test ({size * 1000:.0f} mm square)",
                   "put a fresh sheet under the pen (or `--marks-on scrap`), then repeat `fd tool calibrate --stage "
                   "direction`", photo=bpath)
    can = INK.can_show_ink(before, run.cam, run.tz, (spot[0] - size / 2, spot[1] - size / 2, spot[0] + size / 2, spot[1] + size / 2))
    if not can["ok"]:
        raise Stop("visual", f"the photo cannot show ink where the direction test would go: {can['reason']}",
                   "nothing was drawn; check light and exposure, then repeat", photo=bpath)
    c = np.array(spot)
    r_hat = ST.radial_lean()(c)                        # away from the robot (from the pan axis), at this place
    t_hat = np.array([-r_hat[1], r_hat[0]])
    lines, kinds = [], []
    c_rad = c - r_hat * 0.007
    for k, off in enumerate((-0.009, -0.003, 0.003, 0.009)):
        a, b = c_rad + t_hat * off - r_hat * ln / 2.0, c_rad + t_hat * off + r_hat * ln / 2.0
        lines.append(np.array([a, b]) if k % 2 == 0 else np.array([b, a]))
        kinds.append("away" if k % 2 == 0 else "toward")
    for k, off in enumerate((0.007, 0.013)):
        a, b = c_rad + r_hat * (ln / 2.0 + off) - t_hat * ln / 2.0, c_rad + r_hat * (ln / 2.0 + off) + t_hat * ln / 2.0
        lines.append(np.array([a, b]) if k == 0 else np.array([b, a]))
        kinds.append("across")
    ink_lines = [ST.resample(pl, 0.0012) for pl in lines]
    q = run.arm.read_q()
    p, moves = run.tip(q), []
    press = min(o.press_m, o.after_ink_max_m)
    for pl in ink_lines:
        cm = ST.invert_map(pl, tool.corr_map)
        pts = np.array([[x, y, _marks_z(run, (x, y), press)] for x, y in cm])
        moves += run.piece(p, [pts], pts[0][2] + press + o.lift_m, o.v_mm_s)
        p = np.array([pts[-1][0], pts[-1][1], pts[-1][2] + press + o.lift_m + 0.004])
    try:
        run.check(q, moves)
    except SafetyError as e:
        raise Stop("plan", f"the direction test did not pass the safety checks: {e}", "nothing was drawn") from e
    run.run(moves)
    run.tm.mark("lines")
    after, apath = run.look("dir_after")
    # The lines lie 6 mm apart and are read with a strip of +-2 mm. The map is good to 1 mm at its marks but
    # worse beside them (4 mm in a corner), and ink beside the strip would count as "missing" for every
    # direction alike: the whole block is first shifted (up to 3 mm) to where its ink lies best.
    cov, shift = None, (0.0, 0.0)
    for dx in (0.0, -0.0015, 0.0015, -0.003, 0.003):
        for dy in (0.0, -0.0015, 0.0015, -0.003, 0.003):
            c2 = INK.stroke_coverage(after, run.cam, run.tz, [pl + (dx, dy) for pl in ink_lines], before=before, tol_m=0.002)
            if cov is None or c2["drawn_fraction"] > cov["drawn_fraction"] + 0.01:
                cov, shift = c2, (dx, dy)
    out = {"ok": False, "at": _r(spot), "photos": [bpath, apath], "block_shift_mm": _r([shift[0] * 1000.0, shift[1] * 1000.0], 1),
           "per_line": [{"kind": k, "drawn": f} for k, f in zip(kinds, cov["per_stroke"])]}
    if not cov["judgeable"] or any(f is None for f in cov["per_stroke"]):
        raise Stop("visual", "the photo cannot show whether the test lines are whole "
                   f"({(cov.get('can') or {}).get('reason', 'too much of them cannot be judged')})",
                   "look at the photo; repeat `fd tool calibrate --stage direction` when the lines can be seen", photo=apath,
                   direction=out)
    missing = {k: round(1.0 - float(np.mean([f for kk, f in zip(kinds, cov["per_stroke"]) if kk == k])), 3)
               for k in ("away", "toward", "across")}
    diff = missing["toward"] - missing["away"]
    prefer = "any"
    if diff > o.direction_min_diff and missing["toward"] > 2.0 * missing["away"] + 0.02:
        prefer = "away"
    elif -diff > o.direction_min_diff and missing["away"] > 2.0 * missing["toward"] + 0.02:
        prefer = "toward"
    cal = dict(tool.calibrated or {})
    cal["direction"] = {"prefer": prefer, "missing": missing, "at": _now(), "lines": len(lines),
                        "line_mm": round(ln * 1000.0, 1), "offset_mm": cov["offset_mm"]}
    tool.calibrated = cal
    _save(run, tool)
    out.update(ok=True, prefer=prefer, missing=missing, offset_mm=cov["offset_mm"])
    if near_ink:
        out["note"] = "no spot 3 cm from older ink was left: the test lines lie 1.5 to 3 cm from it"
    return out


_STAGE_FN = {"tip": _stage_tip, "contact": _stage_contact, "map": _stage_map, "direction": _stage_direction}


# ----------------------------------------------------------------------------- draw
def stroke_policy(tool, strokes: str = "auto") -> dict:
    """Which way the pieces of a drawing travel: {"policy": "as-is" | "away" | "toward", "source", "note"}.
    "auto" takes what the direction test of the tool measured; without a measurement the drawing is drawn as
    its file says and the result says that no direction was chosen from a measurement."""
    if strokes != "auto":
        return {"policy": strokes, "source": "option", "note": f"stroke direction '{strokes}' was asked for"}
    d = (getattr(tool, "calibrated", None) or {}).get("direction")
    if not d:
        return {"policy": "as-is", "source": "not measured",
                "note": "the stroke direction was NOT chosen from a measurement (no direction test in the tool file: "
                        "`fd tool calibrate --stage direction`); the strokes are drawn as the file has them"}
    if d.get("prefer") in ("away", "toward"):
        return {"policy": d["prefer"], "source": "measured", "missing": d.get("missing"),
                "note": f"the direction test of {d.get('at')} lost less ink travelling {d['prefer']} "
                        f"(missing: {d.get('missing')})"}
    return {"policy": "as-is", "source": "measured", "missing": d.get("missing"),
            "note": f"the direction test of {d.get('at')} found no direction clearly better (missing: {d.get('missing')})"}


def plan_drawing(run: _Run, drawing: dict, box: dict) -> dict:
    """The laid-out drawing as pen-down pieces: ink polylines (where the ink shall be), commanded polylines
    (through the inverse of the map) and plan heights (press below the contact plane)."""
    o, tool = run.o, run.tool
    polys = ST.layout(drawing, box["centre"], box["half_m"], box["rot_deg"])
    pol = stroke_policy(tool, o.strokes)
    plan = ST.plan_pieces(polys, ST.direction_lean(pol["policy"]), tool.corr_map)
    press = float(o.press_m)
    pieces = []
    for pc in plan["pieces"]:
        parts = []
        for part in pc["parts"]:
            pts = []
            for x, y in part:
                zc = TL.contact_z(tool, (x, y))
                z = zc - press
                run.guard(z, zc - MAX_PRESS_M, "drawing")
                pts.append([x, y, z])
            parts.append(np.array(pts))
        pieces.append({"stroke": pc["stroke"], "ink": pc["ink"], "parts": parts})
    return {"pieces": pieces, "stats": plan["stats"], "policy": pol, "strokes": plan["strokes"], "polys": polys}


def draw(sess: PenSession, drawing, opts: PenOptions | None = None, modules=None) -> dict:
    """`fd draw FILE|NAME`: check the tool -> photo -> layout -> preview -> draw -> photo check.

    drawing: a file, a name in drawings/ or a parsed drawing (strokes.load). Needs a tool with a contact
    plane and a map (else stage "tool" with the command to run). opts.dry stops after the preview and moves
    nothing. ok is decided by ink.stroke_coverage of the final photo (drawn fraction >= opts.min_coverage)."""
    INK = modules[0] if modules else __import__("ink")
    opts = (opts or PenOptions()).check()
    gate = _tool_gate(sess, "draw", need=("contact", "map"))
    if gate:
        return gate
    run = _Run(sess, opts, INK)
    tool = run.tool
    res = _result("draw", run, tool)
    stage = "load"
    try:
        try:
            job = ST.load(drawing)
        except ValueError as e:
            raise Stop("load", str(e), f"drawings that ship with the harness: {', '.join(ST.names())}") from e
        res["drawing"] = {"name": job.get("name"), "strokes": len(job["strokes"]), "box_mm": job["box_mm"]}
        run.arm.speeds(None)
        stage = "photo"
        img, path = run.look("draw_before", move=not opts.dry)
        can = INK.can_show_ink(img, run.cam, run.tz)
        if not can["ok"]:
            raise Stop("visual", f"the photo cannot show ink: {can['reason']}",
                       "nothing touched the paper. Check light and exposure (`fd snap --exposure paper`), then repeat",
                       photo=path)
        stage = "layout"
        ws = run.workspace(img)
        box = run.draw_box(ws, box_mm=job["box_mm"])
        if not box["ok"]:
            raise Stop("layout", f"no place for the drawing: {box['reason']}" + (f" ({ws.old_ink} piece(s) of older ink "
                       "are kept clear)" if ws.old_ink else ""),
                       "a fresh sheet, a smaller `--size MM`, or the sheet moved into reach and view", photo=path)
        res["layout"] = {"centre": _r(box["centre"]), "size_mm": round(2000.0 * box["half_m"], 1), "rot_deg": _r(box["rot_deg"], 1),
                         "sheet_found": box["sheet_found"], "note": box["reason"] or "", "older_ink_pieces": ws.old_ink,
                         "pitch_deg": run.pitch}
        if not box["sheet_found"]:
            run.notes.append("the sheet's outline was not found in the photo: the box was laid out on what the photo "
                             "shows as paper")
        if opts.at is not None and math.dist(opts.at, box["centre"]) > 0.006:
            run.notes.append(f"the box could not be centred at {list(opts.at)}: it is at {_r(box['centre'])}")
        c = np.array(box["corners"])
        can = INK.can_show_ink(img, run.cam, run.tz, (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()))
        if not can["ok"] and not opts.dry:
            raise Stop("visual", f"the photo cannot show ink in the drawing box: {can['reason']}",
                       "nothing touched the paper; check light, exposure and what lies on the sheet", photo=path)
        stage = "plan"
        plan = plan_drawing(run, job, box)
        st, pol = plan["stats"], plan["policy"]
        zs = np.concatenate([pt[:, 2] for pc in plan["pieces"] for pt in pc["parts"]])
        res["plan"] = {"pieces": st["count"], "ink_mm": round(st["ink_m"] * 1000.0, 0), "penup_mm": round(st["penup_m"] * 1000.0, 0),
                       "strokes": pol["policy"], "strokes_source": pol["source"], "press_mm": round(opts.press_m * 1000.0, 2),
                       "v_mm_s": opts.v_mm_s, "plan_z_mm": [round(float(zs.min()) * 1000.0, 2), round(float(zs.max()) * 1000.0, 2)],
                       "expected_s": round(st["ink_m"] * 1000.0 / opts.v_mm_s + 6.0 * st["count"], 0),
                       "fine_mode": run.fine}
        run.notes.append(pol["note"])
        res["preview"] = run.preview(img, "draw_preview", lines=[pc["ink"] for pc in plan["pieces"]], box=box)
        q = run.aside_q()            # (a dry run is checked as the real run starts: from the side pose)
        p, jobs = run.tip(q), []
        for pc in plan["pieces"]:
            s0 = pc["parts"][0][0]
            mv = run.piece(p, pc["parts"], s0[2] + opts.press_m + opts.lift_m, opts.v_mm_s, record=True)
            e = pc["parts"][-1][-1]
            p = np.array([e[0], e[1], s0[2] + opts.press_m + opts.lift_m + 0.004])
            jobs.append(mv)
        try:
            qq = q
            for i, mv in enumerate(jobs):
                qq = run.check(qq, mv)
        except SafetyError as e:
            raise Stop("plan", f"piece {i + 1} of {len(jobs)} did not pass the safety checks: {e}",
                       "nothing was drawn. A smaller `--size MM` or another place (`--at x,y`) may fit") from e
        run.tm.mark("plan")
        if opts.dry:
            res.update(ok=True, stage="planned", dry=True, next="look at the preview image, then run it without --dry",
                       say=f"「{job.get('name')}」を描く計画ができました（{st['count']} 本・線 {st['ink_m'] * 1000:.0f} mm・"
                           f"約 {res['plan']['expected_s']:.0f} 秒）。プレビュー画像を確認してください。")
            return _finish(res, run)
        stage = "draw"
        failed = []
        for i, mv in enumerate(jobs):
            try:
                run.run(mv)
            except MOTION_ERRORS as e:
                # the pen comes off the paper before anything else; a lift that fails ends the drawing
                failed.append({"piece": i + 1, "error": type(e).__name__, "reason": str(e)[:240]})
                TL.lift(run.arm)
                if len(failed) > opts.max_failed:
                    raise Stop("draw", f"{len(failed)} pieces failed (the last: {type(e).__name__}: {e})",
                               "the pen was lifted each time; look at the photo and at the reasons before drawing on",
                               failed=failed) from e
                # the next piece is planned from where the arm really is
                jobs[i + 1:i + 2] = [run.piece(run.tip(), plan["pieces"][i + 1]["parts"],
                                               plan["pieces"][i + 1]["parts"][0][0][2] + opts.press_m + opts.lift_m,
                                               opts.v_mm_s, record=True)] if i + 1 < len(jobs) else []
        run.tm.mark("draw")
        res["drawn"] = {"pieces": len(jobs) - len(failed), "failed": failed, "pen_down_mm": round(run.pen_down_m * 1000.0, 0)}
        if run.track:
            res["tracking"] = {"path_err_mm": {"mean": round(float(np.mean([t["path_err_mm"]["mean"] for t in run.track])), 2),
                                               "max": round(max(t["path_err_mm"]["max"] for t in run.track), 2)},
                               "z_err_mm": {"min": round(min(t["z_err_mm"]["min"] for t in run.track), 2),
                                            "max": round(max(t["z_err_mm"]["max"] for t in run.track), 2)},
                               "parts": len(run.track)}
        stage = "verify"
        after, apath = run.look("draw_after")
        cov = INK.stroke_coverage(after, run.cam, run.tz, [pc["ink"] for pc in plan["pieces"]], before=img)
        res["coverage"] = {"drawn_fraction": cov["drawn_fraction"], "offset_mm": cov["offset_mm"], "judgeable": cov["judgeable"],
                           "unknown_fraction": cov["unknown_fraction"], "ink_mm": round(cov["ink_m"] * 1000.0, 0),
                           "missing_mm": round(cov["missing_m"] * 1000.0, 0), "gaps": cov["gaps"][:40], "gaps_n": len(cov["gaps"]),
                           "min_coverage": opts.min_coverage}
        if not cov["judgeable"]:
            raise Stop("visual", "the final photo cannot show whether the drawing is whole "
                       f"({(cov.get('can') or {}).get('reason', 'too much of it cannot be judged')})",
                       "look at the photo yourself; nothing more was drawn", photo=apath)
        good = cov["drawn_fraction"] >= opts.min_coverage and not failed
        res.update(ok=bool(good), stage="done" if good else "verify")
        if good:
            res["say"] = (f"「{job.get('name')}」を描きました（{st['count']} 本、線の {cov['drawn_fraction'] * 100:.0f} % が写真で確認でき、"
                          f"ずれは平均 {cov['offset_mm']['mean']} mm）。")
        else:
            res["reason"] = (f"{len(failed)} piece(s) failed" if failed else
                             f"only {cov['drawn_fraction'] * 100:.0f} % of the line is on the paper (at least "
                             f"{opts.min_coverage * 100:.0f} % is asked); {len(cov['gaps'])} gap(s)")
            res["next"] = ("look at the final photo and at \"gaps\"; where the pen did not reach the paper, repeat "
                           "`fd tool calibrate --stage contact` (or draw with a larger `--press`, at most 4 mm)")
            res["say"] = f"「{job.get('name')}」を描きましたが、線が足りません（{res['reason']}）。"
    except Stop as e:
        res.update(ok=False, stage=e.stage, reason=e.reason, next=e.next, **{k: v for k, v in e.detail.items()})
        res["say"] = f"描画を止めました（{e.stage}: {e.reason}）。"
    except MOTION_ERRORS as e:
        jaw = _jaw_refused(run, e)
        res.update(ok=False, stage=stage, reason=f"{type(e).__name__}: {e}",
                   next=jaw or "look at the reason and at `fd state`; the pen was lifted when that was possible (see \"pen\")")
        res["say"] = JAW_REFUSED_JA.format(what="描画") if jaw else f"描画の途中で止まりました（{stage}: {type(e).__name__}）。"
    except (KeyboardInterrupt, SystemExit) as e:  # see calibrate
        res.update(ok=False, stage=stage, reason=f"interrupted ({type(e).__name__}) during the stage {stage}", interrupted=True,
                   next=INTERRUPTED_NEXT)
        res["say"] = f"描画が途中で中断されました（{stage}）。"
    except Exception as e:  # a bug here: report it, lift, do not go on
        import traceback
        res.update(ok=False, stage=stage, reason=f"internal error: {type(e).__name__}: {e}",
                   trace=traceback.format_exc().strip().splitlines()[-6:])
        res["say"] = f"描画の途中で内部エラーが起きました（{stage}）。"
    return _finish(res, run, opts.end if res.get("ok") else "aside")


def options_summary(opts: PenOptions) -> dict:
    """The options as plain data (for the run log)."""
    return asdict(opts)
