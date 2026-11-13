"""Grasp planning for taking an object OUT of a container: pure geometry, no motion, no bus.

    plan_grasps(target, container, scene, cfg=None, ...)   -> list[TakeGrasp]   (best first; may be all not-ok)
    explain(cands)                                          -> {"reason": str, "next": str}   (when none is ok)
    choose_place(scene, target, container, cfg=None, ...)  -> (list[dict], reason)   (free table spots, best first)
    hollow_model(target, params=None, hollow=None, assume=False) -> dict   (how the object's cavity is taken to be)

Why this exists (2026-11-11): the arm's own safety check knows the fingertip POINT only. Inside a box that is
not enough: the back of the fixed finger gets thicker with height (5.2 mm at its end, 8.8 mm at 10 mm, 12.5 mm
at 20 mm), both fingers are 12.7 mm wide, and a finger lowered into a slot that is too narrow wedges against
the wall (it turned the whole box by 35 degrees). So the room for the fingers is planned here, from the scene.

Two ways to hold an object that lies in a container:
  "two_sided"  the normal grasp across the object: fixed jaw on one side, moving jaw on the other. Needs free
               space on both sides along one line.
  "pinch"      for hollow objects that are open at the top (a bottle cap, a cup): the fixed jaw goes down
               OUTSIDE the wall, the moving jaw INSIDE the cavity, and the wall is held between the pads. Needs
               free space on one side only. It is released only for an object that is SAID to be hollow and
               open at the top: by the scene (SceneObject.hollow), by the caller (hollow=True) or by the
               request (assume_hollow: it called the thing a cap, a cup, a lid). A cylinder nobody says
               anything about is modelled as hollow, but its wall pinch stays "unconfirmed" (not ok): nothing
               checks the cavity before the moving jaw goes down into it, and on a solid cylinder that jaw
               would land on the top and push.

For every jaw direction the planner measures the free distance from the object's outer surface to whatever comes
first (a container wall, a neighbour) inside a corridor as wide as the finger, and compares it with what the
finger needs THERE: its thickness at the height of that obstacle's top (gripper_model.finger_back_x /
moving_outer_x: the outline, not a point), the gap to the object and a clearance. Container walls and all other
objects are obstacles (vertical prisms up to their top). Conventions as in grasp.py: base frame, metres;
"jaw azimuth" = direction from the fixed jaw to the moving jaw; the gripper points straight down (pitch 90).

Heights come from geometry only: the surface an object in a container stands on is the scene's table height
plus the container's floor (SceneObject.z_bottom), a hollow object's inner bottom is its own bottom thickness
above that. Nothing is measured by touch (Arm.probe_down reads 3-18 mm deep, measured 2026-11-11), and no
table coordinate or height is built into this file: everything is relative to the scene and the container.

Tunables: DEFAULTS, overridden per key by cfg["take"] and by the `params` argument.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

import gripper_model as gm
import kinematics as K

__all__ = ["plan_grasps", "explain", "choose_place", "hollow_model", "fixed_jaw_visible", "TakeGrasp", "DEFAULTS"]

DEFAULTS = {
    # ---- wall pinch: fixed jaw outside, moving jaw inside (the recipe that worked on 2026-11-11)
    "pinch_standoff": 0.005,  # fixed pad face outside the wall's outer surface before closing (works from 1 to 10 mm)
    "pinch_inside_gap": 0.007,  # moving pad face inside the wall's inner surface before closing
    "pinch_inside_gap_min": 0.003,
    # fingertip height above the cavity's EXPECTED inner bottom while closing. The cap of 2026-11-11 was held
    # with the fingertip anywhere from 3 mm below (pressed) to 7 mm above its inner bottom: 2 mm above leaves
    # 5 mm for a bottom that is higher than expected and 5 mm for one that is lower
    "pinch_tip_above": 0.002,
    "pinch_min_depth": 0.008,  # a shallower cavity leaves the pads too little wall to hold
    "cavity_clear": 0.0015,  # moving finger to the cavity's inner surface
    "wall_default": 0.002,  # wall thickness assumed when the scene does not give one
    "bottom_default": 0.002,  # inner bottom above the surface the object stands on, when the scene does not say
    # ---- two-sided grasp across the object
    "standoff": 0.008,  # gap between the fixed pad and the object's face before closing
    "tip_height": 0.004,  # fingertip height above the surface the object stands on while closing
    "max_engage": 0.015,  # deepest the fingertip goes below the object's top
    "min_engage": 0.006,  # objects that allow less are refused
    "min_width": 0.012,
    "max_width": 0.045,
    "open_margin_deg": 12.0,  # pre-grasp opening = contact jaw angle + this (little room in a box)
    "self_clear": 0.004,  # the opened moving pad passes the object's far side by at least this
    "stall_offset_ticks": 40,  # the jaws stall this many ticks below rigid contact (gripper_model)
    # ---- both
    "min_standoff": 0.002,
    "min_clear": 0.002,  # hard minimum between a finger and anything else (also sideways)
    "tight_clear": 0.004,  # below this a candidate is flagged "tight" and ranked lower
    "comfort_clear": 0.006,  # the nominal standoff is kept only while this much stays free behind the finger
    "z_margin": 0.003,  # every obstacle is treated as this much taller than reported
    "azimuth_step_deg": 10.0,  # jaw directions tried around a cylinder
    "rim_margin": 0.012,  # fingertip above the rim (and anything taller nearby) at the hover point
    "approach_drop": 0.015,  # the approach comes straight down onto the hover point from this far above it
    "rim_clear": 0.025,  # object bottom above the rim before it is carried sideways
    "rim_clear_min": 0.010,  # ... the least that is accepted where the arm cannot reach higher
    "hover_radius": 0.09,
    "container_wall_default": 0.004,  # wall thickness assumed for a container without a measured inside
    # ---- where to put the object down
    "place_clear": 0.025,  # free space around the object at the place spot
    "place_radius": 0.20,  # spots are looked for within this distance of the container's centre
    "place_area": None,  # optional [x0, x1, y0, y1] of the table area that may be used (None: no such limit)
    "place_step": 0.015,
    "place_keepout_margin": 0.01,
}

FINGER_WIDTH_MM = 12.7  # both fingers (measured on the real gripper, 2026-11-11)
# Half width (mm) of each finger by distance from its own end: (up to this distance, half width). Never less
# than FINGER_WIDTH_MM / 2. From the mesh slices of wrist_roll_follower_so101_v1.stl / moving_jaw_so101_v1.stl
# (the tables in grasp.py), rounded up.
_FIXED_HW = ((12.0, 6.35), (21.0, 7.4), (36.0, 8.4), (46.0, 9.3), (56.0, 11.0), (66.0, 15.3), (1e9, 27.8))
_MOVING_HW = ((12.8, 6.4), (22.8, 7.5), (32.8, 8.1), (42.8, 8.9), (52.8, 10.1), (57.8, 11.5), (1e9, 24.2))
_HOUSING_MM = (-27.3, 40.6, 27.8)  # gripper body above the palm: from, to (along the jaw azimuth, from the pad), half width
_PALM_Z = gm.PALM_Z  # tool-frame z (mm) above which (more negative) the body begins
_TICKS_PER_DEG = 4096.0 / 360.0
_INF = float("inf")
_X_PAN = float(K._X_PAN)


# ======================================================================================= small helpers
def _wrap180(a_deg: float) -> float:
    return (float(a_deg) + 180.0) % 360.0 - 180.0


def _mm(v: float) -> str:
    return f"{float(v) * 1000.0:.1f} mm"


def _finite(*vals) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


def _unit(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def _lookup(table, d_mm: float) -> float:
    for upto, hw in table:
        if d_mm <= upto:
            return max(hw, FINGER_WIDTH_MM / 2.0)
    return table[-1][1]


def fixed_half_width(z_tool_mm: float) -> float:
    """Half width (m) of the fixed finger at tool height z (mm; gripper_frame_link, +z out of the fingertips)
    and everywhere below it (the finger only gets wider towards the palm)."""
    return _lookup(_FIXED_HW, gm.FIXED_TIP_Z - z_tool_mm) * 1e-3


def moving_half_width(q_deg: float, z_tool_mm: float) -> float:
    """Half width (m) of the moving jaw at tool height z (mm) and below it, at jaw angle q."""
    along = (gm.moving_tip_depth(q_deg) - z_tool_mm) / max(math.cos(math.radians(q_deg)), 0.5)
    return _lookup(_MOVING_HW, along) * 1e-3


def fixed_back(z_tool_mm: float) -> float:
    """Thickness (m) of the fixed finger behind its pad plane at tool height z (mm) and below it:
    gripper_model.finger_back_x (5.2 mm at the end of the finger, +0.36 mm per mm of height)."""
    return gm.finger_back_x(min(z_tool_mm, gm.FIXED_TIP_Z)) * 1e-3


def _rect(center, yaw: float, length: float, width: float) -> np.ndarray:
    """Corners (4, 2), counter-clockwise, of a rectangle whose `length` side points along yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    ax, ay = np.array([c, s]) * (length / 2.0), np.array([-s, c]) * (width / 2.0)
    ctr = np.array([float(center[0]), float(center[1])])
    return np.array([ctr - ax - ay, ctr + ax - ay, ctr + ax + ay, ctr - ax + ay])


def _hull(points) -> np.ndarray:
    """Convex hull (counter-clockwise, (N, 2)) of a point list (monotone chain)."""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) < 3:
        return np.array(pts, dtype=float)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0.0:
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]), dtype=float)


def _clip(poly: list, a: float, b: float, c: float) -> list:
    """The part of a polygon [(x, y), ...] where a*x + b*y <= c (Sutherland-Hodgman, one half plane)."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp, dq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if dp <= 0.0:
            out.append(p)
        if (dp < 0.0 < dq) or (dq < 0.0 < dp):
            f = dp / (dp - dq)
            out.append((p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])))
    return out


# ======================================================================================= obstacles
class _Obstacle:
    """Something the gripper must not touch: a vertical prism, footprint x everything below z_top.
    Footprint = a disc, a convex polygon, or - for the container the object lies in - everything OUTSIDE
    a convex polygon (its walls, seen from the inside)."""

    __slots__ = ("id", "label", "z_top", "disc", "poly", "cavity", "wall", "_edges")

    def __init__(self, obj_id, label: str, z_top: float, disc=None, poly=None, cavity=None, wall: bool = False):
        self.id, self.label, self.z_top, self.wall = obj_id, label, float(z_top), bool(wall)
        self.disc = None if disc is None else tuple(float(v) for v in disc)
        self.poly = None if poly is None else np.asarray(poly, dtype=float)
        self.cavity = None if cavity is None else np.asarray(cavity, dtype=float)
        self._edges = None
        ring = self.poly if self.poly is not None else self.cavity
        if ring is not None:  # outward unit normals and offsets of the edge lines (counter-clockwise ring)
            nxt = np.roll(ring, -1, axis=0)
            d = nxt - ring
            ln = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)
            n = np.column_stack([d[:, 1], -d[:, 0]]) / ln[:, None]
            self._edges = (ring, d, ln, n, (n * ring).sum(axis=1))

    # ---- sweep: where does a bar moving along s first touch the material?
    def first_contact(self, c, s, half_width: float, t_min: float = 0.0) -> float:
        """A bar of the given half width, at right angles to the unit vector s, travels from c + t_min*s
        along s. Returns the t at which it first touches the footprint (t_min if it starts in it),
        inf if it never does."""
        c = np.asarray(c, dtype=float)
        sx, sy = float(s[0]), float(s[1])
        if self.disc is not None:
            ox, oy, r = self.disc
            to = (ox - c[0]) * sx + (oy - c[1]) * sy
            wo = -(ox - c[0]) * sy + (oy - c[1]) * sx
            dw = max(0.0, abs(wo) - half_width)
            if dw >= r:
                return _INF
            half = math.sqrt(r * r - dw * dw)
            if to + half < t_min:
                return _INF
            return max(to - half, t_min)
        ring = self.poly if self.poly is not None else self.cavity
        rel = ring - c
        loc = [(float(x * sx + y * sy), float(-x * sy + y * sx)) for x, y in rel]  # (t, w) of every corner
        if self.poly is not None:
            part = _clip(_clip(_clip(loc, 0.0, 1.0, half_width), 0.0, -1.0, half_width), -1.0, 0.0, -t_min)
            return min(p[0] for p in part) if part else _INF
        # walls around a cavity: the bar is free while both its ends are inside the (convex) cavity
        n = len(loc)
        best = _INF
        for w in (-half_width, half_width):
            for i in range(n):
                (t0, w0), (t1, w1) = loc[i], loc[(i + 1) % n]
                et, ew = t1 - t0, w1 - w0
                ln = math.hypot(et, ew)
                if ln < 1e-12:
                    continue
                nt, nw = ew / ln, -et / ln  # outward normal of a counter-clockwise ring
                dist = nt * (t_min - t0) + nw * (w - w0)  # signed distance of the bar end from this edge line
                if dist > 0.0:
                    return t_min  # already in (or beyond) the wall
                if nt > 1e-12:
                    best = min(best, t_min - dist / nt)
        return best

    # ---- signed distance (negative inside the material)
    def sdist(self, xy) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        if self.disc is not None:
            return np.hypot(xy[:, 0] - self.disc[0], xy[:, 1] - self.disc[1]) - self.disc[2]
        ring, d, ln, n, off = self._edges
        line = xy @ n.T - off[None, :]  # (N, E): signed distance to every edge line (positive outside)
        inside = (line <= 0.0).all(axis=1)
        ap = xy[:, None, :] - ring[None, :, :]
        t = np.clip((ap * d[None, :, :]).sum(axis=2) / (ln * ln)[None, :], 0.0, 1.0)
        seg = np.hypot(*(ap - t[:, :, None] * d[None, :, :]).transpose(2, 0, 1)).min(axis=1)
        dist = np.where(inside, -seg, seg)
        return dist if self.poly is not None else -dist


def _inner_of(container, p: dict):
    """(corners (4, 2), rim_z) of the cavity of a container: its measured inside when there is one, else the
    outline shrunk by an assumed wall."""
    inner = getattr(container, "inner", None)
    if isinstance(inner, dict):
        try:
            c, size = inner["center"], inner["size"]
            vals = (float(c[0]), float(c[1]), float(inner["yaw"]), float(size[0]), float(size[1]), float(inner["rim_z"]))
            if _finite(*vals) and min(vals[3], vals[4]) > 0.0:
                return _rect(vals[:2], vals[2], vals[3], vals[4]), vals[5]
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    w = float(p["container_wall_default"])
    length, width = float(container.size[0]) - 2.0 * w, float(container.size[1]) - 2.0 * w
    return _rect(container.center, float(container.yaw), max(length, 0.01), max(width, 0.01)), float(container.z_top)


def _obstacles(scene, target, container, p: dict) -> list:
    out = []
    for o in (scene.objects if scene is not None else []):
        if o is target or o.id == target.id:
            continue
        if container is not None and o.id == container.id:
            ring, rim_z = _inner_of(o, p)
            out.append(_Obstacle(o.id, "the container wall", rim_z, cavity=ring, wall=True))
            continue
        label = f"{getattr(o, 'color', '?')} {getattr(o, 'kind', '?')} #{o.id}"
        z_top = float(o.z_top)
        if getattr(o, "kind", None) == "container" and isinstance(getattr(o, "inner", None), dict):
            z_top = max(z_top, float(o.inner.get("rim_z", z_top)))
        size = getattr(o, "size", (0.0, 0.0, 0.0))
        if getattr(o, "kind", None) == "cylinder" and _finite(o.center[0], o.center[1], size[0]) and size[0] > 0:
            out.append(_Obstacle(o.id, label, z_top, disc=(o.center[0], o.center[1], 0.5 * max(size[0], size[1]))))
            continue
        fp = getattr(o, "footprint", None)
        poly = None
        if fp is not None and len(fp) >= 3:
            try:
                poly = _hull(fp)
            except (TypeError, ValueError, IndexError):
                poly = None
        if poly is None or len(poly) < 3 or not np.all(np.isfinite(poly)):
            if _finite(o.center[0], o.center[1], getattr(o, "yaw", 0.0), size[0], size[1]) and min(size[0], size[1]) > 0:
                poly = _rect(o.center, float(o.yaw), float(size[0]), float(size[1]))
            else:  # nothing usable: assume a generous disc rather than ignore the object
                cx, cy = (float(o.center[0]), float(o.center[1])) if _finite(*o.center[:2]) else (0.0, 0.0)
                out.append(_Obstacle(o.id, label + " (no usable outline: taken as an 80 mm disc)", z_top,
                                     disc=(cx, cy, 0.04)))
                continue
        out.append(_Obstacle(o.id, label, z_top, poly=poly))
    return out


# ======================================================================================= the object
class _Geom:
    """The object to take: outline, heights, and (for hollow objects) its cavity."""

    def __init__(self, obj):
        self.kind = obj.kind
        self.c = np.array([float(obj.center[0]), float(obj.center[1])])
        self.length, self.width, self.height = (float(v) for v in obj.size[:3])
        self.yaw = float(obj.yaw)
        self.z_bottom = float(obj.z_bottom)
        self.z_top = self.z_bottom + self.height
        self.round = self.kind == "cylinder"
        self.r = 0.5 * max(self.length, self.width)
        self.ax = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        self.ay = np.array([-self.ax[1], self.ax[0]])

    def r_out(self, s) -> float:
        """Distance from the centre to the outer surface in direction s."""
        if self.round:
            return self.r
        return abs(float(s @ self.ax)) * self.length / 2.0 + abs(float(s @ self.ay)) * self.width / 2.0

    def across(self, u) -> float:
        """Size of the object across jaws that open along u."""
        return 2.0 * self.r_out(u)

    def directions(self, step_deg: float) -> list:
        """(azimuth_deg, description) of the jaw directions worth trying."""
        if self.round:
            n = max(8, int(round(360.0 / max(step_deg, 1.0))))
            return [(_wrap180(k * 360.0 / n), "") for k in range(n)]
        yaw = math.degrees(self.yaw)
        what = "" if self.kind == "box" else "shape unknown, planned on its bounding box; "
        return [(_wrap180(yaw + 90.0), what + "across the width"), (_wrap180(yaw - 90.0), what + "across the width"),
                (_wrap180(yaw), what + "across the length"), (_wrap180(yaw + 180.0), what + "across the length")]

    def outline(self, n: int = 36) -> np.ndarray:
        if self.round:
            a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
            return self.c + self.r * np.column_stack([np.cos(a), np.sin(a)])
        ring = _rect(self.c, self.yaw, self.length, self.width)
        out = []
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            k = max(2, int(math.ceil(float(np.hypot(*(b - a))) / 0.004)))
            out.append(a[None, :] + np.linspace(0.0, 1.0, k, endpoint=False)[:, None] * (b - a)[None, :])
        return np.vstack(out)


def _target_problem(target) -> str:
    kind = getattr(target, "kind", None)
    if kind == "container":
        return "the target is a container"
    if kind not in ("box", "cylinder", "unknown"):
        return f"cannot plan a grasp for an object of kind {kind!r}"
    try:
        vals = [target.center[0], target.center[1], target.size[0], target.size[1], target.size[2], target.yaw,
                target.z_bottom]
    except (TypeError, IndexError, AttributeError):
        return "the object has no usable centre / size"
    if not _finite(*vals):
        return "the object's centre / size / yaw / z_bottom is not a finite number"
    if min(float(v) for v in target.size[:3]) <= 0.0:
        return f"the object's size {tuple(round(float(v), 4) for v in target.size[:3])} is not positive"
    return ""


def hollow_model(target, params: dict | None = None, hollow=None, assume: bool = False) -> dict:
    """How the object's cavity is taken to be: {"hollow", "source", "wall", "inner_radius", "z_inner_bottom",
    "depth", "notes"}.

    hollow: an explicit statement (True / False); it wins. Otherwise what the scene says (the optional
    SceneObject fields hollow, inner_radius, wall, z_inner_bottom). When neither says: with assume=True the
    object is taken as hollow and open at the top (the request called it a cap, a cup, a lid); else a cylinder
    is modelled as hollow and anything else as solid. Missing numbers: wall 2 mm, inner bottom 2 mm above the
    surface the object stands on. "source" says where the flag came from: "option", "scene", "request" or
    "default". A flag from "default" is a guess that nothing verifies: plan_grasps plans the wall pinch
    with it but does not release it (see TakeGrasp.unconfirmed)."""
    p = {**DEFAULTS, **(params or {})}
    notes = []
    flag = hollow if hollow is not None else getattr(target, "hollow", None)
    source = "option" if hollow is not None else ("scene" if flag is not None else "default")
    if flag is None and assume:
        flag, source = True, "request"
    if flag is None:
        flag = getattr(target, "kind", None) == "cylinder"
    r_out = 0.5 * max(float(target.size[0]), float(target.size[1])) if getattr(target, "kind", None) == "cylinder" \
        else 0.5 * float(target.size[1])
    z_bottom, z_top = float(target.z_bottom), float(target.z_bottom) + float(target.size[2])

    def given(name):
        v = getattr(target, name, None)
        return float(v) if v is not None and _finite(v) else None

    wall, r_in, z_ib = given("wall"), given("inner_radius"), given("z_inner_bottom")
    if wall is None and r_in is not None and 0.0 < r_in < r_out:
        wall = r_out - r_in
    if wall is None or not (0.0003 <= wall < 0.6 * r_out):
        if wall is not None:
            notes.append(f"wall thickness {_mm(wall)} from the scene is not plausible: {_mm(p['wall_default'])} assumed")
        wall = min(float(p["wall_default"]), 0.5 * r_out)
    if r_in is None or not (0.0 < r_in < r_out):
        r_in = r_out - wall
    if z_ib is None or not (z_bottom <= z_ib < z_top):
        if z_ib is not None:
            notes.append("inner bottom height from the scene is not plausible: default used")
        z_ib = min(z_bottom + float(p["bottom_default"]), z_top)
    return {"hollow": bool(flag), "source": source, "wall": wall, "inner_radius": r_in, "z_inner_bottom": z_ib,
            "depth": z_top - z_ib, "notes": notes}


# ======================================================================================= result type
@dataclass
class TakeGrasp:
    ok: bool
    reason: str  # why it was rejected ("" when ok)
    mode: str  # "two_sided" | "pinch"
    jaw_azimuth_deg: float  # base-frame direction fixed jaw -> moving jaw
    roll_deg: float  # wrist roll that realises it (closed form; the arm's solver has the last word)
    pitch_deg: float  # always 90: straight down
    tip_grasp: tuple  # fingertip (Arm.tip_tool point) while closing, from the scene's heights
    tip_hover: tuple  # fingertip above the rim, where the jaws are looked at before descending
    width: float  # what ends up between the pads: the object (two_sided) or its wall (pinch)
    standoff: float  # gap between the fixed pad and the object's outer face before closing
    open_raw: int  # gripper goal before the descent
    expected_raw: int  # gripper position expected when the jaws hold the object
    release_raw: int  # gripper goal that lets it go
    clearance_fixed: float  # free space behind the fixed finger (inf: nothing near)
    clearance_moving: float  # free space around the moving jaw
    score: float
    object_offset: tuple = (0.0, 0.0)  # object centre minus fingertip (x, y) while held
    h_below: float = 0.0  # how far the object's bottom hangs below the fingertip while held
    z_surface: float = 0.0  # the surface the fingertip height is measured from (cavity bottom / floor)
    tip_above_surface: float = 0.0  # planned fingertip height above it
    contact_offset: float = 0.0  # fingertip height above that surface when the lowest jaw point touches it
    margin: float = 0.0  # lowest jaw point above that surface at the grasp height
    look_closed: bool = False  # look at the jaws with the gripper closed (the moving jaw would hide the fixed one)
    # a wall pinch that has room, but rests on the GUESS that the object is hollow and open at the top (a
    # cylinder the scene says nothing about): not ok, because nothing checks the cavity before the moving jaw
    # goes down into it (on a solid object it would land on the top and push)
    unconfirmed: bool = False
    slot: dict = field(default_factory=dict)  # free distance on the fixed-jaw side and what bounds it
    blockers: list = field(default_factory=list)  # what makes it infeasible: [{"id", "label", "wall", "side", ...}]
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = None if not math.isfinite(v) else round(v, 5)
            elif isinstance(v, tuple):
                d[k] = [round(float(x), 5) for x in v]
        return d


class _Ctx:
    """What is the same for every candidate of one planning call."""

    def __init__(self, cfg: dict | None, p: dict, tip_offset):
        cfg = cfg or {}
        self.p = p
        off = tip_offset if tip_offset is not None else cfg.get("tip_offset", (0.0, 0.0, 0.0))
        self.off = (float(off[0]), float(off[1]), float(off[2]))  # fingertip in tool axes (m)
        self.tip_z_mm = self.off[2] * 1e3
        self.touch_raw = float(cfg.get("gripper_touch_raw", gm.TOUCH_RAW))
        g = cfg.get("gripper") or {}
        margin = float(g.get("margin", 25))
        self.raw_lo = max(float(g.get("closed_raw", 1998)) + margin, self.touch_raw)
        self.raw_hi = float(g.get("open_raw", 3648)) - margin
        soft = (cfg.get("soft_limits_deg") or {}).get("wrist_roll") or (-160.0, 160.0)
        self.roll_lim = (float(soft[0]), float(soft[1]))

    def raw_for_q(self, q_deg: float) -> int:
        raw = self.touch_raw + (q_deg - gm.Q_CLOSED) * _TICKS_PER_DEG
        return int(round(max(self.raw_lo, min(self.raw_hi, raw))))

    def q_for_raw(self, raw: float) -> float:
        return gm.Q_CLOSED + (float(raw) - self.touch_raw) / _TICKS_PER_DEG

    def tip_xy(self, pad_xy: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Fingertip xy when the fixed pad's origin is at pad_xy and the jaws open towards u (vertical
        gripper: tool x axis = -u, tool y axis = u turned by +90 deg)."""
        return pad_xy - self.off[0] * u + self.off[1] * np.array([-u[1], u[0]])

    def tool_z(self, z: float, z_tip: float) -> float:
        """Tool-frame z (mm) of the base height z when the fingertip is at height z_tip."""
        return self.tip_z_mm - (z - z_tip) * 1e3

    def roll_for(self, psi_deg: float, tip_xy) -> float:
        pan = -math.degrees(math.atan2(float(tip_xy[1]), float(tip_xy[0]) - _X_PAN))
        return float(K.roll_for_azimuth(psi_deg, pan, 90.0))


def fixed_jaw_visible(azimuth_deg: float, tip_xy, camera_xy, gap: float) -> bool:
    """Can the camera see the end of the fixed jaw past the moving jaw (jaws `gap` apart, gripper vertical)?
    Seen from the camera the two fingers stand gap * |sin| apart sideways; the moving one hides the fixed one
    when it is the nearer of the two and that distance is less than about 60 % of a finger width."""
    if camera_xy is None:
        return True
    to_cam = np.asarray(camera_xy, dtype=float)[:2] - np.asarray(tip_xy, dtype=float)[:2]
    n = float(np.hypot(*to_cam))
    if n < 1e-6:
        return True
    u = _unit(azimuth_deg)
    along = float(u @ to_cam) / n  # > 0: the moving jaw is on the camera's side
    side = abs(float(u[0] * to_cam[1] - u[1] * to_cam[0])) / n
    return along <= 0.0 or gap * side >= 0.6 * FINGER_WIDTH_MM * 1e-3


# ======================================================================================= one side of the gripper
def _fixed_side(ctx: _Ctx, geom: _Geom, obstacles, u: np.ndarray, z_tip: float) -> dict:
    """Room behind the object for the fixed finger (it comes down on the object's -u side).
    For every obstacle: the free distance from the object's outer surface to it inside the finger's
    corridor, minus the finger's thickness at the height of that obstacle's top = what is left for the gap
    to the object plus the clearance behind the finger. Returns the smallest such "room" and its owner."""
    p = ctx.p
    s = -u
    r_surf = geom.r_out(s)
    best = {"room": _INF, "free": _INF, "need_finger": 0.0, "who": None}
    rows = []
    for ob in obstacles:
        zt = ctx.tool_z(ob.z_top + float(p["z_margin"]), z_tip)
        if zt >= gm.FIXED_TIP_Z:
            continue  # its top is below the end of the finger
        if zt < _PALM_Z:  # tall enough to reach the gripper body: that is checked as a box (_body_hits)
            zt = _PALM_Z
        back, hw = fixed_back(zt), fixed_half_width(zt)
        t = ob.first_contact(geom.c, s, hw + float(p["min_clear"]))
        if not math.isfinite(t):
            continue
        free = t - r_surf
        room = free - back
        rows.append({"ob": ob, "free": free, "back": back, "room": room})
        if room < best["room"]:
            best = {"room": room, "free": free, "need_finger": back, "who": ob}
    best["rows"] = rows
    return best


def _moving_side(ctx: _Ctx, geom: _Geom, obstacles, u: np.ndarray, z_tip: float, q_open: float, width: float) -> dict:
    """Room beyond the object for the opened moving jaw (two-sided grasp). For every obstacle: how far the
    jaw's outer face, at the height of that obstacle's top, reaches beyond the object's far surface when
    the fixed pad is `standoff` from the near one is (reach - standoff); the obstacle allows a standoff of
    at least reach - free + clearance. Returns the largest such lower bound "m" (without the clearance)."""
    p = ctx.p
    r_surf = geom.r_out(u)
    best = {"m": -_INF, "free": _INF, "reach": 0.0, "who": None}
    rows = []
    for ob in obstacles:
        zt = ctx.tool_z(ob.z_top + float(p["z_margin"]), z_tip)
        zt = max(zt, _PALM_Z)
        outer = gm.moving_outer_x(q_open, zt)
        if outer is None:
            continue  # the jaw has no part that low
        hw = moving_half_width(q_open, zt)
        t = ob.first_contact(geom.c, u, hw + float(p["min_clear"]))
        if not math.isfinite(t):
            continue
        free = t - r_surf
        reach = -outer * 1e-3 - width  # beyond the far surface when the standoff is zero
        m = reach - free
        rows.append({"ob": ob, "free": free, "reach": reach, "m": m})
        if m > best["m"]:
            best = {"m": m, "free": free, "reach": reach, "who": ob}
    best["rows"] = rows
    return best


def _body_hits(ctx: _Ctx, pad_xy: np.ndarray, u: np.ndarray, obstacles, z_tip: float):
    """An obstacle tall enough to reach the gripper body (above the palm) under the gripper: its label."""
    a0, a1, hw = (v * 1e-3 for v in _HOUSING_MM)
    for ob in obstacles:
        if ctx.tool_z(ob.z_top + float(ctx.p["z_margin"]), z_tip) >= _PALM_Z:
            continue
        t = ob.first_contact(pad_xy, u, hw + float(ctx.p["min_clear"]), t_min=a0)
        if t <= a1 + float(ctx.p["min_clear"]):
            return ob
    return None


def _push_travel(geom: _Geom, obstacles, u: np.ndarray, s_max: float, z_floor: float):
    """Closing pushes the object along -u until it rests on the fixed pad. Returns (travel, obstacle): how
    far (1 mm steps, at most s_max) its outline can move before it comes within 0.5 mm of another object
    (one it already touches and merely slides along does not count), and what stops it."""
    limit, who = s_max, None
    steps = np.arange(0.001, s_max + 1e-9, 0.001)
    if len(steps) == 0:
        return limit, who
    edge = geom.outline()
    lead = edge[((edge - geom.c) @ u) <= 1e-9]  # the half that leads the motion
    for ob in obstacles:
        if ob.z_top <= z_floor + 0.002:
            continue
        d0 = ob.sdist(lead)
        if float(d0.min()) > s_max + 0.001:
            continue
        for k, t in enumerate(steps):
            d1 = ob.sdist(lead - t * u)
            if bool(((d1 < 0.0005) & (d1 < d0 - 0.001)).any()):
                travel = float(t) - 0.001
                if travel < limit:
                    limit, who = travel, ob
                break
    return limit, who


def _blocker(ob: _Obstacle, side: str, free: float, need: float) -> dict:
    return {"id": ob.id, "label": ob.label, "wall": ob.wall, "side": side, "free_mm": round(free * 1e3, 1),
            "need_mm": round(need * 1e3, 1)}


# ======================================================================================= candidates
def _new(ctx: _Ctx, mode: str, psi: float, tip, z_tip: float, z_hover: float, width: float, standoff: float,
         label: str) -> dict:
    return dict(ok=False, reason="", mode=mode, jaw_azimuth_deg=round(psi, 3), roll_deg=round(ctx.roll_for(psi, tip), 3),
                pitch_deg=90.0, tip_grasp=(tip[0], tip[1], z_tip), tip_hover=(tip[0], tip[1], z_hover), width=width,
                standoff=standoff, open_raw=0, expected_raw=0, release_raw=0, clearance_fixed=_INF,
                clearance_moving=_INF, score=0.0, notes=[label] if label else [])


def _done(f: dict, reason: str = "") -> TakeGrasp:
    f["ok"], f["reason"] = (reason == ""), reason
    for key in ("tip_grasp", "tip_hover", "object_offset"):
        if key in f:
            f[key] = tuple(round(float(v), 5) for v in f[key])
    if not f["ok"]:
        f["score"] = 0.0
    return TakeGrasp(**f)


def _roll_problem(ctx: _Ctx, roll: float) -> str:
    lo, hi = ctx.roll_lim
    if not (lo + 1.0 <= roll <= hi - 1.0):
        return f"wrist roll {roll:.0f} deg needed for this jaw direction is outside the limits [{lo:.0f}, {hi:.0f}]"
    return ""


def _score_clear(ctx: _Ctx, f: dict, sides) -> float:
    """Penalty for little room; adds the "tight" notes."""
    p = ctx.p
    pen = 0.0
    for name, c, who in sides:
        if c < float(p["tight_clear"]):
            pen += 10.0 + 4000.0 * (float(p["tight_clear"]) - c)
            f["notes"].append(f"tight: {_mm(c)} between the {name} and {who}")
        elif c < 0.010:
            pen += 1000.0 * (0.010 - c)
    return pen


def _pinch(ctx: _Ctx, geom: _Geom, hm: dict, obstacles, psi: float, z_hover: float, label: str) -> TakeGrasp:
    """Wall pinch with the moving jaw towards psi: the fixed finger comes down outside the wall on the
    object's -psi side, the moving jaw inside the cavity."""
    p = ctx.p
    u = _unit(psi)
    wall = float(hm["wall"])
    z_tip = float(hm["z_inner_bottom"]) + float(p["pinch_tip_above"])
    r_out = geom.r_out(-u)
    s0 = float(p["pinch_standoff"])

    def tip_for(standoff):
        return ctx.tip_xy(geom.c - (r_out + standoff) * u, u)

    f = _new(ctx, "pinch", psi, tip_for(s0), z_tip, z_hover, wall, s0, label)
    held = (r_out + ctx.off[0]) * u - ctx.off[1] * np.array([-u[1], u[0]])
    f.update(object_offset=(held[0], held[1]), z_surface=float(hm["z_inner_bottom"]),
             tip_above_surface=float(p["pinch_tip_above"]), h_below=z_tip - geom.z_bottom, blockers=[])
    if hm["depth"] < float(p["pinch_min_depth"]):
        return _done(f, f"the cavity is only {_mm(hm['depth'])} deep: too little wall for the pads "
                        f"(minimum {_mm(p['pinch_min_depth'])})")
    # ---- outside: room for the fixed finger
    fx = _fixed_side(ctx, geom, obstacles, u, z_tip)
    room = fx["room"]
    min_room = float(p["min_standoff"]) + float(p["min_clear"])
    f["slot"] = {"free_mm": None if not math.isfinite(fx["free"]) else round(fx["free"] * 1e3, 1),
                 "finger_mm": round(fx["need_finger"] * 1e3, 1),
                 "bounded_by": fx["who"].label if fx["who"] is not None else None}
    if room < min_room:
        f["blockers"] = [_blocker(r["ob"], "fixed", r["free"], r["back"] + min_room) for r in fx["rows"]
                         if r["room"] < min_room]
        who = fx["who"]
        return _done(f, f"no room for the outside finger: {_mm(max(fx['free'], 0.0))} free up to {who.label}, "
                        f"{_mm(fx['need_finger'] + min_room)} needed (the finger is {_mm(fx['need_finger'])} thick "
                        f"at the height of its top)")
    standoff = s0 if room >= s0 + float(p["comfort_clear"]) else max(float(p["min_standoff"]), min(s0, 0.5 * room))
    # the wall is pushed onto the fixed pad when the jaws close: that way must be free too
    travel, stop = _push_travel(geom, obstacles, u, standoff, geom.z_bottom)
    if travel < float(p["min_standoff"]) - 1e-9:
        f["blockers"] = [_blocker(stop, "push", travel, float(p["min_standoff"]))]
        return _done(f, f"closing would push the object into {stop.label} ({_mm(travel)} of free travel)")
    if travel < standoff:
        standoff = travel
        f["notes"].append(f"standoff limited to {_mm(standoff)}: {stop.label} is in the way of the push")
    tip = tip_for(standoff)
    pad = geom.c - (r_out + standoff) * u
    f.update(standoff=standoff, tip_grasp=(tip[0], tip[1], z_tip), tip_hover=(tip[0], tip[1], z_hover),
             roll_deg=round(ctx.roll_for(psi, tip), 3), clearance_fixed=room - standoff)
    problem = _roll_problem(ctx, f["roll_deg"])
    if problem:
        return _done(f, problem)
    # ---- inside: the moving finger must fit into the cavity at the opening it comes down with
    zt_top = ctx.tool_z(geom.z_top, z_tip)  # the cavity's rim in tool coordinates
    fit, thick = None, 0.0
    gap = float(p["pinch_inside_gap"])
    while gap >= float(p["pinch_inside_gap_min"]) - 1e-9:
        opening = standoff + wall + gap
        q_open = gm.q_for_width(opening * 1e3)
        outer = gm.moving_outer_x(q_open, zt_top)
        if outer is not None:
            a_in, a_out = opening, -outer * 1e-3  # along u, from the fixed pad plane
            thick = a_out - a_in
            hw = moving_half_width(q_open, zt_top)
            clear = _cavity_clear(geom, hm, r_out, standoff, a_in, a_out, hw, u)
            if clear >= float(p["cavity_clear"]):
                fit = (gap, opening, q_open, clear)
                break
        gap -= 0.001
    if fit is None:
        return _done(f, f"the cavity (about {_mm(2.0 * hm['inner_radius'])} across inside) is too small for the moving "
                        f"finger, which is {FINGER_WIDTH_MM:.1f} mm wide and {_mm(thick)} thick at the height of the rim")
    gap, opening, q_open, clear = fit
    nose_below = (gm.moving_tip_depth(q_open) - ctx.tip_z_mm) * 1e-3  # moving nose below the fingertip point
    margin = float(p["pinch_tip_above"]) - nose_below
    if margin < 0.0005:
        return _done(f, "the moving jaw would touch the cavity's bottom at the planned height")
    # anything taller than the object next to it: the moving jaw (above the cavity) and the gripper body
    for ob in obstacles:
        zt = max(ctx.tool_z(ob.z_top + float(p["z_margin"]), z_tip), _PALM_Z)
        if zt >= zt_top:
            continue  # not taller than the object: the moving jaw is inside the cavity down there
        outer = gm.moving_outer_x(q_open, zt)
        if outer is None:
            continue
        t = ob.first_contact(pad, u, moving_half_width(q_open, zt) + float(p["min_clear"]), t_min=opening - 0.002)
        if t <= -outer * 1e-3 + float(p["min_clear"]):
            f["blockers"] = [_blocker(ob, "moving", t, -outer * 1e-3)]
            return _done(f, f"the moving jaw would hit {ob.label}, which is taller than the object")
    hit = _body_hits(ctx, pad, u, obstacles, z_tip)
    if hit is not None:
        f["blockers"] = [_blocker(hit, "body", 0.0, 0.0)]
        return _done(f, f"the gripper body would hit {hit.label} (it is taller than the fingers are long)")
    expected = ctx.raw_for_q(gm.q_for_width(max(wall, 0.0005) * 1e3))
    f.update(open_raw=ctx.raw_for_q(q_open), expected_raw=expected, release_raw=ctx.raw_for_q(q_open),
             clearance_moving=clear, contact_offset=nose_below, margin=margin)
    f["slot"]["standoff_mm"] = round(standoff * 1e3, 1)
    if gap < float(p["pinch_inside_gap"]) - 1e-9:
        f["notes"].append(f"opening reduced: moving pad {_mm(gap)} inside the wall (small cavity)")
    if standoff < s0 - 1e-9:
        f["notes"].append(f"standoff reduced to {_mm(standoff)} (little room behind the finger)")
    score = 90.0
    score -= _score_clear(ctx, f, (("fixed finger", f["clearance_fixed"], fx["who"].label if fx["who"] else "?"),))
    score -= 1500.0 * (s0 - standoff) + 500.0 * (float(p["pinch_inside_gap"]) - gap)
    score -= 0.05 * abs(_wrap180(f["roll_deg"] - gm.WRIST_ROLL_SKEW_DEG))
    f["score"] = round(max(1.0, score), 2)
    return _done(f)


def _cavity_clear(geom: _Geom, hm: dict, r_out: float, standoff: float, a_in: float, a_out: float, hw: float,
                  u: np.ndarray) -> float:
    """Smallest distance between the moving finger (a rectangle from a_in to a_out along u, measured from the
    fixed pad plane, and +-hw across) and the cavity's inner surface; negative when it does not fit."""
    x0, x1 = a_in - (r_out + standoff), a_out - (r_out + standoff)  # along u, from the object's centre
    if geom.round:
        return float(hm["inner_radius"]) - max(math.hypot(x, hw) for x in (x0, x1))
    wall = float(hm["wall"])
    along = geom.r_out(u) - wall  # half the cavity along u
    across = geom.r_out(np.array([-u[1], u[0]])) - wall
    return min(along - max(abs(x0), abs(x1)), across - hw)


def _two_sided(ctx: _Ctx, geom: _Geom, obstacles, psi: float, z_hover: float, label: str) -> TakeGrasp:
    """Grasp across the object: fixed finger on its -psi side, moving jaw (opened) on its +psi side."""
    p = ctx.p
    u = _unit(psi)
    width = geom.across(u)
    r_near = geom.r_out(-u)
    s0 = float(p["standoff"])
    tip_above = max(float(p["tip_height"]), geom.height - float(p["max_engage"]))
    q_c = gm.q_for_width(min(max(width * 1e3, 0.0), gm.W_MAX)) if _finite(width) else float("nan")
    q_open = q_c + float(p["open_margin_deg"])
    nose_below = (gm.lowest_point_depth(q_open) - ctx.tip_z_mm) * 1e-3 if math.isfinite(q_open) else 0.0
    if tip_above - nose_below < 0.0032 and geom.height - (0.0032 + nose_below) >= float(p["min_engage"]):
        tip_above = 0.0032 + nose_below  # keep the lowest jaw point 3 mm above the floor when the object allows it
    z_tip = geom.z_bottom + tip_above
    engage = geom.z_top - z_tip

    def tip_for(standoff):
        return ctx.tip_xy(geom.c - (r_near + standoff) * u, u)

    f = _new(ctx, "two_sided", psi, tip_for(s0), z_tip, z_hover, width, s0, label)
    held = (r_near + ctx.off[0]) * u - ctx.off[1] * np.array([-u[1], u[0]])
    f.update(object_offset=(held[0], held[1]), z_surface=geom.z_bottom, tip_above_surface=tip_above,
             h_below=tip_above, contact_offset=nose_below, margin=tip_above - nose_below, blockers=[])
    if width > float(p["max_width"]) + 1e-9:
        return _done(f, f"too wide for a grasp across it: {_mm(width)}, the jaws pinch at most {_mm(p['max_width'])}")
    if width < float(p["min_width"]) - 1e-9:
        return _done(f, f"too narrow: {_mm(width)} across the jaws (minimum {_mm(p['min_width'])})")
    if engage < float(p["min_engage"]) - 1e-9:
        return _done(f, f"too low: an object {_mm(geom.height)} high leaves the fingertips {_mm(engage)} of grip "
                        f"(minimum {_mm(p['min_engage'])})")
    opening = gm.width_for_q(q_open) * 1e-3  # pad to pad before closing
    s_hi = min(s0, opening - width - float(p["self_clear"]))
    min_standoff, min_clear = float(p["min_standoff"]), float(p["min_clear"])
    if s_hi < min_standoff:
        return _done(f, "the pre-grasp opening leaves no room around the object")
    fx = _fixed_side(ctx, geom, obstacles, u, z_tip)
    mv = _moving_side(ctx, geom, obstacles, u, z_tip, q_open, width)
    room, m = fx["room"], mv["m"]
    f["slot"] = {"free_mm": None if not math.isfinite(fx["free"]) else round(fx["free"] * 1e3, 1),
                 "finger_mm": round(fx["need_finger"] * 1e3, 1),
                 "bounded_by": fx["who"].label if fx["who"] is not None else None,
                 "free_moving_mm": None if not math.isfinite(mv["free"]) else round(mv["free"] * 1e3, 1),
                 "bounded_moving_by": mv["who"].label if mv["who"] is not None else None}
    blockers = [_blocker(r["ob"], "fixed", r["free"], r["back"] + min_standoff + min_clear) for r in fx["rows"]
                if r["room"] < min_standoff + min_clear]
    blockers += [_blocker(r["ob"], "moving", r["free"], r["reach"] - s_hi + min_clear) for r in mv["rows"]
                 if r["m"] + min_clear > s_hi]
    lo = max(min_standoff, m + min_clear)
    hi = min(s_hi, room - min_clear)
    if blockers or lo > hi + 1e-9:
        if not blockers:  # each side alone would do, both together do not
            blockers = [_blocker(fx["who"], "fixed", fx["free"], fx["need_finger"] + lo + min_clear),
                        _blocker(mv["who"], "moving", mv["free"], mv["reach"] - hi + min_clear)]
        f["blockers"] = blockers
        parts = []
        for b in blockers[:3]:
            side = "outside (fixed) finger" if b["side"] == "fixed" else "opened moving jaw"
            parts.append(f"{side}: {max(b['free_mm'], 0.0):.1f} mm free up to {b['label']}, {b['need_mm']:.1f} mm needed")
        return _done(f, "no room for both jaws: " + "; ".join(parts))
    # the gap to the object, the room behind the fixed finger and the room beyond the moving jaw are
    # balanced: standoff = the middle of what the two sides allow, never more than the nominal one
    want = 0.5 * (room + max(m, 0.0)) if math.isfinite(room) else s_hi
    standoff = max(lo, min(hi, want))
    travel, stop = _push_travel(geom, obstacles, u, standoff, geom.z_bottom)
    if travel < lo - 1e-9:
        f["blockers"] = [_blocker(stop, "push", travel, lo)]
        return _done(f, f"closing would push the object into {stop.label} ({_mm(travel)} of free travel)")
    if travel < standoff:
        standoff = travel
        f["notes"].append(f"standoff limited to {_mm(standoff)}: {stop.label} is in the way of the push")
    tip = tip_for(standoff)
    pad = geom.c - (r_near + standoff) * u
    c_f, c_m = room - standoff, standoff - m
    f.update(standoff=standoff, tip_grasp=(tip[0], tip[1], z_tip), tip_hover=(tip[0], tip[1], z_hover),
             roll_deg=round(ctx.roll_for(psi, tip), 3), clearance_fixed=c_f, clearance_moving=c_m)
    problem = _roll_problem(ctx, f["roll_deg"])
    if problem:
        return _done(f, problem)
    hit = _body_hits(ctx, pad, u, obstacles, z_tip)
    if hit is not None:
        f["blockers"] = [_blocker(hit, "body", 0.0, 0.0)]
        return _done(f, f"the gripper body would hit {hit.label} (it is taller than the fingers are long)")
    contact_raw = ctx.touch_raw + (q_c - gm.Q_CLOSED) * _TICKS_PER_DEG
    f.update(expected_raw=int(round(contact_raw - float(p["stall_offset_ticks"]))), open_raw=ctx.raw_for_q(q_open),
             release_raw=ctx.raw_for_q(q_open))
    f["slot"]["standoff_mm"] = round(standoff * 1e3, 1)
    if f["expected_raw"] - ctx.touch_raw < 55.0:
        return _done(f, f"too narrow: a {_mm(width)} object would stall too close to the empty-gripper position "
                        "to tell a grasp from a miss")
    score = 100.0
    score -= _score_clear(ctx, f, (("fixed finger", c_f, fx["who"].label if fx["who"] else "?"),
                                   ("opened moving jaw", c_m, mv["who"].label if mv["who"] else "?")))
    score -= 1000.0 * (s0 - standoff)
    score -= 3.0 * max(0.0, q_c - 13.7)  # beyond about 32 mm the pads are no longer parallel
    if "length" in label:
        score -= 2.0
    if geom.kind == "unknown":
        score -= 10.0
    score -= 0.05 * abs(_wrap180(f["roll_deg"] - gm.WRIST_ROLL_SKEW_DEG))
    f["score"] = round(max(1.0, score), 2)
    return _done(f)


# ======================================================================================= public: grasps
def plan_grasps(target, container, scene, cfg: dict | None = None, *, hollow=None, assume_hollow: bool = False,
                mode: str | None = None, tip_offset=None, camera_xy=None, reach=None,
                params: dict | None = None) -> list:
    """Ways to take `target` (a SceneObject lying inside `container`) with the gripper pointing straight
    down, best first. Never empty; may hold no ok candidate (then explain() says what is in the way).

    hollow: True / False overrides what the scene says about the object; assume_hollow: what to take it
    for when neither says (see hollow_model). A wall pinch is released only when the option, the scene or
    the request says that the object is hollow; on the built-in guess for cylinders it is planned but
    marked unconfirmed (not ok). mode: "pinch" or "two_sided" restricts the search (None: both, the wall
    pinch only for hollow objects; "pinch" is itself a statement that the object is hollow).
    tip_offset: fingertip in tool axes (default cfg["tip_offset"]). camera_xy: where the camera stands;
    directions across its line of sight are preferred, and one in which the moving jaw would hide the
    fixed one from it is marked look_closed (the jaws are then looked at closed).
    reach(xyz, roll_deg) -> "" or a reason: the arm's say on whether it can point
    straight down there; asked for the grasp point, the hover point, the point the approach arrives at
    above it (approach_drop higher) and the lift point of every candidate that has room."""
    p = {**DEFAULTS, **((cfg or {}).get("take") or {}), **(params or {})}
    ctx = _Ctx(cfg, p, tip_offset)
    problem = _target_problem(target)
    if problem:
        return [TakeGrasp(ok=False, reason=problem, mode=mode or "two_sided", jaw_azimuth_deg=0.0, roll_deg=0.0,
                          pitch_deg=90.0, tip_grasp=(0.0, 0.0, 0.0), tip_hover=(0.0, 0.0, 0.0), width=0.0, standoff=0.0,
                          open_raw=0, expected_raw=0, release_raw=0, clearance_fixed=_INF, clearance_moving=_INF,
                          score=0.0)]
    geom = _Geom(target)
    hm = hollow_model(target, p, True if (mode == "pinch" and hollow is None) else hollow, assume_hollow)
    guess = bool(hm["hollow"]) and hm["source"] == "default"
    obstacles = _obstacles(scene, target, container, p)
    rim_z = geom.z_top
    if container is not None:
        rim_z = max(rim_z, _inner_of(container, p)[1])
    r = float(p["hover_radius"])
    near = [ob.z_top for ob in obstacles if not ob.wall and float(ob.sdist(geom.c)[0]) < r]
    z_hover = max([rim_z, geom.z_top] + near) + float(p["rim_margin"])
    z_rim_clear = rim_z + min(float(p["rim_clear"]), float(p["rim_clear_min"]))
    step = float(p["azimuth_step_deg"])
    by_mode = {"two_sided": [], "pinch": []}
    for psi, label in geom.directions(step):
        if mode in (None, "two_sided"):
            by_mode["two_sided"].append(_two_sided(ctx, geom, obstacles, psi, z_hover, label))
        if mode in (None, "pinch") and hm["hollow"]:
            by_mode["pinch"].append(_pinch(ctx, geom, hm, obstacles, psi, z_hover, label))
    extra = list(hm["notes"])
    if hm["hollow"] and hm["source"] in ("default", "request"):
        extra.append(("hollow (open at the top) is a guess for a cylinder" if guess else
                      "hollow (open at the top) because the request says so") +
                     f": wall {_mm(hm['wall'])}, inner bottom {_mm(hm['z_inner_bottom'] - geom.z_bottom)} above the floor")
    if scene is None:
        extra.append("no scene given: other objects were not checked")
    cands = []
    for name, lst in by_mode.items():
        n = len(lst)
        for i, c in enumerate(lst):
            if c.ok and geom.round and n >= 8:
                # prefer the middle of a free range of directions: the position is only known to a few mm
                k = 1
                while k < n // 2 and lst[(i + k) % n].ok and lst[(i - k) % n].ok:
                    k += 1
                c.score = round(c.score + 0.3 * min((k - 1) * 360.0 / n, 40.0), 2)
            if c.ok and reach is not None:
                lift = (c.tip_grasp[0], c.tip_grasp[1], z_rim_clear + c.h_below)
                # (the approach arrives straight above the hover point: the highest point of the way in)
                above = (c.tip_hover[0], c.tip_hover[1], c.tip_hover[2] + float(p["approach_drop"]))
                for what, xyz in (("grasp point", c.tip_grasp), ("hover point", c.tip_hover),
                                  ("point above the hover point", above), ("lift point", lift)):
                    why = reach(xyz, c.roll_deg)
                    if why:
                        c.ok, c.score = False, 0.0
                        c.reason = f"{_REACH} {what}: {why}"
                        side = _reach_side(reach, xyz, c.roll_deg)
                        if side:
                            c.notes.append(side)
                        break
            if c.ok and name == "pinch" and guess:
                c.ok, c.score, c.unconfirmed = False, 0.0, True
                c.reason = ("a wall pinch has room here, but that the object is hollow and open at the top is only "
                            "a guess (the scene does not say so, and nothing checks the cavity before the moving jaw "
                            "goes down into it)")
            if c.ok and camera_xy is not None:
                # The camera measures where the jaws are; that is weakest ALONG its line of sight (a height
                # error of the arm shifts the reading there) and that is also where one jaw can hide the
                # other. So, other things equal, the jaws should open across the line of sight.
                los = np.asarray(camera_xy, dtype=float)[:2] - np.asarray(c.tip_hover[:2], dtype=float)
                far = float(np.hypot(*los))
                if far > 1e-6:
                    c.score = round(c.score - 6.0 * abs(float(_unit(c.jaw_azimuth_deg) @ los) / far), 2)
                opening = gm.width_for_q(ctx.q_for_raw(c.open_raw)) * 1e-3
                if not fixed_jaw_visible(c.jaw_azimuth_deg, c.tip_hover, camera_xy, opening):
                    c.look_closed = True
                    c.score = round(c.score - 3.0, 2)
                    c.notes.append("the moving jaw hides the fixed one from the camera: the jaws are looked at closed")
            c.notes.extend(extra)
            cands.append(c)
    if not cands:  # e.g. mode "pinch" for a solid object
        cands.append(TakeGrasp(ok=False, reason="the object is not hollow: no wall to pinch", mode="pinch",
                               jaw_azimuth_deg=0.0, roll_deg=0.0, pitch_deg=90.0, tip_grasp=(0.0, 0.0, 0.0),
                               tip_hover=(0.0, 0.0, 0.0), width=0.0, standoff=0.0, open_raw=0, expected_raw=0,
                               release_raw=0, clearance_fixed=_INF, clearance_moving=_INF, score=0.0))
    # best first; among those that are not ok, the ones that only lack a statement (they have room) come first
    cands.sort(key=lambda c: (not c.ok, not c.unconfirmed, -c.score))
    return cands


SAY_HOLLOW = "say so: `fd take --hollow`"  # the caller that knows the object's id puts it in (skills.plan_take)
# a candidate that has room for the jaws, but not the arm's reach (how its reason begins; explain() looks for it)
_REACH = "the arm cannot point straight down at the"
_TOO_FAR, _TOO_NEAR = "reach: too far from the robot", "reach: too close to the robot"
# what to do about it (also used by skills.plan_take when a path fails for the same reason)
REACH_NEXT = {
    _TOO_FAR: "the object lies too far from the robot for a grasp from straight above: move the container nearer to "
              "the robot (by hand), then repeat. More room around the object or another place spot (--to) does not help",
    _TOO_NEAR: "the object lies too close to the robot for a grasp from straight above: move the container further "
               "away from the robot (by hand), then repeat. More room around the object or another place spot (--to) "
               "does not help",
    "": "the arm cannot hold the gripper straight down over the object: move the container to straight in front of "
        "the robot, well inside its reach (by hand), then repeat. More room around the object or another place spot "
        "(--to) does not help",
}


def _reach_side(reach, xyz, roll_deg) -> str:
    """Which way the arm's reach ends at a point where it cannot hold the gripper straight down: _TOO_FAR when
    the same point 3 cm nearer to the robot can be reached, _TOO_NEAR when 3 cm further away can, else ""."""
    r = math.hypot(float(xyz[0]) - _X_PAN, float(xyz[1]))
    if r < 1e-6:
        return ""
    ux, uy = (float(xyz[0]) - _X_PAN) / r, float(xyz[1]) / r
    for d, side in ((-0.03, _TOO_FAR), (0.03, _TOO_NEAR)):
        if not reach((float(xyz[0]) + d * ux, float(xyz[1]) + d * uy, float(xyz[2])), roll_deg):
            return side
    return ""


def explain(cands: list) -> dict:
    """Why no candidate is ok, in one sentence that names what is in the way, and what to do next
    (for example which neighbour to take out first). {"reason", "next"}."""
    bad = [c for c in cands if not c.ok]
    if not bad:
        return {"reason": "", "next": ""}
    if any(getattr(c, "unconfirmed", False) for c in cands):
        # A wall pinch has room, but only on the guess that the object is hollow. If it is a cap, saying so
        # is all that is needed; if it is solid, what counts is why it cannot be held across.
        reason = ("it can be held by its wall (fixed jaw outside, moving jaw inside), but that it is hollow and "
                  "open at the top is only a guess: the scene does not say so")
        nxt = "if it is a cap or a cup lying open side up, " + SAY_HOLLOW
        across = [c for c in bad if c.mode == "two_sided"]
        if across:
            ex = _blocked(across)
            reason += "; across it, " + ex["reason"]
            if ex["next"]:
                nxt += "; if it is solid: " + ex["next"]
        return {"reason": reason[:600], "next": nxt}
    reach = [c for c in bad if c.reason.startswith(_REACH)]
    if reach:
        # These directions have room for the jaws; what is missing is the arm's reach. Making room or
        # another place spot would change nothing, so say what would.
        sides = [n for c in reach for n in c.notes if n in (_TOO_FAR, _TOO_NEAR)]
        side = max(set(sides), key=sides.count) if sides else ""
        c = reach[0]
        r = math.hypot(float(c.tip_hover[0]) - _X_PAN, float(c.tip_hover[1]))
        reason = (f"{len(reach)} of the {len(cands)} jaw directions tried have room for the jaws, but {c.reason} "
                  f"(that is {r * 100:.0f} cm from the robot's shoulder axis"
                  + (f": {side[len('reach: '):]}" if side else "") + ")")
        rest = [x for x in bad if not x.reason.startswith(_REACH)]
        if rest:
            reason += "; the other directions: " + _blocked(rest)["reason"]
        return {"reason": reason[:600], "next": REACH_NEXT[side]}
    ex = _blocked(bad)
    if all(c.mode != "pinch" for c in cands) and not ex["next"].startswith("take"):
        ex["next"] = ((ex["next"] + "; " if ex["next"] else "") +
                      "if the object is hollow and open at the top (a cap, a cup), " + SAY_HOLLOW)
    return ex


def _blocked(bad: list) -> dict:
    """explain() for candidates that have no room: what is in the way, and what to move first."""
    modes = sorted({c.mode for c in bad})
    walls, neigh, other = {}, {}, {}
    frees = {}
    for c in bad:
        if not c.blockers:
            other[c.reason] = other.get(c.reason, 0) + 1
            continue
        ids = {b["id"] for b in c.blockers}
        for b in c.blockers:
            book = walls if b["wall"] else neigh
            book[b["label"]] = book.get(b["label"], 0) + 1
        removable = {b["id"]: b["label"] for b in c.blockers if not b["wall"]}
        if len(ids) == 1 and removable:  # one neighbour alone blocks this direction
            (i, label), = removable.items()
            frees.setdefault((i, label), []).append(c)
    parts = []
    if neigh:
        top = sorted(neigh.items(), key=lambda kv: -kv[1])[:3]
        parts.append("in the way: " + ", ".join(k for k, _ in top))
    if walls:
        parts.append("the container wall is too close on the other sides" if neigh else "the container wall is too close")
    for reason, _n in sorted(other.items(), key=lambda kv: -kv[1])[:2]:
        parts.append(reason)
    # the closest miss, with its numbers
    with_num = [c for c in bad if c.blockers]
    if with_num:
        best = min(with_num, key=lambda c: max(b["need_mm"] - b["free_mm"] for b in c.blockers))
        b = max(best.blockers, key=lambda b: b["need_mm"] - b["free_mm"])
        parts.append(f"closest: jaws towards {best.jaw_azimuth_deg:.0f} deg ({best.mode}) lack "
                     f"{max(b['need_mm'] - b['free_mm'], 0.0):.0f} mm at {b['label']}")
    what = "the outside finger" if modes == ["pinch"] else ("both jaws" if modes == ["two_sided"] else "the jaws")
    reason = f"no jaw direction leaves room for {what}: " + "; ".join(parts)
    nxt = ""
    if frees:
        (i, label), lst = max(frees.items(), key=lambda kv: len(kv[1]))
        nxt = (f"take {label} out first (`fd take --id {i}`): it alone blocks {len(lst)} of the "
               f"{len(bad)} jaw directions tried")
    elif neigh:
        top = sorted(neigh.items(), key=lambda kv: -kv[1])[:2]
        nxt = "make room first: take out " + " and ".join(k for k, _ in top)
    elif walls:
        nxt = ("the object stands too close to the container wall on every usable side: move it towards the "
               "middle of the container (by hand or by pushing it with the closed gripper)")
    return {"reason": reason[:600], "next": nxt}


# ======================================================================================= public: place spot
def choose_place(scene, target, container, cfg: dict | None = None, *, to=None, ok_xy=None, area=None,
                 params: dict | None = None, limit: int = 8):
    """Free spots on the table where the object can be put down, best (shortest carry) first.
    Returns (spots, reason): spots = [{"xy": (x, y), "clear_mm": free space around the object there}];
    reason says why there is none ("" otherwise).

    A spot is free when the object's outline keeps place_clear from every other object and from the
    container and lies outside the arm's static keep-outs. Where to look: the grid points (place_step)
    within place_radius of the container's centre; `area` = (x0, x1, y0, y1) limits them further
    (default cfg["take"]["place_area"], normally not set). No table coordinate is built in.
    ok_xy(xy) -> True / "" (usable) or False / a reason: the caller's own test of a spot (can the arm
    reach it, does the camera see it); it is asked last, and only until `limit` spots are found.
    to = (x, y): only that spot is examined."""
    cfg = cfg or {}
    p = {**DEFAULTS, **(cfg.get("take") or {}), **(params or {})}
    kind = getattr(target, "kind", None)
    radius = 0.5 * max(float(target.size[0]), float(target.size[1])) if kind == "cylinder" else \
        0.5 * math.hypot(float(target.size[0]), float(target.size[1]))
    need = radius + float(p["place_clear"])
    obstacles = []
    for o in (scene.objects if scene is not None else []):
        if o is target or o.id == target.id or getattr(o, "in_container", None) is not None:
            continue  # what lies in a container is covered by the container's own outline
        size = getattr(o, "size", (0.0, 0.0, 0.0))
        label = f"{getattr(o, 'color', '?')} {getattr(o, 'kind', '?')} #{o.id}"
        if getattr(o, "kind", None) == "cylinder":
            obstacles.append(_Obstacle(o.id, label, o.z_top, disc=(o.center[0], o.center[1], 0.5 * max(size[0], size[1]))))
            continue
        fp = getattr(o, "footprint", None)
        poly = _hull(fp) if fp is not None and len(fp) >= 3 else _rect(o.center, float(o.yaw), float(size[0]), float(size[1]))
        if len(poly) >= 3:
            obstacles.append(_Obstacle(o.id, label, o.z_top, poly=poly))
    table_z = float(getattr(scene, "table_z", cfg.get("table_z", 0.0)) or 0.0)
    boxes = []
    for name, box in (cfg.get("keepouts") or {}).items():
        if str(name).startswith("obj"):
            continue  # scene keep-outs: the same objects again
        try:
            x0, x1, y0, y1, z0, z1 = (float(v) for v in box[:6])
        except (TypeError, ValueError):
            continue
        if z0 <= table_z + 0.03 and z1 >= table_z:
            boxes.append((name, x0, x1, y0, y1))
    ko = radius + float(p["place_keepout_margin"])

    def judge(xy):
        """(clearance, "") or (None, why not)."""
        x, y = float(xy[0]), float(xy[1])
        for name, x0, x1, y0, y1 in boxes:
            if x0 - ko <= x <= x1 + ko and y0 - ko <= y <= y1 + ko:
                return None, f"inside the keep-out '{name}'"
        best, who = _INF, None
        for ob in obstacles:
            d = float(ob.sdist((x, y))[0])
            if d < best:
                best, who = d, ob
        if best < need:
            return None, (f"{who.label} is {_mm(max(best, 0.0))} away, {_mm(need)} needed "
                          f"(object radius {_mm(radius)} + {_mm(p['place_clear'])})")
        if ok_xy is not None:
            why = ok_xy((x, y))
            if why is False:
                why = "out of the arm's reach or out of the camera's view"
            if why not in (True, None, ""):
                return None, str(why)
        return best, ""

    def spot(xy, clear, **more):
        return {"xy": (round(float(xy[0]), 4), round(float(xy[1]), 4)),
                "clear_mm": None if not math.isfinite(clear) else round(clear * 1e3, 1), **more}

    if to is not None:
        if not _finite(to[0], to[1]):
            return [], "the place spot is not a pair of numbers"
        clear, why = judge(to)
        if clear is None:
            return [], f"place spot ({float(to[0]):.3f}, {float(to[1]):.3f}) cannot be used: {why}"
        return [{**spot(to, clear, given=True), "xy": (float(to[0]), float(to[1]))}], ""
    if area is None:
        area = p.get("place_area")
    centre = np.array([float(v) for v in (container.center if container is not None else target.center)[:2]])
    here = np.array([float(target.center[0]), float(target.center[1])])
    reach_r, step = float(p["place_radius"]), float(p["place_step"])
    n = int(math.ceil(reach_r / step))
    grid = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = centre[0] + i * step, centre[1] + j * step
            if math.hypot(x - centre[0], y - centre[1]) > reach_r:
                continue
            if area is not None and not (float(area[0]) <= x <= float(area[1]) and float(area[2]) <= y <= float(area[3])):
                continue
            # short carry first; a little preference for the robot's centre line (best view, far from the cables);
            # of two equally good spots the one nearer to the robot (rounded, so that the order does not hang on
            # the last bit of a float)
            grid.append((round(math.hypot(x - here[0], y - here[1]) + 0.3 * abs(y), 6), round(x, 6), round(y, 6)))
    grid.sort()
    spots, whys = [], {}
    for _cost, x, y in grid:
        clear, why = judge((x, y))
        if clear is None:
            key = "too close to something" if "needed" in why else why
            whys[key] = whys.get(key, 0) + 1
            continue
        spots.append(spot((x, y), clear))
        if len(spots) >= limit:
            break
    if not spots:
        top = ", ".join(f"{k} ({v} spots)" for k, v in sorted(whys.items(), key=lambda kv: -kv[1])[:3])
        return [], (f"no free spot on the table within {reach_r * 100:.0f} cm of the container"
                    + (f": {top}" if top else " (the search area is empty)"))
    return spots, ""
