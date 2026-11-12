"""Grasp and release planning for the SO-101 follower gripper (closed form, no search over poses).

    plan_grasp(target, scene, arm, cfg=None)                          -> list[GraspCandidate]   (best first)
    plan_release(target, container, cand, arm, cfg=None, scene=None)  -> ReleasePlan

No motion and no bus access: the planner only uses arm.solve / arm.check_q / arm.q_limits / arm.tip_tool /
arm.cfg, kinematics and gripper_model. On the Jetson a box takes 7-9 ms, a cylinder (12 jaw directions)
17-18 ms, a cap at far reach (pitch search on all 12) 30 ms, the slowest of 460 random cluttered scenes
37 ms; a release plan 5 ms; the first call of a process 16-24 ms (test_grasp.py: test_timing and the
property test).

How the gripper works (refs/research/gripper-geometry.md, re-checked here against the URDF meshes):
- The FIXED jaw is rigid with the wrist; the URDF tool frame `gripper_frame_link` has its origin ON the
  fixed jaw's inner pad. The MOVING jaw swings open about a pivot 75 mm up the gripper.
- When the jaws close on an object the moving jaw pushes it until it rests against the fixed pad. A grasp
  is therefore planned as: fixed pad `standoff` away from one side face, moving jaw opened beyond the
  opposite one. The two directions across the same faces are physically different grasps.
- jaw azimuth psi = base-frame direction fixed jaw -> moving jaw. For a vertical gripper
  psi = wrist_roll - shoulder_pan - 2.789 deg, which gives the wrist roll in closed form (general formula
  for a tilted gripper: _roll_for_azimuth). The result is verified through kinematics.jaw_open_axis.

Collision model: the gripper (fixed finger, housing, and the moving jaw at every angle it passes through
between the pre-grasp opening and the angle at which it holds the object) is a cloud of sample points in
the tool frame: a 2 mm lattice on the parts rigid with the wrist; the moving jaw's outline every 1.3 mm,
across its width every 2 mm and every 2 deg of its travel (2.9 mm at the nose). A candidate is refused
when a sample that is not above another object's top comes closer than `min_clear` to that object's
footprint; counting every sample below the top (not only those inside the object's height range) also
covers the vertical descent from the hover point. Between samples a reported clearance can be optimistic
by up to about 1.8 mm at a grazing corner, which is why `min_clear` cannot be configured below _MODEL_RES
(2 mm). The same cloud is checked against the surface the object stands on (finding KI-2).

Order of the checks for one jaw direction (_evaluate): size -> push travel -> reachability and pitch
(closed form) -> table and jaw-line tilt at the planned standoff -> clearances (moving side first; the
fixed side and a smaller standoff are only looked at when the moving side is free) -> arm.solve and
arm.check_q for the grasp and the hover pose. The first failure is the reported reason.

Units: metres / radians inside, degrees in public fields named *_deg, raw servo ticks for the gripper.
Tunables: DEFAULTS, overridden per key by cfg["grasp"] = {...}.
"""

from __future__ import annotations

import inspect
import math
from functools import lru_cache

import numpy as np

import gripper_model as gm
import kinematics as K
from fdtypes import GraspCandidate, ReleasePlan

__all__ = ["plan_grasp", "plan_release", "DEFAULTS"]

DEFAULTS = {
    # ---- grasp geometry
    "standoff": 0.012,  # gap between the fixed pad and the object's face before closing (used 2026-11-10)
    "min_standoff": 0.005,  # the gap is never planned smaller than this
    "tip_height": 0.004,  # fingertip height above the surface the object stands on while closing
    "max_engage": 0.015,  # deepest the fingertip goes below the object's top (tall objects)
    "min_engage": 0.006,  # objects that allow less than this are refused
    "min_width": 0.012,  # narrower objects stall too close to the empty-gripper position
    "max_width": 0.045,  # widest antipodal pinch (pad corner still opposite the fixed pad)
    "marginal_width": 0.040,  # wider than this: flagged "marginal" and ranked low
    "open_margin_deg": 24.0,  # pre-grasp opening = contact jaw angle + this (used 2026-11-10)
    "stall_offset_ticks": 40,  # the jaw stalls this many ticks below rigid contact (measured: 50, 44, 27)
    "empty_margin_ticks": 45,  # Arm.grasp_fast calls the grasp empty at touch_raw + this or less
    # ---- approach
    "hover_clear": 0.030,  # fingertip height above the object's top at the hover point
    "hover_clear_min": 0.015,  # lowest hover used when the nominal one is out of reach
    "hover_neighbour_margin": 0.012,  # hover at least this far above every object within hover_radius
    "hover_radius": 0.09,
    "min_pitch": 70.0,  # approach pitch range searched when 90 deg (straight down) is not reachable
    "max_pitch": 110.0,
    "max_jaw_tilt_deg": 12.0,  # largest elevation of the jaw line (non-zero only with a degraded pitch)
    "cyl_azimuths": 12,  # number of jaw directions tried around a cylinder (radial first)
    # ---- clearances
    "min_clear": 0.002,  # hard minimum distance between the gripper and any other object
    "tight_clear": 0.005,  # below this a candidate stays ok but is flagged "tight" and ranked lower
    "z_margin": 0.003,  # every other object is treated as this much taller than reported
    "table_margin": 0.001,  # lowest allowed gripper point above the surface the object stands on
    "self_clear": 0.004,  # the opened moving jaw must pass the object's far side by at least this
    # ---- release
    "release_margin_deg": 12.0,  # release opening = contact jaw angle + this ("only as far as needed")
    "release_fit_margin": 0.005,  # minimum free space around the object's footprint inside the container
    "release_wall_margin": 0.006,  # gripper parts below the rim keep this distance from the walls
    "release_travel_clear": 0.015,  # the fingertip travels at rim + object height + this
    "release_inside_depth": 0.002,  # fingertip height below the rim when releasing inside
    "release_above_clear": 0.005,  # object bottom above the rim when releasing from above
    "release_content_clear": 0.005,  # object bottom above things already in the container (scene given)
    "wall_default": 0.004,  # wall / floor thickness assumed when the container has no measured `inner`
}

_MODEL_RES = 0.002  # resolution of the gripper sample cloud (m): hard floor of "min_clear"
_NEAR = 0.030  # clearances beyond this are reported as inf ("nothing is near")
_TICKS_PER_DEG = 4096.0 / 360.0
_INF = float("inf")


# ======================================================================================= small helpers
def _wrap180(a_deg: float) -> float:
    """Angle in [-180, 180)."""
    return (float(a_deg) + 180.0) % 360.0 - 180.0


def _mm(v: float) -> str:
    return f"{float(v) * 1000.0:.1f} mm"


def _finite(*vals) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


class _Reject(Exception):
    """Internal: this candidate / option cannot be realised; str(e) is the reason shown to the caller."""


# ======================================================================================= gripper cloud
# Constants in gripper_frame_link coordinates, millimetres (gripper_model conventions: +x = fixed-jaw body
# side, +z = out of the fingertips, the moving jaw is on the -x side and opens towards -x).
#
# Half width (y) of the fixed finger by height z, measured on wrist_roll_follower_so101_v1.stl (slabs of
# +-0.5 mm, rounded up). gripper_model has its x profile (inner staircase, back face).
_FIXED_HW = ((6.298, 2.1), (5.5, 2.6), (4.0, 3.7), (2.0, 4.6), (0.5, 5.1), (-2.0, 5.7), (-5.0, 6.2),
             (-9.5, 6.9), (-14.0, 7.3), (-19.5, 7.7), (-25.0, 8.1), (-29.5, 8.4), (-35.0, 8.9), (-39.5, 9.3),
             (-45.0, 10.1), (-49.5, 11.0), (-55.0, 12.9), (-59.5, 15.3), (-61.08, 21.2))
# Housing + gripper servo behind the palm: bounding box of both meshes (x0, x1, y0, y1, z0, z1), widened
# on the -x side to PIVOT_X - 12.5 so that it also contains the hub of the moving jaw at every jaw angle.
_HOUSING = (-40.6, 27.3, -27.8, 24.2, -99.2, -61.1)
# Side-view outline of the moving jaw's finger at q = 0 as (x, z): inner profile heel -> nose
# (gripper_model.INNER_PROFILE_Q0, with the chamfer of the heel plate squared off because the plate's
# wide part has none), flat nose end, outer face nose -> heel (the hull points of
# gripper_model.jaw_reach_minus_x), closed by a straight cut at the heel.
_JAW_OUTLINE_Q0 = (tuple(gm.INNER_PROFILE_Q0[:2]) + ((gm.INNER_PROFILE_Q0[1][0], gm.INNER_PROFILE_Q0[2][1]),)
                   + tuple(gm.INNER_PROFILE_Q0[2:])
                   + ((-19.44, 7.273), (-21.38, 6.6), (-31.93, -15.0), (-36.07, -54.5), (-37.35, -67.30)))
# Half width (y) of the moving jaw by distance r from the pivot axis, from moving_jaw_so101_v1.stl
# (largest |y| per bin): bin upper radii and half widths.
_JAW_R_HI = np.array([15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 80.0])
_JAW_HW = np.array([24.2, 22.7, 21.2, 11.5, 10.1, 8.9, 8.1, 7.45, 6.4, 5.5, 3.75])
_GRID_MM = 2.0  # spacing of the sample points (keep <= _MODEL_RES)
_SWEEP_STEP_DEG = 2.0  # the moving jaw is sampled every <= 2 deg of its travel (2.9 mm at the nose)
_H_MAX = 0.125  # no gripper point is further than this from the tool z axis (jaw opened 100 deg: 0.114)


def _fixed_inner_x(z: float) -> float:
    """x of the fixed finger's inner (gripping-side) face at height z: the staircase of gripper_model,
    taking the more protruding level within 1 mm of a step (the risers are not perfectly sharp)."""
    xs = [x for x, z_lo, z_hi, _hw in gm.FIXED_STEPS if z_lo - 1.0 <= z <= z_hi + 1.0]
    return min(xs) if xs else (0.0 if z > 0.0 else gm.FIXED_STEPS[-1][0])


def _lattice(a0: float, a1: float, b0: float, b1: float):
    """Regular grid over [a0, a1] x [b0, b1] with both borders included and spacing <= _GRID_MM."""
    na = max(2, int(math.ceil((a1 - a0) / _GRID_MM)) + 1)
    nb = max(2, int(math.ceil((b1 - b0) / _GRID_MM)) + 1)
    aa, bb = np.meshgrid(np.linspace(a0, a1, na), np.linspace(b0, b1, nb))
    return aa.ravel(), bb.ravel()


class _Cloud:
    """Sample points of a gripper part in the tool frame, METRES: coordinate arrays x, y, z of shape (N,)
    - or (K, N) for the moving jaw at K angles - ordered along the last axis by `key`.
    key (N,), ascending = -(highest tool-frame z the point can have): it lets _low_points take only the
    points near the fingertips. lo / hi: bounding box (x, y, z) of all points."""

    __slots__ = ("x", "y", "z", "key", "lo", "hi")

    def __init__(self, x, y, z, key):
        self.x, self.y, self.z = (np.ascontiguousarray(a) for a in (x, y, z))
        self.key = np.ascontiguousarray(key)
        for a in (self.x, self.y, self.z, self.key):
            a.setflags(write=False)
        self.lo = (float(self.x.min()), float(self.y.min()), float(self.z.min()))
        self.hi = (float(self.x.max()), float(self.y.max()), float(self.z.max()))


@lru_cache(maxsize=1)
def _fixed_cloud() -> _Cloud:
    """Everything rigid with the wrist: the fixed finger (solid lattice, nose to palm) and the surface of
    the housing box behind it, ordered from the fingertip upwards."""
    # finger: one rectangle [inner face, back face] x [-half width, +half width] per layer, borders included
    n = int(math.ceil((gm.FIXED_TIP_Z - gm.PALM_Z) / _GRID_MM)) + 1
    z = np.linspace(gm.FIXED_TIP_Z, gm.PALM_Z, n)
    hw = np.interp(z, [v for v, _ in _FIXED_HW][::-1], [h for _, h in _FIXED_HW][::-1])
    x_in = np.array([_fixed_inner_x(v) for v in z])
    x_out = np.array([gm.fixed_back_face_x(v) for v in z])
    nx = np.maximum(2, np.ceil((x_out - x_in) / _GRID_MM).astype(int) + 1)
    ny = np.maximum(2, np.ceil(2.0 * hw / _GRID_MM).astype(int) + 1)
    count = nx * ny
    layer = np.repeat(np.arange(n), count)
    k = np.arange(layer.size) - np.repeat(np.cumsum(count) - count, count)
    ix, iy = k // ny[layer], k % ny[layer]
    finger = np.column_stack([x_in[layer] + ix * ((x_out - x_in) / (nx - 1))[layer],
                              -hw[layer] + iy * (2.0 * hw / (ny - 1))[layer], z[layer]])
    # housing: underside and the four side walls
    x0, x1, y0, y1, z0, z1 = _HOUSING
    xs, ys = _lattice(x0, x1, y0, y1)
    parts = [finger, np.column_stack([xs, ys, np.full(xs.shape, z1)])]
    xs, zz = _lattice(x0, x1, z0, z1)
    parts += [np.column_stack([xs, np.full(xs.shape, y), zz]) for y in (y0, y1)]
    ys, zz = _lattice(y0, y1, z0, z1)
    parts += [np.column_stack([np.full(ys.shape, x), ys, zz]) for x in (x0, x1)]
    pts = np.vstack(parts) * 1e-3
    pts = pts[np.argsort(-pts[:, 2], kind="stable")]
    return _Cloud(pts[:, 0], pts[:, 1], pts[:, 2], -pts[:, 2])


@lru_cache(maxsize=1)
def _jaw_cloud_q0():
    """Surface of the moving jaw's finger at q = 0: its side-view outline every <= _GRID_MM / 1.5, repeated
    across the finger's width every <= _GRID_MM. Returns (pts_mm, key): pts (N, 3) in MILLIMETRES ordered
    from the nose towards the pivot; key as in _Cloud (a point at distance r from the pivot can never be
    lower along the tool z axis than PIVOT_Z + r)."""
    poly = np.array(_JAW_OUTLINE_Q0, dtype=float)
    segs = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / (_GRID_MM / 1.5))))
        segs.append(a[None, :] + np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * (b - a)[None, :])
    out = np.vstack(segs)  # outline points (x, z); each edge's end point belongs to the next edge
    r = np.hypot(out[:, 0] - gm.PIVOT_X, out[:, 1] - gm.PIVOT_Z)
    hw = _JAW_HW[np.searchsorted(_JAW_R_HI, r, side="right")]
    ny = np.maximum(2, np.ceil(2.0 * hw / _GRID_MM).astype(int) + 1)
    idx = np.repeat(np.arange(len(out)), ny)
    j = np.arange(idx.size) - np.repeat(np.cumsum(ny) - ny, ny)
    y = -hw[idx] + j * (2.0 * hw[idx] / (ny[idx] - 1))
    pts = np.column_stack([out[idx, 0], y, out[idx, 1]])
    order = np.argsort(-r[idx], kind="stable")
    pts = pts[order]
    key = np.ascontiguousarray(-(gm.PIVOT_Z + r[idx][order]) * 1e-3)
    pts.setflags(write=False)
    return pts, key


def _rotate_jaw(q_deg) -> _Cloud:
    """The q = 0 jaw cloud turned to jaw angle(s) q about the pivot (gripper_model.jaw_point, vectorised):
    coordinates (N,) for a scalar q, (K, N) for K angles."""
    base, key = _jaw_cloud_q0()
    q = np.radians(np.asarray(q_deg, dtype=float))[..., None]
    c, s = np.cos(q), np.sin(q)
    dx, dz = base[:, 0] - gm.PIVOT_X, base[:, 2] - gm.PIVOT_Z
    x = (gm.PIVOT_X + c * dx - s * dz) * 1e-3
    z = (gm.PIVOT_Z + s * dx + c * dz) * 1e-3
    return _Cloud(x, np.broadcast_to(base[:, 1] * 1e-3, x.shape), z, key)


@lru_cache(maxsize=16)
def _jaw_at(q_centi: int) -> _Cloud:
    """The moving jaw at one angle (centi-degrees)."""
    return _rotate_jaw(q_centi / 100.0)


@lru_cache(maxsize=16)
def _jaw_sweep(q_lo_half: int, q_hi_half: int) -> _Cloud:
    """The moving jaw at every angle between q_lo and q_hi (given in half degrees), i.e. the volume it
    sweeps while closing or opening."""
    q_lo, q_hi = q_lo_half / 2.0, q_hi_half / 2.0
    k = max(2, int(math.ceil((q_hi - q_lo) / _SWEEP_STEP_DEG)) + 1)
    return _rotate_jaw(np.linspace(q_lo, q_hi, k))


def _sweep_for(q_from: float, q_to: float) -> _Cloud:
    """Sweep cloud covering [q_from, q_to] (rounded outwards to half degrees, which is conservative)."""
    return _jaw_sweep(int(math.floor(min(q_from, q_to) * 2.0)), int(math.ceil(max(q_from, q_to) * 2.0)))


class _Pts:
    """Gripper sample points in the base frame: coordinate arrays x, y, z (N,), their bounding box in the
    horizontal plane and their highest z."""

    __slots__ = ("x", "y", "z", "n", "bbox", "z_max")

    def __init__(self, x, y, z):
        self.x, self.y, self.z, self.n = x, y, z, len(x)
        if self.n:
            self.bbox = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
            self.z_max = float(z.max())
        else:
            self.bbox, self.z_max = (_INF, -_INF, _INF, -_INF), -_INF

    def shifted(self, dx: float, dy: float) -> "_Pts":
        return _Pts(self.x + dx, self.y + dy, self.z)


_NO_POINTS = _Pts(np.empty(0), np.empty(0), np.empty(0))


def _low_points(t_gf: np.ndarray, cloud: _Cloud, z_cut: float):
    """Base-frame positions (_Pts) of the cloud points that are not above z_cut, for the tool pose
    t_gf = kinematics.fk(q), and the lowest z of the whole cloud. Only the part of the cloud near the
    fingertips (found through cloud.key) is transformed."""
    r, tz = t_gf[2, :3], float(t_gf[2, 3])
    n = cloud.x.shape[-1]
    if r[2] < -0.3:  # tool z points (mostly) down: a point can only be low if its tool-frame z is large
        thr = (tz - z_cut - _H_MAX * math.hypot(r[0], r[1])) / (-r[2])
        n = int(np.searchsorted(cloud.key, -thr, side="right"))
    if n == 0:
        return _NO_POINTS, z_cut  # every point is above z_cut
    x, y, z = cloud.x[..., :n], cloud.y[..., :n], cloud.z[..., :n]
    zb = r[0] * x + r[1] * y + r[2] * z + tz
    z_min = float(zb.min())
    keep = zb <= z_cut
    if not keep.any():
        return _NO_POINTS, z_min
    x, y, z, zb = x[keep], y[keep], z[keep], zb[keep]
    m = t_gf
    return _Pts(m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3],
                m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3], zb), z_min


def _lowest_vertical(t_gf: np.ndarray, cloud: _Cloud) -> float:
    """Lowest base-frame z of a cloud when the tool z axis points straight down (the lowest point then is
    the one with the largest tool-frame z)."""
    return float(t_gf[2, 3]) - cloud.hi[2]


def _footprint_box(t_gf: np.ndarray, clouds) -> tuple:
    """Axis-aligned box (x0, x1, y0, y1) in the base frame that contains the horizontal footprint of the
    given clouds at tool pose t_gf (interval arithmetic on their bounding boxes: cheap, a little generous)."""
    out = []
    for row in (0, 1):
        lo = hi = float(t_gf[row, 3])
        for i in range(3):
            c = float(t_gf[row, i])
            lo += min(min(c * cl.lo[i], c * cl.hi[i]) for cl in clouds)
            hi += max(max(c * cl.lo[i], c * cl.hi[i]) for cl in clouds)
        out += [lo, hi]
    return tuple(out)


# ======================================================================================= other objects
def _rect(center, yaw: float, length: float, width: float) -> np.ndarray:
    """Corners (4, 2), counter-clockwise, of a rectangle whose `length` side points along yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    ax, ay = np.array([c, s]) * (length / 2.0), np.array([-s, c]) * (width / 2.0)
    ctr = np.array([float(center[0]), float(center[1])])
    return np.array([ctr - ax - ay, ctr + ax - ay, ctr + ax + ay, ctr - ax + ay])


def _footprint(obj):
    """Footprint polygon (E, 2) of a scene object: its own polygon when usable, else the rectangle given
    by centre / yaw / size; None when neither is usable."""
    fp = getattr(obj, "footprint", None)
    if fp is not None and len(fp) >= 3:
        try:
            poly = np.array([[float(p[0]), float(p[1])] for p in fp])
            if np.all(np.isfinite(poly)):
                return poly
        except (TypeError, ValueError, IndexError):
            pass
    try:
        if _finite(obj.center[0], obj.center[1], obj.yaw, obj.size[0], obj.size[1]) and \
                obj.size[0] > 0 and obj.size[1] > 0:
            return _rect(obj.center, float(obj.yaw), float(obj.size[0]), float(obj.size[1]))
    except (TypeError, IndexError, AttributeError):
        pass
    return None


class _Polygon:
    """Signed horizontal distance to a simple polygon (negative inside); any winding, convex or not."""

    def __init__(self, poly: np.ndarray):
        a = np.asarray(poly, dtype=float)
        keep = np.hypot(*(a - np.roll(a, 1, axis=0)).T) > 1e-9  # drop repeated vertices
        a = a[keep] if keep.sum() >= 3 else a
        nxt = np.roll(a, -1, axis=0)
        if float((a[:, 0] * nxt[:, 1] - nxt[:, 0] * a[:, 1]).sum()) < 0.0:  # make it counter-clockwise
            a = a[::-1].copy()
        b = np.roll(a, -1, axis=0)
        ab = b - a
        len2 = np.maximum((ab * ab).sum(axis=1), 1e-18)
        self.a, self.ab = a, ab  # vertices and edge vectors, (E, 2)
        self.ax, self.ay, self.by = a[:, 0].copy(), a[:, 1].copy(), b[:, 1].copy()
        self.abx, self.aby, self.inv_len2 = ab[:, 0].copy(), ab[:, 1].copy(), 1.0 / len2
        dy = self.by - self.ay
        self.slope = np.where(dy != 0.0, self.abx / np.where(dy != 0.0, dy, 1.0), 0.0)  # dx/dy, 0 for level edges
        self.bbox = (float(a[:, 0].min()), float(a[:, 0].max()), float(a[:, 1].min()), float(a[:, 1].max()))
        self._edges = [(float(a[i, 0]), float(a[i, 1]), float(ab[i, 0]), float(ab[i, 1]), float(1.0 / len2[i]))
                       for i in range(len(a))]
        # outward unit normals (nx, ny) and offsets of the edge lines, as plain floats
        nrm = np.column_stack([ab[:, 1], -ab[:, 0]]) / np.sqrt(len2)[:, None]
        self._lines = [(float(nrm[i, 0]), float(nrm[i, 1]), float(nrm[i] @ a[i])) for i in range(len(a))]
        nab = np.roll(ab, -1, axis=0)  # convex = it never turns right
        self.convex = bool(np.all(ab[:, 0] * nab[:, 1] - ab[:, 1] * nab[:, 0] >= -1e-12))

    def lower(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Convex polygons only: largest signed distance of each point to an edge LINE. Equals the signed
        distance for points inside (<= 0); outside it is a lower bound (too small only where a corner is
        the nearest feature)."""
        nx, ny, off = self._lines[0]
        h = x * nx + y * ny - off
        for nx, ny, off in self._lines[1:]:
            np.maximum(h, x * nx + y * ny - off, out=h)
        return h

    def sdist_point(self, px: float, py: float) -> float:
        """sdist for a single point, in plain Python (cheaper than numpy for one point)."""
        best, inside = _INF, False
        for ax, ay, abx, aby, inv in self._edges:
            apx, apy = px - ax, py - ay
            t = min(1.0, max(0.0, (apx * abx + apy * aby) * inv))
            dx, dy = apx - t * abx, apy - t * aby
            best = min(best, dx * dx + dy * dy)
            if (ay > py) != (ay + aby > py) and px < ax + apy * abx / aby:
                inside = not inside
        return -math.sqrt(best) if inside else math.sqrt(best)

    def sdist(self, xy: np.ndarray) -> np.ndarray:
        """Exact signed distance (distance to the nearest edge segment; crossing-number test for the sign)."""
        x, y = xy[:, 0][:, None], xy[:, 1][:, None]
        apx, apy = x - self.ax, y - self.ay  # (N, E)
        t = (apx * self.abx + apy * self.aby) * self.inv_len2
        np.clip(t, 0.0, 1.0, out=t)
        dx, dy = apx - t * self.abx, apy - t * self.aby
        dist = np.sqrt((dx * dx + dy * dy).min(axis=1))
        straddle = (self.ay > y) != (self.by > y)
        inside = np.count_nonzero(straddle & (x < self.ax + apy * self.slope), axis=1) & 1
        return np.where(inside == 1, -dist, dist)


def _inner_rect(container):
    """Validated copy of container.inner, or None when it is missing / unusable."""
    inner = getattr(container, "inner", None)
    if not isinstance(inner, dict):
        return None
    try:
        c, size = inner["center"], inner["size"]
        vals = dict(center=(float(c[0]), float(c[1])), yaw=float(inner["yaw"]),
                    size=(float(size[0]), float(size[1])), floor_z=float(inner["floor_z"]),
                    rim_z=float(inner["rim_z"]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    ok = _finite(*vals["center"], vals["yaw"], *vals["size"], vals["floor_z"], vals["rim_z"])
    if not ok or min(vals["size"]) <= 0.0 or vals["rim_z"] <= vals["floor_z"]:
        return None
    return vals


class _Obstacle:
    """Another object as a vertical prism: footprint x (-inf, z_top]. Footprint = disc (cylinders),
    polygon, or - for a container with a measured inner rectangle - the ring of its walls."""

    def __init__(self, obj):
        self.obj_id = getattr(obj, "id", None)
        self.label = f"{getattr(obj, 'color', '?')} {getattr(obj, 'kind', '?')} #{self.obj_id}"
        self.z_top = float(obj.z_top)
        self.guessed = False
        self.circle = None  # (cx, cy, r)
        self.outer = None  # _Polygon
        self.inner = None  # _Polygon: the hollow of a container
        kind = getattr(obj, "kind", "unknown")
        if kind == "cylinder" and _finite(obj.center[0], obj.center[1], obj.size[0], obj.size[1]) \
                and max(obj.size[0], obj.size[1]) > 0:
            r = 0.5 * float(max(obj.size[0], obj.size[1]))
            self.circle = (float(obj.center[0]), float(obj.center[1]), r)
        else:
            poly = _footprint(obj)
            if poly is None:  # nothing usable: assume a generous disc rather than ignore the object
                self.guessed = True
                cx, cy = (float(obj.center[0]), float(obj.center[1])) if _finite(*obj.center[:2]) else (0.0, 0.0)
                self.circle = (cx, cy, 0.04)
            else:
                self.outer = _Polygon(poly)
                inner = _inner_rect(obj) if kind == "container" else None
                if inner is not None:
                    self.inner = _Polygon(_rect(inner["center"], inner["yaw"], inner["size"][0], inner["size"][1]))
                    self.z_top = max(self.z_top, inner["rim_z"])
        if self.circle is not None:
            cx, cy, r = self.circle
            self.bbox = (cx - r, cx + r, cy - r, cy + r)
        else:
            self.bbox = self.outer.bbox

    def sdist(self, xy: np.ndarray) -> np.ndarray:
        """Signed distance of each point (N, 2) to the footprint (negative inside the material)."""
        if self.circle is not None:
            cx, cy, r = self.circle
            return np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r
        d = self.outer.sdist(xy)
        if self.inner is not None:  # inside the cavity the nearest wall is what counts
            d = np.maximum(d, -self.inner.sdist(xy))
        return d

    def min_sdist(self, x: np.ndarray, y: np.ndarray) -> float:
        """min over the points (x, y) of sdist, without computing every distance. For a convex outline the
        largest distance to an edge line is the exact signed distance of a point inside it and a lower
        bound outside (too small only where a corner is the nearest feature). So: take the point with the
        smallest bound, get its exact distance, and re-examine exactly only the points whose bound is
        below that. Exact whenever the result is positive; when points reach into the material the sign
        is exact and the depth may be understated next to a corner."""
        if self.circle is not None:
            return float(np.hypot(x - self.circle[0], y - self.circle[1]).min()) - self.circle[2]
        if not self.outer.convex or (self.inner is not None and not self.inner.convex):
            return float(self.sdist(np.column_stack([x, y])).min())
        bound = self.outer.lower(x, y)
        in_outer = bound <= 0.0
        if self.inner is not None and in_outer.any():
            # inside the outline of a container: wall material, or the cavity, where the distance to the
            # nearest wall face (exact, the cavity is convex) is what counts
            bound[in_outer] = np.maximum(bound[in_outer], -self.inner.lower(x[in_outer], y[in_outer]))
        i = int(np.argmin(bound))
        low = float(bound[i])
        if low <= 0.0 or in_outer[i]:
            return low
        best = self.outer.sdist_point(float(x[i]), float(y[i]))
        rivals = bound < best
        rivals[i] = False
        if rivals.any():
            exact = rivals & in_outer  # cavity points: their bound is already exact
            if exact.any():
                best = min(best, float(bound[exact].min()))
            rest = rivals & ~in_outer
            if rest.any():
                best = min(best, float(self.outer.sdist(np.column_stack([x[rest], y[rest]])).min()))
        return best


def _clearance(pts: _Pts, obstacles, z_margin: float, near=None):
    """Smallest signed horizontal distance between gripper sample points and the other objects. For each
    object only the points that are not above its top (+ z_margin) count, which makes the result valid
    for the pose itself AND for a vertical descent into it. Returns (distance, obstacle); (inf, None) when
    nothing is within _NEAR. near: optional list that receives every obstacle whose bounding box is within
    _NEAR of the points' bounding box (a superset of the obstacles within _NEAR).
    The result is exact; the work is kept small by visiting the objects nearest bounding box first,
    skipping an object whose box is further away than the best distance found so far, and measuring only
    the points inside the object's box grown by that distance (no other point can be closer)."""
    best, who = _INF, None
    if pts.n == 0:
        return best, who
    bx0, bx1, by0, by1 = pts.bbox
    order = []
    for ob in obstacles:
        x0, x1, y0, y1 = ob.bbox
        gap = math.hypot(max(x0 - bx1, bx0 - x1, 0.0), max(y0 - by1, by0 - y1, 0.0))  # between the two boxes
        if gap > _NEAR:
            continue
        order.append((gap, len(order), ob))
        if near is not None and ob not in near:
            near.append(ob)
    order.sort()
    x, y = pts.x, pts.y
    for gap, _i, ob in order:
        reach = min(max(best, 0.0), _NEAR)  # only a point closer than this to the object can change the result
        if gap > reach:
            break  # the remaining objects are at least as far away
        top = ob.z_top + z_margin
        sel = pts.z <= top if pts.z_max > top else None
        if ob.circle is None:  # (a disc is measured faster than its box could be tested)
            x0, x1, y0, y1 = ob.bbox
            box = (x >= x0 - reach) & (x <= x1 + reach) & (y >= y0 - reach) & (y <= y1 + reach)
            sel = box if sel is None else sel & box
        if sel is None:
            d = ob.min_sdist(x, y)
        elif sel.any():
            d = ob.min_sdist(x[sel], y[sel])
        else:
            continue
        if d <= _NEAR and d < best:
            best, who = d, ob
    return best, who


def _boundary_points(ob: _Obstacle, step: float = 0.002) -> np.ndarray:
    """Points every <= step along an object's footprint outline."""
    if ob.circle is not None:
        cx, cy, r = ob.circle
        n = max(12, int(math.ceil(2.0 * math.pi * r / step)))
        a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    out = []
    for a, ab in zip(ob.outer.a, ob.outer.ab):
        n = max(1, int(math.ceil(float(np.hypot(ab[0], ab[1])) / step)))
        out.append(a[None, :] + np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * ab[None, :])
    return np.vstack(out)


def _push_limit(boundary: np.ndarray, u: np.ndarray, obstacles, s_max: float, z_base: float):
    """The closing jaw pushes the object along -u until it rests on the fixed pad. Returns (travel, obstacle):
    the largest travel <= s_max (1 mm steps) during which the object's outline does not come within 0.5 mm
    of another object standing higher than its base (a neighbour it already touches and merely slides
    along does not count: only an approach by more than 1 mm does), and the object that stops it (None
    when s_max is free)."""
    limit, who = s_max, None
    steps = np.arange(0.001, s_max + 1e-9, 0.001)
    if len(steps) == 0:
        return limit, who
    # box swept by the outline during the push (+ 1 mm): only objects that reach into it can stop it
    dx, dy = -s_max * float(u[0]), -s_max * float(u[1])
    bx0, bx1 = float(boundary[:, 0].min()) + min(0.0, dx) - 0.001, float(boundary[:, 0].max()) + max(0.0, dx) + 0.001
    by0, by1 = float(boundary[:, 1].min()) + min(0.0, dy) - 0.001, float(boundary[:, 1].max()) + max(0.0, dy) + 0.001
    pushed = None
    for ob in obstacles:
        x0, x1, y0, y1 = ob.bbox
        if ob.z_top <= z_base + 0.002 or bx0 > x1 or bx1 < x0 or by0 > y1 or by1 < y0:
            continue
        if ob.min_sdist(boundary[:, 0], boundary[:, 1]) > s_max + 0.001:  # too far to be reached by the push
            continue
        if pushed is None:
            pushed = (boundary[None, :, :] - steps[:, None, None] * u[None, None, :]).reshape(-1, 2)
        d0 = ob.sdist(boundary)
        d1 = ob.sdist(pushed).reshape(len(steps), -1)
        blocked = ((d1 < 0.0005) & (d1 < d0[None, :] - 0.001)).any(axis=1)
        if blocked.any():
            travel = float(steps[int(np.argmax(blocked))]) - 0.001
            if travel < limit:
                limit, who = travel, ob
    return limit, who


# ======================================================================================= kinematics helpers
_F0 = K.link_frames(np.zeros(5))
_X_PAN = float(_F0[0][0, 3])  # x of the shoulder-pan axis


def _plane(p) -> np.ndarray:
    """(distance from the pan axis along the arm heading, height) of a zero-pose point."""
    return np.array([float(p[0]) - _X_PAN, float(p[2])])


_SH = _plane(_F0[1][:3, 3])  # shoulder-lift axis
_EL = _plane(_F0[2][:3, 3])  # elbow axis (zero pose)
_WR = _plane(_F0[3][:3, 3])  # wrist-flex axis (zero pose)
_L1, _L2 = float(np.linalg.norm(_EL - _SH)), float(np.linalg.norm(_WR - _EL))
_A1 = math.atan2(_EL[1] - _SH[1], _EL[0] - _SH[0])
_A2 = math.atan2(_WR[1] - _EL[1], _WR[0] - _EL[0])
_SKEW = math.radians(gm.WRIST_ROLL_SKEW_DEG)


@lru_cache(maxsize=8)
def _tool_model_cached(tool_bytes: bytes):
    tool = np.frombuffer(tool_bytes, dtype=float).reshape(4, 4)
    vals = []
    for roll in (0.0, 0.5 * math.pi, math.pi):
        t0 = K.fk(np.array([0.0, 0.0, 0.0, 0.0, roll]), tool)
        u3 = _plane(t0[:3, 3]) - _WR
        vals.append((float(t0[1, 3]), float(u3[0]), float(u3[1])))
    out = []
    for k in range(3):
        v0, v90, v180 = vals[0][k], vals[1][k], vals[2][k]
        c0 = 0.5 * (v0 + v180)
        out.append((c0, 0.5 * (v0 - v180), v90 - c0))
    return tuple(out)


def _tool_model(tool: np.ndarray):
    """Where the tool point sits relative to the wrist as a function of the wrist roll. The point is a few
    mm off the roll axis, so each coordinate is c0 + c1*cos(roll) + c2*sin(roll) (three FK calls fix the
    coefficients exactly). Returns the coefficient triples of (e, u3x, u3z): e = distance from the arm's
    vertical plane, (u3x, u3z) = vector wrist-flex axis -> tool point in that plane at the zero pose, i.e.
    the quantities kinematics.ik computes with one FK call per solution."""
    return _tool_model_cached(np.ascontiguousarray(tool, dtype=float).tobytes())


def _wrap_pi(a: float) -> float:
    """Angle (rad) modulo a full turn, in [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _tool_plane(roll_deg: float, model):
    """(u3x, u3z, e) at this wrist roll (see _tool_model)."""
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    (e0, e1, e2), (x0, x1, x2), (z0, z1, z2) = model
    return x0 + x1 * c + x2 * s, z0 + z1 * c + z2 * s, e0 + e1 * c + e2 * s


def _roll_for_azimuth(psi_deg: float, x: float, y: float, pitch_deg: float, lat) -> float:
    """Wrist roll (deg, in [-180, 180)) that makes the jaws open towards base azimuth psi when the tool
    point is at (x, y) with approach pitch pitch_deg. Closed form:
        pan   = -(atan2(y, x - x_pan) - asin(e(roll) / r))          (as in kinematics.ik)
        alpha = psi + pan                                           (azimuth relative to the arm heading)
        roll  = atan2(sin(alpha) * sin(pitch), cos(alpha)) + 2.789 deg
    i.e. roll = psi + pan + 2.789 deg at pitch 90. e(roll) is a few mm (lat = its coefficient triple, see
    _tool_model), so three passes over the pan <-> roll coupling converge far below 0.001 deg (no search)."""
    e_c, a, b = lat
    az_t = math.atan2(y, x - _X_PAN)
    r_h = max(math.hypot(x - _X_PAN, y), 1e-6)
    psi, th = math.radians(psi_deg), math.radians(pitch_deg)
    roll = psi - az_t + _SKEW
    for _ in range(3):
        e = e_c + a * math.cos(roll) + b * math.sin(roll)
        pan = -(az_t - math.asin(max(-1.0, min(1.0, e / r_h))))
        alpha = psi + pan
        roll = math.atan2(math.sin(alpha) * math.sin(th), math.cos(alpha)) + _SKEW
    return _wrap180(math.degrees(roll))


def _ik_fast(xyz, pitch_deg: float, roll_deg: float, model, lim: dict):
    """Closed-form joint vector for the tool point at xyz with this approach pitch and wrist roll: the
    analytic elbow-up solution of kinematics.ik in plain floats (about 10 x faster). Returns (q, "") or
    (None, reason) with kinematics.ik's wording. Whatever is planned with it is confirmed by arm.solve."""
    u3x, u3z, e = _tool_plane(roll_deg, model)
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    r_h = math.hypot(x - _X_PAN, y)
    if r_h <= abs(e) + 1e-6:
        return None, "target on the pan axis"
    rho = math.sqrt(r_h * r_h - e * e)
    pan = _wrap_pi(-(math.atan2(y, x - _X_PAN) - math.atan2(e, rho)))
    th = math.radians(pitch_deg)
    c, s = math.cos(th), math.sin(th)
    dx = rho - (c * u3x + s * u3z) - _SH[0]  # wrist-flex axis relative to the shoulder-lift axis
    dz = z - (-s * u3x + c * u3z) - _SH[1]
    dist = math.hypot(dx, dz)
    cos_g = (dist * dist - _L1 * _L1 - _L2 * _L2) / (2.0 * _L1 * _L2)
    if cos_g > 1.0 or cos_g < -1.0:
        return None, f"wrist centre out of reach (distance {dist:.3f} m, max {_L1 + _L2:.3f})"
    gamma = math.acos(cos_g)  # elbow-up
    elbow = gamma - (_A1 - _A2)
    lift = _wrap_pi(_A1 - math.atan2(dz, dx) - math.atan2(_L2 * math.sin(gamma), _L1 + _L2 * math.cos(gamma)))
    q = (pan, lift, elbow, _wrap_pi(th - lift - elbow), math.radians(roll_deg))
    for name, v in zip(K.ARM_JOINTS, q):
        lo, hi = lim[name]
        if not (lo <= v <= hi):
            return None, f"{name}={math.degrees(v):.1f} deg outside [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]"
    return np.array(q), ""


def _pitch_feasible(xy, zs, roll_deg: float, pitches_deg, model, lim: dict) -> np.ndarray:
    """Which approach pitches put the tool point at (xy, z) inside the joint limits? _ik_fast for all the
    heights zs and all the pitches at once: bool array (len(zs), len(pitches)). Used to pick the pitch
    when 90 deg does not work."""
    pitches_deg = np.asarray(pitches_deg, dtype=float)
    zs = np.asarray(zs, dtype=float).reshape(-1, 1)
    u3x, u3z, e = _tool_plane(roll_deg, model)
    roll = math.radians(roll_deg)
    r_h = math.hypot(xy[0] - _X_PAN, xy[1])
    none = np.zeros((len(zs), len(pitches_deg)), dtype=bool)
    if r_h <= abs(e) + 1e-6:
        return none
    rho = math.sqrt(r_h * r_h - e * e)
    pan = _wrap_pi(-(math.atan2(xy[1], xy[0] - _X_PAN) - math.atan2(e, rho)))
    for name, val in (("shoulder_pan", pan), ("wrist_roll", roll)):
        if not (lim[name][0] <= val <= lim[name][1]):
            return none
    th = np.radians(pitches_deg)[None, :]
    c, s = np.cos(th), np.sin(th)
    dx = rho - (c * u3x + s * u3z) - _SH[0]
    dz = zs - (-s * u3x + c * u3z) - _SH[1]
    cos_g = (dx * dx + dz * dz - _L1 * _L1 - _L2 * _L2) / (2.0 * _L1 * _L2)
    ok = np.abs(cos_g) <= 1.0
    gamma = np.arccos(np.clip(cos_g, -1.0, 1.0))
    elbow = gamma - (_A1 - _A2)
    two_pi = 2.0 * math.pi  # lift and wrist_flex modulo a full turn, as kinematics.ik takes them
    lift = (_A1 - np.arctan2(dz, dx) - np.arctan2(_L2 * np.sin(gamma), _L1 + _L2 * np.cos(gamma)) + math.pi) % two_pi \
        - math.pi
    wrist = (th - lift - elbow + math.pi) % two_pi - math.pi
    for name, val in (("shoulder_lift", lift), ("elbow_flex", elbow), ("wrist_flex", wrist)):
        ok &= (val >= lim[name][0]) & (val <= lim[name][1])
    return ok


def _solve_pair(ctx, xy, z_low: float, z_highs, psi_deg: float, what_low: str, what_high: str) -> dict:
    """One (pitch, roll) at which the low point (xy, z_low), a point straight above it and the point half
    way between them are all reachable with the jaws opening towards psi (the fingertip then travels the
    vertical line between them without changing its orientation). z_highs: acceptable heights of the
    upper point, best first. Preference: pitch nearest to 90 (ties: the one below 90), then the highest
    upper point; so straight down with a lower upper point beats a degraded pitch. A degraded pitch is
    taken 1 deg inside its feasible range when there is room, so that the arm does not sit on a joint limit.
    Returns {pitch, roll, q_low, q_high, z_high, high_index, degraded, why}; raises _Reject."""
    p, lim, model = ctx.p, ctx.lim, ctx.model
    x, y = float(xy[0]), float(xy[1])
    low = (x, y, float(z_low))
    highs = [(x, y, float(zh)) for zh in z_highs]
    mids = [(x, y, 0.5 * (float(z_low) + float(zh))) for zh in z_highs]
    lo_r, hi_r = math.degrees(lim["wrist_roll"][0]), math.degrees(lim["wrist_roll"][1])
    roll = _roll_for_azimuth(psi_deg, x, y, 90.0, model[0])
    if not (lo_r <= roll <= hi_r):
        raise _Reject(f"wrist roll {roll:.1f} deg needed for this jaw direction is outside the limits "
                      f"[{lo_r:.1f}, {hi_r:.1f}]")

    def result(pitch, roll_p, q_low, q_high, i, why):
        return dict(pitch=pitch, roll=roll_p, q_low=q_low, q_high=q_high, z_high=float(z_highs[i]), high_index=i,
                    degraded=pitch != 90.0, why=why if pitch != 90.0 else "")

    # ---- the normal case: straight down, nominal upper point
    q_low, reason = _ik_fast(low, 90.0, roll, model, lim)
    if q_low is not None:
        q_high, reason = _ik_fast(highs[0], 90.0, roll, model, lim)
        if q_high is not None:
            q_mid, reason = _ik_fast(mids[0], 90.0, roll, model, lim)
            if q_mid is not None:
                return result(90.0, roll, q_low, q_high, 0, "")
            why = f"the vertical line between the {what_low} and the {what_high}: {reason}"
        else:
            why = f"{what_high}: {reason}"
    else:
        why = f"{what_low}: {reason}"
    # ---- otherwise pick (pitch, upper point) from the joint-limit test over all pitches, then solve it;
    #      a choice that does not solve after all is struck out and the next best one is tried
    pitches = np.arange(float(p["min_pitch"]), float(p["max_pitch"]) + 0.5, 1.0)
    span = f"{p['min_pitch']:.0f}..{p['max_pitch']:.0f} deg"
    n = len(highs)
    table = _pitch_feasible((x, y), [low[2]] + [h[2] for h in highs] + [m[2] for m in mids], roll, pitches, model, lim)
    if not table[0].any():
        raise _Reject(f"{what_low} out of reach at any approach pitch {span} ({why})")
    feas = [table[0] & table[1 + i] & table[1 + n + i] for i in range(n)]
    off90, above = np.abs(pitches - 90.0), pitches > 90.0
    # with a margin: a pitch below 90 also needs the next higher one feasible, a pitch above 90 the next lower
    step = np.where(pitches < 90.0, 1, np.where(above, -1, 0))
    idx = np.clip(np.arange(len(pitches)) + step, 0, len(pitches) - 1)
    inside = [f & f[idx] for f in feas]
    for sets in (inside, feas):
        for _ in range(6):
            best = None
            for i, f in enumerate(sets):
                ks = np.flatnonzero(f)
                if len(ks):
                    k = int(ks[np.lexsort((above[ks], off90[ks]))[0]])
                    key = (float(off90[k]), bool(above[k]), i)
                    if best is None or key < best[0]:
                        best = (key, k, i)
            if best is None:
                break
            _, k, i = best
            pitch = float(pitches[k])
            roll_p = roll if pitch == 90.0 else _roll_for_azimuth(psi_deg, x, y, pitch, model[0])
            q_low = _ik_fast(low, pitch, roll_p, model, lim)[0] if lo_r <= roll_p <= hi_r else None
            if q_low is None:
                for f in sets:  # the low point fails at this pitch whatever the upper point
                    f[k] = False
                continue
            q_high = _ik_fast(highs[i], pitch, roll_p, model, lim)[0]
            if q_high is not None and _ik_fast(mids[i], pitch, roll_p, model, lim)[0] is not None:
                return result(pitch, roll_p, q_low, q_high, i, why)
            sets[i][k] = False
    raise _Reject(f"no approach pitch in {span} reaches both the {what_low} and the {what_high} ({why})")


def _confirm(ctx, xyz, pitch: float, roll: float, q_planned: np.ndarray, what: str, jaws=None, **check_kw):
    """The arm has the last word: arm.solve (kinematics.ik with the arm's own limits and tool) and
    arm.check_q must accept the pose (their SafetyError is a RuntimeError), and arm.solve must return the
    joint vector the plan was built on. jaws: the jaw openings (deg) the gripper will have at this pose;
    they go to arm.check_q, whose floor check of the moving jaw would otherwise assume the opening last
    seen on the bus (closed, while planning) and could pass a pose that run_path refuses later."""
    try:
        q = np.asarray(ctx.arm.solve(np.asarray(xyz, dtype=float), float(pitch), float(roll)), dtype=float)
        _check_q(ctx, q, what, jaws, **check_kw)
    except _Reject:
        raise
    except RuntimeError as e:
        raise _Reject(f"{what} refused by the arm: {e}") from None
    if q.shape != q_planned.shape or float(np.max(np.abs(q - q_planned))) > 5e-4:  # 0.03 deg (rounding: 0.003)
        raise _Reject(f"{what}: arm.solve does not return the joint angles the planner computed "
                      "(grasp.py is out of date with kinematics.py / robot.py)")
    return q


def _check_q(ctx, q, what: str, jaws=None, **check_kw):
    """arm.check_q with the jaw openings of this pose (when the arm's check_q takes them)."""
    if jaws is not None and ctx.check_takes_jaws:
        check_kw["jaw_q_deg"] = [float(v) for v in jaws]
    try:
        ctx.arm.check_q(q, what=what, **check_kw)
    except RuntimeError as e:
        raise _Reject(f"{what} refused by the arm: {e}") from None


# ======================================================================================= gripper numbers
@lru_cache(maxsize=256)
def _contact_q_scan(w_tenth_mm: int, z_top_tenth_mm: int) -> float:
    return float(gm.contact_angle(w_tenth_mm / 10.0, z_top_tenth_mm / 10.0, step=1.0))


def _contact_q(width: float, engage_nose: float) -> float:
    """Jaw angle (deg) at which the closing jaw meets an object `width` wide whose top is `engage_nose`
    above the fixed nose's end. Pad-corner closed form inside its validated window (16..45 mm wide,
    10..15 mm engagement: within 0.01 deg of the profile model), gripper_model.contact_angle otherwise
    (cached per 0.1 mm)."""
    w, e = width * 1e3, engage_nose * 1e3
    if 16.0 <= w <= 45.0 and 10.0 <= e <= 15.0:
        return float(gm.q_contact_pad_corner(w))
    return _contact_q_scan(int(round(w * 10.0)), int(round((gm.FIXED_TIP_Z - e) * 10.0)))


class _Ctx:
    """Everything that is the same for all candidates of one planning call."""

    def __init__(self, arm, cfg):
        self.arm = arm
        self.cfg = cfg if cfg is not None else arm.cfg
        self.p = {**DEFAULTS, **(self.cfg.get("grasp") or {})}
        self.p["min_clear"] = max(float(self.p["min_clear"]), _MODEL_RES)
        self.tool = np.asarray(arm.tip_tool(), dtype=float)
        self.lim = arm.q_limits()
        off = np.linalg.inv(K.tool_offset()) @ self.tool  # the fingertip point in tool-frame axes
        self.off = (float(off[0, 3]), float(off[1, 3]), float(off[2, 3]))
        self.model = _tool_model(self.tool)
        try:  # Arm.check_q(..., jaw_q_deg=...) exists since the KI-2 floor check of the moving jaw
            params = inspect.signature(arm.check_q).parameters
            self.check_takes_jaws = "jaw_q_deg" in params or any(v.kind is v.VAR_KEYWORD for v in params.values())
        except (TypeError, ValueError):
            self.check_takes_jaws = False
        self.touch_raw = float(self.cfg.get("gripper_touch_raw", 2050))
        g = self.cfg.get("gripper") or {}
        margin = float(g.get("margin", 25))
        self.raw_lo = float(g.get("closed_raw", 1998)) + margin
        self.raw_hi = float(g.get("open_raw", 3648)) - margin

    def raw_for_q(self, q_deg: float) -> int:
        """Gripper raw position for jaw angle q (deg): the tips touch (q = -11.41) at gripper_touch_raw."""
        raw = self.touch_raw + (q_deg - gm.Q_CLOSED) * _TICKS_PER_DEG
        return int(round(max(self.raw_lo, min(self.raw_hi, raw))))

    def q_for_raw(self, raw: float) -> float:
        return gm.Q_CLOSED + (float(raw) - self.touch_raw) / _TICKS_PER_DEG

    def tip_xy(self, pad_xy: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Fingertip (tool point) xy when the pad origin is at pad_xy and the jaws open towards u
        (vertical gripper: tool x axis = -u, tool y axis = u turned by +90 deg)."""
        return pad_xy - self.off[0] * u + self.off[1] * np.array([-u[1], u[0]])

    def nose_above_tip(self) -> float:
        """How far the end of the fixed nose is above the fingertip point (vertical gripper)."""
        return self.off[2] - gm.FIXED_TIP_Z * 1e-3


# ======================================================================================= grasp planning
class _Target:
    def __init__(self, obj):
        self.kind = obj.kind
        self.c = np.array([float(obj.center[0]), float(obj.center[1])])
        self.length, self.width, self.height = (float(v) for v in obj.size[:3])
        self.yaw = float(obj.yaw)
        self.z_bottom = float(obj.z_bottom)
        self.z_top = self.z_bottom + self.height
        self.obstacle = _Obstacle(obj)
        self.boundary = _boundary_points(self.obstacle)


def _target_problem(target) -> str:
    """Why no grasp can be planned for this object at all ('' = plannable)."""
    kind = getattr(target, "kind", None)
    if kind == "container":
        return "the target is a container"
    if kind not in ("box", "cylinder", "unknown"):
        return f"cannot plan a grasp for an object of kind {kind!r}"
    if getattr(target, "in_container", None) is not None:
        return (f"the object lies inside container {target.in_container} "
                "(grasping inside a container is not supported)")
    try:
        vals = [target.center[0], target.center[1], target.size[0], target.size[1], target.size[2],
                target.yaw, target.z_bottom]
    except (TypeError, IndexError, AttributeError):
        return "the object has no usable centre / size"
    if not _finite(*vals):
        return "the object's centre / size / yaw / z_bottom is not a finite number"
    if min(float(v) for v in target.size[:3]) <= 0.0:
        return f"the object's size {tuple(round(float(v), 4) for v in target.size[:3])} is not positive"
    return ""


def _jaw_directions(tgt: _Target, n_cyl: int):
    """(azimuth_deg, width across the jaws, description) for every jaw direction worth trying."""
    if tgt.kind == "cylinder":
        d = max(tgt.length, tgt.width)
        n = max(4, int(n_cyl))
        step = 360.0 / n
        radial = math.degrees(math.atan2(tgt.c[1], tgt.c[0] - _X_PAN))  # moving jaw away from the robot
        out = [(radial, d, "cylinder, radial (moving jaw away from the robot)")]
        k = 1
        while len(out) < n:
            out.append((radial + k * step, d, f"cylinder, {k * step:+.0f} deg from radial"))
            if len(out) < n and abs(k * step - 180.0) > 1e-6:
                out.append((radial - k * step, d, f"cylinder, {-k * step:+.0f} deg from radial"))
            k += 1
        return out
    yaw = math.degrees(tgt.yaw)
    what = "" if tgt.kind == "box" else "shape unknown, planned on its bounding box; "
    return [(yaw + 90.0, tgt.width, what + "across the width"), (yaw - 90.0, tgt.width, what + "across the width"),
            (yaw, tgt.length, what + "across the length"), (yaw + 180.0, tgt.length, what + "across the length")]


def _best_standoff(s0: float, s_min: float, c_f: float, c_m: float, fixed: _Pts, moving: _Pts, obstacles, u,
                   z_margin: float) -> float:
    """Trade the gap in front of the fixed pad against the space behind the finger. Moving the whole
    gripper towards the object by d shrinks the standoff by d and takes the fixed finger d away from
    whatever is straight behind it, so the balanced choice is standoff = (standoff + clearance_fixed) / 2
    (not below s_min). It is used only if, measured at the shifted position, it really improves
    min(standoff, clearance_fixed, clearance_moving) by more than 0.2 mm; else the nominal one is kept.
    c_f, c_m: the clearances at s0; obstacles: those near the gripper."""
    if c_f >= s0:
        return s0
    s1 = max(s_min, 0.5 * (s0 + c_f))
    if s0 - s1 < 2e-4:
        return s0
    dx, dy = (s0 - s1) * u[0], (s0 - s1) * u[1]
    v1 = min(s1, _clearance(fixed.shifted(dx, dy), obstacles, z_margin)[0],
             _clearance(moving.shifted(dx, dy), obstacles, z_margin)[0])
    return s1 if v1 > min(s0, c_f, c_m) + 2e-4 else s0


def _place_gripper(ctx, tgt: _Target, obstacles, q: np.ndarray, tip, q_closed: float, q_open: float) -> dict:
    """Put the gripper model at joint pose q (planned for the fingertip at `tip`): where its parts are.
    The moving jaw is taken at every angle from q_open (pre-grasp) down to q_closed (where it ends up
    holding the object). Clearances are measured separately (_measure) because they cost the most."""
    z_margin = float(ctx.p["z_margin"])
    t_gf = K.fk(q)
    o = K.jaw_open_axis(t_gf)
    ox, oy, oz = ctx.off  # FK check of the closed-form joint vector: is the fingertip where it was asked to be?
    tip_err = math.sqrt(sum((t_gf[i, 0] * ox + t_gf[i, 1] * oy + t_gf[i, 2] * oz + t_gf[i, 3] - tip[i]) ** 2
                            for i in range(3)))
    fixed_c, sweep_c, open_c = _fixed_cloud(), _sweep_for(q_closed, q_open), _jaw_at(int(round(q_open * 100.0)))
    # only objects within _NEAR of the gripper's footprint can matter
    bx0, bx1, by0, by1 = _footprint_box(t_gf, (fixed_c, sweep_c))
    reach = [ob for ob in obstacles if not (ob.bbox[0] > bx1 + _NEAR or ob.bbox[1] < bx0 - _NEAR
                                            or ob.bbox[2] > by1 + _NEAR or ob.bbox[3] < by0 - _NEAR)]
    z_cut = max([ob.z_top for ob in reach] + [tgt.z_top]) + z_margin
    fixed, z_min_fixed = _low_points(t_gf, fixed_c, z_cut)
    opened, _ = _low_points(t_gf, open_c, z_cut)
    if reach or t_gf[2, 2] > -0.9999999:
        moving, z_min_moving = _low_points(t_gf, sweep_c, z_cut)
    else:  # nothing near and the gripper points straight down: only the jaw's lowest point is needed
        moving, z_min_moving = _NO_POINTS, _lowest_vertical(t_gf, sweep_c)
    return dict(azimuth=math.degrees(math.atan2(o[1], o[0])),
                elev=math.degrees(math.asin(max(-1.0, min(1.0, float(o[2]))))), tip_err=tip_err,
                fixed=fixed, moving=moving, opened=opened, reach=reach,
                z_min_fixed=z_min_fixed, z_min_moving=z_min_moving,
                near=[], c_f=_INF, who_f=None, c_m=_INF, who_m=None, self_f=_INF, self_m=_INF)


def _measure(ctx, tgt: _Target, place: dict) -> None:
    """Clearances of a placed gripper, written into `place`: to the object itself (self_f, self_m), of the
    moving jaw's sweep (c_m) and of the fixed side (c_f) to the other objects. The fixed side is skipped
    (left at inf) when the moving side already rules the candidate out: nothing can cure that."""
    z_margin = float(ctx.p["z_margin"])
    place["self_f"] = _clearance(place["fixed"], [tgt.obstacle], z_margin)[0]
    place["self_m"] = _clearance(place["opened"], [tgt.obstacle], z_margin)[0]
    place["c_m"], place["who_m"] = _clearance(place["moving"], place["reach"], z_margin, near=place["near"])
    if place["c_m"] >= float(ctx.p["min_clear"]):
        place["c_f"], place["who_f"] = _clearance(place["fixed"], place["reach"], z_margin, near=place["near"])


def _pose_problem(ctx, tgt: _Target, pose: dict, place: dict, psi_deg: float, width: float, engage: float) -> str:
    """What is wrong with a placed gripper before any other object is looked at ('' = nothing): the FK
    checks of the closed forms, the table (finding KI-2) and the tilt of the jaw line."""
    p = ctx.p
    if place["tip_err"] > 2e-4:
        return (f"internal error: the closed-form joint angles put the fingertip {_mm(place['tip_err'])} from "
                "its target (grasp.py is out of date with kinematics.py)")
    if abs(_wrap180(place["azimuth"] - psi_deg)) > 0.5:
        return (f"wrist roll {pose['roll']:.1f} deg opens the jaws towards {place['azimuth']:.1f} deg instead "
                f"of {psi_deg:.1f} deg")
    # table: with a tilted gripper the closing moving jaw can hang below the fingertip
    floor = tgt.z_bottom + float(p["table_margin"])
    for part, z_min in (("opened moving jaw", place["z_min_moving"]), ("fixed finger", place["z_min_fixed"])):
        if z_min < floor:
            h = z_min - tgt.z_bottom
            where = f"{_mm(h)} above" if h >= 0 else f"{_mm(-h)} below"
            return (f"the {part} would touch the surface the object stands on: its lowest point would be "
                    f"{where} that surface at pitch {pose['pitch']:.0f} deg, roll {pose['roll']:.0f} deg "
                    f"(it must stay {_mm(p['table_margin'])} above)")
    # a tilted jaw line makes the moving pad meet the object higher (or lower) than the fixed pad
    tilt = place["elev"]
    if abs(tilt) > float(p["max_jaw_tilt_deg"]):
        return (f"jaw line tilted {tilt:+.1f} deg at pitch {pose['pitch']:.0f} deg "
                f"(maximum {p['max_jaw_tilt_deg']:.0f} deg): the pads would not meet the object squarely")
    rise = width * math.sin(math.radians(tilt))
    if engage - rise < float(p["min_engage"]):
        return (f"jaw line tilted {tilt:+.1f} deg: the moving pad would meet the object {_mm(rise)} higher "
                f"and hold only {_mm(engage - rise)} of it (minimum {_mm(p['min_engage'])})")
    return ""


def _evaluate(ctx, tgt: _Target, obstacles, hover_base: float, psi_deg: float, width: float, label: str):
    """Plan one jaw direction completely: an ok GraspCandidate, or a rejected one with the reason."""
    p = ctx.p
    psi_deg = _wrap180(psi_deg)
    u = np.array([math.cos(math.radians(psi_deg)), math.sin(math.radians(psi_deg))])
    nominal = float(p["standoff"])
    z_tip = tgt.z_bottom + max(float(p["tip_height"]), tgt.height - float(p["max_engage"]))
    engage = tgt.z_top - z_tip  # how far the fingertip is below the object's top
    z_hover_nom = max(tgt.z_top + float(p["hover_clear"]), hover_base)
    z_hover_min = max(tgt.z_top + float(p["hover_clear_min"]), hover_base)
    hovers = [z_hover_nom] if z_hover_nom - z_hover_min < 1e-3 else \
        [z_hover_nom, 0.5 * (z_hover_nom + z_hover_min), z_hover_min]

    def tip_for(standoff):
        return ctx.tip_xy(tgt.c - (width / 2.0 + standoff) * u, u)

    held = (width / 2.0) * u - ctx.tip_xy(np.zeros(2), u)  # object centre minus fingertip once it is held
    tip0 = tip_for(nominal)
    # every field is a finite number from the start (the nominal plan), so that a rejected candidate still
    # says where it would have been; clearances stay inf until they are measured
    f = dict(ok=False, reason="", jaw_azimuth_deg=round(psi_deg, 3),
             roll_deg=round(_roll_for_azimuth(psi_deg, tip0[0], tip0[1], 90.0, ctx.model[0]), 3), pitch_deg=90.0,
             tip_grasp=(tip0[0], tip0[1], z_tip), tip_hover=(tip0[0], tip0[1], z_hover_nom), width=width,
             standoff=nominal, open_raw=0, expected_raw=0, clearance_fixed=_INF, clearance_moving=_INF,
             score=0.0, object_offset=(held[0], held[1]), notes=[label])

    def done(reason=""):
        f["ok"], f["reason"] = (reason == ""), reason
        for key in ("tip_grasp", "tip_hover", "object_offset"):
            f[key] = tuple(round(float(v), 5) for v in f[key])
        for key in ("roll_deg", "pitch_deg", "width", "standoff", "clearance_fixed", "clearance_moving", "score"):
            f[key] = float(f[key])
        if not f["ok"]:
            f["score"] = 0.0
        return GraspCandidate(**f)

    # ---- can the jaws hold this size at all?
    if width > float(p["max_width"]) + 1e-9:
        return done(f"too wide: {_mm(width)} across the jaws, the gripper pinches at most {_mm(p['max_width'])}")
    if width < float(p["min_width"]) - 1e-9:
        return done(f"too narrow: {_mm(width)} across the jaws (minimum {_mm(p['min_width'])})")
    if engage < float(p["min_engage"]) - 1e-9:
        return done(f"too low: an object {_mm(tgt.height)} high leaves the fingertips {_mm(engage)} of grip "
                    f"(minimum {_mm(p['min_engage'])})")
    q_c = _contact_q(width, engage - ctx.nose_above_tip())
    if not math.isfinite(q_c) or not (gm.Q_CLOSED < q_c < 60.0):
        return done(f"the jaw model has no contact angle for a width of {_mm(width)}")
    contact_raw = ctx.touch_raw + (q_c - gm.Q_CLOSED) * _TICKS_PER_DEG
    f["expected_raw"] = int(round(contact_raw - float(p["stall_offset_ticks"])))
    f["open_raw"] = ctx.raw_for_q(q_c + float(p["open_margin_deg"]))
    q_open = ctx.q_for_raw(f["open_raw"])  # what the gripper will really be commanded to
    q_stall = ctx.q_for_raw(f["expected_raw"])  # where the servo reads when the jaws hold the object
    if f["expected_raw"] - ctx.touch_raw < float(p["empty_margin_ticks"]) + 10.0:
        return done(f"too narrow: a {_mm(width)} object would stall at raw {f['expected_raw']}, too close to the "
                    f"empty-gripper position ({ctx.touch_raw:.0f}) to tell a grasp from a miss")

    # ---- how far may the object be pushed towards the fixed jaw?
    min_standoff = min(float(p["min_standoff"]), nominal)
    travel, blocker = _push_limit(tgt.boundary, u, obstacles, nominal, tgt.z_bottom)
    if travel < min_standoff - 1e-9:
        return done(f"closing would push the object into {blocker.label}: only {_mm(travel)} of free travel "
                    f"towards the fixed jaw (minimum standoff {_mm(min_standoff)})")
    standoff = min(nominal, travel)
    if blocker is not None:
        f["notes"].append(f"standoff limited to {_mm(standoff)}: {blocker.label} is in the way of the push")

    # ---- pose at the nominal standoff; when something is close, once more at a better standoff
    tip = tip_for(standoff)
    f.update(standoff=standoff, tip_grasp=(tip[0], tip[1], z_tip), tip_hover=(tip[0], tip[1], z_hover_nom))
    try:
        pose = _solve_pair(ctx, tip, z_tip, hovers, psi_deg, "grasp point", "hover point above the object")
    except _Reject as e:
        return done(str(e))
    place = _place_gripper(ctx, tgt, obstacles, pose["q_low"], (tip[0], tip[1], z_tip), q_stall, q_open)
    f.update(tip_hover=(tip[0], tip[1], pose["z_high"]), roll_deg=round(pose["roll"], 3), pitch_deg=pose["pitch"])
    problem = _pose_problem(ctx, tgt, pose, place, psi_deg, width, engage)
    if problem:  # nothing about the other objects is measured for a pose that is ruled out anyway
        return done(problem)
    _measure(ctx, tgt, place)
    min_clear = float(p["min_clear"])
    # a smaller standoff can win back at most (standoff - min_standoff) behind the fixed finger and only
    # makes things worse on the moving side: do not search when that cannot be enough
    hopeless = place["c_m"] < min_clear or place["c_f"] + (standoff - min_standoff) < min_clear
    if place["c_f"] < standoff and standoff > min_standoff and not hopeless:
        better = _best_standoff(standoff, min_standoff, place["c_f"], place["c_m"], place["fixed"], place["moving"],
                                place["near"], u, float(p["z_margin"]))
        if better < standoff - 2e-4:
            tip2 = tip_for(better)
            try:
                pose2 = _solve_pair(ctx, tip2, z_tip, hovers, psi_deg, "grasp point", "hover point above the object")
                place2 = _place_gripper(ctx, tgt, obstacles, pose2["q_low"], (tip2[0], tip2[1], z_tip), q_stall, q_open)
                if _pose_problem(ctx, tgt, pose2, place2, psi_deg, width, engage):
                    pose2 = None
            except _Reject:
                pose2 = None  # keep the nominal standoff; the checks below decide
            if pose2 is not None:
                who = place["who_f"].label if place["who_f"] is not None else "a neighbour"
                f["notes"].append(f"standoff reduced from {_mm(standoff)} to {_mm(better)} to keep clear of {who}")
                pose, place, standoff = pose2, place2, better
                _measure(ctx, tgt, place)
    tip = tip_for(standoff)
    f.update(standoff=standoff, tip_grasp=(tip[0], tip[1], z_tip), tip_hover=(tip[0], tip[1], pose["z_high"]),
             roll_deg=round(pose["roll"], 3), pitch_deg=pose["pitch"], clearance_fixed=place["c_f"],
             clearance_moving=place["c_m"])
    tilt = place["elev"]
    if pose["degraded"]:
        f["notes"].append(f"degraded pitch {pose['pitch']:.0f} deg (straight down not reachable: {pose['why']})")
    if pose["high_index"] > 0:
        f["notes"].append(f"hover lowered to {_mm(pose['z_high'] - tgt.z_top)} above the object "
                          "(the nominal hover point is out of reach)")
    if abs(tilt) > 0.5:
        f["notes"].append(f"jaw line tilted {tilt:+.1f} deg")
    # ---- the gripper must come down around the object, not on it
    if place["self_f"] < min(standoff, 0.003) - 5e-4:
        return done(f"the fixed finger would land on the object itself (distance {_mm(place['self_f'])})")
    if place["self_m"] < float(p["self_clear"]):
        return done(f"the opened moving jaw would not clear the object's far side "
                    f"(distance {_mm(place['self_m'])}, minimum {_mm(p['self_clear'])})")
    # ---- other objects (the moving side first: no standoff can cure it)
    sides = (("opened moving jaw", "sweep into", place["c_m"], place["who_m"]),
             ("fixed finger", "hit", place["c_f"], place["who_f"]))
    too_close = [f"the {side} would {verb} {who.label} ("
                 + (f"overlap {_mm(-c)}" if c < 0 else f"clearance {_mm(c)}, minimum {_mm(min_clear)}") + ")"
                 for side, verb, c, who in sides if c < min_clear]
    if too_close:
        return done("; ".join(too_close))
    # ---- the arm has the last word: its solver and its safety check, for both poses, with exactly the
    #      (rounded) numbers the candidate will carry
    for key in ("tip_grasp", "tip_hover"):
        f[key] = tuple(round(float(v), 5) for v in f[key])
    # the openings the gripper goes through at these poses. Not only the pre-grasp one: with a tilted
    # gripper the closing jaw is lowest about half way (the planner's own table check above covers every
    # 2 deg of it; these samples are for the arm's floor check of the moving jaw)
    jaws = np.linspace(q_stall, q_open, 7)
    try:
        _confirm(ctx, f["tip_grasp"], f["pitch_deg"], f["roll_deg"], pose["q_low"], "grasp pose", jaws)
        _confirm(ctx, f["tip_hover"], f["pitch_deg"], f["roll_deg"], pose["q_high"], "hover pose", jaws)
    except _Reject as e:
        return done(str(e))

    # ---- score (higher is better)
    score = 100.0
    score -= 0.08 * abs(_wrap180(pose["roll"] - gm.WRIST_ROLL_SKEW_DEG))  # mid-range wrist; radial jaws first
    score -= 3.0 * max(0.0, q_c - 13.7)  # beyond about 32 mm the pads are no longer parallel
    if "length" in label:
        score -= 2.0
    if width > float(p["marginal_width"]):
        score -= 15.0
        f["notes"].append(f"marginal: {_mm(width)} across the jaws is close to the {_mm(p['max_width'])} limit "
                          f"(jaw angle {q_c:.0f} deg, pads not parallel)")
    for side, _verb, c, who in sides:
        if c < float(p["tight_clear"]):
            score -= 10.0 + 4000.0 * (float(p["tight_clear"]) - c)
            f["notes"].append(f"tight: {_mm(c)} between the {side} and {who.label}")
        elif c < 0.010:
            score -= 1000.0 * (0.010 - c)
    score -= 1000.0 * (nominal - standoff)
    if pose["degraded"]:
        score -= 20.0 + 2.0 * abs(90.0 - pose["pitch"])
    score -= 1.5 * abs(tilt)
    score -= 200.0 * (z_hover_nom - pose["z_high"])
    if tgt.kind == "unknown":
        score -= 10.0
    f["score"] = round(max(1.0, score), 2)
    return done()


def plan_grasp(target, scene, arm, cfg=None) -> list:
    """Grasp candidates for `target` (a SceneObject standing on the table), best first.

    Boxes: both jaw directions across the width and both across the length (refused when the length
    does not fit the jaws); cylinders: cfg["grasp"]["cyl_azimuths"] directions, radial first. Every
    candidate carries the closed-form wrist roll, the approach pitch (90 when reachable; a degraded pitch
    is flagged in notes and ranked down), standoff, pre-grasp opening, expected stall position, fingertip
    grasp / hover points, the clearances to every other object of `scene` on both sides and a score.
    Rejected candidates say why in `.reason`. The list is never empty but may hold no ok candidate."""
    ctx = _Ctx(arm, cfg)
    problem = _target_problem(target)
    if problem:  # nothing can be planned: one rejected candidate that says why (all numbers are placeholders)
        return [GraspCandidate(ok=False, reason=problem, jaw_azimuth_deg=0.0, roll_deg=0.0, pitch_deg=90.0,
                               tip_grasp=(0.0, 0.0, 0.0), tip_hover=(0.0, 0.0, 0.0), width=0.0,
                               standoff=float(ctx.p["standoff"]), open_raw=0, expected_raw=0,
                               clearance_fixed=_INF, clearance_moving=_INF, score=0.0)]
    tgt = _Target(target)
    others = [o for o in (scene.objects if scene is not None else []) if o is not target and o.id != target.id]
    obstacles = [_Obstacle(o) for o in others]
    cx, cy, r = tgt.c[0], tgt.c[1], float(ctx.p["hover_radius"])
    near_tops = [ob.z_top for ob in obstacles
                 if ob.bbox[0] - r <= cx <= ob.bbox[1] + r and ob.bbox[2] - r <= cy <= ob.bbox[3] + r]
    hover_base = max(near_tops) + float(ctx.p["hover_neighbour_margin"]) if near_tops else -_INF
    cands = [_evaluate(ctx, tgt, obstacles, hover_base, psi, w, label)
             for psi, w, label in _jaw_directions(tgt, int(ctx.p["cyl_azimuths"]))]
    extra = []
    if scene is None:
        extra.append("no scene given: other objects were not checked")
    extra += [f"{ob.label} has no usable footprint: treated as an 80 mm disc" for ob in obstacles if ob.guessed]
    for c in cands:
        c.notes.extend(extra)
    cands.sort(key=lambda c: (not c.ok, -c.score))  # stable: rejected ones keep their generation order
    return cands


# ======================================================================================= release planning
def _release_fail(reason: str, notes=(), tip_above=(0.0, 0.0, 0.0), tip_release=(0.0, 0.0, 0.0)) -> ReleasePlan:
    """A refused release: ok=False and the reason; the numbers are placeholders (never NaN, so that the
    plan can always be printed as JSON) and must not be used."""
    return ReleasePlan(ok=False, reason=reason, tip_above=tuple(float(v) for v in tip_above),
                       tip_release=tuple(float(v) for v in tip_release), pitch_deg=90.0, roll_deg=0.0,
                       open_raw=0, drop_height=0.0, notes=list(notes))


def _keepouts_over(ctx, bbox) -> tuple:
    """Names of the arm's keep-out boxes that overlap the container's footprint: they would forbid the
    release itself (the static 'box' keep-out, or the scene keep-out of the container)."""
    x0, x1, y0, y1 = bbox
    out = []
    for name, box in (ctx.arm.cfg.get("keepouts") or {}).items():
        try:
            bx0, bx1, by0, by1 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        if bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0:
            out.append(name)
    return tuple(out)


def plan_release(target, container, cand, arm, cfg=None, scene=None) -> ReleasePlan:
    """Where to hold the grasped `target` to drop it into `container`.

    The OBJECT's centre (not the fingertip) is aimed at the centre of the container's inner rectangle:
    while held, the object centre sits cand.object_offset from the fingertip, turned with the jaws.
    The jaw direction is chosen among the four aligned with the container so that the object's footprint
    fits with the largest margin (then: low release possible, least wrist rotation).
    Heights: the fingertip travels at rim + object height + 15 mm (tip_above) and comes straight down to
    tip_release, where the gripper opens `release_margin_deg` beyond the contact angle, no further.
    tip_release is as low as is safe:
      - with `scene` given (so that what already lies in the container is known): fingertip
        `release_inside_depth` below the rim, but the object's bottom `release_content_clear` above the
        tallest thing inside;
      - without `scene`: the object's bottom level with the rim (nothing below the rim can be touched);
      - in both cases only if every gripper part less than `release_above_clear` above the rim stays
        `release_wall_margin` inside the inner rectangle while the jaw opens; otherwise from above the
        rim (object bottom `release_above_clear` above it), where the jaw opens over the walls.
    ok=False with a reason when the object's footprint cannot fit or no pose is reachable."""
    ctx = _Ctx(arm, cfg)
    p = ctx.p
    notes = []
    if getattr(container, "kind", None) != "container":
        return _release_fail(f"object {getattr(container, 'id', '?')} is a {getattr(container, 'kind', '?')}, "
                             "not a container")
    if not getattr(cand, "ok", False):
        return _release_fail("the grasp candidate is not ok: " + str(getattr(cand, "reason", "")))
    try:
        width = float(cand.width)
        height, z_obj = float(target.size[2]), float(target.z_bottom)
        size_l, size_w = float(target.size[0]), float(target.size[1])
        tip_z, psi_g, roll_g = float(cand.tip_grasp[2]), float(cand.jaw_azimuth_deg), float(cand.roll_deg)
        offset = np.array([float(cand.object_offset[0]), float(cand.object_offset[1])])
    except (TypeError, ValueError, IndexError, AttributeError):
        return _release_fail("the object or the grasp candidate has no usable size / pose")
    if not _finite(width, height, z_obj, size_l, size_w, tip_z, psi_g, roll_g, *offset) or min(width, height) <= 0.0:
        return _release_fail("the object or the grasp candidate has no usable size / pose")

    # ---- the container's inner rectangle
    inner = _inner_rect(container)
    if inner is None:
        t = float(p["wall_default"])
        try:
            vals = (container.center[0], container.center[1], container.yaw, container.size[0], container.size[1],
                    container.z_bottom, container.size[2])
            usable = _finite(*vals) and min(float(container.size[0]), float(container.size[1])) > 2.0 * t \
                and float(container.size[2]) > t
        except (TypeError, IndexError, AttributeError):
            usable = False
        if not usable:
            return _release_fail("the container has neither a measured inner rectangle nor a usable centre / size")
        inner = dict(center=(float(container.center[0]), float(container.center[1])), yaw=float(container.yaw),
                     size=(float(container.size[0]) - 2.0 * t, float(container.size[1]) - 2.0 * t),
                     floor_z=float(container.z_bottom) + t, rim_z=float(container.z_top))
        notes.append(f"container inner area not measured: assumed the outer size minus {_mm(t)} walls")
    inner_poly = _Polygon(_rect(inner["center"], inner["yaw"], inner["size"][0], inner["size"][1]))
    c_in = np.array(inner["center"])
    rim, floor_z = inner["rim_z"], inner["floor_z"]
    len_in, wid_in = inner["size"]

    # ---- the held object
    if getattr(target, "kind", None) == "cylinder":
        span = width  # a disc
    else:  # the horizontal side that is NOT between the jaws
        span = size_l if abs(width - size_w) <= abs(width - size_l) else size_w
    hang = max(0.0, tip_z - z_obj)  # the object's bottom is this far below the fingertip
    engage = max(height - hang, float(p["min_engage"]))
    q_c = _contact_q(width, engage - ctx.nose_above_tip())
    if not math.isfinite(q_c):
        return _release_fail(f"the jaw model has no contact angle for a width of {_mm(width)}")
    open_raw = ctx.raw_for_q(q_c + float(p["release_margin_deg"]))
    q_rel = ctx.q_for_raw(open_raw)
    q_held = q_c - float(p["stall_offset_ticks"]) / _TICKS_PER_DEG  # servo reading while the object is held
    jaws = np.linspace(q_held, q_rel, 5)  # the openings the gripper goes through above and in the container
    ug = np.array([math.cos(math.radians(psi_g)), math.sin(math.radians(psi_g))])
    if float(np.hypot(*offset)) < 1e-6:  # not filled in: the object rests against the fixed pad
        offset = (width / 2.0) * ug - ctx.tip_xy(np.zeros(2), ug)
        notes.append("cand.object_offset was empty: assumed the object rests against the fixed pad")
    # The held object's centre in the tool frame. cand.object_offset is (object centre - fingertip) in base
    # xy for the vertical grasp pose, where the tool x axis is -u and the tool y axis is u turned by +90
    # deg; along the tool z axis the centre is (hang - height / 2) beyond the fingertip. Forward kinematics
    # at the release pose then turns it with the pan / roll change (and with a tilted gripper).
    centre_gf = np.array([ctx.off[0] - float(offset @ ug), ctx.off[1] + float(offset @ np.array([-ug[1], ug[0]])),
                          ctx.off[2] + hang - height / 2.0])

    # ---- heights
    content_top = -_INF
    if scene is not None:
        tops = [float(o.z_top) for o in scene.objects
                if getattr(o, "in_container", None) == container.id and o.id != getattr(target, "id", None)]
        content_top = max(tops) if tops else -_INF
        over_content = content_top + hang + float(p["release_content_clear"])
        z_low = max(rim - float(p["release_inside_depth"]), floor_z + hang + 0.003, over_content)
    else:  # what lies in the container is unknown: do not take the object below the rim
        over_content = -_INF
        z_low = rim + hang
    z_out = max(rim + hang + float(p["release_above_clear"]), over_content)
    z_above = max(rim + height + float(p["release_travel_clear"]), z_out + 0.005)
    modes = ((("low", z_low),) if z_low < z_out - 1e-4 else ()) + (("above", z_out),)
    wall_bbox = _Obstacle(container).bbox
    ignore = _keepouts_over(ctx, wall_bbox)
    if ignore:
        notes.append("checked with the keep-out(s) over the container ignored (" + ", ".join(repr(k) for k in ignore)
                     + "): pass them as ignore_keepouts for the carry and release moves")

    # ---- the four jaw directions aligned with the container
    fit_min = float(p["release_fit_margin"])
    options, refusals, no_fit = [], [], 0
    for k in range(4):
        psi_deg = _wrap180(math.degrees(inner["yaw"]) + 90.0 * k)
        along, across = (len_in, wid_in) if k % 2 == 0 else (wid_in, len_in)
        fit = min((along - width) / 2.0, (across - span) / 2.0)
        if fit < fit_min:
            no_fit += 1
            continue
        d_psi = math.radians(_wrap180(psi_deg - psi_g))
        tip0 = c_in - np.array([math.cos(d_psi) * offset[0] - math.sin(d_psi) * offset[1],
                                math.sin(d_psi) * offset[0] + math.cos(d_psi) * offset[1]])
        for mode, z_rel in modes:
            try:
                tip = tip0  # exact for a gripper pointing straight down
                pose = _solve_pair(ctx, tip, z_rel, [z_above], psi_deg, "release point", "point above the container")
                t_gf = K.fk(pose["q_low"])
                for _ in range(2):  # aim the OBJECT's centre (by FK): matters when the gripper is tilted
                    miss = c_in - (t_gf[:2, :3] @ centre_gf + t_gf[:2, 3])
                    if float(np.hypot(*miss)) <= 2e-4:
                        break
                    tip = tip + miss
                    pose = _solve_pair(ctx, tip, z_rel, [z_above], psi_deg, "release point",
                                       "point above the container")
                    t_gf = K.fk(pose["q_low"])
                centre_err = float(np.hypot(*(c_in - (t_gf[:2, :3] @ centre_gf + t_gf[:2, 3]))))
                o = K.jaw_open_axis(t_gf)
                az = math.degrees(math.atan2(o[1], o[0]))
                tip_fk = t_gf[:3, :3] @ np.array(ctx.off) + t_gf[:3, 3]
                if float(np.linalg.norm(tip_fk - np.array([tip[0], tip[1], z_rel]))) > 2e-4:
                    raise _Reject("internal error: the closed-form joint angles miss the release point "
                                  "(grasp.py is out of date with kinematics.py)")
                if abs(_wrap180(az - psi_deg)) > 0.5:
                    raise _Reject(f"wrist roll {pose['roll']:.1f} deg opens the jaws towards {az:.1f} deg instead "
                                  f"of {psi_deg:.1f} deg")
                wall = _INF
                if mode == "low":
                    # every gripper part that is not clearly above the rim must stay inside the inner
                    # rectangle while the jaw opens ("clearly above" = by release_above_clear, as when
                    # releasing from above the rim)
                    z_cut = rim + max(float(p["z_margin"]), float(p["release_above_clear"]))
                    fixed, zf = _low_points(t_gf, _fixed_cloud(), z_cut)
                    moving, zm = _low_points(t_gf, _sweep_for(q_held, q_rel), z_cut)
                    for part in (fixed, moving):  # distance to the nearest wall face (the cavity is convex)
                        if part.n:
                            wall = min(wall, -float(inner_poly.lower(part.x, part.y).max()))
                    if wall < float(p["release_wall_margin"]):
                        how = f"come within {_mm(wall)} of" if wall >= 0 else f"reach {_mm(-wall)} beyond"
                        raise _Reject(f"while opening, the gripper would {how} the container's inner wall "
                                      f"(it must stay {_mm(p['release_wall_margin'])} inside)")
                    if min(zf, zm) < floor_z + float(p["table_margin"]):
                        raise _Reject("the gripper would touch the container floor")
                _check_q(ctx, pose["q_high"], "pose above the container", jaws, ignore_keepouts=ignore)
                _check_q(ctx, pose["q_low"], "release pose", jaws, ignore_keepouts=ignore)
                score = 10.0 * min(fit, 0.015) * 1000.0 + (20.0 if mode == "low" else 0.0) \
                    - 0.1 * abs(_wrap180(pose["roll"] - roll_g)) \
                    - (20.0 + 2.0 * abs(90.0 - pose["pitch"]) if pose["degraded"] else 0.0)
                options.append(dict(mode=mode, z_rel=z_rel, pose=pose, tip=tip, fit=fit, wall=wall, score=score,
                                    psi=psi_deg, centre_err=centre_err))
                break
            except _Reject as e:
                refusals.append((psi_deg, mode, str(e)))
    centre_xy = (float(c_in[0]), float(c_in[1]))
    if not options:
        if no_fit == 4:
            reason = (f"the object's footprint {width * 1000:.0f} x {span * 1000:.0f} mm does not fit the container's "
                      f"inner area {len_in * 1000:.0f} x {wid_in * 1000:.0f} mm with {_mm(fit_min)} to spare on "
                      "every side")
        else:
            reason = "no release pose: " + "; ".join(f"jaws towards {a:.0f} deg, {m} release: {r}"
                                                     for a, m, r in refusals)
        return _release_fail(reason, notes, (*centre_xy, z_above), (*centre_xy, z_out))
    best = max(options, key=lambda opt: opt["score"])
    pose, tip = best["pose"], best["tip"]
    # the numbers the plan will carry (rounded), confirmed by the arm's own solver and safety check
    tip_above = (round(float(tip[0]), 5), round(float(tip[1]), 5), round(float(z_above), 5))
    tip_release = (tip_above[0], tip_above[1], round(float(best["z_rel"]), 5))
    pitch, roll = float(pose["pitch"]), round(float(pose["roll"]), 3)
    try:
        _confirm(ctx, tip_above, pitch, roll, pose["q_high"], "pose above the container", jaws, ignore_keepouts=ignore)
        _confirm(ctx, tip_release, pitch, roll, pose["q_low"], "release pose", jaws, ignore_keepouts=ignore)
    except _Reject as e:
        return _release_fail(str(e), notes, tip_above, tip_release)
    notes.append(f"jaws towards {best['psi']:.0f} deg: the object's {width * 1000:.0f} x {span * 1000:.0f} mm "
                 f"footprint keeps {_mm(best['fit'])} to the inner walls")
    dz = best["z_rel"] - rim
    where = f"fingertip {_mm(abs(dz))} {'above' if dz >= 0 else 'below'} the rim"
    if best["mode"] == "low":
        near_wall = f", gripper {_mm(best['wall'])} from the walls while opening" if best["wall"] < _INF else ""
        notes.append(("release inside the rim: " if dz < 0 else "release at the rim: ") + where + near_wall)
        if scene is None:
            notes.append("no scene given, so what lies in the container is unknown and the object is not taken "
                         "below the rim; pass scene= for a lower release")
    else:
        why = [r for a, m, r in refusals if a == best["psi"] and m == "low"]
        notes.append("release from above the rim: " + where + (f" ({why[0]})" if why else ""))
    if best["fit"] < 0.010:
        notes.append(f"tight fit: only {_mm(best['fit'])} between the object and the container walls")
    if pose["degraded"]:
        notes.append(f"degraded pitch {pose['pitch']:.0f} deg (straight down not reachable: {pose['why']}): the "
                     "object hangs tilted; its centre is still aimed at the container centre")
    if best["centre_err"] > 0.001:
        notes.append(f"the object centre lands {_mm(best['centre_err'])} from the container centre")
    if content_top > -_INF:
        notes.append(f"things already in the container reach z = {content_top:.4f} m")
    return ReleasePlan(ok=True, reason="", tip_above=tip_above, tip_release=tip_release, pitch_deg=pitch,
                       roll_deg=roll, open_raw=int(open_raw),
                       drop_height=round(float(best["z_rel"] - hang - floor_z), 5), notes=notes)


# ======================================================================================= demo (no hardware)
if __name__ == "__main__":
    import copy
    import time

    import robot
    import sim_bus
    from fdtypes import Scene, SceneObject

    def _obj(i, kind, color, center, yaw_deg, size, z0, inner=None):
        yaw = math.radians(yaw_deg)
        fp = [tuple(v) for v in _rect(center, yaw, size[0], size[1])] if kind != "cylinder" else \
            [(center[0] + size[0] / 2 * math.cos(a), center[1] + size[0] / 2 * math.sin(a))
             for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
        return SceneObject(id=i, kind=kind, color=color, center=tuple(center), yaw=yaw, size=tuple(size),
                           z_bottom=z0, footprint=fp, pixel_bbox=(0, 0, 0, 0), confidence=1.0, inner=inner)

    tz = -0.0235
    box = _obj(9, "container", "black", (0.262, 0.085), 94.0, (0.111, 0.100, 0.030), tz,
               inner=dict(center=(0.262, 0.085), yaw=math.radians(94.0), size=(0.105, 0.094), floor_z=tz + 0.003,
                          rim_z=tz + 0.030))
    blue = _obj(1, "box", "blue", (0.2783, -0.0663), 6.8, (0.0446, 0.0204, 0.0179), tz)
    ivory = _obj(2, "cylinder", "ivory", (0.2745, -0.0188), 0.0, (0.0258, 0.0258, 0.0216), tz)
    sc = Scene(objects=[blue, ivory, box], table_z=tz)
    sim_arm = robot.Arm(bus=sim_bus.SimBus(), cfg=copy.deepcopy(robot.load_config())).open()  # never the real bus
    t0 = time.perf_counter()
    found = plan_grasp(blue, sc, sim_arm)
    dt = time.perf_counter() - t0
    print(f"blue block of 2026-11-11 00:30 with the ivory cylinder beside it: {len(found)} candidates, {dt * 1e3:.1f} ms")
    for cnd in found:
        print(f"  ok={cnd.ok!s:5} azimuth {cnd.jaw_azimuth_deg:7.1f} roll {cnd.roll_deg:7.1f} pitch {cnd.pitch_deg:4.0f} "
              f"standoff {cnd.standoff * 1e3:4.1f} mm score {cnd.score:6.1f}  {cnd.reason or '; '.join(cnd.notes)}")
    if found[0].ok:
        print("release:", plan_release(blue, box, found[0], sim_arm, scene=sc).to_dict())
