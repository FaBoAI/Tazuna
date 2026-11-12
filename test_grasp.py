"""Tests of grasp.py. No hardware: the arm is robot.Arm on sim_bus.SimBus, scenes are built by hand.

Run:  /home/jetson/miniconda/envs/lerobot/bin/python test_grasp.py          (all tests)
      /home/jetson/miniconda/envs/lerobot/bin/python test_grasp.py real      (tests whose name contains "real")

Ground truth that does not come from grasp.py itself:
- the three objects really grasped on 2026-11-10/11 (testdata/frames.json + runs/20261110_230151/events.jsonl),
- the official SO-101 meshes in refs/urdf/assets: MeshTruth below places the real jaw geometry at a planned
  pose and checks it against the scene with code that shares nothing with the planner's gripper model.
"""

import copy
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# The scenes and frames used here were measured with the calibration of 2026-11-10 (before the
# shoulder_lift zero was corrected on 2026-11-11). Pin that calibration: re-calibrating the real arm
# must not change these tests (and they must not depend on it).
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "calib_20261110", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "calib_20261110", "camera_model.json")
# The gripper in these tests is bare: they never read the tool file of the tree (tooling.py), so they pass
# whether or not a tool is attached to the real arm, and nobody has to `fd tool detach` to run them.
os.environ["FD_TOOL"] = os.path.join(HERE, "testdata", "no_tool_attached.json")
sys.path.insert(0, HERE)

import gripper_model as gm  # noqa: E402
import kinematics as K  # noqa: E402
import robot  # noqa: E402
import sim_bus  # noqa: E402
from fdtypes import GraspCandidate, ReleasePlan, Scene, SceneObject  # noqa: E402

import grasp as G  # noqa: E402

PY = sys.executable
TABLE = -0.0235
TOUCH = 2050
TPD = 4096.0 / 360.0
ASSETS = os.path.join(HERE, "refs", "urdf", "assets")
STATS = {}  # numbers worth reporting, printed at the end

_ARM = None


def arm():
    """The simulated arm (never the real bus), with an in-memory copy of the live configuration."""
    global _ARM
    if _ARM is None:
        bus = sim_bus.SimBus()
        assert type(bus).__module__ == "sim_bus"
        _ARM = robot.Arm(bus=bus, cfg=copy.deepcopy(robot.load_config())).open()
    return _ARM


# ------------------------------------------------------------------------------------- scene builders
def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def rect(center, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        out.append((center[0] + sx * length / 2 * c - sy * width / 2 * s,
                    center[1] + sx * length / 2 * s + sy * width / 2 * c))
    return out


def box(i, color, center, yaw_deg, size, z0=TABLE, **kw):
    yaw = math.radians(yaw_deg)
    return SceneObject(id=i, kind=kw.pop("kind", "box"), color=color, center=tuple(center), yaw=yaw,
                       size=tuple(size), z_bottom=z0, footprint=rect(center, yaw, size[0], size[1]),
                       pixel_bbox=(0, 0, 0, 0), confidence=0.9, **kw)


def cyl(i, color, center, diameter, height, z0=TABLE, **kw):
    fp = [(center[0] + diameter / 2 * math.cos(a), center[1] + diameter / 2 * math.sin(a))
          for a in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)]
    return SceneObject(id=i, kind="cylinder", color=color, center=tuple(center), yaw=0.0,
                       size=(diameter, diameter, height), z_bottom=z0, footprint=fp, pixel_bbox=(0, 0, 0, 0),
                       confidence=0.9, **kw)


def container(i, center, yaw_deg, outer, rim_h, wall=0.003, z0=TABLE, color="black", with_inner=True):
    yaw = math.radians(yaw_deg)
    inner = dict(center=tuple(center), yaw=yaw, size=(outer[0] - 2 * wall, outer[1] - 2 * wall),
                 floor_z=z0 + wall, rim_z=z0 + rim_h) if with_inner else None
    return SceneObject(id=i, kind="container", color=color, center=tuple(center), yaw=yaw,
                       size=(outer[0], outer[1], rim_h), z_bottom=z0, footprint=rect(center, yaw, outer[0], outer[1]),
                       pixel_bbox=(0, 0, 0, 0), confidence=0.9, inner=inner)


def black_box(i=9, **kw):
    """The black box of the rig: centre about (0.262, 0.085), outer 111 x 100 mm, rim 30 mm, yaw 94 deg."""
    return container(i, (0.262, 0.085), 94.0, (0.111, 0.100), 0.030, **kw)


# the three objects really grasped (testdata/frames.json "truth")
def red_block(i=1):
    return box(i, "red", (0.2541, -0.0373), 97.1, (0.040, 0.023, 0.018))


def white_cap(i=2):
    return cyl(i, "white", (0.2711, -0.0751), 0.030, 0.0148)


def blue_block(i=3):
    return box(i, "blue", (0.2783, -0.0663), 6.8, (0.0446, 0.0204, 0.0179))


def ivory_cylinder(i=4):
    return cyl(i, "ivory", (0.2745, -0.0188), 0.0258, 0.0216)


def by_azimuth(cands, az, tol=1.0):
    hits = [c for c in cands if abs(wrap(c.jaw_azimuth_deg - az)) <= tol]
    assert len(hits) == 1, f"{len(hits)} candidates at azimuth {az}: {[c.jaw_azimuth_deg for c in cands]}"
    return hits[0]


def describe(cands):
    return "\n".join(f"   ok={c.ok!s:5} az {c.jaw_azimuth_deg:7.1f} roll {c.roll_deg:7.1f} pitch {c.pitch_deg:5.1f} "
                     f"standoff {c.standoff * 1e3:4.1f} cf {c.clearance_fixed * 1e3:6.1f} cm {c.clearance_moving * 1e3:6.1f} "
                     f"score {c.score:6.1f} | {c.reason or '; '.join(c.notes)}" for c in cands)


def near(a, b, tol):
    return abs(a - b) <= tol


FK_IDLE_US = 57.0  # one kinematics.fk call on this Jetson with nothing else running (measured 2026-11-11 01:20)


def machine_slowdown():
    """How much slower than idle the machine is right now (other jobs share the CPU tonight): the time of
    kinematics.fk, which is plain numpy code that nobody is editing, against its idle time."""
    q = np.zeros(5)
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(200):
            K.fk(q)
        best = min(best, (time.perf_counter() - t0) / 200.0)
    return max(1.0, best * 1e6 / FK_IDLE_US)


def contact_q(c):
    """Jaw angle at rigid contact implied by a candidate's expected stall position."""
    return gm.Q_CLOSED + (c.expected_raw + G.DEFAULTS["stall_offset_ticks"] - TOUCH) / TPD


def open_q(raw):
    return gm.Q_CLOSED + (raw - TOUCH) / TPD


# ------------------------------------------------------------------------------------- mesh ground truth
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _tf(xyz, rpy):
    t = np.eye(4)
    t[:3, :3] = _rpy(*rpy)
    t[:3, 3] = xyz
    return t


def _load_stl(name):
    with open(os.path.join(ASSETS, name), "rb") as f:
        data = f.read()
    n = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    assert len(data) == 84 + 50 * n, name
    rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(float)  # (n, 3, 3) metres


def _sample(tri, per_mm2, rng):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) * 1e6
    n = np.maximum(1, np.ceil(area * per_mm2)).astype(int)
    idx = np.repeat(np.arange(len(tri)), n)
    s, r2 = np.sqrt(rng.random(len(idx))), rng.random(len(idx))
    pts = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
    return np.vstack([pts, tri.reshape(-1, 3)[::3]])


def _apply(t, p):
    return p @ t[:3, :3].T + t[:3, 3]


class MeshTruth:
    """The gripper's real surface (URDF visual meshes, so101_new_calib.urdf lines 256-340) as sample points."""

    def __init__(self):
        rng = np.random.default_rng(7)
        follower = _apply(_tf((8.32667e-17, -0.000218214, 0.000949706), (-3.14159, -5.55112e-17, 0.0)),
                          _sample(_load_stl("wrist_roll_follower_so101_v1.stl"), 0.2, rng))
        motor = _apply(_tf((0.0077, 0.0001, -0.0234), (-1.5708, -5.19179e-17, -1.66533e-16)),
                       _sample(_load_stl("sts3215_03a_v1.stl"), 0.1, rng))
        self.fixed_gl = np.vstack([follower, motor])  # gripper_link coordinates
        self.jaw_local = _apply(_tf((0.0, 0.0, 0.0189), (0.0, 0.0, 0.0)),
                                _sample(_load_stl("moving_jaw_so101_v1.stl"), 0.2, rng))  # jaw-joint coordinates
        self.t_jaw = _tf((0.0202, 0.0188, -0.0234), (1.5708, -5.24284e-08, -1.41553e-15))
        self.gl_to_gf = np.linalg.inv(K.tool_offset())  # gripper_link -> gripper_frame_link coordinates

    def fixed(self, q):
        return _apply(K.gripper_link_pose(q), self.fixed_gl)

    def jaw_gl(self, q_jaw_deg):
        c, s = math.cos(math.radians(q_jaw_deg)), math.sin(math.radians(q_jaw_deg))
        rz = np.eye(4)
        rz[0, 0], rz[0, 1], rz[1, 0], rz[1, 1] = c, -s, s, c
        return _apply(self.t_jaw @ rz, self.jaw_local)

    def jaw(self, q, q_jaw_deg):
        return _apply(K.gripper_link_pose(q), self.jaw_gl(q_jaw_deg))


_MESH = None


def mesh():
    global _MESH
    if _MESH is None:
        if not os.path.isdir(ASSETS):
            raise Skip(f"mesh ground truth not available ({ASSETS} missing)")
        _MESH = MeshTruth()
    return _MESH


class Skip(Exception):
    pass


def footprint_distance(obj, xy):
    """Signed distance (negative inside) of points to an object's footprint; written independently of
    grasp.py: discs analytically, convex polygons by half planes + corner distances; a container with an
    inner rectangle is hollow (distance to the nearest wall face)."""
    xy = np.asarray(xy, dtype=float)
    if obj.kind == "cylinder":
        return np.hypot(xy[:, 0] - obj.center[0], xy[:, 1] - obj.center[1]) - obj.size[0] / 2.0

    def convex(poly):
        poly = np.asarray(poly, dtype=float)
        area = 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                         for i in range(len(poly)))
        if area < 0:
            poly = poly[::-1]
        inside_depth = np.full(len(xy), np.inf)
        outside = np.full(len(xy), np.inf)
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            e = b - a
            ln = float(np.hypot(*e))
            nrm = np.array([e[1], -e[0]]) / ln  # outward normal of a counter-clockwise polygon
            d = (xy - a) @ nrm
            inside_depth = np.minimum(inside_depth, -d)
            t = np.clip((xy - a) @ e / (ln * ln), 0.0, 1.0)
            outside = np.minimum(outside, np.hypot(*(xy - (a + t[:, None] * e)).T))
        return np.where(inside_depth >= 0.0, -inside_depth, outside)

    d = convex(obj.footprint)
    if obj.kind == "container" and obj.inner is not None:
        inn = obj.inner
        d = np.maximum(d, -convex(rect(inn["center"], inn["yaw"], inn["size"][0], inn["size"][1])))
    return d


def mesh_clearances(c, target, others, z_margin=0.0, n_jaw=9, tip=None):
    """Mesh-based truth for a candidate: smallest distance of the fixed jaw (+ housing) and of the moving jaw
    (at n_jaw angles from contact to the pre-grasp opening) to every other object, counting only mesh points
    not above the object's top (+ z_margin); lowest mesh point; distances to the target itself."""
    m = mesh()
    tip = c.tip_grasp if tip is None else tip
    q = arm().solve(np.array(tip, dtype=float), c.pitch_deg, c.roll_deg)
    fixed = m.fixed(q)
    q_c, q_o = contact_q(c), open_q(c.open_raw)
    jaws = [m.jaw(q, a) for a in np.linspace(q_c, q_o, n_jaw)]
    moving = np.vstack(jaws)
    out = dict(fixed=np.inf, moving=np.inf, z_min=float(min(fixed[:, 2].min(), moving[:, 2].min())),
               z_min_moving=float(moving[:, 2].min()))
    for ob in others:
        top = ob.z_top if ob.inner is None else max(ob.z_top, ob.inner["rim_z"])
        for name, pts in (("fixed", fixed), ("moving", moving)):
            sel = pts[pts[:, 2] <= top + z_margin]
            if len(sel):
                out[name] = min(out[name], float(footprint_distance(ob, sel[:, :2]).min()))
    sel = fixed[fixed[:, 2] <= target.z_top]
    out["self_fixed"] = float(footprint_distance(target, sel[:, :2]).min()) if len(sel) else np.inf
    sel = jaws[-1][jaws[-1][:, 2] <= target.z_top]
    out["self_open"] = float(footprint_distance(target, sel[:, :2]).min()) if len(sel) else np.inf
    return out


# ===================================================================================== the three real grasps
def test_real_red_block():
    """(a) 2026-11-10 23:49: red block 40 x 23 x 18 mm, grasped across its 23 mm width, jaw azimuth 7.1 deg,
    roll 21 deg, standoff 12 mm, fingertip (0.2318, -0.0401, -0.0195); stalled at raw 2194, pre-grasp 2518."""
    tgt = red_block()
    cands = G.plan_grasp(tgt, Scene([tgt, black_box()], TABLE), arm())
    best = cands[0]
    assert best.ok and best.reason == "", describe(cands)
    assert abs(wrap(best.jaw_azimuth_deg - 7.1)) <= 3.0, describe(cands)
    assert near(best.roll_deg, 21.0, 3.0) and best.pitch_deg == 90.0, describe(cands)
    assert near(best.standoff, 0.012, 0.002) and near(best.width, 0.023, 1e-9)
    assert np.linalg.norm(np.array(best.tip_grasp) - (0.2318, -0.0401, -0.0195)) <= 0.002, best.tip_grasp
    assert near(best.tip_grasp[2], TABLE + 0.004, 1e-6)
    assert best.tip_hover[:2] == best.tip_grasp[:2] and near(best.tip_hover[2], TABLE + 0.018 + 0.030, 1e-6)
    assert near(best.expected_raw, 2194, 20), best.expected_raw
    assert near(best.open_raw, 2518, 15), best.open_raw
    assert math.isinf(best.clearance_fixed) and math.isinf(best.clearance_moving)
    u = np.array([math.cos(math.radians(best.jaw_azimuth_deg)), math.sin(math.radians(best.jaw_azimuth_deg))])
    assert np.allclose(best.object_offset, (0.0115 - 0.001) * u, atol=1e-4), best.object_offset
    # the same faces from the other side need the wrist beyond its limit (v1: "no wrist roll reaches ...")
    opp = by_azimuth(cands, -172.9)
    assert not opp.ok and "wrist roll" in opp.reason and "limits" in opp.reason, opp.reason
    assert all(c.score <= best.score for c in cands)
    STATS["red block"] = f"az {best.jaw_azimuth_deg:.1f} roll {best.roll_deg:.1f} standoff {best.standoff * 1e3:.1f} mm " \
                         f"tip {best.tip_grasp} expected_raw {best.expected_raw} open_raw {best.open_raw}"


def test_real_white_cap():
    """(b) 2026-11-11 00:05: white cap d 30 x 15 mm, radial jaw azimuth -17.9 deg, roll 2.5 deg,
    fingertip (0.2464, -0.0671, -0.0195); stalled at raw 2263, pre-grasp 2580."""
    tgt = white_cap()
    inside = box(5, "red", (0.262, 0.085), 10.0, (0.040, 0.023, 0.018), z0=TABLE + 0.003, in_container=9)
    cands = G.plan_grasp(tgt, Scene([tgt, black_box(), inside], TABLE), arm())
    best = cands[0]
    assert len(cands) == 12 and best.ok, describe(cands)
    assert abs(wrap(best.jaw_azimuth_deg - (-17.9))) <= 3.0, describe(cands)
    radial = math.degrees(math.atan2(tgt.center[1], tgt.center[0] - 0.0388353))
    assert abs(wrap(best.jaw_azimuth_deg - radial)) < 0.01  # radial first
    assert near(best.roll_deg, 2.5, 3.0) and best.pitch_deg == 90.0
    assert near(best.standoff, 0.012, 0.002)
    assert np.linalg.norm(np.array(best.tip_grasp) - (0.2464, -0.0671, -0.0195)) <= 0.002, best.tip_grasp
    assert near(best.expected_raw, 2263, 20), best.expected_raw
    assert near(best.open_raw, 2580, 15), best.open_raw
    azs = sorted(round(wrap(c.jaw_azimuth_deg - radial)) % 360 for c in cands)
    assert azs == list(range(0, 360, 30)), azs
    STATS["white cap"] = f"az {best.jaw_azimuth_deg:.1f} roll {best.roll_deg:.1f} standoff {best.standoff * 1e3:.1f} mm " \
                         f"tip {best.tip_grasp} expected_raw {best.expected_raw} open_raw {best.open_raw}"


def test_real_blue_block():
    """(c) 2026-11-11 00:32: blue block 44.6 x 20.4 x 17.9 mm with the ivory cylinder 24.6 mm beside it:
    grasped across the 20 mm width with the FIXED jaw in that gap (azimuth -83.2 deg, roll -67 deg, standoff
    7 mm, fingertip (0.2764, -0.0502, -0.0195)); the opposite direction would sweep the opened moving jaw
    into the cylinder; along the 44.6 mm length the grasp is marginal and must rank below."""
    tgt, ivory = blue_block(), ivory_cylinder()
    cands = G.plan_grasp(tgt, Scene([tgt, ivory, black_box()], TABLE), arm())
    best = cands[0]
    assert best.ok, describe(cands)
    assert abs(wrap(best.jaw_azimuth_deg - (-83.2))) <= 3.0, describe(cands)
    assert near(best.roll_deg, -67.0, 3.0) and best.pitch_deg == 90.0, describe(cands)
    assert near(best.standoff, 0.007, 0.002), best.standoff
    assert np.linalg.norm(np.array(best.tip_grasp) - (0.2764, -0.0502, -0.0195)) <= 0.002, best.tip_grasp
    assert 0.003 <= best.clearance_fixed <= 0.009 and math.isinf(best.clearance_moving), describe(cands)
    assert any("standoff reduced" in n and "ivory" in n for n in best.notes), best.notes
    assert near(best.expected_raw, 2194, 20), best.expected_raw
    assert near(best.open_raw, 2495, 15), best.open_raw
    # the gap really was 24.6 mm
    u = np.array([math.cos(math.radians(-83.2)), math.sin(math.radians(-83.2))])
    face = np.array(tgt.center) - 0.0102 * u
    gap = (np.array(ivory.center) - face) @ (-u) - ivory.size[0] / 2
    assert near(gap, 0.0246, 0.0005), gap
    # opposite direction: the opened moving jaw would sweep into the cylinder
    opp = by_azimuth(cands, 96.8)
    assert not opp.ok and "moving jaw" in opp.reason and "ivory cylinder #4" in opp.reason, opp.reason
    # along the length: marginal, ranked below
    for az in (6.8, -173.2):
        c = by_azimuth(cands, az)
        assert near(c.width, 0.0446, 1e-9)
        assert cands.index(c) > 0 and c.score < best.score - 20.0, describe(cands)
        if c.ok:
            assert any("marginal" in n for n in c.notes), c.notes
        else:
            assert c.reason
    assert by_azimuth(cands, 6.8).ok, describe(cands)
    STATS["blue block"] = f"az {best.jaw_azimuth_deg:.1f} roll {best.roll_deg:.1f} standoff {best.standoff * 1e3:.1f} mm " \
                          f"tip {best.tip_grasp} clearance_fixed {best.clearance_fixed * 1e3:.1f} mm " \
                          f"expected_raw {best.expected_raw} open_raw {best.open_raw}; +96.8 deg: {opp.reason}"


def test_real_cases_against_meshes():
    """The three real grasps, judged by the real jaw meshes instead of the planner's model."""
    cases = ((red_block(), [black_box()]), (white_cap(), [black_box()]), (blue_block(), [ivory_cylinder(), black_box()]))
    for tgt, others in cases:
        cands = G.plan_grasp(tgt, Scene([tgt] + others, TABLE), arm())
        best = cands[0]
        mc = mesh_clearances(best, tgt, others)
        assert mc["fixed"] > 0.002 and mc["moving"] > 0.002, (tgt.color, mc)
        assert mc["z_min"] > TABLE + 0.002, mc  # the moving nose is 1.3 mm longer than the fingertip point
        # fixed pad `standoff` in front of the object's face, opened jaw well beyond the far face
        assert near(mc["self_fixed"], best.standoff, 0.0012), (tgt.color, mc["self_fixed"], best.standoff)
        assert mc["self_open"] > 0.012, mc
        if tgt.color == "blue":
            # reported clearance (uses z_margin 3 mm) vs mesh truth with the same margin: within 1 mm
            assert near(mesh_clearances(best, tgt, others, z_margin=0.003)["fixed"], best.clearance_fixed, 0.001)
            assert near(mc["fixed"], 0.0246 - best.standoff - 0.0114, 0.001), mc  # 24.6 gap - standoff - finger
            opp = by_azimuth(cands, 96.8)
            assert mesh_clearances(opp, tgt, others)["moving"] < 0.0, "the rejected direction really collides"


# ===================================================================================== closed-form kinematics
def test_roll_closed_form_matches_fk():
    """psi = roll - pan - 2.789 deg (and its tilted generalisation), verified through kinematics.jaw_open_axis."""
    a = arm()
    tool, lim = a.tip_tool(), a.q_limits()
    lat = G._tool_model(tool)[0]
    rng = np.random.default_rng(11)
    worst, n = {}, {}
    for _ in range(1500):
        x, y, z = rng.uniform(0.12, 0.30), rng.uniform(-0.15, 0.15), rng.uniform(-0.02, 0.05)
        psi = rng.uniform(-180, 180)
        for pitch in (90.0, 80.0, 70.0, 100.0):
            roll = G._roll_for_azimuth(psi, x, y, pitch, lat)
            assert -180.0 <= roll < 180.0
            q, info = K.ik((x, y, z), pitch, math.radians(roll), tool=tool, limits=lim)
            if not info["ok"]:
                continue
            o = K.jaw_open_axis(K.fk(q))
            err = abs(wrap(math.degrees(math.atan2(o[1], o[0])) - psi))
            worst[pitch] = max(worst.get(pitch, 0.0), err)
            n[pitch] = n.get(pitch, 0) + 1
            if pitch == 90.0:  # the textbook form
                assert abs(wrap(roll - math.degrees(q[0]) - gm.WRIST_ROLL_SKEW_DEG - psi)) < 0.01
                assert abs(o[2]) < 1e-4
    assert all(v > 300 for v in n.values()), n
    assert max(worst.values()) < 0.01, worst
    STATS["closed-form roll"] = f"max azimuth error vs FK {max(worst.values()):.5f} deg over {sum(n.values())} poses"


def test_closed_form_ik_matches_kinematics_ik():
    """grasp._ik_fast (one pose) and grasp._pitch_feasible (all pitches at once) against kinematics.ik:
    same verdict everywhere, same joint angles, same wording of the refusals."""
    a = arm()
    tool, lim = a.tip_tool(), a.q_limits()
    model = G._tool_model(tool)
    pitches = np.arange(60.0, 121.0, 1.0)
    total = bad = solved = 0
    worst_q = 0.0
    for x in np.arange(0.08, 0.40, 0.03):
        for y in (-0.12, 0.0, 0.07):
            for z in (-0.0195, 0.03, 0.08):
                for roll in (-120.0, 3.0, 95.0):
                    feas = G._pitch_feasible((x, y), [z], roll, pitches, model, lim)[0]
                    for p, f in zip(pitches, feas):
                        q_ref, info = K.ik((x, y, z), float(p), math.radians(roll), tool=tool, limits=lim)
                        q, reason = G._ik_fast((x, y, z), float(p), roll, model, lim)
                        total += 1
                        bad += int(info["ok"] != bool(f)) + int(info["ok"] != (q is not None))
                        if info["ok"] and q is not None:
                            solved += 1
                            worst_q = max(worst_q, float(np.abs(q - q_ref).max()))
                        elif q is None and not info["ok"]:
                            assert reason.split("=")[0].split(" (")[0] == info["reason"].split("=")[0].split(" (")[0], \
                                (reason, info["reason"])
    assert total > 5000 and solved > 1500 and bad == 0, (bad, solved, total)
    assert worst_q < 1e-5, worst_q
    # the tool model reproduces kinematics.ik's per-call FK quantities for any roll
    for roll in (-150.0, -33.0, 0.0, 61.0, 140.0):
        t0 = K.fk(np.array([0.0, 0.0, 0.0, 0.0, math.radians(roll)]), tool)
        u3x, u3z, e = G._tool_plane(roll, model)
        assert near(e, t0[1, 3], 1e-12) and near(u3x, t0[0, 3] - G._X_PAN - G._WR[0], 1e-12)
        assert near(u3z, t0[2, 3] - G._WR[1], 1e-12)
    STATS["closed-form IK"] = (f"{total} poses: 0 verdicts different from kinematics.ik, joint angles within "
                               f"{worst_q:.1e} rad on {solved} solutions")


def test_contact_angle_closed_form_window():
    """The pad-corner formula is only used where it equals the profile model."""
    worst = 0.0
    for w in np.arange(16.0, 45.01, 1.0):
        for e in np.arange(10.0, 15.01, 1.0):
            worst = max(worst, abs(gm.q_contact_pad_corner(w) - gm.contact_angle(w, gm.FIXED_TIP_Z - e, step=1.0)))
    assert worst < 0.02, worst
    assert near(G._contact_q(0.023, 0.0135), 5.69, 0.02) and near(G._contact_q(0.030, 0.0103), 11.19, 0.02)
    # outside the window the profile model is used (a low object: the pad face, not its corner, makes contact)
    assert near(G._contact_q(0.030, 0.0055), gm.contact_angle(30.0, gm.FIXED_TIP_Z - 5.5), 0.02)
    assert abs(G._contact_q(0.030, 0.0055) - gm.q_contact_pad_corner(30.0)) > 0.2


def test_min_distance_is_exact():
    """The planner's fast minimum distance (half-plane bound + exact re-check) against the plain formula and
    against the independent implementation of this file, for boxes, hollow containers and discs."""
    rng = np.random.default_rng(5)
    n_pos = 0
    for trial in range(1500):
        kind = trial % 3
        c = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))
        if kind == 0:
            o = box(1, "g", c, rng.uniform(-180, 180), (rng.uniform(0.01, 0.08), rng.uniform(0.01, 0.05), 0.03))
        elif kind == 1:
            o = container(1, c, rng.uniform(-180, 180), (rng.uniform(0.06, 0.12), rng.uniform(0.05, 0.11)), 0.03,
                          wall=rng.uniform(0.002, 0.006))
        else:
            o = cyl(1, "g", c, rng.uniform(0.01, 0.06), 0.03)
        ob = G._Obstacle(o)
        pts = rng.uniform(-0.12, 0.12, 2) + rng.uniform(-0.03, 0.03, (int(rng.integers(1, 300)), 2))
        full = ob.sdist(pts)
        assert np.abs(full - footprint_distance(o, pts)).max() < 1e-9
        fast = ob.min_sdist(np.ascontiguousarray(pts[:, 0]), np.ascontiguousarray(pts[:, 1]))
        if full.min() > 0:
            n_pos += 1
            assert abs(fast - full.min()) < 1e-12, (kind, fast, full.min())
        else:  # reaching into the material: the sign is exact, the depth may be understated at a corner
            assert full.min() - 1e-12 <= fast <= 0.0, (kind, fast, full.min())
        if ob.outer is not None:
            for p in pts[:3]:
                assert abs(ob.outer.sdist_point(float(p[0]), float(p[1])) - ob.outer.sdist(p[None, :])[0]) < 1e-12
    assert n_pos > 700
    # a non-convex outline (L shape) falls back to the plain formula and is still right
    ell = SceneObject(id=1, kind="unknown", color="other", center=(0.0, 0.0), yaw=0.0, size=(0.04, 0.04, 0.02),
                      z_bottom=TABLE, footprint=[(0, 0), (0.04, 0), (0.04, 0.01), (0.01, 0.01), (0.01, 0.04), (0, 0.04)],
                      pixel_bbox=(0, 0, 0, 0), confidence=0.5)
    ob = G._Obstacle(ell)
    assert not ob.outer.convex
    probe = np.array([[0.02, 0.02], [0.005, 0.02], [0.03, 0.005], [0.05, 0.005], [0.02, 0.012]])
    assert np.allclose(ob.sdist(probe), [0.01, -0.005, -0.005, 0.01, 0.002], atol=1e-12)
    assert near(ob.min_sdist(probe[:, 0].copy(), probe[:, 1].copy()), -0.005, 1e-12)


def test_clearance_matches_brute_force():
    """_clearance (objects visited nearest bounding box first, points culled by the object's box grown by the
    best distance so far) against the plain definition: for every object, the exact distance of every
    sample point that is not above its top (+ margin); the smallest one, if within _NEAR."""
    rng = np.random.default_rng(11)
    z_margin = 0.003
    n_hit = n_free = n_none = 0
    for trial in range(1500):
        obs = []
        spread = rng.uniform(0.03, 0.12)
        for k in range(int(rng.integers(1, 7))):
            c = (rng.uniform(-spread, spread), rng.uniform(-spread, spread))
            h = rng.uniform(0.005, 0.06)
            kind = int(rng.integers(0, 3))
            if kind == 0:
                o = box(10 + k, "g", c, rng.uniform(-180, 180), (rng.uniform(0.01, 0.08), rng.uniform(0.01, 0.05), h))
            elif kind == 1:
                o = container(10 + k, c, rng.uniform(-180, 180), (rng.uniform(0.06, 0.12), rng.uniform(0.05, 0.11)), h,
                              wall=rng.uniform(0.002, 0.006))
            else:
                o = cyl(10 + k, "g", c, rng.uniform(0.01, 0.06), h)
            obs.append(G._Obstacle(o))
        n = int(rng.integers(1, 400))
        xy = rng.uniform(-0.1, 0.1, 2) + rng.uniform(-1.0, 1.0, (n, 2)) * rng.uniform(0.003, 0.05, 2)
        z = TABLE + rng.uniform(0.0, 0.07, n)
        pts = G._Pts(np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1]), z)
        seen = []
        d, who = G._clearance(pts, obs, z_margin, near=seen)
        best, best_ob, within, uncut = np.inf, None, [], {}
        for ob in obs:  # the definition, with the plain exact distance
            sel = z <= ob.z_top + z_margin
            if sel.any():
                dd = float(ob.sdist(xy[sel]).min())
                if dd <= G._NEAR:
                    within.append((dd, ob))
                    uncut[id(ob)] = ob.min_sdist(pts.x[sel], pts.y[sel])  # the planner's distance, nothing culled
                    if dd < best:
                        best, best_ob = dd, ob
        assert all(any(ob is s for s in seen) for _dd, ob in within), "an object within _NEAR is missing from `near`"
        if best_ob is None:
            assert d == np.inf and who is None, (trial, d)
            n_none += 1
            continue
        # culling never changes the number: it equals the smallest per-object distance over ALL the points
        assert abs(d - min(uncut.values())) < 1e-12 and abs(uncut[id(who)] - d) < 1e-12, (trial, d, uncut)
        if best > 0.0:
            assert abs(d - best) < 1e-12 and who is best_ob, (trial, d, best)
            n_free += 1
        else:  # reaching into an object: the sign is exact, the depth may be understated next to a corner
            assert best - 1e-12 <= d <= 0.0, (trial, d, best)
            assert any(ob is who and dd <= 0.0 for dd, ob in within), trial
            n_hit += 1
        assert G._clearance(pts, obs, z_margin) == (d, who)  # the same without the `near` list
    assert n_hit > 150 and n_free > 300 and n_none > 100, (n_hit, n_free, n_none)
    STATS["clearance vs brute force"] = (f"1500 random clouds x 1-6 objects: {n_free} free (equal within 1e-12 m), "
                                         f"{n_hit} touching (sign equal, value equal to the unculled one), {n_none} with nothing within 30 mm")


# ===================================================================================== candidates are consistent
def _check_candidate(c, tgt, cfg_p=G.DEFAULTS):
    a = arm()
    assert isinstance(c, GraspCandidate)
    assert (c.ok and c.reason == "") or (not c.ok and len(c.reason) > 10), (c.ok, c.reason)
    json.dumps(c.to_dict())
    numbers = [c.jaw_azimuth_deg, c.roll_deg, c.pitch_deg, c.width, c.standoff, c.score, *c.tip_grasp, *c.tip_hover,
               *c.object_offset]
    assert all(type(v) is float and math.isfinite(v) for v in numbers), numbers  # never NaN, also when rejected
    assert not math.isnan(c.clearance_fixed) and not math.isnan(c.clearance_moving)
    assert isinstance(c.open_raw, int) and isinstance(c.expected_raw, int)
    assert isinstance(c.notes, list) and all(isinstance(n, str) for n in c.notes)
    if not c.ok:
        assert c.score == 0.0
        return
    assert c.score >= 1.0
    assert cfg_p["min_standoff"] - 1e-9 <= c.standoff <= cfg_p["standoff"] + 1e-9
    assert c.clearance_fixed >= cfg_p["min_clear"] and c.clearance_moving >= cfg_p["min_clear"]
    assert c.open_raw > c.expected_raw + 150 and 2023 <= c.open_raw <= 3623
    assert c.expected_raw > TOUCH + 50
    assert 70.0 <= c.pitch_deg <= 110.0
    lim = a.q_limits()["wrist_roll"]
    assert math.degrees(lim[0]) <= c.roll_deg <= math.degrees(lim[1])
    assert c.tip_hover[:2] == c.tip_grasp[:2] and c.tip_hover[2] >= tgt.z_top + 0.015 - 1e-5  # (rounded to 10 um)
    assert near(c.tip_grasp[2], tgt.z_bottom + max(0.004, tgt.size[2] - 0.015), 1e-5)
    q_open = open_q(c.open_raw)
    for tip in (c.tip_grasp, c.tip_hover):  # reachable and safe according to the arm itself ...
        q = a.solve(np.array(tip), c.pitch_deg, c.roll_deg)
        a.check_q(q)
        # ... also with every opening the gripper goes through there (the arm's floor check of the jaw)
        a.check_q(q, jaw_q_deg=np.linspace(open_q(c.expected_raw), q_open, 13))
    q = a.solve(np.array(c.tip_grasp), c.pitch_deg, c.roll_deg)
    if hasattr(a, "check_path"):  # what run_path will check before the descent, gripper opening on the way
        a.check_path([a.solve(np.array(c.tip_hover), c.pitch_deg, c.roll_deg), q], 12.0, gripper_goal_raw=c.open_raw)
    t = K.fk(q)
    o = K.jaw_open_axis(t)
    assert abs(wrap(math.degrees(math.atan2(o[1], o[0])) - c.jaw_azimuth_deg)) < 0.5  # jaws open where planned
    u = np.array([math.cos(math.radians(c.jaw_azimuth_deg)), math.sin(math.radians(c.jaw_azimuth_deg))])
    centre = np.array(tgt.center)
    if c.pitch_deg == 90.0:
        # the fixed pad (tool origin) is `standoff` in front of the object's face, centred on the object
        pad = t[:2, 3]
        assert near((centre - pad) @ u - c.width / 2, c.standoff, 3e-4), ((centre - pad) @ u - c.width / 2, c.standoff)
        assert abs((centre - pad) @ np.array([-u[1], u[0]])) < 3e-4
        # once held, the object's centre is object_offset from the fingertip
        held = centre - c.standoff * u
        assert np.allclose(held - np.array(c.tip_grasp[:2]), c.object_offset, atol=3e-4)


def test_candidates_are_consistent():
    scenes = [(red_block(), [black_box()]), (white_cap(), [black_box()]), (blue_block(), [ivory_cylinder(), black_box()]),
              (box(1, "red", (0.34, 0.0), 90.0, (0.040, 0.023, 0.018)), []),
              (cyl(1, "white", (0.33, 0.0), 0.030, 0.0148), []),
              (box(1, "green", (0.20, 0.09), 30.0, (0.035, 0.030, 0.045)), []),
              (cyl(1, "yellow", (0.18, -0.10), 0.020, 0.060), [])]
    n_ok = 0
    for tgt, others in scenes:
        cands = G.plan_grasp(tgt, Scene([tgt] + others, TABLE), arm())
        assert len(cands) >= 4
        assert [c.ok for c in cands] == sorted([c.ok for c in cands], reverse=True)  # ok ones first
        oks = [c.score for c in cands if c.ok]
        assert oks == sorted(oks, reverse=True)  # best first
        for c in cands:
            _check_candidate(c, tgt)
            n_ok += c.ok
    assert n_ok >= 25, n_ok
    tall = G.plan_grasp(scenes[-1][0], Scene([scenes[-1][0]], TABLE), arm())[0]  # 60 mm tall: grip the top 15 mm
    assert tall.ok and near(tall.tip_grasp[2], TABLE + 0.060 - 0.015, 1e-6)


# ===================================================================================== cylinders
def test_cylinder_azimuth_choice():
    cap = cyl(1, "white", (0.25, -0.05), 0.030, 0.0148)
    radial = math.degrees(math.atan2(cap.center[1], cap.center[0] - 0.0388353))
    ur = np.array([math.cos(math.radians(radial)), math.sin(math.radians(radial))])
    # free: radial first, 12 directions; fewer when configured
    cands = G.plan_grasp(cap, Scene([cap], TABLE), arm())
    assert len(cands) == 12 and cands[0].ok and abs(wrap(cands[0].jaw_azimuth_deg - radial)) < 0.01, describe(cands)
    assert "radial" in cands[0].notes[0]
    cfg = copy.deepcopy(arm().cfg)
    cfg["grasp"] = {"cyl_azimuths": 8}
    assert len(G.plan_grasp(cap, Scene([cap], TABLE), arm(), cfg)) == 8
    # a neighbour 15 mm beyond the far side: the radial direction would open the moving jaw into it
    far = cyl(2, "green", tuple(np.array(cap.center) + (0.015 + 0.015 + 0.015) * ur), 0.030, 0.025)
    cands = G.plan_grasp(cap, Scene([cap, far], TABLE), arm())
    rad = by_azimuth(cands, radial)
    assert not rad.ok and "moving jaw" in rad.reason and "green cylinder #2" in rad.reason, rad.reason
    assert cands[0].ok and abs(wrap(cands[0].jaw_azimuth_deg - radial)) >= 29.0, describe(cands)
    assert min(cands[0].clearance_fixed, cands[0].clearance_moving) >= 0.002
    # a neighbour 12 mm in front of the near side: no room for the fixed finger there
    nearby = cyl(2, "green", tuple(np.array(cap.center) - (0.015 + 0.012 + 0.015) * ur), 0.030, 0.025)
    cands = G.plan_grasp(cap, Scene([cap, nearby], TABLE), arm())
    rad = by_azimuth(cands, radial)
    assert not rad.ok and "fixed finger" in rad.reason and "green cylinder #2" in rad.reason, rad.reason
    assert cands[0].ok and abs(wrap(cands[0].jaw_azimuth_deg - radial)) >= 29.0, describe(cands)
    # fenced in on all sides: nothing is feasible, and every candidate says why
    ring = [cyl(10 + k, "green", (cap.center[0] + 0.040 * math.cos(a), cap.center[1] + 0.040 * math.sin(a)), 0.030, 0.025)
            for k, a in enumerate(np.linspace(0, 2 * math.pi, 8, endpoint=False))]
    cands = G.plan_grasp(cap, Scene([cap] + ring, TABLE), arm())
    assert not any(c.ok for c in cands) and all(len(c.reason) > 10 for c in cands), describe(cands)


# ===================================================================================== neighbours
def _block_with_neighbour(gap, side=+1, n_height=0.025):
    """Block (long axis along x) with a 30 x 30 mm neighbour `gap` beside its +y (side=+1) or -y face."""
    tgt = box(1, "red", (0.24, -0.02), 0.0, (0.040, 0.023, 0.018))
    nb = box(2, "green", (0.24, -0.02 + side * (0.0115 + gap + 0.015)), 0.0, (0.030, 0.030, n_height))
    return tgt, nb


def test_neighbour_rejection_each_side():
    for side in (+1, -1):
        az_fixed_in_gap, az_moving_in_gap = -90.0 * side, 90.0 * side
        # 22 mm gap: the fixed finger fits (tightly, with a reduced standoff); the opened moving jaw does not
        tgt, nb = _block_with_neighbour(0.022, side)
        cands = G.plan_grasp(tgt, Scene([tgt, nb], TABLE), arm())
        fx, mv = by_azimuth(cands, az_fixed_in_gap), by_azimuth(cands, az_moving_in_gap)
        assert fx.ok and fx.standoff < 0.008 and 0.002 <= fx.clearance_fixed < 0.005, describe(cands)
        assert any("tight" in n and "fixed finger" in n and "green box #2" in n for n in fx.notes), fx.notes
        assert not mv.ok and "moving jaw" in mv.reason and "sweep into green box #2" in mv.reason, mv.reason
        assert cands[0] is fx or cands[0].score >= fx.score
        # 12 mm gap: neither fits; the reasons name the part and the object
        tgt, nb = _block_with_neighbour(0.012, side)
        cands = G.plan_grasp(tgt, Scene([tgt, nb], TABLE), arm())
        fx, mv = by_azimuth(cands, az_fixed_in_gap), by_azimuth(cands, az_moving_in_gap)
        assert not fx.ok and "fixed finger would hit green box #2" in fx.reason, fx.reason
        assert not mv.ok and "moving jaw" in mv.reason and "green box #2" in mv.reason, mv.reason
        # 45 mm gap: both fit comfortably, nominal standoff
        tgt, nb = _block_with_neighbour(0.045, side)
        cands = G.plan_grasp(tgt, Scene([tgt, nb], TABLE), arm())
        fx, mv = by_azimuth(cands, az_fixed_in_gap), by_azimuth(cands, az_moving_in_gap)
        assert fx.ok and mv.ok and near(fx.standoff, 0.012, 1e-9) and near(mv.standoff, 0.012, 1e-9), describe(cands)
        assert not any("tight" in n for n in fx.notes + mv.notes)
    # a neighbour lower than the fingertips' height above the table is not in the way of anything
    tgt, nb = _block_with_neighbour(0.012, +1, n_height=0.0005)
    cands = G.plan_grasp(tgt, Scene([tgt, nb], TABLE), arm())
    assert by_azimuth(cands, -90.0).ok and by_azimuth(cands, 90.0).ok, describe(cands)


def test_tight_candidates_stay_ok_but_rank_lower():
    tgt, nb = _block_with_neighbour(0.022)
    tight = by_azimuth(G.plan_grasp(tgt, Scene([tgt, nb], TABLE), arm()), -90.0)
    tgt, nb = _block_with_neighbour(0.045)
    free = by_azimuth(G.plan_grasp(tgt, Scene([tgt, nb], TABLE), arm()), -90.0)
    assert tight.ok and free.ok and tight.score < free.score - 15.0, (tight.score, free.score)
    assert tight.clearance_fixed < 0.005 <= free.clearance_fixed
    # the hard minimum cannot be configured away: 1 mm asked, 2 mm (model resolution) enforced
    cfg = copy.deepcopy(arm().cfg)
    cfg["grasp"] = {"min_clear": 0.0}
    tgt, nb = _block_with_neighbour(0.0195)
    c = by_azimuth(G.plan_grasp(tgt, Scene([tgt, nb], TABLE), arm(), cfg), -90.0)
    assert (not c.ok and "fixed finger" in c.reason) or c.clearance_fixed >= 0.002, (c.ok, c.reason, c.clearance_fixed)


def test_container_wall_is_an_obstacle():
    bx = black_box()
    # block 8 mm in front of the box's near end wall, long axis parallel to the wall
    tgt = box(1, "red", (0.262, 0.0100), 0.0, (0.040, 0.023, 0.018))
    cands = G.plan_grasp(tgt, Scene([tgt, bx], TABLE), arm())
    for az, part in ((-90.0, "fixed finger"), (90.0, "moving jaw")):
        c = by_azimuth(cands, az)
        assert not c.ok and part in c.reason and "black container #9" in c.reason, (az, c.reason)
    best = cands[0]
    assert best.ok and near(best.width, 0.040, 1e-9), describe(cands)  # along the length, parallel to the wall
    assert best.clearance_fixed > 0.004 and best.clearance_moving > 0.004, describe(cands)
    # hover above the 30 mm rim next to it, not just above the 18 mm block
    assert best.tip_hover[2] >= TABLE + 0.030 + 0.012 - 1e-6
    # the same block next to a container without a measured interior: treated as solid, same verdicts
    solid = black_box(with_inner=False)
    cands2 = G.plan_grasp(tgt, Scene([tgt, solid], TABLE), arm())
    assert [c.ok for c in sorted(cands2, key=lambda c: c.jaw_azimuth_deg)] == \
           [c.ok for c in sorted(cands, key=lambda c: c.jaw_azimuth_deg)]


def test_push_into_neighbour():
    """Closing pushes the object `standoff` towards the fixed jaw: a neighbour in front of that face but
    beside the finger limits the standoff, or forbids the direction."""
    tgt = box(1, "blue", (0.25, -0.03), 0.0, (0.0446, 0.0204, 0.0179))
    # small post 8 mm in front of the +y face, near its end (16 mm from the finger's centre line)
    post = cyl(2, "green", (0.25 + 0.016, -0.03 + 0.0102 + 0.008 + 0.004), 0.008, 0.030)
    cands = G.plan_grasp(tgt, Scene([tgt, post], TABLE), arm())
    c = by_azimuth(cands, -90.0)  # fixed jaw on the +y side
    assert c.ok and c.standoff <= 0.0075 and any("in the way of the push" in n for n in c.notes), describe(cands)
    post = cyl(2, "green", (0.25 + 0.016, -0.03 + 0.0102 + 0.003 + 0.004), 0.008, 0.030)
    cands = G.plan_grasp(tgt, Scene([tgt, post], TABLE), arm())
    c = by_azimuth(cands, -90.0)
    assert not c.ok and "push the object into green cylinder #2" in c.reason, c.reason
    # a block lying flush against the long side does not block a grasp along the length (it slides along)
    side = box(2, "green", (0.25, -0.03 + 0.0102 + 0.0125), 0.0, (0.0446, 0.025, 0.0179))
    cands = G.plan_grasp(tgt, Scene([tgt, side], TABLE), arm())
    c = by_azimuth(cands, 0.0)
    assert not (not c.ok and "push" in c.reason), c.reason


# ===================================================================================== refusals
def test_unreachable_target():
    for tgt in (box(1, "red", (0.45, 0.0), 0.0, (0.040, 0.023, 0.018)), cyl(1, "white", (0.42, 0.10), 0.030, 0.015),
                box(1, "red", (0.25, 0.35), 20.0, (0.040, 0.023, 0.018))):
        t0 = time.perf_counter()
        cands = G.plan_grasp(tgt, Scene([tgt], TABLE), arm())
        dt = time.perf_counter() - t0
        assert cands and not any(c.ok for c in cands), describe(cands)
        assert all(("out of reach" in c.reason or "wrist roll" in c.reason or "no approach pitch" in c.reason
                    or "refused by the arm" in c.reason) for c in cands), describe(cands)
        assert any("out of reach" in c.reason for c in cands), describe(cands)
        assert dt / machine_slowdown() < 0.05, dt
    # inside the keep-out over the cables: reachable, but the arm's own safety check refuses
    tgt = box(1, "red", (0.24, -0.165), 0.0, (0.040, 0.023, 0.018))
    cands = G.plan_grasp(tgt, Scene([tgt], TABLE), arm())
    assert not any(c.ok for c in cands), describe(cands)
    assert sum("keep-out 'cables'" in c.reason and "refused by the arm" in c.reason for c in cands) >= 3, describe(cands)


def test_too_wide_object():
    big = box(1, "red", (0.24, 0.0), 30.0, (0.070, 0.060, 0.030))
    cands = G.plan_grasp(big, Scene([big], TABLE), arm())
    assert len(cands) == 4 and not any(c.ok for c in cands)
    assert all("too wide" in c.reason and "45.0 mm" in c.reason for c in cands), describe(cands)
    # long but narrow: across the width only
    plank = box(1, "red", (0.24, 0.0), 30.0, (0.060, 0.030, 0.030))
    cands = G.plan_grasp(plank, Scene([plank], TABLE), arm())
    assert [near(c.width, 0.030, 1e-9) for c in cands if c.ok] == [True, True], describe(cands)
    assert all("too wide: 60.0 mm" in c.reason for c in cands if not c.ok) and sum(not c.ok for c in cands) == 2
    wide_cyl = cyl(1, "white", (0.24, 0.0), 0.050, 0.030)
    cands = G.plan_grasp(wide_cyl, Scene([wide_cyl], TABLE), arm())
    assert not any(c.ok for c in cands) and all("too wide" in c.reason for c in cands)
    # 44.6 mm is still inside the limit (marginal), 45.5 mm is not
    edge = box(1, "red", (0.24, 0.0), 0.0, (0.0455, 0.020, 0.018))
    assert all("too wide" in c.reason for c in G.plan_grasp(edge, Scene([edge], TABLE), arm()) if near(c.width, 0.0455, 1e-9))


def test_other_refusals():
    a = arm()

    def only(tgt, text):
        cands = G.plan_grasp(tgt, Scene([tgt], TABLE), a)
        assert len(cands) >= 1 and not any(c.ok for c in cands), describe(cands)
        assert all(text in c.reason for c in cands), describe(cands)
        for c in cands:
            assert isinstance(c, GraspCandidate) and c.score == 0.0
            assert "NaN" not in json.dumps(c.to_dict())

    only(box(1, "red", (0.24, 0.0), 0.0, (0.040, 0.023, 0.008)), "too low")  # 8 mm high: 4 mm of grip
    only(box(1, "red", (0.24, 0.0), 0.0, (0.008, 0.006, 0.020)), "too narrow")
    only(cyl(1, "white", (0.24, 0.0), 0.011, 0.020), "too narrow")
    only(black_box(), "is a container")
    only(box(1, "red", (0.262, 0.085), 0.0, (0.040, 0.023, 0.018), z0=TABLE + 0.003, in_container=9), "inside container 9")
    only(box(1, "red", (0.24, 0.0), 0.0, (0.040, float("nan"), 0.018)), "not a finite number")
    only(box(1, "red", (0.24, 0.0), 0.0, (0.040, 0.023, 0.0)), "not positive")
    only(box(1, "red", (0.24, 0.0), 0.0, (0.040, 0.023, 0.018), kind="blob"), "kind 'blob'")
    # an object of unknown shape is planned on its bounding box, flagged and ranked below a real box
    unk = box(1, "other", (0.24, 0.0), 0.0, (0.040, 0.023, 0.018), kind="unknown")
    cu = G.plan_grasp(unk, Scene([unk], TABLE), a)
    kb = box(1, "red", (0.24, 0.0), 0.0, (0.040, 0.023, 0.018))
    cb = G.plan_grasp(kb, Scene([kb], TABLE), a)
    assert cu[0].ok and "shape unknown" in cu[0].notes[0] and cu[0].score < cb[0].score


def test_scene_edge_cases():
    a = arm()
    tgt = red_block()
    # no scene at all: plans, and says that nothing was checked
    cands = G.plan_grasp(tgt, None, a)
    assert cands[0].ok and any("no scene given" in n for n in cands[0].notes)
    # the target itself listed in the scene (same id) is not an obstacle
    twin = red_block()
    assert G.plan_grasp(tgt, Scene([twin], TABLE), a)[0].ok
    # a neighbour without a usable footprint or size is not ignored: generous disc + note
    def ghost_at(x, y):
        return SceneObject(id=7, kind="unknown", color="other", center=(x, y), yaw=0.0,
                           size=(float("nan"), float("nan"), 0.03), z_bottom=TABLE, footprint=[],
                           pixel_bbox=(0, 0, 0, 0), confidence=0.2)

    cands = G.plan_grasp(tgt, Scene([tgt, ghost_at(0.2541, -0.0373 + 0.085)], TABLE), a)
    assert all(any("no usable footprint" in n and "#7" in n for n in c.notes) for c in cands)
    assert cands[0].ok, describe(cands)
    cands = G.plan_grasp(tgt, Scene([tgt, ghost_at(0.2541 - 0.055, -0.0373)], TABLE), a)  # where the fixed finger goes
    c = by_azimuth(cands, 7.1)
    assert not c.ok and "#7" in c.reason, describe(cands)
    # a neighbour given only by centre / yaw / size (no polygon) is used as that rectangle
    tgt, nb = _block_with_neighbour(0.012)
    nb.footprint = []
    cands = G.plan_grasp(tgt, Scene([tgt, nb], TABLE), a)
    assert not by_azimuth(cands, -90.0).ok and not by_azimuth(cands, 90.0).ok
    # numpy scalars / lists in the scene objects are fine
    t2 = box(1, "red", np.array([0.2541, -0.0373]), 97.1, np.array([0.040, 0.023, 0.018]))
    t2.footprint = np.array(t2.footprint)
    c2 = G.plan_grasp(t2, Scene([t2], np.float64(TABLE)), a)[0]
    assert c2.ok and all(type(v) is float for v in c2.tip_grasp + c2.tip_hover + c2.object_offset)


def test_config_overrides():
    a = arm()
    tgt = red_block()
    cfg = copy.deepcopy(a.cfg)
    cfg["grasp"] = {"standoff": 0.009, "tip_height": 0.006, "open_margin_deg": 18.0, "hover_clear": 0.040,
                    "stall_offset_ticks": 30}
    c = G.plan_grasp(tgt, Scene([tgt], TABLE), a, cfg)[0]
    d = G.plan_grasp(tgt, Scene([tgt], TABLE), a)[0]
    assert c.ok and near(c.standoff, 0.009, 1e-9) and near(c.tip_grasp[2], TABLE + 0.006, 1e-6)
    assert near(c.tip_hover[2], TABLE + 0.018 + 0.040, 1e-6)
    assert near(c.open_raw - d.open_raw, -6.0 * TPD, 3) and c.expected_raw - d.expected_raw in (9, 10, 11)
    # a different touch position shifts both gripper numbers
    cfg = copy.deepcopy(a.cfg)
    cfg["gripper_touch_raw"] = 2060
    e = G.plan_grasp(tgt, Scene([tgt], TABLE), a, cfg)[0]
    assert e.expected_raw - d.expected_raw == 10 and e.open_raw - d.open_raw == 10
    assert G.DEFAULTS["standoff"] == 0.012  # defaults untouched


def test_stall_model_matches_measurements():
    """Measured stalls: raw 2194 on the 23 mm and 20.4 mm blocks, 2263 on the 30 mm cap; tips touch at 2050."""
    a = arm()
    errs = []
    for tgt, measured in ((red_block(), 2194), (blue_block(), 2194), (white_cap(), 2263)):
        c = G.plan_grasp(tgt, Scene([tgt], TABLE), a)[0]
        errs.append(c.expected_raw - measured)
        # the skill's own tolerance is 110 ticks; the empty-gripper threshold is touch + 45
        assert abs(errs[-1]) <= 20 and c.expected_raw > TOUCH + 45 + 10
        # pre-grasp opening = contact + 24 deg
        assert near(open_q(c.open_raw) - contact_q(c), 24.0, 0.15)
    STATS["stall model"] = f"expected_raw - measured stall = {errs} ticks (red, blue, cap)"


# ===================================================================================== degraded pitch (KI-2)
def test_degraded_pitch_at_far_reach():
    a = arm()
    # the same block at normal and at far reach, jaws radial
    near_blk = box(1, "red", (0.26, 0.0), 90.0, (0.040, 0.023, 0.018))
    far_blk = box(1, "red", (0.34, 0.0), 90.0, (0.040, 0.023, 0.018))
    c_near = by_azimuth(G.plan_grasp(near_blk, Scene([near_blk], TABLE), a), 0.0)
    cands = G.plan_grasp(far_blk, Scene([far_blk], TABLE), a)
    c_far = by_azimuth(cands, 0.0)
    assert c_near.ok and c_near.pitch_deg == 90.0 and not any("degraded" in n for n in c_near.notes)
    assert c_far.ok and 70.0 <= c_far.pitch_deg < 90.0, describe(cands)
    assert any("degraded pitch" in n and "not reachable" in n for n in c_far.notes), c_far.notes
    assert any("jaw line tilted" in n for n in c_far.notes), c_far.notes
    assert c_far.score < c_near.score - 20.0, (c_far.score, c_near.score)
    q = a.solve(np.array(c_far.tip_grasp), c_far.pitch_deg, c_far.roll_deg)
    assert near(K.pose_summary(q, a.tip_tool())["pitch_down_deg"], c_far.pitch_deg, 0.05)
    # hover and grasp share pitch and roll: the final descent is a pure vertical translation
    q_h = a.solve(np.array(c_far.tip_hover), c_far.pitch_deg, c_far.roll_deg)
    assert np.allclose(K.fk(q_h)[:3, :3], K.fk(q)[:3, :3], atol=1e-6)
    # KI-2: at far reach, with the moving jaw on the robot side, the opened jaw hangs below the fingertip
    skew = box(1, "red", (0.33, 0.0), 40.0, (0.040, 0.023, 0.018))
    cands = G.plan_grasp(skew, Scene([skew], TABLE), a)
    hit = 0
    for az in (130.0, -140.0):
        c = by_azimuth(cands, az)
        assert not c.ok and "moving jaw" in c.reason and "surface the object stands on" in c.reason, c.reason
        assert c.pitch_deg < 90.0 and abs(c.roll_deg) > 100.0
        # and that is true on the meshes: the opened jaw would be below the table surface + 1 mm
        mc = mesh_clearances(c, skew, [])
        assert mc["z_min_moving"] < TABLE + 0.001, mc
        hit += 1
    assert hit == 2
    # The arm's own floor check (Arm.check_q) sees this only when it is told the jaw openings. With the
    # opening last seen on the bus (closed while planning) the pose passes; with the pre-grasp opening
    # alone it passes too, because the jaw gets LOWER as it closes (on the meshes: 2.7 mm above the table
    # at the pre-grasp 43 deg, 3-4 mm below it from 29 deg down to where it holds the object); with the
    # openings in between it is refused.
    worst = by_azimuth(cands, -140.0)
    q_bad = a.solve(np.array(worst.tip_grasp), worst.pitch_deg, worst.roll_deg)
    q_pre, q_held = open_q(worst.open_raw), open_q(worst.expected_raw)
    a.check_q(q_bad)
    a.check_q(q_bad, jaw_q_deg=q_pre)
    lows = [mesh().jaw(q_bad, qj)[:, 2].min() - TABLE for qj in (q_held, 0.5 * (q_held + q_pre), q_pre)]
    assert min(lows[:2]) < -0.003 and lows[2] > 0.001, lows
    try:
        a.check_q(q_bad, jaw_q_deg=np.linspace(q_held, q_pre, 7))
    except robot.SafetyError as e:
        assert "moving jaw" in str(e), e
    else:
        raise AssertionError("arm.check_q accepts a pose whose closing jaw goes 4 mm below the table")
    ok = [c for c in cands if c.ok]
    assert ok and all(mesh_clearances(c, skew, [])["z_min"] > TABLE + 0.0005 for c in ok), describe(cands)
    # near the base the wrist cannot reach straight down either; pitch > 90 with the moving jaw in front dips it
    close = box(1, "red", (0.10, 0.0), 90.0, (0.040, 0.023, 0.018))
    cands = G.plan_grasp(close, Scene([close], TABLE), a)
    c = by_azimuth(cands, 0.0)
    assert not c.ok and c.pitch_deg > 90.0 and "moving jaw" in c.reason, describe(cands)
    # a jaw line tilted so much that the moving pad would ride over a low object is refused
    cap = cyl(1, "white", (0.36, 0.05), 0.030, 0.0148)
    cands = G.plan_grasp(cap, Scene([cap], TABLE), a)
    assert any((not c.ok) and "jaw line tilted" in c.reason for c in cands), describe(cands)
    STATS["degraded pitch"] = f"block at x=0.34: pitch {c_far.pitch_deg:.0f} deg, score {c_far.score:.1f} vs {c_near.score:.1f} at x=0.26"


# ===================================================================================== mesh ground truth
def test_gripper_model_covers_meshes():
    """Every point of the real jaw meshes lies inside the planner's closed-form gripper model (0.35 mm tol)."""
    m = mesh()
    tol = 0.35
    fixed = _apply(m.gl_to_gf, m.fixed_gl) * 1e3  # gripper_frame coordinates, mm
    fin = fixed[fixed[:, 2] >= gm.PALM_Z + 0.2]
    hw = np.interp(fin[:, 2], [z for z, _ in G._FIXED_HW][::-1], [h for _, h in G._FIXED_HW][::-1])
    x_out = np.array([gm.fixed_back_face_x(z) for z in fin[:, 2]])
    x_in = np.array([G._fixed_inner_x(z) for z in fin[:, 2]])
    assert len(fin) > 5000
    assert (fin[:, 0] - x_out).max() < tol and (x_in - fin[:, 0]).max() < tol and (np.abs(fin[:, 1]) - hw).max() < tol
    assert fin[:, 2].max() < gm.FIXED_TIP_Z + 0.01
    x0, x1, y0, y1, z0, z1 = G._HOUSING
    rest = fixed[fixed[:, 2] < gm.PALM_Z + 0.2]
    assert rest[:, 0].min() > x0 - tol and rest[:, 0].max() < x1 + tol and rest[:, 1].min() > y0 - tol
    assert rest[:, 1].max() < y1 + tol and rest[:, 2].min() > z0 - tol
    poly = _OutlineTest(G._JAW_OUTLINE_Q0)
    for q in (-11.0, 0.0, 12.0, 30.0, 47.0, 70.0):
        jaw = _apply(m.gl_to_gf, m.jaw_gl(q)) * 1e3
        # undo the jaw rotation with the planner's own convention: must land on the q = 0 outline
        c, s = math.cos(math.radians(-q)), math.sin(math.radians(-q))
        dx, dz = jaw[:, 0] - gm.PIVOT_X, jaw[:, 2] - gm.PIVOT_Z
        x0q, z0q = gm.PIVOT_X + c * dx - s * dz, gm.PIVOT_Z + s * dx + c * dz
        r = np.hypot(dx, dz)
        outside = poly.depth_outside(np.column_stack([x0q, z0q])) > tol
        assert not (outside & (r > 12.6)).any(), (q, int((outside & (r > 12.6)).sum()))
        hwj = G._JAW_HW[np.searchsorted(G._JAW_R_HI, r, side="right")]
        assert (np.abs(jaw[:, 1]) - hwj)[r > 12.6].max() < tol
        hub = jaw[r <= 12.6]  # the hub is covered by the housing box at every angle
        assert hub[:, 0].min() > x0 - tol and hub[:, 0].max() < x1 and hub[:, 2].min() > z0 and hub[:, 2].max() < z1 + tol
        assert r.max() < gm.R_TIP + 0.05
    # and the sample cloud really has the promised resolution (<= 2 mm between neighbours on a face)
    fc = G._fixed_cloud()
    assert np.all(np.diff(fc.key) >= 0) and len(fc.x) > 3000
    assert near(fc.hi[2], gm.FIXED_TIP_Z * 1e-3, 1e-9) and near(fc.lo[2], G._HOUSING[4] * 1e-3, 1e-9)
    for z_lo in (0.004, -0.010, -0.030, -0.050):
        sel = np.abs(fc.z - z_lo) < 0.0011
        back = fc.x[sel].max()
        ys = np.sort(fc.y[sel][np.abs(fc.x[sel] - back) < 1e-6])
        assert len(ys) >= 3 and np.diff(ys).max() <= 0.00201, (z_lo, np.diff(ys).max())
        zs = np.unique(np.round(fc.z[fc.z > -0.061], 6))
        assert np.diff(zs).max() <= 0.00201
    sw = G._sweep_for(5.0, 29.0)
    assert sw.x.shape[0] >= 13 and np.hypot(np.diff(sw.x, axis=0), np.diff(sw.z, axis=0)).max() <= 0.0030
    # along the jaw: neighbouring outline samples at most 1.4 mm apart, across it at most 2 mm
    pts, key = G._jaw_cloud_q0()
    assert np.all(np.diff(key) >= 0)
    outline = np.unique(np.round(pts[:, [0, 2]], 6), axis=0)
    d = np.hypot(outline[:, None, 0] - outline[None, :, 0], outline[:, None, 1] - outline[None, :, 1])
    np.fill_diagonal(d, np.inf)
    assert d.min(axis=1).max() <= 1.4, d.min(axis=1).max()
    for x, z in outline[::7]:
        ys = np.sort(pts[(np.abs(pts[:, 0] - x) < 1e-6) & (np.abs(pts[:, 2] - z) < 1e-6), 1])
        assert len(ys) >= 2 and np.diff(ys).max() <= 2.001, (x, z, ys)
    # the jaw model also agrees with gripper_model's own reach and nose depth (from 10 deg on the reach is
    # set by the finger; below, by the hub, which the planner keeps inside the housing box)
    for q in (10.0, 20.0, 30.0, 45.0):
        c = G._jaw_at(int(q * 100))
        assert near(-c.lo[0] * 1e3, gm.jaw_reach_minus_x(q), 0.7), (q, -c.lo[0] * 1e3, gm.jaw_reach_minus_x(q))
        assert near(c.hi[2] * 1e3, gm.moving_tip_depth(q), 0.35), (q, c.hi[2] * 1e3, gm.moving_tip_depth(q))


class _OutlineTest:
    """How far outside a (possibly non-convex) polygon points lie; independent of grasp._Polygon (uses cv2)."""

    def __init__(self, poly):
        import cv2
        self.cv2 = cv2
        self.poly = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)

    def depth_outside(self, pts):
        return np.array([-self.cv2.pointPolygonTest(self.poly, (float(x), float(y)), True) for x, y in pts])


def _random_scene(rng, far=False):
    kind = "box" if rng.random() < 0.6 else "cylinder"
    c = (rng.uniform(0.305, 0.345), rng.uniform(-0.06, 0.06)) if far else (rng.uniform(0.17, 0.30), rng.uniform(-0.11, 0.11))
    if kind == "box":
        length = rng.uniform(0.024, 0.045)
        tgt = box(1, "red", c, rng.uniform(-90, 90), (length, rng.uniform(0.014, min(0.032, length)), rng.uniform(0.012, 0.032)))
        r_t = 0.5 * math.hypot(tgt.size[0], tgt.size[1])
    else:
        tgt = cyl(1, "white", c, rng.uniform(0.016, 0.042), rng.uniform(0.012, 0.032))
        r_t = tgt.size[0] / 2
    others = []
    for k in range(int(rng.integers(1, 4))):
        for _ in range(20):
            a, gap = rng.uniform(0, 2 * math.pi), rng.uniform(0.004, 0.045)
            h = rng.uniform(0.008, 0.050)
            if rng.random() < 0.5:
                d = rng.uniform(0.012, 0.040)
                o = cyl(10 + k, "green", (c[0] + (r_t + gap + d / 2) * math.cos(a), c[1] + (r_t + gap + d / 2) * math.sin(a)), d, h)
            else:
                ln, wd = rng.uniform(0.02, 0.05), rng.uniform(0.015, 0.03)
                rr = r_t + gap + 0.5 * math.hypot(ln, wd) * rng.uniform(0.6, 1.0)
                o = box(10 + k, "green", (c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)), rng.uniform(-90, 90), (ln, wd, h))
            # objects do not interpenetrate
            probe = np.array(tgt.footprint)
            if footprint_distance(o, probe).min() > 0.003 and footprint_distance(tgt, np.array(o.footprint)).min() > 0.003:
                others.append(o)
                break
    if rng.random() < 0.25:  # a container wall somewhere near
        a = rng.uniform(0, 2 * math.pi)
        rr = r_t + rng.uniform(0.010, 0.050) + 0.05
        bx = container(20, (c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)), math.degrees(a) + rng.uniform(-10, 10),
                       (0.100, 0.100), 0.030)
        if footprint_distance(bx, np.array(tgt.footprint)).min() > 0.004 and \
                (footprint_distance(tgt, np.array(bx.footprint)).min() > 0.004):
            others.append(bx)
    return tgt, others


def test_never_intersects_mesh_truth():
    """Property test on random cluttered scenes. For EVERY ok candidate the real jaw meshes, placed at the
    planned grasp pose and at heights of the descent, do not touch any other object's footprint at that
    object's height, the table, or the target; and candidates refused for a neighbour really are that close."""
    a = arm()
    mesh()
    rng = np.random.default_rng(2026)
    n_ok = n_rej = n_scenes = n_tilted = n_table = 0
    worst_ok, timed = np.inf, []
    over_rejected = []
    for k in range(110):
        tgt, others = _random_scene(rng, far=k >= 80)  # the last 30 at far reach: tilted approaches
        sc = Scene([tgt] + others, TABLE)
        t0 = time.perf_counter()
        cands = G.plan_grasp(tgt, sc, a)
        timed.append((time.perf_counter() - t0, k, tgt, others))
        n_scenes += 1
        for c in cands:
            _check_candidate(c, tgt)
            if c.ok:
                mc = mesh_clearances(c, tgt, others)
                # 2 mm required by the planner; its sample cloud may be optimistic by up to 1.6 mm at a corner
                assert mc["fixed"] > 0.0004 and mc["moving"] > 0.0004, (describe([c]), mc, tgt, others)
                assert mc["z_min"] > TABLE + 0.0005, (describe([c]), mc)
                assert mc["self_fixed"] > 0.002 and mc["self_open"] > 0.0025, (describe([c]), mc)
                worst_ok = min(worst_ok, mc["fixed"], mc["moving"])
                n_tilted += c.pitch_deg != 90.0
                # on the way down (same pitch / roll, gripper opened) nothing is touched either
                for frac in (0.35, 0.7):
                    z = c.tip_grasp[2] + frac * (c.tip_hover[2] - c.tip_grasp[2])
                    md = mesh_clearances(c, tgt, others, n_jaw=1, tip=(c.tip_grasp[0], c.tip_grasp[1], z))
                    assert md["fixed"] > 0.0004 and md["self_fixed"] > 0.002, (describe([c]), md)
                    q_o = open_q(c.open_raw)
                    qa = a.solve(np.array([c.tip_grasp[0], c.tip_grasp[1], z]), c.pitch_deg, c.roll_deg)
                    jaw = mesh().jaw(qa, q_o)
                    for ob in others:
                        top = ob.z_top if ob.inner is None else max(ob.z_top, ob.inner["rim_z"])
                        sel = jaw[jaw[:, 2] <= top]
                        assert not len(sel) or footprint_distance(ob, sel[:, :2]).min() > 0.0004, describe([c])
                n_ok += 1
            elif ("would hit" in c.reason or "would sweep into" in c.reason) and math.isfinite(c.roll_deg):
                # refused for a neighbour: on the meshes (same 3 mm height margin) it must be nearly as close
                mc = mesh_clearances(c, tgt, others, z_margin=0.003, n_jaw=13)
                side = "fixed" if "fixed finger" in c.reason else "moving"
                if mc[side] > 0.0045:
                    over_rejected.append((round(mc[side] * 1e3, 2), c.reason))
                n_rej += 1
            elif "surface the object stands on" in c.reason:
                # refused for the table (KI-2): on the meshes the gripper really is lower than 1 mm + model slack
                assert mesh_clearances(c, tgt, [], n_jaw=13)["z_min"] < TABLE + 0.001 + 0.0008, (describe([c]),)
                n_table += 1
    assert n_ok >= 100 and n_rej >= 60 and n_tilted >= 15 and n_table >= 5, (n_ok, n_rej, n_tilted, n_table)
    assert not over_rejected, over_rejected[:5]
    # timing: the five scenes that took longest above, re-measured (best of 5 each); the slowest of them
    # in idle-machine milliseconds
    best = 0.0
    slow = machine_slowdown()
    for _dt, _k, tgt_s, others_s in sorted(timed, key=lambda r: -r[0])[:5]:
        sc_s = Scene([tgt_s] + others_s, TABLE)
        t_s = min(_timed(lambda: G.plan_grasp(tgt_s, sc_s, a)) for _ in range(5))
        if t_s > best:
            best, tgt, others = t_s, tgt_s, others_s
    assert best / slow < 0.05, (best, slow)
    STATS["mesh property test"] = (f"{n_scenes} random scenes: {n_ok} ok candidates verified on the meshes ({n_tilted} "
                                   f"with a tilted gripper; smallest true clearance {worst_ok * 1e3:.2f} mm), {n_rej} "
                                   f"neighbour rejections and {n_table} table rejections confirmed; slowest "
                                   f"scene ({len(others)} neighbours, {tgt.kind}) {best * 1e3:.1f} ms best of 5 "
                                   f"= {best / slow * 1e3:.1f} ms idle-equivalent (machine {slow:.2f}x slower than idle)")


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ===================================================================================== release
def _release_checks(tgt, bx, cand, rel, mode=None):
    a = arm()
    assert isinstance(rel, ReleasePlan) and rel.ok and rel.reason == "", rel
    json.dumps(rel.to_dict())
    inn = bx.inner
    rim, floor = inn["rim_z"], inn["floor_z"]
    hang = cand.tip_grasp[2] - tgt.z_bottom
    # reachable and safe according to the arm (the container's own keep-out, if any, is the caller's to ignore)
    q_above = a.solve(np.array(rel.tip_above), rel.pitch_deg, rel.roll_deg)
    q_rel = a.solve(np.array(rel.tip_release), rel.pitch_deg, rel.roll_deg)
    a.check_q(q_above)
    a.check_q(q_rel)
    lim = a.q_limits()["wrist_roll"]
    assert math.degrees(lim[0]) <= rel.roll_deg <= math.degrees(lim[1])
    # travel height: fingertip >= rim + object height + 15 mm; straight above the release point
    assert rel.tip_above[2] >= rim + tgt.size[2] + 0.015 - 1e-6, rel.tip_above
    assert rel.tip_above[:2] == rel.tip_release[:2] and rel.tip_above[2] > rel.tip_release[2]
    # the OBJECT centre is at the inner centre. By FK, independently of the planner: from the fingertip, go
    # |object_offset| along the jaw axis (fixed -> moving jaw) and (hang - height / 2) along the fingers
    t_rel = K.fk(q_rel)
    o = K.jaw_open_axis(t_rel)
    tip_fk = K.fk(q_rel, a.tip_tool())[:3, 3]
    assert np.linalg.norm(tip_fk - np.array(rel.tip_release)) < 2e-4
    centre3 = tip_fk + float(np.hypot(*cand.object_offset)) * o + (hang - tgt.size[2] / 2.0) * K.approach_axis(t_rel)
    centre = centre3[:2]
    assert np.linalg.norm(centre - np.array(inn["center"])) < 0.0005, (centre, inn["center"])
    assert np.linalg.norm(np.array(rel.tip_release[:2]) - np.array(inn["center"])) > 0.005  # i.e. NOT the fingertip
    if rel.pitch_deg == 90.0:  # ... which, for a vertical gripper, is object_offset turned by the pan / roll change
        d = math.radians(wrap(math.degrees(math.atan2(o[1], o[0])) - cand.jaw_azimuth_deg))
        off = np.array(cand.object_offset)
        turned = np.array([math.cos(d) * off[0] - math.sin(d) * off[1], math.sin(d) * off[0] + math.cos(d) * off[1]])
        assert np.linalg.norm(np.array(rel.tip_release[:2]) + turned - centre) < 2e-4
        q_g = a.solve(np.array(cand.tip_grasp), cand.pitch_deg, cand.roll_deg)
        d_joint = (rel.roll_deg - math.degrees(q_rel[0])) - (cand.roll_deg - math.degrees(q_g[0]))
        if cand.pitch_deg == 90.0:
            assert abs(wrap(math.degrees(d) - d_joint)) < 0.01  # jaw azimuth change = change of (roll - pan)
    # jaws aligned with the container; the object's footprint inside the inner rectangle with margin
    jaw_az = math.degrees(math.atan2(o[1], o[0]))
    rel_az = wrap(jaw_az - math.degrees(inn["yaw"]))
    assert min(abs(wrap(rel_az - k)) for k in (0, 90, 180, 270)) < 0.6, rel_az
    span = tgt.size[0] if tgt.kind != "cylinder" and abs(cand.width - tgt.size[1]) < abs(cand.width - tgt.size[0]) else \
        (cand.width if tgt.kind == "cylinder" else tgt.size[1])
    foot = SceneObject(id=0, kind="box", color="", center=tuple(centre), yaw=math.radians(jaw_az),
                       size=(cand.width, span, tgt.size[2]), z_bottom=0.0,
                       footprint=rect(centre, math.radians(jaw_az), cand.width, span), pixel_bbox=(0, 0, 0, 0), confidence=1)
    inner_obj = box(0, "", inn["center"], math.degrees(inn["yaw"]), (inn["size"][0], inn["size"][1], 0.03))
    assert footprint_distance(inner_obj, np.array(foot.footprint)).max() < -0.005 + 1e-6
    # opening: beyond the stall position, but only as far as needed (12 deg past contact, less than pre-grasp)
    assert near(open_q(rel.open_raw) - contact_q(cand), 12.0, 0.2), (rel.open_raw, cand.expected_raw)
    assert cand.expected_raw + 100 < rel.open_raw < cand.open_raw
    # heights and drop
    assert near(rel.drop_height, rel.tip_release[2] - hang - floor, 1e-5) and rel.drop_height > 0.0
    got = [k for k, text in (("inside", "release inside the rim"), ("at rim", "release at the rim"),
                             ("above", "release from above the rim")) if any(text in n for n in rel.notes)]
    assert len(got) == 1, rel.notes
    got = got[0]
    if mode is not None:
        assert got == mode, rel.notes
    m = mesh()
    fixed = m.fixed(q_rel)
    jaws = np.vstack([m.jaw(q_rel, qa) for qa in np.linspace(contact_q(cand), open_q(rel.open_raw), 7)])
    z_low = min(fixed[:, 2].min(), jaws[:, 2].min())
    assert z_low > floor + 0.001
    below = np.vstack([fixed[fixed[:, 2] <= rim], jaws[jaws[:, 2] <= rim]])
    close = np.vstack([fixed[fixed[:, 2] <= rim + 0.005], jaws[jaws[:, 2] <= rim + 0.005]])
    if got == "inside":  # fingertips just inside the rim
        assert rim - 0.006 <= rel.tip_release[2] < rim, rel.tip_release
        assert len(below) > 0
    elif got == "at rim":  # the object's bottom level with the rim, the fingertips just above it
        assert near(rel.tip_release[2] - hang, rim, 1e-5), rel.tip_release
    else:  # object bottom 5 mm above the rim (or above what lies inside); no gripper part below the rim
        assert rel.tip_release[2] - hang >= rim + 0.005 - 1e-6
        assert len(below) == 0 and z_low > rim + 0.003
    if got != "above" and len(close):
        # mesh truth: every gripper part less than 5 mm above the rim stays inside the inner rectangle,
        # 6 mm from the walls (minus the 1.6 mm resolution of the planner's model), while the jaw opens
        worst = footprint_distance(inner_obj, close[:, :2]).max()
        assert worst < -(0.006 - 0.0016), worst
    return got


def test_release_fit_real_cases():
    """The three objects of 2026-11-10/11 into the black box (inner about 105 x 94 mm, rim 30 mm)."""
    a = arm()
    bx = black_box()
    for tgt, others in ((red_block(), []), (white_cap(), []), (blue_block(), [ivory_cylinder()])):
        sc = Scene([tgt, bx] + others, TABLE)
        cand = G.plan_grasp(tgt, sc, a)[0]
        # as skills.py calls it (no scene): object bottom level with the rim, 27 mm above the box floor
        rel = G.plan_release(tgt, bx, cand, a, a.cfg)
        got = _release_checks(tgt, bx, cand, rel, mode="at rim")
        assert rel.pitch_deg == 90.0, rel
        assert near(rel.tip_above[2], TABLE + 0.030 + tgt.size[2] + 0.015, 1e-5)
        assert near(rel.tip_release[2], TABLE + 0.030 + 0.004, 1e-5) and near(rel.drop_height, 0.027, 1e-5), rel
        assert any("no scene given" in n for n in rel.notes)
        # with the scene (box known to be empty): fingertip 2 mm inside the rim, 21 mm drop
        low = G.plan_release(tgt, bx, cand, a, scene=sc)
        _release_checks(tgt, bx, cand, low, mode="inside")
        assert near(low.tip_release[2], TABLE + 0.030 - 0.002, 1e-5) and near(low.drop_height, 0.021, 1e-5), low
        assert low.tip_release[:2] == rel.tip_release[:2] and low.roll_deg == rel.roll_deg
        STATS[f"release {tgt.color}"] = (f"{got}: tip_above {rel.tip_above} tip_release {rel.tip_release} roll "
                                         f"{rel.roll_deg:.1f} open_raw {rel.open_raw} drop {rel.drop_height * 1e3:.0f} mm "
                                         f"(with scene=: fingertip 2 mm inside the rim, drop {low.drop_height * 1e3:.0f} mm)")


def test_release_long_side_follows_the_container():
    """An object that fits only one way round is turned that way."""
    a = arm()
    slot = container(9, (0.25, 0.09), 0.0, (0.120, 0.050), 0.030)  # inner 114 x 44 mm, long side along x
    tgt = box(1, "red", (0.24, -0.04), 60.0, (0.045, 0.023, 0.018))
    for cand in [c for c in G.plan_grasp(tgt, Scene([tgt, slot], TABLE), a) if c.ok]:
        rel = G.plan_release(tgt, slot, cand, a)
        _release_checks(tgt, slot, cand, rel)
        q = a.solve(np.array(rel.tip_release), rel.pitch_deg, rel.roll_deg)
        o = K.jaw_open_axis(K.fk(q))
        az = math.degrees(math.atan2(o[1], o[0]))
        across_width = near(cand.width, 0.023, 1e-9)
        # object's 45 mm side must lie along x: jaws along y when it is held across the width, along x otherwise
        assert near(abs(wrap(az)) % 180.0, 90.0 if across_width else 0.0, 1.0) or \
            near(abs(abs(wrap(az)) - 180.0), 0.0, 1.0) and not across_width, (az, cand.width)


def test_release_with_a_tilted_gripper():
    """A container at the edge of reach: straight down is not reachable above it, the gripper is tilted, and
    the OBJECT's centre (not the fingertip, not the vertical-gripper guess) still ends over the centre."""
    a = arm()
    tgt = red_block()
    cand = G.plan_grasp(tgt, Scene([tgt], TABLE), a)[0]
    seen = []
    for c in ((0.30, 0.10), (0.31, 0.11), (0.33, 0.10)):
        far = container(9, c, 94.0, (0.111, 0.100), 0.030)
        rel = G.plan_release(tgt, far, cand, a)
        assert rel.ok and 70.0 <= rel.pitch_deg < 90.0, rel
        assert any("degraded pitch" in n and "tilted" in n for n in rel.notes), rel.notes
        _release_checks(tgt, far, cand, rel)
        seen.append(rel.pitch_deg)
    assert seen == sorted(seen, reverse=True)  # the further away, the flatter
    # a tall object (60 mm) is carried and released much higher: the same applies
    tall = cyl(1, "yellow", (0.20, -0.08), 0.025, 0.060)
    ct = G.plan_grasp(tall, Scene([tall], TABLE), a)[0]
    bx = black_box()
    rel = G.plan_release(tall, bx, ct, a)
    assert rel.ok and near(rel.tip_above[2], TABLE + 0.030 + 0.060 + 0.015, 1e-5), rel
    assert near(rel.tip_release[2], TABLE + 0.030 + 0.045, 1e-5)  # it hangs 45 mm below the fingertip
    _release_checks(tall, bx, ct, rel)
    # beyond reach: refused with the kinematic reason
    rel = G.plan_release(tgt, container(9, (0.36, 0.08), 94.0, (0.111, 0.100), 0.030), cand, a)
    assert not rel.ok and "no release pose" in rel.reason and "out of reach" in rel.reason, rel.reason


def test_release_no_fit():
    a = arm()
    tgt = red_block()
    cand = G.plan_grasp(tgt, Scene([tgt], TABLE), a)[0]
    # 40 x 23 mm block, inner area 34 x 34 mm
    small = container(9, (0.262, 0.085), 94.0, (0.040, 0.040), 0.030)
    rel = G.plan_release(tgt, small, cand, a)
    assert not rel.ok and "does not fit" in rel.reason and "23 x 40 mm" in rel.reason and "34 x 34 mm" in rel.reason, rel.reason
    json.dumps(rel.to_dict())
    # fits one way only with less than the 5 mm margin: 40 + 2 * 5 > 48
    snug = container(9, (0.262, 0.085), 94.0, (0.054, 0.054), 0.030)
    rel = G.plan_release(tgt, snug, cand, a)
    assert not rel.ok and "does not fit" in rel.reason, rel.reason
    # the cap (30 mm) into a 34 mm hole: no
    cap = white_cap()
    ccap = G.plan_grasp(cap, Scene([cap], TABLE), a)[0]
    assert not G.plan_release(cap, small, ccap, a).ok
    # container out of the arm's reach: refused with the kinematic reason, not silently
    far = container(9, (0.45, 0.15), 0.0, (0.111, 0.100), 0.030)
    rel = G.plan_release(tgt, far, cand, a)
    assert not rel.ok and "no release pose" in rel.reason and "out of reach" in rel.reason, rel.reason


def test_release_above_rim_when_the_gripper_does_not_fit():
    a = arm()
    cap = white_cap()
    cand = G.plan_grasp(cap, Scene([cap], TABLE), a)[0]
    # inner 54 x 54 mm: the 30 mm cap fits with 12 mm to spare, but the opening jaw would reach the wall
    tight = container(9, (0.25, 0.08), 0.0, (0.060, 0.060), 0.030)
    for sc in (None, Scene([cap, tight], TABLE)):
        rel = G.plan_release(cap, tight, cand, a, scene=sc)
        got = _release_checks(cap, tight, cand, rel, mode="above")
        assert any("wall" in n for n in rel.notes), rel.notes
        assert near(rel.tip_release[2], TABLE + 0.030 + 0.004 + 0.005, 1e-5)  # object bottom 5 mm above the rim
        assert near(rel.drop_height, 0.030 + 0.005 - 0.003, 1e-5)
    STATS["release above rim"] = f"{got}: {[n for n in rel.notes if 'above the rim' in n][0]}"
    # roomy box: low
    bx = black_box()
    _release_checks(cap, bx, cand, G.plan_release(cap, bx, cand, a), mode="at rim")
    _release_checks(cap, bx, cand, G.plan_release(cap, bx, cand, a, scene=Scene([cap, bx], TABLE)), mode="inside")


def test_release_contents_and_assumed_inner():
    a = arm()
    tgt = red_block()
    bx = black_box()
    cand = G.plan_grasp(tgt, Scene([tgt, bx], TABLE), a)[0]
    base = G.plan_release(tgt, bx, cand, a, scene=Scene([tgt, bx], TABLE))  # empty box: 2 mm inside the rim
    assert near(base.tip_release[2], TABLE + 0.030 - 0.002, 1e-5)
    # something already inside that reaches above the rim: stay 5 mm above it
    tall = cyl(5, "ivory", (0.262, 0.085), 0.0258, 0.0216 + 0.012, z0=TABLE + 0.003, in_container=9)
    rel = G.plan_release(tgt, bx, cand, a, scene=Scene([tgt, bx, tall], TABLE))
    assert rel.ok and near(rel.tip_release[2] - 0.004, tall.z_top + 0.005, 1e-5), rel
    assert any("already in the container" in n for n in rel.notes)
    _release_checks(tgt, bx, cand, rel, mode="above")
    # the ivory cylinder of tonight (21.6 mm) inside: 24.6 mm above the table, so the bottom stays at 29.6 mm
    ivory = cyl(5, "ivory", (0.262, 0.085), 0.0258, 0.0216, z0=TABLE + 0.003, in_container=9)
    rel = G.plan_release(tgt, bx, cand, a, scene=Scene([tgt, bx, ivory], TABLE))
    assert rel.ok and near(rel.tip_release[2] - 0.004, ivory.z_top + 0.005, 1e-5), rel
    assert base.tip_release[2] < rel.tip_release[2] < TABLE + 0.030 + 0.004 + 1e-9
    # low contents change nothing
    low = cyl(5, "white", (0.262, 0.085), 0.030, 0.0148, z0=TABLE + 0.003, in_container=9)
    rel = G.plan_release(tgt, bx, cand, a, scene=Scene([tgt, bx, low], TABLE))
    assert rel.ok and rel.tip_release == base.tip_release
    # container without a measured interior: walls assumed, and said so
    bare = black_box(with_inner=False)
    rel = G.plan_release(tgt, bare, cand, a)
    assert rel.ok and any("not measured" in n for n in rel.notes), rel.notes
    assert near(rel.tip_above[2], TABLE + 0.030 + 0.018 + 0.015, 1e-5)
    # a keep-out box over the container (as skills.py installs) is reported, not a reason to refuse
    a2 = robot.Arm(bus=sim_bus.SimBus(), cfg=copy.deepcopy(a.cfg)).open()
    a2.cfg["keepouts"]["obj9"] = [0.20, 0.33, 0.02, 0.15, -0.07, 0.0125]
    rel = G.plan_release(tgt, bx, cand, a2)
    assert rel.ok and any("'obj9'" in n and "ignore_keepouts" in n for n in rel.notes), rel.notes


def test_release_bad_inputs():
    a = arm()
    tgt, bx = red_block(), black_box()
    cands = G.plan_grasp(tgt, Scene([tgt, bx], TABLE), a)
    good, bad = cands[0], [c for c in cands if not c.ok][0]
    r = G.plan_release(tgt, tgt, good, a)
    assert not r.ok and "not a container" in r.reason
    r = G.plan_release(tgt, bx, bad, a)
    assert not r.ok and "not ok" in r.reason and bad.reason in r.reason
    broken = black_box(with_inner=False)
    broken.size = (float("nan"), 0.1, 0.03)
    r = G.plan_release(tgt, broken, good, a)
    assert not r.ok and "neither a measured inner rectangle nor a usable centre / size" in r.reason
    weird = copy.deepcopy(good)
    weird.width = float("nan")
    r = G.plan_release(tgt, bx, weird, a)
    assert not r.ok and "no usable size / pose" in r.reason
    for r in (G.plan_release(tgt, tgt, good, a), G.plan_release(tgt, bx, bad, a)):
        assert isinstance(r, ReleasePlan) and r.reason
        assert "NaN" not in json.dumps(r.to_dict())
    # object_offset left at its default: the planner assumes the object rests on the fixed pad and says so
    blank = copy.deepcopy(good)
    blank.object_offset = (0.0, 0.0)
    r = G.plan_release(tgt, bx, blank, a)
    assert r.ok and any("object_offset was empty" in n for n in r.notes)
    assert np.allclose(r.tip_release, G.plan_release(tgt, bx, good, a).tip_release, atol=1e-4)
    # cfg can be passed by position (as skills.py does) or omitted
    assert G.plan_release(tgt, bx, good, a, a.cfg).tip_release == G.plan_release(tgt, bx, good, a).tip_release


def test_scene_keepouts_installed_by_the_skill():
    """skills.plan_pick installs a keep-out box around every other object (its bounding box + 6 mm, up to
    its top + 6 mm) BEFORE it calls the planner. A candidate whose fingertip would enter one is refused by
    the arm's own check, with the keep-out named; the release ignores the boxes over the container."""
    base = arm()
    a = robot.Arm(bus=sim_bus.SimBus(), cfg=copy.deepcopy(base.cfg)).open()
    tgt, ivory, bx = blue_block(), ivory_cylinder(), black_box()
    sc = Scene([tgt, ivory, bx], TABLE)

    def keepout(o, margin=0.006):
        xs, ys = [p[0] for p in o.footprint], [p[1] for p in o.footprint]
        return [min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin, TABLE - 0.05, o.z_top + margin]

    a.cfg["keepouts"].update({"obj4": keepout(ivory), "obj9": keepout(bx)})
    cands = G.plan_grasp(tgt, sc, a, a.cfg)
    best = cands[0]
    assert best.ok and abs(wrap(best.jaw_azimuth_deg + 83.2)) < 1.0, describe(cands)  # fingertip 13 mm outside obj4
    rel = G.plan_release(tgt, bx, best, a, a.cfg)
    assert rel.ok and any("'obj9'" in n and "ignore_keepouts" in n for n in rel.notes), rel
    # a fatter box around the cylinder (as around a rotated block) swallows the fingertip: refused, and why
    a.cfg["keepouts"]["obj4"][2] -= 0.02
    cands = G.plan_grasp(tgt, sc, a, a.cfg)
    c = by_azimuth(cands, -83.2)
    assert not c.ok and "refused by the arm" in c.reason and "keep-out 'obj4'" in c.reason, c.reason
    assert base.cfg["keepouts"].keys() == {"base_column", "cables"}  # the shared test arm is untouched


class _SpyArm(robot.Arm):
    """robot.Arm that records what the planner asks its safety check."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def check_q(self, q, *args, **kw):
        self.calls.append((kw.get("what"), kw.get("jaw_q_deg"), kw.get("ignore_keepouts")))
        return super().check_q(q, *args, **kw)


class _OldArm:
    """An arm whose check_q has the signature of DESIGN.md 5.1 (no jaw_q_deg)."""

    def __init__(self, arm_):
        self._arm = arm_
        self.cfg = arm_.cfg

    def solve(self, xyz, pitch_down_deg, roll_deg, q_seed=None, use_tip=True):
        return self._arm.solve(xyz, pitch_down_deg, roll_deg, q_seed=q_seed, use_tip=use_tip)

    def check_q(self, q, allow_contact=False, ignore_keepouts=(), what="target", q_start=None):
        return self._arm.check_q(q, allow_contact, ignore_keepouts, what, q_start)

    def q_limits(self):
        return self._arm.q_limits()

    def tip_tool(self):
        return self._arm.tip_tool()


def test_arm_floor_check_gets_the_jaw_openings():
    """Arm.check_q checks the moving jaw against the floor for the opening last seen on the bus unless it
    is told the opening (jaw_q_deg). While planning the gripper is closed, so the planner passes the
    openings the gripper will really have: stalled .. pre-grasp at the grasp and hover poses, stalled ..
    release opening above and in the container."""
    spy = _SpyArm(bus=sim_bus.SimBus(), cfg=copy.deepcopy(arm().cfg)).open()
    tgt, bx = red_block(), black_box()
    cands = G.plan_grasp(tgt, Scene([tgt, bx], TABLE), spy)
    best = cands[0]
    seen = {what: jaws for what, jaws, _ in spy.calls}
    assert set(seen) == {"grasp pose", "hover pose"}, seen
    for what in ("grasp pose", "hover pose"):
        jaws = [j for w, j, _ in spy.calls if w == what][0]  # first call = the best direction's
        assert jaws is not None and near(max(jaws), spy.gripper_q_deg(best.open_raw), 0.05), (what, jaws)
        assert near(min(jaws), spy.gripper_q_deg(best.expected_raw), 0.05), (what, jaws)
        assert len(jaws) >= 5 and np.diff(sorted(jaws)).max() < 6.0  # ... and the openings in between
    assert all(j is not None for _, j, _ in spy.calls)
    spy.calls.clear()
    rel = G.plan_release(tgt, bx, best, spy)
    assert rel.ok and {w for w, _, _ in spy.calls} == {"pose above the container", "release pose"}
    for _, jaws, _ in spy.calls:
        assert near(max(jaws), spy.gripper_q_deg(rel.open_raw), 0.05) and near(min(jaws), spy.gripper_q_deg(best.expected_raw), 0.05)
    # an arm whose check_q does not take jaw_q_deg is planned for all the same (the planner's own
    # table check of the moving jaw does not depend on it)
    old = _OldArm(arm())
    c_old = G.plan_grasp(tgt, Scene([tgt, bx], TABLE), old)
    assert [c.to_dict() for c in c_old] == [c.to_dict() for c in cands]
    assert G.plan_release(tgt, bx, c_old[0], old).to_dict() == rel.to_dict()
    far = box(1, "red", (0.33, 0.0), 40.0, (0.040, 0.023, 0.018))
    c_far = G.plan_grasp(far, Scene([far], TABLE), old)
    assert not by_azimuth(c_far, 130.0).ok and "surface the object stands on" in by_azimuth(c_far, 130.0).reason


def test_inputs_are_not_modified():
    a = arm()
    tgt, ivory, bx = blue_block(), ivory_cylinder(), black_box()
    sc = Scene([tgt, ivory, bx], TABLE)
    cfg_before, scene_before = copy.deepcopy(a.cfg), copy.deepcopy(sc.to_dict())
    defaults_before = copy.deepcopy(G.DEFAULTS)
    cands = G.plan_grasp(tgt, sc, a)
    best_before = copy.deepcopy(cands[0])
    G.plan_release(tgt, bx, cands[0], a, scene=sc)
    G.plan_release(tgt, bx, cands[0], a)
    assert a.cfg == cfg_before and sc.to_dict() == scene_before and cands[0] == best_before
    assert G.DEFAULTS == defaults_before
    # and the same call gives the same answer (no hidden state)
    again = G.plan_grasp(tgt, sc, a)
    assert [c.to_dict() for c in again] == [c.to_dict() for c in cands]


def test_integration_with_skills_plan_pick():
    """The integrator's planning path (skills.plan_pick: plan_grasp -> plan_release -> build_plan; nothing
    moves) accepts the planner's output for the three real scenes. skills.py is the integrator's file and
    still changing: when its interface is no longer the one this test was written against, it is skipped."""
    try:
        import skills as SK
        opts, plan_pick = SK.PickOptions(), SK.plan_pick
    except Exception as e:  # not this module's to fix
        raise Skip(f"skills.plan_pick is not usable right now: {type(e).__name__}: {e}")
    a = robot.Arm(bus=sim_bus.SimBus(), cfg=copy.deepcopy(arm().cfg)).open()
    bx = black_box()
    for tgt, others, az in ((red_block(), [], 7.1), (white_cap(), [], -17.9), (blue_block(), [ivory_cylinder()], -83.2)):
        sc = Scene([tgt] + others + [bx], TABLE)
        try:
            plan, _keepouts, report = plan_pick(a, sc, [tgt], bx, opts, G)
            cand, rel, hover, p_hover = plan.cand, plan.rel, plan.hover, plan.p_hover
        except (TypeError, AttributeError) as e:
            raise Skip(f"skills.plan_pick has a different interface now: {e}")
        assert abs(wrap(cand.jaw_azimuth_deg - az)) < 3.0 and rel.ok, report
        # the skill hovers where the planner said, with the planner's pitch: the final descent is vertical
        assert np.allclose(hover, cand.tip_hover, atol=1e-6) and near(p_hover, cand.pitch_deg, 1e-6)
    assert a.cfg["keepouts"].keys() == arm().cfg["keepouts"].keys()


# ===================================================================================== isolation and timing
def test_planner_never_touches_the_bus():
    """Importing and running the planner must not load the serial bus layer, and must not write to the bus."""
    code = ("import sys; sys.path.insert(0, %r); import grasp; "
            "assert 'servo_bus' not in sys.modules and 'scservo_sdk' not in sys.modules and 'robot' not in sys.modules, "
            "sorted(m for m in sys.modules if 'servo' in m or m == 'robot'); print('clean')" % HERE)
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "clean" in out.stdout, out.stderr[-600:]
    a = arm()
    n_log = len(a.bus.log)
    goal = dict(a.bus.goal)
    tgt, bx = blue_block(), black_box()
    cands = G.plan_grasp(tgt, Scene([tgt, ivory_cylinder(), bx], TABLE), a)
    G.plan_release(tgt, bx, cands[0], a)
    assert len(a.bus.log) == n_log and a.bus.goal == goal and not any(a.bus.torque.values())


_COLD = r"""
import sys, time
sys.path.insert(0, %r)
import numpy as np
from fdtypes import Scene
import test_grasp as T
a = T.arm()
slow = T.machine_slowdown()
tgt, others = %s
t0 = time.perf_counter(); c = T.G.plan_grasp(tgt, Scene([tgt] + others, T.TABLE), a); t1 = time.perf_counter()
r = T.G.plan_release(tgt, T.black_box(), c[0], a); t2 = time.perf_counter()
print("COLD", t1 - t0, t2 - t1, slow, c[0].ok, r.ok)
"""


def test_timing():
    """Planning one object must take < 50 ms: warm (same process) and cold (first call of a fresh process).
    Other engineers' jobs share this machine tonight, so the limit is applied to idle-equivalent time
    (measured time / machine_slowdown()); the raw numbers are reported next to it."""
    a = arm()
    bx = black_box()
    ring = [cyl(10 + k, "green", (0.25 + 0.040 * math.cos(al), -0.05 + 0.040 * math.sin(al)), 0.030, 0.025)
            for k, al in enumerate(np.linspace(0, 2 * math.pi, 8, endpoint=False))]
    cases = {"red block": (red_block(), [bx]), "white cap": (white_cap(), [bx]),
             "blue block + cylinder": (blue_block(), [ivory_cylinder(), bx]),
             "cap at far reach beside the box (pitch search on all 12 directions)":
                 (cyl(1, "white", (0.33, 0.0), 0.030, 0.0148), [bx]),
             "cap fenced in by 8 cylinders": (cyl(1, "white", (0.25, -0.05), 0.030, 0.0148), ring),
             "unreachable cap": (cyl(1, "white", (0.42, 0.10), 0.030, 0.0148), [bx])}
    lines = []
    worst = 0.0
    for name, (tgt, others) in cases.items():
        sc = Scene([tgt] + others, TABLE)
        cands = G.plan_grasp(tgt, sc, a)
        slow = machine_slowdown()
        ts = sorted(_timed(lambda: G.plan_grasp(tgt, sc, a)) for _ in range(9))
        assert ts[0] / slow < 0.05 and ts[0] < 0.15, (name, ts, slow)
        worst = max(worst, ts[0] / slow)
        line = (f"{name}: {ts[0] / slow * 1e3:.1f} ms idle-equivalent (raw min/median {ts[0] * 1e3:.1f}/{ts[4] * 1e3:.1f} ms "
                f"at slowdown {slow:.2f}, {len(cands)} candidates)")
        if cands[0].ok:
            tr = sorted(_timed(lambda: G.plan_release(tgt, bx, cands[0], a)) for _ in range(9))
            assert tr[0] / slow < 0.05, (name, tr, slow)
            line += f"; release {tr[0] / slow * 1e3:.1f} ms (raw {tr[0] * 1e3:.1f})"
        lines.append(line)
    cold = []
    for expr in ("(T.blue_block(), [T.ivory_cylinder(), T.black_box()])", "(T.white_cap(), [T.black_box()])"):
        best = None
        for _ in range(3):
            out = subprocess.run([PY, "-c", _COLD % (HERE, expr)], capture_output=True, text=True, timeout=120)
            assert out.returncode == 0, out.stderr[-800:]
            tok = [ln for ln in out.stdout.splitlines() if ln.startswith("COLD")][0].split()
            assert tok[4] == "True" and tok[5] == "True", out.stdout
            trip = (float(tok[1]) / float(tok[3]), float(tok[2]) / float(tok[3]), float(tok[1]), float(tok[3]))
            best = trip if best is None or trip[0] < best[0] else best
        assert best[0] < 0.05 and best[1] < 0.05, best
        cold.append(f"{best[0] * 1e3:.1f} ms grasp + {best[1] * 1e3:.1f} ms release idle-equivalent "
                    f"(raw grasp {best[2] * 1e3:.1f} ms at slowdown {best[3]:.2f})")
    STATS["timing warm"] = " | ".join(lines)
    STATS["timing warm, worst case"] = f"{worst * 1e3:.1f} ms idle-equivalent"
    STATS["timing cold (first call in a fresh process: blue block, white cap)"] = " | ".join(cold)


# ===================================================================================== runner
if __name__ == "__main__":
    import traceback

    np.set_printoptions(precision=4, suppress=True)
    wanted = sys.argv[1:]
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    if wanted:
        tests = [(n, f) for n, f in tests if any(w in n for w in wanted)]
    passed = failed = skipped = 0
    t_all = time.perf_counter()
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            passed += 1
            print(f"PASS {name} ({time.perf_counter() - t0:.2f} s)")
        except Skip as e:
            skipped += 1
            print(f"SKIP {name}: {e}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print()
    for k, v in STATS.items():
        print(f"  {k}: {v}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped of {len(tests)} tests in {time.perf_counter() - t_all:.1f} s")
    sys.exit(1 if failed else 0)
