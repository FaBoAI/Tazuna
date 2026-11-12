"""Tests of the tracking measurement (tracking.py), the opt-in fine mode of Arm.run_path, the
stick-slip option of the physics simulation (phys_bus.py) and `fd track-test`.

Run:  /home/jetson/miniconda/envs/lerobot/bin/python test_tracking.py [-v] [--only NAME]
      ... test_tracking.py --golden      print the reference digests of test_default_mode_is_bit_for_bit
                                         (JH_TRACK_TREE=<other tree> computes them with that tree's modules:
                                         that is how GOLDEN below was made from the pristine copy `base/`)
No hardware is touched: Bus.open is replaced, and an audit hook fails the run on any attempt to open
the servo port, the camera device or v4l2-ctl.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "pen_20261111", "camera_model.json")
# No tool unless a test attaches one: these tests never read the tool file of the tree (tooling.py), so they pass
# whether or not a tool is attached to the real arm, and nobody has to `fd tool detach` to run them.
os.environ["FD_TOOL"] = os.path.join(HERE, "testdata", "no_tool_attached.json")
TREE = os.environ.get("JH_TRACK_TREE", HERE)  # --golden only: take robot / sim_bus / phys_bus from another tree
sys.path.insert(0, HERE)
if TREE != HERE:
    sys.path.insert(0, TREE)


def _audit(event, args):
    s = " ".join(str(a) for a in args[:2])
    if event in ("open", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn") and any(
            k in s for k in ("/dev/ttyACM", "/dev/ttyUSB", "/dev/video", "v4l2-ctl")):
        raise RuntimeError(f"HARDWARE ACCESS FORBIDDEN IN TESTS: {event} {s}")


sys.addaudithook(_audit)

import kinematics as K  # noqa: E402
import phys_bus as PB  # noqa: E402
import servo_bus  # noqa: E402
from robot import ARM, Arm, MotionAborted, SafetyError, load_config  # noqa: E402
from sim_bus import SimBus  # noqa: E402


def _forbidden(*a, **k):
    raise RuntimeError("HARDWARE ACCESS FORBIDDEN IN TESTS")


servo_bus.Bus.open = _forbidden  # these tests can never open the serial port

TRACKS = os.path.join(HERE, "testdata", "pen_20261111", "tracks")
MANIFEST = json.load(open(os.path.join(HERE, "testdata", "pen_20261111", "manifest.json")))
TPR = 4096.0 / (2.0 * math.pi)
SIDE_Q = np.radians([43.15, -20.31, 10.46, 63.28, -0.05])  # the pose beside the paper the night's tests started from
PEN_TIP = [0.00153, -0.00809, 0.06211]  # the marker's tip (tip_offset_model of the manifest, from 22:46)


def raises(exc, fn, match=None):
    try:
        fn()
    except exc as e:
        if match is not None:
            assert match in str(e), f"{match!r} not in {str(e)!r}"
        return e
    raise AssertionError(f"{exc.__name__} not raised")


def phys_arm(stick=True, tip_offset=None, **bus_kw):
    """Arm on the torque-level servo model at the side pose, torqued (45 Hz loop like the real bus)."""
    cfg = load_config()
    if tip_offset is not None:
        cfg["tip_offset"] = list(tip_offset)
    par = dict(PB.STICK_SLIP_20261111) if stick else {}
    par.update(bus_kw)
    par.setdefault("io_s", 0.003)
    bus = PB.PhysBus(cfg, **par)
    arm = Arm(bus=bus, cfg=cfg).open()
    bus.set_q(SIDE_Q)
    arm.torque_on()
    bus.sleep(0.5)
    return arm, bus


def figure(arm, kind="circle", size=0.016, centre=(0.223, 0.003), pitch=75.0, v_mm_s=6.0, height=0.03):
    """Go to the start of a test figure (settled) and return (waypoints, joint speed)."""
    import tracking as TR
    pts = TR.shape_points(kind, size, centre, float(arm.cfg["table_z"]) + height)
    q0 = arm.read_q()
    arm.run_path([q0, arm.solve(np.array(pts[0]), pitch, 0.0, q_seed=q0)], 30.0, settle=True)
    wps = arm.plan_tip_path([(np.array(p), pitch, 0.0) for p in pts[1:]], q_start=arm.read_q(), step_m=0.0015)
    length = sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))
    return wps, TR.joint_speed_for(wps, length, v_mm_s / 1000.0)


# =============================================================== Trace.analyse against the real traces
def _manifest_numbers(text):
    """'... mean 0.59 max 2.50, z error -3.9..+2.5 mm' -> (0.59, 2.50, (-3.9, 2.5) or None)"""
    import re
    m = re.search(r"mean ([0-9.]+) max ([0-9.]+)", text)
    z = re.search(r"z (?:error )?(-?[0-9.]+)\.\.\+?(-?[0-9.]+)", text)
    return float(m.group(1)), float(m.group(2)), (float(z.group(1)), float(z.group(2))) if z else None


def test_analyse_reproduces_the_manifest():
    """Section 7: Trace.analyse on tracks/*.npy gives the numbers written down on the night."""
    import tracking as TR
    files = MANIFEST["tracks"]["files"]
    assert len(files) == 9 and sorted(files) == sorted(os.listdir(TRACKS))
    out = {}
    for name, text in files.items():
        mean, mx, z = _manifest_numbers(text)
        tr = TR.Trace.load(os.path.join(TRACKS, name))
        assert not tr.has_joints and tr.tip_plan.shape == tr.tip_meas.shape == (len(tr), 3)
        a = tr.analyse()
        assert abs(a["path_err_mm"]["mean"] - mean) < 0.006 and abs(a["path_err_mm"]["max"] - mx) < 0.006, (name, a, text)
        assert a["path_err_mm"]["mean"] <= a["path_err_mm"]["p95"] <= a["path_err_mm"]["max"]
        if z is not None:
            assert abs(a["z_err_mm"]["min"] - z[0]) < 0.06 and abs(a["z_err_mm"]["max"] - z[1]) < 0.06, (name, a, text)
        assert set(a) >= {"path_err_mm", "z_err_mm", "lag_s", "end_err_mm", "duration_s"}
        assert 40.0 < a["rate_hz"] < 46.0  # the real loop ran at 43 Hz
        out[name[6:-4]] = (a["path_err_mm"]["mean"], a["path_err_mm"]["max"])
    # the headline of the night: frozen correction 3.35 / 7.8 mm at 4 mm/s, integral on 0.6-0.8 / 2.5-3.6 at 6 mm/s
    assert out["circle_4_1"] == (3.35, 7.8) and out["square_6_0"] == (0.59, 2.5) and out["circle_6_0"] == (0.8, 3.6)
    return out


def test_breakdown_of_the_real_traces():
    """What the error is: split along / across / height, spectrum, stick-slip signature."""
    import tracking as TR
    tr = TR.Trace.load(os.path.join(TRACKS, "track_square_6_0.npy"))
    b = tr.breakdown()
    tf = b["tool_frame_mm"]
    assert set(tf) == {"along", "across", "height"} and all(set(v) == {"mean", "rms", "min", "max"} for v in tf.values())
    # the height wander of the manifest (-3.9 .. +2.5) is the height component over the moving part
    assert -4.0 < tf["height"]["min"] < -3.0 and 2.0 < tf["height"]["max"] < 2.6, tf
    assert 0.8 < tf["along"]["rms"] < 1.5 and 0.6 < tf["across"]["rms"] < 1.2 and 0.8 < tf["height"]["rms"] < 1.4, tf
    # stick-slip: the measured tip stands still a third of the time and then jumps
    assert b["stick_slip"]["stall_frac"] > 0.25 and b["stick_slip"]["burst_ratio"] > 4.0, b["stick_slip"]
    assert "joints" not in b  # a Cartesian file has no joint angles ...
    raises(ValueError, tr.joint_stats, "joints_from_tips")  # ... and says how to get them
    # frozen correction: the tip sags (mean height error well below zero) and lags
    fz = TR.Trace.load(os.path.join(TRACKS, "track_circle_4_1.npy")).breakdown()["tool_frame_mm"]
    assert fz["height"]["mean"] < -2.0 and fz["along"]["rms"] > 3.0, fz
    return {"square_6_0": tf, "stick_slip": b["stick_slip"]}


def test_joints_behind_a_cartesian_trace():
    """The measured tip of the night's files is the FK of whole encoder ticks: they can be recovered,
    which gives the per-joint picture (elbow and lift slip 7 - 13 ticks at a time)."""
    import tracking as TR
    cfg = load_config()
    tr = TR.Trace.from_tip_array(np.load(os.path.join(TRACKS, "track_circle_10_0.npy"))[:140])  # the first 3.2 s
    t0 = time.time()
    got = TR.joints_from_tips(tr, cfg, PEN_TIP, 80.0, first_range=(1, 14, 14, 14, 1))
    dt = time.time() - t0
    assert got is not None, "the first sample did not fit whole ticks"
    ticks, bad = got
    assert bad == 0 and ticks.shape == (len(tr), 5) and np.all(ticks == np.round(ticks))
    z = np.array([cfg["zero_ticks"][j] for j in ARM])
    s = np.array([cfg["sign"][j] for j in ARM])
    xyz = TR.fk_points(s * (ticks - z) / TPR, K.tool_offset(*PEN_TIP))
    assert float(np.max(np.abs(xyz - tr.tip_meas))) < 1e-7
    # fk_points is kinematics.fk, vectorised
    assert np.allclose(xyz[5], K.fk(s * (ticks[5] - z) / TPR, K.tool_offset(*PEN_TIP))[:3, 3], atol=1e-12)
    # the slips of the elbow: it stands, then moves several ticks at once
    plan = np.array([K.ik(p, 80.0, 0.0, tool=K.tool_offset(*PEN_TIP))[0] for p in tr.tip_plan])
    ev = TR.slip_events(tr.t, z[2] + s[2] * plan[:, 2] * TPR, ticks[:, 2])
    assert len(ev) >= 1 and max(e[4] for e in ev) >= 8, ev
    # a wrong tool offset is noticed, not fitted
    assert TR.joints_from_tips(tr, cfg, [0.0, 0.0, 0.05], 80.0, first_range=(1, 8, 8, 8, 0)) is None
    return {"seconds": round(dt, 1), "elbow_slips": len(ev), "largest_ticks": max(e[4] for e in ev)}


# =============================================================== recording
def test_record_and_trace_files():
    import tracking as TR
    cfg = load_config()
    bus = SimBus()
    arm = Arm(bus=bus, cfg=cfg).open()
    arm.torque_on()
    q = arm.read_q()
    seen = []
    tr = TR.record(arm, [q, arm.ready_q()], 25.0, settle=0.5, monitor=lambda t, qp, qm: seen.append(t) and False)
    assert tr.has_joints and len(tr) == len(seen) > 50 and tr.result["aborted"] is False
    assert tr.q_plan.shape == tr.q_meas.shape == (len(tr), 5) and np.all(np.diff(tr.t) > 0)
    tool = arm.tip_tool()
    pp, pm = tr.tip(tool)
    assert np.allclose(pm[-1], K.fk(tr.q_meas[-1], tool)[:3, 3], atol=1e-12)
    assert not np.allclose(tr.tip(None)[1][-1], pm[-1], atol=1e-4)  # another tool, another tip
    a = tr.analyse(tool)
    assert a["end_err_mm"] < 3.0 and a["n"] == len(tr) and abs(a["rate_hz"] - 50.0) < 1.0, a
    js = tr.joint_stats(cfg, tool)
    assert set(js) == set(ARM) and js["shoulder_lift"]["travel_deg"] > 30 and "err_rms_ticks" in js["elbow_flex"]
    with tempfile.TemporaryDirectory() as d:
        path = tr.save(os.path.join(d, "t"), tool)
        assert path.endswith(".npz") and os.path.getsize(path) < 200_000
        back = TR.Trace.load(path)
        assert np.array_equal(back.q_meas, tr.q_meas) and np.allclose(back.tip_meas, pm) and back.meta["fine"] is False
        assert back.analyse() == a  # stored tips: no tool needed
        np.save(os.path.join(d, "c.npy"), np.hstack([tr.t[:, None], pp, pm]))
        assert TR.Trace.load(os.path.join(d, "c.npy")).analyse() == a  # the night's file format
    # a monitor that stops the motion still stops it, and the part flown so far comes with the exception
    q = arm.read_q()
    e = raises(MotionAborted, lambda: TR.record(arm, [q, q + np.array([0.5, 0, 0, 0, 0])], 20.0,
                                                monitor=lambda t, qp, qm: t > 0.5))
    assert e.info["reason"] == "monitor" and e.trace is not None and 0.4 < e.trace.t[-1] < 0.7
    # refused before it starts: nothing recorded
    e = raises(SafetyError, lambda: TR.record(arm, [q + 1.0, q], 20.0))
    assert e.trace is None
    for shape, n in (("circle", 41), ("square", 5), ("line", 3)):
        pts = TR.shape_points(shape, 0.02, (0.22, 0.0), 0.03)
        assert len(pts) == n and pts[0] == pts[-1] and all(p[2] == 0.03 for p in pts)
    raises(ValueError, lambda: TR.shape_points("star", 0.02, (0.22, 0.0), 0.03))


# =============================================================== the default mode is untouched
def _digest(bus, extra=""):
    h = hashlib.sha256(extra.encode())
    for t, j, v in bus.log:
        h.update(f"{t:.6f} {j} {int(v)}\n".encode())
    for j in sorted(bus.pos):
        h.update(f"{j} {bus.pos[j]:.6f}\n".encode())
    return f"{h.hexdigest()[:16]}:{len(bus.log)}"


def _square(cfg, centre=(0.223, 0.003), s=0.012, h=0.03):
    cx, cy = centre
    z = float(cfg["table_z"]) + h
    return [(cx - s, cy - s, z), (cx - s, cy + s, z), (cx + s, cy + s, z), (cx + s, cy - s, z), (cx - s, cy - s, z)]


def _reference_motions(run_path_kw=None) -> dict:
    """A set of motions on the simulated buses with the DEFAULT mode: every goal write (time, joint,
    ticks) and the final positions, hashed. Uses nothing that the tree of 2026-11-12 00:00 did not
    have, so the same function gives the reference when run with that tree's modules."""
    kw = dict(run_path_kw or {})
    out = {}
    cfg = load_config()
    # 1. ideal servos: rest -> ready, a square with blended corners at drawing speed, a joint move with the gripper
    bus = SimBus()
    arm = Arm(bus=bus, cfg=cfg).open()
    arm.torque_on()
    q = arm.read_q()
    arm.run_path([q, arm.ready_q()], 25.0, settle=0.5, **kw)
    pts = _square(cfg)
    q0 = arm.read_q()
    arm.run_path([q0, arm.solve(np.array(pts[0]), 75.0, 0.0, q_seed=q0)], 20.0, settle=True, **kw)
    wps = arm.plan_tip_path([(np.array(p), 75.0, 0.0) for p in pts[1:]], q_start=arm.read_q(), step_m=0.0015)
    r = arm.run_path(wps, 2.5, settle=0.8, **kw)
    arm.run_path([arm.read_q(), arm.read_q() + np.array([0.3, 0.1, -0.1, 0.05, 0.4])], 30.0, settle=False,
                 gripper_goal_raw=2600, time_law="minjerk", **kw)
    out["simbus_square"] = _digest(bus, json.dumps(r["corr_deg"]))
    # 2. torque-level servos (default friction), the night's circle at 6 mm/s, integral on, then frozen
    for name, ki in (("physbus_circle", None), ("physbus_circle_frozen", 0.0)):
        cfg = load_config()
        bus = PB.PhysBus(cfg, io_s=0.003)
        arm = Arm(bus=bus, cfg=cfg).open()
        bus.set_q(SIDE_Q)
        arm.torque_on()
        bus.sleep(0.5)
        z = float(cfg["table_z"]) + 0.03
        pts = [(0.223 + 0.008 * math.cos(a), 0.003 + 0.008 * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 41)]
        q0 = arm.read_q()
        arm.run_path([q0, arm.solve(np.array(pts[0]), 75.0, 0.0, q_seed=q0)], 30.0, settle=True, **kw)
        wps = arm.plan_tip_path([(np.array(p), 75.0, 0.0) for p in pts[1:]], q_start=arm.read_q(), step_m=0.0015)
        if ki is not None:
            arm.cfg["ki_per_s"] = ki
        r = arm.run_path(wps, 2.4, allow_contact=True, settle=0.8, **kw)
        out[name] = _digest(bus, json.dumps([r["corr_deg"], r["peak_err_deg"]]))
    # 3. a descent that lands on something 8 mm above its target: the pinned abort of the settle phase
    cfg = load_config()
    bus = PB.PhysBus(cfg, z_surface=float(cfg["table_z"]) + 0.012, k_contact=5000.0)
    arm = Arm(bus=bus, cfg=cfg).open()
    bus.set_q(arm.ready_q())
    arm.torque_on()
    arm.move_tip(np.array([0.22, 0.0, float(cfg["table_z"]) + 0.03]), 80.0, 0.0, 20.0, **kw)
    try:
        arm.move_tip(np.array([0.22, 0.0, float(cfg["table_z"]) + 0.004]), 80.0, 0.0, 12.0, straight=True, **kw)
        why = "reached"
    except MotionAborted as e:
        why = e.info["reason"]
    bus.sleep(1.0)
    out["physbus_blocked"] = _digest(bus, why)
    out["physbus_blocked_why"] = why
    return out


# made with `JH_TRACK_TREE=<jetharness_work/base> python test_tracking.py --golden` (the tree before this package)
GOLDEN = {
    "simbus_square": "b1d775bd382a4f6c:6082",
    "physbus_circle": "015a893931dcb950:2506",
    "physbus_circle_frozen": "977a893de28249c7:2636",
    "physbus_blocked": "24f850e52eccc4e1:1681",
    "physbus_blocked_why": "target_not_reached",
}


def test_default_mode_is_bit_for_bit():
    """Section 7 / 5.5: with `fine` off every goal write is what it was before this package: the same
    motions as the pristine tree, on the ideal and on the torque-level simulated servos (default
    friction model: the stiction option of phys_bus is off unless asked for)."""
    got = _reference_motions()
    assert got["physbus_blocked_why"] == "target_not_reached", got
    assert GOLDEN, "no reference digests in this file"
    assert got == GOLDEN, {k: (got[k], GOLDEN.get(k)) for k in got if got[k] != GOLDEN.get(k)}
    # fine=False said out loud is the same call
    assert _reference_motions({"fine": False}) == GOLDEN
    return got


def test_fine_off_changes_nothing_in_the_result():
    bus = SimBus()
    arm = Arm(bus=bus, cfg=load_config()).open()
    arm.torque_on()
    q = arm.read_q()
    r = arm.run_path([q, arm.ready_q()], 25.0, settle=0.5)
    assert "fine" not in r and "fine_ff_deg" not in r
    q = arm.read_q()
    r2 = arm.run_path([q, q + np.array([0.2, 0, 0, 0, 0])], 5.0, fine=True)
    assert r2["fine"] is True and r2["fine_ff_deg"] == [0.12, 0.6, 0.6, 0.12, 0.0] and set(r2) - set(r) == {"fine", "fine_ff_deg"}
    # the stiction option is off by default
    assert PB.PhysBus(load_config()).tau_static is None


# =============================================================== the fine mode keeps every safety property
def test_fine_mode_stays_inside_the_correction_bounds():
    """Whatever the feed-forward adds, the correction written to the servos stays inside the bounds of
    the default mode (how hard the arm can push on an obstacle), and it is zero for a resting plan."""
    arm, bus = phys_arm()
    wps, speed = figure(arm, "square", 0.024)
    lo, hi = arm._path_bounds([np.asarray(w, float) for w in wps], float(arm._c("friction_corr_max_deg")))
    seen = []
    real = arm._write_q

    def spy(q_plan, integ=None, start=None):
        seen.append(np.array(integ, dtype=float))
        return real(q_plan, integ, start)

    arm._write_q = spy
    r = arm.run_path(wps, speed, settle=0.8, fine=True)
    c = np.array(seen)
    assert len(c) > 300 and np.all(c >= lo - 1e-12) and np.all(c <= hi + 1e-12), (c.min(0), c.max(0), lo, hi)
    assert r["fine"] is True and r["aborted"] is False
    # the helper itself: clipped, proportional below v0, zero at rest
    fine = arm._fine_params()
    ff = Arm._fine_corr(fine, np.zeros(5), np.radians([5.0, -5.0, 0.25, 0.0, 9.0]), np.full(5, -1.0), np.full(5, 1.0))
    assert np.allclose(np.degrees(ff), [0.12, -0.6, 0.3, 0.0, 0.0], atol=1e-9), np.degrees(ff)
    tight = Arm._fine_corr(fine, np.radians([0.0, 1.4, 0, 0, 0]), np.radians([5.0, 5.0, 0, 0, 0]),
                           np.radians(np.full(5, -1.5)), np.radians(np.full(5, 1.5)))
    assert math.degrees(tight[1]) <= 1.5 + 1e-9
    # a typing error in the config refuses before anything moves
    for bad in ({"ff_deg": {"elbow_flex": 5.0}}, {"ff_deg": {"elbow": 0.3}}, {"ff_deg": {"elbow_flex": -0.1}},
                {"ff_v0_deg_s": 0.0}, {"ff_deg": {"shoulder_pan": "x"}}):
        arm.cfg["fine_motion"] = bad
        n = len(bus.log)
        q = arm.read_q()
        raises(SafetyError, lambda: arm.run_path([q, q + 0.05], 5.0, fine=True), "fine_motion")
        assert len(bus.log) == n
        arm.run_path([q, q], 5.0)  # the default mode does not read it
    arm.cfg["fine_motion"] = {"ff_deg": {"elbow_flex": 0.3}, "ff_v0_deg_s": 1.0}
    assert np.allclose(np.degrees(arm._fine_params()["ff"]), [0.12, 0.6, 0.3, 0.12, 0.0])


def test_fine_mode_aborts_like_the_default_mode():
    """Pinned abort of the settle phase, servo fault, tracking error: all three stop a fine motion and
    leave the arm holding with the compensation the motion started with."""
    cfg = load_config()
    # 1. lands on something 8 mm above the target
    for fine in (False, True):
        bus = PB.PhysBus(cfg, z_surface=float(cfg["table_z"]) + 0.012, k_contact=5000.0)
        arm = Arm(bus=bus, cfg=load_config()).open()
        bus.set_q(arm.ready_q())
        arm.torque_on()
        arm.move_tip(np.array([0.22, 0.0, float(cfg["table_z"]) + 0.03]), 80.0, 0.0, 20.0)
        e = raises(MotionAborted, lambda: arm.move_tip(np.array([0.22, 0.0, float(cfg["table_z"]) + 0.004]), 80.0, 0.0,
                                                       12.0, straight=True, fine=fine))
        assert e.info["reason"] == "target_not_reached", (fine, e.info)
        bus.sleep(2.0)
        assert bus.fz < 6.0, (fine, bus.fz)  # the push was released
    # 2. a servo fault bit during the motion
    arm, bus = phys_arm()
    q = arm.read_q()

    def mon(t, qp, qm):
        if t > 0.6:
            bus.fault["elbow_flex"] = 32
        return False

    e = raises(MotionAborted, lambda: arm.run_path([q, q + np.array([0.3, 0, 0, 0, 0])], 5.0, monitor=mon, fine=True))
    assert e.info["reason"] == "servo_fault" and e.info["faults"] == {"elbow_flex": 32}
    p0 = dict(bus.pos)
    bus.sleep(1.0)
    assert all(abs(bus.pos[j] - p0[j]) < 8 for j in ARM)
    # 3. a joint that cannot follow: tracking-error abort (ideal servos with an obstacle on the pan joint)
    bus = SimBus(blocker=lambda pos: pos["shoulder_pan"] < 1800)
    arm = Arm(bus=bus, cfg=load_config()).open()
    raw = arm.q_to_raw_f(arm.ready_q())
    for j in ARM:
        bus.pos[j], bus.goal[j], bus.sp[j] = raw[j], float(int(round(raw[j]))), raw[j]
    arm.torque_on()
    bus.sleep(0.3)
    q = arm.read_q()
    q2 = q.copy()
    q2[0] = math.radians(-70)
    e = raises(MotionAborted, lambda: arm.run_path([q, q2], 40.0, fine=True))
    assert e.info["reason"] in ("tracking_error", "target_not_reached"), e.info
    e0 = raises(MotionAborted, lambda: Arm.run_path(arm, [arm.read_q(), q2], 40.0))
    assert e0.info["reason"] == e.info["reason"]
    assert abs(bus.goal["shoulder_pan"] - bus.pos["shoulder_pan"]) < 20  # not left pushing


# =============================================================== the physics simulation and the real traces
def test_stiction_option_makes_a_slow_joint_stick_and_slip():
    """Without the option a slowly driven joint follows smoothly one friction band behind; with it the
    joint stands and jumps, as the real elbow did (7 - 12 ticks at a time at 2.4 deg/s)."""
    import tracking as TR
    res = {}
    for name, stick, kw, ki in (("default model", False, {}, 2.0), ("stiction only", True, {"stick_speed_deg_s": 0.0}, 2.0),
                                ("stick-slip, frozen", True, {}, 0.0), ("stick-slip, integral on", True, {}, 2.0)):
        arm, bus = phys_arm(stick=stick, **kw)
        q = arm.read_q()
        q2 = q + np.array([0.0, 0.0, math.radians(-12.0), 0.0, 0.0])
        arm.cfg["ki_per_s"] = ki  # 0: goal = plan + a constant, as in the frozen traces of the night
        tr = TR.record(arm, [q, q2], 2.4, settle=False, time_law="cruise")
        z, s = arm.cfg["zero_ticks"]["elbow_flex"], arm.cfg["sign"]["elbow_flex"]
        ev = TR.slip_events(tr.t, z + s * tr.q_plan[:, 2] * TPR, np.round(z + s * tr.q_meas[:, 2] * TPR))
        ev = [e for e in ev if e[0] > 1.5]  # after the start
        res[name] = (len(ev), float(np.median([e[4] for e in ev])) if ev else 0.0,
                     round(float(np.median([e[2] for e in ev])), 1) if ev else None,
                     round(float(np.median([e[3] for e in ev])), 1) if ev else None)
    assert res["default model"][0] == 0 and res["stiction only"][0] == 0, res  # slides smoothly once it has let go
    assert res["stick-slip, frozen"][0] >= 5 and 4.0 <= res["stick-slip, frozen"][1] <= 14.0, res
    # integral on: the real elbow (track_square_6_0) let go 2.6 ticks behind the plan and stopped 2.8 ahead, 9 - 12 ticks on
    n, slip, e0, e1 = res["stick-slip, integral on"]
    assert n >= 5 and 6.0 <= slip <= 14.0 and -5.0 < e0 < -0.5 and 0.5 < e1 < 5.0, res  # (this ramp runs the other way)
    # the option never lowers the breakaway torque below the kinetic friction, and is off by default
    b = PB.PhysBus(load_config(), tau_fric=0.25, tau_static=0.1, stick_speed_deg_s=2.0)
    assert np.all(b.tau_static == 0.25) and b.w_stick > 0
    b = PB.PhysBus(load_config(), stick_speed_deg_s=2.0)  # without tau_static the speed does nothing
    assert b.tau_static is None and b.w_stick == 0.0
    return {"slips, median ticks, error at start, at end": res}


def test_simulation_against_the_real_traces():
    """How well the physics simulation reproduces the night (written into the report): with the
    stick-slip option the path error of the night's figures comes out within a factor of two of the
    measured one, integral on and frozen, and the tip stands still about as often. It does NOT
    reproduce the detail: the largest error is up to a third too small, and with the correction
    frozen the simulated tip ends above its plan where the real one sagged below it."""
    import tracking as TR
    out = {}
    for name, kind, size, centre, pitch, tip, v, frozen in (
            ("circle_6_0", "circle", 0.016, (0.223, 0.003), 80.0, PEN_TIP, 6.0, False),
            ("square_6_0", "square", 0.024, (0.2575, 0.003), 75.0, [0.017, -0.0028, 0.0583], 6.0, False),
            ("circle_4_1", "circle", 0.016, (0.2575, 0.003), 75.0, [0.017, -0.0028, 0.0583], 4.0, True)):
        real = TR.Trace.load(os.path.join(TRACKS, f"track_{name}.npy"))
        ra, rb = real.analyse(), real.breakdown()
        arm, bus = phys_arm(tip_offset=tip)
        wps, speed = figure(arm, kind, size, centre, pitch, v)
        ki = arm.cfg["ki_per_s"]
        if frozen:
            arm.cfg["ki_per_s"] = 0.0
        tr = TR.record(arm, wps, speed, allow_contact=True, settle=0.8)
        arm.cfg["ki_per_s"] = ki
        sa, sb = tr.analyse(arm.tip_tool()), tr.breakdown(arm.tip_tool(), arm.cfg)
        out[name] = {"real": [ra["path_err_mm"]["mean"], ra["path_err_mm"]["max"]],
                     "sim": [sa["path_err_mm"]["mean"], sa["path_err_mm"]["max"]],
                     "z real": list(ra["z_err_mm"].values()), "z sim": list(sa["z_err_mm"].values()),
                     "stall real/sim": [rb["stick_slip"]["stall_frac"], sb["stick_slip"]["stall_frac"]]}
        for key in ("mean", "max"):
            ratio = sa["path_err_mm"][key] / ra["path_err_mm"][key]
            assert 0.5 <= ratio <= 2.0, (name, key, ratio, sa, ra)
        assert abs(sa["rate_hz"] - ra["rate_hz"]) < 1.5  # same loop rate as the real bus
        assert abs(sb["stick_slip"]["stall_frac"] - rb["stick_slip"]["stall_frac"]) < 0.15, (name, sb["stick_slip"], rb["stick_slip"])
        zr = (ra["z_err_mm"]["max"] - ra["z_err_mm"]["min"], sa["z_err_mm"]["max"] - sa["z_err_mm"]["min"])
        assert 0.5 <= zr[1] / zr[0] <= 2.0, (name, zr)  # the height wander has the right size
    # frozen is worse than integral on in the simulation as on the arm
    assert out["circle_4_1"]["sim"][0] > 2.0 * out["circle_6_0"]["sim"][0]
    return out


# =============================================================== fd track-test
def _fd(argv, arms=None, cfg_edits=None, files=None, hook=None):
    """fd.main(argv) writing nothing into the tree: no run log, traces go to a temporary directory
    (their names are collected in `files`), edit_config_file is replaced (what it would have written
    is collected in cfg_edits). arms collects every Arm fd builds; without --sim fd gets a stand-in
    "real" arm: torque-level servos with stick-slip (never the serial port)."""
    import fd
    real = (fd.make_arm, fd.log_event, fd.edit_config_file, fd.run_dir)

    def make(sim):
        arm = real[0](True) if sim else Arm(bus=PB.PhysBus(load_config(), io_s=0.003, **PB.STICK_SLIP_20261111),
                                            cfg=load_config())
        if arms is not None:
            arms.append(arm)
        if hook is not None:
            hook(arm)  # (before fd opens it)
        return arm

    def edit(mutate):
        raw = {}
        mutate(raw)
        if cfg_edits is not None:
            cfg_edits.append(raw)
        return raw

    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        fd.make_arm, fd.log_event, fd.edit_config_file, fd.run_dir = make, (lambda *a, **k: None), edit, (lambda: tmp)
        try:
            with contextlib.redirect_stdout(buf):
                rc = fd.main(argv)
        finally:
            fd.make_arm, fd.log_event, fd.edit_config_file, fd.run_dir = real
        if files is not None:
            files.extend(sorted(os.listdir(tmp)))
    return rc, json.loads(buf.getvalue())


def _watch_tip(arm, lows):
    """Record the lowest REAL tool-tip height (above the table) the simulated arm ever has."""
    tip, zt = arm.tip_tool(), float(arm.cfg["table_z"])
    real = arm.bus.feedback

    def feedback(*a, **k):
        fb = real(*a, **k)
        z = float(K.fk(arm.raw_to_q({j: arm.bus.pos[j] for j in ARM}), tip)[2, 3]) - zt
        lows.append(z if not lows else min(lows[-1], z))
        return fb

    arm.bus.feedback = feedback


def test_fd_track_test_in_simulation():
    """`fd --sim track-test` runs (section 7), with every shape, and returns the usual result keys."""
    for shape in ("circle", "square", "line"):
        arms = []
        rc, r = _fd(["--sim", "track-test", "--shape", shape], arms)
        assert rc == 0 and r["ok"] is True and r["stage"] == "done" and r["sim"] is True, r
        assert {"reason", "next", "say", "runs", "shape", "v_mm_s", "height_m"} <= set(r), sorted(r)
        d = r["runs"]["default"]
        assert d["path_err_mm"]["max"] < 1.0 and -1.0 < d["z_err_mm"]["min"] <= d["z_err_mm"]["max"] < 1.0, d
        assert set(d["breakdown"]["tool_frame_mm"]) == {"along", "across", "height"} and "joints" in d["breakdown"]
        assert "シミュレーション" in r["say"] and r["traces"] == []
        assert len(arms) == 1 and len(arms[0].bus.log) > 500  # it did fly, on the arm fd built
    # --dry: plan only, not a single bus write
    arms = []
    rc, r = _fd(["--sim", "track-test", "--dry", "--shape", "square", "--size", "30", "--v", "8"], arms)
    assert rc == 0 and r["stage"] == "planned" and r["dry"] is True and r["expected_s"] > 5 and "runs" not in r, r
    assert arms[0].bus.log == [] and arms[0].bus.io["writes"] == 0
    # the stick-slip servo model shows a real-sized error and the fine mode runs in the comparison
    rc, r = _fd(["--sim", "track-test", "--phys", "--fine"])
    assert rc == 0 and set(r["runs"]) == {"default", "fine"} and "fine_verdict" in r, r
    assert 0.2 < r["runs"]["default"]["path_err_mm"]["mean"] < 2.0 and r["runs"]["default"]["rate_hz"] < 46
    assert set(r["fine_verdict"]) >= {"passed", "fails", "mean_ratio", "p95_ratio", "z_range_ratio"}
    return {"phys default": r["runs"]["default"]["path_err_mm"], "phys fine": r["runs"]["fine"]["path_err_mm"],
            "verdict": r["fine_verdict"]}


def test_fd_track_test_stays_in_the_air():
    """Refused below 2 cm above the table and for a negative height, before anything is written to the
    bus; the lowest goal of a flown test is never under 2 cm."""
    for h in ("0.015", "0.0", "-0.01", "nan"):
        arms = []
        rc, r = _fd(["--sim", "track-test", "--height", h], arms)
        assert rc == 1 and r["ok"] is False and r["error"] == "SafetyError" and "in the air" in r["message"], (h, r)
        assert arms[0].bus.log == [] and arms[0].bus.io["writes"] == 0, h
    for argv, word in ((["--size", "200"], "--size"), (["--v", "100"], "--v"), (["--at", "here"], "--at"),
                       (["--accept"], "--fine"), (["--repeat", "9"], "--repeat"), (["--pitch", "5"], "IK failed"),
                       (["--at", "0.60,0.0"], "IK failed")):
        arms = []
        rc, r = _fd(["--sim", "track-test"] + argv, arms)
        assert rc == 1 and r["error"] == "SafetyError" and word in r["message"], (argv, r)
        assert arms[0].bus.log == [], argv
    rc, r = _fd(["track-test", "--phys", "--dry"])  # --phys is a choice of the simulation
    assert rc == 1 and "--sim" in r["message"], r
    # lowest height the tool tip really had during a whole test at the minimum height, way there and back
    # included: on the ideal servos, and on the stick-slip servos in both modes (the tip wanders there)
    out = {}
    for name, argv, floor in (("ideal", ["--sim", "track-test", "--height", "0.02", "--shape", "square"], 0.019),
                              ("stick-slip, fine", ["track-test", "--height", "0.02", "--fine"], 0.015)):
        lows = []
        rc, r = _fd(argv, hook=lambda arm, lows=lows: _watch_tip(arm, lows))
        assert rc == 0 and len(lows) > 500, r
        assert floor < lows[-1] < 0.0215, (name, lows[-1])
        out[name] = round(lows[-1] * 1000, 1)
    return {"lowest_tip_height_mm": out}


def test_fd_track_test_accept_only_on_the_real_arm():
    """--accept never records a simulated run; on the (stand-in) real arm it records only a passed
    comparison, as one more entry of fine_motion.accepted in config.json."""
    import fd
    edits = []
    rc, r = _fd(["--sim", "track-test", "--fine", "--accept", "--repeat", "3"], cfg_edits=edits)
    assert rc == 0 and r["accepted"] is False and "simulation" in r["note"] and edits == [], r
    assert r["runs"]["default"]["flights"] == 3 and len(r["runs"]["fine"]["each"]) == 3 and r["flights"] == 6, r
    rc, r = _fd(["--sim", "track-test", "--phys", "--fine", "--accept", "--repeat", "3", "--shape", "line"], cfg_edits=edits)
    assert rc == 0 and r["accepted"] is False and edits == [], r
    d = r["runs"]["default"]  # several flights: mean of the means, largest maximum, envelope of the height error
    assert abs(d["path_err_mm"]["mean"] - np.mean([e["path_err_mm"]["mean"] for e in d["each"]])) < 0.006
    assert d["path_err_mm"]["max"] == max(e["path_err_mm"]["max"] for e in d["each"])
    assert d["z_err_mm"]["min"] == min(e["z_err_mm"]["min"] for e in d["each"])
    # single flights scatter more than the modes differ: a comparison of single flights is never recorded
    arms = []
    for argv in (["track-test", "--fine", "--accept"], ["track-test", "--fine", "--accept", "--repeat", "2"]):
        rc, r = _fd(argv, arms, cfg_edits=edits)
        assert rc == 1 and r["error"] == "SafetyError" and "--repeat 3" in r["message"] and edits == [], r
        assert arms[-1].bus.log == []
    # the stand-in "real" arm (no --sim): torque-level servos with stick-slip. Force both verdicts.
    real_verdict = fd._track_verdict
    try:
        fd._track_verdict = lambda b, f: {"passed": False, "fails": ["forced"], "mean_ratio": 2.0, "p95_ratio": 2.0,
                                          "z_range_ratio": 2.0}
        files = []
        rc, r = _fd(["track-test", "--fine", "--accept", "--repeat", "3", "--shape", "line"], cfg_edits=edits, files=files)
        assert rc == 0 and "sim" not in r and r["accepted"] is False and edits == [], r
        # on the real arm every flight's trace is kept (one small file each)
        assert len(r["traces"]) == 6 and len(files) == 6 and all(f.endswith(".npz") for f in files), (r["traces"], files)
        fd._track_verdict = lambda b, f: {"passed": True, "fails": [], "mean_ratio": 0.8, "p95_ratio": 0.9,
                                          "z_range_ratio": 0.9}
        rc, r = _fd(["track-test", "--fine", "--accept", "--repeat", "3", "--shape", "line"], cfg_edits=edits)
    finally:
        fd._track_verdict = real_verdict
    assert rc == 0 and r["accepted"] is True and r["accepted_runs"] == 1 and len(edits) == 1, r
    rec = edits[0]["fine_motion"]["accepted"][0]
    assert rec["shape"] == "line" and rec["verdict"]["passed"] is True and rec["repeat"] == 3
    # what a drawing may ask: has the fine mode passed on the real arm (twice by default)?
    import tracking as TR
    assert TR.fine_accepted(load_config()) is False and TR.fine_accepted({"fine_motion": {"ff_deg": {}}}) is False
    one = {"fine_motion": {"accepted": [rec]}}
    assert TR.fine_accepted(one) is False and TR.fine_accepted(one, min_runs=1) is True
    assert TR.fine_accepted({"fine_motion": {"accepted": [rec, rec]}}) is True
    assert TR.fine_accepted({"fine_motion": {"accepted": [rec, {"verdict": {"passed": False}}]}}) is False
    assert set(rec) >= {"at", "default", "fine", "v_mm_s"}
    # the verdict itself
    base = {"path_err_mm": {"mean": 0.8, "p95": 2.0, "max": 3.6}, "z_err_mm": {"min": -2.0, "max": 2.0}}
    better = {"path_err_mm": {"mean": 0.6, "p95": 1.5, "max": 3.0}, "z_err_mm": {"min": -1.5, "max": 1.5}}
    worse_z = {"path_err_mm": {"mean": 0.6, "p95": 1.5, "max": 3.0}, "z_err_mm": {"min": -3.0, "max": 2.0}}
    assert fd._track_verdict(base, better)["passed"] is True and fd._track_verdict(base, base)["passed"] is True
    v = fd._track_verdict(base, worse_z)
    assert v["passed"] is False and "z_range_ratio" in v["fails"][0], v
    assert fd._track_verdict(better, base)["passed"] is False


# ----------------------------------------------------------------------------- runner
def main(argv):
    if "--golden" in argv:
        print(json.dumps(_reference_motions(), indent=1))
        return 0
    verbose = "-v" in argv
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    if only:
        tests = [(n, f) for n, f in tests if only in n]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            info = fn()
            print(f"PASS {name} ({time.time() - t0:.2f} s)" + (f"  {info}" if verbose and info else ""))
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:900]}")
            if verbose:
                traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
