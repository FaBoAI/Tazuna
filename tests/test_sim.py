"""Simulation tests for the motion / safety logic (no hardware). Run: python tests/test_sim.py

The first 15 tests are the original suite; the rest cover the review findings folded into
robot.py (ids as in DESIGN.md section 6) and the v2 motion API (section 5.1)."""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import json
import math
import os
import tempfile

import numpy as np

import kinematics as K
import robot as R
from robot import ARM, DEFAULT_CONFIG, Arm, MotionAborted, SafetyError
from servo_bus import JOINTS, BusError
from sim_bus import LIMITS, SimBus

# The gripper in these tests is bare: they never read the tool file of the tree (tooling.py), so they pass
# whether or not a tool is attached to the real arm, and nobody has to `fd tool detach` to run them.
os.environ["FD_TOOL"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "no_tool_attached.json")

np.set_printoptions(precision=4, suppress=True)
TPR = 4096.0 / (2.0 * math.pi)
TPD = 4096.0 / 360.0


def make(blocker=None, sag=12.0, bus_cls=SimBus, bus_kw=None, **cfg_over):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg.update(cfg_over)
    bus = bus_cls(sag_ticks=sag, blocker=blocker, **(bus_kw or {}))
    arm = Arm(bus=bus, cfg=cfg).open()
    return arm, bus


def max_goal_step(bus, start=0):
    """Largest jump between consecutive goal writes per joint (ticks)."""
    last, worst = {}, 0
    for _, j, v in bus.log[start:]:
        if j in last:
            worst = max(worst, abs(v - last[j]))
        last[j] = v
    return worst


def q_of(pos, cfg=DEFAULT_CONFIG):
    z, s = cfg["zero_ticks"], cfg["sign"]
    return np.array([s[j] * (pos[j] - z[j]) / TPR for j in ARM])


def tip_of(arm, bus=None):
    q = arm.read_q() if bus is None else q_of(bus.pos, arm.cfg)
    return K.fk(q, arm.tip_tool())[:3, 3]


def table_blocker(table_z, cfg=DEFAULT_CONFIG):
    tool = K.tool_offset(*cfg["tip_offset"])

    def blocker(pos):
        return K.fk(q_of(pos, cfg), tool)[2, 3] < table_z

    return blocker


def gmp(bus):
    return {j: int(round(bus.goal[j] - bus.pos[j])) for j in JOINTS}


def place(arm, bus, q):
    """Teleport the (limp) simulated arm to joint pose q."""
    raw = arm.q_to_raw_f(q)
    for j in ARM:
        bus.pos[j] = raw[j]
        bus.goal[j] = float(int(round(raw[j])))
        bus.sp[j] = raw[j]


def ready(blocker=None, sag=12.0, **kw):
    """Arm torqued at the ready pose (skips the simulated move from the rest pose)."""
    arm, bus = make(blocker=blocker, sag=sag, **kw)
    place(arm, bus, arm.ready_q())
    arm.torque_on()
    bus.sleep(0.3)
    return arm, bus


def raises(exc, fn, match=None):
    try:
        fn()
    except exc as e:
        if match is not None:
            assert match in str(e), f"{match!r} not in {str(e)!r}"
        return e
    raise AssertionError(f"{exc.__name__} not raised")


class Skip(Exception):
    """A test that could not run here (reported separately, never as a pass)."""


# =============================================================== original suite (15 tests)
def test_ik_roundtrip():
    rng = np.random.default_rng(1)
    n = 0
    for _ in range(4000):
        q = np.array([rng.uniform(-1.7, 1.7), rng.uniform(-1.6, 1.6), rng.uniform(-1.2, 1.6),
                      rng.uniform(-1.6, 1.6), rng.uniform(-2.5, 2.5)])
        g = q[2] + (K._A1 - K._A2)
        t = K.fk(q)
        fwd = (t[0, 3] - K._X_PAN) * math.cos(q[0]) - t[1, 3] * math.sin(q[0])
        if not (0.05 < g < math.pi - 0.05) or fwd < 0.02:
            continue
        n += 1
        wide = {j: (-10, 10) for j in ARM}
        q2, info = K.ik(t[:3, 3], math.degrees(K.pitch_down_of(t, q[0])), q[4], limits=wide)
        assert info["ok"], info
        assert np.max(np.abs((q2 - q + math.pi) % (2 * math.pi) - math.pi)) < 1e-3
    assert n > 1000


def test_ik_respects_limits():
    arm, _ = make()
    lim = arm.q_limits()
    for x in np.arange(0.08, 0.46, 0.03):
        for z in (0.0, 0.05, 0.1, 0.2):
            for pitch in range(-20, 121, 10):
                q, info = K.ik((x, 0.03, z), pitch, 0.2, limits=lim)
                if info["ok"]:
                    for i, j in enumerate(ARM):
                        assert lim[j][0] <= q[i] <= lim[j][1]
                    t = K.fk(q)
                    assert np.linalg.norm(t[:3, 3] - np.array([x, 0.03, z])) < 1e-3


def test_torque_on_does_not_move():
    arm, bus = make()
    before = dict(bus.pos)
    arm.torque_on()
    bus.sleep(1.0)
    for j in ARM:
        assert abs(bus.goal[j] - round(before[j])) <= 1


def test_requires_torque():
    arm, _ = make()
    try:
        arm.move_joints(arm.read_q() + 0.05)
    except SafetyError:
        return
    raise AssertionError("moved without torque")


def test_rest_to_ready_and_back_smooth():
    arm, bus = make()
    arm.torque_on()
    q_ready = arm.solve(np.array([0.20, 0.0, 0.12]), 70.0, 0.0)
    res = arm.move_joints(q_ready, 35.0)
    assert max(abs(e) for e in res["joint_err_deg"]) < 0.5, res
    assert max_goal_step(bus) < 40, max_goal_step(bus)  # < 3.5 deg per 20 ms tick
    tip = K.fk(arm.read_q())[:3, 3]
    assert np.linalg.norm(tip - np.array([0.20, 0.0, 0.12])) < 0.004, tip
    # ... and back to the (clamped) rest pose, as `park` does
    n0 = len(bus.log)
    res = arm.move_joints(arm.rest_q(), 25.0, allow_contact=True, settle=False)
    assert max_goal_step(bus, n0) < 40
    assert max(abs(e) for e in res["joint_err_deg"]) < 1.5, res


def test_straight_line_stays_straight():
    arm, bus = make(sag=0.0)
    arm.torque_on()
    arm.move_tip(np.array([0.22, 0.05, 0.10]), 80.0, 0.0, 35.0)
    zs, ys = [], []

    def mon(t, q_plan, q_meas):
        p = K.fk(q_plan)[:3, 3]
        zs.append(p[2])
        ys.append(p[1])
        return False

    arm.move_tip(np.array([0.22, 0.05, 0.03]), 88.0, 0.0, 20.0, straight=True, monitor=mon)
    assert max(abs(y - 0.05) for y in ys) < 1e-3
    assert all(zs[i + 1] <= zs[i] + 1e-4 for i in range(len(zs) - 1))  # monotone descent


def test_joint_limit_rejected():
    arm, _ = make()
    arm.torque_on()
    q = arm.read_q()
    q[0] = math.radians(130)
    try:
        arm.move_joints(q)
    except SafetyError:
        return
    raise AssertionError("limit violation accepted")


def test_floor_rejected_unless_contact():
    arm, bus = make()
    arm.torque_on()
    arm.move_tip(np.array([0.22, 0.0, 0.08]), 85.0, 0.0, 35.0)
    try:
        arm.move_tip(np.array([0.22, 0.0, -0.04]), 85.0, 0.0, 20.0, straight=True)
    except SafetyError:
        pass
    else:
        raise AssertionError("went below the floor")
    # with allow_contact the fingertip may go a bounded distance below the floor limit
    limit = arm.cfg["floor_z"] + arm.cfg["floor_clearance"]
    arm.move_tip(np.array([0.22, 0.0, limit - 0.004]), 85.0, 0.0, 20.0, straight=True, allow_contact=True)
    assert abs(tip_of(arm)[2] - (limit - 0.004)) < 0.002


def test_keepout_rejected():
    arm, _ = make()
    arm.torque_on()
    arm.move_tip(np.array([0.22, 0.0, 0.08]), 85.0, 0.0, 35.0)
    for target in ((0.06, 0.0, 0.05),):
        try:
            qq = arm.solve(np.array(target), 100.0, 0.0)
            arm.check_q(qq)
        except SafetyError:
            continue
        raise AssertionError("keep-out not enforced")


def test_unreachable_rejected():
    arm, _ = make()
    arm.torque_on()
    try:
        arm.move_tip(np.array([0.60, 0.0, 0.05]), 45.0, 0.0)
    except SafetyError:
        return
    raise AssertionError("unreachable accepted")


def test_stall_aborts_and_holds():
    # a wall stops shoulder_pan from going below 1800 ticks
    arm, bus = make(blocker=lambda pos: pos["shoulder_pan"] < 1800)
    arm.torque_on()
    q = arm.read_q()
    q[0] = math.radians(-60)
    try:
        arm.move_joints(q, 40.0)
    except MotionAborted as e:
        assert abs(bus.goal["shoulder_pan"] - bus.pos["shoulder_pan"]) < 5  # holding where it is
        assert e.info["reason"] == "tracking_error" and e.info["aborted"] is True
        return
    raise AssertionError("stall not detected")


def test_probe_detects_table():
    table_z = 0.013
    arm, bus = make(blocker=table_blocker(table_z))
    arm.torque_on()
    arm.move_tip(np.array([0.24, -0.05, 0.05]), 90.0, 0.0, 35.0)
    res = arm.probe_down(max_drop_m=0.06)
    assert res["contact"] is True, res
    assert abs(res["contact_tip_z"] - table_z) < 0.004, res
    tip = K.fk(arm.read_q())[:3, 3]
    assert tip[2] > table_z + 0.006  # backed off


def test_probe_no_contact_reports_false():
    arm, bus = make()
    arm.torque_on()
    arm.move_tip(np.array([0.24, -0.05, 0.08]), 90.0, 0.0, 35.0)
    res = arm.probe_down(max_drop_m=0.02)
    assert res["contact"] is False and res["reason"] is None, res
    assert abs(res["lowest_tip_z"] - 0.06) < 0.003, res  # free travel is certified down to here


def test_grasp_detects_object():
    # object stops the jaw at 2600 ticks
    arm, bus = make(blocker=lambda pos: pos["gripper"] < 2600)
    arm.torque_on()
    arm.set_gripper(0.9)
    res = arm.grasp()
    assert res["contact"], res
    assert abs(res["pos_raw"] - 2600) < 15
    assert res["hold_goal_raw"] < res["pos_raw"]  # squeezing
    assert res["pos_raw"] - res["hold_goal_raw"] <= 45  # ... by about 40 ticks (60 sits on the overload threshold)
    arm2, bus2 = make()
    arm2.torque_on()
    arm2.set_gripper(0.9)
    res2 = arm2.grasp()
    assert not res2["contact"], res2


def test_goals_never_leave_servo_limits():
    arm, bus = make()
    arm.torque_on()
    arm.move_tip(np.array([0.20, 0.0, 0.12]), 70.0, 0.0, 40.0)
    arm.move_tip(np.array([0.28, -0.12, 0.04]), 80.0, 30.0, 40.0)
    for _, j, v in bus.log:
        assert LIMITS[j][0] <= v <= LIMITS[j][1], (j, v)


# =============================================================== MS-0: blocked target
def _descend_onto(table_z, target_z, arm_kw=None):
    arm, bus = ready(blocker=table_blocker(table_z), **(arm_kw or {}))
    arm.move_tip(np.array([0.24, 0.0, table_z + 0.04]), 90.0, 0.0, 35.0)
    free = gmp(bus)
    try:
        arm.move_tip(np.array([0.24, 0.0, target_z]), 90.0, 0.0, 12.0, straight=True)
        err = None
    except MotionAborted as e:
        err = e
    return arm, bus, free, err


def test_blocked_descent_aborts_and_releases():
    # the fixed jaw lands on something 12 mm above the commanded height (which passes the floor check)
    arm, bus, free, err = _descend_onto(0.012, 0.0)
    assert err is not None, "blocked descent returned normally"
    assert err.info["reason"] == "target_not_reached", err.info
    assert "Push released" in str(err)
    bus.sleep(1.0)
    after = gmp(bus)
    for j in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        # the standing offset is back at its free-air (gravity) value: nothing keeps pushing
        assert abs(after[j] - free[j]) <= 8, (j, after, free)
    assert abs(tip_of(arm)[2] - 0.012) < 0.003


def test_blocked_descent_shallow_depths():
    for depth_mm in (3.5, 6.0):
        arm, bus, free, err = _descend_onto(0.012, 0.012 - depth_mm / 1000.0)
        assert err is not None, f"{depth_mm} mm blocked descent returned normally"
        after = gmp(bus)
        assert all(abs(after[j] - free[j]) <= 8 for j in ("shoulder_lift", "elbow_flex")), (after, free)


def test_blocked_pan_short_overrun_aborts():
    # wall 5 deg before the target: below the tracking-abort threshold, used to return ok with the push standing
    for tgt, reason in ((-27.0, "target_not_reached"), (-60.0, "tracking_error")):
        arm, bus = ready(blocker=lambda pos: pos["shoulder_pan"] < 1800)
        q = arm.read_q()
        q[0] = math.radians(tgt)
        e = raises(MotionAborted, lambda: arm.move_joints(q, 30.0))
        assert e.info["reason"] == reason, e.info
        assert abs(bus.goal["shoulder_pan"] - bus.pos["shoulder_pan"]) <= 6, gmp(bus)


class StictionBus(SimBus):
    """SimBus with a static-friction band: a joint moves only while |goal - pos| exceeds it."""
    band = 12.0

    def _advance_joint(self, j, dt):
        if not self.torque[j]:
            return
        target = self.goal[j] + (self.sag if j in ("shoulder_lift", "elbow_flex") else 0.0)
        e = target - self.pos[j]
        if abs(e) <= self.band:
            return
        cap = self.vel_cap[j] if self.vel_cap[j] > 0 else 3000.0
        self.pos[j] += max(-cap * dt, min(cap * dt, e - math.copysign(self.band, e)))


def test_no_false_abort_with_friction():
    arm, bus = make(bus_cls=StictionBus)
    arm.torque_on()
    arm.move_joints(arm.ready_q(), 35.0)
    rng = np.random.default_rng(3)
    n = 0
    worst = 0.0
    for _ in range(30):
        xyz = np.array([rng.uniform(0.16, 0.26), rng.uniform(-0.12, 0.12), rng.uniform(0.0, 0.14)])
        try:
            res = arm.move_tip(xyz, rng.uniform(60, 90), rng.uniform(-30, 30), 30.0, straight=bool(rng.integers(0, 2)))
        except SafetyError:
            continue  # unreachable random target
        n += 1
        worst = max(worst, max(abs(e) for e in res["joint_err_deg"]))
    assert n >= 15 and worst < 0.6, (n, worst)


# =============================================================== MS-1: inherited contact offset
def _transfer_deficit(arm):
    tool = arm.tip_tool()
    rows = []
    arm.move_tip(np.array([0.20, -0.12, 0.10]), 80.0, 0.0, 30.0,
                 monitor=lambda t, qp, qm: rows.append((K.fk(qp, tool)[2, 3], K.fk(qm, tool)[2, 3])) and False)
    return max(a - b for a, b in rows)


def test_inherited_push_is_not_carried_into_next_move():
    table_z = 0.012
    arm, bus = ready(blocker=table_blocker(table_z))
    arm.move_tip(np.array([0.24, 0.0, 0.04]), 90.0, 0.0, 35.0)
    arm.move_tip(np.array([0.24, 0.0, table_z + 0.0005]), 90.0, 0.0, 12.0, straight=True)
    # a push left on the servos (another process, an old harness version): lift / elbow pressing down
    bus.goal["shoulder_lift"] = bus.pos["shoulder_lift"] + 60
    bus.goal["elbow_flex"] = bus.pos["elbow_flex"] + 70
    bus.sleep(0.5)
    # a harness process that knows nothing of the past (one CLI call per command) takes over at most
    # the inherit allowance (0.5 deg) of such an offset, not the 5-6 deg that stand on the servos
    fresh = Arm(bus=bus, cfg=arm.cfg).open()
    off = np.degrees(fresh._current_offset())
    assert off[1] <= 0.5 + 1e-6 and off[2] <= 0.5 + 1e-6, off
    # the process that made the previous motion knows the compensation from before the contact
    off = np.degrees(arm._current_offset())
    assert abs(off[1] + 1.05) < 0.2 and abs(off[2] + 1.05) < 0.2, off
    deficit = _transfer_deficit(arm)
    assert deficit < 0.004, f"next move ran {deficit * 1000:.1f} mm below its plan"


def test_inherited_push_fresh_process_is_bounded():
    table_z = 0.012
    arm, bus = ready(blocker=table_blocker(table_z))
    arm.move_tip(np.array([0.24, 0.0, 0.04]), 90.0, 0.0, 35.0)
    arm.move_tip(np.array([0.24, 0.0, table_z + 0.0005]), 90.0, 0.0, 12.0, straight=True)
    bus.goal["shoulder_lift"] = bus.pos["shoulder_lift"] + 60
    bus.goal["elbow_flex"] = bus.pos["elbow_flex"] + 70
    bus.sleep(0.5)
    fresh = Arm(bus=bus, cfg=arm.cfg).open()
    deficit = _transfer_deficit(fresh)
    assert deficit < 0.012, f"{deficit * 1000:.1f} mm"  # the unpatched code ran 26-39 mm below its plan here


def test_gravity_offset_is_still_inherited():
    arm, bus = ready()
    arm.move_tip(np.array([0.24, 0.0, 0.04]), 90.0, 0.0, 35.0)
    off = np.degrees(arm._current_offset())
    # the simulated sag (12 ticks = 1.05 deg on lift / elbow) is the holding direction: kept in full
    assert abs(off[1] + 1.05) < 0.2 and abs(off[2] + 1.05) < 0.2, off


def test_move_after_blocked_abort_is_clean():
    arm, bus, free, err = _descend_onto(0.012, 0.0)
    assert err is not None
    deficit = _transfer_deficit(arm)
    assert deficit < 0.004, f"{deficit * 1000:.1f} mm"


# =============================================================== MS-2 / KI-8: probe
def test_probe_refuses_undetectable_descent():
    arm, bus = ready()
    arm.move_tip(np.array([0.24, -0.05, 0.05]), 90.0, 0.0, 35.0)
    n0 = len(bus.log)
    raises(SafetyError, lambda: arm.probe_down(max_drop_m=0.012), "cannot be detected")
    raises(SafetyError, lambda: arm.probe_down(max_drop_m=0.05, speed_m_s=0.03), "outside")
    raises(SafetyError, lambda: arm.probe_down(max_drop_m=0.05, speed_m_s=float("nan")), "outside")
    assert len(bus.log) == n0, "a refused probe moved the arm"


def test_probe_inconclusive_at_range_end():
    # the surface sits 3.6 mm above the end of the probe range: the tip is lagging when the descent
    # ends, but not long enough for a detection -> neither True nor False
    arm, bus = make(blocker=table_blocker(0.0136))
    arm.torque_on()
    arm.move_tip(np.array([0.24, -0.05, 0.05]), 90.0, 0.0, 35.0)
    res = arm.probe_down(max_drop_m=0.04)
    assert res["contact"] is None and "inconclusive" in res["reason"], res
    assert not res["contact"]  # falsy, so `if not res["contact"]` callers treat it as "no table found"
    assert tip_of(arm)[2] > 0.0136 + 0.006  # released and backed off


def test_probe_push_bound_follows_tip_jacobian():
    # KI-8: +wrist_flex lowers the fingertip only for pitch < ~88 deg; beyond, the other side must be capped
    seen = {}
    for pitch, x in ((60.0, 0.22), (110.0, 0.16)):
        arm, bus = make(floor_z=-0.2)
        arm.torque_on()
        arm.move_tip(np.array([x, 0.0, 0.06]), pitch, 0.0, 35.0)
        orig = arm._track

        def spy(plan, duration, integ0, pitch=pitch, orig=orig, **kw):
            seen.setdefault(pitch, (np.degrees(kw["integ_lo"]), np.degrees(kw["integ_hi"]),
                                    np.degrees(np.asarray(integ0))))  # first call = the probe descent
            return orig(plan, duration, integ0, **kw)

        arm._track = spy
        arm.probe_down(max_drop_m=0.025)
        arm._track = orig
    lo, hi, i0 = seen[60.0]
    assert abs(hi[3] - (i0[3] + 2.0)) < 1e-6, (lo, hi, i0)  # pitch 60: +wrist pushes down -> upper side capped
    lo, hi, i0 = seen[110.0]
    assert abs(lo[3] - (i0[3] - 2.0)) < 1e-6, (lo, hi, i0)  # pitch 110: -wrist pushes down -> lower side capped
    for lo, hi, i0 in seen.values():  # lift and elbow: + is down in every front pose
        assert abs(hi[1] - (i0[1] + 2.0)) < 1e-6 and abs(hi[2] - (i0[2] + 2.0)) < 1e-6
        assert lo[1] < -6.9 and lo[2] < -6.9  # upward holding authority untouched


# =============================================================== MS-3: torque_on idempotent
def test_torque_on_is_idempotent():
    arm, bus = make(blocker=lambda pos: pos["gripper"] < 2600)
    first = arm.torque_on()
    assert sorted(first["newly_enabled"]) == sorted(JOINTS)
    arm.move_tip(np.array([0.24, 0.0, 0.03]), 90.0, 0.0, 35.0)
    arm.set_gripper(0.9)
    assert arm.grasp()["contact"]
    goals = dict(bus.goal)
    z0 = tip_of(arm)[2]
    for _ in range(4):
        again = arm.torque_on()
        assert again["newly_enabled"] == []
        bus.sleep(1.0)
    assert bus.goal == goals, "torque_on changed the goal of a holding joint"
    assert abs(tip_of(arm)[2] - z0) < 2e-4
    assert bus.pos["gripper"] - bus.goal["gripper"] >= 35  # the grasp squeeze survives
    # a joint that lost its torque is re-seated at its present position, the others are left alone
    bus.torque["elbow_flex"] = 0
    res = arm.torque_on()
    assert res["newly_enabled"] == ["elbow_flex"]
    assert abs(bus.goal["elbow_flex"] - bus.pos["elbow_flex"]) <= 1
    assert all(bus.goal[j] == goals[j] for j in JOINTS if j != "elbow_flex")


def test_simbus_mirrors_bus_api():
    """SimBus must offer what servo_bus.Bus offers, with the same call signatures and refusals
    (the real class is only inspected here, never opened)."""
    import inspect

    import servo_bus
    for name in ("read", "write", "sync_read", "sync_write", "positions", "feedback", "snapshot", "limits", "faults",
                 "torque_on", "torque_off", "set_p_gain", "open", "close"):
        real = list(inspect.signature(getattr(servo_bus.Bus, name)).parameters)
        sim = list(inspect.signature(getattr(SimBus, name)).parameters)
        assert real == sim, (name, real, sim)
    real_bus = servo_bus.Bus.__new__(servo_bus.Bus)  # no __init__ side effects needed for the stats keys
    servo_bus.Bus.__init__(real_bus, port="/nonexistent/never-opened")
    bus = SimBus()
    assert set(bus.stats) == set(real_bus.stats) and set(bus.status) == set(real_bus.status)
    assert set(bus.snapshot()["elbow_flex"]) == {"pos", "load", "current", "volt", "temp", "torque", "goal", "status",
                                                 "acc", "vel_cap", "lock"}
    assert set(bus.feedback(("gripper",))["gripper"]) == {"pos", "load", "current", "volt", "temp", "status"}
    assert set(bus.feedback("gripper")) == {"gripper"} and bus.is_open is False
    # the write guard
    for reg, val in (("P_Coefficient", 32), ("Homing_Offset", 5), ("Min_Position_Limit", 0), ("Lock", 0),
                     ("Torque_Enable", 128), ("Goal_Position", 5000), ("Goal_Position", float("nan")),
                     ("Goal_Position", None), ("Goal_Position", "2000")):
        raises(BusError, lambda reg=reg, val=val: bus.write("elbow_flex", reg, val))
    raises(BusError, lambda: bus.sync_write("Goal_Position", {"elbow_flex": 2000, "wrist_flex": 9999}))
    raises(BusError, lambda: bus.sync_write("Goal_Position", {}), "no joints")
    raises(BusError, lambda: bus.read("elbow", "Present_Position"), "unknown joint")
    raises(BusError, lambda: bus.read("elbow_flex", "Present_Speed"), "unknown register")
    assert bus.goal["elbow_flex"] == 3025 and not bus.log  # nothing was written by any of the refused calls
    # torque_on: refusals with nothing changed, goal := present, idempotent writes
    raises(BusError, lambda: bus.torque_on(velocity=0, acceleration=20), "unlimited")
    raises(BusError, lambda: bus.torque_on(velocity=400, acceleration=0), "unlimited")
    bus.pos["shoulder_pan"] = LIMITS["shoulder_pan"][1] + 40.0
    raises(BusError, bus.torque_on, "nothing was changed")
    assert not any(bus.torque.values()) and set(bus.vel_cap.values()) == {0}
    bus.pos["shoulder_pan"] = LIMITS["shoulder_pan"][1] + 5.0  # within max_goal_gap: goal goes to the limit
    assert sorted(bus.torque_on(velocity=1100, acceleration=40)["newly_enabled"]) == sorted(JOINTS)
    assert bus.goal["shoulder_pan"] == LIMITS["shoulder_pan"][1] and set(bus.lock.values()) == {1}
    w0 = bus.io["writes"]
    assert bus.torque_on(velocity=1100, acceleration=40) == {"newly_enabled": []} and bus.io["writes"] == w0
    assert bus.stats["requests"] == bus.io["reads"] + bus.io["writes"]
    # set_p_gain
    assert bus.set_p_gain("elbow_flex", 32) is True and bus.set_p_gain("elbow_flex", 32) is False
    for bad in (41, 7, 16.5, None):
        raises(BusError, lambda bad=bad: bus.set_p_gain("elbow_flex", bad))
    bus.lock["wrist_flex"] = 0
    assert bus.set_p_gain("wrist_flex", 16) is False  # already there: no write, Lock not needed
    raises(BusError, lambda: bus.set_p_gain("wrist_flex", 32), "Lock is not 1")
    # torque_off puts the profile registers back; faults() decodes the reply error bytes
    bus.torque_off()
    assert not any(bus.torque.values()) and set(bus.vel_cap.values()) == {0} and set(bus.acc.values()) == {0}
    bus.fault["gripper"] = 32 | 64
    bus.read("gripper", "Present_Position")
    assert bus.faults() == {"gripper": ["overload", "unknown_0x40"]}


def test_torque_on_refused_by_bus_leaves_arm_as_it_was():
    # a joint far outside its stored limits: the bus refuses; the arm stays limp with the boot gain
    arm, bus = make()
    bus.pos["shoulder_pan"] = LIMITS["shoulder_pan"][1] + 60.0
    raises(BusError, arm.torque_on, "by hand")
    assert not any(bus.torque.values()) and set(bus.p_gain.values()) == {16}


def test_arm_on_real_bus_class_through_pty_fake():
    """The motion core on the REAL servo_bus.Bus (packets, sync reads, write guard) against the
    simulated servos of fakebus.py behind a pseudo-terminal. No hardware: the only tty that can
    be opened here is a pty, and the bus lock is a scratch file. Runs in real time (about 8 s)."""
    try:
        import serial

        import fakebus
    except Exception as e:  # fakebus.py belongs to BUS
        raise Skip(f"fakebus not importable ({type(e).__name__}: {e})")
    real_open = serial.Serial.open

    def guarded_open(self):
        if not str(self.port).startswith("/dev/pts/"):
            raise RuntimeError(f"this test may only open a pty, not {self.port}")
        return real_open(self)

    serial.Serial.open = guarded_open
    fake = fakebus.FakeBus()
    bus = fake.bus()
    try:
        assert bus.port.startswith("/dev/pts/") and bus.lock_path == fake.lock_path
        arm = Arm(bus=bus, cfg=json.loads(json.dumps(DEFAULT_CONFIG)))
        arm.open()
        fake.sag[2] = fake.sag[3] = 12.0  # lift / elbow droop like SimBus
        res = arm.torque_on()
        assert sorted(res["newly_enabled"]) == sorted(JOINTS)
        assert arm.torque_on()["newly_enabled"] == []
        q = arm.read_q()
        q[0] += math.radians(25.0)
        r = arm.move_joints(q, 35.0, gripper_goal_raw=2516)
        assert max(abs(e) for e in r["joint_err_deg"]) < 0.5 and r["aborted"] is False, r
        assert r["loop_hz"] > 30.0, r["loop_hz"]  # one write and one 15-byte sync read per cycle
        assert r["min_volt"] is not None and abs(r["gripper_pos_raw"] - 2516) <= 3
        fake.blocker = lambda sid, pos: sid == 6 and pos < 2194  # an object between the jaws
        g = arm.grasp_fast(expected_raw=2194)
        assert g["contact"] and abs(g["delta_raw"]) <= 3 and g["hold_goal_raw"] == g["pos_raw"] - 40, g
        fake.blocker = None
        s = arm.set_gripper(raw=2400)
        assert abs(s["pos_raw"] - 2400) <= 3 and not s["blocked"], s
        # a servo fault in the middle of a move stops the stream
        fake.set_fault(3, 32)
        e = raises(SafetyError, lambda: arm.move_joints(arm.read_q() + np.array([0.2, 0, 0, 0, 0]), 25.0), "overload")
        fake.set_fault(3, 0)
        q2 = arm.read_q() + np.array([-0.3, 0, 0, 0, 0])
        e = raises(MotionAborted, lambda: arm.move_joints(
            q2, 25.0, monitor=lambda t, qp, qm: fake.set_fault(3, 32) if t > 0.4 else None), "overload")
        assert e.info["reason"] == "servo_fault"
        fake.set_fault(3, 0)
        arm.torque_off()
        v = fake.view()
        assert v["torque"] == [0] * 6
        assert bus.stats["fallbacks"] == 0 and bus.stats["failures"] == 0, bus.stats
    finally:
        bus.close()
        fake.close()
        serial.Serial.open = real_open


# =============================================================== MS-5: check_q relaxations
def test_floor_ratchet_is_capped():
    arm, bus = ready()
    limit = arm.cfg["floor_z"] + arm.cfg["floor_clearance"]
    x, y = 0.24, -0.03
    arm.move_tip(np.array([x, y, 0.03]), 90.0, 0.0, 35.0)
    arm.move_tip(np.array([x, y, limit + 0.002]), 90.0, 0.0, 20.0, straight=True)
    raises(SafetyError, lambda: arm.move_tip(np.array([x, y, limit - 0.010]), 90.0, 0.0, 12.0, straight=True))
    n = 0
    for _ in range(40):  # walking down in 0.9 mm steps (used to be accepted without bound)
        try:
            arm.move_tip(tip_of(arm) + np.array([0.0, 0.0, -0.0009]), 90.0, 0.0, 12.0, straight=True)
            n += 1
        except SafetyError:
            pass
    depth = limit - tip_of(arm)[2]
    assert n <= 8 and depth <= arm.cfg["floor_relax_max_m"] + 0.0005, (n, depth)
    # from down there the arm can still move sideways at the same height and back up
    arm.move_tip(tip_of(arm) + np.array([0.0, 0.02, 0.0]), 90.0, 0.0, 12.0, straight=True)
    arm.move_tip(tip_of(arm) + np.array([0.0, 0.0, 0.02]), 90.0, 0.0, 12.0, straight=True)


def test_relax_function():
    arm, _ = make()
    lim = -0.016
    assert arm._relax(lim, None) == lim
    assert arm._relax(lim, 0.05) == lim  # start above: no relaxation
    assert abs(arm._relax(lim, lim - 0.0005) - (lim - 0.0015)) < 1e-12  # 1 mm of noise slack
    assert abs(arm._relax(lim, lim - 0.0025) - (lim - 0.003)) < 1e-12  # capped at floor_relax_max_m
    assert abs(arm._relax(lim, lim - 0.020) - (lim - 0.0202)) < 1e-12  # far below: 0.2 mm only (may move up)


def test_keepout_depth_rule():
    arm, bus = ready()
    arm.move_tip(np.array([0.06, 0.080, 0.03]), 90.0, 0.0, 35.0)
    raises(SafetyError, lambda: arm.move_tip(np.array([0.06, 0.060, 0.03]), 90.0, 0.0, 20.0), "keep-out")
    arm.move_tip(np.array([0.06, 0.062, 0.03]), 90.0, 0.0, 20.0, ignore_keepouts=("base_column",))  # 3 mm inside
    # once inside, an ordinary command must not go deeper ...
    raises(SafetyError, lambda: arm.move_tip(np.array([0.06, 0.050, 0.03]), 90.0, 0.0, 20.0), "keep-out")
    # ... but may always leave
    arm.move_tip(np.array([0.06, 0.080, 0.03]), 90.0, 0.0, 20.0)


def test_contact_depth_is_bounded():
    arm, bus = ready()
    limit = arm.cfg["floor_z"] + arm.cfg["floor_clearance"]
    q_ok = arm.solve(np.array([0.24, 0.0, limit - 0.004]), 90.0, 0.0)
    q_deep = arm.solve(np.array([0.24, 0.0, limit - 0.02]), 90.0, 0.0)
    raises(SafetyError, lambda: arm.check_q(q_ok))
    arm.check_q(q_ok, allow_contact=True)
    raises(SafetyError, lambda: arm.check_q(q_deep, allow_contact=True), "contact depth")
    arm.check_q(q_deep, allow_contact=True, contact_depth=0.03)  # explicit, larger bound
    arm.check_q(q_deep, allow_contact=True, contact_depth=math.inf)  # what probe_down uses
    # the probe keeps working below the floor limit (it bounds its own descent)
    arm.move_tip(np.array([0.24, 0.0, 0.02]), 90.0, 0.0, 35.0)
    res = arm.probe_down(max_drop_m=0.06)
    assert res["contact"] is False and res["lowest_tip_z"] < limit - 0.015, res


def test_limit_flicker_at_rest_is_tolerated():
    # at rest shoulder_lift sits outside its software limit; a target built from a reading that is one
    # tick further out than the reading the path starts from must not be refused (MS-5e)
    arm, bus = make()
    arm.torque_on()
    bus.sleep(0.5)
    q = arm.read_q()
    tgt = q.copy()
    tgt[0] += math.radians(10)
    tgt[1] -= 1.0 / TPR
    tgt[2] += 1.0 / TPR
    arm.move_joints(tgt, 30.0)
    far = q.copy()
    far[1] -= 3.0 / TPR  # ... but really moving further out of the limits stays forbidden
    raises(SafetyError, lambda: arm.check_q(far, q_start=q))
    near = q.copy()
    near[1] -= 2.0 / TPR
    arm.check_q(near, q_start=q)


# =============================================================== MS-6: gains
def test_torque_on_validates_gain_before_any_write():
    arm, bus = make(p_gain=64)
    w0 = bus.io["writes"]
    raises(SafetyError, arm.torque_on, "outside the allowed 8..40")
    assert bus.io["writes"] == w0 and not any(bus.torque.values()) and set(bus.p_gain.values()) == {16}


def test_torque_on_sets_lock_and_gain_before_torque():
    arm, bus = make(bus_kw={"lock": 0})  # LeRobot leaves Lock = 0 after a session
    order = []
    orig_set, orig_write = bus.set_p_gain, bus.write

    def spy_set(j, p):
        order.append(("gain", j, bus.lock[j], bus.torque[j]))
        return orig_set(j, p)

    def spy_write(j, reg, v):
        if reg == "Torque_Enable":
            order.append(("torque", j, v))
        return orig_write(j, reg, v)

    bus.set_p_gain, bus.write = spy_set, spy_write
    res = arm.torque_on()
    gains = [o for o in order if o[0] == "gain"]
    assert len(gains) == 5 and all(lock == 1 and torque == 0 for _, _, lock, torque in gains), order
    assert order.index(gains[-1]) < min(i for i, o in enumerate(order) if o[0] == "torque")
    assert all(bus.p_gain[j] == 32 for j in ARM) and bus.p_gain["gripper"] == 16
    assert res["p_gain"] == 32 and all(bus.torque.values())


def test_failed_gain_write_leaves_arm_limp():
    arm, bus = make()
    orig = bus.set_p_gain

    def flaky(j, p):
        if j == "elbow_flex":
            raise BusError("write P_Coefficient=32 to elbow_flex failed (comm result -6)")
        return orig(j, p)

    bus.set_p_gain = flaky
    raises(BusError, arm.torque_on)
    assert not any(bus.torque.values()), "a failed gain write left torque on"
    bus.set_p_gain = orig
    assert sorted(arm.torque_on()["newly_enabled"]) == sorted(JOINTS)


def test_gain_step_on_loaded_arm():
    arm, bus = ready()
    arm.move_tip(np.array([0.24, 0.0, 0.04]), 90.0, 0.0, 35.0)  # holds 12 ticks of sag on lift / elbow
    raises(SafetyError, arm.restore_gains, "Nothing was changed")
    assert all(bus.p_gain[j] == 32 for j in ARM)
    seq = []
    orig = bus.set_p_gain
    bus.set_p_gain = lambda j, p: (seq.append((j, p)), orig(j, p))[1]
    res = arm.restore_gains(force=True)
    lift = [p for j, p in seq if j == "shoulder_lift"]
    assert lift == list(range(30, 14, -2)), lift  # ramped 32 -> 16 in steps of 2
    assert sorted(res["ramped"]) == ["elbow_flex", "shoulder_lift"]
    seq.clear()
    arm.apply_gains()  # raising on a loaded arm is ramped, not refused
    assert [p for j, p in seq if j == "shoulder_lift"] == list(range(18, 34, 2))
    assert all(bus.p_gain[j] == 32 for j in ARM)
    seq.clear()
    arm.apply_gains()  # already there: no packet at all
    assert seq == []


def test_torque_off_restores_gain_and_profile():
    arm, bus = ready()
    res = arm.torque_off()
    assert res["torque"] == "off" and not any(bus.torque.values())
    assert set(bus.p_gain.values()) == {16}
    assert set(bus.vel_cap.values()) == {0} and set(bus.acc.values()) == {0}


# =============================================================== MS-7: plan clock, NaN speed
def test_plan_clock_survives_stalled_loop():
    for stall_s in (0.3, 1.0, 2.5):
        arm, bus = ready(sag=0.0)
        arm.move_tip(np.array([0.20, 0.10, 0.06]), 85.0, 0.0, 35.0)
        tool = arm.tip_tool()
        st = {"done": False, "dev": 0.0}
        n0 = len(bus.log)

        def mon(t, q_plan, q_meas, st=st, bus=bus, stall_s=stall_s):
            p = K.fk(q_meas, tool)[:3, 3]
            st["dev"] = max(st["dev"], abs(p[0] - 0.20), abs(p[2] - 0.06))
            if not st["done"] and t > 1.2:
                st["done"] = True
                bus.sleep(stall_s)  # the loop does not run for stall_s seconds
            return False

        res = arm.move_tip(np.array([0.20, -0.14, 0.06]), 85.0, 0.0, 30.0, straight=True, monitor=mon)
        assert st["done"]
        assert max_goal_step(bus, n0) < 2.0 * TPD, (stall_s, max_goal_step(bus, n0))  # no lunge (was 24 deg after 1 s)
        assert st["dev"] < 0.003, (stall_s, st["dev"])  # the tip stays on the checked line
        assert max(abs(e) for e in res["joint_err_deg"]) < 0.5


def test_nan_and_bad_speed_rejected():
    arm, bus = ready()
    q = arm.read_q()
    q[0] = math.radians(60)
    n0 = len(bus.log)
    for bad in (float("nan"), float("inf"), 0.0, -5.0):
        raises(SafetyError, lambda bad=bad: arm.move_joints(q, bad), "speed")
    raises(SafetyError, lambda: arm.move_joints(np.array([0.1, float("nan"), 0, 0, 0]), 20.0))
    assert len(bus.log) == n0


def test_speed_is_capped():
    arm, bus = ready(sag=0.0)
    q = arm.read_q()
    q[0] = math.radians(60)
    res = arm.move_joints(q, 500.0)
    assert res["speed_deg_s"] == 70.0
    last, tl, worst = {}, {}, 0.0
    for t, j, v in bus.log:
        if j in last and t > tl[j]:
            worst = max(worst, abs(v - last[j]) / (t - tl[j]))
        last[j], tl[j] = v, t
    assert worst / TPD < 75.0, worst / TPD


def test_frozen_clock_does_not_hang():
    arm, bus = ready()
    arm._sleep = lambda dt: None  # time never advances
    q = arm.read_q() + np.array([0.3, 0, 0, 0, 0])
    e = raises(MotionAborted, lambda: arm.move_joints(q, 25.0), "clock")
    assert e.info["reason"] == "clock"
    raises(SafetyError, lambda: arm.set_gripper(0.5), "clock")
    raises(SafetyError, arm.grasp_fast, "clock")


# =============================================================== MS-8: torque gate, hold, health
def test_require_torque_checks_all_six_and_profile():
    arm, bus = ready()
    q = arm.read_q() + np.array([0.1, 0, 0, 0, 0])
    bus.torque["gripper"] = 0  # limp gripper
    raises(SafetyError, lambda: arm.move_joints(q, 20.0), "gripper")
    arm.torque_on()
    bus.vel_cap["shoulder_pan"], bus.acc["shoulder_pan"] = 0, 0  # a servo that lost its caps
    raises(SafetyError, lambda: arm.move_joints(q, 20.0), "servo state lost")
    arm.torque_on()
    bus.p_gain["elbow_flex"] = 16
    raises(SafetyError, lambda: arm.move_joints(q, 20.0), "P gain 16")
    arm.torque_on()
    bus.power_cycle(("wrist_flex",))  # reset servo: limp, uncapped, boot gain
    raises(SafetyError, lambda: arm.move_joints(q, 20.0), "torque is off on wrist_flex")
    arm.torque_on()
    arm.move_joints(q, 20.0)  # torque_on repairs every case


def test_abort_keeps_gravity_compensation():
    arm, bus = ready(blocker=lambda pos: pos["shoulder_pan"] < 1800)
    arm.move_tip(np.array([0.22, -0.02, 0.03]), 90.0, 0.0, 35.0)
    z0 = tip_of(arm)[2]
    q = arm.read_q()
    q[0] = math.radians(-70)
    raises(MotionAborted, lambda: arm.move_joints(q, 40.0))
    bus.sleep(1.0)
    after = gmp(bus)
    assert abs(after["shoulder_lift"] + 12) <= 3 and abs(after["elbow_flex"] + 12) <= 3, after
    assert abs(tip_of(arm)[2] - z0) < 0.001  # used to sag 5 mm after an abort


def test_hold_is_verified_and_repeated():
    # the hold after an abort is one unacknowledged broadcast: if a servo misses it, it is written again
    arm, bus = ready(blocker=lambda pos: pos["shoulder_pan"] < 1800)
    orig = bus.sync_write
    lost = {"n": 0}

    def lossy(reg, values):
        if lost["armed"]:
            values = {j: v for j, v in values.items() if j != "shoulder_pan"}  # pan never hears the broadcast
            lost["n"] += 1
        return orig(reg, values)

    lost["armed"] = False
    bus.sync_write = lossy
    q = arm.read_q()
    q[0] = math.radians(-27.0)
    orig_hold = arm._hold

    def hold(q_meas, integ=None):
        lost["armed"] = True
        try:
            return orig_hold(q_meas, integ)
        finally:
            lost["armed"] = False

    arm._hold = hold
    raises(MotionAborted, lambda: arm.move_joints(q, 30.0))
    assert lost["n"] == 1
    assert abs(bus.goal["shoulder_pan"] - bus.pos["shoulder_pan"]) <= 6, gmp(bus)  # released all the same


def test_hold_here_releases_push_keeps_compensation():
    arm, bus = ready()
    arm.move_tip(np.array([0.24, 0.0, 0.04]), 90.0, 0.0, 35.0)
    bus.goal["shoulder_pan"] += 150  # some push that is not gravity
    arm._hold_here()
    g = gmp(bus)
    assert abs(g["shoulder_pan"]) <= 7 and abs(g["shoulder_lift"] + 12) <= 3, g


def test_health_is_checked_before_every_motion():
    arm, bus = ready()
    q = arm.read_q() + np.array([0.1, 0, 0, 0, 0])
    bus.temp["elbow_flex"] = 70
    raises(SafetyError, lambda: arm.move_joints(q, 20.0), "temperature")
    bus.temp["elbow_flex"] = 25
    bus.volt = 3.7
    raises(SafetyError, lambda: arm.move_joints(q, 20.0), "voltage")
    bus.volt = 4.1  # the loaded 5 V rail: fine for a motion, not for torque_on
    arm.move_joints(q, 20.0)
    raises(SafetyError, arm.torque_on, "voltage")
    bus.volt = 4.9
    bus.fault["wrist_roll"] = 32
    raises(SafetyError, lambda: arm.move_joints(q, 20.0), "overload")
    bus.fault["wrist_roll"] = 1  # voltage bit alone does not refuse a motion
    arm.move_joints(arm.read_q() + np.array([0.05, 0, 0, 0, 0]), 20.0)


# =============================================================== SP-3: fault bits abort a motion
def test_servo_fault_aborts_motion():
    for bit, name in ((32, "overload"), (4, "overheat"), (8, "overcurrent"), (2, "sensor")):
        arm, bus = ready()
        q = arm.read_q() + np.array([0.6, 0, 0, 0, 0])

        def mon(t, qp, qm, bus=bus, bit=bit):
            if t > 0.8:
                bus.fault["elbow_flex"] = bit
            return False

        e = raises(MotionAborted, lambda: arm.move_joints(q, 25.0, monitor=mon), name)
        assert e.info["reason"] == "servo_fault" and e.info["faults"] == {"elbow_flex": bit}
        p0 = dict(bus.pos)
        bus.sleep(1.0)
        assert all(abs(bus.pos[j] - p0[j]) < 8 for j in ARM), "the arm kept moving after the fault"
    # the voltage bit alone (5 V supply) does not abort
    arm, bus = ready()
    q = arm.read_q() + np.array([0.3, 0, 0, 0, 0])
    arm.move_joints(q, 25.0, monitor=lambda t, qp, qm: bus.fault.__setitem__("elbow_flex", 1))
    assert arm.state()["faults"] == {"elbow_flex": ["voltage"]}


def test_position_only_stream_fallback():
    # cfg stream_feedback = False: the 2-byte position read v1 used; fault bits still abort (reply error byte)
    arm, bus = ready(stream_feedback=False)
    q = arm.read_q() + np.array([0.6, 0, 0, 0, 0])
    res = arm.move_joints(q, 25.0)
    assert max(abs(e) for e in res["joint_err_deg"]) < 0.3 and res["min_volt"] is None and res["peak_load"] == 0
    e = raises(MotionAborted, lambda: arm.move_joints(arm.read_q() - np.array([0.6, 0, 0, 0, 0]), 25.0,
                                                      monitor=lambda t, qp, qm: bus.fault.__setitem__("elbow_flex", 32)))
    assert e.info["reason"] == "servo_fault"
    bus.fault["elbow_flex"] = 0
    arm2, bus2 = ready(blocker=lambda pos: pos["gripper"] < 2300, stream_feedback=False)
    arm2.set_gripper(raw=2500)
    g = arm2.grasp_fast()
    assert g["contact"] and abs(g["pos_raw"] - 2300) <= 3, g


# =============================================================== MS-9: gripper
def test_closed_clamp_is_touch_raw():
    arm, bus = ready()
    assert arm.gripper_raw_for(0.0) == 2050 and arm.gripper_raw_for(-1.0) == 2050
    assert arm.gripper_raw_for(1.0) == 3648 - 25
    res = arm.set_gripper(0.0)
    assert res["target_raw"] == 2050 and res["pos_raw"] == 2050 and not res["blocked"] and res["load"] == 0


def test_set_gripper_blocked_is_relieved():
    arm, bus = ready(blocker=lambda pos: pos["gripper"] < 2600)
    arm.set_gripper(0.9)
    res = arm.set_gripper(0.0)
    assert res["blocked"] is True and abs(res["pos_raw"] - 2600) <= 2, res
    assert res["hold_goal_raw"] == res["pos_raw"] - 40
    assert 0 < bus.pos["gripper"] - bus.goal["gripper"] <= 42  # bounded squeeze, not 550 ticks of stall
    # opening against something is relieved the same way
    arm2, bus2 = ready(blocker=lambda pos: pos["gripper"] > 2400)
    res2 = arm2.set_gripper(0.9)
    assert res2["blocked"] and res2["hold_goal_raw"] == res2["pos_raw"] + 40, res2


def test_set_gripper_raw_and_no_wait():
    arm, bus = ready()
    t0 = bus.now
    res = arm.set_gripper(raw=2700, wait=False)
    assert res == {"target_raw": 2700, "waited": False} and bus.now == t0  # returns at once
    assert bus.goal["gripper"] == 2700
    bus.sleep(1.0)
    assert abs(bus.pos["gripper"] - 2700) <= 1
    res = arm.set_gripper(0.9, raw=2300)  # raw overrides frac
    assert res["target_raw"] == 2300 and abs(res["pos_raw"] - 2300) <= 2 and not res["moving"]
    assert arm.set_gripper(raw=99999, wait=False)["target_raw"] == 3648 - 25  # clamped into the range
    raises(SafetyError, lambda: arm.set_gripper())
    res = arm.set_gripper(0.1, wait_s=0.2)  # a long stroke that is still under way is not "blocked"
    assert res["moving"] and not res["blocked"], res
    bus.torque["gripper"] = 0
    raises(SafetyError, lambda: arm.set_gripper(0.5), "torque is off")


def test_grasp_min_frac_guard_does_not_open():
    arm, bus = ready(blocker=lambda pos: pos["gripper"] < 2600)
    arm.set_gripper(0.9)
    assert arm.grasp()["contact"]
    p0, g0 = bus.pos["gripper"], bus.goal["gripper"]
    res = arm.grasp(min_frac=0.6)  # used to OPEN the gripper by 387 ticks
    assert res["contact"] is False and "already" in res["note"], res
    assert bus.pos["gripper"] == p0 and bus.goal["gripper"] == g0


def test_grasp_contact_found_on_last_step():
    # the jaw is stopped only a little above the closed position: the stepping loop reaches its floor
    # with a single stalled step, which used to be reported as "no contact" with the servo stalled
    arm, bus = ready(blocker=lambda pos: pos["gripper"] < 2080)
    arm.set_gripper(raw=2120)
    res = arm.grasp()
    assert res["contact"] is True, res
    assert bus.pos["gripper"] - bus.goal["gripper"] <= 42


# =============================================================== grasp_fast
def test_grasp_fast_contact():
    arm, bus = ready(blocker=lambda pos: pos["gripper"] < 2194, bus_kw={"servo_profile": True})
    arm.set_gripper(raw=2516)
    t0 = bus.now
    res = arm.grasp_fast(expected_raw=2194)
    assert res["contact"] is True and res["empty"] is False and res["stalled"], res
    assert abs(res["pos_raw"] - 2194) <= 2 and res["delta_raw"] == res["pos_raw"] - 2194
    assert res["hold_goal_raw"] == res["pos_raw"] - 40 and bus.goal["gripper"] == res["hold_goal_raw"]
    assert bus.now - t0 < 0.8 and res["duration_s"] < 0.8, res  # the stepping grasp() needs about 1.8 s
    assert abs(res["q_jaw_deg"] - (-11.41 + 144 * 360 / 4096)) < 0.3
    assert 15.0 < res["width_mm"] < 20.0, res  # geometric opening at the stall position
    for key in ("contact", "pos_raw", "hold_goal_raw", "q_jaw_deg", "width_mm", "empty", "duration_s", "load"):
        assert key in res


def test_grasp_fast_empty():
    arm, bus = ready(bus_kw={"servo_profile": True})
    arm.set_gripper(raw=2516)
    res = arm.grasp_fast()
    assert res["contact"] is False and res["empty"] is True and not res["timeout"], res
    assert res["pos_raw"] <= 2052 and bus.goal["gripper"] == 2050
    assert abs(bus.pos["gripper"] - bus.goal["gripper"]) <= 2  # no standing load on the closed jaws


def test_grasp_fast_thin_object_counts_as_empty():
    arm, bus = ready(blocker=lambda pos: pos["gripper"] < 2080)
    arm.set_gripper(raw=2400)
    res = arm.grasp_fast()
    assert res["contact"] is False and res["empty"] is True and res["stalled"] is True, res
    assert abs(bus.pos["gripper"] - bus.goal["gripper"]) <= 6  # relieved


def test_grasp_fast_never_squeezes_below_closed_clamp():
    arm, bus = ready(blocker=lambda pos: pos["gripper"] < 2100)
    arm.set_gripper(raw=2400)
    res = arm.grasp_fast(squeeze_ticks=80, empty_margin=20)
    assert res["contact"] and res["hold_goal_raw"] == 2050, res


def test_grasp_fast_already_holding():
    arm, bus = ready(blocker=lambda pos: pos["gripper"] < 2300)
    arm.set_gripper(raw=2305)
    res = arm.grasp_fast()
    assert res["contact"] and abs(res["pos_raw"] - 2300) <= 5 and res["duration_s"] < 0.5, res


def test_grasp_fast_timeout_on_yielding_object():
    # something soft that keeps giving way: the jaw is still creeping when the time is up
    class Creep(SimBus):
        def _advance_joint(self, j, dt):
            if j == "gripper" and self.torque[j] and self.pos[j] < 2400 and self.goal[j] < self.pos[j]:
                self.pos[j] = max(self.goal[j], self.pos[j] - 60.0 * dt)  # 60 ticks/s below 2400
                return
            super()._advance_joint(j, dt)

    arm, bus = ready(bus_cls=Creep)
    arm.set_gripper(raw=2600)
    res = arm.grasp_fast(timeout_s=1.0)
    assert res["timeout"] and res["contact"] and not res["empty"], res
    assert res["hold_goal_raw"] == res["pos_raw"] - 40 and res["moved_ticks"] > 200


def test_grasp_fast_gripper_fault_and_torque():
    arm, bus = ready(blocker=lambda pos: pos["gripper"] < 2300)
    arm.set_gripper(raw=2600)
    bus.fault["gripper"] = 32
    e = raises(MotionAborted, arm.grasp_fast, "overload")
    assert e.info["reason"] == "servo_fault"
    assert abs(bus.goal["gripper"] - bus.pos["gripper"]) <= 25  # not left squeezing
    bus.fault["gripper"] = 0
    bus.torque["gripper"] = 0
    raises(SafetyError, arm.grasp_fast, "torque is off")


def test_gripper_angle_mapping():
    arm, _ = make()
    assert abs(arm.gripper_q_deg(2050) + 11.41) < 1e-9
    assert abs(arm.gripper_q_deg(2194) - (-11.41 + 144 * 360 / 4096)) < 1e-9
    for q in (-11.41, 0.0, 12.5, 40.0):
        assert abs(arm.gripper_q_deg(arm.gripper_raw_for_q(q)) - q) < 0.05
    arm2, _ = make(gripper_touch_raw=2060)
    assert abs(arm2.gripper_q_deg(2060) + 11.41) < 1e-9 and arm2.gripper_raw_for_q(-11.41) == 2060


# =============================================================== KI-2: moving jaw
REAL_TIP = [-0.001, 0.0, 0.0068]  # fingertip point of the live config (0.5 mm past the fixed jaw's nose)


def test_moving_jaw_floor_check():
    arm, _ = make(tip_offset=REAL_TIP)
    limit = arm.cfg["floor_z"] + arm.cfg["floor_clearance"]
    # geometry: at pitch 70 / roll -150 the jaw opened 20 deg hangs about 12 mm below the fingertip
    q = arm.solve(np.array([0.22, 0.0, 0.05]), 70.0, -150.0)
    drop = arm._jaw_low_z(K.fk(q), [20.0]) - K.fk(q, arm.tip_tool())[2, 3]
    assert -0.016 < drop < -0.009, drop
    q_low = arm.solve(np.array([0.22, 0.0, limit + 0.006]), 70.0, -150.0)
    raises(SafetyError, lambda: arm.check_q(q_low, jaw_q_deg=20.0), "moving jaw")
    arm.check_q(q_low, jaw_q_deg=-11.0)  # closed jaw: fine
    arm.check_q(arm.solve(np.array([0.22, 0.0, limit + 0.006]), 70.0, 0.0), jaw_q_deg=20.0)  # jaw on the upper side
    # a vertical approach is unchanged: any opening may go down to the floor limit
    q_v = arm.solve(np.array([0.22, 0.0, limit + 0.0002]), 90.0, 30.0)
    for jq in (-11.0, 0.0, 8.0, 20.0, 45.0):
        arm.check_q(q_v, jaw_q_deg=jq)


def test_moving_jaw_check_uses_measured_and_commanded_opening():
    arm, bus = ready(tip_offset=REAL_TIP)
    limit = arm.cfg["floor_z"] + arm.cfg["floor_clearance"]
    target = np.array([0.22, 0.0, limit + 0.006])
    arm.move_tip(np.array([0.22, 0.0, 0.06]), 70.0, -150.0, 35.0)
    arm.set_gripper(raw=2100)  # nearly closed: the descent is fine
    arm.move_tip(target, 70.0, -150.0, 12.0, straight=True)
    arm.move_tip(np.array([0.22, 0.0, 0.06]), 70.0, -150.0, 20.0, straight=True)
    # opening the gripper DURING the descent is checked with the commanded opening
    wps = arm.plan_tip_path([(target, 70.0, -150.0)])
    raises(SafetyError, lambda: arm.run_path(wps, 12.0, gripper_goal_raw=2600), "moving jaw")
    arm.set_gripper(raw=2600)
    raises(SafetyError, lambda: arm.move_tip(target, 70.0, -150.0, 12.0, straight=True), "moving jaw")


def test_state_reports_jaw_tilt():
    arm, bus = ready()
    st = arm.state()
    assert abs(st["jaw_tilt_deg"]) < 30 and st["jaw_yaw_deg"] is not None
    assert st["p_gain"]["shoulder_lift"] == 32 and st["profile"]["elbow_flex"] == [1100, 40]
    assert st["speed_tier"] == "slow" and st["faults"] == {}
    arm.move_tip(np.array([0.30, 0.0, 0.10]), 20.0, 0.0, 35.0)  # nearly horizontal gripper, roll 0:
    st = arm.state()  # the jaw line is close to vertical, its "yaw" means nothing
    assert st["jaw_yaw_deg"] is None and st["jaw_tilt_deg"] > 60, st


# =============================================================== KI-3: wrist_roll limits, no silent clamp
def test_wrist_roll_limits_follow_zero_tick():
    arm, _ = make()
    lim = arm.q_limits()["wrist_roll"]
    assert abs(math.degrees(lim[0]) + 160) < 1e-6 and abs(math.degrees(lim[1]) - 160) < 1e-6
    zt = dict(DEFAULT_CONFIG["zero_ticks"], wrist_roll=984.0)
    arm2, _ = make(zero_ticks=zt)
    lo, hi = arm2.q_limits()["wrist_roll"]
    assert abs(lo - (45 - 984.0) / TPR) < 1e-9, math.degrees(lo)  # -82.5 deg: the servo cannot go further
    assert abs(math.degrees(hi) - 160) < 1e-6
    raises(SafetyError, lambda: arm2.solve(np.array([0.22, 0.0, 0.05]), 90.0, -150.0), "wrist_roll")
    arm2.solve(np.array([0.22, 0.0, 0.05]), 90.0, -80.0)


def test_write_q_never_clamps_a_setpoint_silently():
    arm, bus = ready()
    q = arm.read_q()
    q[0] = (LIMITS["shoulder_pan"][1] + 60 - 2047.0) / TPR  # 60 ticks beyond the servo's own limit
    n0 = len(bus.log)
    raises(SafetyError, lambda: arm._write_q(q), "outside the servo limits")
    assert len(bus.log) == n0
    q = arm.read_q()
    arm._write_q(q, np.array([0.0, -0.1, 0.0, 0.0, 0.0]))  # the bounded correction may be limited: no error
    # an arm that rests beyond a stored servo limit (pushed onto its stop) can still be driven back in
    arm2, bus2 = make()
    bus2.pos["shoulder_lift"] = bus2.goal["shoulder_lift"] = LIMITS["shoulder_lift"][0] - 12.0
    arm2.torque_on()
    res = arm2.move_joints(arm2.ready_q(), 30.0)
    assert max(abs(e) for e in res["joint_err_deg"]) < 0.5


# =============================================================== KI-6 / plan_tip_path
def test_straight_reorientation_on_short_segment():
    arm, bus = ready()
    arm.move_tip(np.array([0.22, 0.0, 0.05]), 90.0, 60.0, 35.0)
    # 10 mm straight with roll 60 -> -30 used to fail as an "IK branch jump of 30 deg"
    arm.move_tip(np.array([0.22, 0.0, 0.06]), 90.0, -30.0, 20.0, straight=True)
    assert abs(math.degrees(arm.read_q()[4]) + 30) < 0.5
    # zero-length segment with a pitch change of 30 deg
    wps = arm.plan_tip_path([((0.22, 0.0, 0.06), 60.0, -30.0)])
    assert len(wps) >= 1 + 8, len(wps)  # at least one waypoint per 4 deg
    tool = arm.tip_tool()
    assert max(np.linalg.norm(K.fk(q, tool)[:3, 3] - np.array([0.22, 0.0, 0.06])) for q in wps) < 2e-3


def test_plan_tip_path():
    arm, bus = ready()
    q0 = arm.read_q()
    pts = [((0.24, -0.05, 0.03), 85.0, 20.0), ((0.24, -0.05, 0.09), 85.0, 20.0), ((0.22, 0.08, 0.09), 80.0, -15.0)]
    n0 = len(bus.log)
    wps = arm.plan_tip_path(pts)
    assert len(bus.log) == n0  # planning does not move anything
    assert np.allclose(wps[0], q0)
    tool = arm.tip_tool()
    tips = np.array([K.fk(q, tool)[:3, 3] for q in wps])
    steps = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    assert steps.max() <= 0.0051, steps.max()  # >= 1 waypoint per 5 mm
    assert np.linalg.norm(tips[-1] - np.array([0.22, 0.08, 0.09])) < 1e-4
    # the last segment is a straight line in Cartesian space
    a, b = np.array([0.24, -0.05, 0.09]), np.array([0.22, 0.08, 0.09])
    i0 = int(np.argmin(np.linalg.norm(tips - a, axis=1)))
    for p in tips[i0:]:
        s = np.dot(p - a, b - a) / np.dot(b - a, b - a)
        assert np.linalg.norm(p - (a + s * (b - a))) < 2e-4
    # pitch / roll None keep the previous value; q_start plans from another pose
    wps2 = arm.plan_tip_path([((0.22, 0.08, 0.05), None, None)], q_start=wps[-1])
    assert np.allclose(wps2[0], wps[-1]) and abs(math.degrees(wps2[-1][4]) + 15) < 1e-6
    assert abs(K.pose_summary(wps2[-1], tool)["pitch_down_deg"] - 80.0) < 0.2
    raises(SafetyError, lambda: arm.plan_tip_path([((0.60, 0.0, 0.05), 45.0, 0.0)]), "IK failed")
    raises(SafetyError, lambda: arm.plan_tip_path([((0.2, float("nan"), 0.05), 45.0, 0.0)]))


# =============================================================== continuous paths, gripper concurrency
def _goal_rate(bus, start, joints=ARM):
    """[(t, largest goal rate over the joints in ticks/s)] from the goal log."""
    last, out = {}, {}
    for t, j, v in bus.log[start:]:
        if j not in joints:
            continue
        if j in last and t > last[j][0]:
            out[t] = max(out.get(t, 0.0), abs(v - last[j][1]) / (t - last[j][0]))
        last[j] = (t, v)
    return sorted(out.items())


def test_path_is_continuous_at_intermediate_waypoints():
    arm, bus = ready()
    arm.move_tip(np.array([0.24, -0.05, 0.03]), 85.0, 20.0, 35.0)
    pts = [((0.24, -0.05, 0.09), 85.0, 20.0), ((0.22, 0.08, 0.09), 80.0, -15.0), ((0.22, 0.08, 0.05), 80.0, -15.0)]
    n0, t0 = len(bus.log), bus.now
    res = arm.move_tip_path(pts, 25.0)
    rate = _goal_rate(bus, n0)
    T = res["plan_s"]
    mid = [r for t, r in rate if 0.2 * T < t - t0 < 0.8 * T]
    peak = max(r for _, r in rate)
    assert min(mid) > 0.2 * peak, (min(mid), peak)  # never slows to a stop between the points
    assert max(r for _, r in _goal_rate(bus, n0, ARM[:4])) / TPD < 28.0  # 25 deg/s plus tick quantisation
    assert max(r for _, r in _goal_rate(bus, n0, ("wrist_roll",))) / TPD < 1.6 * 28.0  # roll: its own scale
    assert res["blend_s"] > 0 and res["aborted"] is False
    assert max(abs(e) for e in res["joint_err_deg"]) < 0.3
    assert np.linalg.norm(tip_of(arm) - np.array([0.22, 0.08, 0.05])) < 0.003
    # the same three legs as separate moves stop twice and take longer
    arm2, bus2 = ready()
    arm2.move_tip(np.array([0.24, -0.05, 0.03]), 85.0, 20.0, 35.0)
    n1, t1 = len(bus2.log), bus2.now
    for p in pts:
        arm2.move_tip(np.array(p[0]), p[1], p[2], 25.0, straight=True)
    T2 = bus2.now - t1
    assert min(r for t, r in _goal_rate(bus2, n1) if 0.2 * T2 < t - t1 < 0.8 * T2) == 0.0  # stops in between
    assert res["duration_s"] < T2, (res["duration_s"], T2)


def test_corner_blend_is_bounded_and_checked():
    arm, bus = ready(sag=0.0)
    arm.move_tip(np.array([0.24, -0.05, 0.03]), 85.0, 20.0, 35.0)
    corner = np.array([0.24, -0.05, 0.09])
    pts = [(corner, 85.0, 20.0), ((0.22, 0.08, 0.09), 85.0, 20.0)]
    tool = arm.tip_tool()
    near = []
    res = arm.move_tip_path(pts, 25.0, monitor=lambda t, qp, qm: near.append(
        np.linalg.norm(K.fk(qp, tool)[:3, 3] - corner)) and False)
    assert 0.0005 < min(near) < 0.006, min(near)  # the corner is rounded by a few millimetres, not more
    # a two-point move has no corner and is not blended (identical to the validated motion)
    res2 = arm.move_tip(np.array([0.24, -0.05, 0.05]), 85.0, 20.0, 25.0)
    assert res2["blend_s"] == 0.0
    # the blended reference is what gets checked: a corner cut that enters a keep-out is refused
    q0 = arm.read_q()
    q1 = q0 + np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    q2 = q1 + np.array([0.0, -0.35, 0.0, 0.0, 0.0])
    plan, duration, total, blend = arm._path_plan([q0, q1, q2], 40.0)  # (min-jerk law)
    assert blend > 0
    ts = np.linspace(0.0, duration, 400)
    tips = np.array([K.fk(plan(t), tool)[:3, 3] for t in ts])
    p_corner = K.fk(q1, tool)[:3, 3]
    p_cut = tips[int(np.argmin(np.linalg.norm(tips - p_corner, axis=1)))]  # where the rounded path passes the corner
    poly = np.array([K.fk(q0 + (q1 - q0) * a, tool)[:3, 3] for a in np.linspace(0, 1, 300)]
                    + [K.fk(q1 + (q2 - q1) * a, tool)[:3, 3] for a in np.linspace(0, 1, 300)])
    gap = float(np.min(np.linalg.norm(poly - p_cut, axis=1)))
    assert 0.002 < gap < 0.02, gap  # the cut leaves the polyline by a few millimetres
    r = 0.0012
    arm.cfg["keepouts"]["post"] = [p_cut[0] - r, p_cut[0] + r, p_cut[1] - r, p_cut[1] + r, p_cut[2] - r, p_cut[2] + r]
    for q in (q1, q2):
        arm.check_q(q)  # the waypoints themselves (and the straight polyline) stay clear of it
    n0 = len(bus.log)
    raises(SafetyError, lambda: arm.run_path([q0, q1, q2], 40.0), "blended path")
    assert len(bus.log) == n0


def test_gripper_moves_while_arm_moves():
    arm, bus = ready()
    arm.set_gripper(raw=2100)
    q = arm.read_q() + np.array([0.7, 0.0, 0.0, 0.0, 0.0])
    seen = []
    n0, t0 = len(bus.log), bus.now
    res = arm.move_joints(q, 25.0, gripper_goal_raw=2600, monitor=lambda t, qp, qm: seen.append(
        (t, bus.pos["gripper"], abs(qm[0] - q[0]))) and False)
    writes = [(t, v) for t, j, v in bus.log[n0:] if j == "gripper"]
    assert writes == [(t0, 2600)], writes  # written once, when the motion starts
    t_open = min(t for t, g, _ in seen if g >= 2598)
    assert t_open < 0.5 * res["plan_s"], (t_open, res["plan_s"])  # open long before the arm arrives
    arrived_then = [d for t, g, d in seen if t <= t_open][-1]
    assert arrived_then > math.radians(20), "the arm had not moved while the gripper opened"
    assert res["gripper_goal_raw"] == 2600 and abs(res["gripper_pos_raw"] - 2600) <= 2
    # an out-of-range goal is clamped, a non-number refused before any motion
    assert arm.move_joints(arm.read_q(), 20.0, gripper_goal_raw=10)["gripper_goal_raw"] == 2050
    raises(SafetyError, lambda: arm.move_joints(q, 20.0, gripper_goal_raw=float("nan")))


def test_one_bus_read_per_cycle():
    arm, bus = ready()
    q = arm.read_q() + np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    r0, w0, t0 = bus.io["reads"], bus.io["writes"], bus.now
    res = arm.run_path([arm.read_q(), q], 25.0, settle=False)
    cycles = round((bus.now - t0) / 0.02) + 1
    reads, writes = bus.io["reads"] - r0, bus.io["writes"] - w0
    # per cycle: one feedback read + one goal write; per motion: one position read (ours), one snapshot
    # (6 block reads), 5 gain reads, one more feedback read
    assert reads <= cycles + 14 and writes <= cycles + 1, (cycles, reads, writes)
    assert res["peak_load"] > 0 and res["min_volt"] == 4.9 and res["max_temp"] == 25


def test_settle_is_only_as_long_as_needed():
    arm, bus = ready()
    q0 = arm.read_q()
    a = arm.move_joints(q0 + np.array([0.3, 0, 0, 0, 0]), 25.0)  # settle=True: leaves once inside tolerance
    assert a["settled"] is True and a["settle_s"] <= 0.45, a
    b = arm.move_joints(q0, 25.0, settle=False)
    assert b["settle_s"] == 0.0 and b["settled"] is None, b
    c = arm.move_joints(q0 + np.array([0.3, 0, 0, 0, 0]), 25.0, settle=0.3)
    assert c["settle_s"] <= 0.31
    d = arm.move_joints(q0, 25.0, tol_deg=2.0)  # coarse tolerance: no waiting beyond the dwell
    assert d["settle_s"] <= 0.25
    raises(SafetyError, lambda: arm.move_joints(q0, 25.0, settle=-1.0))
    for key in ("q_meas", "joint_err_deg", "corr_deg", "duration_s", "settle_s", "peak_err_deg", "aborted"):
        assert key in a
    assert a["aborted"] is False and a["peak_err_deg"] >= 0 and a["duration_s"] >= a["plan_s"]


def test_unsettled_fast_move_still_checks_arrival():
    # settle=False must not hide a blocked target either
    arm, bus = ready(blocker=lambda pos: pos["shoulder_pan"] < 1800)
    q = arm.read_q()
    q[0] = math.radians(-28.0)
    e = raises(MotionAborted, lambda: arm.move_joints(q, 30.0, settle=False))
    assert e.info["reason"] == "target_not_reached"
    assert abs(bus.goal["shoulder_pan"] - bus.pos["shoulder_pan"]) <= 6


def test_path_must_start_at_measured_pose():
    arm, bus = ready()
    q = arm.read_q()
    n0 = len(bus.log)
    raises(SafetyError, lambda: arm.run_path([q + 0.2, q + 0.3], 20.0), "measured pose")
    raises(SafetyError, lambda: arm.run_path([], 20.0))
    assert len(bus.log) == n0
    arm.run_path([q + np.array([0.004, 0, 0, 0, 0]), q + 0.1], 20.0)  # a slightly stale start is re-anchored


# =============================================================== time laws, per-joint speed scale
def test_time_laws():
    arm, _ = make()
    for law, total_deg, speed in (("minjerk", 60.0, 25.0), ("cruise", 60.0, 25.0), ("cruise", 200.0, 70.0),
                                  ("cruise", 10.0, 25.0), ("minjerk", 0.5, 25.0)):
        total = math.radians(total_deg)
        T, s = arm._time_law(total, speed, law)
        ts = np.linspace(0.0, T, 2001)
        ss = np.array([s(t) for t in ts])
        assert abs(ss[0]) < 1e-12 and abs(ss[-1] - total) < 1e-9 and np.all(np.diff(ss) >= -1e-12), law
        v = np.degrees(np.diff(ss) / np.diff(ts))
        assert v.max() <= speed * 1.001, (law, v.max())  # the requested speed is a true bound
        acc = np.abs(np.diff(v) / np.diff(ts)[1:])
        if law == "cruise" and total_deg >= 60:
            assert abs(T - (total_deg / speed + 0.5)) < 1e-9  # ramp up, hold the speed, ramp down
            assert v.max() > 0.999 * speed and acc.max() < 1.9 * speed / 0.5 * 1.05
        else:
            assert abs(T - max(0.35, 1.875 * total_deg / speed)) < 1e-9  # min-jerk (also for short cruise paths)
    assert arm._time_law(math.radians(60), 25.0, "cruise")[0] < 0.7 * arm._time_law(math.radians(60), 25.0, "minjerk")[0]


def test_cruise_tier():
    arm, bus = ready()
    assert arm.speeds("slow_cruise", allow_unvalidated=True)["time_law"] == "cruise"
    assert arm.speeds()["time_law"] == "minjerk"
    raises(SafetyError, lambda: arm.speeds("slow_cruise"), "not been validated")
    q0 = arm.read_q()
    q1 = q0 + np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    a = arm.move_joints(q1, 25.0)
    assert a["time_law"] == "minjerk" and abs(a["plan_s"] - 1.875 * math.degrees(1.0) / 25.0) < 1e-3
    n0 = len(bus.log)
    b = arm.move_joints(q0, 25.0, time_law="cruise")  # explicit per call
    assert b["time_law"] == "cruise" and abs(b["plan_s"] - (math.degrees(1.0) / 25.0 + 0.5)) < 0.02
    assert max(r for _, r in _goal_rate(bus, n0)) / TPD < 28.0 and max(abs(e) for e in b["joint_err_deg"]) < 0.3
    raises(SafetyError, lambda: arm.move_joints(q1, 25.0, time_law="warp"), "time_law")
    arm.cfg["speed_tiers"]["odd"] = {"transit": 25.0, "approach": 12.0, "precise": 8.0, "time_law": "bang"}
    raises(SafetyError, lambda: arm.speeds("odd", allow_unvalidated=True), "time_law")
    # as the configured tier (after a hardware bench) every move uses it without being asked
    arm2, bus2 = ready(speed_tier="slow_cruise", speed_tiers_validated=["slow", "slow_cruise"])
    c = arm2.move_joints(arm2.read_q() + np.array([1.0, 0.0, 0.0, 0.0, 0.0]), arm2.speeds()["transit"])
    assert c["time_law"] == "cruise" and c["plan_s"] < 0.7 * a["plan_s"]


def test_joint_speed_scale():
    arm, bus = ready(sag=0.0)
    sp = arm.speeds()
    assert sp["joint_speed_scale"] == {"shoulder_pan": 1.0, "shoulder_lift": 1.0, "elbow_flex": 1.0,
                                       "wrist_flex": 1.0, "wrist_roll": 1.6}
    q0 = arm.read_q()
    # a roll-only move takes 1 / 1.6 of the time of the same angle on another joint
    n0 = len(bus.log)
    r_roll = arm.move_joints(q0 + np.array([0, 0, 0, 0, 1.0]), 25.0)
    r_pan = arm.move_joints(arm.read_q() + np.array([1.0, 0, 0, 0, 0]), 25.0)
    assert abs(r_roll["plan_s"] - r_pan["plan_s"] / 1.6) < 2e-3, (r_roll["plan_s"], r_pan["plan_s"])
    assert r_roll["joint_speed_scale"]["wrist_roll"] == 1.6 and r_roll["joint_speed_scale"]["shoulder_pan"] == 1.0
    roll_rate = max(r for _, r in _goal_rate(bus, n0, ("wrist_roll",))) / TPD
    assert 36.0 < roll_rate < 43.0, roll_rate  # 40 deg/s, as the v1 roll flips on the real arm
    assert max(abs(e) for e in r_roll["joint_err_deg"]) < 0.3
    # a mixed move that another joint dominates is unchanged
    q = arm.read_q()
    mixed = arm.move_joints(q + np.array([-1.0, 0, 0, 0, -0.6]), 25.0)
    assert abs(mixed["plan_s"] - r_pan["plan_s"]) < 2e-3
    # ... and one that roll dominates is limited by roll at ITS speed
    q = arm.read_q()
    n1 = len(bus.log)
    mixed2 = arm.move_joints(q + np.array([0.3, 0, 0, 0, 1.2]), 25.0)
    assert abs(mixed2["plan_s"] - 1.875 * math.degrees(1.2) / 40.0) < 2e-3
    assert max(r for _, r in _goal_rate(bus, n1, ARM[:4])) / TPD < 15.0  # pan: 10 deg/s plus tick quantisation
    # the scaled joint never exceeds max_joint_speed_deg_s or the servo-side cap / 1.25
    q = arm.read_q()
    fast = arm.move_joints(q + np.array([0, 0, 0, 0, -1.0]), 60.0)
    assert abs(fast["joint_speed_scale"]["wrist_roll"] - 70.0 / 60.0) < 1e-3, fast["joint_speed_scale"]
    n2 = len(bus.log)
    top = arm.move_joints(arm.read_q() + np.array([0, 0, 0, 0, 1.0]), 70.0)
    assert top["joint_speed_scale"]["wrist_roll"] == 1.0
    assert max(r for _, r in _goal_rate(bus, n2, ("wrist_roll",))) / TPD < 75.0
    arm3, bus3 = ready(speed_tier="normal", speed_tiers_validated=["slow", "normal"])  # cap 770 steps/s = 67.7 deg/s
    r3 = arm3.move_joints(arm3.read_q() + np.array([0, 0, 0, 0, 1.0]), 45.0)
    assert abs(r3["joint_speed_scale"]["wrist_roll"] - 770 / TPD / 1.25 / 45.0) < 1e-3  # 1.2, not 1.6
    # configuration: off, per tier, and nonsense
    arm4, _ = ready(joint_speed_scale={})
    r4 = arm4.move_joints(arm4.read_q() + np.array([0, 0, 0, 0, 1.0]), 25.0)
    assert abs(r4["plan_s"] - r_pan["plan_s"]) < 2e-3 and r4["joint_speed_scale"]["wrist_roll"] == 1.0
    arm.cfg["speed_tiers"]["slow"]["joint_speed_scale"] = {"wrist_roll": 1.2, "shoulder_pan": 0.5}
    assert arm.speeds()["joint_speed_scale"]["wrist_roll"] == 1.2 and arm.speeds()["joint_speed_scale"]["shoulder_pan"] == 0.5
    for bad in ({"wrist_roll": 9.0}, {"wrist_roll": float("nan")}, {"gripper": 1.5}, {"wrist_roll": "fast"}):
        arm.cfg["speed_tiers"]["slow"]["joint_speed_scale"] = bad
        raises(SafetyError, arm.speeds, "joint_speed_scale")


def test_bench_validates_the_roll_scale():
    arm, bus = ready()
    res = arm.bench_motion("slow")
    names = [m["name"] for m in res["moves"]]
    assert "roll_swing_90" in names and res["pass"] is True and res["passed"] is True, res
    swing = res["moves"][names.index("roll_swing_90")]
    assert swing["duration_s"] < 1.875 * 90.0 / 25.0 * 0.75  # roll ran above the tier speed
    assert res["joint_speed_scale"]["wrist_roll"] == 1.6 and res["time_law"] == "minjerk"


# =============================================================== speed tiers
def test_speed_tiers():
    arm, bus = make()
    s = arm.speeds()
    assert s["tier"] == "slow" and s["validated"] is True
    assert (s["transit"], s["approach"], s["precise"]) == (25.0, 12.0, 8.0)
    n = arm.speeds("normal", allow_unvalidated=True)
    assert (n["transit"], n["approach"], n["precise"], n["validated"]) == (45.0, 20.0, 10.0, False)
    f = arm.speeds("fast", allow_unvalidated=True)
    assert (f["transit"], f["approach"], f["precise"]) == (70.0, 30.0, 12.0)
    for t in (n, f):  # servo-side cap about 1.5 x the planned peak
        assert 1.4 <= t["servo_velocity_cap"] / (t["transit"] * TPD) <= 1.6, t
    raises(SafetyError, lambda: arm.speeds("normal"), "not been validated")
    raises(SafetyError, lambda: arm.speeds("fast"), "not been validated")
    raises(SafetyError, lambda: arm.speeds("warp"), "unknown speed tier")
    arm.cfg["speed_tiers"]["bad"] = {"transit": 500.0, "approach": 10.0, "precise": 5.0}
    raises(SafetyError, lambda: arm.speeds("bad", allow_unvalidated=True), "outside")
    arm.cfg["speed_tiers"]["bad"] = {"transit": 30.0, "approach": 10.0}
    raises(SafetyError, lambda: arm.speeds("bad", allow_unvalidated=True), "precise")
    arm.cfg["speed_tiers"]["bad"] = {"transit": 60.0, "approach": 10.0, "precise": 5.0, "servo_velocity_cap": 600}
    raises(SafetyError, lambda: arm.speeds("bad", allow_unvalidated=True), "servo_velocity_cap")


def test_slow_tier_is_what_ran_on_the_arm():
    arm, bus = make()
    calls = []
    orig = bus.torque_on
    bus.torque_on = lambda joints=JOINTS, **kw: (calls.append((tuple(joints), kw)), orig(joints, **kw))[1]
    arm.torque_on()
    assert calls == [(tuple(JOINTS), {"velocity": 1100, "acceleration": 40})], calls  # one call, as on 2026-11-10
    assert set(bus.vel_cap.values()) == {1100} and set(bus.acc.values()) == {40}
    assert [bus.p_gain[j] for j in JOINTS] == [32, 32, 32, 32, 32, 16]


def test_unvalidated_tier_is_refused_everywhere():
    arm, bus = make(speed_tier="fast")
    raises(SafetyError, arm.torque_on, "not been validated")
    assert not any(bus.torque.values())
    arm, bus = make(speed_tier="normal", speed_tiers_validated=["slow", "normal"])  # after a hardware bench
    res = arm.torque_on()
    assert res["tier"] == "normal" and res["servo_velocity_cap"] == 770
    assert all(bus.vel_cap[j] == 770 and bus.acc[j] == 130 for j in ARM)
    assert bus.vel_cap["gripper"] == 1100 and bus.acc["gripper"] == 40  # the gripper keeps its validated profile
    arm.move_joints(arm.ready_q(), arm.speeds()["transit"])
    # a harness configured for another tier refuses to drive servos that carry this one
    arm2 = Arm(bus=bus, cfg=json.loads(json.dumps(DEFAULT_CONFIG))).open()
    raises(SafetyError, lambda: arm2.move_joints(arm2.read_q(), 20.0), "servo state lost or changed")
    arm2.torque_on()
    assert all(bus.vel_cap[j] == 1100 and bus.acc[j] == 40 for j in JOINTS)


def test_load_config_keeps_working_with_the_live_file():
    path = R.CONFIG_PATH
    before = open(path, "rb").read() if os.path.exists(path) else None
    cfg = R.load_config()
    assert cfg["speed_tier"] in cfg["speed_tiers"] and cfg["speed_tier"] in cfg["speed_tiers_validated"]
    assert {"slow", "normal", "fast"} <= set(cfg["speed_tiers"]) and len(cfg["ready"]["xyz"]) == 3
    live = Arm(bus=SimBus(), cfg=cfg).open()
    sp = live.speeds()  # the configured tier is usable as it stands
    assert sp["validated"] and sp["precise"] <= sp["approach"] <= sp["transit"] <= cfg["max_joint_speed_deg_s"]
    live.check_q(live.ready_q())
    if before is not None:
        user = json.loads(before)
        for k in ("floor_z", "tip_offset", "table_z"):
            if k in user:
                assert cfg[k] == user[k]
        assert set(DEFAULT_CONFIG["keepouts"]) <= set(cfg["keepouts"])
        assert open(path, "rb").read() == before  # never written
    # nested overrides are merged, not replaced
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "config.json")
        with open(tmp, "w") as f:
            json.dump({"speed_tiers": {"normal": {"transit": 40.0}}, "bench": {"peak_err_deg": 5.0}}, f)
        c = R.load_config(tmp)
        assert c["speed_tiers"]["normal"] == {"transit": 40.0, "approach": 20.0, "precise": 10.0,
                                              "servo_velocity_cap": 770, "servo_acceleration": 130}
        assert c["bench"]["peak_err_deg"] == 5.0 and c["bench"]["overshoot_deg"] == 1.5
        assert c["keepouts"] == DEFAULT_CONFIG["keepouts"] and c["p_gain"] == 32
    arm = Arm(bus=SimBus(), cfg={k: v for k, v in DEFAULT_CONFIG.items() if k not in ("speed_tiers", "ready")}).open()
    assert arm.speeds()["transit"] == 25.0 and arm.ready_q().shape == (5,)  # an old config dict still works


# =============================================================== ready / rest poses
def test_ready_and_rest_poses():
    arm, bus = make()
    q = arm.ready_q()
    assert np.linalg.norm(K.fk(q, arm.tip_tool())[:3, 3] - np.array([0.20, 0.0, 0.12])) < 1e-4
    assert abs(K.pose_summary(q, arm.tip_tool())["pitch_down_deg"] - 70.0) < 0.1
    rest = arm.rest_q()
    lim = arm.q_limits()
    true_rest = arm.raw_to_q({j: v for j, v in bus.pos.items() if j in ARM})
    for i, j in enumerate(ARM):
        assert lim[j][0] < rest[i] < lim[j][1]
        assert abs(rest[i] - true_rest[i]) < math.radians(2.5)
    arm.check_q(rest, allow_contact=True)


# =============================================================== bench_motion
def test_bench_refuses_unless_torqued_and_ready():
    arm, bus = make()
    raises(SafetyError, lambda: arm.bench_motion("slow"), "torque is off")
    arm.torque_on()
    n0 = len(bus.log)
    raises(SafetyError, lambda: arm.bench_motion("slow"), "ready pose")
    raises(SafetyError, lambda: arm.bench_motion("warp"), "unknown speed tier")
    assert len(bus.log) == n0


def test_bench_motion_slow_passes_in_sim():
    arm, bus = ready()
    res = arm.bench_motion("slow")
    assert res["pass"] is True and not res["aborted"] and res["fails"] == [], res
    assert [m["name"] for m in res["moves"]] == [m["name"] for m in arm.bench_moves()]
    assert all(m["ok"] for m in res["moves"])
    for key in ("peak_err_deg", "overshoot_deg", "settle_s", "settle_1deg_s", "peak_load", "temp_rise_c", "min_volt",
                "duration_s"):
        assert res[key] is not None, key
    assert np.max(np.abs(arm.read_q() - arm.ready_q())) < math.radians(0.6)  # ends where it started
    tool = arm.tip_tool()
    for m in arm.bench_moves():  # every pose is at least 9 cm above the table
        for xyz, *_ in ([m["to"]] if m["kind"] == "joints" else m["points"]):
            assert xyz[2] >= 0.07 - 1e-9


def test_bench_runs_unvalidated_tier_and_restores_profile():
    arm, bus = ready()
    arm.move_joints(arm.read_q() + np.array([0, 0, 0, 0, -1.1]), 25.0)  # stand-by with the wrist rolled: allowed
    seen = set()
    orig = bus.sync_write
    bus.sync_write = lambda reg, values: (seen.add((bus.vel_cap["elbow_flex"], bus.acc["elbow_flex"])), orig(reg, values))[1]
    res = arm.bench_motion("fast")
    assert seen == {(1200, 254)}, seen  # the tier's own servo profile was in force during every move
    assert res["tier"] == "fast" and res["validated"] is False and res["speeds"]["transit"] == 70.0
    assert all(bus.vel_cap[j] == 1100 and bus.acc[j] == 40 for j in JOINTS)  # back to the validated profile
    raises(SafetyError, lambda: arm.speeds("fast"), "not been validated")  # a pass does not unlock the tier
    assert arm.cfg["speed_tiers_validated"] == ["slow"]
    arm.move_joints(arm.ready_q(), arm.speeds()["transit"])  # normal operation continues


def test_bench_stops_at_first_failed_move():
    arm, bus = ready()
    arm.cfg["bench"] = dict(DEFAULT_CONFIG["bench"], peak_err_deg=0.05)
    res = arm.bench_motion("normal")
    assert res["pass"] is False and len(res["moves"]) == 1 and res["moves"][0]["ok"] is False, res
    assert "peak_err_deg" in res["fails"][0] and res["returned_to_ready"] is True
    assert np.max(np.abs(arm.read_q() - arm.ready_q())) < math.radians(0.6)
    assert all(bus.vel_cap[j] == 1100 for j in JOINTS)


def test_bench_early_abort():
    # supply sag during the second move
    arm, bus = ready()
    orig = bus.feedback
    state = {"n": 0}

    def fb(joints=JOINTS):
        state["n"] += 1
        if state["n"] == 420:
            bus.volt = 3.6
        return orig(joints)

    bus.feedback = fb
    res = arm.bench_motion("slow")
    assert res["pass"] is False and res["aborted"] is True and "voltage" in res["reason"], res
    assert all(bus.vel_cap[j] == 1100 for j in JOINTS)
    # a servo fault: stop, hold, do not try to drive back
    arm, bus = ready()
    orig2 = bus.feedback
    state2 = {"n": 0}

    def fb2(joints=JOINTS):
        state2["n"] += 1
        if state2["n"] == 100:
            bus.fault["shoulder_lift"] = 32
        return orig2(joints)

    bus.feedback = fb2
    res = arm.bench_motion("slow")
    assert res["pass"] is False and res["aborted"] and "overload" in res["reason"], res
    assert "returned_to_ready" not in res


# ------------------------------------------------------------------ starting up (added 2026-11-13 with the tools/ layout)
def _refused(fn, *needles):
    try:
        fn()
    except SafetyError as e:
        assert all(n in str(e) for n in needles), (needles, str(e))
        return str(e)
    raise AssertionError("no SafetyError")


def test_real_arm_needs_the_rig_config():
    """Arm() with the real bus and the configuration read from the file is refused BEFORE the servo port is opened
    when that file is missing, unreadable or lacks the rig's measurements (the built-in numbers were measured on
    another arm), and when a file of the old flat layout lies in the project folder (it would be ignored).
    An explicit bus or cfg is not affected."""
    opened = []

    class NoBus:
        def __init__(self, *a, **k):
            opened.append(1)
            raise BusError("stand-in: this test never opens the servo port")

    rig = dict(json.loads(json.dumps(DEFAULT_CONFIG)), table_z=-0.0015, floor_z=-0.0055, tip_offset=[-0.001, 0.0, 0.0068])
    assert all(rig.get(k) is not None for k in R.RIG_KEYS), R.RIG_KEYS
    saved = (R.Bus, R.CONFIG_PATH, tazuna_paths.ROOT, os.environ.get("FD_CONFIG"))
    R.Bus = NoBus
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FD_CONFIG"] = os.path.join(tmp, "nowhere.json")
            _refused(Arm, "configuration file is missing", "nowhere.json")
            del os.environ["FD_CONFIG"]
            R.CONFIG_PATH = os.path.join(tmp, "config", "config.json")
            _refused(Arm, "configuration file is missing", os.path.join("config", "config.json"))
            _refused(R.require_rig_config, "configuration file is missing")
            os.makedirs(os.path.join(tmp, "config"))
            tazuna_paths.ROOT = tmp
            with open(R.CONFIG_PATH, "w") as f:
                f.write("{ not json")
            _refused(Arm, "cannot be read")
            for key in R.RIG_KEYS:              # a file that a `config --set` could have made: some keys, not the rig's
                with open(R.CONFIG_PATH, "w") as f:
                    json.dump({k: v for k, v in rig.items() if k != key}, f)
                _refused(Arm, "has no " + key, "another arm")
            with open(R.CONFIG_PATH, "w") as f:
                json.dump({"speed_tier": "slow"}, f)
            _refused(Arm, "zero_ticks", "tip_offset", "floor_z", "table_z")
            with open(R.CONFIG_PATH, "w") as f:
                json.dump(rig, f)
            for stale in R.OLD_LAYOUT_FILES:
                with open(os.path.join(tmp, stale), "w") as f:
                    f.write("{}")
                _refused(Arm, stale, "config/")
                os.remove(os.path.join(tmp, stale))
            assert opened == [], "the servo port was asked for although the configuration was refused"
            assert R.require_rig_config() == R.CONFIG_PATH
            try:                                    # with the file in place the bus is opened (here: the stand-in)
                Arm()
                raise AssertionError("the stand-in bus was not asked for")
            except BusError:
                assert opened == [1]
            R.CONFIG_PATH = os.path.join(tmp, "missing.json")          # simulation and tests pass bus / cfg themselves
            assert Arm(bus=SimBus(), cfg=json.loads(json.dumps(DEFAULT_CONFIG))).cfg["zero_ticks"]
            assert Arm(bus=SimBus()).cfg["zero_ticks"]
    finally:
        R.Bus, R.CONFIG_PATH, tazuna_paths.ROOT = saved[:3]
        if saved[3] is None:
            os.environ.pop("FD_CONFIG", None)
        else:
            os.environ["FD_CONFIG"] = saved[3]


def test_sim_never_changes_the_rig_files():
    """`fd --sim` with a command that writes the rig's configuration or calibration, or changes the real camera, is
    refused (these commands do not use the arm, so the flag only looked safe); the read-only forms still work.
    config-changing commands never create the file and never write beside $FD_CONFIG."""
    import contextlib
    import io

    import camera as C
    import fd
    touched = []

    def recorder(name):
        return lambda *a, **k: touched.append(name)

    saved = (fd.edit_config_file, fd._append_json, C.CameraModel.save, C.grab, C.lock_settings, C.restore_auto, fd.RUNS,
             R.CONFIG_PATH, os.environ.get("FD_CONFIG"), fd.PROBES)
    real_edit = fd.edit_config_file

    def run(argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fd.main(argv)
        return rc, json.loads([ln for ln in buf.getvalue().splitlines() if ln.strip()][-1])

    with tempfile.TemporaryDirectory() as tmp:
        try:
            fd.edit_config_file, fd._append_json = recorder("config"), recorder("json")
            C.CameraModel.save, C.grab = recorder("camera model"), recorder("photo")
            C.lock_settings, C.restore_auto = recorder("camera lock"), recorder("camera auto")
            fd.RUNS = tmp
            fd.PROBES = os.path.join(tmp, "probes.json")      # (the rig's own probes file may have been moved away)
            with open(fd.PROBES, "w") as f:
                json.dump([{"xy": [0.17, 0.0], "z": -0.002}, {"xy": [0.14, -0.10], "z": -0.003}, {"xy": [0.20, -0.17], "z": -0.002}], f)
            os.environ.pop("FD_CONFIG", None)
            for argv in (["--sim", "config", "--set", "speed_tier=\"slow\""], ["--sim", "table-fit"], ["--sim", "calib-roll"],
                         ["--sim", "calib-fit"], ["--sim", "ref"], ["--sim", "camera", "auto"], ["--sim", "camera", "lock"]):
                rc, res = run(argv)
                assert rc == 1 and res["ok"] is False and res["error"] == "SafetyError" and "--sim" in res["message"], (argv, res)
            assert touched == [], touched
            rc, res = run(["--sim", "config"])                       # reading is fine
            assert rc == 0 and res["ok"] and "zero_ticks" in res, res
            rc, res = run(["--sim", "table-fit", "--dry"])           # so is looking at a fit without saving
            assert rc == 0 and res["ok"] and res["saved"] is False and touched == [], (res, touched)
            # table-fit on the real rig: its probed heights read too deep, so it saves only when told to
            rc, res = run(["table-fit"])
            assert rc == 1 and res["error"] == "SafetyError" and "--force" in res["message"] and touched == [], (res, touched)
            rc, res = run(["table-fit", "--force"])
            assert rc == 0 and res["saved"] is True and touched == ["config"], (res, touched)
            touched.clear()
            # the function that writes the configuration: no file out of nothing, nothing beside $FD_CONFIG
            fd.edit_config_file = real_edit
            R.CONFIG_PATH = os.path.join(tmp, "config.json")
            _refused(lambda: fd.edit_config_file(lambda raw: raw.update(x=1)), "is missing", "does not create")
            assert not os.path.exists(R.CONFIG_PATH)
            with open(R.CONFIG_PATH, "w") as f:
                json.dump({"speed_tier": "slow"}, f)
            os.environ["FD_CONFIG"] = os.path.join(tmp, "other.json")
            _refused(lambda: fd.edit_config_file(lambda raw: raw.update(x=1)), "FD_CONFIG is set")
            with open(R.CONFIG_PATH) as f:
                assert json.load(f) == {"speed_tier": "slow"}
        finally:
            (fd.edit_config_file, fd._append_json, C.CameraModel.save, C.grab, C.lock_settings, C.restore_auto, fd.RUNS,
             R.CONFIG_PATH) = saved[:8]
            fd.PROBES = saved[9]
            if saved[8] is None:
                os.environ.pop("FD_CONFIG", None)
            else:
                os.environ["FD_CONFIG"] = saved[8]


def test_fd_answers_json_when_a_module_is_missing():
    """A Python without one of the packages (or a copy with a harness file missing) still gets ONE JSON answer and
    exit code 1 from fd, and the message says which of the two it is."""
    import subprocess
    import sys
    fd_py = os.path.join(tazuna_paths.TOOLS, "cli", "fd.py")
    code = ("import runpy, sys\n"
            "sys.modules[sys.argv[1]] = None\n"          # makes `import <name>` fail the way a missing package does
            "fd = sys.argv[2]\n"
            "sys.argv = [fd, '--sim', 'state']\n"
            "runpy.run_path(fd, run_name='__main__')\n")
    for name, third_party in (("numpy", True), ("cv2", True), ("scservo_sdk", True), ("serial", True), ("kinematics", False)):
        p = subprocess.run([sys.executable, "-c", code, name, fd_py], capture_output=True, text=True, timeout=120)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        assert p.returncode == 1 and len(lines) == 1, (name, p.returncode, p.stdout[-300:], p.stderr[-300:])
        res = json.loads(lines[0])
        assert res["ok"] is False and res["error"] == "ModuleNotFoundError" and f"'{name}'" in res["message"], (name, res)
        assert ("TAZUNA_PYTHON" in res["message"]) == third_party and ("incomplete" in res["message"]) != third_party, (name, res)
    with tempfile.TemporaryDirectory() as tmp:          # a package that is installed but does not load
        os.makedirs(os.path.join(tmp, "scservo_sdk"))
        with open(os.path.join(tmp, "scservo_sdk", "__init__.py"), "w") as f:
            f.write("raise ImportError('libfoo.so.1: cannot open shared object file')\n")
        env = dict(os.environ, PYTHONPATH=tmp + os.pathsep + os.environ.get("PYTHONPATH", ""))
        p = subprocess.run([sys.executable, fd_py, "--sim", "state"], capture_output=True, text=True, timeout=120, env=env)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        assert p.returncode == 1 and len(lines) == 1, (p.returncode, p.stdout[-300:], p.stderr[-300:])
        res = json.loads(lines[0])
        assert res["ok"] is False and res["error"] == "ImportError", res
        assert "libfoo.so.1" in res["message"] and "TAZUNA_PYTHON" in res["message"], res


def test_fd_keeps_its_answer_and_its_own_run_folder():
    """Two things about the run record: when it cannot be written (disk full, read-only folder) the finished command
    still prints its JSON, with a `log_error`; and the pointer runs/current is only followed inside this project
    folder (a copied folder used to log into the original's runs/)."""
    import contextlib
    import io
    import time as _time

    import fd
    saved = (fd.log_event, fd.RUNS)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fd.RUNS = os.path.join(tmp, "copy", "runs")

            def no_space(*a, **k):
                raise OSError(28, "No space left on device")

            fd.log_event = no_space
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = fd.main(["--sim", "state"])
            res = json.loads([ln for ln in buf.getvalue().splitlines() if ln.strip()][-1])
            assert rc == 0 and res["ok"] and "raw" in res and "No space left" in res["log_error"], res
            fd.log_event = saved[0]
            # a pointer that names another folder's run directory of today is not followed
            other = os.path.join(tmp, "original", "runs", _time.strftime("%Y%m%d") + "_000001")
            os.makedirs(other)
            os.makedirs(fd.RUNS, exist_ok=True)
            with open(os.path.join(fd.RUNS, "current"), "w") as f:
                f.write(other)
            d = fd.run_dir()
            assert os.path.dirname(d) == fd.RUNS and d != other and os.path.isdir(d), d
            assert fd.run_dir() == d                                   # ... and its own one is kept for the day
            assert os.listdir(other) == []
        finally:
            fd.log_event, fd.RUNS = saved


def test_launcher_chooses_the_interpreter():
    """The launcher `fd`: $TAZUNA_PYTHON, else the first non-blank line of config/python (CR and blanks stripped,
    a leading ~/ expanded), else python3; a path that cannot be run gets one JSON error and exit code 1. Works through a symlink."""
    import shutil
    import stat
    import subprocess
    if shutil.which("bash") is None:
        raise Skip("no bash here")
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "tree with space")
        os.makedirs(os.path.join(root, "tools", "cli"))
        os.makedirs(os.path.join(root, "config"))
        os.makedirs(os.path.join(tmp, "bin"))
        shutil.copy(os.path.join(tazuna_paths.ROOT, "fd"), os.path.join(root, "fd"))
        with open(os.path.join(root, "tools", "cli", "fd.py"), "w") as f:
            f.write("# stand-in\n")

        def interpreter(path, tag):       # a stand-in "Python": says who it is and what it was given
            with open(path, "w") as f:
                f.write(f"#!/bin/sh\necho \"{tag}|$1|$2|$3\"\n")
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
            return path

        py_env = interpreter(os.path.join(tmp, "py env"), "ENV")
        py_file = interpreter(os.path.join(tmp, "py_file"), "FILE")
        interpreter(os.path.join(tmp, "bin", "python3"), "PATH")
        interpreter(os.path.join(tmp, "py_home"), "HOME")
        target = os.path.join(root, "tools", "cli", "fd.py")
        link = os.path.join(tmp, "fd_link")
        os.symlink(os.path.join(root, "fd"), link)

        def run(cfg_text=None, env_py=None, via=None):
            cfg = os.path.join(root, "config", "python")
            if cfg_text is None:
                if os.path.exists(cfg):
                    os.remove(cfg)
            else:
                with open(cfg, "wb") as f:
                    f.write(cfg_text.encode())
            env = {k: v for k, v in os.environ.items() if k != "TAZUNA_PYTHON"}
            env["PATH"] = os.path.join(tmp, "bin") + os.pathsep + env.get("PATH", "")
            env["HOME"] = tmp
            if env_py is not None:
                env["TAZUNA_PYTHON"] = env_py
            p = subprocess.run([via or os.path.join(root, "fd"), "do", "a b"], capture_output=True, text=True, env=env, cwd=tmp, timeout=60)
            return p.returncode, p.stdout.strip()

        want = lambda tag: (0, f"{tag}|{target}|do|a b")
        assert run() == want("PATH")                                        # nothing set: python3 of the PATH
        assert run(cfg_text=py_file + "\n") == want("FILE")
        assert run(cfg_text="\r\n  " + py_file + "  \r\nsecond line\n") == want("FILE")     # CRLF, blank first line, blanks
        assert run(cfg_text=py_file + "\n", env_py=py_env) == want("ENV")   # the variable wins
        assert run(cfg_text=py_file + "\n", env_py="") == want("FILE")      # an empty variable counts as unset
        assert run(cfg_text=py_file + "\n", via=link) == want("FILE")       # called through a symlink, from elsewhere
        assert run(cfg_text="~/py_home\n") == want("HOME")                  # a path that starts at the home folder
        for bad in (dict(cfg_text=os.path.join(tmp, "no such python") + "\n"), dict(env_py=os.path.join(tmp, 'no "such" python'))):
            rc, out = run(**bad)
            res = json.loads(out)
            assert rc == 1 and res["ok"] is False and res["error"] == "PythonNotFound" and "config/python" in res["message"], (rc, out)


if __name__ == "__main__":
    import sys
    import time
    import traceback
    only = sys.argv[1:]
    fails = skipped = 0
    n = 0
    t_all = time.time()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and (not only or any(o in name for o in only)):
            n += 1
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f} s)")
            except Skip as e:
                skipped += 1
                print(f"SKIP {name}: {e}")
            except Exception:
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    print(f"tests: {n}  passed: {n - fails - skipped}  skipped: {skipped}  failures: {fails}  ({time.time() - t_all:.0f} s)")
    sys.exit(1 if fails else 0)
