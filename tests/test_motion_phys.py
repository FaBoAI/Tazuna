"""Motion tests on PhysBus, the torque-level arm model (no hardware). Run: python tests/test_motion_phys.py

    python tests/test_motion_phys.py            all tests + the speed-tier study + the bench table
    python tests/test_motion_phys.py study      only the speed-tier study
    python tests/test_motion_phys.py <substr>   tests whose name contains <substr>

PhysBus was calibrated by the review verifier against the real run log, i.e. against static
holds and motions of 25 deg/s OR LESS. Everything printed here for the faster tiers ("normal",
"fast") is a PREDICTION of that model, not a measurement; bench_motion on the real arm decides.
Two model variants are reported side by side:
    quasi   the calibrated quasi-static model (cannot overshoot by construction)
    dyn     the same with link / rotor inertia and the in-servo trapezoid profile (Acceleration /
            Goal_Velocity registers); these two extensions are physically reasoned, NOT calibrated
"""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import json
import math
import os
import sys
import time

import numpy as np

import kinematics as K
import phys_bus as PB
import robot as R
import servo_bus
from phys_bus import PhysBus
from robot import ARM, DEFAULT_CONFIG, Arm, MotionAborted, SafetyError
from servo_bus import JOINTS
from sim_bus import SimBus


def _forbidden(*a, **k):
    raise RuntimeError("HARDWARE ACCESS FORBIDDEN IN TESTS")


servo_bus.Bus.open = _forbidden  # these tests can never open the serial port
# The gripper in these tests is bare: they never read the tool file of the tree (tooling.py), so they pass
# whether or not a tool is attached to the real arm, and nobody has to `fd tool detach` to run them.
os.environ["FD_TOOL"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "no_tool_attached.json")

np.set_printoptions(precision=3, suppress=True, linewidth=170)
TPD = 4096.0 / 360.0
TABLE = -0.0235


# The rig as it was calibrated on 2026-11-10 (the values of config.json at the time the real
# picks ran). Frozen here so that these tests do not change when the live file is edited.
RIG = {
    "zero_ticks": {"shoulder_pan": 2047.0, "shoulder_lift": 2047.0, "elbow_flex": 2047.0, "wrist_flex": 2047.0,
                   "wrist_roll": 1944.6},
    "tip_offset": [-0.001, 0.0, 0.0068],
    "floor_z": -0.027501036388128973,
    "table_z": -0.023501036388128973,
    "keepouts": {"base_column": [-0.12, 0.085, -0.065, 0.065, -0.05, 0.13],
                 "box": [0.227, 0.346, 0.026, 0.137, -0.05, 0.036],
                 "cables": [0.19, 0.45, -0.4, -0.135, -0.05, 0.03]},
}


def real_cfg(**over):
    """DEFAULT_CONFIG with the rig calibration and every speed tier unlocked (in memory only)."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    R._merge(cfg, json.loads(json.dumps(RIG)))
    cfg["speed_tiers_validated"] = sorted(cfg["speed_tiers"])
    cfg.update(over)
    return cfg


def make(cfg=None, at_ready=True, **bus_kw):
    """Arm on a PhysBus, torqued; at_ready teleports it to the ready pose and lets it settle."""
    cfg = real_cfg() if cfg is None else cfg
    bus_kw.setdefault("h", 0.002)
    bus = PhysBus(cfg, **bus_kw)
    arm = Arm(bus=bus, cfg=cfg).open()
    if at_ready:
        bus.set_q(arm.ready_q())
    arm.torque_on()
    if at_ready:
        arm.move_joints(arm.ready_q(), 20.0)
    return arm, bus


def gmp(bus):
    return {j: int(round(bus.goal[j] - bus.pos[j])) for j in ARM}


def tip(arm, q=None):
    return K.fk(arm.read_q() if q is None else q, arm.tip_tool())[:3, 3]


def raises(exc, fn, match=None):
    try:
        fn()
    except exc as e:
        if match is not None:
            assert match in str(e), f"{match!r} not in {str(e)!r}"
        return e
    raise AssertionError(f"{exc.__name__} not raised")


# =============================================================== model self-checks
def test_gravity_and_contact_statics():
    rng = np.random.default_rng(0)
    tool = K.tool_offset(-0.001, 0.0, 0.0068)
    worst = 0.0
    for _ in range(200):
        q = rng.uniform(-1.2, 1.2, 5)
        tg = PB.torques(q, tool)[0]
        num = np.array([-(PB.potential(q + d) - PB.potential(q - d)) / 2e-4 for d in np.eye(5) * 1e-4])
        worst = max(worst, float(np.max(np.abs(tg - num))))
    assert worst < 1e-6, worst  # analytic gravity torque = -dU/dq
    q = np.radians([10.0, 5.0, 20.0, 65.0, 21.0])
    z = K.fk(q, tool)[2, 3] + 0.01
    tg, tc, _, fz = PB.torques(q, tool, z_surface=z, k_contact=1000.0)
    jz = np.array([(K.fk(q + d, tool)[2, 3] - K.fk(q - d, tool)[2, 3]) / 2e-5 for d in np.eye(5) * 1e-5])
    assert abs(fz - 10.0) < 1e-6 and np.max(np.abs(tc - jz * fz)) < 1e-6  # contact torque = Jz^T F
    # total moving mass and the self-weight torque at full reach (research: 0.864 N m at shoulder_lift)
    assert abs(sum(m for _, _, m, _ in PB.INERTIALS) + PB.MOVING_JAW[0] - 0.485) < 0.001


def test_calibration_matches_real_log():
    """The numbers the verifier calibrated on: Present_Load per tick, free-air holding offsets."""
    arm, bus = make()
    r = arm.move_tip(np.array([0.2318, -0.0401, 0.06]), 90.0, 21.0, 20.0)
    # real event #90 (same move, P = 32): corr_deg [-0.19, -1.39, -1.92, -0.4, 0.32]
    assert -3.5 < r["corr_deg"][1] < -0.5 and -3.0 < r["corr_deg"][2] < -0.3, r["corr_deg"]
    snap = bus.snapshot()
    for j in ("shoulder_lift", "elbow_flex"):  # Present_Load = 8.2 permille per tick of (goal - pos), sign as logged
        e = snap[j]["goal"] - snap[j]["pos"]
        assert abs(snap[j]["load"] + 8.2 * e) <= 8.2, (j, e, snap[j]["load"])
        assert e < 0 and snap[j]["load"] > 0  # holding against gravity: goal below present, load positive
    # at P = 16 the same pose needs about twice the standing error (log: -54 / -35 ticks at P=16 vs -27 / -18)
    arm.cfg["p_gain"] = 16
    e32 = gmp(bus)
    arm.restore_gains(force=True)
    bus.sleep(2.0)
    e16 = gmp(bus)
    assert bus.p_gain["shoulder_lift"] == 16
    assert tip(arm)[2] < 0.06 + 0.002  # lowering the gain lets the arm sag (never rise)
    assert abs(e16["shoulder_lift"]) >= abs(e32["shoulder_lift"]), (e32, e16)


def test_probe_sink_matches_real_probes():
    """Real table probes (23:20:08, 23:21:14): the encoder height sank about 5 mm between contact
    onset and detection. k_contact = 1 N/mm reproduces that; the detected height stays within 3 mm."""
    arm, bus = make(z_surface=TABLE, k_contact=1000.0, h=0.001)
    arm.move_tip(np.array([0.22, 0.0, 0.03]), 90.0, 15.0, 20.0)
    arm.move_tip(np.array([0.22, 0.0, 0.0]), 90.0, 15.0, 10.0, straight=True)
    bus.peak_fz = 0.0
    pr = arm.probe_down(max_drop_m=0.06)
    assert pr["contact"] is True, pr
    sink = (pr["contact_tip_z"] - pr["stall_tip_z"]) * 1000
    assert 2.0 < sink < 9.0, sink
    assert abs(pr["contact_tip_z"] - TABLE) < 0.003, pr["contact_tip_z"]
    assert bus.peak_fz < 9.0, bus.peak_fz  # bounded push
    assert tip(arm)[2] > TABLE + 0.005  # released and backed off
    print(f"      probe on a 1 N/mm table: contact z {pr['contact_tip_z']:+.4f} (table {TABLE:+.4f}), sink {sink:.1f} mm, "
          f"peak force {bus.peak_fz:.1f} N")


def test_bounds_never_starve_a_free_air_hold():
    """Pure statics: the offset a joint needs to hold AND to break friction either way must lie
    inside the gravity-aware correction bounds (1.5 deg allowance) at every task pose."""
    cfg = real_cfg()
    arm = Arm(bus=SimBus(), cfg=cfg).open()
    lim, tool = arm.q_limits(), arm.tip_tool()
    rng = np.random.default_rng(7)
    poses = []
    for _ in range(1500):
        xyz = (rng.uniform(0.12, 0.33), rng.uniform(-0.18, 0.18), rng.uniform(-0.02, 0.16))
        q, info = K.ik(xyz, rng.uniform(55, 100), math.radians(rng.uniform(-60, 60)), tool=tool, limits=lim)
        if info["ok"]:
            poses.append(("ik", q))
    n_ik = len(poses)
    for _ in range(1500):
        poses.append(("rand", np.array([rng.uniform(*lim[j]) for j in ARM])))
    poses += [("ik", arm.rest_q()), ("ik", arm.ready_q())]
    kp_deg = PB.KP_PER_TICK_P32 * TPD
    allow = cfg["friction_corr_max_deg"]
    rows = []
    for ts, tf in ((3.0, 0.25), (3.0, 0.10), (1.3, 0.10), (3.0, 0.35), (2.0, 0.25)):
        bad = {"ik": 0, "rand": 0}
        worst = {"ik": 0.0, "rand": 0.0}
        for kind, q in poses:
            tg = PB.torques(q, tool)[0]
            lo, hi = (np.degrees(b) for b in arm._offset_bounds(q, allow))
            need_lo, need_hi = (-tg - tf) / (ts * kp_deg), (-tg + tf) / (ts * kp_deg)
            v = float(np.max(np.maximum(lo - need_lo, need_hi - hi)[1:4]))
            if v > 0:
                bad[kind] += 1
                worst[kind] = max(worst[kind], v)
        rows.append((ts, tf, tf / (ts * kp_deg), bad["ik"], bad["rand"], worst["ik"]))
        print(f"      tau_stall {ts} tau_fric {tf} (friction band {tf / (ts * kp_deg):.2f} deg): task poses short of "
              f"authority {bad['ik']}/{n_ik + 2} (worst {worst['ik']:.2f} deg), random joint poses {bad['rand']}/1500 "
              f"(worst {worst['rand']:.2f} deg)")
    for ts, tf, band, n_ik_bad, n_rand_bad, w in rows[:2]:  # the calibrated set and its low-friction variant
        assert n_ik_bad == 0 and n_rand_bad == 0, (ts, tf, n_ik_bad, n_rand_bad)
    # weaker motors / more friction eat into the allowance: a few near-balance task poses come up short
    assert all(r[3] <= 15 and r[5] < 0.5 for r in rows), rows


def test_live_config_smoke():
    """The live config.json (read only) drives the same motion core: ready pose, one transit, one
    straight descent above the table, bench pre-checks."""
    cfg = R.load_config()
    bus = PhysBus(cfg, h=0.002)
    arm = Arm(bus=bus, cfg=cfg).open()
    bus.set_q(arm.ready_q())
    arm.torque_on()
    sp = arm.speeds()
    assert sp["tier"] in cfg["speed_tiers_validated"]
    table = float(cfg.get("table_z", TABLE))
    r1 = arm.move_joints(arm.solve(np.array([0.24, -0.04, table + 0.06]), 90.0, 10.0), sp["transit"])
    r2 = arm.move_tip_path([((0.24, -0.04, table + 0.03), 90.0, 10.0)], sp["approach"])
    assert max(abs(e) for e in r1["joint_err_deg"] + r2["joint_err_deg"]) < 0.6
    arm.move_joints(arm.ready_q(), sp["transit"])
    res = arm.bench_motion(sp["tier"])
    assert res["pass"] is True, res


# =============================================================== free-air regression
def _random_moves(arm, n=26, seed=11, speed=(8, 30)):
    rng = np.random.default_rng(seed)
    done = aborted = 0
    worst = worst_corr = 0.0
    msgs = []
    for _ in range(n):
        xyz = np.array([rng.uniform(0.15, 0.29), rng.uniform(-0.14, 0.14), rng.uniform(-0.015, 0.14)])
        pitch = rng.uniform(65, 92)
        try:
            res = arm.move_tip(xyz, pitch, rng.uniform(-30, 30), rng.uniform(*speed), straight=bool(rng.integers(0, 2)),
                               ignore_keepouts=("box", "cables"))
        except SafetyError:
            continue  # unreachable random target
        except MotionAborted as e:
            aborted += 1
            done += 1
            msgs.append(str(e)[:100])
            continue
        done += 1
        worst = max(worst, max(abs(e) for e in res["joint_err_deg"]))
        worst_corr = max(worst_corr, max(abs(c) for c in res["corr_deg"]))
    return done, aborted, worst, worst_corr, msgs


def test_free_air_accuracy_and_no_false_aborts():
    """Random free-air moves over the workspace on four parameter sets: no MotionAborted, final
    joint error as in the real log (<= 0.56 deg there)."""
    for ts, tf, payload in ((3.0, 0.25, 0.0), (3.0, 0.10, 0.03), (1.3, 0.10, 0.0), (3.0, 0.35, 0.03)):
        arm, bus = make(tau_stall=ts, tau_fric=tf, payload_kg=payload)
        done, aborted, worst, worst_corr, msgs = _random_moves(arm)
        print(f"      tau_stall {ts} tau_fric {tf} payload {payload * 1000:.0f} g: {done} moves, {aborted} aborted, "
              f"worst final joint error {worst:.2f} deg, largest correction {worst_corr:.2f} deg")
        assert done >= 18 and aborted == 0, msgs
        assert worst < 0.6, worst


def test_friction_beyond_allowance_is_reported_not_hidden():
    """If the real friction band were wider than friction_corr_max_deg (here 1.9 deg vs 1.5), free-air
    moves would end in 'target not reached' aborts: loud, with the push released, never silent. The
    remedy on the arm is a larger friction_corr_max_deg, to be decided with the standing offsets seen."""
    arm, bus = make(tau_stall=2.0, tau_fric=0.35, at_ready=False)
    aborted = 0
    try:
        arm.move_joints(arm.ready_q(), 25.0)
    except MotionAborted as e:
        aborted += 1
        assert e.info["reason"] in ("target_not_reached", "tracking_error")
    done, ab2, worst, worst_corr, msgs = _random_moves(arm, n=10)
    print(f"      friction band 1.9 deg (allowance 1.5): {aborted + ab2} of {done + 1} moves aborted loudly"
          + (f", e.g. '{msgs[0][:70]}...'" if msgs else ""))
    arm.cfg["friction_corr_max_deg"] = 2.5
    done, ab3, worst, _, _ = _random_moves(arm, n=10)
    assert ab3 == 0 and worst < 0.6, (ab3, worst)  # with the allowance raised the same arm runs clean


SEQ = [  # the tip moves of the first real pick-and-place (events #90 .. #108): xyz, pitch, roll, speed, straight
    ((0.2318, -0.0401, 0.06), 90.0, 21.0, 20.0, False), ((0.2318, -0.0401, 0.0045), 90.0, 21.0, 12.0, True),
    ((0.2318, -0.0401, -0.0195), 90.0, 21.0, 8.0, True), ((0.2318, -0.0401, 0.07), 90.0, 21.0, 10.0, True),
    ((0.278, 0.0754, 0.07), 82.0, -15.0, 12.0, True), ((0.278, 0.0754, 0.02), 82.0, -15.0, 8.0, True),
    ((0.278, 0.0754, 0.085), 80.0, -15.0, 12.0, True)]


def test_real_pick_sequence_replay():
    """The exact command sequence of the logged pick (gripper included, 20 g in the jaws after the
    grasp) runs without an abort on the calibrated model and on a weak-motor variant."""
    for ts, tf in ((3.0, 0.25), (1.3, 0.10)):
        arm, bus = make(tau_stall=ts, tau_fric=tf, blocker=lambda pos: pos["gripper"] < 2194, servo_profile=True)
        t0 = bus.now
        errs = []
        for i, (xyz, pitch, roll, spd, straight) in enumerate(SEQ):
            if i == 2:
                arm.set_gripper(0.315)
            if i == 3:
                g = arm.grasp()
                assert g["contact"] and abs(g["pos_raw"] - 2194) <= 3, g
                bus.payload = 0.02
            if i == 6:
                arm.set_gripper(0.22)
                bus.payload = 0.0
            res = arm.move_tip(np.array(xyz), pitch, roll, spd, straight=straight, ignore_keepouts=("box",))
            errs.append(max(abs(e) for e in res["joint_err_deg"]))
            assert res["corr_deg"][1] < 0.6 and res["corr_deg"][2] < 0.6, res  # lift / elbow never push down
        arm.set_gripper(0.06)
        arm.move_joints(arm.ready_q(), 25.0)
        assert max(errs) < 0.6, errs
        print(f"      tau_stall {ts} tau_fric {tf}: 7 tip moves + grasp + home, worst final error {max(errs):.2f} deg, "
              f"{bus.now - t0:.0f} s of robot time (v1 command set, slow speeds)")


# =============================================================== MS-0 / MS-1 on a compliant contact
XY = (0.2318, -0.0401)


def _blocked(z_obstacle, z_target, k, ts=3.0, tf=0.25, arm_cfg=None):
    arm, bus = make(cfg=arm_cfg, tau_stall=ts, tau_fric=tf, z_surface=z_obstacle, k_contact=k,
                    h=0.00025 if k > 5000 else 0.001)
    arm.move_tip(np.array([XY[0], XY[1], 0.06]), 90.0, 21.0, 20.0)
    z_pre = 0.0045 if z_obstacle is None else max(0.0045, z_obstacle + 0.01)
    arm.move_tip(np.array([XY[0], XY[1], z_pre]), 90.0, 21.0, 12.0, straight=True)
    free = gmp(bus)
    bus.peak_fz = 0.0
    t0 = bus.now
    try:
        res = arm.move_tip(np.array([XY[0], XY[1], z_target]), 90.0, 21.0, 12.0, straight=True)
        err = None
    except MotionAborted as e:
        res, err = None, e
    took = bus.now - t0
    bus.sleep(5.0)  # nothing relieves the goal while the operator thinks
    return arm, bus, res, err, free, took


def test_blocked_descent_on_compliant_contact():
    """Fixed jaw lands on the block top, 14 mm above the commanded height (the verifier's case A;
    the unpatched code returned ok with 56-98 % PWM and 8.5-19.5 N standing)."""
    for ts, tf, k in ((3.0, 0.25, 1000.0), (3.0, 0.25, 5000.0), (3.0, 0.25, 20000.0), (1.3, 0.10, 1000.0),
                      (1.3, 0.10, 5000.0)):
        arm, bus, res, err, free, took = _blocked(-0.0055, -0.0195, k, ts, tf)
        assert err is not None and err.info["reason"] == "target_not_reached", (ts, tf, k, res)
        pwm = bus.pwm()
        after = gmp(bus)
        print(f"      block top, k {k / 1000:4.0f} N/mm, tau_stall {ts}: aborted after {took:.1f} s; standing PWM lift/elbow "
              f"{pwm['shoulder_lift']:+d}/{pwm['elbow_flex']:+d} %, force {bus.fz:.1f} N (peak {bus.peak_fz:.1f} N)")
        assert bus.fz < 6.0 and bus.peak_fz < 12.0, (bus.fz, bus.peak_fz)
        assert abs(pwm["shoulder_lift"]) <= 35 and abs(pwm["elbow_flex"]) <= 25, pwm
        for j in ("shoulder_lift", "elbow_flex"):
            assert after[j] <= free[j] + 12, (j, after, free)  # no downward push left standing
    # a table 8 mm higher than believed, stiff: abort; 1.5 mm, soft: reaches the target with a light touch
    arm, bus, res, err, free, took = _blocked(TABLE + 0.008, TABLE, 5000.0)
    assert err is not None and bus.fz < 6.0, (res, bus.fz)
    arm, bus, res, err, free, took = _blocked(TABLE + 0.0015, TABLE, 1000.0)
    assert bus.fz < 3.0, bus.fz
    print(f"      table +1.5 mm, 1 N/mm (shallow, soft): {'aborted' if err else 'target reached'}, standing force "
          f"{bus.fz:.1f} N: such a light contact can pass as ok")


def _transfer_deficit(arm, bus):
    tool = arm.tip_tool()
    rows = []
    arm.move_tip(np.array([0.278, 0.0754, 0.085]), 80.0, -15.0, 25.0,
                 monitor=lambda t, qp, qm: rows.append((K.fk(qp, tool)[2, 3], K.fk(qm, tool)[2, 3])) and False)
    return max(a - b for a, b in rows) * 1000


def test_contact_offset_is_not_carried_over():
    """After a descent that ended in contact, the next transfer must not run centimetres below its
    plan (verifier: 18-33 mm low with the unpatched code, baseline 9 mm on a compliant model)."""
    arm, bus, res, err, free, took = _blocked(None, -0.0195, 0.0)
    assert err is None
    base = _transfer_deficit(arm, bus)
    out = []
    for depth_mm, k in ((6.0, 1000.0), (3.5, 5000.0), (14.0, 1000.0), (3.5, 1000.0), (14.0, 300.0)):
        arm, bus, res, err, free, took = _blocked(-0.0195 + depth_mm / 1000.0, -0.0195, k)
        d = _transfer_deficit(arm, bus)
        out.append((depth_mm, k, err is not None, d))
        assert d < base + 3.0, (depth_mm, k, d, base)
    print(f"      height deficit of the next transfer: free-air baseline {base:.1f} mm; after contact "
          + ", ".join(f"{d:.0f} mm/{k / 1000:.1f} N/mm {'(aborted)' if ab else '(passed as ok)'}: {v:.1f} mm"
                      for d, k, ab, v in out))


def test_overload_protection_bit_aborts_motion():
    """SP-3: the servo's own overload protection (status bit 32) stops the stream."""
    arm, bus = make(overload=(0.25, 0.3, 0.2))
    q = arm.read_q() + np.array([0.9, 0.3, 0.0, 0.0, 0.0])
    e = raises(MotionAborted, lambda: arm.move_joints(q, 60.0), "overload")
    assert e.info["reason"] == "servo_fault" and any(bus.protect.values())
    raises(SafetyError, lambda: arm.move_joints(arm.read_q(), 10.0), "overload")  # and nothing moves until it is cleared


# =============================================================== new motion API on the physical model
def test_continuous_path_and_gripper_on_phys():
    arm, bus = make(blocker=lambda pos: pos["gripper"] < 2194, servo_profile=True, io_s=0.003)
    sp = arm.speeds()
    hover = ((0.25, -0.05, 0.02), 90.0, 0.0)
    grasp = ((0.25, -0.05, -0.01), 90.0, 0.0)
    t0 = bus.now
    r1 = arm.move_joints(arm.solve(np.array(hover[0]), hover[1], hover[2]), sp["transit"], settle=False,
                         gripper_goal_raw=2516)
    assert abs(r1["gripper_pos_raw"] - 2516) <= 3  # the gripper opened on the way
    r2 = arm.move_tip_path([grasp], sp["approach"])
    g = arm.grasp_fast(expected_raw=2194)
    assert g["contact"] and abs(g["delta_raw"]) <= 3 and g["duration_s"] < 0.8, g
    bus.payload = 0.02
    r3 = arm.move_tip_path([((0.25, -0.05, 0.05), 90.0, 0.0), ((0.26, 0.085, 0.07), 85.0, 0.0),
                            ((0.26, 0.085, 0.03), 85.0, 0.0)], sp["transit"], ignore_keepouts=("box",))
    assert np.linalg.norm(tip(arm) - np.array([0.26, 0.085, 0.03])) < 0.004
    rel = arm.set_gripper(raw=2400)
    bus.payload = 0.0
    r4 = arm.move_tip_path([((0.26, 0.085, 0.085), 85.0, 0.0)], sp["transit"], settle=False, ignore_keepouts=("box",))
    r5 = arm.move_joints(arm.ready_q(), sp["transit"], settle=False, gripper_goal_raw=2100)
    total = bus.now - t0
    parts = [r1["duration_s"], r2["duration_s"], g["duration_s"], r3["duration_s"], rel["duration_s"], r4["duration_s"],
             r5["duration_s"]]
    print(f"      one pick with the v2 calls at the slow tier: {total:.1f} s of robot time (hover {parts[0]:.1f}, descend "
          f"{parts[1]:.1f}, grasp {parts[2]:.2f}, lift+transfer+lower {parts[3]:.1f}, release {parts[4]:.2f}, retreat "
          f"{parts[5]:.1f}, ready {parts[6]:.1f})")
    assert total < 30.0
    assert all(r["aborted"] is False for r in (r1, r2, r3, r4, r5))


def _pick_like_sequence(arm, bus, sp):
    """The motion pattern skills.pick_place uses (short settles, gripper on the way, continuous
    carry) in free air with 20 g in the jaws; returns the per-move results."""
    out = []
    hover = arm.solve(np.array([0.25, -0.05, 0.02]), 90.0, 30.0)
    out.append(arm.run_path([arm.read_q(), hover], sp["transit"], settle=0.8, gripper_goal_raw=2516))
    out.append(arm.run_path(arm.plan_tip_path([((0.25, -0.05, -0.01), 90.0, 30.0)]), sp["approach"], settle=1.0))
    bus.payload = 0.02
    out.append(arm.run_path(arm.plan_tip_path([((0.25, -0.05, 0.05), 90.0, 30.0), ((0.26, 0.085, 0.07), 85.0, -40.0),
                                               ((0.26, 0.085, 0.03), 85.0, -40.0)]), sp["transit"], settle=0.6,
                            ignore_keepouts=("box",)))
    bus.payload = 0.0
    out.append(arm.run_path(arm.plan_tip_path([((0.26, 0.085, 0.085), 85.0, -40.0)]), sp["transit"], settle=0.3,
                            ignore_keepouts=("box",)))
    out.append(arm.run_path([arm.read_q(), arm.ready_q()], sp["transit"], settle=0.5, gripper_goal_raw=2100))
    return out


def test_no_false_aborts_with_short_settles():
    """Free air, the settle times the pick skill uses (0.3 .. 1.0 s): the blocked-target test must
    stay quiet on the calibrated model, with heavier friction and on the dynamic variant."""
    variants = {"quasi": dict(), "quasi, friction 0.35": dict(tau_fric=0.35), "dyn": dict(inertia=True, servo_profile=True)}
    print("      robot time of the pick-like sequence (s) / worst end error (deg), free air:")
    for name, kw in variants.items():
        row = []
        for tier in STUDY_TIERS:
            arm, bus = make(cfg=real_cfg(speed_tier=tier), io_s=0.003, **kw)
            t0 = bus.now
            try:
                rs = _pick_like_sequence(arm, bus, arm.speeds())
                worst = max(max(abs(e) for e in r["joint_err_deg"]) for r in rs)
                row.append(f"{tier} {bus.now - t0:.1f}/{worst:.2f}")
                ok = True
            except MotionAborted as e:
                row.append(f"{tier} ABORT ({str(e)[:50]})")
                ok = False
            if tier in ("slow", "slow_cruise") or (tier == "normal" and name != "dyn"):
                assert ok, (name, tier, row[-1])
        print(f"        {name:21s}: " + "; ".join(row))


# =============================================================== speed tiers
STUDY_TIERS = ("slow", "normal", "fast", "slow_cruise")
MODELS = {"quasi": dict(), "dyn": dict(inertia=True, servo_profile=True)}


def _cross_track(plan_tips, meas_tips) -> float:
    """Largest distance (m) of a measured fingertip position from the planned fingertip PATH (the
    polyline through the planned positions), i.e. the geometric error without the time lag."""
    a, b = plan_tips[:-1], plan_tips[1:]
    ab = b - a
    den = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    worst = 0.0
    for m in meas_tips:
        s = np.clip(np.einsum("ij,ij->i", m - a, ab) / den, 0.0, 1.0)
        worst = max(worst, float(np.min(np.linalg.norm(a + ab * s[:, None] - m, axis=1))))
    return worst


def _study_motion(arm, bus, name, tier_speeds):
    """Run one of the three representative motions; returns the run_path result plus the peak
    fingertip deviation from the plan at the same instant ("tip_dev_mm", mostly time lag along the
    path) and from the planned path ("xtrack_mm", what matters for clearances)."""
    tool = arm.tip_tool()
    dev = [0.0]
    plan_tips, meas_tips = [], []

    def mon(t, qp, qm):
        a, b = K.fk(qp, tool)[:3, 3], K.fk(qm, tool)[:3, 3]
        plan_tips.append(a)
        meas_tips.append(b)
        dev[0] = max(dev[0], float(np.linalg.norm(a - b)))
        return False

    if name == "hover":  # ready pose -> hover above (0.25, -0.05) at z = 0.02, pitch 90
        r = arm.move_joints(arm.solve(np.array([0.25, -0.05, 0.02]), 90.0, 0.0), tier_speeds["transit"], monitor=mon)
    elif name == "descent":  # 30 mm straight down to the grasp height
        r = arm.move_tip_path([((0.25, -0.05, -0.01), 90.0, 0.0)], tier_speeds["approach"], monitor=mon)
    else:  # lift 60 mm + transfer to (0.26, 0.085, 0.07) + lower 40 mm as ONE path, 20 g in the jaws
        bus.payload = 0.02
        r = arm.move_tip_path([((0.25, -0.05, 0.05), 90.0, 0.0), ((0.26, 0.085, 0.07), 85.0, 0.0),
                               ((0.26, 0.085, 0.03), 85.0, 0.0)], tier_speeds["transit"], monitor=mon,
                              ignore_keepouts=("box",))
        bus.payload = 0.0
    r["tip_dev_mm"] = round(dev[0] * 1000, 1)
    r["xtrack_mm"] = round(_cross_track(np.array(plan_tips), np.array(meas_tips)) * 1000, 1)
    return r


def speed_tier_study(verbose=True):
    """duration / peak tracking error / overshoot / settle time of three representative motions for
    every tier on both model variants. Returns {(model, tier, motion): result or exception}."""
    out = {}
    for model, kw in MODELS.items():
        for tier in STUDY_TIERS:
            cfg = real_cfg(speed_tier=tier)
            arm, bus = make(cfg=cfg, io_s=0.003, **kw)
            sp = arm.speeds()
            for motion in ("hover", "descent", "lift_transfer_lower"):
                try:
                    out[(model, tier, motion)] = _study_motion(arm, bus, motion, sp)
                except (MotionAborted, SafetyError) as e:
                    out[(model, tier, motion)] = e
                    break
    if verbose:
        print("\n  SPEED-TIER STUDY on PhysBus (calibrated on motions <= 25 deg/s: 'normal' and 'fast' are PREDICTIONS)")
        print("  model: quasi = calibrated quasi-static model; dyn = + inertia and in-servo profile (not calibrated)")
        print("  speed = peak speed of the fastest joint; duration includes the settle; settle = time to 0.25 deg;")
        print("  <1deg = time after the plan until every joint stays within 1 deg; lag mm = peak fingertip distance")
        print("  from the planned position at the same instant; xtrack mm = peak distance from the planned PATH")
        print(f"  {'model':5s} {'tier':11s} {'motion':20s} {'speed':>5s} {'dur s':>6s} {'plan s':>6s} {'peak err':>8s} "
              f"{'overshoot':>9s} {'settle s':>8s} {'<1deg s':>7s} {'lag mm':>6s} {'xtrack':>6s} {'load':>5s} {'volt':>5s}")
        for (model, tier, motion), r in out.items():
            if isinstance(r, Exception):
                print(f"  {model:5s} {tier:11s} {motion:20s} {type(r).__name__}: {str(r)[:90]}")
                continue
            print(f"  {model:5s} {tier:11s} {motion:20s} {r['speed_deg_s']:5.0f} {r['duration_s']:6.2f} {r['plan_s']:6.2f} "
                  f"{r['peak_err_deg']:7.2f}d {r['overshoot_deg']:8.2f}d {r['settle_s']:8.2f} {r['settle_1deg_s']:7.2f} "
                  f"{r['tip_dev_mm']:6.1f} {r['xtrack_mm']:6.1f} {r['peak_load']:5d} {r['min_volt']:5.2f}")
        print("  totals for the three motions (s): " + "; ".join(
            f"{model}/{tier} " + (f"{sum(out[(model, tier, m)]['duration_s'] for m in ('hover', 'descent', 'lift_transfer_lower')):.1f}"
                                  if all(not isinstance(out.get((model, tier, m)), (Exception, type(None)))
                                         for m in ("hover", "descent", "lift_transfer_lower")) else "aborted")
            for model in MODELS for tier in STUDY_TIERS))
    return out


def test_speed_tier_study():
    out = speed_tier_study()
    th = DEFAULT_CONFIG["bench"]
    for model in MODELS:
        tot = {}
        for tier in STUDY_TIERS:
            rs = [out.get((model, tier, m)) for m in ("hover", "descent", "lift_transfer_lower")]
            if tier in ("slow", "slow_cruise"):  # the validated speeds must run clean on both variants
                assert all(isinstance(r, dict) for r in rs), (model, tier, rs)
                for r in rs:
                    assert r["peak_err_deg"] < th["peak_err_deg"] and r["overshoot_deg"] < th["overshoot_deg"], (model, tier, r)
                    assert max(abs(e) for e in r["joint_err_deg"]) < 0.6
            if all(isinstance(r, dict) for r in rs):
                tot[tier] = sum(r["duration_s"] for r in rs)
        assert tot["slow_cruise"] < tot["slow"]
        if "normal" in tot:
            assert tot["normal"] < tot["slow"]
        if "fast" in tot and "normal" in tot:
            assert tot["fast"] < tot["normal"] + 0.5


def test_bench_motion_on_phys():
    """bench_motion runs on the physical model: the validated tier passes on the calibrated variant;
    the others are printed as predictions (not asserted)."""
    print("\n  BENCH_MOTION on PhysBus (thresholds: " + ", ".join(
        f"{k} {v}" for k, v in DEFAULT_CONFIG["bench"].items() if k not in ("start_tol_deg",)) + ")")
    print(f"  {'model':5s} {'tier':11s} {'pass':5s} {'dur s':>6s} {'peak err':>8s} {'overshoot':>9s} {'settle s':>8s} "
          f"{'<1deg s':>7s} {'load':>5s} {'volt':>5s}  fails / abort")
    for model, kw in MODELS.items():
        for tier in STUDY_TIERS:
            arm, bus = make(cfg=real_cfg(speed_tiers_validated=["slow"]), io_s=0.003, **kw)
            res = arm.bench_motion(tier)
            why = res["reason"] or "; ".join(res["fails"]) or ""
            print(f"  {model:5s} {tier:11s} {str(res['pass']):5s} {res['duration_s']:6.1f} "
                  + (f"{res['peak_err_deg']:7.2f}d {res['overshoot_deg']:8.2f}d {res['settle_s']:8.2f} "
                     f"{res['settle_1deg_s']:7.2f} {res['peak_load']:5d} {res['min_volt']:5.2f}" if res["peak_err_deg"] is not None
                     else " " * 47) + f"  {why[:80]}")
            assert all(bus.vel_cap[j] == 1100 and bus.acc[j] == 40 for j in JOINTS)  # the validated profile is back
            assert arm.speeds()["tier"] == "slow"
            if tier == "slow" and model == "quasi":
                assert res["pass"] is True, res
            if res["pass"] or res.get("returned_to_ready"):
                assert np.max(np.abs(arm.read_q() - arm.ready_q())) < math.radians(1.0)


def test_grasp_fast_timing():
    """One closing command with the in-servo profile (Goal_Velocity 1100, Acceleration 40): about 0.5 s
    for the stroke used in the real picks (2516 -> 2194), against about 1.8 s for the stepping grasp()."""
    out = {}
    for name in ("grasp_fast", "grasp"):
        cfg = real_cfg()
        bus = SimBus(sag_ticks=0.0, blocker=lambda pos: pos["gripper"] < 2194, servo_profile=True, io_s=0.003)
        arm = Arm(bus=bus, cfg=cfg).open()
        arm.torque_on()
        arm.set_gripper(raw=2516)
        t0 = bus.now
        res = getattr(arm, name)()
        out[name] = bus.now - t0
        assert res["contact"] and abs(res["pos_raw"] - 2194) <= 3, res
    print(f"      closing 2516 -> stall at 2194: grasp_fast {out['grasp_fast']:.2f} s, stepping grasp {out['grasp']:.2f} s")
    assert out["grasp_fast"] < 0.8 and out["grasp_fast"] < 0.5 * out["grasp"]


if __name__ == "__main__":
    import traceback
    only = sys.argv[1:]
    if only == ["study"]:
        speed_tier_study()
        sys.exit(0)
    fails = n = 0
    t_all = time.time()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and (not only or any(o in name for o in only)):
            n += 1
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.0f} s)")
            except Exception:
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
            sys.stdout.flush()
    print(f"tests: {n}  passed: {n - fails}  failures: {fails}  ({time.time() - t_all:.0f} s)")
    sys.exit(1 if fails else 0)
