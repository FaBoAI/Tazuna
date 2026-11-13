"""Tests of skills.pick_place and skills.take_out: control flow with stand-ins for the arm, the
camera, perception and the grasp planner (every failure path), the geometry of the take-out
planner (takeout.py), then end-to-end on the simulated arm with the real perception and
planning on saved frames.

Run:  python tests/test_skills.py [-v] [--only NAME]
No hardware is touched.
"""

from __future__ import annotations

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import copy
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The recorded frames in tests/testdata/ were measured with the calibration of 2026-11-10 (before the
# shoulder_lift zero was corrected on 2026-11-11), and the stand-in world of these tests stands on
# that calibration's table. Pin it, as test_grasp.py and test_scene.py do: re-calibrating the real
# arm must not change these tests. A tree whose calibration has moved on keeps the old one in
# tests/testdata/calib_20261110; in a tree without that directory the tree's own files still are it.
# (test_e2e_sim_take_out_with_the_trees_own_calibration alone runs on the current calibration.)
_CALIB = os.path.join(HERE, "testdata", "calib_20261110")
if os.path.isfile(os.path.join(_CALIB, "config.json")):
    os.environ["FD_CONFIG"] = os.path.join(_CALIB, "config.json")
if os.path.isfile(os.path.join(_CALIB, "camera_model.json")):
    os.environ["FD_CAMERA_MODEL"] = os.path.join(_CALIB, "camera_model.json")
# The gripper in these tests is bare: they never read the tool file of the tree (tooling.py), so they pass
# whether or not a tool is attached to the real arm, and nobody has to `fd tool detach` to run them.
os.environ["FD_TOOL"] = os.path.join(HERE, "testdata", "no_tool_attached.json")

import fdtypes as T  # noqa: E402
import gripper_model as GM  # noqa: E402
import kinematics as K  # noqa: E402
import skills  # noqa: E402
import takeout  # noqa: E402
from robot import DEFAULT_CONFIG, MotionAborted, SafetyError, load_config  # noqa: E402

# Table height of the stand-in world (FakeArm, the synthetic scenes): read from the (pinned) configuration,
# never written here. Every height in the tests is relative to it.
TABLE_Z = float(load_config()["table_z"])
TOUCH = 2050


# ----------------------------------------------------------------------------- stand-ins
class FakeBus:
    def __init__(self, arm):
        self.arm = arm

    def read(self, joint, reg):
        assert joint == "gripper" and reg in ("Present_Position", "Present_Load"), (joint, reg)
        return self.arm.gripper_raw if reg == "Present_Position" else self.arm.gripper_load()


class FakeArm:
    """Ideal arm: real kinematics, instant motion, every call recorded in .calls."""

    def __init__(self, parked=True):
        self.cfg = copy.deepcopy(DEFAULT_CONFIG)
        self.cfg.update(table_z=TABLE_Z, floor_z=TABLE_Z - 0.004, gripper_touch_raw=TOUCH,
                        keepouts={"cables": [0.19, 0.45, -0.4, -0.135, -0.05, 0.03]})
        self.bus = FakeBus(self)
        self._tool = K.tool_offset(-0.001, 0.0, 0.0068)
        self._lim = {j: (math.radians(lo), math.radians(hi)) for j, (lo, hi) in
                     zip(K.ARM_JOINTS, [(-100, 100), (-98, 100), (-100, 90), (-100, 100), (-170, 185)])}
        self._ready = self.solve((0.20, 0.0, 0.12), 70.0, 0.0)
        self._rest = np.radians([-4.2, -96.0, 86.0, 68.0, 3.4])
        self.q = self._rest.copy() if parked else self._ready.copy()
        self.torqued = not parked
        self.gripper_raw = 2184
        self.gripper_goal = 2184
        self.temp = 30
        self.calls = []
        self.stall_raw = 2194  # where the jaws stop when something is between them (None = nothing)
        self.dropped = False  # the object left the jaws at the release (a later closing finds them empty)
        self.stuck_in_jaws = False  # ... unless it sticks to them
        self.lose_object_in_carry = False
        self.fail_run_path_at = None  # index of the run_path call that raises MotionAborted
        self._n_run = 0
        self.holding = False
        # ---- take-out
        self.pinch_raw = 2075  # where the jaws stop on a thin wall (None = nothing between them)
        self.surface_z = None  # fingertip height at which a touch descent lands (None = where the skill expects it)
        self.surface_offset = 0.0  # ... plus this (a height calibration error)
        self.expected_touch = None  # set by the test's listener when the skill announces a probe
        self.probe_script = []  # results for the next probe_down calls, instead of the model below
        self.probe_slide = 0.0  # sideways slip of the fingertip during a probe (m)
        self.slip_at = None  # the object slips out of the jaws during the n-th motion after they closed
        self._moves_held = 0
        self.released_at = None  # fingertip position where the jaws let the object go
        self.lost_at = None  # ... or where it slipped out

    # -- geometry
    def tip_tool(self):
        return self._tool

    def q_limits(self):
        return self._lim

    def solve(self, xyz, pitch_down_deg, roll_deg, q_seed=None, use_tip=True):
        q, info = K.ik(np.asarray(xyz, dtype=float), float(pitch_down_deg), math.radians(float(roll_deg)),
                       q_init=q_seed, tool=self._tool, limits=self._lim)
        if not info["ok"]:
            raise SafetyError(f"IK failed at {np.round(xyz, 3).tolist()} pitch {pitch_down_deg}: {info.get('reason')}")
        return np.asarray(q, dtype=float)

    def tip(self, q=None):
        return K.fk(self.q if q is None else q, self._tool)[:3, 3]

    def check_q(self, q, allow_contact=False, ignore_keepouts=(), what="target", q_start=None):
        p = self.tip(q)
        floor = self.cfg["floor_z"] + 0.004
        if q_start is not None:  # a motion that starts below the limit may go up from there (as robot.Arm allows)
            floor = min(floor, float(self.tip(q_start)[2]) - 0.001)
        if p[2] < floor - 1e-6 and not allow_contact:
            raise SafetyError(f"{what}: fingertip below the floor limit ({p[2]:.4f})")
        for name, b in self.cfg.get("keepouts", {}).items():
            if name in ignore_keepouts:
                continue
            if b[0] <= p[0] <= b[1] and b[2] <= p[1] <= b[3] and b[4] <= p[2] <= b[5]:
                raise SafetyError(f"{what}: fingertip inside keep-out '{name}'")

    def plan_tip_path(self, points, q_start=None):
        q = self.q.copy() if q_start is None else np.asarray(q_start, dtype=float)
        wps = [q.copy()]
        s = K.pose_summary(q, self._tool)
        p0, pitch0, roll0 = self.tip(q), float(s["pitch_down_deg"]), math.degrees(float(q[4]))
        for xyz, pitch, roll in points:
            xyz = np.asarray(xyz, dtype=float)
            n = max(1, int(math.ceil(np.linalg.norm(xyz - p0) / 0.005)), int(math.ceil(abs(pitch - pitch0) / 4.0)))
            for i in range(1, n + 1):
                a = i / n
                q = self.solve(p0 + a * (xyz - p0), pitch0 + a * (pitch - pitch0), roll0 + a * (roll - roll0), q_seed=q)
                wps.append(q)
            p0, pitch0, roll0 = xyz, float(pitch), float(roll)
        return wps

    # -- state
    def read_q(self):
        return self.q.copy()

    def is_torqued(self):
        return self.torqued

    def torque_on(self):
        self.calls.append(("torque_on",))
        self.torqued = True

    def torque_off(self):
        self.calls.append(("torque_off",))
        self.torqued = False

    def state(self):
        return {"temp_c": {j: self.temp for j in K.ARM_JOINTS}}

    def speeds(self, tier=None, allow_unvalidated=False):
        return {"tier": tier or "slow", "transit": 25.0, "approach": 12.0, "precise": 8.0, "validated": True}

    def ready_q(self):
        return self._ready.copy()

    def rest_q(self):
        return self._rest.copy()

    def gripper_q_deg(self, raw):
        return -11.41 + (raw - TOUCH) * 360.0 / 4096.0

    def gripper_raw_for(self, frac):
        return int(1998 + frac * (3648 - 1998))

    # -- motion
    def run_path(self, waypoints, speed_deg_s, allow_contact=False, ignore_keepouts=(), settle=True, hz=50.0,
                 monitor=None, tol_deg=0.25, gripper_goal_raw=None, contact_depth=None):
        assert self.torqued, "run_path without torque"
        assert np.allclose(waypoints[0], self.q, atol=1e-6), "path does not start at the current pose"
        self._n_run += 1
        if self.fail_run_path_at is not None and self._n_run == self.fail_run_path_at:
            self.q = np.asarray(waypoints[max(1, len(waypoints) // 2)], dtype=float).copy()
            raise MotionAborted("simulated tracking fault")
        for q in waypoints[1:]:
            if not (len(waypoints) == 2 or allow_contact):
                pass
            self.check_q(q, allow_contact=allow_contact or np.allclose(q, self._rest), ignore_keepouts=ignore_keepouts,
                         q_start=waypoints[0])
        if gripper_goal_raw is not None:
            self._gripper_to(gripper_goal_raw)
            self.dropped = False  # opening for the next object
        self.q = np.asarray(waypoints[-1], dtype=float).copy()
        if self.holding:
            self._moves_held += 1
        if self.holding and (self.lose_object_in_carry or (self.slip_at is not None and self._moves_held >= self.slip_at)):
            self.holding = False
            self.lost_at = self.tip().copy()
            self.gripper_raw = self.gripper_goal  # nothing between the jaws any more: they close on the goal
        self.calls.append(("run_path", len(waypoints), speed_deg_s, tuple(ignore_keepouts), gripper_goal_raw,
                           self.tip().round(4).tolist()))
        return {"joint_err_deg": [0.0] * 5, "corr_deg": [0.0] * 5, "duration_s": 0.0, "settle_s": 0.0,
                "peak_err_deg": 0.0, "aborted": False}

    def move_tip_path(self, points, speed_deg_s, **kw):
        return self.run_path(self.plan_tip_path(points), speed_deg_s, **kw)

    def _gripper_to(self, raw):
        self.gripper_goal = int(raw)
        if self.holding and raw > self.gripper_raw:
            self.holding = False
            self.dropped = not self.stuck_in_jaws
            self.released_at = self.tip().copy()
        if not self.holding:
            self.gripper_raw = int(raw)

    def set_gripper(self, frac=None, wait_s=1.2, raw=None, wait=True):
        goal = int(raw) if raw is not None else self.gripper_raw_for(frac)
        self.calls.append(("set_gripper", goal, wait))
        self._gripper_to(goal)
        return {"goal_raw": goal}

    def grasp_fast(self, expected_raw=None, squeeze_ticks=40, timeout_s=1.5, empty_margin=45):
        self.calls.append(("grasp_fast", self.tip().round(4).tolist()))
        if self.stall_raw is None or self.dropped:
            self.gripper_raw = self.gripper_goal = TOUCH
            return {"contact": False, "pos_raw": TOUCH, "hold_goal_raw": TOUCH, "q_jaw_deg": -11.4, "width_mm": 0.0,
                    "empty": True, "duration_s": 0.5, "load": 0}
        self.holding, self._moves_held = True, 0
        self.gripper_raw = self.stall_raw
        self.gripper_goal = max(TOUCH, self.stall_raw - squeeze_ticks)
        return {"contact": True, "pos_raw": self.stall_raw, "hold_goal_raw": self.gripper_goal, "q_jaw_deg": 1.2,
                "width_mm": 21.0, "empty": False, "duration_s": 0.5, "load": 300}

    def gripper_load(self):
        return 160 if self.holding else 70

    def pinch(self, extra_ticks=None, wait_s=1.2):
        """Arm.pinch: close on something thin; the numbers are those logged on 2026-11-11."""
        self.calls.append(("pinch", self.tip().round(4).tolist()))
        if self.pinch_raw is None or self.dropped:
            self.gripper_raw, self.gripper_goal = TOUCH + 1, TOUCH - 23
            return {"pos_raw": TOUCH + 1, "pos_before_squeeze": TOUCH + 17, "hold_goal_raw": TOUCH - 23, "load": 100,
                    "above_touch_ticks": 1}
        self.holding, self._moves_held = True, 0
        self.gripper_raw, self.gripper_goal = self.pinch_raw, self.pinch_raw - 37
        return {"pos_raw": self.pinch_raw, "pos_before_squeeze": self.pinch_raw + 3, "hold_goal_raw": self.pinch_raw - 37,
                "load": 160, "above_touch_ticks": self.pinch_raw - TOUCH}

    def probe_down(self, max_drop_m=0.06, speed_m_s=0.006, backoff_m=0.015, **kw):
        """Arm.probe_down: contact when the surface lies well inside the range; "inconclusive" when the
        descent starts on it or ends within a few mm of it; no contact when it is below the range.
        Ends backoff_m above the lowest point reached."""
        tip = self.tip()
        self.calls.append(("probe_down", round(float(max_drop_m), 4), tip.round(4).tolist()))
        lowest = float(tip[2] - max_drop_m)
        if self.probe_script:
            res = dict(self.probe_script.pop(0))
        else:
            z = self.surface_z if self.surface_z is not None else self.expected_touch
            z = None if z is None else float(z) + self.surface_offset
            if z is None or z < lowest - 1e-9:
                res = {"contact": False, "contact_tip_z": None, "lowest_tip_z": round(lowest, 4), "reason": None,
                       "free_air_gap": 0.0002}
            elif z > tip[2] - 0.012:
                res = {"contact": None, "contact_tip_z": None, "lowest_tip_z": round(z, 4),
                       "reason": "inconclusive: no free-air baseline could be formed", "free_air_gap": None}
            elif z < lowest + 0.006:
                res = {"contact": None, "contact_tip_z": None, "lowest_tip_z": round(z - 0.001, 4),
                       "reason": "inconclusive: the tip was 2.0 mm behind the plan at the end of the descent",
                       "free_air_gap": 0.0002}
            else:
                res = {"contact": True, "contact_tip_z": z, "lowest_tip_z": round(z - 0.001, 4), "reason": None,
                       "free_air_gap": 0.0002}
        end = res["lowest_tip_z"] if res.get("lowest_tip_z") is not None else lowest
        s = K.pose_summary(self.q, self._tool)
        self.q = self.solve((tip[0] + self.probe_slide, tip[1], end + backoff_m), float(s["pitch_down_deg"]),
                            math.degrees(float(self.q[4])))
        return res


def box(i, color, cx, cy, size=(0.045, 0.020, 0.018), yaw=0.0, z_bottom=TABLE_Z, kind="box", in_container=None):
    length, width, _h = size
    c, s = math.cos(yaw), math.sin(yaw)
    fp = [(cx + sx * length / 2 * c - sy * width / 2 * s, cy + sx * length / 2 * s + sy * width / 2 * c)
          for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return T.SceneObject(id=i, kind=kind, color=color, center=(cx, cy), yaw=yaw, size=size, z_bottom=z_bottom,
                         footprint=fp, pixel_bbox=(0, 0, 10, 10), confidence=0.9, in_container=in_container)


def container(i=9, cx=0.262, cy=0.085):
    o = box(i, "black", cx, cy, size=(0.111, 0.100, 0.030), yaw=math.radians(94), kind="container")
    o.inner = {"center": (cx, cy), "yaw": o.yaw, "size": (0.100, 0.090), "floor_z": TABLE_Z + 0.003,
               "rim_z": TABLE_Z + 0.030}
    return o


def scene(*objs):
    return T.Scene(objects=list(objs), table_z=TABLE_Z)


class FakeScene:
    """Perception stand-in. `scenes`: one Scene per perceive() call (the last repeats)."""

    def __init__(self, scenes, jaws=(), present_after=False, arrives=True):
        self.scenes = scenes if isinstance(scenes, list) else [scenes]
        self.jaws = list(jaws)
        self.present_after = present_after
        self.arrives = arrives
        self.n_perceive = 0
        self.n_inside = 0
        self.n_jaws = 0

    def perceive(self, img, cam, cfg, arm_q=None, jaw_q_deg=None, debug=None):
        sc = self.scenes[min(self.n_perceive, len(self.scenes) - 1)]
        self.n_perceive += 1
        self._sets_at_look = self._n_sets()
        return sc

    def _n_sets(self):
        return 0 if self.arm is None else sum(1 for c in self.arm.calls if c[0] == "set_gripper")

    def annotate(self, img, sc, cam):
        return img

    def objects_in_container(self, img, cam, cfg, con, arm_q=None, jaw_q_deg=None):
        self.n_inside += 1
        released = self._n_sets() > self._sets_at_look  # the jaws were opened since the last look
        if not released or not self.arrives:
            return []
        return [{"color": "blue", "area_px": 900.0}]

    arm = None  # set by run(): the stand-in looks at the arm's call log to know "before" from "after"

    def object_present(self, img, cam, cfg, obj, arm_q=None):
        return self.present_after

    def detect_jaws(self, img, ref, cam, arm_q, jaw_q_deg, cfg):
        self.n_jaws += 1
        if not self.jaws:
            return {"ok": False, "confidence": 0.0, "dxy_base": None}
        return self.jaws[min(self.n_jaws - 1, len(self.jaws) - 1)]


class FakeGrasp:
    """Planner stand-in: top-down grasp at the object's centre, release at the container centre."""

    def __init__(self, clearance=float("inf"), ok=True, n_cands=1, pitch=90.0):
        self.clearance, self.ok, self.n_cands, self.pitch = clearance, ok, n_cands, pitch
        self.calls = 0

    def plan_grasp(self, target, sc, arm, cfg=None):
        self.calls += 1
        cx, cy = target.center
        z = target.z_bottom + max(0.006, target.size[2] - 0.012)
        out = []
        for i in range(self.n_cands):
            out.append(T.GraspCandidate(
                ok=self.ok, reason="" if self.ok else "blocked by a neighbour", jaw_azimuth_deg=90.0 * i, roll_deg=0.0,
                pitch_deg=self.pitch, tip_grasp=(cx, cy, z), tip_hover=(cx, cy, target.z_top + 0.012),
                width=target.size[1], standoff=0.008, open_raw=2420, expected_raw=2194,
                clearance_fixed=self.clearance, clearance_moving=self.clearance, score=1.0 - 0.1 * i - 0.01 * target.id))
        return out

    def plan_release(self, target, con, cand, arm, cfg=None, scene=None):
        cx, cy = con.inner["center"]
        h_below = cand.tip_grasp[2] - target.z_bottom
        z_rel = con.inner["floor_z"] + h_below + 0.008
        z_above = con.inner["rim_z"] + h_below + 0.020
        p = min(skills._feasible_pitch(arm, (cx, cy, z), cand.roll_deg, cand.pitch_deg) for z in (z_rel, z_above))
        return T.ReleasePlan(ok=True, reason="", tip_above=(cx, cy, z_above), tip_release=(cx, cy, z_rel), pitch_deg=p,
                             roll_deg=cand.roll_deg, open_raw=cand.open_raw, drop_height=0.008)


def session(arm):
    cam = skills.ReplayCamera([np.zeros((4, 4, 3), np.uint8)])
    return skills.Session(arm, cam, cam_model=None, image_dir=None)


def run(arm, S, G, query=None, **opt):
    opt.setdefault("release_wait_s", 0.0)
    S.arm = arm
    return skills.pick_place(session(arm), query if query is not None else {"color": "blue"}, None,
                             skills.PickOptions(**opt), modules=(G, S))


BLUE = (0.2783, -0.0663)


def std_scene(extra=()):
    return scene(box(1, "blue", *BLUE, yaw=math.radians(6.8)), container(), *extra)


# ----------------------------------------------------------------------------- control-flow tests
def test_happy_path_from_parked():
    arm = FakeArm(parked=True)
    S, G = FakeScene(std_scene()), FakeGrasp()
    saved = copy.deepcopy(arm.cfg["keepouts"])
    r = run(arm, S, G, visual="off")
    assert r["ok"] and r["stage"] == "done" and r["arm_state"] == "standby", r
    assert r["attempts"] == 1 and r["target"]["id"] == 1 and r["container"]["id"] == 9
    kinds = [c[0] for c in arm.calls]
    assert kinds == ["torque_on", "run_path", "run_path", "run_path", "grasp_fast", "run_path", "set_gripper",
                     "run_path"], kinds
    # unfold without touching the gripper, then open it on the way to the hover point
    assert arm.calls[1][4] is None and arm.calls[2][4] == 2420
    # grasp happened at the planned fingertip position
    g_tip = arm.calls[4][1]
    assert np.allclose(g_tip[:2], BLUE, atol=1e-3) and abs(g_tip[2] - (TABLE_Z + 0.006)) < 1e-3, g_tip
    # release happened inside the container footprint, above its floor, and the jaws opened there
    rel_tip = arm.calls[5][5]
    assert np.allclose(rel_tip[:2], (0.262, 0.085), atol=1e-3) and rel_tip[2] > TABLE_Z + 0.003
    assert arm.calls[6] == ("set_gripper", 2420, False)
    # the arm waits straight above the release point, clear of the rim, torque on
    end = arm.tip()
    assert np.allclose(end[:2], rel_tip[:2], atol=1e-3) and end[2] >= TABLE_Z + 0.030 + 0.04 - 1e-6 and arm.torqued, end
    assert r["plan"]["standby"] == "above the container"
    assert arm.cfg["keepouts"] == saved, "keep-outs must be restored"
    assert r["checks"]["verify"]["arrived"] and not r["checks"]["verify"]["object_still_on_table"]
    assert set(("perceive", "plan", "approach", "descend", "grasp", "transfer", "release", "verify")) <= set(r["timings"])


def test_from_ready_is_one_approach_motion():
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene()), FakeGrasp(), visual="off")
    assert r["ok"], r
    kinds = [c[0] for c in arm.calls]
    assert kinds == ["run_path", "run_path", "grasp_fast", "run_path", "set_gripper", "run_path"], kinds


def test_standby_ready_option():
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene()), FakeGrasp(), visual="off", standby="ready")
    assert r["ok"] and r["arm_state"] == "ready" and np.allclose(arm.q[:4], arm.ready_q()[:4]), r


def test_second_pick_starts_from_standby():
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene()), FakeGrasp(), visual="off")
    assert r["ok"] and r["arm_state"] == "standby", r
    n0 = len(arm.calls)
    sc2 = scene(box(2, "blue", 0.25, -0.11), container())
    r = run(arm, FakeScene(sc2), FakeGrasp(), visual="off")
    assert r["ok"] and r["arm_state"] == "standby", r
    second = arm.calls[n0:]
    assert [c[0] for c in second] == ["run_path", "run_path", "grasp_fast", "run_path", "set_gripper", "run_path"]
    # straight across from above the container: 1 start + 1 transit waypoint + the vertical arrival
    assert second[0][1] == 1 + r["plan"]["waypoints"]["approach"], second[0]


def test_hidden_target_gets_a_second_look_from_ready():
    arm = FakeArm(parked=False)
    arm.q = arm.solve((0.262, 0.085, TABLE_Z + 0.095), 80.0, 0.0)  # waiting above the container
    assert skills._arm_state(arm, arm.q) == "standby"
    S = FakeScene([scene(container()), std_scene()])  # first look: the object is hidden
    r = run(arm, S, FakeGrasp(), visual="off")
    assert r["ok"] and S.n_perceive == 2, r
    first = arm.calls[0]
    assert first[0] == "run_path" and np.allclose(first[5], arm.tip(arm.ready_q()).round(4), atol=1e-3), first
    # ... but not when the arm already stands at the ready pose, and never in a dry run
    arm = FakeArm(parked=False)
    S = FakeScene([scene(container()), std_scene()])
    r = run(arm, S, FakeGrasp(), visual="off")
    assert not r["ok"] and r["stage"] == "perceive" and S.n_perceive == 1 and arm.calls == [], r
    arm = FakeArm(parked=False)
    arm.q = arm.solve((0.262, 0.085, TABLE_Z + 0.095), 80.0, 0.0)
    S = FakeScene([scene(container()), std_scene()])
    r = run(arm, S, FakeGrasp(), dry=True)
    assert not r["ok"] and S.n_perceive == 1 and arm.calls == [], r


def test_hidden_container_gets_a_second_look_from_ready():
    arm = FakeArm(parked=False)
    arm.q = arm.solve((0.262, 0.085, TABLE_Z + 0.095), 80.0, 0.0)  # waiting above the container, hiding its rim
    S = FakeScene([scene(box(1, "blue", *BLUE)), std_scene()])
    r = run(arm, S, FakeGrasp(), visual="off")
    assert r["ok"] and S.n_perceive == 2 and r["container"]["id"] == 9, r


def test_plan_transit():
    arm = FakeArm(parked=False)
    hi = arm.solve((0.25, -0.10, 0.06), 90.0, 0.0)
    over = arm.solve((0.262, 0.085, 0.07), 85.0, 0.0)
    w = skills.plan_transit(arm, over, hi, z_clear=0.03)
    assert len(w) == 1 and np.allclose(w[0], hi), "high to high: straight across"
    low = arm.solve((0.25, 0.0, TABLE_Z + 0.01), 90.0, 0.0)
    w = skills.plan_transit(arm, low, hi, z_clear=0.03)
    assert len(w) > 1 and np.allclose(w[-1], hi)
    tips = [arm.tip(q) for q in w]
    assert np.allclose(tips[0][:2], (0.25, 0.0), atol=1e-3) and tips[0][2] > TABLE_Z + 0.01, "a low start is raised straight up first"
    assert abs(max(t[2] for t in tips[:-1]) - 0.03) < 2e-3
    # a keep-out wall between the two poses: go round through the ready pose
    arm.cfg["keepouts"]["wall"] = [0.20, 0.30, -0.02, 0.0, -0.05, 0.10]
    w = skills.plan_transit(arm, over, hi, z_clear=0.03)
    assert len(w) == 2 and np.allclose(w[0], arm.ready_q()) and np.allclose(w[1], hi)
    # ... and an honest failure when that is blocked too
    arm.cfg["keepouts"]["wall"] = [0.10, 0.30, -0.02, 0.0, -0.05, 0.30]
    try:
        skills.plan_transit(arm, over, hi, z_clear=0.03)
        raise AssertionError("should be blocked")
    except SafetyError:
        pass


def test_end_park():
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene()), FakeGrasp(), visual="off", end="park")
    assert r["ok"] and r["arm_state"] == "parked" and not arm.torqued, r
    assert np.allclose(arm.q, arm.rest_q())


def test_dry_moves_nothing():
    arm = FakeArm(parked=True)
    q0 = arm.q.copy()
    r = run(arm, FakeScene(std_scene()), FakeGrasp(), dry=True)
    assert r["ok"] and r["stage"] == "planned" and r["arm_state"] == "parked", r
    assert arm.calls == [] and np.allclose(arm.q, q0) and not arm.torqued
    assert r["plan"]["waypoints"]["carry"] > 5 and "grasp" in r and "release" in r


def test_target_not_found():
    arm = FakeArm()
    r = run(arm, FakeScene(std_scene()), FakeGrasp(), query={"color": "green"})
    assert not r["ok"] and r["stage"] == "perceive" and "target not found" in r["reason"], r
    assert arm.calls == [] and r["arm_state"] == "parked"


def test_object_already_inside_is_not_a_target():
    arm = FakeArm()
    sc = scene(box(1, "blue", 0.262, 0.085, z_bottom=TABLE_Z + 0.003, in_container=9), container())
    r = run(arm, FakeScene(sc), FakeGrasp())
    assert not r["ok"] and r["stage"] == "perceive" and "target not found" in r["reason"], r
    r = run(arm, FakeScene(sc), FakeGrasp(), query={"id": 1})
    assert not r["ok"] and "already inside" in r["reason"], r
    r = run(arm, FakeScene(sc), FakeGrasp(), query={"id": 9})
    assert not r["ok"] and "is a container" in r["reason"], r


def test_ambiguous_asks_unless_any_match():
    sc = std_scene(extra=(box(2, "blue", 0.25, -0.12),))
    arm = FakeArm()
    r = run(arm, FakeScene(sc), FakeGrasp())
    assert not r["ok"] and r["stage"] == "perceive" and "2 objects match" in r["reason"], r
    assert r["checks"]["failures"][0]["matches"] == [1, 2] and arm.calls == []
    arm = FakeArm()
    r = run(arm, FakeScene(sc), FakeGrasp(), any_match=True, visual="off")
    assert r["ok"] and r["target"]["id"] == 1, r  # the better-scoring one
    # nearest_to / largest resolve the ambiguity without asking
    arm = FakeArm()
    r = run(arm, FakeScene(sc), FakeGrasp(), query={"color": "blue", "nearest_to": (0.25, -0.12)}, visual="off")
    assert r["ok"] and r["target"]["id"] == 2, r


def test_no_container_or_two():
    arm = FakeArm()
    r = run(arm, FakeScene(scene(box(1, "blue", *BLUE))), FakeGrasp())
    assert not r["ok"] and "container not found" in r["reason"], r
    sc = scene(box(1, "blue", *BLUE), container(8, 0.20, 0.14), container(9))
    r = run(arm, FakeScene(sc), FakeGrasp())
    assert not r["ok"] and "2 containers match" in r["reason"], r


def test_plan_failure_reports_reasons():
    arm = FakeArm()
    r = run(arm, FakeScene(std_scene()), FakeGrasp(ok=False))
    assert not r["ok"] and r["stage"] == "plan" and "blocked by a neighbour" in r["reason"], r
    assert arm.calls == []


def test_unreachable_target_fails_before_moving():
    arm = FakeArm()
    sc = scene(box(1, "blue", 0.46, 0.0), container())
    r = run(arm, FakeScene(sc), FakeGrasp())
    assert not r["ok"] and r["stage"] == "plan" and arm.calls == [], r


def test_keepout_blocks_plan():
    arm = FakeArm()
    sc = scene(box(1, "blue", 0.21, -0.15), container())  # inside the "cables" keep-out
    r = run(arm, FakeScene(sc), FakeGrasp())
    assert not r["ok"] and r["stage"] == "plan" and "keep-out" in r["reason"] and arm.calls == [], r


def test_hot_servo_refused():
    arm = FakeArm()
    arm.temp = 61
    r = run(arm, FakeScene(std_scene()), FakeGrasp())
    assert not r["ok"] and r["stage"] == "approach" and "cool" in r["reason"], r
    assert arm.calls == [] and r["arm_state"] == "parked"


def test_empty_grasp_retries_then_reports():
    arm = FakeArm(parked=False)
    arm.stall_raw = None
    S = FakeScene(std_scene())
    r = run(arm, S, FakeGrasp(), visual="off", retries=1)
    assert not r["ok"] and r["stage"] == "grasp" and r["attempts"] == 2 and S.n_perceive == 2, r
    assert r["arm_state"] == "ready" and np.allclose(arm.q[:4], arm.ready_q()[:4])
    assert len(r["checks"]["failures"]) == 2
    assert not any(c[0] == "set_gripper" for c in arm.calls), "nothing to release"
    # after the miss the arm first rose with the jaws as they were, then opened on the way back
    i = [k for k, c in enumerate(arm.calls) if c[0] == "grasp_fast"][0]
    assert arm.calls[i + 1][0] == "run_path" and arm.calls[i + 1][4] is None
    assert arm.calls[i + 2][0] == "run_path" and arm.calls[i + 2][4] == 2420


def test_empty_grasp_then_success():
    arm = FakeArm(parked=False)
    arm.stall_raw = None
    S = FakeScene(std_scene())
    real = arm.grasp_fast

    def second_time_lucky(**kw):
        if S.n_perceive >= 2:
            arm.stall_raw = 2194
        return real(**kw)

    arm.grasp_fast = second_time_lucky
    r = run(arm, S, FakeGrasp(), visual="off", retries=1)
    assert r["ok"] and r["attempts"] == 2 and r["arm_state"] == "standby", r


def test_object_lost_in_carry():
    arm = FakeArm(parked=False)
    arm.lose_object_in_carry = True
    r = run(arm, FakeScene(std_scene()), FakeGrasp(), visual="off")
    assert not r["ok"] and r["stage"] == "transfer" and "lost" in r["reason"], r
    assert r["arm_state"] == "ready" and r["attempts"] == 1
    assert not any(c[0] == "set_gripper" for c in arm.calls)


def test_verify_failures():
    # still standing where it was: a failure, after one more look from the ready pose
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), present_after=True), FakeGrasp(), visual="off")
    assert not r["ok"] and r["stage"] == "verify" and r["arm_state"] == "ready" and r["attempts"] == 1, r
    assert r["checks"]["verify"]["first_look"] == {"object_still_on_table": True, "arrived": True}
    assert np.allclose(arm.q[:4], arm.ready_q()[:4])
    # verify=False skips all of it
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), arrives=False, present_after=True), FakeGrasp(), visual="off", verify=False)
    assert r["ok"], r


def test_verify_inferred_when_the_camera_cannot_tell():
    # gone from the table, nothing new seen in the container (it landed on something of its own
    # colour), nothing of that colour on the table, jaws empty: accepted, and said so
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), arrives=False), FakeGrasp(), visual="off")
    v = r["checks"]["verify"]
    assert r["ok"] and v["confirmed"] and v["jaws_empty"] is True and v["how"].startswith("inferred"), r
    assert [c[0] for c in arm.calls].count("grasp_fast") == 2
    # ... but not when the jaws still hold something
    arm = FakeArm(parked=False)
    arm.stuck_in_jaws = True
    r = run(arm, FakeScene(std_scene(), arrives=False), FakeGrasp(), visual="off")
    assert not r["ok"] and r["stage"] == "verify" and "still in the gripper" in r["reason"], r
    assert r["arm_state"] == "holding_object"


def test_verify_finds_an_object_dropped_beside_the_container():
    arm = FakeArm(parked=False)
    after = scene(box(7, "blue", 0.20, 0.02), container())  # a blue object where there was none
    S = FakeScene([std_scene(), after], arrives=False)
    r = run(arm, S, FakeGrasp(), visual="off")
    assert not r["ok"] and r["stage"] == "verify" and "outside the container" in r["reason"], r
    assert r["checks"]["verify"]["stray"]["center"] == [0.2, 0.02] and "jaws_empty" not in r["checks"]["verify"]


def test_motion_fault_while_holding():
    arm = FakeArm(parked=False)
    arm.fail_run_path_at = 3  # the carry
    r = run(arm, FakeScene(std_scene()), FakeGrasp(), visual="off")
    assert not r["ok"] and "MotionAborted" in r["reason"] and r["arm_state"] == "holding_object", r
    assert r["stage"] == "transfer" and arm.torqued


def test_motion_fault_on_approach():
    arm = FakeArm(parked=False)
    arm.fail_run_path_at = 1
    r = run(arm, FakeScene(std_scene()), FakeGrasp(), visual="off")
    assert not r["ok"] and r["stage"] == "approach" and r["arm_state"] == "stopped", r


def jaws(dx_mm, dy_mm, conf=0.9, ok=True):
    return {"ok": ok, "confidence": conf, "dxy_base": (dx_mm / 1000.0, dy_mm / 1000.0)}


def test_visual_small_offset_no_correction():
    arm = FakeArm(parked=False)
    S = FakeScene(std_scene(), jaws=[jaws(1.0, -1.0)])
    r = run(arm, S, FakeGrasp())
    v = r["checks"]["visual"]
    assert r["ok"] and v["seen"] and not v["applied"] and v["corr_mm"] == [0.0, 0.0] and S.n_jaws == 1, r


def test_visual_correction_shifts_grasp():
    arm = FakeArm(parked=False)
    S = FakeScene(std_scene(), jaws=[jaws(4.0, -3.0)])
    r = run(arm, S, FakeGrasp(), visual="on")
    v = r["checks"]["visual"]
    assert r["ok"] and v["applied"] and v["corr_mm"] == [-4.0, 3.0] and S.n_jaws == 2, r
    g_tip = [c for c in arm.calls if c[0] == "grasp_fast"][0][1]
    assert np.allclose(g_tip[:2], (BLUE[0] - 0.004, BLUE[1] + 0.003), atol=3e-4), g_tip


def test_visual_auto_records_only_when_roomy():
    arm = FakeArm(parked=False)
    S = FakeScene(std_scene(), jaws=[jaws(4.0, -3.0)])
    r = run(arm, S, FakeGrasp())
    v = r["checks"]["visual"]
    assert r["ok"] and v["seen"] and not v["applied"] and v["steps"][0]["dxy_mm"] == [4.0, -3.0] and S.n_jaws == 1, r
    g_tip = [c for c in arm.calls if c[0] == "grasp_fast"][0][1]
    assert np.allclose(g_tip[:2], BLUE, atol=3e-4), g_tip
    # a wild reading does not stop a roomy grasp either; it is reported
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), jaws=[jaws(12.0, 6.0)]), FakeGrasp())
    assert r["ok"] and "13 mm" in r["checks"]["visual"]["warning"], r
    # opt-in: correct roomy grasps too
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), jaws=[jaws(4.0, -3.0)]), FakeGrasp(), visual_apply_when_roomy=True)
    assert r["ok"] and r["checks"]["visual"]["corr_mm"] == [-4.0, 3.0], r


def test_options_from_config():
    o = skills.options_from_config({"pick": {"hover_margin": 0.02, "visual": "off", "bogus": 1}}, tier=None, retries=3)
    assert o.hover_margin == 0.02 and o.visual == "off" and o.retries == 3 and o.tier is None
    o = skills.options_from_config({}, visual="on")
    assert o.visual == "on" and o.hover_margin == skills.PickOptions().hover_margin


def test_visual_tight_grasp_confirms():
    arm = FakeArm(parked=False)
    S = FakeScene(std_scene(), jaws=[jaws(4.0, -3.0), jaws(4.4, -2.6)])
    r = run(arm, S, FakeGrasp(clearance=0.003))
    v = r["checks"]["visual"]
    assert r["ok"] and S.n_jaws == 2 and v["corr_mm"] == [-4.0, 3.0], r  # second look agreed: keep the first
    g_tip = [c for c in arm.calls if c[0] == "grasp_fast"][0][1]
    assert np.allclose(g_tip[:2], (BLUE[0] - 0.004, BLUE[1] + 0.003), atol=3e-4), g_tip


def test_visual_tight_grasp_unstable_detection_aborts():
    arm = FakeArm(parked=False)
    S = FakeScene(std_scene(), jaws=[jaws(4.0, -3.0), jaws(-2.0, 3.0)])
    r = run(arm, S, FakeGrasp(clearance=0.003))
    assert not r["ok"] and r["stage"] == "visual" and "disagree" in r["reason"], r
    assert r["arm_state"] == "ready" and not any(c[0] == "grasp_fast" for c in arm.calls)


def test_visual_tight_needs_the_fixed_jaw_in_view():
    partial = {"ok": True, "confidence": 0.8, "dxy_base": (0.0, -0.0005), "fixed": {"visible": False}}
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), jaws=[partial]), FakeGrasp(clearance=0.003))
    assert not r["ok"] and r["stage"] == "visual" and "fixed jaw is hidden" in r["reason"], r
    assert r["arm_state"] == "ready" and not any(c[0] == "grasp_fast" for c in arm.calls)
    # a roomy grasp does not need it
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), jaws=[partial]), FakeGrasp())
    assert r["ok"] and r["checks"]["visual"]["steps"][0]["full"] is False, r
    # garbage from the detector is treated as "not seen", never as a number
    for bad in ({"ok": True, "confidence": 0.9, "dxy_base": (float("nan"), 0.0)}, {"ok": True, "confidence": 0.9, "dxy_base": (0.001,)},
                {"ok": True, "confidence": None, "dxy_base": "x"}):
        arm = FakeArm(parked=False)
        r = run(arm, FakeScene(std_scene(), jaws=[bad]), FakeGrasp())
        assert r["ok"] and not r["checks"]["visual"]["seen"] and not r["checks"]["visual"]["applied"], (bad, r)


def test_visual_too_far_aborts():
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), jaws=[jaws(12.0, 6.0)]), FakeGrasp(), visual="on")
    assert not r["ok"] and r["stage"] == "visual" and "13 mm" in r["reason"] and r["arm_state"] == "ready", r
    assert not any(c[0] == "grasp_fast" for c in arm.calls)


def test_visual_unseen():
    # roomy grasp: carry on without the visual confirmation
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), jaws=[jaws(1, 1, conf=0.3)]), FakeGrasp())
    assert r["ok"] and not r["checks"]["visual"]["seen"], r
    # tight grasp, or visual="on": stop
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), jaws=[jaws(1, 1, conf=0.3)]), FakeGrasp(clearance=0.003))
    assert not r["ok"] and r["stage"] == "visual" and "tight" in r["reason"] and r["arm_state"] == "ready", r
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(), jaws=[]), FakeGrasp(), visual="on")
    assert not r["ok"] and r["stage"] == "visual", r


def test_carry_height_clears_other_objects():
    tall = box(3, "red", 0.27, 0.0, size=(0.04, 0.04, 0.06))  # between the object and the container
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene(extra=(tall,))), FakeGrasp(), visual="off")
    assert r["ok"], r
    h_below = 0.006
    assert r["plan"]["z_carry"] >= tall.z_top + h_below + 0.015 - 1e-6, r["plan"]
    assert any("carry height raised" in n for n in r["plan"]["notes"])


def test_second_candidate_used_when_first_path_fails():
    # pitch 90 is not feasible this far out at carry height: FakeGrasp offers it first; the plan
    # must not die on it if _feasible_pitch can degrade, and must report failures otherwise
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(std_scene()), FakeGrasp(n_cands=3), visual="off")
    assert r["ok"] and len(r["grasp_candidates"]) >= 1, r


def test_select_objects():
    sc = std_scene(extra=(box(2, "red", 0.25, -0.12, kind="cylinder"), box(3, "blue", 0.2, 0.0, size=(0.06, 0.03, 0.02))))
    ids = lambda q, **k: [o.id for o in skills.select_objects(sc, q, **k)]  # noqa: E731
    assert ids({}) == [1, 2, 3] and ids({"color": "blue"}) == [1, 3] and ids({"kind": "cylinder"}) == [2]
    assert ids({"color": "blue", "largest": True}) == [3] and ids({}, containers=True) == [9]
    assert ids({"color": "black"}, containers=True) == [9]
    for bad in ({"color": "white"}, {"id": 77}, {"id": 9}):
        try:
            skills.select_objects(sc, bad)
            raise AssertionError(f"{bad} should not match")
        except LookupError:
            pass


def test_inside_change():
    f = skills._inside_change
    assert f([], [{"color": "blue", "area_px": 3000}], "blue")["arrived"]
    assert not f([], [{"color": "red", "area_px": 1200}], "blue")["arrived"]
    assert not f([{"color": "blue", "area_px": 3000}], [{"color": "blue", "area_px": 3100}], "blue")["arrived"]
    assert f([{"color": "blue", "area_px": 3000}], [{"color": "blue", "area_px": 6500}], "blue")["arrived"]  # merged blob
    # a speck, or something of that colour at the arm (still in the jaws), is not evidence
    assert not f([], [{"color": "blue", "area_px": 300}], "blue")["arrived"]
    r = f([], [{"color": "blue", "area_px": 5000, "near_arm": True}], "blue")
    assert not r["arrived"] and r["near_arm"] is True
    # the real case of 2026-11-11: an ivory cylinder landed on a white cap and the pile reads "white"
    before = [{"color": "blue", "area_px": 19096}, {"color": "red", "area_px": 11885}, {"color": "white", "area_px": 11720}]
    after = [{"color": "blue", "area_px": 18040}, {"color": "white", "area_px": 17253}, {"color": "red", "area_px": 12108}]
    r = f(before, after, "ivory", expect_px=0.6 * 112 * 140)
    assert r["arrived"] and set(r["evidence"]) == {"more of its colour", "more in total"}, r
    # frame-to-frame jitter of the same contents is not an arrival
    jitter = [{"color": "blue", "area_px": 19300}, {"color": "red", "area_px": 12000}, {"color": "white", "area_px": 11900}]
    assert not f(before, jitter, "ivory", expect_px=0.6 * 112 * 140)["arrived"]
    # an object that lands on one of ANOTHER colour still shows up in the total
    after2 = [{"color": "blue", "area_px": 23500}, {"color": "red", "area_px": 11885}, {"color": "white", "area_px": 11720}]
    assert f(before, after2, "green", expect_px=9000)["evidence"] == ["more in total"]


def test_second_look_confirms_what_the_arm_was_hiding():
    arm = FakeArm(parked=False)
    S = FakeScene(std_scene())
    looks = {"n": 0}

    def present(img, cam, cfg, obj, arm_q=None):  # hidden behind the arm on the first look ("cannot tell" = True)
        looks["n"] += 1
        return looks["n"] == 1

    S.object_present = present
    r = run(arm, S, FakeGrasp(), visual="off")
    assert r["ok"] and r["arm_state"] == "ready" and looks["n"] == 2, r
    assert r["checks"]["verify"]["first_look"]["object_still_on_table"] is True


def test_uncertain_detections_need_an_explicit_id():
    vague = box(1, "blue", *BLUE)
    vague.confidence, vague.notes = 0.3, ["base hidden behind another object"]
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(scene(vague, container())), FakeGrasp(), visual="off")
    assert not r["ok"] and r["stage"] == "perceive" and "uncertain" in r["reason"] and "id 1" in r["reason"], r
    assert arm.calls == []
    r = run(arm, FakeScene(scene(vague, container())), FakeGrasp(), visual="off", any_match=True)
    assert not r["ok"] and arm.calls == [], "--all must not take an uncertain detection either"
    r = run(arm, FakeScene(scene(vague, container())), FakeGrasp(), query={"id": 1}, visual="off")
    assert r["ok"], r
    # a sure one and an unsure one of the same colour: the sure one is taken without asking
    sure = box(2, "blue", 0.25, -0.12)
    arm = FakeArm(parked=False)
    r = run(arm, FakeScene(scene(vague, sure, container())), FakeGrasp(), visual="off")
    assert r["ok"] and r["target"]["id"] == 2, r


def test_other_colours_match_through_their_note():
    o = box(1, "other", *BLUE)
    o.notes = ["colour: orange"]
    sc = scene(o, container())
    assert [x.id for x in skills.select_objects(sc, {"color": "orange"})] == [1]
    try:
        skills.select_objects(sc, {"color": "purple"})
        raise AssertionError("purple should not match an orange object")
    except LookupError:
        pass


class _DeadCamera:
    def __init__(self, good=1):
        self.good = good

    def latest(self, n_avg=1, max_age_s=0.2):
        if self.good <= 0:
            raise RuntimeError("camera stalled")
        self.good -= 1
        return np.zeros((4, 4, 3), np.uint8)

    def wait_fresh(self, after_time):
        raise RuntimeError("camera stalled")


def _run_with_camera(arm, S, camera, **opt):
    S.arm = arm
    opt.setdefault("release_wait_s", 0.0)
    sess = skills.Session(arm, camera, cam_model=None, image_dir=None)
    return skills.pick_place(sess, {"color": "blue"}, None, skills.PickOptions(**opt), modules=(FakeGrasp(), S))


def test_camera_failures_never_crash_the_sequence():
    # no frame at all: nothing moves
    arm = FakeArm(parked=False)
    r = _run_with_camera(arm, FakeScene(std_scene()), _DeadCamera(good=0))
    assert not r["ok"] and r["stage"] == "perceive" and "no camera frame" in r["reason"] and arm.calls == [], r
    # camera dies after the first frame: the look at the jaws counts as "not seen" (roomy grasp
    # goes on), the pick is completed, only the final confirmation is missing
    arm = FakeArm(parked=False)
    r = _run_with_camera(arm, FakeScene(std_scene()), _DeadCamera(good=1))
    assert not r["ok"] and r["stage"] == "verify" and "no camera frame" in r["reason"], r
    assert r["arm_state"] == "standby" and any(c[0] == "set_gripper" for c in arm.calls)
    assert r["checks"]["visual"]["seen"] is False


def test_internal_error_is_reported_not_raised():
    arm = FakeArm(parked=False)
    S = FakeScene(std_scene())

    def boom(*a, **k):
        raise ValueError("bad array")

    S.objects_in_container = boom
    r = run(arm, S, FakeGrasp(), visual="off")
    assert not r["ok"] and "internal error: ValueError" in r["reason"] and r["arm_state"] == "ready" and arm.calls == [], r
    assert arm.cfg["keepouts"] == FakeArm().cfg["keepouts"]


def test_container_geometry_remembered_within_a_session():
    arm = FakeArm(parked=False)
    good = container()
    S = FakeScene(std_scene())
    S.arm = arm
    sess = skills.Session(arm, skills.ReplayCamera([np.zeros((4, 4, 3), np.uint8)]), None, image_dir=None)
    r = skills.pick_place(sess, {"color": "blue"}, None, skills.PickOptions(visual="off", release_wait_s=0.0),
                          modules=(FakeGrasp(), S))
    assert r["ok"] and len(sess.known_containers) == 1, r
    # second pick: the arm waits above the container, which is now seen badly (shifted, rim unknown)
    vague = container(9, 0.262 + 0.006, 0.085 - 0.004)
    vague.confidence, vague.inner = 0.4, {**good.inner, "rim_z": TABLE_Z + 0.05, "center": (0.268, 0.081)}
    S2 = FakeScene(scene(box(2, "blue", 0.25, -0.11), vague))
    S2.arm = arm
    r = skills.pick_place(sess, {"color": "blue"}, None, skills.PickOptions(visual="off", release_wait_s=0.0),
                          modules=(FakeGrasp(), S2))
    assert r["ok"] and abs(r["container"]["inner"]["rim_z"] - (TABLE_Z + 0.030)) < 1e-9, r["container"]
    assert np.allclose(r["release"]["tip_release"][:2], (0.262, 0.085), atol=1e-6)


# ----------------------------------------------------------------------------- take-out: stand-ins and scenes
FLOOR = TABLE_Z + 0.005  # inner floor of the test container
RIM = TABLE_Z + 0.030  # its rim
BOX_C = (0.225, 0.085)  # its centre; the inside is 105 x 94 mm
X_WALL = BOX_C[0] - 0.0525  # inner face of the wall nearest to the robot
CAP_R = 0.0155  # the white bottle cap of 2026-11-11: 31 mm across, 16.5 mm high, open side up
CAP_INNER = FLOOR + 0.002  # its inner bottom (the default model: 2 mm above the surface it stands on)
TIP_ABOVE = 0.002  # the jaws close with the fingertip this far above the expected inner bottom
H_BELOW = CAP_INNER + TIP_ABOVE - FLOOR  # ... so the cap's bottom hangs this far below the fingertip
TIP_OFF = (-0.001, 0.0, 0.0068)  # the fingertip in tool axes, as on the real arm
PLAN_CFG = {"tip_offset": TIP_OFF, "gripper_touch_raw": TOUCH, "table_z": TABLE_Z,
            "gripper": {"closed_raw": 1998, "open_raw": 3648, "margin": 25}}


def cyl(i, color, cx, cy, d=2 * CAP_R, h=0.0165, z_bottom=FLOOR, in_container=9, hollow=None):
    fp = [(cx + d / 2 * math.cos(2 * math.pi * k / 16), cy + d / 2 * math.sin(2 * math.pi * k / 16)) for k in range(16)]
    o = T.SceneObject(id=i, kind="cylinder", color=color, center=(cx, cy), yaw=0.0, size=(d, d, h), z_bottom=z_bottom,
                      footprint=fp, pixel_bbox=(0, 0, 60, 50), confidence=0.9, in_container=in_container)
    if hollow is not None:
        o.hollow = hollow  # the optional SceneObject field (set whether or not fdtypes declares it yet)
    return o


def take_box(yaw=0.0, cx=BOX_C[0], cy=BOX_C[1]):
    o = box(9, "black", cx, cy, size=(0.111, 0.100, 0.030), yaw=yaw, kind="container")
    o.inner = {"center": (cx, cy), "yaw": yaw, "size": (0.105, 0.094), "floor_z": FLOOR, "rim_z": RIM}
    return o


def cap_scene(*extra, cap_xy=BOX_C, **cap_kw):
    """A white cap in the test container, plus whatever else is given."""
    return scene(cyl(1, "white", *cap_xy, **cap_kw), take_box(), *extra)


def hemmed_scene(wall_gap=0.022, fourth=False):
    """The cap as on 2026-11-11: the wall `wall_gap` behind it (towards the robot), an ivory cylinder
    touching it on one side, a low blue disc touching it in front, a red block 3 mm away on the other
    side. fourth=True: the wall is only 8 mm away, so no side is left."""
    cx = X_WALL + (0.008 if fourth else wall_gap) + CAP_R
    cy = 0.070
    return scene(cyl(1, "white", cx, cy),
                 cyl(2, "ivory", cx, cy - CAP_R - 0.013 - 0.0005, d=0.026, h=0.022),
                 cyl(3, "blue", cx + CAP_R + 0.024 + 0.0005, cy, d=0.048, h=0.012),
                 box(4, "red", cx + 0.010, cy + CAP_R + 0.012 + 0.003, size=(0.045, 0.024, 0.020), z_bottom=FLOOR,
                     in_container=9),
                 take_box())


class FakeTakeScene:
    """Perception stand-in for take-out: what the camera sees follows the arm. The object lies in the
    container until the jaws hold it, is gone from there while it is carried, and stands on the table
    where the jaws let go of it (or where it slipped out of them)."""

    arm = None

    def __init__(self, before, target_id=1, jaws=None, stays=False, lands=True, hidden=False, behind_arm=None):
        self.before, self.target_id = before, target_id
        self.jaws = [globals()["jaws"](0.4, -0.3)] if jaws is None else list(jaws)
        self.stays, self.lands = stays, lands
        self.hidden = hidden  # True: the arm hides the object's place while it is lifted; "always": at the end too
        self.behind_arm = behind_arm  # what perceive returns while the arm is not folded back (None: the truth)
        self.n_perceive = self.n_jaws = 0
        self.look_z = []  # fingertip height at every look at the jaws

    def _target(self):
        return [o for o in self.before.objects if o.id == self.target_id][0]

    def _where(self):
        """None: in the container; "jaws": held; "gone": nowhere to be seen; (x, y): on the table."""
        a = self.arm
        if self.stays or a is None:
            return None
        if a.holding:
            return "jaws"
        p = a.released_at if a.released_at is not None else a.lost_at
        if p is None:
            return None
        con = self.before.containers()[0]
        if math.hypot(p[0] - con.center[0], p[1] - con.center[1]) < 0.05:
            return None  # it fell back into the container
        return (float(p[0]), float(p[1])) if self.lands else "gone"

    def perceive(self, img, cam, cfg, arm_q=None, jaw_q_deg=None, debug=None):
        self.n_perceive += 1
        w = self._where()
        if w is None and self.behind_arm is not None and not skills._is_parked(self.arm.read_q()):
            return self.behind_arm
        if w is None:
            return self.before
        objs = [o for o in self.before.objects if o.id != self.target_id]
        tz = self.before.table_z
        if isinstance(w, tuple):
            t = self._target()
            objs.append(cyl(50, t.color, w[0], w[1], d=t.size[0], h=t.size[2], z_bottom=tz, in_container=None))
        return T.Scene(objects=objs, table_z=tz)

    def annotate(self, img, sc, cam):
        return img

    def objects_in_container(self, img, cam, cfg, con, arm_q=None, jaw_q_deg=None):
        out = [{"color": o.color, "area_px": 3000.0} for o in self.before.objects
               if o.in_container == con.id and o.id != self.target_id]
        if self._where() is None or self.hidden == "always":
            out.append({"color": self._target().color, "area_px": 3000.0, "near_arm": self.hidden == "always"})
        return out

    def object_presence(self, img, cam, cfg, obj, arm_q=None, jaw_q_deg=None):
        w = self._where()
        if self.hidden == "always" or (self.hidden and w == "jaws"):
            return {"present": None, "color_now": None}  # the arm hides the place
        return {"present": w is None, "color_now": obj.color if w is None else None}

    def detect_jaws(self, img, ref, cam, arm_q, jaw_q_deg, cfg):
        self.n_jaws += 1
        self.look_z.append(float(K.fk(self.arm.read_q(), self.arm.tip_tool())[2, 3]))
        return self.jaws[min(self.n_jaws - 1, len(self.jaws) - 1)]


def take(arm, S, query=None, planner=None, **opt):
    opt.setdefault("release_wait_s", 0.0)
    S.arm = arm
    sess = skills.Session(arm, skills.ReplayCamera([np.zeros((4, 4, 3), np.uint8)]), cam_model=None, image_dir=None)
    # the stand-in arm "feels" a surface where the skill expects one (plus arm.surface_offset)
    sess.on_event = lambda name, info: setattr(arm, "expected_touch", info["expected_tip_z"]) if name == "probe" else None
    return skills.take_out(sess, query if query is not None else {"color": "white"}, None, skills.TakeOptions(**opt),
                           modules=(planner or takeout, S))


def kinds_of(arm):
    return [c[0] for c in arm.calls]


def lowest_tip(arm):
    return min(c[5][2] for c in arm.calls if c[0] == "run_path")


# approach, straight down, close, little lift, clear of the rim, carry out + down, open, back to stand-by
PINCH_CALLS = ["run_path", "run_path", "pinch", "run_path", "run_path", "run_path", "set_gripper", "run_path"]


# ----------------------------------------------------------------------------- take-out: control flow
def test_take_pinch_happy_path():
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene())
    saved = copy.deepcopy(arm.cfg["keepouts"])
    r = take(arm, S, mode="pinch")
    assert r["ok"] and r["stage"] == "done" and r["arm_state"] == "ready" and r["attempts"] == 1, r
    assert r["skill"] == "take_out" and r["target"]["id"] == 1 and r["container"]["id"] == 9
    assert kinds_of(arm) == PINCH_CALLS, kinds_of(arm)
    g = r["grasp"]
    assert g["mode"] == "pinch" and g["open_raw"] == 2165 and abs(g["standoff"] - 0.005) < 1e-9, g
    # the jaws opened to 14 mm on the way; the look at them happened above the rim, before anything went lower
    assert arm.calls[0][4] == 2165 and arm.calls[0][5][2] >= RIM + 0.012 - 1e-4, arm.calls[0]
    assert S.n_jaws == 1 and min(S.look_z) >= RIM + 0.012 - 1e-4
    assert r["checks"]["visual"]["seen"] and not r["checks"]["visual"]["applied"]
    # nothing was touched to find a height: the jaws closed 2 mm above the inner bottom the scene implies
    # (table + container floor + the cap's own bottom), fixed pad 5 mm outside the wall
    assert "probe" not in r["checks"] and "probe_down" not in kinds_of(arm)
    pinch_tip = arm.calls[2][1]
    assert abs(pinch_tip[2] - (TABLE_Z + 0.005 + 0.002 + TIP_ABOVE)) < 2e-4, pinch_tip
    u = np.array([math.cos(math.radians(g["jaw_azimuth_deg"])), math.sin(math.radians(g["jaw_azimuth_deg"]))])
    pad = np.array(pinch_tip[:2]) - 0.001 * u
    assert abs((np.array(BOX_C) - pad) @ u - (CAP_R + 0.005)) < 3e-4, pad
    c = r["checks"]["grasp"]
    assert c["pos_raw"] == 2075 and not c["empty"] and c["above_touch_ticks"] == 25 and c["gap_mm"] == 3.0, c
    # lifted a little, jaws checked; then straight up until the object's bottom was 25 mm above the rim
    assert abs(arm.calls[3][5][2] - (pinch_tip[2] + 0.012)) < 2e-4, arm.calls[3]
    top = arm.calls[4][5]
    assert top[2] - H_BELOW >= RIM + 0.025 - 2e-4 and np.allclose(top[:2], pinch_tip[:2], atol=1e-3), top
    lift = r["checks"]["lift"]
    assert lift["held"] and lift["top"]["held"] and lift["camera"]["gone"] is True and lift["load"] == 160, lift
    # put down on a free spot outside the container, object bottom 3 mm above the table; jaws opened there
    place = np.array(r["place"]["xy"])
    outline = takeout._Obstacle(9, "container", RIM, poly=np.array(take_box().footprint))
    assert float(outline.sdist(place)[0]) >= CAP_R + 0.025 - 1e-6, place
    rel = arm.calls[5][5]
    assert abs(rel[2] - (TABLE_Z + H_BELOW + 0.003)) < 2e-4, rel
    assert arm.calls[6] == ("set_gripper", 2165, False)
    assert np.allclose(arm.q[:4], arm.ready_q()[:4]) and arm.torqued
    ver = r["checks"]["verify"]
    assert ver["confirmed"] and ver["left_container"] and ver["on_table"]["dist_mm"] < 25, ver
    assert arm.cfg["keepouts"] == saved, "keep-outs must be restored"
    assert {"perceive", "plan", "approach", "visual", "descend", "grasp", "lift", "transfer", "release",
            "verify"} <= set(r["timings"]) and "probe" not in r["timings"]
    assert r["target"]["confidence"] == 0.9


def test_take_from_parked_unfolds_first():
    arm = FakeArm(parked=True)
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch")
    assert r["ok"] and r["arm_state"] == "ready", r
    assert kinds_of(arm) == ["torque_on", "run_path"] + PINCH_CALLS, kinds_of(arm)
    assert arm.calls[1][4] is None and arm.calls[2][4] == 2165  # the gripper stays shut while unfolding


def test_take_two_sided_for_a_solid_object():
    arm = FakeArm(parked=False)
    sc = scene(cyl(1, "ivory", *BOX_C, d=0.026, h=0.022, hollow=False), take_box())
    r = take(arm, FakeTakeScene(sc), query={"color": "ivory"})
    assert r["ok"] and r["grasp"]["mode"] == "two_sided", r
    # no touch needed (the fingertip stays 4 mm above the floor): straight down, one closing motion
    assert kinds_of(arm) == ["run_path", "run_path", "grasp_fast", "run_path", "run_path", "run_path", "set_gripper",
                             "run_path"], kinds_of(arm)
    assert "probe" not in r["checks"] and r["checks"]["grasp"]["contact"]
    g_tip = arm.calls[2][1]
    assert abs(g_tip[2] - (FLOOR + 0.022 - 0.015)) < 2e-4, g_tip  # 15 mm below its top


def test_take_dry_moves_nothing():
    arm = FakeArm(parked=True)
    q0 = arm.q.copy()
    r = take(arm, FakeTakeScene(cap_scene()), dry=True)
    assert r["ok"] and r["stage"] == "planned" and r["arm_state"] == "parked", r
    assert arm.calls == [] and np.allclose(arm.q, q0) and not arm.torqued
    assert r["plan"]["waypoints"]["carry"] > 5 and "grasp" in r and len(r["place"]["xy"]) == 2


def test_take_object_not_found():
    arm = FakeArm()
    r = take(arm, FakeTakeScene(cap_scene()), query={"color": "green"})
    assert not r["ok"] and r["stage"] == "perceive" and r["reason"].startswith("target not found"), r
    assert "1:white cylinder" in r["reason"] and arm.calls == [] and r["arm_state"] == "parked"


def test_take_object_not_in_a_container():
    arm = FakeArm()
    sc = scene(cyl(1, "white", 0.25, -0.05, z_bottom=TABLE_Z, in_container=None), take_box())
    r = take(arm, FakeTakeScene(sc))
    assert not r["ok"] and r["stage"] == "perceive" and r["reason"].startswith("not in a container"), r
    r = take(arm, FakeTakeScene(sc), query={"id": 1})
    assert not r["ok"] and r["reason"].startswith("not in a container") and "fd pick" in r["reason"], r
    # no container at all
    r = take(arm, FakeTakeScene(scene(cyl(1, "white", 0.25, -0.05, z_bottom=TABLE_Z, in_container=None))))
    assert not r["ok"] and r["reason"].startswith("container not found") and arm.calls == [], r


def test_take_ambiguous_and_uncertain():
    arm = FakeArm()
    two = cap_scene(cyl(2, "white", BOX_C[0], BOX_C[1] + 0.03, d=0.02))
    r = take(arm, FakeTakeScene(two))
    assert not r["ok"] and r["stage"] == "perceive" and "2 objects inside the container match" in r["reason"], r
    assert r["checks"]["failures"][0]["matches"] == [1, 2] and arm.calls == []
    # what lies in a container is always seen roughly (scene.py rates it 0.2 - 0.3): chosen by colour all the same
    vague = cap_scene()
    vague.objects[0].confidence = 0.2
    r = take(arm, FakeTakeScene(vague), dry=True)
    assert r["ok"] and r["target"] == {"id": 1, "kind": "cylinder", "color": "white", "confidence": 0.2}, r
    # ... unless the threshold says otherwise; then only by its id
    r = take(arm, FakeTakeScene(vague), min_confidence=0.25)
    assert not r["ok"] and "uncertain" in r["reason"] and "id 1" in r["reason"] and arm.calls == [], r
    r = take(arm, FakeTakeScene(vague), query={"id": 1}, dry=True, min_confidence=0.25)
    assert r["ok"], r


def test_take_no_feasible_direction():
    arm = FakeArm()
    r = take(arm, FakeTakeScene(hemmed_scene(fourth=True)))
    assert not r["ok"] and r["stage"] == "plan" and arm.calls == [], r
    assert "no jaw direction leaves room" in r["reason"] and "the container wall" in r["reason"], r["reason"]
    assert "ivory cylinder #2" in r["reason"] or "red box #4" in r["reason"] or "blue cylinder #3" in r["reason"]
    assert "fd take --id" in r["next"] and "out first" in r["next"], r["next"]
    assert r["grasp_candidates"] and not any(c["ok"] for c in r["grasp_candidates"])
    assert "candidates" not in r["checks"]["failures"][0]


def test_take_hemmed_in_uses_the_free_side():
    arm = FakeArm(parked=False)
    S = FakeTakeScene(hemmed_scene())
    r = take(arm, S, assume_hollow=True)  # the request called it a cap (`fd do "...キャップ..."`)
    assert r["ok"] and r["grasp"]["mode"] == "pinch", r
    # the fixed jaw went into the slot between the cap and the wall: the jaws open away from the wall (+x)
    assert abs(r["grasp"]["jaw_azimuth_deg"]) <= 31.0, r["grasp"]["jaw_azimuth_deg"]
    assert r["grasp"]["slot"]["bounded_by"] == "the container wall"
    assert any("because the request says so" in n for n in r["grasp"]["notes"]), r["grasp"]["notes"]


def test_take_wall_pinch_needs_to_be_told_that_the_object_is_hollow():
    """Nothing checks a cavity before the moving jaw goes down into it (no touch by default). So the wall of
    a cylinder is pinched only when somebody says that it is hollow and open at the top: the scene, the
    option (--hollow) or the request's own word. Otherwise the take-out stops at the plan stage, nothing
    moves, and the next step names both ways on."""
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(hemmed_scene()))
    assert not r["ok"] and r["stage"] == "plan" and arm.calls == [] and r["arm_state"] == "ready", r
    assert r["reason"].startswith("it can be held by its wall") and "only a guess" in r["reason"], r["reason"]
    assert "across it, no jaw direction leaves room for both jaws" in r["reason"], r["reason"]
    # if it is a cap: the same command for this object with --hollow; if it is solid: what to move first
    assert "if it is a cap or a cup lying open side up, say so: `fd take --id 1 --hollow`" in r["next"], r["next"]
    assert "; if it is solid: " in r["next"] and ("out first" in r["next"] or "make room first" in r["next"]), r["next"]
    unconfirmed = [c for c in r["grasp_candidates"] if "only a guess" in c["reason"]]
    assert unconfirmed and not any(c["ok"] for c in r["grasp_candidates"]), r["grasp_candidates"]
    # said by the option, by the scene, by the request: the same pinch each time
    by_option = take(FakeArm(parked=False), FakeTakeScene(hemmed_scene()), hollow=True)
    told = hemmed_scene()
    told.by_id(1).hollow = True
    by_scene = take(FakeArm(parked=False), FakeTakeScene(told))
    by_request = take(FakeArm(parked=False), FakeTakeScene(hemmed_scene()), assume_hollow=True)
    for res in (by_option, by_scene, by_request):
        assert res["ok"] and res["grasp"]["mode"] == "pinch", res
        assert res["grasp"]["jaw_azimuth_deg"] == by_option["grasp"]["jaw_azimuth_deg"]
        assert np.allclose(res["grasp"]["tip_grasp"], by_option["grasp"]["tip_grasp"], atol=1e-9)
    # the scene saying that it is NOT hollow wins over the request's word; the option wins over the scene
    closed = hemmed_scene()
    closed.by_id(1).hollow = False
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(closed), assume_hollow=True)
    assert not r["ok"] and r["stage"] == "plan" and arm.calls == [], r
    assert "no jaw direction leaves room for both jaws" in r["reason"] and "only a guess" not in r["reason"], r["reason"]
    assert take(FakeArm(parked=False), FakeTakeScene(closed), hollow=True)["ok"]
    # where there is room for both jaws nothing needs to be said: the cap is taken across
    r = take(FakeArm(parked=False), FakeTakeScene(cap_scene()))
    assert r["ok"] and r["grasp"]["mode"] == "two_sided", r


def test_take_jaws_not_measurable_stops_above_the_rim():
    unseen = {"ok": False, "confidence": 0.0, "dxy_base": None, "reason": "its nose is in front of something pale"}
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene(), jaws=[unseen])
    r = take(arm, S, mode="pinch")
    assert not r["ok"] and r["stage"] == "visual" and r["arm_state"] == "ready", r
    assert "could not be measured" in r["reason"] and "not descending" in r["reason"], r["reason"]
    # one more try with the jaws closed, one more from 25 mm higher, then back to the stand-by pose: nothing
    # ever went below the hover point
    assert kinds_of(arm) == ["run_path", "set_gripper", "run_path", "run_path"], kinds_of(arm)
    assert S.n_jaws == 3 and lowest_tip(arm) >= RIM + 0.012 - 1e-4 and np.allclose(arm.q[:4], arm.ready_q()[:4])
    steps = r["checks"]["visual"]["steps"]
    assert steps[1]["retry_closed"] is True and steps[2]["raised_mm"] == 25 and r["checks"]["visual"]["raised"]
    assert abs(S.look_z[2] - (RIM + 0.012 + 0.025)) < 2e-4 and abs(S.look_z[0] - (RIM + 0.012)) < 2e-4, S.look_z
    # a low-confidence reading, a hidden fixed jaw and garbage from the detector count as "not measured" too
    partial = {"ok": True, "confidence": 0.8, "dxy_base": (0.0, -0.0005), "fixed": {"visible": False}}
    for bad in (jaws(1, 1, conf=0.3), partial, {"ok": True, "confidence": 0.9, "dxy_base": (float("nan"), 0.0)},
                {"ok": True, "confidence": None, "dxy_base": "x"}, None):
        arm = FakeArm(parked=False)
        r = take(arm, FakeTakeScene(cap_scene(), jaws=[bad]), mode="pinch")
        assert not r["ok"] and r["stage"] == "visual" and "pinch" not in kinds_of(arm), (bad, r["reason"])
        assert lowest_tip(arm) >= RIM + 0.012 - 1e-4
    assert "fixed jaw is hidden" in take(FakeArm(parked=False), FakeTakeScene(cap_scene(), jaws=[partial]))["reason"]


def test_take_jaws_seen_only_when_closed():
    partial = {"ok": True, "confidence": 0.8, "dxy_base": (0.0, -0.0005), "fixed": {"visible": False}}
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene(), jaws=[partial, jaws(0.5, 0.5)])
    r = take(arm, S, mode="pinch")
    assert r["ok"] and r["checks"]["visual"]["closed"] is True and S.n_jaws == 2, r
    # closed for the second look, opened again (and waited for) before going down
    assert kinds_of(arm)[:4] == ["run_path", "set_gripper", "set_gripper", "run_path"], kinds_of(arm)
    assert arm.calls[1] == ("set_gripper", 1998, True) and arm.calls[2] == ("set_gripper", 2165, True)


def test_take_jaws_seen_only_from_higher_up():
    # over a white object the jaws cannot be told from what is behind them; 25 mm higher they can
    unseen = {"ok": False, "confidence": 0.0, "dxy_base": None, "reason": "its nose is in front of something pale"}
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene(), jaws=[unseen, unseen, jaws(3.0, 1.0), jaws(3.1, 0.9)])
    r = take(arm, S, mode="pinch")
    v = r["checks"]["visual"]
    assert r["ok"] and v["raised"] and v["closed"] and v["applied"] and v["corr_mm"] == [-3.0, -1.0] and S.n_jaws == 4, v
    # the correction was driven and confirmed at the raised point; no look from below the hover point
    assert abs(S.look_z[2] - (RIM + 0.012 + 0.025)) < 2e-4 and abs(S.look_z[3] - S.look_z[2]) < 2e-4, S.look_z
    planned = np.array(r["grasp"]["tip_grasp"][:2])
    i = kinds_of(arm).index("pinch")
    pinch_tip = arm.calls[i][1]
    assert np.allclose(pinch_tip[:2], planned + (-0.003, -0.001), atol=3e-4), (pinch_tip, planned)
    assert abs(pinch_tip[2] - (CAP_INNER + TIP_ABOVE)) < 2e-4
    assert ("set_gripper", 2165, True) in arm.calls[:i], "opened again (and waited for) before the descent"
    # seen from higher up without an offset worth correcting: straight down from there
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene(), jaws=[unseen, unseen, jaws(0.3, 0.2)])
    r = take(arm, S, mode="pinch")
    assert r["ok"] and r["checks"]["visual"]["raised"] and not r["checks"]["visual"]["applied"], r
    assert np.allclose([c for c in arm.calls if c[0] == "pinch"][0][1][:2], r["grasp"]["tip_grasp"][:2], atol=3e-4)
    # no second chance from higher up when that is switched off
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene(), jaws=[unseen, unseen, jaws(0.3, 0.2)])
    r = take(arm, S, mode="pinch", visual_raise=0.0)
    assert not r["ok"] and r["stage"] == "visual" and S.n_jaws == 2, r


def test_take_visual_correction_shifts_the_descent():
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene(), jaws=[jaws(4.0, -3.0), jaws(4.2, -2.8)])
    r = take(arm, S, mode="pinch")
    v = r["checks"]["visual"]
    assert r["ok"] and v["applied"] and v["corr_mm"] == [-4.0, 3.0] and S.n_jaws == 2 and v["residual_mm"] < 0.5, v
    planned = np.array(r["grasp"]["tip_grasp"][:2])
    pinch_tip = [c for c in arm.calls if c[0] == "pinch"][0][1]
    assert np.allclose(pinch_tip[:2], planned + (-0.004, 0.003), atol=3e-4), (pinch_tip, planned)
    # the correction was driven at hover height; from there the jaws went straight down
    assert abs(arm.calls[1][5][2] - arm.calls[0][5][2]) < 1e-4 and min(S.look_z) >= RIM + 0.012 - 1e-4
    assert np.allclose(arm.calls[1][5][:2], planned + (-0.004, 0.003), atol=3e-4), arm.calls[1]
    assert np.allclose(arm.calls[2][5][:2], arm.calls[1][5][:2], atol=3e-4) and arm.calls[3][0] == "pinch"
    # a correction that does not hold at the second look is driven again and looked at a third time
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene(), jaws=[jaws(4.0, -3.0), jaws(6.5, -3.0), jaws(6.6, -3.1)])
    r = take(arm, S, mode="pinch")
    assert r["ok"] and S.n_jaws == 3 and r["checks"]["visual"]["corr_mm"] == [-6.5, 3.0], r["checks"]["visual"]


def test_take_visual_refuses_wild_readings():
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), jaws=[jaws(25.0, 25.0)]), mode="pinch")
    assert not r["ok"] and r["stage"] == "visual" and "35 mm" in r["reason"] and r["arm_state"] == "ready", r
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), jaws=[jaws(4.0, -3.0), jaws(-2.0, 3.0)]), mode="pinch")
    assert not r["ok"] and r["stage"] == "visual" and "disagree" in r["reason"], r
    assert "pinch" not in kinds_of(arm) and "probe_down" not in kinds_of(arm)
    # a reading of the size of 2026-11-11's error with the old calibration (15 mm) is beyond what the look
    # can be trusted to measure (see the next test): it is no longer "corrected", the take-out stops
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), jaws=[jaws(-14.0, 4.6), jaws(-14.2, 4.5)]), mode="pinch")
    assert not r["ok"] and r["stage"] == "visual" and "15 mm" in r["reason"] and "calibration?" in r["reason"], r
    assert "pinch" not in kinds_of(arm) and lowest_tip(arm) >= RIM + 0.012 - 1e-4 and r["arm_state"] == "ready"


def _window_mm(cam, xyz, px):
    """What `px` pixels around a point are on the table: (sideways, along the camera's line of sight) in mm,
    and how many mm along the line of sight the camera reads for 1 mm of height."""
    p0 = np.asarray(xyz, dtype=float)
    pj = np.asarray(cam.project([p0, p0 + [0.001, 0, 0], p0 + [0, 0.001, 0], p0 + [0, 0, 0.001]]), dtype=float)
    J = np.column_stack([(pj[1] - pj[0]) / 0.001, (pj[2] - pj[0]) / 0.001])  # pixels per metre of base x, y
    los = np.asarray(cam.center, dtype=float)[:2] - p0[:2]
    los = los / np.linalg.norm(los)
    side = np.array([-los[1], los[0]])
    span = [1e3 * px / float(np.abs(J @ u).max()) for u in (side, los)]
    return span[0], span[1], float(abs(np.linalg.solve(J, (pj[3] - pj[0]) / 0.001) @ los))


def test_take_look_stops_where_it_cannot_measure():
    """What the look at the jaws can measure is bounded by the detector's search window (36 px around where
    the joint angles put the jaws): about 10 mm sideways and 15-22 mm along the camera's line of sight at
    the hover point, where a height error of the arm also reads as 1.2-1.8 times as much. The accepted
    correction stays inside that; a jaw found at the edge of the window, a reading above the limit and a
    correction that no further look confirms all stop the take-out above the rim."""
    import camera as C
    import scene as real_scene
    o = skills.TakeOptions()
    side, depth, per_dz = _window_mm(C.CameraModel.load(), (BOX_C[0], BOX_C[1], RIM + 0.018), real_scene._JAW_SEARCH_PX)
    assert real_scene._JAW_SEARCH_PX == 36 and 8.5 < side < 12.0 and 14.0 < depth < 23.0 and 1.1 < per_dz < 1.9, \
        (side, depth, per_dz)
    assert o.max_visual_corr == 0.012 and o.max_visual_corr * 1e3 < depth and o.visual_edge * side < o.max_visual_corr * 1e3
    stopped = ("visual", "ready")

    def look(readings, **opt):
        arm = FakeArm(parked=False)
        S = FakeTakeScene(cap_scene(), jaws=readings)
        r = take(arm, S, mode="pinch", **opt)
        if not r["ok"]:  # stopped above the rim: nothing went lower, nothing was closed
            assert "pinch" not in kinds_of(arm) and lowest_tip(arm) >= RIM + 0.012 - 1e-4, kinds_of(arm)
            assert np.allclose(arm.q[:4], arm.ready_q()[:4]) and "not descending" in r["reason"], r["reason"]
        return r, S

    # found 33.5 px from where the joint angles put it, in a window of 36: the real offset may lie outside it
    edge = {**jaws(9.0, 1.0), "fixed": {"visible": True, "shift_px": (33.5, 4.0)}}
    r, S = look([edge])
    assert (r["stage"], r["arm_state"]) == stopped and "edge of what the look can measure" in r["reason"], r
    assert S.n_jaws == 1 and r["checks"]["visual"]["steps"][0]["window"] == 0.93, r["checks"]["visual"]
    # the same reading well inside the window (22 px) is a measurement: corrected and confirmed
    inside = {**jaws(6.0, 1.0), "fixed": {"visible": True, "shift_px": (22.0, 3.0)}}
    r, S = look([inside, inside])
    assert r["ok"] and r["checks"]["visual"]["corr_mm"] == [-6.0, -1.0] and r["checks"]["visual"]["steps"][0]["window"] == 0.61
    # above the limit (12 mm): "calibration?", whatever the window says
    r, _S = look([jaws(-9.0, 9.0)])
    assert (r["stage"], r["arm_state"]) == stopped and "13 mm" in r["reason"] and "more than 12 mm" in r["reason"], r
    # the size the look read on 2026-11-11 with the corrected calibration (3-5 mm) is corrected as before
    r, _S = look([jaws(-4.4, 1.9), jaws(-4.5, 1.8)])
    assert r["ok"] and r["checks"]["visual"]["corr_mm"] == [4.4, -1.9], r["checks"]["visual"]
    # three looks that each ask for a further correction (2.5 mm apart: less than "disagree", more than
    # "confirmed"): the last correction was never seen to hold, so it is not driven down
    r, S = look([jaws(4.0, 0.0), jaws(6.5, 0.0), jaws(9.0, 0.0)])
    assert (r["stage"], r["arm_state"]) == stopped and "not confirmed" in r["reason"] and S.n_jaws == 3, r
    assert "2.5 mm" in r["reason"], r["reason"]
    # one look only: a correction can never be confirmed, so only a reading within the tolerance descends
    r, S = look([jaws(5.0, 0.0)], visual_iters=1)
    assert (r["stage"], r["arm_state"]) == stopped and "not confirmed" in r["reason"] and S.n_jaws == 1, r
    r, S = look([jaws(1.0, 0.5)], visual_iters=1)
    assert r["ok"] and not r["checks"]["visual"]["applied"], r["checks"]["visual"]


def test_take_empty_pinch_retries_then_reports():
    arm = FakeArm(parked=False)
    arm.pinch_raw = None
    S = FakeTakeScene(cap_scene())
    r = take(arm, S, mode="pinch", retries=1)
    assert not r["ok"] and r["stage"] == "grasp" and r["attempts"] == 2 and S.n_perceive == 2, r
    assert "closed on nothing" in r["reason"] and r["arm_state"] == "ready" and np.allclose(arm.q[:4], arm.ready_q()[:4])
    assert r["checks"]["grasp"]["empty"] and len(r["checks"]["failures"]) == 2
    assert not any(c[0] == "set_gripper" for c in arm.calls), "nothing to release"
    # second time lucky
    arm = FakeArm(parked=False)
    arm.pinch_raw = None
    S = FakeTakeScene(cap_scene())
    real = arm.pinch

    def pinch(**kw):
        if S.n_perceive >= 2:
            arm.pinch_raw = 2075
        return real(**kw)

    arm.pinch = pinch
    r = take(arm, S, mode="pinch", retries=1)
    assert r["ok"] and r["attempts"] == 2 and r["arm_state"] == "ready", r


def test_take_object_slips_while_lifting():
    # at the little lift: the jaws close on to the holding goal (2026-11-11: 2081 -> 2058, load 172 -> 80)
    arm = FakeArm(parked=False)
    arm.slip_at = 1
    S = FakeTakeScene(cap_scene())
    r = take(arm, S, mode="pinch", retries=0)
    assert not r["ok"] and r["stage"] == "lift" and "slipped out of the jaws" in r["reason"], r
    assert r["arm_state"] == "ready" and r["attempts"] == 1 and not r["checks"]["lift"]["held"]
    assert r["checks"]["lift"]["raw"] == 2038 and r["checks"]["lift"]["raw_min"] == 2057
    assert not any(c[0] == "set_gripper" for c in arm.calls)
    # ... one more attempt by default
    arm = FakeArm(parked=False)
    arm.slip_at = 1
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch")
    assert not r["ok"] and r["stage"] == "lift" and r["attempts"] == 2, r
    # on the way up to the rim
    arm = FakeArm(parked=False)
    arm.slip_at = 2
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", retries=0)
    assert not r["ok"] and r["stage"] == "lift" and "clear of the rim" in r["reason"] and r["arm_state"] == "ready", r
    # the load alone gives it away too (jaws jammed open, nothing between them)
    held, info = skills._held({"pos_raw": 2075, "hold_goal_raw": 2038}, {"raw": 2075, "load": 80}, load_ref=160)
    assert not held and info["load_min"] == 112
    # jaws resting on each other after a slip read about 100 at a position the raw test alone would accept
    assert not skills._held({"pos_raw": 2075, "hold_goal_raw": 2038}, {"raw": 2058, "load": 100}, load_ref=160)[0]
    assert skills._held({"pos_raw": 2075, "hold_goal_raw": 2038}, {"raw": 2075, "load": 160}, load_ref=160)[0]
    assert not skills._held({"pos_raw": 2100, "hold_goal_raw": 2060}, {"raw": 2071, "load": 56}, load_ref=172)[0]


def test_take_object_lost_while_carrying():
    arm = FakeArm(parked=False)
    arm.slip_at = 3
    S = FakeTakeScene(cap_scene())
    r = take(arm, S, mode="pinch")
    assert not r["ok"] and r["stage"] == "transfer" and "lost on the way" in r["reason"], r
    assert r["arm_state"] == "ready" and r["attempts"] == 1 and not any(c[0] == "set_gripper" for c in arm.calls)
    # the camera found where it lies now
    where = r["checks"]["failures"][0]["camera"]["on_table"]
    assert where is not None and "lies on the table at" in r["reason"], r["checks"]["failures"][0]


def test_take_place_spot_not_free():
    arm = FakeArm()
    r = take(arm, FakeTakeScene(cap_scene()), to=(0.225, 0.02))  # 15 mm from the container
    assert not r["ok"] and r["stage"] == "plan" and "cannot be used" in r["reason"] and "black container #9" in r["reason"], r
    assert arm.calls == [] and "--to" in r["next"]
    # a spot the arm cannot reach straight down: said by the path planner, nothing moves
    r = take(arm, FakeTakeScene(cap_scene()), to=(0.40, 0.0))
    assert not r["ok"] and r["stage"] == "plan" and "(0.400, 0.000) cannot be reached straight down" in r["reason"], r
    assert arm.calls == []
    # a free one is used as given
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene()), to=(0.22, -0.07), mode="pinch")
    assert r["ok"] and r["place"]["xy"] == [0.22, -0.07], r
    assert math.hypot(arm.released_at[0] - 0.22, arm.released_at[1] + 0.07) < CAP_R + 1e-3  # fingertip beside the centre
    # the table is full, as far as the search goes (20 cm around the container)
    crowd = [box(20 + k, "red", x, y, size=(0.05, 0.05, 0.02), in_container=None)
             for k, (x, y) in enumerate((BOX_C[0] + i * 0.05, BOX_C[1] + j * 0.05) for i in range(-5, 6) for j in range(-5, 6)
                                        if not (abs(i * 0.05) < 0.09 and abs(j * 0.05) < 0.085))]
    r = take(FakeArm(), FakeTakeScene(cap_scene(*crowd)))
    assert not r["ok"] and r["stage"] == "plan" and "no free spot" in r["reason"], r
    assert "too close to something" in r["reason"] and "clear some space" in r["next"], (r["reason"], r["next"])


class _SpyPlanner:
    """takeout, with the place-spot test the skill hands to choose_place kept for inspection."""

    ok_xy = None

    def __getattr__(self, name):
        return getattr(takeout, name)

    def choose_place(self, *a, **k):
        self.ok_xy = k.get("ok_xy")
        return takeout.choose_place(*a, **k)


def test_take_place_spot_is_tested_for_reach():
    spy = _SpyPlanner()
    arm = FakeArm()
    r = take(arm, FakeTakeScene(cap_scene()), planner=spy, dry=True)
    assert r["ok"] and spy.ok_xy is not None, r
    x, y = r["place"]["xy"]
    assert spy.ok_xy((x, y)) == ""
    for far in ((0.40, 0.0), (0.0, 0.42), (0.30, -0.30)):
        assert "cannot point straight down" in spy.ok_xy(far), far
    # the chosen spot can be reached straight down at the table and above the rim, 2 cm nearer and further too
    d = math.hypot(x - K._X_PAN, y)
    for m in (0.0, 0.02, -0.02):
        for z in (TABLE_Z + 0.005, RIM + 0.015):
            arm.solve((x + m * (x - K._X_PAN) / d, y + m * y / d, z), 90.0, 0.0)
    # it lies outside the container, next to it (no table coordinate is built into the search)
    outline = takeout._Obstacle(9, "container", RIM, poly=np.array(take_box().footprint))
    assert CAP_R + 0.025 - 1e-6 <= float(outline.sdist((x, y))[0]) < 0.09, (x, y)
    # a spot given by hand is not pre-judged for reach: the path planner has the last word on it (it is
    # still asked whether it lies where the arm parks: test_take_place_spot_keeps_clear_of_where_the_arm_parks)
    spy = _SpyPlanner()
    take(FakeArm(), FakeTakeScene(cap_scene()), planner=spy, dry=True, to=(0.22, -0.07))
    assert spy.ok_xy is not None and all(spy.ok_xy(far) == "" for far in ((0.40, 0.0), (0.0, 0.42), (0.30, -0.30)))


def test_place_view_test():
    """Which place spots the camera will see with the arm back at the stand-by pose (the tree's own camera
    model and the real arm mask; the test points are built from the camera and the arm, not from numbers)."""
    import camera as C
    import scene as real_scene
    cfg, cam = load_config(), C.CameraModel.load()
    tz = float(cfg["table_z"])
    with _sim_arm() as arm:
        sess = skills.Session(arm, skills.ReplayCamera([np.zeros((4, 4, 3), np.uint8)]), cam, image_dir=None)
        cap = cyl(1, "white", 0.0, 0.0, z_bottom=tz, in_container=None)
        ok = skills._place_view(sess, real_scene, (1080, 1920, 3), cap, tz)
        arm.cfg["scene"] = {"workspace": [-2.0, 2.0, -2.0, 2.0]}  # (to try the image tests on their own)
        ok_wide = skills._place_view(sess, real_scene, (1080, 1920, 3), cap, tz)
        del arm.cfg["scene"]
        tip = K.fk(arm.ready_q(), arm.tip_tool())[:3, 3]
    c = np.asarray(cam.center, dtype=float)
    # the table point exactly behind the fingertip, as the camera sees it: hidden by the arm (it lies behind
    # the robot, outside the watched area: asked without that limit)
    behind = c + (c[2] - tz) / (c[2] - tip[2]) * (tip - c)
    assert ok_wide(behind[:2]) == "hidden from the camera by the arm at its stand-by pose", ok_wide(behind[:2])
    assert ok_wide((tip[0], 1.5)) == "outside the camera image"
    # straight below the fingertip, 6 cm towards the camera: in full view
    assert ok((tip[0] + 0.06, tip[1])) == "" and ok_wide((tip[0] + 0.06, tip[1])) == "", ok((tip[0] + 0.06, tip[1]))
    # right in front of the camera, and far to the side: not in the area the camera watches
    assert ok((c[0] - 0.05, c[1])) == "outside the area the camera watches"
    assert ok((tip[0], 0.5)) == "outside the area the camera watches"
    # no camera model, or a stand-in for perception: no test (the spot is then taken on trust)
    sess.cam = None
    assert skills._place_view(sess, real_scene, (1080, 1920, 3), cap, tz) is None
    sess.cam = cam
    assert skills._place_view(sess, FakeTakeScene(cap_scene()), (1080, 1920, 3), cap, tz) is None


def test_take_verification_negative():
    # put down, but the camera does not find it on the table
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), lands=False), mode="pinch")
    assert not r["ok"] and r["stage"] == "verify" and r["arm_state"] == "ready" and r["attempts"] == 1, r
    assert "left the container" in r["reason"] and r["checks"]["verify"]["on_table"] is None
    # nothing changed in the picture (looked at only at the end)
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), stays=True), mode="pinch", lift_look=False)
    assert not r["ok"] and r["stage"] == "verify" and "still seems to be in the container" in r["reason"], r
    # verify=False skips the camera checks of the result
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), stays=True), mode="pinch", verify=False)
    assert r["ok"] and "verify" not in r["checks"] and "camera" not in r["checks"]["lift"], r
    # the arm hides the place during the lift: decided at the end, from the stand-by pose
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), hidden=True), mode="pinch")
    assert r["ok"] and r["checks"]["lift"]["camera"]["gone"] is None and r["checks"]["verify"]["confirmed"], r


def test_take_camera_contradicts_the_jaws():
    # the jaws hold something, the camera still sees the object where it was: stop and hold
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), stays=True), mode="pinch")
    assert not r["ok"] and r["stage"] == "lift" and r["arm_state"] == "holding_object", r
    assert "still sees the object" in r["reason"] and arm.holding and r["attempts"] == 1
    assert not any(c[0] == "set_gripper" for c in arm.calls)
    # ... unless something of the same colour lies right next to it (the camera may be looking at that)
    arm = FakeArm(parked=False)
    twin = cyl(2, "ivory", BOX_C[0], BOX_C[1] + 2 * CAP_R + 0.002, d=0.026, h=0.02)
    r = take(arm, FakeTakeScene(cap_scene(twin), stays=True), mode="pinch", any_match=True)
    assert r["stage"] != "lift" and any(c[0] == "set_gripper" for c in arm.calls), r


def test_take_never_touches_to_find_a_height_by_default():
    # the heights come from geometry (table + container floor + the object's own bottom): nothing is probed,
    # even where a touch would read differently, and the jaws close 2 mm above the expected inner bottom
    assert skills.TakeOptions().probe == "off" and skills.TakeOptions().probe_may_lower is False
    arm = FakeArm(parked=False)
    arm.surface_offset = -0.008
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch")
    assert r["ok"] and "probe_down" not in kinds_of(arm) and "probe" not in r["checks"] and r["plan"]["probe"] is False, r
    pinch_tip = [c for c in arm.calls if c[0] == "pinch"][0][1]
    assert abs(pinch_tip[2] - (CAP_INNER + TIP_ABOVE)) < 2e-4, pinch_tip


def _expected_touch():
    """Fingertip height at which the open jaws should land in the cap (from a dry plan)."""
    g = take(FakeArm(), FakeTakeScene(cap_scene()), mode="pinch", dry=True)["grasp"]
    return g["z_surface"] + g["contact_offset"]


def test_take_touch_check_never_sets_the_grasp_height():
    # probe="on": a slow touch before the last stretch, as a CHECK. It goes 2 mm past where the jaws should
    # land, not further (more would push what it touches), and whatever it reads the grasp height stays the
    # one from the scene: Arm.probe_down reports its contact 3-18 mm deeper than the first touch
    for offset, contact in ((0.0, None), (-0.001, None), (-0.008, False)):
        arm = FakeArm(parked=False)
        arm.surface_offset = offset
        r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", probe="on")
        p = r["checks"]["probe"]
        assert r["ok"] and p["mode"] == "check" and len(p["tries"]) == 1 and p["tries"][0]["contact"] is contact, (offset, p)
        probes = [c for c in arm.calls if c[0] == "probe_down"]
        assert len(probes) == 1 and abs((probes[0][2][2] - probes[0][1]) - (p["expected_tip_z"] - 0.002)) < 2e-4, probes
        pinch_tip = [c for c in arm.calls if c[0] == "pinch"][0][1]
        assert abs(pinch_tip[2] - (CAP_INNER + TIP_ABOVE)) < 2e-4 and abs(p["tip_grasp_z"] - pinch_tip[2]) < 2e-4, pinch_tip
        assert "from the scene" in p["note"] and "contact_tip_z" not in p
    # a reported contact, at the expected height or (reading deep) below it, changes nothing either
    expected = _expected_touch()
    for z in (expected + 0.002, expected - 0.004):
        arm = FakeArm(parked=False)
        arm.probe_script = [{"contact": True, "contact_tip_z": z, "lowest_tip_z": z - 0.001, "reason": None,
                             "free_air_gap": 0.0002}]
        r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", probe="on")
        p = r["checks"]["probe"]
        assert r["ok"] and abs(p["contact_tip_z"] - z) < 1e-4 and "not used" in p["note"], p
        assert abs([c for c in arm.calls if c[0] == "pinch"][0][1][2] - (CAP_INNER + TIP_ABOVE)) < 2e-4


def test_take_touch_check_stops_when_something_is_higher_than_the_scene_says():
    # the jaws would land on the cap's rim (it is closed on top, or it does not lie where it was seen): the
    # check meets it before its run-in is over
    arm = FakeArm(parked=False)
    arm.surface_offset = +0.014
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", probe="on")
    assert not r["ok"] and r["stage"] == "probe" and "right where it started" in r["reason"] and "rim" in r["reason"], r
    assert r["arm_state"] == "ready" and "pinch" not in kinds_of(arm) and r["attempts"] == 1
    # a surface 5 mm higher than expected: felt; the jaws do not go on to press into it
    arm = FakeArm(parked=False)
    arm.surface_offset = +0.005
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", probe="on")
    assert not r["ok"] and r["stage"] == "probe" and "5 mm above the expected height" in r["reason"], r
    assert r["arm_state"] == "ready" and "pinch" not in kinds_of(arm)
    # the fingertip slid sideways while it pressed: the object may have been pushed
    arm = FakeArm(parked=False)
    arm.probe_slide = 0.009
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", probe="on")
    assert not r["ok"] and r["stage"] == "probe" and "slid 9 mm" in r["reason"], r


# the touch as a MEASUREMENT: kept for a probe that reports the first touch (Arm.probe_down does not, today)
MEASURE = {"probe": "on", "probe_may_lower": True}


def test_take_probe_measurement_follows_the_touch():
    # the scene's heights are 8 mm too high here: the grasp follows what was touched
    arm = FakeArm(parked=False)
    arm.surface_offset = -0.008
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", **MEASURE)
    p = r["checks"]["probe"]
    assert r["ok"] and p["mode"] == "measure" and abs(p["deviation_mm"] + 8.0) < 0.3 and len(p["tries"]) == 1, p
    pinch_tip = [c for c in arm.calls if c[0] == "pinch"][0][1]
    assert abs(pinch_tip[2] - (CAP_INNER - 0.008 + TIP_ABOVE)) < 2e-4, pinch_tip
    # put down by the table height of the scene all the same (object bottom 3 mm above it)
    assert abs(arm.released_at[2] - (TABLE_Z + H_BELOW + 0.003)) < 2e-4


def test_take_probe_measurement_repeats_once_with_a_longer_drop():
    # first descent: the range ends 2 mm above the surface ("no contact"); the repeat goes deeper and finds it
    arm = FakeArm(parked=False)
    arm.surface_offset = -0.024
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", probe_max_dev=0.030, **MEASURE)
    p = r["checks"]["probe"]
    assert r["ok"] and len(p["tries"]) == 2 and p["tries"][0]["contact"] is False and p["tries"][1]["contact"] is True, p
    probes = [c for c in arm.calls if c[0] == "probe_down"]
    assert len(probes) == 2 and probes[1][2][2] - probes[1][1] < probes[0][2][2] - probes[0][1] - 0.009, probes
    assert probes[1][2][2] >= p["tries"][0]["lowest_tip_z"] + 0.0126 - 1e-4, "the repeat needs its run-in"
    # "inconclusive" (the tip was lagging when the range ended) is repeated in the same way
    arm = FakeArm(parked=False)
    arm.surface_offset = -0.019
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", probe_max_dev=0.025, **MEASURE)
    p = r["checks"]["probe"]
    assert r["ok"] and p["tries"][0]["contact"] is None and "inconclusive" in p["tries"][0]["reason"], p
    # two descents without a clear contact: stop, say so, back to the stand-by pose with nothing closed
    arm = FakeArm(parked=False)
    arm.probe_script = [{"contact": None, "contact_tip_z": None, "lowest_tip_z": TABLE_Z + 0.004,
                         "reason": "inconclusive: lagging"},
                        {"contact": False, "contact_tip_z": None, "lowest_tip_z": TABLE_Z - 0.010, "reason": None}]
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", **MEASURE)
    assert not r["ok"] and r["stage"] == "probe" and "no clear contact in two touch descents" in r["reason"], r
    assert "inconclusive: lagging" in r["reason"] and r["arm_state"] == "ready" and "pinch" not in kinds_of(arm)
    assert r["attempts"] == 1, "a failed measurement is not retried blindly"


def test_take_probe_measurement_implausible_results():
    # the jaws landed 14 mm high: on the rim of the cap (or it is closed on top), not on its inner bottom
    arm = FakeArm(parked=False)
    arm.probe_script = [{"contact": True, "contact_tip_z": CAP_INNER + 0.014, "lowest_tip_z": CAP_INNER + 0.013,
                         "reason": None}]
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", **MEASURE)
    assert not r["ok"] and r["stage"] == "probe" and "14 mm above" in r["reason"] and "rim" in r["reason"], r
    assert r["arm_state"] == "ready" and "pinch" not in kinds_of(arm)
    # far below what the scene says: not used either
    arm = FakeArm(parked=False)
    arm.probe_script = [{"contact": True, "contact_tip_z": CAP_INNER - 0.021, "lowest_tip_z": CAP_INNER - 0.022,
                         "reason": None}]
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", **MEASURE)
    assert not r["ok"] and r["stage"] == "probe" and "below" in r["reason"] and "calibration" in r["reason"], r
    # the fingertip slid sideways while pressing: the object may have moved
    arm = FakeArm(parked=False)
    arm.probe_slide = 0.009
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", **MEASURE)
    assert not r["ok"] and r["stage"] == "probe" and "slid 9 mm" in r["reason"], r
    # a small slip is put right before the last stretch down
    arm = FakeArm(parked=False)
    arm.probe_slide = 0.003
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", **MEASURE)
    planned = np.array(r["grasp"]["tip_grasp"][:2])
    assert r["ok"] and np.allclose([c for c in arm.calls if c[0] == "pinch"][0][1][:2], planned, atol=3e-4), r


def test_take_probe_option():
    # "auto": a touch check only where the planned grasp leaves the jaws less than 3 mm (the wall pinch) ...
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", probe="auto")
    assert r["ok"] and kinds_of(arm).count("probe_down") == 1 and r["plan"]["probe"] is True, r
    # ... not for a grasp across a solid object, which stays 4 mm and more above the floor
    sc = scene(cyl(1, "ivory", *BOX_C, d=0.026, h=0.022, hollow=False), take_box())
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(sc), query={"color": "ivory"}, probe="auto")
    assert r["ok"] and "probe_down" not in kinds_of(arm) and "probe" not in r["checks"], r
    # "on": there too (in the free slot beside the object); the grasp height is the planned one
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(sc), query={"color": "ivory"}, probe="on")
    assert r["ok"] and kinds_of(arm).count("probe_down") == 1 and r["grasp"]["mode"] == "two_sided", r
    assert abs(arm.calls[kinds_of(arm).index("grasp_fast")][1][2] - (FLOOR + 0.007)) < 2e-4


def test_take_table_met_early_counts_as_put_down():
    arm = FakeArm(parked=False)
    real = arm.run_path

    def run_path(waypoints, speed, **kw):
        if arm.holding and kw.get("settle") == 0.6:  # the carry: the table is 5 mm higher than calibrated
            end = arm.tip(waypoints[-1])
            arm.q = arm.solve((end[0], end[1], end[2] + 0.005), 90.0, math.degrees(float(waypoints[-1][4])))
            raise MotionAborted("target not reached: something is in the way? Push released", info={"reason": "target_not_reached"})
        return real(waypoints, speed, **kw)

    arm.run_path = run_path
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch")
    assert r["ok"] and r["checks"]["place_contact"]["above_planned_mm"] == 5.0, r
    assert any(c[0] == "set_gripper" for c in arm.calls) and r["arm_state"] == "ready"
    # a fault anywhere else on the way is a fault: stop where it is, holding
    arm = FakeArm(parked=False)
    arm.fail_run_path_at = 5  # the carry (approach, descent, little lift, rise, carry)
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch")
    assert not r["ok"] and "MotionAborted" in r["reason"] and r["stage"] == "transfer", r
    assert r["arm_state"] == "holding_object" and arm.torqued and not any(c[0] == "set_gripper" for c in arm.calls)


def test_take_hot_servo_and_bad_tier():
    arm = FakeArm()
    arm.temp = 61
    r = take(arm, FakeTakeScene(cap_scene()))
    assert not r["ok"] and r["stage"] == "approach" and "cool" in r["reason"] and arm.calls == [], r

    def speeds(tier=None, allow_unvalidated=False):
        raise SafetyError("speed tier 'fast' has not been validated on the hardware")

    arm = FakeArm()
    arm.speeds = speeds
    r = take(arm, FakeTakeScene(cap_scene()), tier="fast")
    assert not r["ok"] and r["stage"] == "start" and "validated" in r["reason"] and arm.calls == [], r


def test_take_starts_with_a_clear_view():
    # after a pick the arm waits over the container: it goes to the stand-by pose before it looks
    arm = FakeArm(parked=False)
    arm.q = arm.solve((BOX_C[0], BOX_C[1], TABLE_Z + 0.095), 80.0, 0.0)
    assert skills._arm_state(arm, arm.q) == "standby"
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch")
    assert r["ok"] and np.allclose(arm.calls[0][5], arm.tip(arm.ready_q()).round(4), atol=1e-3), arm.calls[0]
    # ... but never in a dry run
    arm = FakeArm(parked=False)
    arm.q = arm.solve((BOX_C[0], BOX_C[1], TABLE_Z + 0.095), 80.0, 0.0)
    r = take(arm, FakeTakeScene(cap_scene()), dry=True)
    assert r["ok"] and arm.calls == [], r


def test_take_folds_the_arm_back_when_it_hides_the_container():
    # 2026-11-11: with the arm at the stand-by pose one corner of the box was hidden and no container was reported
    behind = scene(box(7, "red", 0.25, -0.1, in_container=None))
    behind.warnings.append("a dark box-like region near x=0.22 y=+0.09 has 1 corner(s) hidden by the robot arm and is "
                           "not reported as a container: move the arm away and look again")
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene(), behind_arm=behind)
    r = take(arm, S, mode="pinch")
    assert r["ok"] and S.n_perceive == 3 and "folded back" in r["checks"]["second_look"], r
    assert kinds_of(arm) == ["run_path", "run_path"] + PINCH_CALLS, kinds_of(arm)
    assert np.allclose(arm.tip(arm.rest_q()).round(4), arm.calls[0][5], atol=1e-3), arm.calls[0]  # folded back first
    assert arm.torqued and not any(c[0] == "torque_off" for c in arm.calls)
    # nothing to take after the clear look either: back to the stand-by pose, and said so
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene(), behind_arm=behind)
    r = take(arm, S, query={"color": "green"})
    assert not r["ok"] and r["stage"] == "perceive" and r["reason"].startswith("target not found"), r
    assert r["arm_state"] == "ready" and np.allclose(arm.q[:4], arm.ready_q()[:4]) and len(arm.calls) == 2
    # a dry run never moves: it says what to do instead
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), behind_arm=behind), dry=True)
    assert not r["ok"] and arm.calls == [] and "hides part of the scene" in r["next"], r
    # no fold for a plain "not there" when the camera saw the container unobstructed
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene()), query={"color": "green"})
    assert not r["ok"] and arm.calls == [] and r["arm_state"] == "ready", r


def test_take_verification_inferred_when_the_place_is_hidden():
    # neither the look during the lift nor the final one can see the object's old place (the arm is in the
    # way), but the jaws held it all the way and it stands on the table: accepted, and said so
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), hidden="always"), mode="pinch")
    v = r["checks"]["verify"]
    assert r["ok"] and v["confirmed"] and not v["left_container"] and v["how"].startswith("inferred"), v
    # ... not when the object is not on the table either
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), hidden="always", lands=False), mode="pinch")
    assert not r["ok"] and r["stage"] == "verify", r


def test_take_internal_error_is_reported_not_raised():
    arm = FakeArm(parked=False)
    S = FakeTakeScene(cap_scene())

    def boom(*a, **k):
        raise ValueError("bad array")

    S.objects_in_container = boom
    r = take(arm, S)
    assert not r["ok"] and "internal error: ValueError" in r["reason"] and arm.calls == [], r
    assert arm.cfg["keepouts"] == FakeArm().cfg["keepouts"]


def test_select_inside():
    sc = cap_scene(cyl(2, "ivory", 0.21, 0.06, d=0.026), box(3, "red", 0.25, -0.1, in_container=None))
    ids = lambda q, **k: [o.id for o in skills.select_inside(sc, q, **k)[0]]  # noqa: E731
    assert ids({}) == [1, 2] and ids({"color": "white"}) == [1] and ids({"kind": "cylinder", "largest": True}) == [1]
    assert ids({"id": 2}) == [2] and ids({}, container_query={"color": "black"}) == [1, 2]
    assert ids({"nearest_to": (0.21, 0.06)}) == [2]
    assert skills.select_inside(sc, {"color": "white"})[1].id == 9
    for bad, start in (({"color": "red"}, "not in a container"), ({"id": 3}, "not in a container"),
                       ({"color": "green"}, "target not found"), ({"id": 77}, "target not found"),
                       ({"id": 9}, "target not found")):
        try:
            skills.select_inside(sc, bad)
            raise AssertionError(f"{bad} should not match")
        except LookupError as e:
            assert str(e).startswith(start), (bad, str(e))
    try:
        skills.select_inside(sc, {"color": "white"}, container_query={"color": "blue"})
        raise AssertionError("no blue container")
    except LookupError as e:
        assert str(e).startswith("container not found"), str(e)


def test_take_options_from_config():
    o = skills.take_options_from_config({"take": {"rim_margin": 0.02, "probe": "off", "pinch_standoff": 0.004}}, tier=None,
                                        retries=3)
    assert o.rim_margin == 0.02 and o.probe == "off" and o.retries == 3 and o.visual == "on"
    assert skills.take_options_from_config({}, probe="on").probe == "on"
    # the request's word for a hollow thing can be made the rule (every cylinder nobody describes counts as a cup)
    assert skills.take_options_from_config({}).assume_hollow is False
    assert skills.take_options_from_config({"take": {"assume_hollow": True}}).assume_hollow is True


def test_take_command_line_says_what_is_hollow():
    """`fd do`: a sentence that calls the thing a cap, a cup or a lid releases the wall pinch for that
    request (assume_hollow); any other sentence does not. `fd take`: --hollow / --solid state it outright,
    and the touch stays off unless asked for. (No arm motion, no camera: the skill itself is replaced.)"""
    import fd
    seen, real_cmd = [], fd.cmd_take

    def fake_cmd_take(arm, a):
        seen.append({k: getattr(a, k, None) for k in ("color", "kind", "from_color", "assume_hollow", "hollow", "solid",
                                                      "probe")})
        return {"ok": True, "skill": "take_out", "stage": "done", "target": {"color": a.color, "kind": a.kind},
                "timings": {"total": 1.0}}

    fd.cmd_take = fake_cmd_take
    try:
        for text in ("箱の中の白いキャップを取り出して", "take the white cup out of the black box",
                     "箱から赤いブロックを出して", "箱の中の白い円柱を取り出して"):
            res = fd.cmd_do(None, fd.build_parser().parse_args(["--sim", "do", text]))
            assert res["ok"] and res["request"]["action"] == "take_out" and "取り出して机に置きました" in res["say"], res
    finally:
        fd.cmd_take = real_cmd
    assert [s["assume_hollow"] for s in seen] == [True, True, False, False], seen
    assert seen[1]["from_color"] == "black" and seen[0]["color"] == "white" and seen[2]["kind"] == "box", seen
    assert all(s["hollow"] is False and s["solid"] is False and s["probe"] is None for s in seen), seen
    # fd take: the options as the skill receives them
    got, real_take, real_run_dir = [], skills.take_out, fd.run_dir
    skills.take_out = lambda sess, query, con, opts: got.append((query, con, opts)) or {
        "ok": True, "skill": "take_out", "stage": "planned", "reason": "", "arm_state": "parked"}
    fd.run_dir = lambda: None  # (no run directory for this test)
    try:
        with _sim_arm() as arm:
            default_probe = skills.take_options_from_config(arm.cfg).probe
            for argv in (["take", "--color", "white", "--dry"], ["take", "--color", "white", "--hollow", "--dry"],
                         ["take", "--id", "3", "--solid", "--from-color", "black", "--dry"],
                         ["take", "--color", "white", "--probe", "on", "--dry"]):
                a = fd.build_parser().parse_args(["--sim"] + argv)
                a._camera = skills.ReplayCamera([np.zeros((4, 4, 3), np.uint8)])
                assert fd.cmd_take(arm, a)["ok"]
            a = fd.build_parser().parse_args(["--sim", "take", "--color", "white", "--dry"])
            a._camera, a.assume_hollow = skills.ReplayCamera([np.zeros((4, 4, 3), np.uint8)]), True  # as `fd do` sets it
            assert fd.cmd_take(arm, a)["ok"]
    finally:
        skills.take_out, fd.run_dir = real_take, real_run_dir
    opts = [g[2] for g in got]
    assert [o.hollow for o in opts] == [None, True, False, None, None], [o.hollow for o in opts]
    assert [o.assume_hollow for o in opts] == [False, False, False, False, True]
    assert [o.probe for o in opts] == [default_probe] * 3 + ["on", default_probe] and skills.TakeOptions().probe == "off"
    assert got[2][0] == {"id": 3} and got[2][1] == {"color": "black"} and all(o.dry for o in opts), got[2][:2]
    import contextlib
    import io
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            fd.build_parser().parse_args(["take", "--color", "white", "--hollow", "--solid"])
        raise AssertionError("--hollow and --solid exclude each other")
    except SystemExit:
        pass


# ----------------------------------------------------------------------------- take-out: findings of the 2026-11-11 review
class _OtherTaken(FakeTakeScene):
    """What the camera sees when the jaws carried out the object `taken_id`, which need not be the target:
    every other object stays where it lies; the one taken is gone from the container while it is held and
    stands on the table where the jaws let go of it (`table_color`: the colour name the camera gives it
    there, default its own)."""

    def __init__(self, before, taken_id, table_color=None, **kw):
        super().__init__(before, **kw)
        self.taken_id, self.table_color = taken_id, table_color

    def _out(self):
        return self.arm.holding or self.arm.released_at is not None

    def perceive(self, img, cam, cfg, arm_q=None, jaw_q_deg=None, debug=None):
        self.n_perceive += 1
        if not self._out():
            return self.before
        taken = self.before.by_id(self.taken_id)
        objs = [o for o in self.before.objects if o.id != self.taken_id]
        p = self.arm.released_at
        if p is not None:
            objs.append(cyl(50, self.table_color or taken.color, float(p[0]), float(p[1]), d=taken.size[0], h=taken.size[2],
                            z_bottom=self.before.table_z, in_container=None))
        return T.Scene(objects=objs, table_z=self.before.table_z)

    def objects_in_container(self, img, cam, cfg, con, arm_q=None, jaw_q_deg=None):
        return [{"color": o.color, "area_px": 3000.0} for o in self.before.objects
                if o.in_container == con.id and not (self._out() and o.id == self.taken_id)]

    def object_presence(self, img, cam, cfg, obj, arm_q=None, jaw_q_deg=None):
        gone = self._out() and obj.id == self.taken_id
        return {"present": not gone, "color_now": None if gone else obj.color}


def _alike_scene():
    """The scene of the review: the white cap with an ivory cylinder 4 mm beside it (both said to be solid,
    so that the cap is held across: the grasp that can close on the neighbour instead)."""
    cap_xy = (BOX_C[0] + 0.012, BOX_C[1] + 0.012)
    ivory_xy = (cap_xy[0] - CAP_R - 0.013 - 0.004, cap_xy[1] - 0.018)
    return scene(cyl(1, "white", *cap_xy, hollow=False), cyl(2, "ivory", *ivory_xy, d=0.026, h=0.022, hollow=False),
                 take_box())


def test_take_wrong_object_is_never_confirmed():
    """F2. The jaws close on the ivory cylinder beside the white cap (a position error of 2 cm) and carry
    it out. The camera saw the cap at its place during the lift, sees it there at the end, and sees an
    ivory object on the table: that must not come back as ok (it did: white and ivory count as one colour
    inside a dark container, and "one thing of that colour fewer in the container" was taken as "it left")."""
    arm = FakeArm(parked=False)
    arm.stall_raw = 2231  # 26 mm between the jaws, not the cap's 31 mm
    r = take(arm, _OtherTaken(_alike_scene(), taken_id=2))
    v = r["checks"]["verify"]
    assert not r["ok"] and r["stage"] == "verify" and r["arm_state"] == "ready" and r["attempts"] == 1, r
    assert r["reason"].startswith("the wrong object was taken out: the ivory cylinder"), r["reason"]
    assert "is on the table at (" in r["reason"] and "the white cylinder asked for is still in the container" in r["reason"]
    assert v["wrong_object"] and not v["confirmed"] and not v["left_container"] and v["on_table"]["color"] == "ivory", v
    # (what the camera said, as in the review: seen at its place twice, one thing of the family fewer inside)
    assert r["checks"]["lift"]["camera"]["gone"] is False and not v["place_in_container_empty"] and v["inside"]["less"], v
    assert "repeat the command" in r["next"] and "fd pick --nearest" in r["next"] and "tell the user" in r["next"], r["next"]
    assert arm.released_at is not None and f"{arm.released_at[0]:.2f}"[:4] in r["next"], r["next"]
    # the same scene, the RIGHT object taken: confirmed as before (white on the table, its place seen empty)
    r = take(FakeArm(parked=False), FakeTakeScene(_alike_scene()))
    v = r["checks"]["verify"]
    assert r["ok"] and v["confirmed"] and v["on_table"]["color"] == "white" and "wrong_object" not in v, r
    # the ivory cylinder asked for, the white cap being ITS look-alike: confirmed
    arm = FakeArm(parked=False)
    arm.stall_raw = 2231
    r = take(arm, FakeTakeScene(_alike_scene(), target_id=2), query={"color": "ivory"})
    assert r["ok"] and r["target"]["id"] == 2 and r["checks"]["verify"]["on_table"]["color"] == "ivory", r
    # no look-alike in the container: nothing changes
    r = take(FakeArm(parked=False), FakeTakeScene(cap_scene()), mode="pinch")
    assert r["ok"] and r["checks"]["verify"]["confirmed"] and "look_alikes_in_container" not in r["checks"]["verify"], r
    # the colour names cannot decide (both are called white; taken by id): seen at its place twice while one
    # white thing fewer lies in the container is "something else left", not "it left"
    both = _alike_scene()
    both.by_id(2).color = "white"
    arm = FakeArm(parked=False)
    arm.stall_raw = 2231
    r = take(arm, _OtherTaken(both, taken_id=2), query={"id": 1})
    v = r["checks"]["verify"]
    assert not r["ok"] and r["stage"] == "verify" and "was the right object taken?" in r["reason"], r
    assert v["inside"]["less"] and not v["left_container"] and not v["confirmed"] and "wrong_object" not in v, v
    # the right object, but on the table the camera gives it the look-alike's colour name: not confirmed
    # either (the camera cannot tell which of the two came out), and the reason says exactly that
    r = take(FakeArm(parked=False), _OtherTaken(_alike_scene(), taken_id=1, table_color="ivory"))
    v = r["checks"]["verify"]
    assert not r["ok"] and r["stage"] == "verify" and "cannot tell which of the two was taken out" in r["reason"], r
    assert v["left_container"] and not v["confirmed"] and "wrong_object" not in v and "final" in r["next"], (v, r["next"])
    # _placed: of two new objects on the table the one of exactly the target's colour is "it"
    before = _alike_scene()
    after = scene(*before.objects, cyl(60, "ivory", 0.24, -0.02, z_bottom=TABLE_Z, in_container=None),
                  cyl(61, "white", 0.20, -0.07, z_bottom=TABLE_Z, in_container=None))
    assert skills._placed(before, after, before.by_id(1), (0.24, -0.02))["id"] == 61
    assert skills._placed(before, after, before.by_id(2), (0.20, -0.07))["id"] == 60


def test_frame_without_sim_never_reaches_the_real_arm():
    """F4. `fd pick | take | do | repl --frame IMG` replays a saved image. Without --sim (and without --dry)
    the REAL arm would be driven by it: refused with a SafetyError before the camera, the bus or anything
    else is opened. (Every way to the hardware is replaced by a recorder here, so this test touches nothing
    on a tree that does not refuse.)"""
    import contextlib
    import io
    import json

    import fd
    opened = []
    real = (fd.make_arm, fd._open_camera, fd.log_event)

    class _NoArm:
        def __enter__(self):
            raise SafetyError("test: the arm must not be opened")

        def __exit__(self, *exc):
            return False

    fd.make_arm = lambda sim: opened.append(("arm", sim)) or _NoArm()
    fd._open_camera = lambda a: opened.append(("camera", len(a.frame or ()))) or skills.ReplayCamera(
        [np.zeros((4, 4, 3), np.uint8)])
    fd.log_event = lambda *a, **k: None
    frame = os.path.join(HERE, "testdata", "0146_cap_final_raw.jpg")

    def call(argv):
        del opened[:]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fd.main(argv)
        return rc, json.loads(buf.getvalue())

    try:
        for argv in (["take", "--color", "white", "--frame", frame], ["pick", "--color", "blue", "--frame", frame],
                     ["take", "--id", "1", "--hollow", "--frame", frame, "--frame", frame],
                     ["do", "箱の中の白いキャップを取り出して", "--frame", frame], ["repl", "--frame", frame]):
            rc, res = call(argv)
            assert rc == 1 and res["ok"] is False and res["error"] == "SafetyError", (argv, res)
            assert "--frame" in res["message"] and "--sim" in res["message"] and "--dry" in res["message"], res
            assert opened == [], (argv, opened)  # neither the camera nor the arm
        # with --sim, as a dry run, and for commands that only look, the replay goes on (to the recorder here)
        for argv in (["--sim", "take", "--color", "white", "--frame", frame], ["--sim", "repl", "--frame", frame],
                     ["take", "--color", "white", "--dry", "--frame", frame], ["pick", "--color", "blue", "--dry", "--frame", frame],
                     ["do", "箱の中の白いキャップを取り出して", "--dry", "--frame", frame], ["scene", "--frame", frame]):
            rc, res = call(argv)
            assert [o[0] for o in opened] == ["camera", "arm"] and opened[1][1] == (argv[0] == "--sim"), (argv, opened)
            assert res.get("message") == "test: the arm must not be opened", res
    finally:
        fd.make_arm, fd._open_camera, fd.log_event = real


def test_take_place_spot_keeps_clear_of_where_the_arm_parks():
    """F5. `fd park` and the fold for a clear view lower the gripper onto the rest pose with contact
    allowed, knowing the static keep-outs only. A place spot therefore keeps the object's outline 40 mm
    away from where the parked gripper lies (its ground track at arm.rest_q(): no coordinate is built in)."""
    arm = FakeArm()
    rest_tip = arm.tip(arm.rest_q())[:2]  # where the fingertip lies when the arm is parked
    # a container close to the robot: the nearest free spot on the table is where the arm parks
    cx, cy = 0.215, 0.095
    sc = scene(cyl(1, "white", cx, cy), take_box(cx=cx, cy=cy))
    r = take(arm, FakeTakeScene(sc), dry=True)
    assert r["ok"] and arm.calls == [], r
    spot = np.array(r["place"]["xy"])
    assert float(np.hypot(*(spot - rest_tip))) >= CAP_R + 0.040, (spot, rest_tip)  # (the review's measure)
    # ... and it is not far away either: still next to the container
    outline = takeout._Obstacle(9, "container", RIM, poly=np.array(take_box(cx=cx, cy=cy).footprint))
    assert CAP_R + 0.025 - 1e-6 <= float(outline.sdist(spot)[0]) < 0.09, spot
    # a spot given by hand is refused there too, with the reason; one next to it is taken as given
    r = take(FakeArm(), FakeTakeScene(cap_scene()), to=(float(rest_tip[0]) + 0.02, float(rest_tip[1]) - 0.03), dry=True)
    assert not r["ok"] and r["stage"] == "plan" and "cannot be used: where the arm folds down to park" in r["reason"], r
    assert "--to" in r["next"], r["next"]
    r = take(FakeArm(), FakeTakeScene(cap_scene()), to=(0.22, -0.07), dry=True)
    assert r["ok"] and r["place"]["xy"] == [0.22, -0.07], r
    # the measure itself: the parked gripper's ground track (from under the palm to the foremost jaw nose)
    track = skills._park_track(arm)
    assert track is not None and skills._park_dist(track, rest_tip) < 0.006  # the fingertip lies on that track
    assert skills.TakeOptions().park_clear == 0.04 and skills.TakeOptions().fold_clear == 0.02
    assert skills._park_dist(track, spot) - CAP_R >= 0.040 - 1e-9, spot
    # without the rule the same scene puts the cap right there (the scenario is a real one)
    near = np.array(take(FakeArm(), FakeTakeScene(sc), dry=True, park_clear=-1.0)["place"]["xy"])
    assert skills._park_dist(track, near) - CAP_R < 0.030, near
    # the rule follows the arm's rest pose: another rest pose, another place
    turned = FakeArm()
    turned._rest = turned._rest + np.radians([35.0, 0.0, 0.0, 0.0, 0.0])
    t2 = skills._park_track(turned)
    assert float(np.hypot(*(t2[1] - track[1]))) > 0.05 and skills._park_dist(t2, turned.tip(turned.rest_q())[:2]) < 0.006
    assert abs(t2[2] - track[2]) < 1e-5 and TABLE_Z + 0.05 < track[2] < TABLE_Z + 0.12  # (where the gripper's body begins)
    r = take(turned, FakeTakeScene(cap_scene()), to=(float(rest_tip[0]) + 0.02, float(rest_tip[1]) - 0.03), dry=True)
    assert r["ok"], r
    # an arm that cannot say where it rests: no such test (as with a missing camera model)
    assert skills._park_track(object()) is None


def test_take_does_not_fold_back_onto_something():
    """F5. The fold for a clear view of a hidden container goes down onto the rest pose. When the first
    look shows something standing there, the arm is not folded: with nothing to take yet the take-out
    stops and says what is in the way; with the target already found it goes on with the first look."""
    arm = FakeArm(parked=False)
    rest_tip = arm.tip(arm.rest_q())[:2]  # where the fingertip comes down when the arm folds
    block = dict(size=(0.03, 0.03, 0.035), in_container=None)  # a block 35 mm high, 5 mm beside that point
    behind = scene(box(7, "red", float(rest_tip[0]) - 0.01, float(rest_tip[1]) - 0.02, **block))
    behind.warnings.append("a dark box-like region near x=0.22 y=+0.09 has 1 corner(s) hidden by the robot arm and is "
                           "not reported as a container: move the arm away and look again")
    S = FakeTakeScene(cap_scene(), behind_arm=behind)
    r = take(arm, S, mode="pinch")
    assert arm.calls == [] and S.n_perceive == 1, arm.calls  # nothing moved, least of all down onto the block
    assert not r["ok"] and r["stage"] == "perceive" and r["arm_state"] == "ready" and r["attempts"] == 1, r
    assert r["reason"].startswith("container not found") and "NOT folded back" in r["reason"], r["reason"]
    assert "red box (id 7)" in r["reason"] and "stands where the folded arm would come down" in r["reason"], r["reason"]
    assert "Move that object away" in r["next"], r["next"]
    # the target was found, only a warning says that the arm hides something: no fold, the first look is used
    seen = cap_scene(box(7, "red", float(rest_tip[0]) - 0.01, float(rest_tip[1]) - 0.02, **block))
    seen.warnings.append("one corner of the container is hidden by the robot arm")
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(seen), mode="pinch")
    assert kinds_of(arm) == PINCH_CALLS and not np.allclose(arm.tip(arm.rest_q()).round(4), arm.calls[0][5], atol=1e-3)
    assert r["ok"] and r["checks"]["second_look"].startswith("not made (the red box (id 7)"), r
    # ... and the cap is not put down beside the parked gripper either
    assert float(np.hypot(*(np.array(r["place"]["xy"]) - rest_tip))) >= CAP_R + 0.040, r["place"]
    # the same block 6 cm clear of that place: folded back for the second look, as before
    far = scene(box(7, "red", float(rest_tip[0]) + 0.09, float(rest_tip[1]), **block))
    far.warnings.append("a dark box-like region has 1 corner(s) hidden by the robot arm and is not reported as a container")
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene(), behind_arm=far), mode="pinch")
    assert r["ok"] and "folded back" in r["checks"]["second_look"], r
    assert np.allclose(arm.tip(arm.rest_q()).round(4), arm.calls[0][5], atol=1e-3), arm.calls[0]
    # what counts as "standing there": the outline, not the centre (a container next to the parking place)
    track = skills._park_track(arm)
    on_it = 0.5 * (track[0] + track[1])
    assert skills._park_clearance(track, take_box(cx=float(on_it[0]), cy=float(on_it[1]))) == 0.0
    box_beside = take_box(cx=float(on_it[0]), cy=float(on_it[1]) + 0.05 + 0.025)
    assert abs(skills._park_clearance(track, box_beside) - 0.025) < 0.004, skills._park_clearance(track, box_beside)
    assert abs(skills._park_clearance(track, cyl(3, "blue", float(track[1][0]) + 0.033, float(track[1][1]), d=0.026)) - 0.020) < 1e-6
    # the container of these tests stands 25 mm beside that place, lower than the gripper's body: clear of the
    # fingers. Something as tall as the body (a tower of blocks) is in its way at that distance
    fold_clear = skills.TakeOptions().fold_clear
    assert skills._parking_blocked(arm, cap_scene(), fold_clear) == ""
    assert 0.020 < skills._park_clearance(track, take_box()) < 0.030 and take_box().z_top < track[2] - 0.03
    tower = box(8, "red", float(track[1][0]), float(track[1][1]) + 0.03 + 0.015, size=(0.03, 0.03, 0.10), in_container=None)
    assert "red box (id 8)" in skills._parking_blocked(arm, scene(tower), fold_clear)
    tower.size = (0.03, 0.03, 0.03)
    assert skills._parking_blocked(arm, scene(tower), fold_clear) == ""


def test_take_look_cannot_be_switched_off_on_a_real_arm():
    """F6. The look at the jaws is what keeps a jaw off a wall: cfg["take"]["visual"] = "off" (which
    `fd config --set` can write) must not switch it off on a real arm, and a result must not call an arm
    "simulated" that is not."""
    o = skills.take_options_from_config({"take": {"visual": "off", "max_visual_corr": 0.2, "min_visual_conf": 0.0}})
    assert o.visual == "off"  # (the setting is read: the skill is where it is refused)
    arm = FakeArm(parked=False)  # a stand-in for a REAL arm: nothing about it says "simulation"
    S = FakeTakeScene(cap_scene())
    r = take(arm, S, mode="pinch", visual=o.visual)
    assert not r["ok"] and r["stage"] == "start" and r["arm_state"] == "ready", r
    assert arm.calls == [] and S.n_perceive == 0 and S.n_jaws == 0 and "visual" not in r["checks"], arm.calls
    assert "switched off" in r["reason"] and "not a simulated arm" in r["reason"], r["reason"]
    assert "nothing moved" in r["next"] and "take.visual" in r["next"] and "--sim" in r["next"], r["next"]
    # a value that is neither "on" nor "off" is a mistake, not a way to say "on"
    for bad in ("of", "auto", "ON", "", None, True, 0):
        arm = FakeArm(parked=False)
        r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", visual=bad)
        assert not r["ok"] and r["stage"] == "start" and arm.calls == [] and "it must be \"on\"" in r["reason"], (bad, r)
        assert "take.visual" in r["next"], r["next"]
    # a dry run moves nothing: it plans whatever the setting
    arm = FakeArm(parked=False)
    r = take(arm, FakeTakeScene(cap_scene()), mode="pinch", visual="off", dry=True)
    assert r["ok"] and r["stage"] == "planned" and arm.calls == [], r
    # the simulated arm (a SimBus), which no camera sees, is the one place for it; the note says why
    assert not skills._simulated(FakeArm()) and not skills._simulated(object())
    with _sim_arm() as sim:
        assert skills._simulated(sim)
    r, _tip, _off = _sim_take(_SimScene(cap_scene()), {"color": "white"}, [np.zeros((4, 4, 3), np.uint8)], mode="pinch")
    note = r["checks"]["visual"]["note"]
    assert r["ok"] and r["checks"]["visual"]["mode"] == "off" and "simulated arm, which no camera sees" in note, note
    assert "NOT" not in note


def test_do_runs_only_a_plain_positive_command():
    """F1, end to end through `fd do`: a prohibition, an exclusion, a condition, a question, a statement
    and a request with a second part name the same object and container as the plain command; none of
    them may reach `take` or `pick`. (The sentences themselves: `python tools/cli/nlq.py`.)"""
    import fd
    ran = []
    real = (fd.cmd_take, fd.cmd_pick, fd.cmd_scene)
    done = {"ok": True, "stage": "done", "timings": {"total": 1.0}}
    fd.cmd_take = lambda arm, a: ran.append(("take", a.color, a.kind)) or {**done, "skill": "take_out", "target": {"color": a.color}}
    fd.cmd_pick = lambda arm, a: ran.append(("pick", a.color, a.kind)) or {**done, "target": {"color": a.color}}
    fd.cmd_scene = lambda arm, a: {"objects": [{"id": 1, "color": "white"}], "warnings": [], "view": "scene.jpg"}

    def do(text):
        del ran[:]
        return fd.cmd_do(None, fd.build_parser().parse_args(["--sim", "do", text]))

    try:
        for text, kind in (("箱の中の白いキャップを取り出さないで", "prohibition"), ("箱の中の白いキャップは取り出すな", "prohibition"),
                           ("青いキューブを箱に入れないで", "prohibition"), ("don't take the white cap out of the box", "prohibition"),
                           ("do not put the blue cube into the box", "prohibition"),
                           ("箱から白いキャップ以外を出して", "exclusion"), ("put everything except the blue cube into the box", "exclusion"),
                           ("箱の中の白くないキャップを取り出して", "negation"), ("箱から白いキャップを出したら教えて", "condition"),
                           ("箱の中の白いキャップを取り出した?", "question"), ("箱の中の白いキャップを取り出した", "statement"),
                           ("did you put the blue cube into the box", "question"), ("I took the white cap out of the box", "statement"),
                           ("take the white cap out of the box and then put it back", "extra"),
                           ("箱の中の白いキャップを取り出して机の右に置いて", "extra")):
            res = do(text)
            assert ran == [], (text, ran)
            assert res["ok"] is False and res["need_llm"] is True and res["not_plain"] == kind and res["text"] == text, (text, res)
            assert res["next"].startswith("nothing was moved: the sentence is not a plain command") and "say" not in res, res
            assert res["objects"] == [{"id": 1, "color": "white"}]
        # two containers in one sentence: neither is dropped, nothing runs
        res = do("赤い箱から白い箱に白いキャップを出して")
        assert ran == [] and res["need_llm"] is True and "one container into another" in res["reason"], res
        # the plain commands run as before, with the same queries
        assert do("箱の中の白いキャップを取り出して")["ok"] and ran == [("take", "white", "cylinder")], ran
        assert do("Boxの中のペットボトルのキャップ(白)を取り出して")["ok"] and ran == [("take", "white", "cylinder")], ran
        assert do("ペットボトルキャップをBoxから取り出して")["ok"] and ran == [("take", None, "cylinder")], ran
        assert do("青いキューブをボックスに入れて")["ok"] and ran == [("pick", "blue", "box")], ran
        assert do("赤いBoxを黒い箱の中に入れてください")["ok"] and ran == [("pick", "red", None)], ran
        res = do("全部箱に入れて")  # (as before: no 「を」, object and destination cannot be told apart)
        assert ran == [] and res["need_llm"] is True and "not_plain" not in res and "fd pick --id N" in res["next"], res
    finally:
        fd.cmd_take, fd.cmd_pick, fd.cmd_scene = real


def test_nearest_needs_two_numbers():
    """N3. `--nearest x,y`: anything but two finite numbers is refused with the same clear message as --to."""
    import fd
    p = fd.build_parser()
    for cmd in ("take", "pick"):
        assert fd._query(p.parse_args([cmd, "--nearest", "0.2,0.05"])) == {"nearest_to": (0.2, 0.05)}
        assert fd._query(p.parse_args([cmd, "--color", "white"])) == {"color": "white"}
        for bad in ("foo", "0.2", "0.2,0.05,0.1", "0.2,abc", "nan,0.1", "inf,0", ","):
            try:
                fd._query(p.parse_args([cmd, "--nearest", bad]))
                raise AssertionError(f"--nearest {bad} should be refused")
            except SafetyError as e:
                assert str(e) == "--nearest needs x,y in metres (base frame), e.g. --nearest 0.23,-0.03", (bad, str(e))


def test_take_out_of_reach_says_so():
    """N4. A container so far out that the arm cannot hold the gripper straight down at the point above the
    hover point (where the approach arrives): refused at the plan stage, for that reason, with a next step
    that is about reach (not "make room around it or give the place with --to")."""
    cx, cy = 0.31, 0.06
    sc = scene(cyl(1, "white", cx, cy), take_box(cx=cx, cy=cy))
    arm = FakeArm()
    r = take(arm, FakeTakeScene(sc), mode="pinch", dry=True)
    assert not r["ok"] and r["stage"] == "plan" and arm.calls == [], r
    assert "have room for the jaws, but the arm cannot point straight down at the point above the hover point" in r["reason"]
    assert "too far from the robot" in r["reason"] and "cm from the robot's shoulder axis" in r["reason"], r["reason"]
    assert "move the container nearer to the robot" in r["next"] and "does not help" in r["next"], r["next"]
    assert "make room" not in r["next"]
    assert any("point above the hover point" in c["reason"] for c in r["grasp_candidates"]), r["grasp_candidates"]
    # that point really is out of reach while the hover point below it is not (so only this check finds it)
    g = [c for c in takeout.plan_grasps(sc.by_id(1), sc.by_id(9), sc, arm.cfg, mode="pinch", tip_offset=TIP_OFF) if c.ok][0]
    arm.solve(g.tip_hover, 90.0, g.roll_deg)
    try:
        arm.solve((g.tip_hover[0], g.tip_hover[1], g.tip_hover[2] + 0.015), 90.0, g.roll_deg)
        raise AssertionError("the point above the hover point should be out of reach here")
    except SafetyError:
        pass
    # fd's own one-liner for a plan that fails on reach without a next step from the skill
    import fd
    tip = fd._advice_take({"stage": "plan", "arm_state": "ready",
                           "reason": "no feasible path: IK failed for xyz=[0.2984, 0.0576, 0.0334] pitch=90.0"})
    assert "reach" in tip and "make room" not in tip, tip
    assert "make room" in fd._advice_take({"stage": "plan", "arm_state": "ready", "reason": "no jaw direction leaves room"})


def test_docs_say_what_the_code_does():
    """N1, N2, F3 (docs/MANUAL.md and CLAUDE.md are what the operator acts on): the numbers and the few
    statements the review found wrong are held against the code, so that they cannot drift again."""
    import contextlib
    import io
    import re

    import nlq
    readme = open(os.path.join(tazuna_paths.ROOT, "docs", "MANUAL.md"), encoding="utf-8").read()     # the detailed manual
    claude = open(os.path.join(tazuna_paths.ROOT, "CLAUDE.md"), encoding="utf-8").read()
    # N2: the test counts of chapter 7 are those of the files
    counts = {}
    for name in ("test_skills.py", "test_sim.py", "test_motion_phys.py", "test_kinematics.py", "test_scene.py",
                 "test_grasp.py", "test_bus_fake.py"):
        counts[name] = len(re.findall(r"^def test_\w+\(", open(os.path.join(HERE, name), encoding="utf-8").read(), flags=re.M))
        line = [ln for ln in readme.splitlines() if ln.startswith(f"$PY tests/{name} ")]
        tail = readme[readme.index(line[0]):readme.index(line[0]) + 400].split("$PY", 2)[1] if line else ""
        assert f"{counts[name]} 件" in tail, (name, counts[name], tail[:160])
    src = open(os.path.join(HERE, "test_skills.py"), encoding="utf-8").read()
    take_part = src[src.index("# ----------------------------------------------------------------------------- take-out: stand-ins"):]
    n_take = len([n for n in re.findall(r"^def (test_\w+)\(", take_part, flags=re.M)
                  if n not in ("test_e2e_dry_blue_block", "test_e2e_sim_pick_blue_block")])
    assert f"（{counts['test_skills.py']} 件。うち取り出し {n_take} 件）" in readme, (counts["test_skills.py"], n_take)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert nlq._selftest() == 0
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("PASS")]
    refused = sum("-> not a plain command:" in ln for ln in lines)
    assert f"（{len(lines)} 例。うち「素直な肯定の命令文ではない」として断る文 {refused} 例）" in readme, (len(lines), refused)
    # N2: --sim keeps away from the servos, not from the camera (without --frame the real one is opened and locked)
    sim_row = [ln for ln in readme.splitlines() if ln.startswith("| `./fd --sim <コマンド>`")][0]
    assert "実機に一切触れない" not in sim_row and "カメラは実物を使う" in sim_row and "`--frame`" in sim_row, sim_row
    assert "実カメラを開いて" in claude and "`--frame` は `--sim` か `--dry`" in claude
    # N1: after a failed grasp / lift "repeat the command" holds only when the arm is not holding or stopped
    advice = claude[claude.index("`grasp` / `lift` →"):]
    advice = advice[:advice.index("`verify`")]
    assert "holding_object" in advice and "stopped" in advice and "`next` に従う" in advice, advice
    assert advice.index("holding_object") < advice.index("もう一度同じコマンド"), advice
    # F3: the documented limit of the look is the code's, and both documents say what the look did not catch
    o = skills.TakeOptions()
    for key, value in (("max_visual_corr", f"{o.max_visual_corr:.3f}"), ("visual_edge", f"{o.visual_edge:g}"),
                       ("park_clear", f"{o.park_clear:.2f}"), ("fold_clear", f"{o.fold_clear:.2f}"), ("visual", f"`{o.visual}`")):
        assert f"| `take.{key}` | {value} |" in readme, (key, value)
    assert "30 mm）" not in readme and f"{o.max_visual_corr * 1000:.0f} mm" in readme and f"{o.max_visual_corr * 1000:.0f} mm" in claude
    assert "爪の目視で分かること・分からないこと" in readme and "捉えられていなかった" in readme and "防げていなかった" in claude
    # ... and the example sentences the documents promise are read as they say
    for text, action in (("青いキューブを箱に入れて", "put_in"), ("箱に白いキャップを入れて", "put_in"), ("赤と青のブロックを箱に入れて", "put_in"),
                         ("ブロックを全部箱に片付けて", "put_in"), ("テーブルの上のものを全部箱に入れて", "put_in"),
                         ("箱の中の白いキャップを取り出して", "take_out"), ("白いキャップを箱から出して", "take_out"),
                         ("箱から赤いブロックを取って机に置いて", "take_out")):
        assert text in readme + claude and nlq.parse(text).get("action") == action and nlq.parse(text)["ok"], text
    for text in ("箱の中の白いキャップを取り出さないで", "全部箱に入れて"):
        assert text in readme + claude and not nlq.parse(text)["ok"], text


# ----------------------------------------------------------------------------- take-out: geometry (takeout.py)
def plan(sc, target_id=1, **kw):
    kw.setdefault("tip_offset", TIP_OFF)
    return takeout.plan_grasps(sc.by_id(target_id), sc.by_id(9), sc, PLAN_CFG, **kw)


def az_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_geometry_cap_in_the_middle_of_an_empty_box():
    cands = plan(cap_scene(), hollow=True)  # said to be a cap, open side up (--hollow)
    pinch = [c for c in cands if c.mode == "pinch"]
    two = [c for c in cands if c.mode == "two_sided"]
    # nothing is near: the wall pinch works in every direction; the grasp across needs room for both jaws
    # (it fails only where the opened moving jaw would meet the nearer walls, 31.5 mm from the cap)
    # (a few directions need a wrist roll beyond +-160 deg: the only reason for a refusal here)
    refused = [c for c in pinch if not c.ok]
    assert len(pinch) == 36 and len(refused) <= 5 and all("wrist roll" in c.reason for c in refused), \
        [c.reason for c in refused][:3]
    assert sum(c.ok for c in two) >= 20 and cands[0].ok
    c = [c for c in pinch if c.ok][0]
    assert c.open_raw == 2165 and c.release_raw == 2165 and c.expected_raw == 2067  # 14 mm apart; a 2 mm wall
    # the height is pure geometry: table + container floor (5 mm) + the cap's own bottom (2 mm) + 2 mm
    assert abs(c.standoff - 0.005) < 1e-9 and abs(c.tip_grasp[2] - (TABLE_Z + 0.005 + 0.002 + 0.002)) < 1e-5
    assert abs(c.z_surface - CAP_INNER) < 1e-9 and abs(c.tip_above_surface - TIP_ABOVE) < 1e-12
    assert abs(c.tip_hover[2] - (RIM + 0.012)) < 1e-5 and c.pitch_deg == 90.0
    assert 0.0015 < c.margin < 0.002 and abs(c.h_below - H_BELOW) < 1e-9  # moving nose 1.76 mm above the inner bottom
    # the fingertip stands beside the wall: pad 5 mm outside the outer surface, on the side opposite to the azimuth
    u = np.array([math.cos(math.radians(c.jaw_azimuth_deg)), math.sin(math.radians(c.jaw_azimuth_deg))])
    assert abs((np.array(BOX_C) - np.array(c.tip_grasp[:2])) @ u - (CAP_R + 0.005 - 0.001)) < 2e-5
    # the slot is bounded by the wall; the finger is taken at its thickness at the RIM height, not at its tip
    assert c.slot["bounded_by"] == "the container wall" and 13.0 < c.slot["finger_mm"] < 14.5, c.slot
    assert abs(c.slot["finger_mm"] - GM.finger_back_x(6.8 - (RIM + 0.003 - c.tip_grasp[2]) * 1e3)) < 0.06
    # plenty of room on both sides: the normal grasp across the cap ranks first (no touch needed)
    assert cands[0].mode == "two_sided" and cands[0].margin >= 0.003, cands[0]
    assert all(c.look_closed is False for c in cands)


def test_geometry_cap_5mm_from_a_wall():
    sc = cap_scene(cap_xy=(X_WALL + 0.005 + CAP_R, BOX_C[1]))
    cands = plan(sc, mode="pinch")
    ok = [c for c in cands if c.ok]
    assert ok and len(ok) < 36
    # the fixed jaw stands on the side opposite to the azimuth: azimuth 0 would put it between cap and wall
    at_wall = [c for c in cands if az_diff(c.jaw_azimuth_deg, 0.0) < 1.0][0]
    assert not at_wall.ok and "5.0 mm free up to the container wall" in at_wall.reason, at_wall.reason
    # needed: the finger's thickness at the height of the rim (13.7 mm) + 2 mm to the cap + 2 mm to the wall
    thick = takeout.fixed_back(6.8 - (RIM + 0.003 - (CAP_INNER + TIP_ABOVE)) * 1e3)
    assert 0.0135 < thick < 0.0139 and f"{(thick + 0.004) * 1e3:.1f} mm needed" in at_wall.reason, (thick, at_wall.reason)
    assert at_wall.blockers[0]["wall"] is True
    assert all(az_diff(c.jaw_azimuth_deg, 0.0) > 45.0 for c in ok), sorted(c.jaw_azimuth_deg for c in ok)
    # the best one has its fixed jaw well away from the wall and room behind it
    assert az_diff(cands[0].jaw_azimuth_deg, 0.0) >= 90.0 and cands[0].clearance_fixed > 0.008
    # across the cap, parallel to the wall, both jaws have room; with a jaw on the wall side they have not
    two = plan(sc, mode="two_sided")
    assert two[0].ok and min(az_diff(two[0].jaw_azimuth_deg, 90.0), az_diff(two[0].jaw_azimuth_deg, -90.0)) <= 20.0
    assert not any(c.ok for c in two if min(az_diff(c.jaw_azimuth_deg, 0.0), az_diff(c.jaw_azimuth_deg, 180.0)) < 25.0)


def test_geometry_cap_hemmed_in_on_three_sides():
    sc = hemmed_scene()  # wall 22 mm behind, ivory cylinder touching, blue disc touching, red block 3 mm away
    cands = plan(sc, hollow=True)
    ok = [c for c in cands if c.ok]
    assert ok and all(c.mode == "pinch" for c in ok), [(c.mode, c.jaw_azimuth_deg) for c in ok]
    # only the slot between the cap and the wall takes the outside finger: the jaws open away from the wall
    assert all(az_diff(c.jaw_azimuth_deg, 0.0) <= 30.0 for c in ok), sorted(c.jaw_azimuth_deg for c in ok)
    best = cands[0]
    assert best.slot["bounded_by"] == "the container wall" and 21.0 < best.slot["free_mm"] <= 22.01, best.slot
    # 22 mm = 13.5 mm of finger at rim height + the gap to the cap + the clearance behind: the gap gives way
    assert 0.002 <= best.standoff < 0.005 and best.clearance_fixed >= 0.002, (best.standoff, best.clearance_fixed)
    assert abs(best.standoff + best.clearance_fixed + best.slot["finger_mm"] / 1e3 - best.slot["free_mm"] / 1e3) < 2e-4
    # no room for two jaws anywhere, and the reasons name what is in the way
    two = [c for c in cands if c.mode == "two_sided"]
    assert not any(c.ok for c in two)
    named = " ".join(c.reason for c in cands if not c.ok)
    assert "ivory cylinder #2" in named and "blue cylinder #3" in named and "red box #4" in named
    # the low disc in front is in the way of a finger although a fingertip POINT would pass over it
    front = [c for c in cands if c.mode == "pinch" and az_diff(c.jaw_azimuth_deg, 180.0) < 1.0][0]
    assert not front.ok and front.blockers[0]["label"] == "blue cylinder #3", front.reason
    # the wall 8 mm away instead: nothing is left, and the advice names a neighbour to take out first
    boxed = plan(hemmed_scene(fourth=True))
    assert not any(c.ok for c in boxed)
    ex = takeout.explain(boxed)
    assert "the container wall" in ex["reason"] and "closest:" in ex["reason"], ex
    assert ex["next"].startswith("take ") and "out first (`fd take --id " in ex["next"], ex


def test_geometry_solid_cylinder_with_room_on_both_sides():
    sc = scene(cyl(1, "ivory", *BOX_C, d=0.026, h=0.022, hollow=False), take_box())
    cands = plan(sc)
    assert {c.mode for c in cands} == {"two_sided"} and cands[0].ok
    c = cands[0]
    assert abs(c.width - 0.026) < 1e-9 and abs(c.standoff - 0.008) < 1e-9
    assert c.expected_raw == GM.stall_raw_for_width(26.0) and c.open_raw > GM.raw_from_q(GM.q_for_width(26.0)) + 100
    assert abs(c.tip_grasp[2] - (FLOOR + 0.022 - 0.015)) < 1e-5 and c.margin > 0.003
    assert c.clearance_fixed > 0.01 and c.clearance_moving > 0.01
    u = np.array([math.cos(math.radians(c.jaw_azimuth_deg)), math.sin(math.radians(c.jaw_azimuth_deg))])
    assert abs((np.array(BOX_C) - np.array(c.tip_grasp[:2])) @ u - (0.013 + 0.008 - 0.001)) < 2e-5
    # it cannot be pinched by a wall it does not have; asked for all the same, the planner says so
    forced = plan(sc, mode="pinch", hollow=False)
    assert len(forced) == 1 and not forced[0].ok and "not hollow" in forced[0].reason
    # a neighbour 10 mm away on one side: the grasp turns so that neither jaw has to go there
    near = scene(cyl(1, "ivory", *BOX_C, d=0.026, h=0.022, hollow=False),
                 cyl(2, "blue", BOX_C[0] + 0.013 + 0.010 + 0.012, BOX_C[1], d=0.024, h=0.02), take_box())
    cands = plan(near)
    assert cands[0].ok and min(az_diff(cands[0].jaw_azimuth_deg, 90.0), az_diff(cands[0].jaw_azimuth_deg, -90.0)) <= 30.0
    assert not [c for c in cands if c.ok and min(az_diff(c.jaw_azimuth_deg, 0.0), az_diff(c.jaw_azimuth_deg, 180.0)) < 5.0]


def test_geometry_finger_thickness_at_the_height_of_the_obstacle():
    # the same 12 mm slot: enough beside something 8 mm high (the finger is 7.5 mm thick down there), not
    # enough beside the wall (13.7 mm thick at rim height) and not beside something 20 mm high (11.9 mm:
    # the fingertip stands 4 mm above the floor, the obstacle counts 3 mm taller than reported)
    def slot(obstacle_h):
        x = BOX_C[0] - CAP_R - 0.012 - 0.015
        low = box(2, "blue", x, BOX_C[1], size=(0.03, 0.05, obstacle_h), z_bottom=FLOOR, in_container=9)
        c = [c for c in plan(cap_scene(low), mode="pinch") if az_diff(c.jaw_azimuth_deg, 0.0) < 1.0][0]
        return c

    low = slot(0.008)
    assert low.ok and 7.0 < low.slot["finger_mm"] < 8.0 and abs(low.slot["free_mm"] - 12.0) < 0.05, low.slot
    assert abs(low.standoff + low.clearance_fixed + low.slot["finger_mm"] / 1e3 - 0.012) < 1e-4
    high = slot(0.020)
    assert not high.ok and "12.0 mm free up to blue box #2" in high.reason, high.reason
    assert 11.5 < high.blockers[0]["need_mm"] - 4.0 < 12.5, high.blockers
    wall = [c for c in plan(cap_scene(cap_xy=(X_WALL + 0.012 + CAP_R, BOX_C[1])), mode="pinch")
            if az_diff(c.jaw_azimuth_deg, 0.0) < 1.0][0]
    assert not wall.ok and wall.blockers[0]["wall"]
    # something lower than the fingertips are when they close is no obstacle at all
    flat = slot(0.0015)
    assert flat.ok and flat.slot["bounded_by"] == "the container wall"
    # the outline functions, as the planner uses them
    assert abs(takeout.fixed_back(GM.FIXED_TIP_Z) - 0.00518) < 2e-5 and abs(takeout.fixed_back(GM.FIXED_TIP_Z - 10) - 0.00882) < 2e-5
    assert abs(takeout.fixed_back(GM.FIXED_TIP_Z - 20) - 0.01245) < 2e-5
    assert takeout.fixed_half_width(GM.FIXED_TIP_Z - 5.0) == 0.00635 and takeout.moving_half_width(0.0, 0.0) == 0.0064
    assert takeout.fixed_half_width(GM.FIXED_TIP_Z - 30.0) > 0.008


def test_geometry_cavity_and_hollow_model():
    cap = cyl(1, "white", *BOX_C)
    hm = takeout.hollow_model(cap)
    assert hm["hollow"] and hm["source"] == "default" and abs(hm["wall"] - 0.002) < 1e-12
    assert abs(hm["inner_radius"] - 0.0135) < 1e-12 and abs(hm["z_inner_bottom"] - CAP_INNER) < 1e-12
    assert takeout.hollow_model(cap, hollow=False)["hollow"] is False
    assert takeout.hollow_model(box(2, "red", 0.2, 0.08))["hollow"] is False  # only a cylinder is taken as hollow unasked
    # what the scene says wins over the defaults; the option wins over the scene
    cap.hollow, cap.inner_radius, cap.z_inner_bottom = True, 0.0140, FLOOR + 0.0015
    hm = takeout.hollow_model(cap)
    assert hm["source"] == "scene" and abs(hm["wall"] - 0.0015) < 1e-9 and abs(hm["z_inner_bottom"] - (FLOOR + 0.0015)) < 1e-12
    c = plan(scene(cap, take_box()), mode="pinch")[0]
    assert c.ok and abs(c.tip_grasp[2] - (FLOOR + 0.0015 + TIP_ABOVE)) < 1e-5 and c.expected_raw == GM.raw_from_q(GM.q_for_width(1.5))
    assert takeout.hollow_model(cap, hollow=False)["source"] == "option"
    cap.hollow = False
    assert not any(c.mode == "pinch" for c in plan(scene(cap, take_box())))
    assert any(c.mode == "pinch" and c.ok for c in plan(scene(cap, take_box()), hollow=True))
    # nonsense from the scene is replaced by the defaults, and said
    odd = cyl(1, "white", *BOX_C)
    odd.wall, odd.z_inner_bottom = 0.03, FLOOR + 0.5
    hm = takeout.hollow_model(odd)
    assert abs(hm["wall"] - 0.002) < 1e-12 and abs(hm["z_inner_bottom"] - CAP_INNER) < 1e-12 and len(hm["notes"]) == 2
    # a cavity too small for the moving finger (12.7 mm wide): no pinch, and the reason says so
    small = plan(scene(cyl(1, "white", *BOX_C, d=0.020), take_box()), mode="pinch")
    assert not any(c.ok for c in small) and "too small for the moving finger" in small[0].reason, small[0].reason
    # a cavity too shallow for the pads
    shallow = plan(scene(cyl(1, "white", *BOX_C, h=0.008), take_box()), mode="pinch")
    assert not any(c.ok for c in shallow) and "deep" in shallow[0].reason
    # a wider cup: the opening follows wall + gaps, the moving jaw stays inside
    cup = plan(scene(cyl(1, "white", *BOX_C, d=0.05, h=0.03), take_box()), mode="pinch")
    assert cup[0].ok and cup[0].open_raw == 2165 and cup[0].clearance_moving > 0.004, cup[0]


def test_geometry_wall_pinch_only_for_an_object_said_to_be_hollow():
    """A cylinder nobody says anything about is modelled as hollow, but its wall pinch is not released:
    nothing checks the cavity before the moving jaw goes down into it (on a solid cylinder it would land
    on the top and push). The scene, the option or the request's word release it."""
    cap = cyl(1, "white", *BOX_C)
    assert takeout.hollow_model(cap)["source"] == "default"
    assert takeout.hollow_model(cap, assume=True) == {**takeout.hollow_model(cap), "source": "request"}
    assert takeout.hollow_model(cap, hollow=True, assume=True)["source"] == "option"
    said_not = cyl(1, "white", *BOX_C, hollow=False)
    hm = takeout.hollow_model(said_not, assume=True)  # what the camera saw wins over the request's word
    assert hm["hollow"] is False and hm["source"] == "scene"
    box_like = takeout.hollow_model(box(2, "red", 0.2, 0.08), assume=True)  # "the red cup" seen as a box
    assert box_like["hollow"] is True and box_like["source"] == "request"
    # hemmed in: only the wall pinch has room, and it rests on the guess
    guessed = plan(hemmed_scene())
    assert not any(c.ok for c in guessed)
    unconfirmed = [c for c in guessed if c.unconfirmed]
    assert unconfirmed and all(c.mode == "pinch" and "only a guess" in c.reason and c.score == 0.0 for c in unconfirmed)
    assert all(az_diff(c.jaw_azimuth_deg, 0.0) <= 30.0 for c in unconfirmed)  # the slot between cap and wall
    assert any("is a guess for a cylinder" in n for n in unconfirmed[0].notes), unconfirmed[0].notes
    assert unconfirmed[0].to_dict()["unconfirmed"] is True
    ex = takeout.explain(guessed)
    assert ex["reason"].startswith("it can be held by its wall") and "only a guess" in ex["reason"], ex
    assert "across it, no jaw direction leaves room for both jaws" in ex["reason"] and "blue cylinder #3" in ex["reason"], ex
    assert ex["next"].startswith("if it is a cap or a cup lying open side up, say so: `fd take --hollow`"), ex
    assert "; if it is solid: " in ex["next"], ex
    # said: by the option, by the scene, by the request; mode="pinch" is such a statement too
    told = hemmed_scene()
    told.by_id(1).hollow = True
    ways = [plan(hemmed_scene(), hollow=True), plan(told), plan(hemmed_scene(), assume_hollow=True),
            plan(hemmed_scene(), mode="pinch")]
    for cands in ways:
        ok = [c for c in cands if c.ok]
        assert ok and all(c.mode == "pinch" and not c.unconfirmed for c in ok)
        assert (cands[0].jaw_azimuth_deg, cands[0].tip_grasp) == (ways[0][0].jaw_azimuth_deg, ways[0][0].tip_grasp)
    assert any("because the request says so" in n for n in ways[2][0].notes), ways[2][0].notes
    assert not any("guess" in n or "request" in n for n in ways[0][0].notes + ways[1][0].notes)
    # the scene says it is closed: no pinch, whatever the request calls it; the option still wins
    closed = hemmed_scene()
    closed.by_id(1).hollow = False
    assert not any(c.mode == "pinch" for c in plan(closed, assume_hollow=True))
    assert "only a guess" not in takeout.explain(plan(closed, assume_hollow=True))["reason"]
    assert any(c.ok and c.mode == "pinch" for c in plan(closed, hollow=True))
    # room on both sides: the grasp across needs no statement and ranks first; the pinch stays unconfirmed
    free = plan(cap_scene())
    assert free[0].ok and free[0].mode == "two_sided"
    assert all(c.mode == "two_sided" for c in free if c.ok) and any(c.unconfirmed for c in free)
    # a solid object that cannot be held across: the advice still mentions --hollow, for the case that it is not solid
    solid = hemmed_scene()
    solid.by_id(1).kind = "unknown"
    ex = takeout.explain(plan(solid))
    assert ex["reason"].startswith("no jaw direction leaves room for both jaws") and "only a guess" not in ex["reason"], ex
    assert ex["next"].startswith("take ") or "`fd take --hollow`" in ex["next"], ex


def test_geometry_jaws_open_across_the_cameras_line_of_sight():
    """The look at the jaws is weakest ALONG the camera's line of sight (an error of the arm's height shifts
    the reading there, and one jaw can hide the other). Other things equal the jaws open across it: each
    candidate loses 6 points times |cos| of the angle between its jaw direction and the line of sight."""
    cam_xy = (0.55, 0.04)
    sc = cap_scene()
    blind = {round(c.jaw_azimuth_deg, 1): c for c in plan(sc, mode="pinch") if c.ok}
    seen = {round(c.jaw_azimuth_deg, 1): c for c in plan(sc, mode="pinch", camera_xy=cam_xy) if c.ok}
    assert len(seen) >= 30 and set(seen) == set(blind), (len(seen), len(blind))
    for az, c in seen.items():
        los = np.array(cam_xy) - np.array(c.tip_hover[:2])
        cos = abs(float(np.array([math.cos(math.radians(az)), math.sin(math.radians(az))]) @ los) / float(np.hypot(*los)))
        want = 6.0 * cos + (3.0 if c.look_closed else 0.0)  # (3 more where the moving jaw hides the fixed one)
        assert abs((blind[az].score - c.score) - want) < 0.02, (az, blind[az].score, c.score, want)
    best, best_blind = plan(sc, mode="pinch", camera_xy=cam_xy)[0], plan(sc, mode="pinch")[0]
    los = np.array(cam_xy) - np.array(best.tip_hover[:2])
    to_cam = math.degrees(math.atan2(los[1], los[0]))
    assert 60.0 <= az_diff(best.jaw_azimuth_deg, to_cam) <= 120.0, (best.jaw_azimuth_deg, to_cam)
    assert not (60.0 <= az_diff(best_blind.jaw_azimuth_deg, to_cam) <= 120.0), "the test scene must show a difference"
    # it is a preference, never a reason to give up room: hemmed in, the only free slot is used all the same
    tight = plan(hemmed_scene(), hollow=True, camera_xy=cam_xy)
    assert tight[0].ok and tight[0].jaw_azimuth_deg == plan(hemmed_scene(), hollow=True)[0].jaw_azimuth_deg
    assert {round(c.jaw_azimuth_deg, 1) for c in tight if c.ok} == \
        {round(c.jaw_azimuth_deg, 1) for c in plan(hemmed_scene(), hollow=True) if c.ok}


def test_geometry_sweep_primitive_matches_brute_force():
    rng = np.random.default_rng(7)
    wall = takeout._Obstacle(9, "wall", RIM, cavity=takeout._rect(BOX_C, math.radians(25.0), 0.105, 0.094), wall=True)
    disc = takeout._Obstacle(2, "disc", 0.0, disc=(0.25, 0.10, 0.013))
    poly = takeout._Obstacle(3, "poly", 0.0, poly=takeout._rect((0.20, 0.06), math.radians(40.0), 0.045, 0.02))
    for ob in (wall, disc, poly):
        for _ in range(40):
            c = np.array(BOX_C) + rng.uniform(-0.02, 0.02, 2)
            a = rng.uniform(0.0, 2.0 * math.pi)
            s, hw = np.array([math.cos(a), math.sin(a)]), rng.uniform(0.004, 0.012)
            t = ob.first_contact(c, s, hw)
            # brute force: march the bar outwards in 0.1 mm steps until one of its points is in the material
            perp = np.array([-s[1], s[0]])
            ws = np.linspace(-hw, hw, 41)
            hit = math.inf
            for k in range(1200):
                pts = c + (k * 1e-4) * s + ws[:, None] * perp
                if float(ob.sdist(pts).min()) <= 0.0:
                    hit = k * 1e-4
                    break
            if math.isinf(hit):
                assert math.isinf(t) or t > 0.119, (ob.label, t)
            else:
                assert abs(t - hit) < 4e-4, (ob.label, t, hit)
    # a rotated container: the free distance is measured along the jaw direction, to the wall actually met
    sc = scene(cyl(1, "white", *BOX_C), take_box(yaw=math.radians(35.0)))
    for c in plan(sc, mode="pinch")[:36]:
        if c.slot.get("free_mm") is None:
            continue
        s = -np.array([math.cos(math.radians(c.jaw_azimuth_deg)), math.sin(math.radians(c.jaw_azimuth_deg))])
        ring = takeout._Obstacle(9, "w", RIM, cavity=takeout._rect(BOX_C, math.radians(35.0), 0.105, 0.094))
        hw = takeout.fixed_half_width(6.8 - (RIM + 0.003 - c.tip_grasp[2]) * 1e3) + 0.002
        assert abs(c.slot["free_mm"] / 1e3 - (ring.first_contact(BOX_C, s, hw) - CAP_R)) < 1e-4


def test_geometry_look_and_reach_hints():
    cam_xy = (0.55, 0.04)
    cands = plan(cap_scene(), mode="pinch", camera_xy=cam_xy)
    for c in cands:
        to_cam = math.degrees(math.atan2(cam_xy[1] - c.tip_hover[1], cam_xy[0] - c.tip_hover[0]))
        d = az_diff(c.jaw_azimuth_deg, to_cam)
        # moving jaw between the camera and the fixed jaw (within about 33 deg of the line of sight): look closed
        assert c.look_closed == (d < 33.0) or 30.0 < d < 36.0, (c.jaw_azimuth_deg, d, c.look_closed)
    assert not cands[0].look_closed, "a direction the camera sees is preferred"
    assert takeout.fixed_jaw_visible(180.0, (0.2, 0.0), cam_xy, 0.014) and not takeout.fixed_jaw_visible(5.0, (0.2, 0.0), cam_xy, 0.014)
    # the arm's say: candidates whose grasp, hover or lift point it cannot reach straight down are dropped
    asked = []

    def reach(xyz, roll_deg):
        asked.append(tuple(xyz))
        return "" if xyz[0] < 0.215 else "shoulder_lift outside its limits"

    cands = plan(cap_scene(), mode="pinch", reach=reach)
    ok = [c for c in cands if c.ok]
    assert ok and all(c.tip_grasp[0] < 0.215 for c in ok) and any("cannot point straight down" in c.reason for c in cands)
    assert max(p[2] for p in asked) >= RIM + 0.010 + H_BELOW - 1e-9  # the lift point was asked about too


def test_geometry_reach_is_asked_above_the_hover_point():
    """N4. The approach arrives straight above the hover point (approach_drop higher): the highest point of
    the way in, and the first the arm cannot hold the gripper straight down at when the container stands
    far out. The planner asks about it, and explain() then gives a next step about reach."""
    hover_z = RIM + 0.012
    asked = []

    def reach_to(r_high):
        def reach(xyz, roll_deg):  # like the real arm: the higher the fingertip, the less far out it reaches
            asked.append(tuple(xyz))
            r = math.hypot(xyz[0] - K._X_PAN, xyz[1])
            high = xyz[2] > hover_z + 0.008  # (above the hover and the lift points of this scene)
            return "" if (r_high[0] <= r <= r_high[1] if high else r <= 0.30) else "wrist_flex outside its limits"
        return reach

    cands = plan(cap_scene(), mode="pinch", reach=reach_to((0.0, 0.18)))  # the cap lies 20 cm out
    assert any(abs(p[2] - (hover_z + 0.015)) < 1e-4 for p in asked), "the point above the hover point is asked about"
    assert not any(c.ok for c in cands)
    out = [c for c in cands if c.reason.startswith("the arm cannot point straight down at the point above the hover point")]
    assert len(out) >= 30 and all(c.score == 0.0 for c in out), len(out)
    ex = takeout.explain(cands)
    assert "have room for the jaws, but the arm cannot point straight down at the point above the hover point" in ex["reason"]
    assert "too far from the robot" in ex["reason"] and ex["next"] == takeout.REACH_NEXT["reach: too far from the robot"], ex
    assert "nearer to the robot" in ex["next"] and "--to" in ex["next"] and "does not help" in ex["next"], ex["next"]
    # the other way round (too close to the robot), and a reach that ends for a reason no shift cures
    ex = takeout.explain(plan(cap_scene(), mode="pinch", reach=reach_to((0.235, 0.40))))
    assert "too close to the robot" in ex["reason"] and "further away from the robot" in ex["next"], ex
    ex = takeout.explain(plan(cap_scene(), mode="pinch", reach=reach_to((9.0, 9.0))))
    assert ex["next"] == takeout.REACH_NEXT[""] and "straight in front of the robot" in ex["next"], ex
    # with no drop onto the hover point there is no such point, and the same reach is enough
    assert any(c.ok for c in plan(cap_scene(), mode="pinch", reach=reach_to((0.0, 0.18)), params={"approach_drop": 0.0}))
    assert takeout.DEFAULTS["approach_drop"] == skills.TakeOptions().approach_drop == 0.015


def test_geometry_choose_place():
    sc = cap_scene(box(5, "red", BOX_C[0] - 0.015, BOX_C[1] - 0.105, size=(0.04, 0.02, 0.02), in_container=None))
    cap, con = sc.by_id(1), sc.by_id(9)
    cables = [BOX_C[0] - 0.035, BOX_C[0] + 0.225, BOX_C[1] - 0.485, BOX_C[1] - 0.22, TABLE_Z - 0.03, TABLE_Z + 0.05]
    cfg = {**PLAN_CFG, "keepouts": {"cables": cables}}
    spots, why = takeout.choose_place(sc, cap, con, cfg)
    assert len(spots) == 8 and why == ""
    outline = takeout._Obstacle(9, "c", RIM, poly=np.array(con.footprint))
    red = takeout._Obstacle(5, "r", 0.0, poly=np.array(sc.by_id(5).footprint))
    for sp in spots:
        x, y = sp["xy"]
        # free: clear of the container and of the other object, outside the keep-out; and near the container
        assert float(outline.sdist((x, y))[0]) >= CAP_R + 0.025 - 1e-9 and float(red.sdist((x, y))[0]) >= CAP_R + 0.025 - 1e-9
        assert not (cables[0] - 0.0255 <= x <= cables[1] + 0.0255 and y <= cables[3] + 0.0255), (x, y)
        assert math.hypot(x - BOX_C[0], y - BOX_C[1]) <= 0.20 + 1e-9
    cost = [math.hypot(sp["xy"][0] - BOX_C[0], sp["xy"][1] - BOX_C[1]) + 0.3 * abs(sp["xy"][1]) for sp in spots]
    assert cost == sorted(cost), "the shortest carry comes first"
    # no table coordinate is built in: the same scene 5 cm further from the robot gives the same spots, 5 cm further
    d = 0.05
    moved = scene(cyl(1, "white", BOX_C[0] + d, BOX_C[1]), take_box(cx=BOX_C[0] + d),
                  box(5, "red", BOX_C[0] - 0.015 + d, BOX_C[1] - 0.105, size=(0.04, 0.02, 0.02), in_container=None))
    cfg_moved = {**PLAN_CFG, "keepouts": {"cables": [cables[0] + d, cables[1] + d] + cables[2:]}}
    spots_moved, _ = takeout.choose_place(moved, moved.by_id(1), moved.by_id(9), cfg_moved)
    assert takeout.DEFAULTS["place_area"] is None
    assert np.allclose([sp["xy"] for sp in spots_moved], np.array([sp["xy"] for sp in spots]) + (d, 0.0), atol=2e-4)
    # an area, when one is given, limits the search; so does the caller's own test, with its reason
    x0 = BOX_C[0] + 0.08
    inside, _ = takeout.choose_place(sc, cap, con, cfg, area=(x0, x0 + 0.5, -1.0, 1.0))
    assert inside and all(sp["xy"][0] >= x0 - 1e-9 for sp in inside)
    asked = []

    def near_only(xy):
        asked.append(xy)
        return "" if xy[0] < BOX_C[0] - 0.08 else "the arm cannot point straight down there"

    picky, _ = takeout.choose_place(sc, cap, con, cfg, ok_xy=near_only, limit=3)
    assert len(picky) == 3 and all(sp["xy"][0] < BOX_C[0] - 0.08 for sp in picky)
    assert len(asked) < 150, "the caller's test is asked only until enough spots are found"
    none, why = takeout.choose_place(sc, cap, con, cfg, ok_xy=lambda xy: False)
    assert none == [] and why.startswith("no free spot on the table within 20 cm of the container") and "reach" in why, why
    none, why = takeout.choose_place(sc, cap, con, cfg, ok_xy=lambda xy: "hidden by the arm")
    assert none == [] and "hidden by the arm (" in why, why
    # a given spot is taken or refused as it is
    free = (BOX_C[0] + 0.035, BOX_C[1] - 0.145)
    assert takeout.choose_place(sc, cap, con, cfg, to=free)[0][0]["xy"] == free
    for to, word in (((BOX_C[0] - 0.01, BOX_C[1] - 0.115), "red box #5"), (BOX_C, "black container #9"),
                     ((BOX_C[0] + 0.025, BOX_C[1] - 0.215), "keep-out 'cables'"), ((float("nan"), 0.0), "not a pair of numbers"),
                     (free, "hidden by the arm")):
        spots, why = takeout.choose_place(sc, cap, con, cfg, to=to, ok_xy=lambda xy: "hidden by the arm")
        assert spots == [] and word in why, (to, why)


def test_geometry_planner_is_fast():
    sc = hemmed_scene()
    plan(sc)
    t0 = time.perf_counter()
    for _ in range(5):
        plan(sc)
    dt = (time.perf_counter() - t0) / 5
    assert dt < 0.15, f"plan_grasps took {dt * 1000:.0f} ms"
    return {"plan_grasps_ms": round(dt * 1000, 1)}


# ----------------------------------------------------------------------------- end to end (simulated arm)
def _sim_arm(cfg=None):
    """A simulated arm with the (pinned) configuration of these tests; cfg: another one."""
    from robot import Arm
    from sim_bus import SimBus
    return Arm(bus=SimBus(realtime=False), cfg=copy.deepcopy(cfg if cfg is not None else load_config()))


def _frame(name):
    import cv2
    img = cv2.imread(os.path.join(HERE, "testdata", name))
    assert img is not None, name
    return img


def test_e2e_dry_blue_block():
    import camera as C
    with _sim_arm() as arm:
        cam = skills.ReplayCamera([_frame("0148_cube_scene_raw.jpg")])
        sess = skills.Session(arm, cam, C.CameraModel.load(), image_dir=None)
        t0 = time.time()
        r = skills.pick_place(sess, {"color": "blue"}, None, skills.PickOptions(dry=True))
        dt = time.time() - t0
    assert r["ok"] and r["stage"] == "planned", {k: r[k] for k in ("stage", "reason", "scene", "grasp_candidates") if k in r}
    g = r["grasp"]
    # the fingertip (fixed jaw pad) stands beside the block: half its width plus the stand-off away
    assert 0.008 < math.hypot(g["tip_grasp"][0] - 0.2783, g["tip_grasp"][1] + 0.0663) < 0.03, g["tip_grasp"]
    assert r["target"]["color"] == "blue" and r["plan"]["standby"] == "above the container", r["plan"]
    assert dt < 1.5, f"perceive + plan took {dt:.2f} s"
    return {"perceive_plan_s": round(dt, 2), "timings": r["timings"]}


def test_e2e_sim_pick_blue_block():
    """Whole sequence on the simulated arm: scene frame before, final frame after."""
    import camera as C
    with _sim_arm() as arm:
        stall = {"raw": None}
        arm.bus.blocker = lambda pos: stall["raw"] is not None and pos["gripper"] < stall["raw"]
        real_grasp = arm.grasp_fast

        def grasp_with_object(expected_raw=None, **kw):
            stall["raw"] = expected_raw  # the simulated jaws stall where the planner expects the object
            return real_grasp(expected_raw=expected_raw, **kw)

        arm.grasp_fast = grasp_with_object
        cam = skills.ReplayCamera([_frame("0148_cube_scene_raw.jpg"), _frame("0168_cube_final_raw.jpg")])
        sess = skills.Session(arm, cam, C.CameraModel.load(), image_dir=None)
        r = skills.pick_place(sess, {"color": "blue"}, None, skills.PickOptions(visual="off", release_wait_s=0.0))
        tip = arm.state()["tip_xyz"]
    assert r["ok"] and r["stage"] == "done" and r["arm_state"] == "standby", \
        {k: r.get(k) for k in ("stage", "reason", "checks", "arm_state")}
    assert math.hypot(tip[0] - r["release"]["tip_release"][0], tip[1] - r["release"]["tip_release"][1]) < 0.01, tip
    return {"timings": r["timings"], "checks": r["checks"]}


# ----------------------------------------------------------------------------- take-out end to end (simulated arm)
def _sim_take(S, query, frames, cam_model=None, ready=True, take_cfg=None, cfg=None, **opt):
    """take_out on the simulated arm (it feels the object, and the surface under an optional touch,
    through skills.sim_take_hooks). cfg: the arm's configuration (default: the tree's own); take_cfg:
    cfg["take"] entries, e.g. {"sim_surface_offset": -0.009} to put the simulated surface 9 mm lower
    than the scene says. Returns (result, final fingertip, joint error to the stand-by pose in deg)."""
    cfg = None if cfg is None else copy.deepcopy(cfg)
    with _sim_arm(cfg) as arm:
        if take_cfg:
            arm.cfg["take"] = dict(take_cfg)
        if ready:  # start torqued at the stand-by pose (skips the simulated unfolding)
            raw = arm.q_to_raw_f(arm.ready_q())
            for j in K.ARM_JOINTS:
                arm.bus.pos[j] = arm.bus.sp[j] = raw[j]
                arm.bus.goal[j] = float(int(round(raw[j])))
            arm.torque_on()
            arm.bus.sleep(0.3)
        if hasattr(S, "arm"):
            S.arm = arm
        sess = skills.Session(arm, skills.ReplayCamera(frames), cam_model, image_dir=None)
        skills.sim_take_hooks(sess)
        hook = sess.on_event

        def on_event(name, info):
            hook(name, info)
            if hasattr(S, "on_event"):
                S.on_event(name, info, arm)

        sess.on_event = on_event
        opt.setdefault("visual", "off")  # no camera sees a simulated arm
        opt.setdefault("release_wait_s", 0.35)
        r = skills.take_out(sess, query, None, skills.TakeOptions(**opt), modules=(takeout, S))
        off = float(np.degrees(np.abs(arm.read_q()[:4] - arm.ready_q()[:4])).max())
        return r, arm.state()["tip_xyz"], off


class _SimScene(FakeTakeScene):
    """FakeTakeScene for the simulated arm (which has no stand-in flags): the object's whereabouts
    follow what the skill announces: held from the closing on, on the table where the jaws opened."""

    state = None

    def on_event(self, name, info, arm):
        if name == "grasp":
            self.state = "jaws"
        elif name == "release":
            tip = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
            self.state = (float(tip[0]), float(tip[1]))

    def _where(self):
        return self.state


def test_e2e_sim_take_out_cap_by_its_wall():
    """Whole sequence on the simulated arm, heights from geometry only: hover above the rim, straight down,
    pinch the wall, lift, carry out, put down, stand-by pose. The scene is given (a cap hemmed in as on
    2026-11-11), with the tree's own calibration."""
    blank = np.zeros((4, 4, 3), np.uint8)
    S = _SimScene(hemmed_scene())
    r, tip, off = _sim_take(S, {"color": "white"}, [blank], assume_hollow=True)  # the request said "cap"
    assert r["ok"] and r["stage"] == "done" and r["arm_state"] == "ready", \
        {k: r.get(k) for k in ("stage", "reason", "checks", "arm_state", "trace")}
    assert r["grasp"]["mode"] == "pinch" and off < 1.5, (r["grasp"]["mode"], off)
    g = r["checks"]["grasp"]
    assert "probe" not in r["checks"] and "probe" not in r["timings"]
    assert abs(r["plan"]["tip_grasp"][2] - (CAP_INNER + TIP_ABOVE)) < 1e-4, r["plan"]
    assert not g["empty"] and abs(g["pos_raw"] - r["grasp"]["expected_raw"]) <= 3, g
    assert r["checks"]["lift"]["held"] and r["checks"]["held_before_release"]["held"]
    assert r["checks"]["verify"]["confirmed"], r["checks"]["verify"]
    assert r["timings"]["total"] < 40.0, r["timings"]
    return {"timings": r["timings"]}


def test_e2e_sim_take_out_with_the_optional_touch():
    """The optional touch on the simulated arm (the real Arm.probe_down on SimBus). As a check (probe="on")
    it changes nothing where the surface is as the scene says, and stops the take-out when something lies
    higher under the jaws. As a measurement (probe_may_lower) the grasp follows a surface 9 mm lower."""
    blank = np.zeros((4, 4, 3), np.uint8)
    keys = ("stage", "reason", "checks", "arm_state", "trace")
    r, _tip, off = _sim_take(_SimScene(cap_scene()), {"color": "white"}, [blank], mode="pinch", probe="on")
    p = r["checks"].get("probe", {})
    assert r["ok"] and off < 1.5 and p.get("mode") == "check" and len(p["tries"]) == 1, {k: r.get(k) for k in keys}
    assert abs(p["tip_grasp_z"] - (CAP_INNER + TIP_ABOVE)) < 1e-4 and r["timings"]["total"] < 40.0, (p, r["timings"])
    lowest = min(z for z in (p["tries"][0]["lowest_tip_z"],) if z is not None)
    assert lowest > p["expected_tip_z"] - 0.004, "the check does not press further than 2 mm past the expected height"
    check = {"timings": r["timings"], "probe": p}
    # something 12 mm higher than the scene says lies under the jaws: stopped before closing, arm at stand-by
    r, _tip, off = _sim_take(_SimScene(cap_scene()), {"color": "white"}, [blank], mode="pinch", probe="on",
                             take_cfg={"sim_surface_offset": 0.012})
    assert not r["ok"] and r["stage"] == "probe" and r["arm_state"] == "ready" and off < 1.5, {k: r.get(k) for k in keys}
    assert "grasp" not in r["checks"] and ("right where it started" in r["reason"] or "above the expected" in r["reason"])
    # as a measurement: the simulated surface lies 9 mm lower than the scene says, the jaws close 2 mm above it
    measure = {"probe": "on", "probe_may_lower": True}
    r, _tip, _off = _sim_take(_SimScene(cap_scene()), {"color": "white"}, [blank], mode="pinch",
                              take_cfg={"sim_surface_offset": -0.009}, **measure)
    p = r["checks"].get("probe", {})
    assert r["ok"] and p.get("mode") == "measure" and -10.5 < p["deviation_mm"] < -7.5, {k: r.get(k) for k in keys}
    assert abs(p["tip_grasp_z"] - (CAP_INNER - 0.009 + TIP_ABOVE)) < 0.0015 and len(p["tries"]) == 1, p
    # the first descent stops short of it (a range of only 8 mm below the expected height): repeated, deeper
    r, _tip, _off = _sim_take(_SimScene(cap_scene()), {"color": "white"}, [blank], mode="pinch",
                              take_cfg={"sim_surface_offset": -0.009}, probe_below=0.008, **measure)
    p2 = r["checks"].get("probe", {})
    assert r["ok"] and len(p2["tries"]) == 2 and p2["tries"][0]["contact"] is not True, {k: r.get(k) for k in keys}
    assert -10.5 < p2["deviation_mm"] < -7.5 and abs(p2["tip_grasp_z"] - p["tip_grasp_z"]) < 0.0015, p2
    # a surface so far off that the grasp would lie below what the floor limit allows: refused, arm at stand-by
    r, _tip, off = _sim_take(_SimScene(cap_scene()), {"color": "white"}, [blank], mode="pinch",
                             take_cfg={"sim_surface_offset": -0.02}, probe_max_dev=0.03, **measure)
    assert not r["ok"] and r["stage"] == "descend" and "cannot be reached" in r["reason"] and off < 1.5, r["reason"]
    assert r["arm_state"] == "ready" and "grasp" not in r["checks"]
    return {"check": check, "measure": {"probe": p, "repeat": p2["tries"]}}


def test_e2e_sim_take_out_looks_and_corrects():
    """The look at the jaws on the simulated arm, with stand-in readings: the fixed jaw hidden at first
    (jaws closed for a second look), then 5 mm off; the descent is shifted and the jaws re-opened."""
    blank = np.zeros((4, 4, 3), np.uint8)
    partial = {"ok": True, "confidence": 0.8, "dxy_base": (0.0, -0.0005), "fixed": {"visible": False}}
    S = _SimScene(cap_scene(), jaws=[partial, jaws(4.0, -3.0), jaws(4.1, -2.9)])
    r, _tip, off = _sim_take(S, {"color": "white"}, [blank], mode="pinch", visual="on")
    v = r["checks"]["visual"]
    assert r["ok"] and off < 1.5, {k: r.get(k) for k in ("stage", "reason", "checks", "trace")}
    assert v["closed"] and v["applied"] and v["corr_mm"] == [-4.0, 3.0] and S.n_jaws == 3, v
    assert min(S.look_z) >= RIM + 0.012 - 0.0015, S.look_z  # every look from above the rim
    assert not r["checks"]["grasp"]["empty"] and r["checks"]["verify"]["confirmed"]
    return {"timings": r["timings"], "visual": v}


def test_e2e_sim_take_out_folds_back_for_a_clear_view():
    blank = np.zeros((4, 4, 3), np.uint8)
    behind = scene()
    behind.warnings.append("a dark box-like region has 1 corner(s) hidden by the robot arm and is not reported as a "
                           "container: move the arm away and look again")
    S = _SimScene(cap_scene(), behind_arm=behind)
    r, _tip, off = _sim_take(S, {"color": "white"}, [blank], mode="pinch")
    assert r["ok"] and "folded back" in r["checks"]["second_look"] and off < 1.5, \
        {k: r.get(k) for k in ("stage", "reason", "checks", "trace")}
    assert r["timings"]["total"] < 40.0, r["timings"]
    return {"timings": r["timings"]}


def test_e2e_sim_take_out_two_sided_from_parked():
    blank = np.zeros((4, 4, 3), np.uint8)
    sc = scene(cyl(1, "ivory", *BOX_C, d=0.026, h=0.022, hollow=False), take_box())
    r, _tip, off = _sim_take(_SimScene(sc), {"color": "ivory"}, [blank], ready=False)
    assert r["ok"] and r["grasp"]["mode"] == "two_sided" and "probe" not in r["checks"] and off < 1.5, \
        {k: r.get(k) for k in ("stage", "reason", "checks", "arm_state", "trace")}
    assert r["timings"]["total"] < 40.0
    return {"timings": r["timings"]}


def _own_calibration():
    """The configuration this tree has NOW, whatever the tests of this file are pinned to."""
    import robot
    return load_config(robot.CONFIG_PATH)


def _on_table(sc, table_z):
    """The same scene standing on a table at another height."""
    dz = float(table_z) - float(sc.table_z)
    out = copy.deepcopy(sc)
    out.table_z = float(table_z)
    for o in out.objects:
        o.z_bottom += dz
        if o.inner:
            o.inner = {**o.inner, "floor_z": o.inner["floor_z"] + dz, "rim_z": o.inner["rim_z"] + dz}
    return out


def test_e2e_sim_take_out_with_the_trees_own_calibration():
    """The take-out on the simulated arm with the calibration this tree uses now (zero ticks, table height,
    floor limit, keep-outs, stand-by pose), not the pinned one: the same synthetic scenes, standing on that
    calibration's table. Heights come from that table alone; no number of an older calibration is involved.
    (In a tree whose calibration is still the recorded one this repeats the tests above.)"""
    cfg = _own_calibration()
    tz = float(cfg["table_z"])
    blank = np.zeros((4, 4, 3), np.uint8)
    keys = ("stage", "reason", "checks", "arm_state", "trace", "next")
    # the cap hemmed in on three sides: by its wall, from the stand-by pose
    r, _tip, off = _sim_take(_SimScene(_on_table(hemmed_scene(), tz)), {"color": "white"}, [blank], cfg=cfg,
                             assume_hollow=True)
    assert r["ok"] and r["stage"] == "done" and r["arm_state"] == "ready" and off < 1.5, {k: r.get(k) for k in keys}
    assert r["grasp"]["mode"] == "pinch" and "probe" not in r["checks"], r["grasp"]["mode"]
    # grasp height = this table + the container's floor (5 mm) + the cap's own bottom (2 mm) + 2 mm
    assert abs(r["plan"]["tip_grasp"][2] - (tz + 0.005 + 0.002 + TIP_ABOVE)) < 1e-4, (r["plan"]["tip_grasp"], tz)
    # put down with the cap's bottom 3 mm above this table
    assert abs(r["plan"]["tip_place"][2] - (tz + H_BELOW + 0.003)) < 1e-4, (r["plan"]["tip_place"], tz)
    assert r["checks"]["verify"]["confirmed"] and r["timings"]["total"] < 40.0, r["timings"]
    pinch = r["timings"]
    # a solid cylinder across, from the parked pose
    sc = scene(cyl(1, "ivory", *BOX_C, d=0.026, h=0.022, hollow=False), take_box())
    r, _tip, off = _sim_take(_SimScene(_on_table(sc, tz)), {"color": "ivory"}, [blank], cfg=cfg, ready=False)
    assert r["ok"] and r["grasp"]["mode"] == "two_sided" and off < 1.5, {k: r.get(k) for k in keys}
    assert abs(r["plan"]["tip_grasp"][2] - (tz + 0.005 + 0.022 - 0.015)) < 1e-4 and r["timings"]["total"] < 40.0
    return {"table_z": tz, "shoulder_lift_zero": cfg["zero_ticks"]["shoulder_lift"], "pinch": pinch,
            "two_sided_from_parked": r["timings"]}


class _CapAsItIs:
    """The real scene module, except that the white thing inside the container is described as what it
    is: the bottle cap of 2026-11-10/11 (31 mm across, 16.5 mm high, open side up: hollow), at the place
    where perception sees it (in tests/testdata/0146_cap_final_raw.jpg the blob's centre is within 2 mm of the
    cap's rim centre). A scene.py without SceneObject.hollow reports it as an "unknown" 44 x 32 mm blob with
    a guessed height; one with it describes the cap itself (test_e2e_sim_take_out_on_saved_frames_as_perceived)."""

    def __init__(self):
        import scene as real
        self.real = real

    def __getattr__(self, name):
        return getattr(self.real, name)

    def perceive(self, img, cam, cfg, **kw):
        sc = self.real.perceive(img, cam, cfg, **kw)
        for o in sc.objects:
            if o.in_container is not None and o.color == "white":
                x, y = o.center
                o.kind, o.yaw, o.size, o.confidence = "cylinder", 0.0, (2 * CAP_R, 2 * CAP_R, 0.0165), 0.6
                o.hollow = True
                o.footprint = [(x + CAP_R * math.cos(2 * math.pi * k / 16), y + CAP_R * math.sin(2 * math.pi * k / 16))
                               for k in range(16)]
        return sc


def test_e2e_sim_take_out_on_saved_frames():
    """Real camera frames and the real perception around a simulated arm: the cap lies in the box in the
    first frame and stands on the table in the second (the same scene before it was put in), so the
    container, the contents check, the look at its old place, the choice of a place the camera can see
    and the final verification all run on real images. The cap's own shape is supplied (see _CapAsItIs).
    Recorded frames: they need the calibration they were recorded with (pinned at the top of this file)."""
    import camera as C
    import scene as real_scene
    cfg, cam_model = load_config(), C.CameraModel.load()
    frames = [_frame("0146_cap_final_raw.jpg"), _frame("0134_cap_scene_raw.jpg")]
    # where the cap stands on the table in the second frame, as the camera sees it: it is "put down" there
    on_table = [o for o in real_scene.perceive(frames[1], cam_model, cfg).graspable() if o.color == "white"]
    assert len(on_table) == 1, on_table
    spot = (round(float(on_table[0].center[0]), 4), round(float(on_table[0].center[1]), 4))
    r, tip, off = _sim_take(_CapAsItIs(), {"color": "white"}, frames, cam_model=cam_model, cfg=cfg, to=spot,
                            release_wait_s=0.0)
    assert r["ok"] and r["stage"] == "done" and r["arm_state"] == "ready" and off < 1.5, \
        {k: r.get(k) for k in ("stage", "reason", "checks", "arm_state", "trace", "next")}
    ver = r["checks"]["verify"]
    assert ver["confirmed"] and ver["on_table"]["color"] == "white" and ver["on_table"]["dist_mm"] < 10.0, ver
    assert ver["place_in_container_empty"] or ver["inside"]["less"], ver
    assert r["checks"]["lift"]["camera"]["gone"] is True, r["checks"]["lift"]
    assert r["container"]["id"] is not None and r["grasp"]["mode"] in ("pinch", "two_sided")
    # the grasp height came from the scene alone: 2 mm above the cap's inner bottom
    if r["grasp"]["mode"] == "pinch":
        bottom = float(r["grasp"]["z_surface"])  # the inner bottom the plan was made for
        assert abs(r["plan"]["tip_grasp"][2] - (bottom + TIP_ABOVE)) < 1e-4, r["plan"]
        geometric = float(cfg["table_z"]) + 0.005 + 0.002  # table + the container's floor + the cap's own bottom
        seen = [getattr(o, "z_inner_bottom", None) for o in real_scene.perceive(frames[0], cam_model, cfg).objects
                if o.in_container is not None and o.color == "white"]
        if seen and seen[0] is not None:
            # perception says where the inner bottom is (SceneObject.z_inner_bottom): that is what is used,
            # and it lies within what this cap tolerates on its tight side (3 mm) of the geometric height
            assert abs(bottom - float(seen[0])) < 5e-4, (bottom, seen)
            assert abs(bottom - geometric) < 0.003, (bottom, geometric)
        else:  # it does not say (scene.py without that field): the geometric height itself
            assert abs(bottom - geometric) < 1e-5, (bottom, geometric)
    assert "probe" not in r["checks"]
    return {"timings": r["timings"], "mode": r["grasp"]["mode"], "verify": {k: ver[k] for k in ("on_table", "inside")}}


def test_e2e_sim_take_out_on_saved_frames_as_perceived():
    """The same two recorded frames with the perception as it is: no shape supplied, nobody says "hollow".
    A scene module that says what is hollow (SceneObject.hollow) describes the cap in the box as a cylinder
    that is open at the top, and the one call takes it out by its wall. The one before it sees an "unknown"
    blob: the call stops at the plan stage with a reason and a next step, the arm where it was.
    Recorded frames: they need the calibration they were recorded with (pinned at the top of this file)."""
    import camera as C
    import fdtypes
    import scene as real_scene
    cfg, cam_model = load_config(), C.CameraModel.load()
    frames = [_frame("0146_cap_final_raw.jpg"), _frame("0134_cap_scene_raw.jpg")]
    on_table = [o for o in real_scene.perceive(frames[1], cam_model, cfg).graspable() if o.color == "white"]
    assert len(on_table) == 1, on_table
    spot = (round(float(on_table[0].center[0]), 4), round(float(on_table[0].center[1]), 4))
    r, tip, off = _sim_take(real_scene, {"color": "white"}, frames, cam_model=cam_model, cfg=cfg, to=spot,
                            release_wait_s=0.0)
    if not hasattr(fdtypes.SceneObject, "hollow"):
        assert not r["ok"] and r["stage"] == "plan" and r["reason"] and r.get("next") and off < 1.5, \
            {k: r.get(k) for k in ("ok", "stage", "reason", "next", "arm_state")}
        return {"stage": r["stage"], "reason": r["reason"][:200], "next": r.get("next")}
    assert r["ok"] and r["stage"] == "done" and r["arm_state"] == "ready" and off < 1.5, \
        {k: r.get(k) for k in ("stage", "reason", "checks", "arm_state", "trace", "next")}
    assert r["grasp"]["mode"] == "pinch" and not r["grasp"]["unconfirmed"], r["grasp"]  # the scene said: hollow
    ver = r["checks"]["verify"]
    assert ver["confirmed"] and ver["on_table"]["color"] == "white" and ver["on_table"]["dist_mm"] < 10.0, ver
    assert r["checks"]["lift"]["camera"]["gone"] is True and "probe" not in r["checks"], r["checks"]
    return {"timings": r["timings"], "mode": r["grasp"]["mode"], "jaw_azimuth_deg": r["grasp"]["jaw_azimuth_deg"]}


def test_e2e_take_out_dry_on_a_saved_frame_as_perceived():
    """What the take-out makes of a saved frame with the perception as it is: either a plan, or a refusal
    at the perceive / plan stage with a reason and a next step. Never a crash, never a motion.
    Recorded frame: it needs the calibration it was recorded with (pinned at the top of this file)."""
    import camera as C
    with _sim_arm() as arm:
        cam = skills.ReplayCamera([_frame("0146_cap_final_raw.jpg")])
        sess = skills.Session(arm, cam, C.CameraModel.load(), image_dir=None)
        t0 = time.time()
        r = skills.take_out(sess, {"color": "white"}, None, skills.TakeOptions(dry=True))
        dt = time.time() - t0
        moved = len(arm.bus.log)
    assert "trace" not in r and moved == 0, r.get("trace")
    assert r["stage"] in ("planned", "plan") and (r["ok"] or (r["reason"] and r["arm_state"] == "parked")), r
    assert r["ok"] or r.get("next"), r
    assert dt < 1.5, f"perceive + plan took {dt:.2f} s"
    return {"stage": r["stage"], "reason": r["reason"][:200], "next": r.get("next"), "perceive_plan_s": round(dt, 2)}


# ----------------------------------------------------------------------------- runner
def main(argv):
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
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:600]}")
            if verbose:
                traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
