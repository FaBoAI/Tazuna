"""High-level skills: ONE call = one complete, self-checking pick-and-place.

Why this exists: with one LLM tool call per primitive, a pick-and-place took 23-28 calls and
60-70 % of the wall time was the LLM thinking between them. Here the whole sequence
(perceive -> plan -> approach -> check -> grasp -> verify -> transfer -> release -> verify)
runs inside one process with numeric checks after every stage. The LLM is needed only to
choose the object when the request is ambiguous and when a check fails; the result says
exactly which stage stopped, why, and in what state the arm was left.

    pick_place(sess, target_query, container_query, opts)   table -> container
    take_out(sess, target_query, container_query, opts)     container -> table (second half of this file)

Principles
- Fail before moving: every path of the sequence is planned and checked (IK, joint limits,
  floor, keep-outs) before the first motion command.
- A failed check with nothing in the jaws ends at the stand-by pose; with an object in the
  jaws, or after a motion fault, the arm stops where it is and holds (the caller decides).
- Nothing here talks to hardware directly: motion goes through robot.Arm (same safety checks
  as the low-level commands), perception through scene.py, planning through grasp.py.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

import gripper_model as GM
import kinematics as K
from robot import MotionAborted, SafetyError
from servo_bus import BusError

MOTION_ERRORS = (SafetyError, MotionAborted, BusError)


# ----------------------------------------------------------------------------- options
@dataclass
class PickOptions:
    tier: str | None = None  # speed tier name; None = cfg["speed_tier"]
    # look at the jaws at hover before descending: "off" = never; "on" = always, correct the grasp
    # position by what is seen, stop if the jaws cannot be seen; "auto" = always look and record,
    # but correct (and insist on seeing them) only when the grasp is tight
    visual: str = "auto"
    visual_apply_when_roomy: bool = False  # "auto": also correct roomy grasps (enable once detect_jaws is trusted)
    end: str = "ready"  # "ready" | "park"
    retries: int = 1  # extra attempts after an empty grasp
    dry: bool = False  # perceive + plan only (nothing moves, torque untouched)
    any_match: bool = False  # several objects match the query: take the easiest instead of asking
    hover_margin: float = 0.012  # fingertip height above the tallest nearby object at hover (m)
    approach_drop: float = 0.015  # the last part of the approach is a vertical descent of this length
    carry_clearance: float = 0.015  # gap between the carried object and anything it passes over
    standby: str = "container"  # where the arm waits after the release: "container" (straight above it) | "ready"
    standby_above_rim: float = 0.065  # fingertip height above the rim there (the camera sees past the gripper)
    transit_clearance: float = 0.02  # free-space moves keep the fingertip this far above everything
    max_visual_corr: float = 0.010  # largest xy correction accepted from the visual check (m)
    min_visual_conf: float = 0.7
    visual_iters: int = 2  # looks at the jaws for a tight grasp (the second confirms the correction)
    visual_tol: float = 0.002  # residual below which no (further) correction is made (m)
    visual_repeat_tol: float = 0.004  # two looks may differ by at most this much (m)
    tight_clearance: float = 0.008  # free space below which a grasp counts as tight (positioning is good to ~5 mm)
    stall_tol_ticks: int = 110  # |stall position - expected| above which the grasp is suspicious
    release_wait_s: float = 0.35
    verify: bool = True
    max_temp_c: float = 55.0
    save_images: bool = True


def arm_clock(arm):
    """Seconds as the real robot would experience them. On a simulated bus that does not run
    in real time, the simulated motion time is added to the wall time spent computing, so the
    reported timings are an estimate of the real duration."""
    bus = getattr(arm, "bus", None)
    if bus is not None and hasattr(bus, "now") and not getattr(bus, "realtime", False):
        return lambda: time.perf_counter() + float(bus.now)
    return time.perf_counter


def arm_sleep(arm, dt: float):
    """Wait dt seconds of robot time (simulated time on a simulated bus)."""
    if dt > 0:
        getattr(arm, "_sleep", time.sleep)(dt)


class Timer:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.t0 = clock()
        self.last = self.t0
        self.marks = {}

    def mark(self, name: str):
        now = self.clock()
        self.marks[name] = round(self.marks.get(name, 0.0) + now - self.last, 3)
        self.last = now

    def total(self) -> float:
        return round(self.clock() - self.t0, 3)


class StageError(RuntimeError):
    """A stage of a skill could not be completed; .stage and .detail say which and why."""

    def __init__(self, stage: str, reason: str, detail: dict | None = None):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason, self.detail = stage, reason, detail or {}


# ----------------------------------------------------------------------------- session
class Session:
    """Owns the arm and the camera for the duration of one command.

    camera: an object with latest(n_avg=1) and wait_fresh(after_time) (fastcam.FastCamera, or a
    stand-in that replays saved frames in simulation). cam_model: camera.CameraModel.
    image_dir: where diagnostic frames are written (None = do not save)."""

    def __init__(self, arm, camera, cam_model, image_dir: str | None = None, log=None):
        self.arm = arm
        self.camera = camera
        self.cam = cam_model
        self.image_dir = image_dir
        self.log = log or (lambda *a, **k: None)
        self.images = {}
        self._n = 0
        self.known_containers = []  # containers seen with full confidence earlier in this session

    def frame(self, fresh: bool = False, n_avg: int = 1):
        """Newest camera frame. fresh=True returns a frame exposed after this call (use it
        after a motion: an averaged or buffered frame could still show the arm moving)."""
        if fresh and hasattr(self.camera, "wait_fresh"):
            return self.camera.wait_fresh(time.time())
        return self.camera.latest(n_avg=n_avg)

    def save(self, tag: str, img) -> str | None:
        if self.image_dir is None or img is None:
            return None
        import cv2
        os.makedirs(self.image_dir, exist_ok=True)
        self._n += 1
        path = os.path.join(self.image_dir, f"{time.strftime('%H%M%S')}_{self._n:02d}_{tag}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.images[tag] = path
        return path


class ReplayCamera:
    """Camera stand-in for simulation and tests: returns the given frames in order. latest()
    repeats the current frame; every wait_fresh() advances to the next (the last one repeats)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0

    def latest(self, n_avg: int = 1, max_age_s: float = 0.2):
        return self.frames[min(self.i, len(self.frames) - 1)]

    def wait_fresh(self, after_time: float):
        self.i += 1
        return self.latest()

    def close(self):
        pass


# ----------------------------------------------------------------------------- helpers
def _feasible_pitch(arm, xyz, roll_deg: float, want: float) -> float:
    """Feasible approach pitch nearest to `want` at this fingertip position."""
    tool, lim = arm.tip_tool(), arm.q_limits()
    best = getattr(K, "best_pitch", None)
    if best is not None:
        p, _degraded = best(xyz, want=want, wrist_roll=math.radians(roll_deg), tool=tool, limits=lim)
        if p is None:
            raise SafetyError(f"fingertip position {np.round(xyz, 4).tolist()} is out of reach")
        return float(p)
    rng = K.pitch_range(xyz, math.radians(roll_deg), tool=tool, limits=lim)
    if rng is None:
        raise SafetyError(f"fingertip position {np.round(xyz, 4).tolist()} is out of reach")
    lo, hi = rng[0] + 1.0, rng[1] - 1.0
    return float(max(lo, min(hi, want))) if lo <= hi else float(rng[0] + rng[1]) / 2.0


def _tip_path(arm, points, q_start) -> list:
    """Waypoints of straight fingertip segments through `points`, WITHOUT the start pose."""
    wps = list(arm.plan_tip_path(points, q_start=q_start))
    if wps and np.allclose(wps[0], q_start, atol=1e-9):
        wps = wps[1:]
    return wps


def check_path(arm, wps, speed: float, opts, ignore=(), gripper_raw=None, what: str = "path"):
    """The checks run_path will make for this path (every 1.5 deg of it and of its corner-blended
    reference, with the jaw opening the gripper will have), without moving. Raises SafetyError."""
    if len(wps) < 2:
        return
    if hasattr(arm, "check_path"):
        arm.check_path(wps, speed, ignore_keepouts=ignore, gripper_goal_raw=gripper_raw, tier=opts.tier)
    else:  # minimal stand-in arms (tests): the waypoints only
        for q in wps[1:]:
            arm.check_q(q, ignore_keepouts=ignore, what=what, q_start=wps[0])


def scene_keepouts(sc, skip_ids=(), margin: float = 0.006) -> dict:
    """Axis-aligned keep-out boxes (fingertip may not enter) for every object except skip_ids:
    the object's footprint grown by `margin`, from below the table up to its top + margin."""
    out = {}
    for o in sc.objects:
        if o.id in skip_ids or not o.footprint:
            continue
        xs = [p[0] for p in o.footprint]
        ys = [p[1] for p in o.footprint]
        out[f"obj{o.id}"] = [min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin,
                             sc.table_z - 0.05, o.z_top + margin]
    return out


def container_keepouts(sc, container) -> tuple:
    """Names of the keep-outs that must be ignored while reaching into a container: its own box
    and the boxes of everything already lying inside it."""
    return (f"obj{container.id}",) + tuple(f"obj{o.id}" for o in sc.objects if o.in_container == container.id)


def _is_parked(q) -> bool:
    """True when the arm is folded back on its rest (lift far back)."""
    return math.degrees(float(q[1])) < -70.0


def standby_q(arm, q_now) -> np.ndarray:
    """The stand-by pose reached from q_now: the configured ready pose, but with the wrist roll
    left as it is. Rolling the wrist back to zero after every release (and forward again for
    the next grasp) was the longest single part of a pick; the roll does not matter there."""
    q = np.array(arm.ready_q(), dtype=float)
    q[4] = float(q_now[4])
    return q


def _arm_state(arm, q) -> str:
    if not arm.is_torqued():
        return "parked" if _is_parked(q) else "limp"
    d = np.asarray(q, dtype=float)[:4] - np.asarray(arm.ready_q(), dtype=float)[:4]  # roll is free at stand-by
    if float(np.degrees(np.abs(d)).max()) < 4.0:
        return "ready"
    tip_z = float(K.fk(np.asarray(q, dtype=float), arm.tip_tool())[2, 3])
    return "standby" if tip_z > float(arm.cfg.get("table_z", 0.0)) + 0.07 else "stopped"


def plan_transit(arm, q_from, q_to, z_clear: float, ignore=()) -> list:
    """Joint waypoints (without q_from) of a free-space move from q_from to q_to.

    A fingertip that starts low is first raised straight up. Then the move goes directly in
    joint space when every sample of it passes the safety check and the fingertip never dips
    below the lower of its two ends (or below z_clear, when both ends are above that);
    otherwise it goes through the ready pose. Raises SafetyError if neither way is clear."""
    tool = arm.tip_tool()
    q = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    wps = []
    tip = K.fk(q, tool)[:3, 3]
    z_to = float(K.fk(q_to, tool)[2, 3])
    z_lift = min(z_clear, z_to)
    if tip[2] < z_lift - 0.005:
        top = np.array([tip[0], tip[1], z_lift])
        roll = math.degrees(float(q[4]))
        pitch = _feasible_pitch(arm, top, roll, float(K.pose_summary(q, tool)["pitch_down_deg"]))
        wps += _tip_path(arm, [(top, pitch, roll)], q)
        q = wps[-1]

    def clear(a, b):
        floor = min(float(K.fk(a, tool)[2, 3]), float(K.fk(b, tool)[2, 3]), z_clear) - 0.005
        for u in np.linspace(0.0, 1.0, 15)[1:]:
            qs = a + u * (b - a)
            arm.check_q(qs, ignore_keepouts=ignore, what="transit")
            z = float(K.fk(qs, tool)[2, 3])
            if z < floor:
                raise SafetyError(f"the transit would dip to z = {z:.3f} m (limit {floor:.3f})")

    try:
        clear(q, q_to)
        return wps + [q_to]
    except SafetyError:
        r = np.asarray(arm.ready_q(), dtype=float)
        clear(q, r)
        clear(r, q_to)
        return wps + [r, q_to]


def _summarise(sc) -> list:
    return [{"id": o.id, "kind": o.kind, "color": o.color,
             "center": [round(float(v), 4) for v in o.center],
             "size_mm": [round(float(v) * 1000, 1) for v in o.size],
             "yaw_deg": round(math.degrees(o.yaw), 1), "in_container": o.in_container,
             "confidence": round(float(o.confidence), 2), "notes": list(o.notes)} for o in sc.objects]


def _gripper_raw(arm) -> int:
    return int(arm.bus.read("gripper", "Present_Position"))


def _color_is(o, color: str) -> bool:
    """Colours outside the perception's basic names come back as "other" with a note such as
    "colour: orange"."""
    return o.color == color or (o.color == "other" and any(f"colour: {color}" in str(n) for n in (o.notes or [])))


def select_objects(sc, query: dict, containers: bool = False, min_confidence: float = 0.5) -> list:
    """Objects of the scene matching `query` (keys: id, color, kind, nearest_to, largest).
    Loose objects on the table by default, containers with containers=True. Raises LookupError
    with a reason when nothing matches. More than one result = the request is ambiguous.

    Loose objects whose confidence is below min_confidence have only estimated geometry (base
    hidden, outline only, flat or oversized): they are never chosen by colour or shape, only
    by an explicit id."""
    query = query or {}
    pool = sc.containers() if containers else sc.graspable()
    what = "container" if containers else "object"
    if query.get("id") is not None:
        hit = [o for o in sc.objects if o.id == query["id"]]
        if not hit:
            raise LookupError(f"no object with id {query['id']} in view")
        o = hit[0]
        if not containers and o.kind == "container":
            raise LookupError(f"object {o.id} is a container")
        if not containers and o.in_container is not None:
            raise LookupError(f"object {o.id} is already inside container {o.in_container}")
        if containers and o.kind != "container":
            raise LookupError(f"object {o.id} is not a container")
        return [o]
    cands = list(pool)
    if query.get("color"):
        cands = [o for o in cands if _color_is(o, query["color"])]
    if query.get("kind") and not containers:
        cands = [o for o in cands if o.kind == query["kind"]]
    if not containers:
        sure = [o for o in cands if float(o.confidence) >= min_confidence]
        if cands and not sure:
            unsure = "; ".join(f"id {o.id} ({o.color} {o.kind}: {', '.join(str(n) for n in o.notes[:2]) or 'uncertain'})"
                               for o in cands[:4])
            raise LookupError(f"only uncertain detections match {query}: {unsure}. If one of them is right, "
                              f"pick it by id")
        cands = sure
    if not cands:
        seen = ", ".join(f"{o.id}:{o.color} {o.kind}" for o in pool) or "none"
        raise LookupError(f"no {what} matches {query} (in view: {seen})")
    if query.get("nearest_to") is not None:
        x, y = query["nearest_to"]
        cands = sorted(cands, key=lambda o: math.hypot(o.center[0] - x, o.center[1] - y))[:1]
    elif query.get("largest"):
        cands = sorted(cands, key=lambda o: -o.size[0] * o.size[1])[:1]
    return cands


# ----------------------------------------------------------------------------- motion plan
@dataclass
class MotionPlan:
    """Everything the pick sequence will command, planned and checked before it moves."""
    target: object
    cand: object  # fdtypes.GraspCandidate
    rel: object  # fdtypes.ReleasePlan
    hover: np.ndarray
    p_hover: float
    above_hover: np.ndarray
    p_above: float
    tip_grasp: np.ndarray
    z_carry: float
    approach: list = field(default_factory=list)  # joint waypoints: above-hover ... hover
    descent: list = field(default_factory=list)  # hover ... grasp
    carry: list = field(default_factory=list)  # grasp ... lift ... above container ... release
    retreat: list = field(default_factory=list)  # release ... straight up (to the stand-by point above the container)
    standby_over: bool = False  # the retreat ends at a stand-by point above the container (else: go to ready)
    z_clear: float = 0.0  # height above which free-space moves are clear of everything in the scene
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"hover": [round(float(v), 4) for v in self.hover], "pitch_hover": round(self.p_hover, 1),
                "tip_grasp": [round(float(v), 4) for v in self.tip_grasp], "z_carry": round(self.z_carry, 4),
                "waypoints": {"approach": len(self.approach), "descent": len(self.descent),
                              "carry": len(self.carry), "retreat": len(self.retreat)},
                "standby": "above the container" if self.standby_over else "ready pose",
                "notes": list(self.notes)}


def build_plan(arm, sc, target, container, cand, rel, opts: PickOptions, corr=(0.0, 0.0), q_hover=None) -> MotionPlan:
    """Plan and check every path of the sequence for one grasp candidate. `corr` shifts the
    grasp position in xy (visual correction); q_hover = measured pose at hover, when the arm
    already stands there. Raises SafetyError when any part is infeasible. No motion."""
    box_ko = container_keepouts(sc, container)
    roll = float(cand.roll_deg)
    corr3 = np.array([float(corr[0]), float(corr[1]), 0.0])
    near = [o.z_top for o in sc.objects if o.id != target.id and o.kind != "container"
            and math.hypot(o.center[0] - target.center[0], o.center[1] - target.center[1]) < 0.09]
    hover = np.array(cand.tip_hover, dtype=float)
    z_min = max(near + [target.z_top]) + opts.hover_margin
    p_hover = float(cand.pitch_deg)  # the planner checked its own hover pose at the grasp pitch
    if hover[2] < z_min - 1e-9:
        hover[2] = z_min
        p_hover = _feasible_pitch(arm, hover, roll, cand.pitch_deg)
        if abs(p_hover - float(cand.pitch_deg)) > 3.0:
            # the planner's clearances hold for a descent at ONE pitch; a hover pose that needs a
            # clearly different pitch would rotate the gripper on the way down (3 deg moves the
            # jaw noses by about half a millimetre: harmless)
            raise SafetyError(f"the raised hover point needs pitch {p_hover:.0f} deg instead of {cand.pitch_deg:.0f}")
    above_hover = hover + np.array([0.0, 0.0, opts.approach_drop])
    p_above = _feasible_pitch(arm, above_hover, roll, p_hover)
    tip_grasp = np.array(cand.tip_grasp, dtype=float) + corr3
    plan = MotionPlan(target=target, cand=cand, rel=rel, hover=hover, p_hover=p_hover, above_hover=above_hover,
                      p_above=p_above, tip_grasp=tip_grasp, z_carry=0.0)

    # approach: arrive from straight above (the caller prepends the transit from where the arm is)
    q_above = arm.solve(above_hover, p_above, roll)
    plan.approach = [q_above] + _tip_path(arm, [(hover, p_hover, roll)], q_above)
    q_h = plan.approach[-1] if q_hover is None else np.asarray(q_hover, dtype=float)

    # descent: (sideways by the visual correction at hover height, then) straight down
    pts = [(tip_grasp, cand.pitch_deg, roll)]
    if float(np.linalg.norm(corr3)) > 1e-6:
        pts.insert(0, (hover + corr3, p_hover, roll))
    plan.descent = _tip_path(arm, pts, q_h)
    q_grasp = plan.descent[-1]

    # carry: straight up, across in joint space (shortest and fastest; within a millimetre of a
    # level line for these distances), straight down to the release point
    above = np.array(rel.tip_above, dtype=float)
    release = np.array(rel.tip_release, dtype=float)
    h_below = float(tip_grasp[2] - target.z_bottom)  # how far the object hangs below the fingertip
    others = [o.z_top for o in sc.objects if o.id != target.id and o.kind != "container"]
    z_carry = max([above[2]] + [z + h_below + opts.carry_clearance for z in others])
    if z_carry > above[2] + 1e-6:
        plan.notes.append(f"carry height raised {1000 * (z_carry - above[2]):.0f} mm to clear other objects")
        above[2] = z_carry
    plan.z_carry = float(z_carry)
    lift = np.array([tip_grasp[0], tip_grasp[1], z_carry])
    p_lift = _feasible_pitch(arm, lift, roll, cand.pitch_deg)
    up = _tip_path(arm, [(lift, p_lift, roll)], q_grasp)
    q_over = arm.solve(above, rel.pitch_deg, rel.roll_deg, q_seed=up[-1])
    down = _tip_path(arm, [(release, rel.pitch_deg, rel.roll_deg)], q_over)
    plan.carry = up + [q_over] + down
    plan.z_clear = max(o.z_top for o in sc.objects) + opts.transit_clearance
    # after the release: straight up to a stand-by point above the container, high enough for the
    # camera to see the whole inside past the gripper. The next pick starts from there, which
    # saves the long way to the ready pose and back.
    if opts.standby == "container":
        rim_z = float((container.inner or {}).get("rim_z", container.z_top))
        for dz in (opts.standby_above_rim, 0.05, 0.04):
            top = np.array([release[0], release[1], max(above[2], rim_z + dz)])
            try:
                p_top = _feasible_pitch(arm, top, rel.roll_deg, rel.pitch_deg)
                path = _tip_path(arm, [(top, p_top, rel.roll_deg)], down[-1])
                for q in path:
                    arm.check_q(q, ignore_keepouts=box_ko, what="retreat")
            except SafetyError:
                continue
            plan.retreat, plan.standby_over = path, True
            if dz < opts.standby_above_rim:
                plan.notes.append(f"stand-by point only {1000 * dz:.0f} mm above the rim (reach)")
            break
    if not plan.standby_over:
        plan.retreat = _tip_path(arm, [(above, rel.pitch_deg, rel.roll_deg)], down[-1])

    # checks: exactly what run_path will check when it executes (corner blending included), so a
    # plan that passes here is not refused half-way through the sequence
    sp = arm.speeds(opts.tier)
    tool = arm.tip_tool()
    check_path(arm, plan.approach, sp["transit"], opts, gripper_raw=cand.open_raw, what="approach")
    check_path(arm, [q_h] + plan.descent, sp["approach"], opts, gripper_raw=cand.open_raw, what="descent")
    check_path(arm, [q_grasp] + plan.carry, sp["transit"], opts, ignore=box_ko, gripper_raw=cand.expected_raw,
               what="carry")
    check_path(arm, [down[-1]] + plan.retreat, sp["transit"], opts, ignore=box_ko, gripper_raw=rel.open_raw,
               what="retreat")
    for s in np.linspace(0.0, 1.0, 13)[1:-1]:
        z = float(K.fk(up[-1] + s * (q_over - up[-1]), tool)[2, 3])
        if z < z_carry - 0.006:
            raise SafetyError(f"the crossing dips {1000 * (z_carry - z):.0f} mm below the carry height")
    return plan


def plan_pick(arm, sc, targets, container, opts: PickOptions, G) -> tuple:
    """Best feasible (MotionPlan, report) over the given target objects and their grasp
    candidates. report lists what was considered. Raises StageError("plan") if nothing works."""
    cfg = arm.cfg
    report, best = [], None
    for target in targets:
        saved = cfg.get("keepouts", {})
        cfg["keepouts"] = {**{k: v for k, v in saved.items() if not k.startswith("obj")},
                           **scene_keepouts(sc, skip_ids=(target.id,))}
        try:
            cands = G.plan_grasp(target, sc, arm, cfg)
            for c in cands[:6]:
                row = {"object": target.id, "ok": bool(c.ok), "azimuth": round(float(c.jaw_azimuth_deg), 1),
                       "score": round(float(c.score), 2), "reason": c.reason}
                report.append(row)
                if not c.ok or (best is not None and best[0] >= c.score):
                    continue
                rel = G.plan_release(target, container, c, arm, cfg, scene=sc)  # scene: release above what is inside
                if not rel.ok:
                    row.update(ok=False, reason=f"release: {rel.reason}")
                    continue
                try:
                    plan = build_plan(arm, sc, target, container, c, rel, opts)
                except SafetyError as e:
                    row.update(ok=False, reason=f"path: {e}")
                    continue
                best = (float(c.score), plan, dict(cfg["keepouts"]))
        finally:
            cfg["keepouts"] = saved
    if best is None:
        reasons = "; ".join(sorted({r["reason"] for r in report if r["reason"]}))[:400]
        raise StageError("plan", "no feasible grasp: " + (reasons or "the planner returned no candidates"),
                         {"candidates": report})
    return best[1], best[2], report


# ----------------------------------------------------------------------------- the skill
def options_from_config(cfg: dict, **over) -> PickOptions:
    """PickOptions with the defaults overridden by cfg["pick"] (tuning without code changes) and
    then by the given keyword arguments (None = not given)."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(PickOptions)}
    vals = {k: v for k, v in (cfg.get("pick") or {}).items() if k in names}
    vals.update({k: v for k, v in over.items() if v is not None})
    return PickOptions(**vals)


def pick_place(sess: Session, target_query: dict, container_query: dict | None = None,
               opts: PickOptions | None = None, modules=None) -> dict:
    """Pick the object matching target_query and drop it into the container matching
    container_query (default: the only container in view).

    Returns a dict: ok, stage (last stage reached), reason, timings, the scene summary, the
    chosen grasp / release, measured values of every check and paths of saved frames.
    "arm_state" says how the arm was left: "parked" (torque off, folded), "ready" (holding the
    stand-by pose), "stopped" (holding some other pose), "holding_object".
    modules: (grasp, scene) replacements for tests."""
    if modules is None:
        import grasp as G
        import scene as S
    else:
        G, S = modules
    opts = opts or PickOptions()
    arm = sess.arm
    res = {"ok": False, "stage": "start", "reason": "", "attempts": 0, "checks": {}, "images": sess.images}
    tm = Timer(arm_clock(arm))
    try:
        arm.speeds(opts.tier)  # refuses an unknown tier or one not yet validated on the hardware
    except SafetyError as e:
        res.update(reason=str(e), arm_state=_arm_state(arm, arm.read_q()), timings={"total": tm.total()})
        return res
    saved_keepouts = dict(arm.cfg.get("keepouts", {}))
    try:
        for attempt in range(1, max(0, opts.retries) + 2):
            res["attempts"] = attempt
            try:
                _pick_place_once(sess, target_query, container_query, opts, res, tm, G, S)
                res.update(ok=True, reason="")
                break
            except StageError as e:
                res.update(ok=False, stage=e.stage, reason=e.reason)
                res["checks"].setdefault("failures", []).append({"attempt": attempt, "stage": e.stage,
                                                                 "reason": e.reason, **e.detail})
                again = (e.stage == "grasp" and res.get("arm_state") == "ready" and attempt <= opts.retries
                         and not opts.dry)
                if not again:
                    break
                sess.log(f"attempt {attempt} failed at {e.stage}: {e.reason}; looking again")
    except MOTION_ERRORS as e:
        res.update(ok=False, reason=f"{type(e).__name__}: {e}")
        res.setdefault("arm_state", "stopped")
    except Exception as e:  # a bug in perception / planning / this file: report it, do not move any further
        import traceback
        res.update(ok=False, reason=f"internal error: {type(e).__name__}: {e}")
        res["trace"] = traceback.format_exc().strip().splitlines()[-6:]
        res.setdefault("arm_state", "stopped")
    finally:
        arm.cfg["keepouts"] = saved_keepouts
    res["timings"] = {**tm.marks, "total": tm.total()}
    return res


def _settle_container(sess, con):
    """A container seen past the arm (waiting above it) comes back with low confidence and
    possibly an unmeasured rim. If this session already measured a container at that place
    with the arm out of the way, its geometry is used instead."""
    if float(con.confidence) >= 0.5:
        sess.known_containers = [k for k in sess.known_containers
                                 if math.hypot(k.center[0] - con.center[0], k.center[1] - con.center[1]) > 0.03] + [con]
        return con
    for k in sess.known_containers:
        if math.hypot(k.center[0] - con.center[0], k.center[1] - con.center[1]) < 0.015:
            con.center, con.yaw, con.size, con.z_bottom = k.center, k.yaw, k.size, k.z_bottom
            con.footprint, con.inner = k.footprint, k.inner
            con.notes = list(con.notes) + ["geometry taken from an earlier, unobstructed look in this session"]
            return con
    con.notes = list(con.notes) + ["seen with low confidence (arm in the way?)"]
    return con


def _retreat(arm, sp, res, via=(), ignore=(), open_raw=None) -> bool:
    """After a failed check with nothing in the jaws: back off through the fingertip poses
    `via` to the stand-by pose in one motion (opening the gripper if open_raw is given)."""
    try:
        q = arm.read_q()
        wps = [q] + (_tip_path(arm, list(via), q) if via else [])
        wps.append(standby_q(arm, wps[-1]))
        arm.run_path(wps, sp["transit"], settle=0.3, ignore_keepouts=ignore, gripper_goal_raw=open_raw)
        res["arm_state"] = "ready"
        return True
    except MOTION_ERRORS as e:
        res["arm_state"] = "stopped"
        res["checks"]["retreat_error"] = f"{type(e).__name__}: {e}"
        return False


def _pick_place_once(sess, target_query, container_query, opts, res, tm, G, S):
    arm, cam = sess.arm, sess.cam
    cfg = arm.cfg

    # ---------------------------------------------------------------- 1. perceive
    res["stage"] = "perceive"
    fresh = False
    while True:
        q0 = arm.read_q()
        jaw0 = arm.gripper_q_deg(_gripper_raw(arm))
        res["arm_state"] = _arm_state(arm, q0)
        try:
            img0 = sess.frame(fresh=fresh)
        except Exception as e:  # camera could not deliver a frame: nothing has moved for this pick
            raise StageError("perceive", f"no camera frame: {e}")
        sc = S.perceive(img0, cam, cfg, arm_q=q0, jaw_q_deg=jaw0)
        res["scene"] = _summarise(sc)
        res["scene_warnings"] = list(sc.warnings)
        missing, targets, cons = None, [], []
        try:
            targets = select_objects(sc, target_query)
        except LookupError as e:
            missing = (f"target not found: {e}", {"query": target_query})
        if missing is None:
            try:
                cons = select_objects(sc, container_query or {}, containers=True)
            except LookupError as e:
                missing = (f"container not found: {e}", {"query": container_query})
        if missing is None:
            break
        if res["arm_state"] in ("ready", "parked", "limp") or opts.dry:
            tm.mark("perceive")
            raise StageError("perceive", missing[0], missing[1])
        # the arm may be standing between the camera and the object (or over the container):
        # look again from the ready pose, which hides nothing that can be reached
        sess.log(f"{missing[0]}; looking again from the ready pose")
        try:
            go_ready(arm, opts.tier)
        except MOTION_ERRORS as e2:
            raise StageError("perceive", f"{missing[0]}; moving to the ready pose for a second look failed: {e2}")
        fresh = True
    tm.mark("perceive")
    if opts.save_images:
        try:
            sess.save("scene", S.annotate(img0, sc, cam))
        except Exception as e:  # annotation is a convenience, never a reason to stop
            res["scene_warnings"].append(f"annotate failed: {e}")
    if len(targets) > 1 and not opts.any_match:
        raise StageError("perceive", f"{len(targets)} objects match; choose one by id",
                         {"query": target_query, "matches": [o.id for o in targets]})
    if len(cons) != 1:
        raise StageError("perceive", f"{len(cons)} containers match; say which one",
                         {"matches": [o.id for o in cons]})
    container = _settle_container(sess, cons[0])
    res["container"] = {"id": container.id, "color": container.color, "inner": container.inner}
    before_inside = S.objects_in_container(img0, cam, cfg, container, arm_q=q0, jaw_q_deg=jaw0)

    # ---------------------------------------------------------------- 2. plan (all paths, no motion)
    res["stage"] = "plan"
    plan, keepouts, report = plan_pick(arm, sc, targets, container, opts, G)
    target, best, rel = plan.target, plan.cand, plan.rel
    res["target"] = {"id": target.id, "kind": target.kind, "color": target.color}
    res["grasp_candidates"] = report
    res["grasp"] = best.to_dict()
    res["release"] = rel.to_dict()
    res["plan"] = plan.summary()
    tm.mark("plan")
    if opts.dry:
        res["stage"] = "planned"
        return
    arm.cfg["keepouts"] = keepouts  # everything the camera sees except the target
    box_ko = container_keepouts(sc, container)
    roll = float(best.roll_deg)

    # ---------------------------------------------------------------- 3. approach
    res["stage"] = "approach"
    hot = max(arm.state()["temp_c"].values())
    if hot > opts.max_temp_c:
        raise StageError("approach", f"a servo is at {hot} C; let the arm cool down")
    try:
        sp = arm.speeds(opts.tier)
        q_now = arm.read_q()
        q_from = np.asarray(arm.ready_q(), dtype=float) if _is_parked(q_now) else q_now
        transit = plan_transit(arm, q_from, plan.approach[0], plan.z_clear)
        check_path(arm, [q_from] + transit + plan.approach[1:], sp["transit"], opts, gripper_raw=best.open_raw,
                   what="approach")
    except SafetyError as e:
        raise StageError("approach", str(e))
    if not arm.is_torqued():
        arm.torque_on()
    if hasattr(arm, "apply_profile"):
        sp = arm.apply_profile(opts.tier)  # servo-side speed profile and time law of the tier
    res["speed_tier"] = sp
    res["arm_state"] = "stopped"
    if _is_parked(q_now):
        # unfold first (the gripper stays closed until the arm is clear of its rest)
        arm.run_path([q_now, arm.ready_q()], sp["transit"], settle=False)
        q_now = arm.read_q()
        transit = plan_transit(arm, q_now, plan.approach[0], plan.z_clear)
    # free-space transit to a point above the hover point, then straight down to it, as one
    # motion; the gripper opens on the way
    arm.run_path([q_now] + transit + plan.approach[1:], sp["transit"], settle=0.8, gripper_goal_raw=best.open_raw)
    tm.mark("approach")
    back_up = [(plan.above_hover, plan.p_above, roll)]

    # ---------------------------------------------------------------- 4. look at the jaws
    # detect_jaws reports d = (jaw position seen by the camera) - (jaw position from the joint
    # angles). The object was located through the same camera, so the commanded fingertip
    # position is shifted by corr = -d: then the camera-seen jaws sit where the plan wants them.
    res["stage"] = "visual"
    corr = np.zeros(2)
    vis = {"mode": opts.visual, "seen": False, "applied": False, "steps": []}
    res["checks"]["visual"] = vis
    if opts.visual != "off":
        tight = min(best.clearance_fixed, best.clearance_moving) < opts.tight_clearance
        strict = tight or opts.visual == "on"
        apply = strict or opts.visual_apply_when_roomy
        vis["tight"] = bool(tight)
        n_iter = max(1, opts.visual_iters) if strict else 1  # a tight grasp gets a confirming look
        d_prev = None
        for it in range(n_iter):
            img1 = None
            try:
                img1 = sess.frame(fresh=True)
                dj = S.detect_jaws(img1, img0, cam, arm.read_q(), arm.gripper_q_deg(_gripper_raw(arm)), cfg)
                if not isinstance(dj, dict):
                    dj = {"ok": False, "confidence": 0.0, "dxy_base": None}
            except MOTION_ERRORS:
                raise
            except Exception as e:  # camera or perception problems must not crash a motion sequence
                dj = {"ok": False, "confidence": 0.0, "dxy_base": None, "error": str(e)}
            conf = float(dj.get("confidence", 0.0) or 0.0)
            dxy = dj.get("dxy_base")
            try:
                dxy = None if dxy is None else [float(dxy[0]), float(dxy[1])]
                if dxy is not None and not all(math.isfinite(v) for v in dxy):
                    dxy = None
            except (TypeError, ValueError, IndexError):
                dxy = None
            usable = bool(dj.get("ok")) and dxy is not None and conf >= opts.min_visual_conf
            # With only the moving jaw in view the offset ALONG the opening direction is unknown
            # (reported as zero). That is the direction a tight grasp depends on.
            full = bool((dj.get("fixed") or {}).get("visible", True))
            vis["steps"].append({"ok": bool(dj.get("ok")), "confidence": round(conf, 2), "full": full,
                                 "dxy_mm": None if dxy is None else [round(v * 1000, 1) for v in dxy]})
            problem = None
            if not usable or (tight and not full):
                if not strict:
                    break  # roomy grasp: carry on with what is known so far
                problem = ("could not confirm the jaw position in the image" if not usable else
                           "the fixed jaw is hidden from the camera, so its position along the closing direction "
                           "cannot be confirmed") + (" and the grasp is tight" if tight else "")
            else:
                d = np.array(dxy, dtype=float)
                if float(np.linalg.norm(d)) > opts.max_visual_corr and not apply:
                    vis["warning"] = (f"the jaws appear {np.linalg.norm(d) * 1000:.0f} mm from where the kinematics "
                                      "put them (recorded only; the grasp is roomy)")
                elif float(np.linalg.norm(d)) > opts.max_visual_corr:
                    problem = (f"the jaws appear {np.linalg.norm(d) * 1000:.0f} mm from where the kinematics "
                               "put them; not descending")
                elif d_prev is not None and float(np.linalg.norm(d - d_prev)) > opts.visual_repeat_tol:
                    problem = f"two looks at the jaws disagree by {np.linalg.norm(d - d_prev) * 1000:.1f} mm; not descending"
            if problem:
                sess.save("visual_fail", img1)
                _retreat(arm, sp, res, via=back_up, open_raw=None)
                raise StageError("visual", problem, dict(vis))
            residual = float(np.linalg.norm(corr + d))  # camera-seen jaws minus the planned position
            d_prev = d
            vis["seen"] = True
            if not apply:
                break  # recorded only: a roomy grasp tolerates this offset
            if residual <= opts.visual_tol:
                break  # already where the plan wants them (within tolerance)
            corr = -d
            if it + 1 < n_iter:
                # look again from the corrected hover point (small, slow, straight move)
                hv = plan.hover.copy()
                hv[:2] += corr
                arm.move_tip_path([(hv, plan.p_hover, roll)], sp["precise"], settle=0.6)
        vis["corr_mm"] = [round(float(v) * 1000, 1) for v in corr]
        vis["applied"] = bool(float(np.linalg.norm(corr)) > 1e-6)
        if vis["applied"]:
            try:  # re-plan everything after the hover point for the shifted grasp position
                plan = build_plan(arm, sc, target, container, best, rel, opts, corr=corr, q_hover=arm.read_q())
            except SafetyError as e:
                _retreat(arm, sp, res, via=back_up)
                raise StageError("visual", f"the corrected grasp position is not feasible: {e}", dict(vis))
    tm.mark("visual")

    # ---------------------------------------------------------------- 5. descend + grasp
    res["stage"] = "descend"
    arm.run_path([arm.read_q()] + plan.descent, sp["approach"], settle=1.0)
    tm.mark("descend")

    res["stage"] = "grasp"
    g = arm.grasp_fast(expected_raw=best.expected_raw)
    chk = {k: g.get(k) for k in ("contact", "empty", "pos_raw", "width_mm", "duration_s", "load")}
    chk["expected_raw"] = best.expected_raw
    res["checks"]["grasp"] = chk
    tm.mark("grasp")
    stuck = g.get("moved_ticks") is not None and int(g["moved_ticks"]) < 15  # "contact" without having moved
    if g.get("empty") or not g.get("contact") or stuck:
        # nothing between the jaws: straight up with the jaws as they are, then open on the way
        # back to the stand-by pose, and look at the table again (the caller may retry)
        hv = (plan.hover + np.array([corr[0], corr[1], 0.0]), plan.p_hover, roll)
        try:
            arm.move_tip_path([hv], sp["approach"], settle=False)
        except MOTION_ERRORS as e:
            res["checks"]["retreat_error"] = f"{type(e).__name__}: {e}"
            raise StageError("grasp", "the jaws closed on nothing, and backing up failed", {"pos_raw": g.get("pos_raw")})
        _retreat(arm, sp, res, via=back_up, open_raw=best.open_raw)
        raise StageError("grasp", "the jaws did not move when closing (were they open?)" if stuck
                         else "the jaws closed on nothing", {"pos_raw": g.get("pos_raw")})
    if abs(int(g["pos_raw"]) - int(best.expected_raw)) > opts.stall_tol_ticks:
        chk["warning"] = "stall position far from the expected width"

    # ---------------------------------------------------------------- 6. lift + cross + lower (one motion)
    res["stage"] = "transfer"
    res["arm_state"] = "holding_object"
    arm.run_path([arm.read_q()] + plan.carry, sp["transit"], settle=0.6, ignore_keepouts=box_ko)
    tm.mark("transfer")
    held = _gripper_raw(arm)
    res["checks"]["held_raw_before_release"] = held
    above_box = [(np.array(rel.tip_above, dtype=float), rel.pitch_deg, rel.roll_deg)]
    above_box[0][0][2] = max(above_box[0][0][2], plan.z_carry)
    # While the object is held the jaws stay where they stalled; without it they close on to the
    # holding goal (stall position minus the squeeze), which is still well open - so compare
    # with the stall position, not with "closed".
    stall = int(g["pos_raw"])
    hold_goal = int(g.get("hold_goal_raw") if g.get("hold_goal_raw") is not None else stall - 40)
    lost_below = stall - max(8, (stall - hold_goal) // 2)
    res["checks"]["held_raw_min"] = lost_below
    if held < lost_below:
        res["arm_state"] = "stopped"
        _retreat(arm, sp, res, via=above_box, ignore=box_ko)
        raise StageError("transfer", "the object was lost on the way (the jaws are closed); nothing released",
                         {"gripper_raw": held})

    # ---------------------------------------------------------------- 7. release + retreat
    res["stage"] = "release"
    arm.set_gripper(raw=rel.open_raw, wait=False)
    arm_sleep(arm, opts.release_wait_s)
    res["arm_state"] = "stopped"
    # the jaws stay open while backing out (closing here could catch the object again)
    wps = [arm.read_q()] + plan.retreat
    if not plan.standby_over:
        wps.append(standby_q(arm, wps[-1]))
    arm.run_path(wps, sp["transit"], settle=0.3, ignore_keepouts=box_ko)
    res["arm_state"] = "standby" if plan.standby_over else "ready"
    tm.mark("release")

    # ---------------------------------------------------------------- 8. verify
    res["stage"] = "verify"
    if opts.verify:
        ver = _verify(sess, S, sc, target, container, before_inside, opts, "final")
        if not ver["confirmed"] and res["arm_state"] != "ready":
            # the arm may be hiding the place the object stood on, or part of the container:
            # look once more from the ready pose before calling it a failure
            try:
                go_ready(arm, opts.tier)
                res["arm_state"] = "ready"
                first = ver
                ver = _verify(sess, S, sc, target, container, before_inside, opts, "final2")
                ver["first_look"] = {k: first[k] for k in ("object_still_on_table", "arrived")}
            except MOTION_ERRORS as e:
                ver["second_look_error"] = f"{type(e).__name__}: {e}"
        if (not ver["confirmed"] and not ver["object_still_on_table"] and "stray" not in ver
                and res["arm_state"] in ("ready", "standby")):
            # Nothing contradicts the pick: the object left its place, the jaws held it up to the
            # release point over the container, and no such object lies on the table. The camera
            # just cannot tell it apart inside the container (it landed on something of its own
            # colour, or is hidden). Rule out the last alternative: that it is still in the jaws.
            try:
                g2 = arm.grasp_fast()
                ver["jaws_empty"] = bool(g2.get("empty"))
            except MOTION_ERRORS as e:
                ver["jaw_check_error"] = f"{type(e).__name__}: {e}"
            if ver.get("jaws_empty"):
                ver["confirmed"] = True
                ver["how"] = "inferred: held until the release over the container, gone from the table, jaws empty"
            elif "jaws_empty" in ver:
                res["checks"]["verify"] = ver
                res["arm_state"] = "holding_object"
                tm.mark("verify")
                raise StageError("verify", "the object seems to be still in the gripper", ver)
        res["checks"]["verify"] = ver
        tm.mark("verify")
        if not ver["confirmed"]:
            reason = "the object is not confirmed inside the container"
            if "stray" in ver:
                reason = (f"the object seems to have landed outside the container, at "
                          f"({ver['stray']['center'][0]:.3f}, {ver['stray']['center'][1]:.3f})")
            raise StageError("verify", reason, ver)

    # ---------------------------------------------------------------- 9. end state
    res["stage"] = "done"
    if opts.end == "park":
        park(arm, sp)
        res["arm_state"] = "parked"
        tm.mark("park")


_ALIKE = ({"white", "ivory"},)  # colours the camera cannot reliably tell apart inside a dark container


def _color_family(color: str) -> set:
    for fam in _ALIKE:
        if color in fam:
            return fam
    return {color}


def _stray(sc_before, sc_after, target):
    """A loose object of the target's colour that lies on the table now but did not before:
    the object was dropped beside the container or bounced out of it."""
    fam = _color_family(target.color)
    known = [o for o in sc_before.graspable() if o.id != target.id and o.color in fam]
    for o in sc_after.graspable():
        if o.color not in fam:
            continue
        if math.hypot(o.center[0] - target.center[0], o.center[1] - target.center[1]) < 0.015:
            continue  # at the old place: that is object_present's question
        if any(math.hypot(o.center[0] - k.center[0], o.center[1] - k.center[1]) < 0.015 for k in known):
            continue
        return o
    return None


def _verify(sess, S, sc, target, container, before_inside, opts, tag: str) -> dict:
    """One look at the result: is the object gone from where it stood, did something arrive in
    the container, and if neither can be seen: does it lie somewhere else on the table?
    (object_present answers True when it cannot see the place.)"""
    arm, cam, cfg = sess.arm, sess.cam, sess.arm.cfg
    try:
        img = sess.frame(fresh=True)
    except Exception as e:  # camera stalled: the motion is finished, only the confirmation is missing
        raise StageError("verify", f"no camera frame for the final check: {e}")
    q = arm.read_q()
    jaw = arm.gripper_q_deg(_gripper_raw(arm))
    still_there = bool(S.object_present(img, cam, cfg, target, arm_q=q))
    after = S.objects_in_container(img, cam, cfg, container, arm_q=q, jaw_q_deg=jaw)
    bbox = getattr(target, "pixel_bbox", None) or (0, 0, 0, 0)
    expect_px = 0.6 * float(bbox[2]) * float(bbox[3]) if len(bbox) == 4 else 0.0
    ver = {"object_still_on_table": still_there, **_inside_change(before_inside, after, target.color, expect_px)}
    if not ver["arrived"] and not still_there:
        try:
            stray = _stray(sc, S.perceive(img, cam, cfg, arm_q=q, jaw_q_deg=jaw), target)
            if stray is not None:
                ver["stray"] = {"id": stray.id, "color": stray.color, "kind": stray.kind,
                                "center": [round(float(v), 4) for v in stray.center]}
        except Exception as e:  # a perception problem here must not turn into a crash
            ver["scene_error"] = f"{type(e).__name__}: {e}"
    ver["confirmed"] = bool(ver["arrived"] and not still_there)
    if opts.save_images:
        sess.save(tag, img)
    return ver


def _inside_change(before: list, after: list, color: str, expect_px: float = 0.0, min_area_px: float = 800.0) -> dict:
    """Did the object arrive in the container? Evidence, from the colour blobs seen inside the
    rim before and after (blobs touching the arm in the image, "near_arm", and specks are
    ignored):
    - one more blob of the object's colour, or
    - clearly more area of that colour (a blob can merge with a neighbour of the same colour;
      white and ivory count as one colour here), or
    - clearly more coloured area in total (the object landed on something and the camera gives
      the pile one colour): at least a quarter of the object's own size in the image."""
    fam = _color_family(color)

    def usable(blobs):
        return [b for b in blobs if not b.get("near_arm") and float(b.get("area_px", 0.0) or 0.0) >= min_area_px]

    def stat(blobs, colors=None):
        mine = [b for b in usable(blobs) if colors is None or b.get("color") in colors]
        return len(mine), sum(float(b.get("area_px", 0.0) or 0.0) for b in mine)

    n0, a0 = stat(before, fam)
    n1, a1 = stat(after, fam)
    _, t0 = stat(before)
    _, t1 = stat(after)
    more_blobs = n1 > n0
    more_color = a1 >= a0 * 1.25 + min_area_px
    more_total = (t1 - t0) >= max(1500.0, 0.25 * float(expect_px or 0.0))
    out = {"in_container_before": n0, "in_container_after": n1, "area_before_px": round(a0), "area_after_px": round(a1),
           "total_before_px": round(t0), "total_after_px": round(t1),
           "arrived": bool(more_blobs or more_color or more_total),
           "evidence": [name for name, hit in (("new blob", more_blobs), ("more of its colour", more_color),
                                               ("more in total", more_total)) if hit]}
    if any(b.get("color") in fam and b.get("near_arm") for b in after):
        out["near_arm"] = True  # something of that colour is at the arm: possibly still in the jaws
    return out


# ----------------------------------------------------------------------------- small skills
def go_ready(arm, tier: str | None = None) -> dict:
    """Torque on if needed and move to the stand-by pose. With a tool taped to the gripper (arm.tool) that
    is the tool's stand-by pose (the TOOL tip at the stand-by point), reached on a way that never brings
    the tool tip lower than 2 cm above the table: the plain way would swing a 6 cm tool through the table."""
    if getattr(arm, "tool", None) is not None:
        import tooling
        return tooling.go_ready(arm, tier)
    sp = arm.speeds(tier)
    if not arm.is_torqued():
        arm.torque_on()
    if hasattr(arm, "apply_profile"):
        sp = arm.apply_profile(tier)
    q = arm.read_q()
    if _is_parked(q):
        return arm.run_path([q, arm.ready_q()], sp["transit"], settle=0.5)
    # from anywhere else: straight up first if the fingertip is low, then across
    z_clear = float(arm.cfg.get("table_z", 0.0)) + 0.06
    return arm.run_path([q] + plan_transit(arm, q, arm.ready_q(), z_clear), sp["transit"], settle=0.5)


def park(arm, sp: dict | None = None) -> dict:
    """Ready pose -> folded rest pose -> torque off (also restores the servo registers).

    Refused (SafetyError, before anything moves) while a tool is taped to the gripper: park opens the
    gripper to 10 % and folds the arm until the FINGERTIP is 28 mm above the table, which drives a 6 cm
    tool 17 mm into it. `fd tool rest` (tooling.rest) is the way to fold away with a tool."""
    if getattr(arm, "tool", None) is not None:
        import tooling
        raise SafetyError(tooling.lock_message(arm.tool, "park"))
    sp = sp or arm.speeds(None)
    q = arm.read_q()
    if _is_parked(q):
        wps = [q, arm.rest_q()]
    else:  # via the ready pose (wrist roll back to zero there), raising the fingertip first if it is low
        z_clear = float(arm.cfg.get("table_z", 0.0)) + 0.06
        wps = [q] + plan_transit(arm, q, arm.ready_q(), z_clear) + [arm.rest_q()]
    out = arm.run_path(wps, min(sp["transit"], 30.0), allow_contact=True, settle=False,
                       gripper_goal_raw=arm.gripper_raw_for(0.1))
    arm_sleep(arm, 0.4)
    arm.torque_off()
    return out


# ============================================================================= take-out: container -> table
# One call = perceive -> choose the object INSIDE the container -> plan the grasp from the free space around it
# (takeout.py) -> hover above the rim -> look at the jaws and shift the descent by what is seen (mandatory) ->
# straight down to the grasp height (from geometry: table + container floor + the object's own bottom) ->
# close -> lift a little and check the jaws -> lift clear of the rim, look that the object left its place ->
# carry out, put it down on a free spot of the table -> open -> stand-by pose -> verify with the camera.
# Written after 2026-11-11, when the same job took 92 hand-driven commands and 42 minutes.
@dataclass
class TakeOptions:
    tier: str | None = None  # speed tier name; None = cfg["speed_tier"]
    # look at the jaws above the rim and shift the descent by what is seen. "on" is the only setting for
    # the real arm (no measurement = no descent). "off" is for the simulated arm, which no camera can see:
    # take_out refuses it on any other arm (except in a dry run, which does not move), and any other value
    visual: str = "on"
    end: str = "ready"  # "ready" | "park"
    retries: int = 1  # extra attempts after the jaws closed on nothing or the object slipped while lifting
    dry: bool = False  # perceive + plan only (nothing moves, torque untouched)
    any_match: bool = False  # several objects match the query: take the easiest instead of asking
    hollow: bool | None = None  # True: open at the top (its wall can be pinched); False: solid; None: what the scene says
    # the scene says nothing about it either: True = take it as hollow and open at the top (the request called
    # it a cap, a cup, a lid); False = a cylinder is still modelled as hollow, but its wall is not pinched on
    # that guess alone (nothing checks the cavity before the moving jaw goes down into it)
    assume_hollow: bool = False
    mode: str | None = None  # "pinch" | "two_sided": hold it this way only; None: the planner chooses
    to: tuple | None = None  # (x, y) on the table where it is put down; None: the nearest free spot
    # Heights come from geometry (the scene's table height + the container's floor + the object's own bottom);
    # nothing needs a touch. Optional touch before the last stretch down: "off" (default) = never; "on" =
    # always; "auto" = when the planned grasp leaves the jaws less than probe_margin above what is under them.
    # It is a CHECK, not a measurement: something met above the expected height stops the take-out, the grasp
    # height stays the one from the scene. Reason (measured 2026-11-11): Arm.probe_down reports its contact
    # 3-18 mm DEEPER than the first touch, because the arm does not stop when the fingertip lands; it keeps
    # following the plan and the fingertip slides outward by 1.0-1.5 mm per mm (it pushes what it touches).
    probe: str = "off"
    probe_may_lower: bool = False  # True: use the touched height as the grasp height (needs a probe that reports the FIRST touch)
    probe_press: float = 0.002  # the touch check goes this far past the expected contact, not further
    probe_margin: float = 0.003
    probe_speed: float = 0.006  # m/s (Arm.probe_down accepts 0.002 .. 0.012)
    # the touch descent starts this far above the expected contact (it needs a run-in), and never below the
    # rim: whatever the jaws meet inside the container, they meet it in a slow, force-limited descent
    probe_above: float = 0.019
    probe_below: float = 0.022  # (probe_may_lower) ... and may go this far below it
    probe_retry_extra: float = 0.010  # (probe_may_lower) the one repeat goes this much deeper than the first try got
    probe_backoff: float = 0.006  # the fingertip rises this far after the touch (the grasp height is just above it)
    probe_max_dev: float = 0.015  # (probe_may_lower) the touched height may differ from the scene's by at most this
    probe_warn_dev: float = 0.005  # ... beyond this it is reported: the heights of arm and scene disagree here
    probe_slide_max: float = 0.006  # fingertip slipping further than this sideways while probing: result not used
    rim_margin: float = 0.012  # fingertip above the rim at the hover point; nothing goes lower before the look
    approach_drop: float = 0.015  # the last part of the approach is a vertical descent of this length
    lift_check: float = 0.012  # the little lift after which the jaws are checked
    lift_look: bool = True  # one camera frame at the top of the lift: has the object left its place?
    rim_clear: float = 0.025  # object bottom above the rim before it is carried sideways
    rim_clear_min: float = 0.010  # ... the least accepted where the arm cannot reach higher
    carry_clearance: float = 0.015  # gap between the carried object and anything it passes over
    place_gap: float = 0.003  # object bottom above the table when the jaws open
    place_clear: float = 0.025  # free space around the object at the place spot
    place_tol: float = 0.035  # further than this from the place spot is reported as a warning
    place_contact_window: float = 0.015  # the table met this far above the planned height counts as "put down"
    # Where the arm folds down onto its rest pose (`fd park`, and the fold for a clear view of a container):
    # neither motion knows what stands on the table there. A place spot keeps the object's outline this far
    # from the parked gripper's ground track (from arm.rest_q(), not a table coordinate) ...
    park_clear: float = 0.04
    # ... and the arm is not folded back for a second look when the first one shows something nearer to it
    # than this (the fingers are 6 to 9 mm wide to each side there; the rest is what calibration and
    # perception may be off by). For something tall enough to meet the gripper's body, 20 mm more.
    fold_clear: float = 0.02
    transit_clearance: float = 0.02
    # Largest xy correction accepted from the look at the jaws (m); a larger reading stops the take-out
    # ("calibration?"). What the look CAN measure is bounded by the detector's search window (scene.py:
    # 36 px around where the joint angles put the jaws = about 10 mm sideways and 18 mm along the camera's
    # line of sight, at the hover point), so a limit above that would never be reached. And it measures
    # the gripper against the camera only: an offset along the line of sight is invisible to it, an error
    # of the arm's height shows as about 1.4 times as much along it.
    max_visual_corr: float = 0.012
    # a jaw found further out than this fraction of that search window is not trusted either: the best
    # match inside the window may be the edge of a larger offset
    visual_edge: float = 0.9
    min_visual_conf: float = 0.7
    # looks at the jaws at most. Each correction is confirmed by the next look; one that is still not
    # confirmed when the looks are used up stops the take-out (it is not driven down unseen)
    visual_iters: int = 3
    visual_tol: float = 0.002  # residual below which no (further) correction is made (m)
    visual_repeat_tol: float = 0.004  # two looks may differ by at most this much (m)
    visual_raise: float = 0.025  # jaws not measurable at the hover point: one more look from this much higher (0 = no)
    # Things inside a container are always seen roughly (scene.py rates them 0.2 - 0.3). A detection rated
    # below this is taken only by an explicit id; the default lets every in-container detection be chosen by
    # colour or shape, because the request names it and the id would add nothing to its geometry.
    min_confidence: float = 0.15
    stall_tol_ticks: int = 110
    release_wait_s: float = 0.35
    verify: bool = True
    max_temp_c: float = 55.0
    save_images: bool = True


def take_options_from_config(cfg: dict, **over) -> TakeOptions:
    """TakeOptions with the defaults overridden by cfg["take"] and then by the keyword arguments
    (None = not given). cfg["take"] may also hold planner settings (takeout.DEFAULTS); those are
    read by takeout.py itself."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(TakeOptions)}
    vals = {k: v for k, v in (cfg.get("take") or {}).items() if k in names}
    vals.update({k: v for k, v in over.items() if v is not None})
    return TakeOptions(**vals)


def _simulated(arm) -> bool:
    """True when the arm's bus is the simulator (sim_bus.SimBus or a subclass of it): an arm that no
    camera can see. Anything else, the stand-in arms of the tests included, counts as a real arm."""
    try:
        from sim_bus import SimBus
    except Exception:  # no simulator in this tree: nothing is simulated
        return False
    return isinstance(getattr(arm, "bus", None), SimBus)


def _park_track(arm):
    """Where the gripper comes down when the arm folds onto its rest pose (park(), _fold_back()): the
    two ends (base xy) of the gripper's ground track there, from under the palm to under the foremost
    jaw nose, and the height above which the gripper is as wide as its body (below it there are only
    the fingers). It comes from the arm's own rest pose (arm.rest_q(), the opening park() closes to),
    so it follows the calibration; no table coordinate is built in. None for an arm that cannot say."""
    try:
        q = np.asarray(arm.rest_q(), dtype=float)
        pts = K.jaw_points(q, float(arm.gripper_q_deg(arm.gripper_raw_for(0.1))), arm.tip_tool())
        palm = np.asarray(pts["palm"], dtype=float)
        ends = [np.asarray(pts[k], dtype=float)[:2] for k in ("fixed_nose", "moving_nose", "tip") if k in pts]
        front = max(ends, key=lambda e: float(np.hypot(*(e - palm[:2]))))
        return palm[:2], front, float(palm[2]) - 0.015
    except Exception:  # a stand-in without these methods: no parking test rather than no take-out
        return None


def _park_dist(track, xy) -> float:
    """Distance (m) from a table point to the gripper's ground track at the rest pose."""
    a, b = track[0], track[1]
    pt = np.asarray(xy, dtype=float)[:2]
    ab = b - a
    t = float(np.clip((pt - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.hypot(*(pt - (a + t * ab))))


def _radius(o) -> float:
    """Radius (m) of the circle around an object's centre that holds its outline."""
    if getattr(o, "kind", None) == "cylinder":
        return 0.5 * max(float(o.size[0]), float(o.size[1]))
    return 0.5 * math.hypot(float(o.size[0]), float(o.size[1]))


def _park_clearance(track, o) -> float:
    """Free distance (m) between an object's outline and that ground track; 0 when it stands on it."""
    fp = [(float(v[0]), float(v[1])) for v in (getattr(o, "footprint", None) or [])]
    if getattr(o, "kind", None) == "cylinder" or len(fp) < 3:
        return max(0.0, _park_dist(track, o.center) - _radius(o))
    for end in track[:2]:  # the track under (inside) the outline: ray casting
        inside = False
        for (x0, y0), (x1, y1) in zip(fp, fp[1:] + fp[:1]):
            if (y0 > end[1]) != (y1 > end[1]) and end[0] < x0 + (end[1] - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
        if inside:
            return 0.0
    best = float("inf")
    for (x0, y0), (x1, y1) in zip(fp, fp[1:] + fp[:1]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.004)))
        for k in range(n):
            best = min(best, _park_dist(track, (x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n)))
    return best


def _parking_blocked(arm, sc, clear: float) -> str:
    """"" or what the scene shows standing where the folded arm would come down: within `clear` of the
    parked gripper's ground track, or within 20 mm more for something tall enough to meet the gripper's
    body. park() and _fold_back() run with contact allowed against the static keep-outs only, so the
    camera's word on that place is all there is."""
    track = _park_track(arm)
    if track is None:
        return ""
    hits = []
    for o in sc.objects:
        d = _park_clearance(track, o)
        need = clear + (0.02 if float(o.z_top) >= track[2] else 0.0)
        if d < need:
            hits.append((d - need, d, need, o))
    if not hits:
        return ""
    _short, d, need, o = min(hits, key=lambda h: h[0])
    mid = 0.5 * (track[0] + track[1])
    return (f"the {o.color} {o.kind} (id {o.id}) at ({o.center[0]:.3f}, {o.center[1]:.3f}) stands where the folded "
            f"arm would come down: {d * 1000:.0f} mm from the parked gripper around ({mid[0]:.3f}, {mid[1]:.3f}), "
            f"{need * 1000:.0f} mm needed")


def select_inside(sc, query: dict, container_query: dict | None = None, min_confidence: float = 0.15) -> tuple:
    """(objects, container): the objects lying INSIDE a container that match `query` (keys: id, color,
    kind, nearest_to, largest) and the container they lie in. Raises LookupError with a reason that
    starts with "container not found", "target not found" or "not in a container".

    A detection rated below min_confidence is taken only by an explicit id."""
    query = query or {}

    def where(o):
        return f"({o.center[0]:.3f}, {o.center[1]:.3f})"

    if not sc.containers():
        seen = ", ".join(f"{o.id}:{o.color} {o.kind}" for o in sc.objects) or "nothing"
        raise LookupError(f"container not found: the camera sees no container (in view: {seen})")
    if query.get("id") is not None:
        hit = [o for o in sc.objects if o.id == query["id"]]
        if not hit:
            raise LookupError(f"target not found: no object with id {query['id']} in view")
        o = hit[0]
        if o.kind == "container":
            raise LookupError(f"target not found: object {o.id} is a container")
        if o.in_container is None:
            raise LookupError(f"not in a container: object {o.id} ({o.color} {o.kind}) stands on the table at "
                              f"{where(o)}; `fd pick` moves things that are on the table")
        con = [c for c in sc.containers() if c.id == o.in_container]
        if not con:
            raise LookupError(f"container not found: object {o.id} is marked as inside container {o.in_container}, "
                              "which is not in view")
        if container_query:
            try:
                wanted = [c.id for c in select_objects(sc, container_query, containers=True)]
            except LookupError as e:
                raise LookupError(f"container not found: {e}") from None
            if con[0].id not in wanted:
                raise LookupError(f"not in a container: object {o.id} lies in container {con[0].id}, not in the "
                                  f"one asked for {container_query}")
        return [o], con[0]
    try:
        cons = select_objects(sc, container_query or {}, containers=True)
    except LookupError as e:
        raise LookupError(f"container not found: {e}") from None
    ids = {c.id for c in cons}
    inside = [o for o in sc.objects if o.kind != "container" and o.in_container in ids]

    def match(o):
        return ((not query.get("color") or _color_is(o, query["color"]))
                and (not query.get("kind") or o.kind == query["kind"]))

    cands = [o for o in inside if match(o)]
    if not cands:
        named = bool(query.get("color") or query.get("kind"))
        outside = [o for o in sc.graspable() if match(o)] if named else []
        if outside:
            o = outside[0]
            raise LookupError(f"not in a container: the {o.color} {o.kind} (id {o.id}) stands on the table at "
                              f"{where(o)}, not inside the container")
        seen = ", ".join(f"{o.id}:{o.color} {o.kind}" for o in inside) or "nothing"
        raise LookupError(f"target not found: nothing inside the container matches {query} (inside: {seen})")
    sure = [o for o in cands if float(o.confidence) >= min_confidence]
    if not sure:
        unsure = "; ".join(f"id {o.id} ({o.color} {o.kind}: {', '.join(str(n) for n in o.notes[:2]) or 'uncertain'})"
                           for o in cands[:4])
        raise LookupError(f"target not found: only uncertain detections inside the container match {query}: "
                          f"{unsure}. If one of them is right, take it by id")
    cands = sure
    if query.get("nearest_to") is not None:
        x, y = query["nearest_to"]
        cands = sorted(cands, key=lambda o: math.hypot(o.center[0] - x, o.center[1] - y))[:1]
    elif query.get("largest"):
        cands = sorted(cands, key=lambda o: -o.size[0] * o.size[1])[:1]
    return cands, [c for c in cons if c.id == cands[0].in_container][0]


@dataclass
class TakePlan:
    """Everything the take-out sequence will command, planned and checked before it moves."""
    target: object
    container: object
    cand: object  # takeout.TakeGrasp
    place: dict  # {"xy": (x, y), ...}: where the object's centre is put down
    roll: float
    hover: np.ndarray
    above_hover: np.ndarray
    tip_grasp: np.ndarray  # from the scene's heights; the touch probe replaces z
    probe: bool
    tip_probe: object = None  # where the touch descent starts (None: from the hover point)
    z_lift: float = 0.0  # fingertip height at which the object's bottom clears the rim
    tip_place: object = None
    approach: list = field(default_factory=list)  # joint waypoints: above-hover ... hover
    descent: list = field(default_factory=list)  # hover ... probe start (probe) | grasp
    lift: list = field(default_factory=list)  # grasp ... a little up
    rise: list = field(default_factory=list)  # ... clear of the rim
    carry: list = field(default_factory=list)  # ... above the place spot ... place
    retreat: list = field(default_factory=list)  # place ... straight up ... stand-by pose
    z_clear: float = 0.0
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"hover": [round(float(v), 4) for v in self.hover],
                "tip_grasp": [round(float(v), 4) for v in self.tip_grasp], "probe": bool(self.probe),
                "z_lift": round(float(self.z_lift), 4),
                "tip_place": None if self.tip_place is None else [round(float(v), 4) for v in self.tip_place],
                "waypoints": {"approach": len(self.approach), "descent": len(self.descent), "lift": len(self.lift),
                              "rise": len(self.rise), "carry": len(self.carry), "retreat": len(self.retreat)},
                "notes": list(self.notes)}


def _emit(sess, name: str, **info):
    """Tell a listener what the skill is about to do (the simulation uses it to put a surface under
    the probe and an object between the jaws). No listener = nothing happens."""
    fn = getattr(sess, "on_event", None)
    if fn is not None:
        fn(name, info)


def _jaws(arm) -> dict:
    """Gripper position and load, as they are now."""
    raw = _gripper_raw(arm)
    try:
        load = int(arm.bus.read("gripper", "Present_Load"))
    except Exception:  # a stand-in bus without that register
        load = None
    return {"raw": raw, "load": load}


def _held(g: dict, now: dict, load_ref=None) -> tuple:
    """Is the object still between the jaws? g: what the closing returned (pos_raw, hold_goal_raw).
    While it is held the jaws stay where they stopped and the servo keeps pressing; without it they
    close on to the holding goal and the load drops (2026-11-11: held 2075 / load 160 all the way;
    slipped 2081 -> 2058, load 172 -> 80; jaws resting on each other read about 100). Returns
    (held, numbers)."""
    stall = int(g["pos_raw"])
    hold = int(g.get("hold_goal_raw") if g.get("hold_goal_raw") is not None else stall - 40)
    lowest = stall - max(8, (stall - hold) // 2)
    ok = now["raw"] >= lowest
    info = {"raw": now["raw"], "raw_min": lowest, "load": now["load"]}
    if load_ref is not None and now["load"] is not None and abs(load_ref) >= 60:
        info["load_min"] = int(round(0.7 * abs(load_ref)))
        ok = ok and abs(now["load"]) >= info["load_min"]
    return bool(ok), info


def _exact_roll(arm, xyz, azimuth_deg: float, roll0: float) -> float:
    """Wrist roll that makes the jaws open towards azimuth_deg with the fingertip at xyz, gripper
    straight down. The closed form needs the pan angle, which depends a little on the roll (the
    fingertip is off the roll axis): two rounds through the arm's solver settle it. Raises
    SafetyError when the pose or the roll is out of reach."""
    roll = float(roll0)
    for _ in range(2):
        q = arm.solve(np.asarray(xyz, dtype=float), 90.0, roll)
        roll = float(K.roll_for_azimuth(float(azimuth_deg), math.degrees(float(q[0])), 90.0))
    q = arm.solve(np.asarray(xyz, dtype=float), 90.0, roll)
    jaw = K.jaw_open_axis(K.fk(q))
    off = (math.degrees(math.atan2(jaw[1], jaw[0])) - float(azimuth_deg) + 180.0) % 360.0 - 180.0
    if abs(off) > 1.5:
        raise SafetyError(f"jaw azimuth {azimuth_deg:.0f} deg cannot be realised here (off by {off:.1f} deg)")
    return roll


def _needs_probe(cand, opts: TakeOptions) -> bool:
    if opts.probe == "off":
        return False
    return opts.probe == "on" or float(cand.margin) < opts.probe_margin - 1e-9


class _SpotError(SafetyError):
    """A path cannot be planned because of the PLACE spot (another spot may do)."""


def _after_grasp(arm, sc, target, container, cand, spot: dict, opts: TakeOptions, q_grasp, roll: float) -> dict:
    """The way out, from the pose q_grasp at which the jaws hold the object: a little lift, straight up
    until the object's bottom clears the rim, across to above the place spot, down, and (after the
    release) straight up and to the stand-by pose. Every part is checked the way run_path will check
    it. Used before anything moves (with the planned grasp pose) and again once the object is held
    (with the measured one). Raises SafetyError (_SpotError when it is the place spot that does not work)."""
    tool = arm.tip_tool()
    sp = arm.speeds(opts.tier)
    box_ko = container_keepouts(sc, container)
    q_grasp = np.asarray(q_grasp, dtype=float)
    tip = K.fk(q_grasp, tool)[:3, 3].copy()
    rim_z = float((container.inner or {}).get("rim_z", container.z_top))
    h_below = float(cand.h_below)
    out = {"notes": []}
    out["lift"] = _tip_path(arm, [(tip + np.array([0.0, 0.0, opts.lift_check]), 90.0, roll)], q_grasp)
    check_path(arm, [q_grasp] + out["lift"], sp["approach"], opts, ignore=box_ko, gripper_raw=cand.expected_raw,
               what="lift")
    others = [o.z_top for o in sc.objects if o.id != target.id and o.kind != "container"]
    z_over = max(others + [sc.table_z]) + h_below + opts.carry_clearance
    # where the fingertip stands when the object's centre is on the spot: the same wrist roll gives
    # another jaw direction there (the pan angle differs), so the offset is turned with the gripper
    u_g = np.array([math.cos(math.radians(cand.jaw_azimuth_deg)), math.sin(math.radians(cand.jaw_azimuth_deg))])
    off = np.array(cand.object_offset, dtype=float)
    along, across = float(off @ u_g), float(off @ np.array([-u_g[1], u_g[0]]))
    spot_xy = np.array(spot["xy"], dtype=float)
    z_place = float(sc.table_z) + h_below + opts.place_gap
    xy = spot_xy - off
    try:
        for _ in range(3):
            jaw = K.jaw_open_axis(K.fk(arm.solve(np.array([xy[0], xy[1], z_place]), 90.0, roll)))
            u_p = np.array([jaw[0], jaw[1]]) / max(math.hypot(jaw[0], jaw[1]), 1e-9)
            xy = spot_xy - along * u_p - across * np.array([-u_p[1], u_p[0]])
    except SafetyError as e:
        raise _SpotError(f"the place spot ({spot_xy[0]:.3f}, {spot_xy[1]:.3f}) cannot be reached straight down: {e}")
    tip_place = np.array([xy[0], xy[1], z_place])
    err, spot_fault = None, False
    for rc in (opts.rim_clear, 0.5 * (opts.rim_clear + opts.rim_clear_min), opts.rim_clear_min):
        z_lift = max(rim_z + rc + h_below, z_over, tip[2] + opts.lift_check + 0.002)
        try:
            spot_fault = False
            rise = _tip_path(arm, [(np.array([tip[0], tip[1], z_lift]), 90.0, roll)], out["lift"][-1])
            check_path(arm, [out["lift"][-1]] + rise, sp["transit"], opts, ignore=box_ko, gripper_raw=cand.expected_raw,
                       what="lift")
            spot_fault = True
            above = np.array([xy[0], xy[1], z_lift])
            q_over = arm.solve(above, 90.0, roll, q_seed=rise[-1])
            down = _tip_path(arm, [(tip_place, 90.0, roll)], q_over)
            for s in np.linspace(0.0, 1.0, 13)[1:-1]:
                z = float(K.fk(rise[-1] + s * (q_over - rise[-1]), tool)[2, 3])
                if z < z_lift - 0.006:
                    raise SafetyError(f"the crossing dips {1000 * (z_lift - z):.0f} mm below the carry height")
            check_path(arm, [rise[-1], q_over] + down, sp["transit"], opts, ignore=box_ko,
                       gripper_raw=cand.expected_raw, what="carry")
        except SafetyError as e:
            err = e
            continue
        if rc < opts.rim_clear:
            out["notes"].append(f"object bottom only {1000 * rc:.0f} mm above the rim while it is carried out (reach)")
        out.update(rise=rise, carry=[q_over] + down, z_lift=float(z_lift), tip_place=tip_place, above_place=above)
        break
    else:
        if spot_fault:
            raise _SpotError(f"no way from above the container to the place spot ({spot_xy[0]:.3f}, "
                             f"{spot_xy[1]:.3f}): {err}")
        raise SafetyError(f"the object cannot be lifted clear of the rim here: {err}")
    z_clear = max(o.z_top for o in sc.objects) + opts.transit_clearance
    try:
        up = _tip_path(arm, [(out["above_place"], 90.0, roll)], out["carry"][-1])
        out["retreat"] = up + plan_transit(arm, up[-1], standby_q(arm, up[-1]), z_clear)
        check_path(arm, [out["carry"][-1]] + out["retreat"], sp["transit"], opts, gripper_raw=cand.release_raw,
                   what="retreat")
    except SafetyError as e:
        raise _SpotError(f"no way back from the place spot ({spot_xy[0]:.3f}, {spot_xy[1]:.3f}): {e}")
    out["z_clear"] = float(z_clear)
    return out


def build_take_plan(arm, sc, target, container, cand, spot: dict, opts: TakeOptions, corr=(0.0, 0.0),
                    q_hover=None) -> TakePlan:
    """Plan and check every path of the take-out for one way of holding the object and one place
    spot. `corr` shifts everything below the hover point in xy (what the look at the jaws measured);
    q_hover = the measured pose when the arm already stands at the hover point. The hover point is
    above the rim; nothing here goes below it except the straight descent. Raises SafetyError when
    any part is infeasible. No motion."""
    box_ko = container_keepouts(sc, container)
    sp = arm.speeds(opts.tier)
    corr3 = np.array([float(corr[0]), float(corr[1]), 0.0])
    hover = np.array(cand.tip_hover, dtype=float)
    above_hover = hover + np.array([0.0, 0.0, opts.approach_drop])
    tip_grasp = np.array(cand.tip_grasp, dtype=float) + corr3
    # (the same jaw direction needs another wrist roll at the shifted position: the pan angle differs)
    roll = _exact_roll(arm, tip_grasp, cand.jaw_azimuth_deg, cand.roll_deg)
    plan = TakePlan(target=target, container=container, cand=cand, place=spot, roll=roll, hover=hover,
                    above_hover=above_hover, tip_grasp=tip_grasp, probe=_needs_probe(cand, opts))
    q_above = arm.solve(above_hover, 90.0, roll)
    plan.approach = [q_above] + _tip_path(arm, [(hover, 90.0, roll)], q_above)
    q_h = plan.approach[-1] if q_hover is None else np.asarray(q_hover, dtype=float)
    # descent: (sideways by the visual correction at hover height, then) straight down
    pts = []
    if float(np.linalg.norm(corr3)) > 1e-6:
        pts.append((hover + corr3, 90.0, roll))
    if plan.probe:
        # down to where the touch descent starts (not below the rim); the last stretch is planned again
        # from the touched height
        rim_z = max(float((container.inner or {}).get("rim_z", container.z_top)), float(target.z_top))
        z_start = max(float(cand.z_surface) + float(cand.contact_offset) + opts.probe_above, rim_z + 0.002)
        if z_start < hover[2] - 0.001:
            plan.tip_probe = np.array([tip_grasp[0], tip_grasp[1], z_start])
            pts.append((plan.tip_probe, 90.0, roll))
        plan.descent = _tip_path(arm, pts, q_h) if pts else []
        q_p = plan.descent[-1] if plan.descent else q_h
        final = _tip_path(arm, [(tip_grasp, 90.0, roll)], q_p)
    else:
        pts.append((tip_grasp, 90.0, roll))
        plan.descent, final = _tip_path(arm, pts, q_h), []
    q_grasp = (final or plan.descent)[-1]
    after = _after_grasp(arm, sc, target, container, cand, spot, opts, q_grasp, roll)
    plan.lift, plan.rise, plan.carry, plan.retreat = after["lift"], after["rise"], after["carry"], after["retreat"]
    plan.z_lift, plan.tip_place, plan.z_clear = after["z_lift"], after["tip_place"], after["z_clear"]
    plan.notes += after["notes"]
    look_raw = arm.gripper_raw_for(0.0) if (cand.look_closed and opts.visual != "off") else cand.open_raw
    check_path(arm, plan.approach, sp["transit"], opts, gripper_raw=look_raw, what="approach")
    if plan.descent:
        check_path(arm, [q_h] + plan.descent, sp["approach"], opts, ignore=box_ko, gripper_raw=cand.open_raw,
                   what="descent")
    if final:
        check_path(arm, [plan.descent[-1] if plan.descent else q_h] + final, sp["precise"], opts, ignore=box_ko,
                   gripper_raw=cand.open_raw, what="descent")
    return plan


def _place_view(sess, S, shape_hw, target, table_z: float):
    """ok(xy) -> "" or why not: will the camera, with the arm back at the stand-by pose, see an object
    put down at xy (inside the area perception watches, inside the image, not behind the arm)? None
    when that cannot be told (no camera model, or a stand-in for perception)."""
    cam, arm = sess.cam, sess.arm
    if cam is None or shape_hw is None or not hasattr(S, "arm_mask"):
        return None
    try:
        h, w = int(shape_hw[0]), int(shape_hw[1])
        k = 0.25
        mask = S.arm_mask((int(round(h * k)), int(round(w * k))), cam, arm.ready_q(), None, scale=k)
        ws = (arm.cfg.get("scene") or {}).get("workspace") or (getattr(S, "DEFAULTS", None) or {}).get("workspace")
        cam_x = float(cam.center[0])
    except Exception:  # an odd camera model or stand-in: no view test rather than no take-out
        return None
    r = 0.5 * max(float(target.size[0]), float(target.size[1]))
    z_lo, z_hi = float(table_z), float(table_z) + float(target.size[2])

    def ok(xy):
        x, y = float(xy[0]), float(xy[1])
        if ws is not None:  # (scene.py also stops watching 10 cm in front of the camera)
            x1 = min(float(ws[1]), cam_x - 0.10) if cam_x > float(ws[0]) + 0.2 else float(ws[1])
            if not (float(ws[0]) + r <= x <= x1 - r and float(ws[2]) + r <= y <= float(ws[3]) - r):
                return "outside the area the camera watches"
        pts = [(x, y, z_lo), (x, y, z_hi), (x + r, y, z_hi), (x - r, y, z_hi), (x, y + r, z_hi), (x, y - r, z_hi)]
        uv = np.asarray(cam.project(pts), dtype=float)
        if (not np.all(np.isfinite(uv)) or uv[:, 0].min() < 20 or uv[:, 0].max() > w - 20
                or uv[:, 1].min() < 20 or uv[:, 1].max() > h - 20):
            return "outside the camera image"
        px = np.rint(uv * k).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, mask.shape[1] - 1)
        px[:, 1] = np.clip(px[:, 1], 0, mask.shape[0] - 1)
        if mask[px[:, 1], px[:, 0]].any():
            return "hidden from the camera by the arm at its stand-by pose"
        return ""

    return ok


def plan_take(arm, sess, sc, targets, container, opts: TakeOptions, P, view_for=None) -> tuple:
    """Best feasible (TakePlan, keep-outs, report) over the given objects, the ways of holding each
    (takeout.plan_grasps) and the free spots on the table (takeout.choose_place). Raises
    StageError("plan") with what is in the way and, where there is one, a next step.
    view_for(target) -> ok(xy) or None: the camera's say on a place spot (see _place_view)."""
    cfg = arm.cfg
    tool, lim = arm.tip_tool(), arm.q_limits()
    t_off = np.linalg.inv(K.tool_offset()) @ np.asarray(tool, dtype=float)
    tip_offset = (float(t_off[0, 3]), float(t_off[1, 3]), float(t_off[2, 3]))
    cam_xy = None
    try:
        c = sess.cam.center
        cam_xy = (float(c[0]), float(c[1]))
    except Exception:  # no camera model (tests): no preference for directions the camera sees well
        pass

    def reach(xyz, roll_deg):
        _q, info = K.ik(np.asarray(xyz, dtype=float), 90.0, math.radians(float(roll_deg)), tool=tool, limits=lim)
        return "" if info["ok"] else str(info["reason"])

    params = {"rim_margin": opts.rim_margin, "rim_clear": opts.rim_clear, "rim_clear_min": opts.rim_clear_min,
              "place_clear": opts.place_clear, "approach_drop": opts.approach_drop}
    x_pan = float(K._X_PAN)
    rim_z = float((container.inner or {}).get("rim_z", container.z_top))
    z_put, z_over = float(sc.table_z) + 0.005, rim_z + opts.rim_clear_min + 0.005  # about where a spot is used
    track = _park_track(arm)

    def parked(xy, radius):
        """"" or why an object of that radius must not be put down at xy: it would stand where the arm
        folds down onto its rest pose (park() and the fold for a clear view know nothing of it)."""
        if track is None:
            return ""
        d = _park_dist(track, xy) - radius
        if d >= opts.park_clear:
            return ""
        return (f"where the arm folds down to park (the object would stand {max(d, 0.0) * 1000:.0f} mm from the "
                f"parked gripper, {opts.park_clear * 1000:.0f} mm needed)")

    def spot_test(view, radius, reach_too=True):
        """ok_xy for choose_place: a spot the arm can reach straight down, with 2 cm to spare towards and
        away from the robot (the fingertip stands beside the object's centre), that is not where the arm
        parks, and that the camera can see. reach_too=False: a spot given by hand (the path planner has
        the last word on its reach)."""
        def ok(xy):
            x, y = float(xy[0]), float(xy[1])
            d = math.hypot(x - x_pan, y)
            ux, uy = ((x - x_pan) / d, y / d) if d > 1e-6 else (1.0, 0.0)
            for m in ((0.0, 0.02, -0.02) if reach_too else ()):
                for z in (z_put, z_over):
                    if reach((x + m * ux, y + m * uy, z), 0.0):
                        return "the arm cannot point straight down there"
            return parked(xy, radius) or (view(xy) if view is not None else "")
        return ok

    report, best, problems, hints = [], None, [], []
    for target in targets:
        saved = cfg.get("keepouts", {})
        cfg["keepouts"] = {**{k: v for k, v in saved.items() if not k.startswith("obj")},
                           **scene_keepouts(sc, skip_ids=(target.id,))}
        try:
            cands = P.plan_grasps(target, container, sc, cfg, hollow=opts.hollow, assume_hollow=opts.assume_hollow,
                                  mode=opts.mode, tip_offset=tip_offset, camera_xy=cam_xy, reach=reach, params=params)
            good = [c for c in cands if c.ok]
            rows = [{"object": target.id, "mode": c.mode, "ok": True, "azimuth": round(float(c.jaw_azimuth_deg), 1),
                     "score": round(float(c.score), 2), "reason": ""} for c in good[:6]]
            seen = set()
            for c in cands:
                if not c.ok and c.reason not in seen and len(seen) < 4:
                    seen.add(c.reason)
                    rows.append({"object": target.id, "mode": c.mode, "ok": False,
                                 "azimuth": round(float(c.jaw_azimuth_deg), 1), "score": 0.0, "reason": c.reason[:200]})
            report += rows
            if not good:
                ex = P.explain(cands)
                problems.append(ex["reason"])
                # the planner does not know which object it is: name it, so that the hint is the next command
                hints.append(str(ex["next"]).replace("`fd take --hollow`", f"`fd take --id {target.id} --hollow`"))
                continue
            view = view_for(target) if (view_for is not None and opts.verify) else None
            # a spot given by hand is judged by the path planner itself (below); it must still be clear of
            # where the arm parks, and the camera must still see it
            given = opts.to is not None
            ok_xy = spot_test(view, _radius(target), reach_too=not given)
            if given and view is None and track is None:
                ok_xy = None  # (nothing to ask: no camera's say and no rest pose)
            spots, no_spot = P.choose_place(sc, target, container, cfg, to=opts.to, params=params, ok_xy=ok_xy)
            if not spots:
                problems.append(no_spot)
                hints.append("give another spot with --to x,y, or clear some space on the table"
                             if opts.to is not None else "clear some space on the table next to the container")
                continue
            last, last_is_reach = None, False
            for i, c in enumerate(good[:6]):
                if best is not None and best[0] >= c.score:
                    break
                for spot in spots[:5]:
                    try:
                        plan = build_take_plan(arm, sc, target, container, c, spot, opts)
                    except _SpotError as e:
                        last, last_is_reach = str(e), False
                        continue
                    except SafetyError as e:  # this way of holding it does not work, whatever the spot
                        last = str(e)
                        last_is_reach = any(k in last for k in ("IK failed", "out of reach", "cannot be realised"))
                        break
                    best = (float(c.score), plan, dict(cfg["keepouts"]))
                    break
                if best is None or best[1].cand is not c:
                    rows[i].update(ok=False, reason=f"path: {last}")
                    continue
                break
            else:
                if last:
                    problems.append(f"no feasible path: {last}")
                    # the way in or out leaves the arm's reach: room around the object or another place spot
                    # would not help, and the hint must not say so
                    hints.append(getattr(P, "REACH_NEXT", {}).get("", "") if last_is_reach else "")
        finally:
            cfg["keepouts"] = saved
    if best is None:
        reason = "; ".join(dict.fromkeys(p for p in problems if p))[:500] or "the planner returned no candidates"
        detail = {"candidates": report[:10]}
        nxt = next((h for h in hints if h), "")
        if nxt:
            detail["next"] = nxt
        raise StageError("plan", reason, detail)
    return best[1], best[2], report


def _fold_back(arm, tier: str | None = None) -> dict:
    """Fold the arm back onto its rest pose WITHOUT switching the torque off (the way park() goes
    there): out of the camera's view of anything that stands close to the robot."""
    sp = arm.speeds(tier)
    if hasattr(arm, "apply_profile"):
        sp = arm.apply_profile(tier)
    q = arm.read_q()
    z_clear = float(arm.cfg.get("table_z", 0.0)) + 0.06
    wps = [q] + plan_transit(arm, q, arm.ready_q(), z_clear) + [arm.rest_q()]
    return arm.run_path(wps, min(sp["transit"], 30.0), allow_contact=True, settle=False,
                        gripper_goal_raw=arm.gripper_raw_for(0.1))


def _look_at_jaws(sess, S, img0, opts: TakeOptions) -> tuple:
    """One look at the jaws: (d, step, problem, frame). d = where the camera sees the gripper minus
    where the joint angles put it (base xy, m); problem = why this look cannot be used (None = ok)."""
    arm, cam, cfg = sess.arm, sess.cam, sess.arm.cfg
    img1 = None
    try:
        img1 = sess.frame(fresh=True)
        dj = S.detect_jaws(img1, img0, cam, arm.read_q(), arm.gripper_q_deg(_gripper_raw(arm)), cfg)
        if not isinstance(dj, dict):
            dj = {"ok": False, "confidence": 0.0, "dxy_base": None}
    except MOTION_ERRORS:
        raise
    except Exception as e:  # camera or perception problems must not crash a motion sequence
        dj = {"ok": False, "confidence": 0.0, "dxy_base": None, "reason": f"{type(e).__name__}: {e}"}
    try:
        conf = float(dj.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    dxy = dj.get("dxy_base")
    try:
        dxy = None if dxy is None else [float(dxy[0]), float(dxy[1])]
        if dxy is not None and not all(math.isfinite(v) for v in dxy):
            dxy = None
    except (TypeError, ValueError, IndexError):
        dxy = None
    usable = bool(dj.get("ok")) and dxy is not None and conf >= opts.min_visual_conf
    # with only the moving jaw in view the offset ALONG the opening direction is not measured (it is
    # reported as zero): exactly the direction in which a jaw must land beside a wall, not on it
    full = bool((dj.get("fixed") or {}).get("visible", True))
    step = {"ok": bool(dj.get("ok")), "confidence": round(conf, 2), "full": full,
            "dxy_mm": None if dxy is None else [round(v * 1000, 1) for v in dxy]}
    # How far out in its search window the detector found the jaw that carries the answer (1.0 = at the
    # edge). It looks only that far around where the joint angles put the jaws (scene._JAW_SEARCH_PX), so
    # a match near the edge may be the visible end of a larger offset.
    try:
        shift = (dj.get("fixed") or {}).get("shift_px")
        window = float(dj.get("search_px") or getattr(S, "_JAW_SEARCH_PX", 36))
        if shift is not None and window > 0:
            step["window"] = round(max(abs(float(shift[0])), abs(float(shift[1]))) / window, 2)
    except (TypeError, ValueError, IndexError):
        pass
    problem = None
    if not usable:
        why = str(dj.get("reason") or "")[:160]
        problem = "the jaws could not be measured in the camera image" + (f" ({why})" if why else "")
    elif not full:
        problem = ("the fixed jaw is hidden from the camera, so where it stands along the closing direction "
                   "cannot be measured")
    return (None if dxy is None else np.array(dxy, dtype=float)), step, problem, img1


def _align_by_look(sess, S, plan: TakePlan, opts: TakeOptions, res: dict, sp: dict, img0) -> np.ndarray:
    """Look at the jaws at the hover point (above the rim) and return the xy correction that puts
    the camera-seen jaws where the plan wants them; the correction is driven and confirmed by
    another look. Jaws that cannot be measured are looked at again closed, then once more from
    opts.visual_raise higher. For a take-out the look is not optional: if they still cannot be
    measured the arm goes back to the stand-by pose and StageError("visual") says so. The same when
    the reading is larger than opts.max_visual_corr, lies at the edge of what the detector can measure
    (opts.visual_edge), differs from the look before by more than opts.visual_repeat_tol, or is a
    correction that no further look has confirmed. With opts.visual == "off" nothing is looked at
    (take_out lets that through only for a simulated arm)."""
    arm, cand = sess.arm, plan.cand
    roll = plan.roll
    vis = {"mode": opts.visual, "seen": False, "applied": False, "closed": False, "raised": False, "steps": []}
    res["checks"]["visual"] = vis
    corr = np.zeros(2)
    if opts.visual == "off":
        vis["note"] = ("not looked at: the look at the jaws is switched off (visual = \"off\"), "
                       + ("and this is a simulated arm, which no camera sees" if _simulated(arm)
                          else "although this is NOT a simulated arm"))
        return corr
    vis["closed"] = bool(cand.look_closed)
    back_up = [(plan.above_hover, 90.0, roll)]

    def stop(problem, img=None):
        if img is not None:
            sess.save("visual_fail", img)
        _retreat(arm, sp, res, via=back_up)
        raise StageError("visual", problem + "; not descending into the container", dict(vis))

    closed = bool(cand.look_closed)
    look_z = float(plan.hover[2])  # the height the jaws are looked at (never below the hover point)
    d_prev, residual = None, None
    n_looks = max(1, opts.visual_iters)
    for it in range(n_looks):
        d, step, problem, img1 = _look_at_jaws(sess, S, img0, opts)
        vis["steps"].append(step)
        if problem and not closed and it == 0:
            # once more with the jaws closed: the camera then sees one body instead of two fingers
            # of which one can hide the other
            arm.set_gripper(raw=arm.gripper_raw_for(0.0), wait=True)
            closed = vis["closed"] = True
            d, step, problem, img1 = _look_at_jaws(sess, S, img0, opts)
            step["retry_closed"] = True
            vis["steps"].append(step)
        if problem and it == 0 and opts.visual_raise > 0:
            # and once more from higher up, where the jaws stand in front of the far wall or the table
            # instead of what lies in the container (2026-11-11: closed jaws 7 mm above the rim, over the
            # white cap, could not be measured; 25 mm and more above the rim they always could)
            hv = plan.hover.copy()
            hv[2] += opts.visual_raise
            try:
                arm.move_tip_path([(hv, 90.0, roll)], sp["approach"], settle=0.6)
            except SafetyError:
                pass  # no room above the hover point: the problem stands
            else:
                look_z, vis["raised"] = float(hv[2]), True
                d, step, problem, img1 = _look_at_jaws(sess, S, img0, opts)
                step["raised_mm"] = round(opts.visual_raise * 1000)
                vis["steps"].append(step)
        if problem:
            stop(problem, img1)
        if step.get("window") is not None and step["window"] >= opts.visual_edge:
            stop(f"the jaws were found at the edge of what the look can measure ({step['window'] * 100:.0f} % of the "
                 f"detector's search window, reading {np.linalg.norm(d) * 1000:.0f} mm): the real offset may be "
                 "larger (calibration?)", img1)
        if float(np.linalg.norm(d)) > opts.max_visual_corr:
            stop(f"the jaws appear {np.linalg.norm(d) * 1000:.0f} mm from where the joint angles put them "
                 f"(more than {opts.max_visual_corr * 1000:.0f} mm: calibration?)", img1)
        if d_prev is not None and float(np.linalg.norm(d - d_prev)) > opts.visual_repeat_tol:
            stop(f"two looks at the jaws disagree by {np.linalg.norm(d - d_prev) * 1000:.1f} mm", img1)
        residual = float(np.linalg.norm(corr + d))  # camera-seen jaws minus the planned position
        d_prev = d
        vis["seen"] = True
        if residual <= opts.visual_tol:
            break
        corr = -d
        if it + 1 < n_looks:
            hv = plan.hover.copy()
            hv[:2] += corr
            hv[2] = look_z
            try:  # the jaw direction stays as planned: the wrist roll follows the pan angle
                roll = _exact_roll(arm, hv, cand.jaw_azimuth_deg, roll)
                arm.move_tip_path([(hv, 90.0, roll)], sp["precise"], settle=0.6)
            except SafetyError as e:
                stop(f"the position corrected by the look ({corr[0] * 1000:+.0f}, {corr[1] * 1000:+.0f} mm) "
                     f"is not feasible: {e}", img1)
    else:
        # every look asked for a (further) correction, so the last one was never confirmed by a look: two
        # consecutive looks that do not agree are no measurement to descend on
        stop(f"the correction by the look is not confirmed: after {n_looks} look(s) the jaws still read "
             f"{residual * 1000:.1f} mm from where the correction should have put them", img1)
    vis["corr_mm"] = [round(float(v) * 1000, 1) for v in corr]
    vis["residual_mm"] = None if residual is None else round(residual * 1000, 1)
    vis["applied"] = bool(float(np.linalg.norm(corr)) > 1e-6)
    if closed:  # open before going down between the object and its neighbours
        arm.set_gripper(raw=cand.open_raw, wait=True)
    return corr


def _back_out(arm, sp: dict, res: dict, plan: TakePlan, corr, box_ko, open_raw=None) -> bool:
    """Nothing in the jaws: straight up out of the container, then to the stand-by pose."""
    c3 = np.array([float(corr[0]), float(corr[1]), 0.0])
    try:
        z = float(K.fk(arm.read_q(), arm.tip_tool())[2, 3])
    except MOTION_ERRORS:
        z = -1.0
    via = [(plan.hover + c3, 90.0, plan.roll)] if z < plan.hover[2] - 0.002 else []
    return _retreat(arm, sp, res, via=via, ignore=box_ko, open_raw=open_raw)


def _probe_height(sess, plan: TakePlan, opts: TakeOptions, res: dict, box_ko, sp: dict):
    """Touch before the last stretch of the descent (optional: opts.probe, off by default).

    As a CHECK (what it is unless opts.probe_may_lower): the open jaws are lowered slowly to
    opts.probe_press past the height at which their lowest point should land (inside the cavity for
    a wall pinch, in the free slot beside the object otherwise). Something met clearly above that
    height - the object's rim, a closed top, a neighbour - stops the take-out; otherwise the grasp
    height stays the one from the scene and None is returned. The touched height is NOT used:
    Arm.probe_down reports its contact 3-18 mm deeper than the first touch (measured 2026-11-11).

    As a MEASUREMENT (opts.probe_may_lower; for a probe that reports the first touch): the descent
    may go opts.probe_below past the expected height, is repeated once with a longer drop when it
    ends without a clear contact, and the touched height is returned (it must lie within
    opts.probe_max_dev of the scene's).

    Raises StageError("probe") with the reason; the arm then stands above where it probed."""
    arm, cand = sess.arm, plan.cand
    tool = arm.tip_tool()
    measure = bool(opts.probe_may_lower)
    expected = float(cand.z_surface) + float(cand.contact_offset)
    detect = 0.004 + 0.35 * opts.probe_speed + 0.001  # what Arm.probe_down needs beyond the surface to be sure
    min_drop = opts.probe_speed * 2.45 + 0.004 + 0.002  # shortest descent it accepts
    run_in = opts.probe_speed * 2.1 + 0.002  # ... and how far above a surface it has to start
    where = "the bottom of the cavity" if cand.mode == "pinch" else "the floor beside the object"
    chk = {"mode": "measure" if measure else "check", "expected_tip_z": round(expected, 4), "tries": []}
    res["checks"]["probe"] = chk
    _emit(sess, "probe", expected_tip_z=expected)
    saved = arm.cfg.get("keepouts", {})
    # probe_down has no way to ignore keep-outs: the container's own box (and what lies in it) is
    # taken out of the list while the fingertip is lowered into it
    arm.cfg["keepouts"] = {k: v for k, v in saved.items() if k not in box_ko}
    z, pr = None, {}
    try:
        start = K.fk(arm.read_q(), tool)[:3, 3].copy()
        drop = max(min_drop, float(start[2] - expected) + (opts.probe_below if measure else opts.probe_press))
        for n in ((1, 2) if measure else (1,)):
            try:
                pr = arm.probe_down(max_drop_m=drop, speed_m_s=opts.probe_speed, backoff_m=opts.probe_backoff)
            except SafetyError as e:
                # refused before moving (the bottom of the range is out of reach?): the shortest range
                # that still detects a surface at the expected height
                short = max(min_drop, float(K.fk(arm.read_q(), tool)[2, 3] - expected) + detect + 0.003)
                if not measure or short >= drop - 1e-4:
                    raise StageError("probe", f"the touch descent was refused: {e}", dict(chk))
                try:
                    pr = arm.probe_down(max_drop_m=short, speed_m_s=opts.probe_speed, backoff_m=opts.probe_backoff)
                    drop = short
                except SafetyError as e2:
                    raise StageError("probe", f"the touch descent was refused: {e2}", dict(chk))
            row = {"drop_mm": round(drop * 1000, 1), "contact": pr.get("contact"),
                   "contact_tip_z": None if pr.get("contact_tip_z") is None else round(float(pr["contact_tip_z"]), 4),
                   "lowest_tip_z": pr.get("lowest_tip_z"), "reason": pr.get("reason")}
            chk["tries"].append(row)
            now = K.fk(arm.read_q(), tool)[:3, 3]
            slide = float(np.hypot(now[0] - start[0], now[1] - start[1]))
            chk["slide_mm"] = round(slide * 1000, 1)
            if pr.get("contact") is True and pr.get("contact_tip_z") is not None:
                z = float(pr["contact_tip_z"])
                break
            if not measure:
                break
            if n == 2:
                low = pr.get("lowest_tip_z")
                raise StageError("probe", "no clear contact in two touch descents (the first: "
                                 f"{chk['tries'][0]['reason'] or 'free travel to the end of its range'}; the second: "
                                 f"{row['reason'] or 'free travel'}, lowest fingertip height "
                                 f"{'?' if low is None else format(float(low), '.4f')} m, expected contact at "
                                 f"{expected:.4f} m). Check the height calibration with `fd probe` over the table",
                                 dict(chk))
            # once more, deeper: up to where the detector has its run-in above the lowest point of the
            # first try, then down to `probe_retry_extra` (plus what the detector needs) below it
            low = pr.get("lowest_tip_z")
            low = float(low) if low is not None else float(now[2]) - opts.probe_backoff
            if now[2] < low + run_in:
                try:  # (straight up from where the first try ended, which may be below the calibrated floor)
                    arm.move_tip_path([(np.array([now[0], now[1], low + run_in]), 90.0, plan.roll)], sp["approach"],
                                      settle=False, allow_contact=True)
                except SafetyError as e:
                    raise StageError("probe", f"no clear contact, and the repeat could not be set up: {e}", dict(chk))
                now = K.fk(arm.read_q(), tool)[:3, 3]
            drop = max(min_drop, float(now[2]) - low + opts.probe_retry_extra + detect)
    finally:
        arm.cfg["keepouts"] = saved
        _emit(sess, "probe_done")
    if chk["slide_mm"] > opts.probe_slide_max * 1000:
        raise StageError("probe", f"the fingertip slid {chk['slide_mm']:.0f} mm sideways while probing: the object "
                                  "may have been pushed; look at the scene again", dict(chk))
    if z is None:  # (check) no contact was reported on the way down
        # no free-air baseline = the jaws were on something before the detector had its run-in
        early = pr.get("contact") is None and (pr["free_air_gap"] is None if "free_air_gap" in pr
                                               else "baseline" in str(pr.get("reason")))
        if early:
            raise StageError("probe", f"the touch check met something right where it started, {1000 * (start[2] - expected):.0f} "
                                      f"mm or less above where the jaws should land on {where}: the object's rim, a "
                                      "closed top or a neighbour is under them", dict(chk))
        chk["note"] = "nothing met above the expected height; the grasp height is the one from the scene"
        return None
    dev = z - expected
    chk.update(contact_tip_z=round(z, 4), deviation_mm=round(dev * 1000, 1))
    above = (min(opts.probe_max_dev, max(0.004, 0.6 * (float(plan.target.z_top) - float(cand.z_surface)))) if measure
             else opts.probe_press + 0.001)
    if dev > above:
        raise StageError("probe", f"touch found a surface {dev * 1000:.0f} mm above the expected height: the jaws landed "
                                  f"on something (the object's rim, a closed top, a neighbour) instead of {where}", dict(chk))
    if not measure:
        chk["note"] = ("touched where the scene puts the surface or below it; the grasp height is the one from the "
                       "scene (the touched height is not used: Arm.probe_down reads deep)")
        return None
    if dev < -opts.probe_max_dev:
        raise StageError("probe", f"touch found a surface {-dev * 1000:.0f} mm below the expected height: the surface is "
                                  "much lower than the scene says (height calibration?)", dict(chk))
    return z


def _alike_neighbour(sc, target) -> bool:
    """Does something of the target's colour family lie right next to it (the camera could take
    the one for the other)?"""
    fam = _color_family(target.color)
    for o in sc.objects:
        if o.id == target.id or o.kind == "container" or o.color not in fam:
            continue
        gap = math.hypot(o.center[0] - target.center[0], o.center[1] - target.center[1]) \
            - 0.5 * (max(o.size[0], o.size[1]) + max(target.size[0], target.size[1]))
        if gap < 0.02:
            return True
    return False


def _presence(S, img, sess, target, q, jaw) -> dict:
    """Is the object still where the scene saw it? {"present": True | False | None, "color_now"}."""
    arm, cam = sess.arm, sess.cam
    if hasattr(S, "object_presence"):
        pr = S.object_presence(img, cam, arm.cfg, target, arm_q=q, jaw_q_deg=jaw)
        return {"present": pr.get("present"), "color_now": pr.get("color_now"),
                "occupied_fraction": pr.get("occupied_fraction"), "visible_fraction": pr.get("visible_fraction")}
    # object_present answers True when it cannot see the place: only its False is information
    return {"present": False if not S.object_present(img, cam, arm.cfg, target, arm_q=q) else None,
            "color_now": None}


def _inside_less(before: list, after: list, color: str, expect_px: float = 0.0, min_area_px: float = 800.0) -> dict:
    """Did something of the object's colour leave the container? Evidence from the colour blobs
    seen inside the rim before and after: one blob fewer, or clearly less area of that colour (at
    least 30 % of the object's own size in the image; white and ivory count as one colour)."""
    fam = _color_family(color)

    def stat(blobs):
        mine = [b for b in blobs if b.get("color") in fam and not b.get("near_arm")
                and float(b.get("area_px", 0.0) or 0.0) >= min_area_px]
        return len(mine), sum(float(b.get("area_px", 0.0) or 0.0) for b in mine)

    n0, a0 = stat(before)
    n1, a1 = stat(after)
    fewer = n1 < n0
    smaller = (a0 - a1) >= max(min_area_px, 0.3 * float(expect_px or 0.0))
    return {"less": bool(fewer or smaller), "blobs_before": n0, "blobs_after": n1, "area_before_px": round(a0),
            "area_after_px": round(a1)}


def _placed(sc_before, sc_after, target, place_xy) -> dict | None:
    """The loose object of the target's colour that stands on the table now and did not before,
    nearest to the place spot: {"id", "color", "kind", "center", "dist_mm"}; None if there is none.
    One of exactly the target's colour comes before one that only looks alike (white / ivory)."""
    fam = _color_family(target.color)
    known = [o for o in sc_before.graspable() if o.id != target.id and o.color in fam]
    best = None
    for o in sc_after.graspable():
        if o.color not in fam and not _color_is(o, target.color):
            continue
        if any(math.hypot(o.center[0] - k.center[0], o.center[1] - k.center[1]) < 0.015 for k in known):
            continue
        d = math.hypot(o.center[0] - place_xy[0], o.center[1] - place_xy[1])
        key = (not _color_is(o, target.color), d)
        if best is None or key < best[0]:
            best = (key, d, o)
    if best is None:
        return None
    _key, d, o = best
    return {"id": o.id, "color": o.color, "kind": o.kind, "center": [round(float(v), 4) for v in o.center],
            "dist_mm": round(d * 1000, 1)}


def _verify_take(sess, S, sc, target, container, place_xy, before_inside, gone_at_lift, opts: TakeOptions,
                 tag: str) -> dict:
    """One look at the result from the stand-by pose: does the object stand on the table (a loose
    object of its colour that was not there before), and has it left the container (its place is
    empty, or the container shows less of its colour, or the look during the lift saw it gone)?

    Two things are NOT taken as "the object left" (2026-11-11 review: an ivory cylinder next to the white
    cap was carried out instead of it and the take-out reported ok): less of the colour family in the
    container while the camera saw the object at its place during the lift and sees that place occupied
    now; and, when something of that family with ANOTHER colour name lay in the container, an object on
    the table that does not carry the target's own colour name. ver["wrong_object"] is set when that
    look-alike stands on the table and the target was seen at its place."""
    arm, cam, cfg = sess.arm, sess.cam, sess.arm.cfg
    try:
        img = sess.frame(fresh=True)
    except Exception as e:  # camera stalled: the motion is finished, only the confirmation is missing
        raise StageError("verify", f"no camera frame for the final check: {e}")
    q = arm.read_q()
    jaw = arm.gripper_q_deg(_gripper_raw(arm))
    fam = _color_family(target.color)
    pres = _presence(S, img, sess, target, q, jaw)
    gone = pres["present"] is False or (pres["present"] is True and pres.get("color_now") is not None
                                        and pres["color_now"] not in fam)
    after = S.objects_in_container(img, cam, cfg, container, arm_q=q, jaw_q_deg=jaw)
    bbox = getattr(target, "pixel_bbox", None) or (0, 0, 0, 0)
    expect_px = 0.6 * float(bbox[2]) * float(bbox[3]) if len(bbox) == 4 else 0.0
    less = _inside_less(before_inside, after, target.color, expect_px)
    ver = {"place_in_container_empty": bool(gone), "gone_at_lift": gone_at_lift, "inside": less, "on_table": None}
    try:
        ver["on_table"] = _placed(sc, S.perceive(img, cam, cfg, arm_q=q, jaw_q_deg=jaw), target, place_xy)
    except Exception as e:  # a perception problem here must not turn into a crash
        ver["scene_error"] = f"{type(e).__name__}: {e}"
    # the object's place still shows its colour (or something the camera cannot name), now
    occupied = pres["present"] is True and not gone
    # seen there twice, with something in the jaws the first time: what the container has less of is
    # SOMETHING of that colour family, not this object
    seen_twice = gone_at_lift is False and occupied
    left = bool(gone or gone_at_lift is True or (less["less"] and not seen_twice))
    ver["left_container"] = left
    # other things of the family that the scene gave another colour name (the ivory cylinder beside the
    # white cap): then the object on the table has to carry the target's own name
    alike = sorted({o.color for o in sc.objects if o.id != target.id and o.kind != "container"
                    and o.in_container == container.id and o.color in fam and o.color != target.color})
    on = ver["on_table"]
    other_name = on is not None and bool(alike) and on["color"] != target.color
    if other_name:
        ver["look_alikes_in_container"] = alike
        if on["color"] in alike and (occupied or (gone_at_lift is False and not gone)):
            ver["wrong_object"] = True  # the look-alike is on the table, the target was seen at its place
    ver["confirmed"] = bool(on is not None and left and not other_name)
    if ver["on_table"] is not None and not left and pres["present"] is None and gone_at_lift is None and not other_name:
        # The camera could not see the object's old place, neither during the lift nor now (the arm is
        # in the way). The jaws held something from the container to the place spot without a break,
        # and an object of that colour stands there now: accepted, and said so.
        ver["confirmed"] = True
        ver["how"] = ("inferred: held without a break from the container to the place spot and seen on the table "
                      "there; its old place in the container is hidden from the camera")
    if ver["on_table"] is not None and ver["on_table"]["dist_mm"] > opts.place_tol * 1000:
        ver["warning"] = (f"the object stands {ver['on_table']['dist_mm']:.0f} mm from the planned spot "
                          "(it slid or rolled when it was put down)")
    if opts.save_images:
        sess.save(tag, img)
    return ver


def take_out(sess: Session, target_query: dict, container_query: dict | None = None,
             opts: TakeOptions | None = None, modules=None) -> dict:
    """Take the object matching target_query OUT of the container it lies in (container_query, default:
    the only container in view) and put it down on a free spot of the table (opts.to, default: the
    nearest one).

    Returns a dict in the style of pick_place: ok, stage (last stage reached: perceive, plan, approach,
    visual, descend, probe, grasp, lift, transfer, release, verify, done), reason, next (what to do
    about it, when the skill knows), arm_state, timings, scene summary, the chosen grasp and place
    spot, the measured values of every check, paths of saved frames.
    modules: (planner, scene) replacements for tests; the planner is the takeout module."""
    if modules is None:
        import scene as S
        import takeout as P
    else:
        P, S = modules
    opts = opts or TakeOptions()
    arm = sess.arm
    res = {"ok": False, "skill": "take_out", "stage": "start", "reason": "", "attempts": 0, "checks": {},
           "images": sess.images}
    tm = Timer(arm_clock(arm))
    # The look at the jaws is what keeps a jaw from coming down on a wall or a neighbour: it is not an
    # option on a real arm. "off" (from cfg["take"]["visual"], or a caller) passes only for the simulated
    # arm, which no camera sees, and for a dry run, which does not move; any other value is a mistake.
    refuse = None
    if opts.visual not in ("on", "off"):
        refuse = (f"take.visual is {opts.visual!r}: it must be \"on\" (\"off\" is for the simulated arm only)",
                  "nothing moved. Set it right: `fd config --set 'take.visual=\"on\"'`, then repeat")
    elif opts.visual == "off" and not opts.dry and not _simulated(arm):
        refuse = ("the look at the jaws is switched off (take.visual = \"off\"), and this is not a simulated arm: a "
                  "take-out never goes down into a container without that look",
                  "nothing moved. Switch it on again: `fd config --set 'take.visual=\"on\"'`, then repeat (to try "
                  "the sequence without a camera use `fd --sim take ...`, to see the plan `--dry`)")
    try:
        if refuse is None:
            arm.speeds(opts.tier)  # refuses an unknown tier or one not yet validated on the hardware
    except SafetyError as e:
        refuse = (str(e), None)
    if refuse is not None:
        res.update(reason=refuse[0], arm_state=_arm_state(arm, arm.read_q()), timings={"total": tm.total()})
        if refuse[1]:
            res["next"] = refuse[1]
        return res
    saved_keepouts = dict(arm.cfg.get("keepouts", {}))
    try:
        for attempt in range(1, max(0, opts.retries) + 2):
            res["attempts"] = attempt
            res.pop("next", None)
            try:
                _take_out_once(sess, target_query, container_query, opts, res, tm, P, S)
                res.update(ok=True, reason="")
                break
            except StageError as e:
                res.update(ok=False, stage=e.stage, reason=e.reason)
                detail = {k: v for k, v in e.detail.items() if k not in ("next", "retry", "candidates")}
                if e.detail.get("next"):
                    res["next"] = e.detail["next"]
                if e.detail.get("candidates"):
                    res["grasp_candidates"] = e.detail["candidates"]  # what was tried and why not
                res["checks"].setdefault("failures", []).append({"attempt": attempt, "stage": e.stage,
                                                                 "reason": e.reason, **detail})
                if e.stage in ("perceive", "plan") and not opts.dry and arm.is_torqued():
                    q = arm.read_q()
                    if _is_parked(q):  # folded back for the second look: nothing to do there, wait at stand-by
                        try:
                            go_ready(arm, opts.tier)
                        except MOTION_ERRORS as e2:  # (the reason for stopping stays the one above)
                            res["checks"]["retreat_error"] = f"{type(e2).__name__}: {e2}"
                        q = arm.read_q()
                    res["arm_state"] = _arm_state(arm, q)
                again = (bool(e.detail.get("retry")) and res.get("arm_state") == "ready" and attempt <= opts.retries
                         and not opts.dry)
                if not again:
                    break
                arm.cfg["keepouts"] = dict(saved_keepouts)
                sess.log(f"attempt {attempt} failed at {e.stage}: {e.reason}; looking again")
    except MOTION_ERRORS as e:
        res.update(ok=False, reason=f"{type(e).__name__}: {e}")
        res.setdefault("arm_state", "stopped")
    except Exception as e:  # a bug in perception / planning / this file: report it, do not move any further
        import traceback
        res.update(ok=False, reason=f"internal error: {type(e).__name__}: {e}")
        res["trace"] = traceback.format_exc().strip().splitlines()[-6:]
        res.setdefault("arm_state", "stopped")
    finally:
        arm.cfg["keepouts"] = saved_keepouts
    res["timings"] = {**tm.marks, "total": tm.total()}
    return res


def _take_out_once(sess, target_query, container_query, opts: TakeOptions, res: dict, tm: Timer, P, S):
    arm, cam = sess.arm, sess.cam
    cfg = arm.cfg
    tool = arm.tip_tool()

    # ---------------------------------------------------------------- 1. perceive
    res["stage"] = "perceive"
    q0 = arm.read_q()
    res["arm_state"] = _arm_state(arm, q0)
    fresh = False
    if res["arm_state"] not in ("ready", "parked", "limp") and not opts.dry:
        # the arm may stand over the container (after a pick it waits there): what lies inside must be
        # seen without it, and the frame is also the reference for the look at the jaws
        try:
            go_ready(arm, opts.tier)
        except MOTION_ERRORS as e:
            raise StageError("perceive", f"moving to the stand-by pose for a clear view of the container failed: {e}")
        q0 = arm.read_q()
        res["arm_state"] = _arm_state(arm, q0)
        fresh = True

    def look(fresh):
        q = arm.read_q()
        jaw = arm.gripper_q_deg(_gripper_raw(arm))
        try:
            img = sess.frame(fresh=fresh)
        except Exception as e:  # camera could not deliver a frame
            raise StageError("perceive", f"no camera frame: {e}")
        sc = S.perceive(img, cam, cfg, arm_q=q, jaw_q_deg=jaw)
        res["scene"] = _summarise(sc)
        res["scene_warnings"] = list(sc.warnings)
        try:
            found = select_inside(sc, target_query, container_query, opts.min_confidence)
            return q, jaw, img, sc, found, None
        except LookupError as e:
            return q, jaw, img, sc, None, str(e)

    q0, jaw0, img0, sc, found, missing = look(fresh)
    hidden = any("robot arm" in str(w) for w in sc.warnings)  # perceive: something is (partly) behind the arm
    unclear = hidden or (missing or "").startswith("container not found")
    if unclear and not opts.dry and arm.is_torqued() and not _is_parked(q0):
        # Even at the stand-by pose the arm can stand between the camera and a container that is close
        # to the robot (2026-11-11: one corner of the box hidden, so no container was reported). Fold
        # the arm back, where it hides nothing, and look once more.
        sess.log(f"{missing or 'part of the scene is hidden by the arm'}; looking again with the arm folded back")
        # The fold goes down onto the rest pose with contact allowed, against the static keep-outs only: what
        # this first look shows standing there decides whether it is made at all.
        blocked = _parking_blocked(arm, sc, opts.fold_clear)
        if blocked and missing:
            tm.mark("perceive")
            raise StageError("perceive", f"{missing}; the arm was NOT folded back for a clear view: {blocked}",
                             {"query": target_query,
                              "next": "nothing was taken. Move that object away from where the arm folds down (or move "
                                      "the container further from the robot, so that the arm at its stand-by pose does "
                                      "not hide it), then repeat"})
        if blocked:
            res["checks"]["second_look"] = f"not made ({blocked}); the first look is used"
        else:
            try:
                _fold_back(arm, opts.tier)
                q0, jaw0, img0, sc, found, missing = look(True)
                res["checks"]["second_look"] = "with the arm folded back (it hid part of the container)"
            except MOTION_ERRORS as e:
                if missing:
                    raise StageError("perceive", f"{missing}; folding the arm back for a clear view failed: {e}")
                res["checks"]["second_look"] = f"not possible ({type(e).__name__}: {e}); the first look is used"
    if missing:
        tm.mark("perceive")
        detail = {"query": target_query}
        if hidden and opts.dry:
            detail["next"] = ("the arm hides part of the scene: run it without --dry (the arm then folds back for a "
                              "second look), or `fd park` first")
        raise StageError("perceive", missing, detail)
    targets, container = found
    tm.mark("perceive")
    if opts.save_images:
        try:
            sess.save("scene", S.annotate(img0, sc, cam))
        except Exception as e:  # annotation is a convenience, never a reason to stop
            res["scene_warnings"].append(f"annotate failed: {e}")
    if len(targets) > 1 and not opts.any_match:
        raise StageError("perceive", f"{len(targets)} objects inside the container match; choose one by id",
                         {"query": target_query, "matches": [o.id for o in targets]})
    container = _settle_container(sess, container)
    res["container"] = {"id": container.id, "color": container.color, "inner": container.inner}
    before_inside = S.objects_in_container(img0, cam, cfg, container, arm_q=q0, jaw_q_deg=jaw0)

    # ---------------------------------------------------------------- 2. plan (all paths, no motion)
    res["stage"] = "plan"
    plan, keepouts, report = plan_take(arm, sess, sc, targets, container, opts, P,
                                       view_for=lambda t: _place_view(sess, S, getattr(img0, "shape", None), t,
                                                                      float(sc.table_z)))
    target, cand = plan.target, plan.cand
    res["target"] = {"id": target.id, "kind": target.kind, "color": target.color,
                     "confidence": round(float(target.confidence), 2)}
    res["grasp_candidates"] = report
    res["grasp"] = cand.to_dict()
    res["place"] = {"xy": [round(float(v), 4) for v in plan.place["xy"]], "clear_mm": plan.place.get("clear_mm")}
    res["plan"] = plan.summary()
    tm.mark("plan")
    if opts.dry:
        res["stage"] = "planned"
        return
    arm.cfg["keepouts"] = keepouts  # everything the camera sees except the target
    box_ko = container_keepouts(sc, container)
    roll = plan.roll
    look_closed = bool(cand.look_closed) and opts.visual != "off"

    # ---------------------------------------------------------------- 3. approach (to above the rim)
    res["stage"] = "approach"
    hot = max(arm.state()["temp_c"].values())
    if hot > opts.max_temp_c:
        raise StageError("approach", f"a servo is at {hot} C; let the arm cool down")
    first_raw = arm.gripper_raw_for(0.0) if look_closed else cand.open_raw
    try:
        sp = arm.speeds(opts.tier)
        q_now = arm.read_q()
        q_from = np.asarray(arm.ready_q(), dtype=float) if _is_parked(q_now) else q_now
        transit = plan_transit(arm, q_from, plan.approach[0], plan.z_clear)
        check_path(arm, [q_from] + transit + plan.approach[1:], sp["transit"], opts, gripper_raw=first_raw,
                   what="approach")
    except SafetyError as e:
        raise StageError("approach", str(e))
    if not arm.is_torqued():
        arm.torque_on()
    if hasattr(arm, "apply_profile"):
        sp = arm.apply_profile(opts.tier)  # servo-side speed profile and time law of the tier
    res["speed_tier"] = sp
    res["arm_state"] = "stopped"
    if _is_parked(q_now):
        # unfold first (the gripper stays closed until the arm is clear of its rest)
        arm.run_path([q_now, arm.ready_q()], sp["transit"], settle=False)
        q_now = arm.read_q()
        transit = plan_transit(arm, q_now, plan.approach[0], plan.z_clear)
    # free-space transit to a point above the hover point, then straight down to it, as one motion;
    # the gripper goes to its pre-grasp opening on the way
    arm.run_path([q_now] + transit + plan.approach[1:], sp["transit"], settle=0.8, gripper_goal_raw=first_raw)
    tm.mark("approach")

    # ---------------------------------------------------------------- 4. look at the jaws (mandatory)
    # detect_jaws reports d = (jaw position seen by the camera) - (jaw position from the joint angles).
    # The object was located through the same camera, so the commanded position is shifted by -d.
    res["stage"] = "visual"
    corr = _align_by_look(sess, S, plan, opts, res, sp, img0)
    if res["checks"]["visual"]["applied"] or res["checks"]["visual"]["raised"]:
        try:  # everything below the hover point again, from where the arm stands and for the shifted position
            plan = build_take_plan(arm, sc, target, container, cand, plan.place, opts, corr=corr, q_hover=arm.read_q())
        except SafetyError as e:
            _retreat(arm, sp, res, via=[(plan.above_hover, 90.0, roll)])
            raise StageError("visual", f"the corrected grasp position is not feasible: {e}",
                             dict(res["checks"]["visual"]))
        roll = plan.roll
    tm.mark("visual")

    # ---------------------------------------------------------------- 5. straight down (optional touch check)
    res["stage"] = "descend"
    tip_grasp = plan.tip_grasp.copy()  # from geometry: table + container floor + the object's own bottom
    if plan.descent:
        arm.run_path([arm.read_q()] + plan.descent, sp["approach"], settle=False if plan.probe else 1.0,
                     ignore_keepouts=box_ko)
    tm.mark("descend")
    if plan.probe:
        res["stage"] = "probe"
        try:
            z_touch = _probe_height(sess, plan, opts, res, box_ko, sp)
        except StageError:
            tm.mark("probe")
            _back_out(arm, sp, res, plan, corr, box_ko)
            raise
        tm.mark("probe")
        if z_touch is not None:  # (probe_may_lower only) the lowest jaw point `margin` above what was touched
            tip_grasp[2] = z_touch + float(cand.margin)
        res["checks"]["probe"]["tip_grasp_z"] = round(float(tip_grasp[2]), 4)
        res["stage"] = "descend"
        try:
            q_now = arm.read_q()
            now = K.fk(q_now, tool)[:3, 3]
            pts = [(tip_grasp, 90.0, roll)]
            if math.hypot(now[0] - tip_grasp[0], now[1] - tip_grasp[1]) > 0.001:  # it slid while pressing: re-centre first
                pts.insert(0, (np.array([tip_grasp[0], tip_grasp[1], max(now[2], tip_grasp[2])]), 90.0, roll))
            # a height that was touched at this very place may pass the floor limit of the calibration by
            # the usual contact allowance (not further); the height from the scene never does
            arm.run_path([q_now] + _tip_path(arm, pts, q_now), sp["precise"], settle=0.6, ignore_keepouts=box_ko,
                         allow_contact=z_touch is not None)
        except SafetyError as e:
            _back_out(arm, sp, res, plan, corr, box_ko)
            raise StageError("descend", f"the grasp height{' found by touch' if z_touch is not None else ''} "
                                        f"cannot be reached: {e}")
        tm.mark("descend")

    # ---------------------------------------------------------------- 6. close
    res["stage"] = "grasp"
    _emit(sess, "grasp", mode=cand.mode, expected_raw=int(cand.expected_raw))
    touch = int(cfg.get("gripper_touch_raw", 2050))
    load_ref = None
    if cand.mode == "pinch":
        g = arm.pinch(wait_s=0.7)  # (the jaws have 12 mm to travel; the default waits 1.2 s for them)
        above = int(g["pos_raw"]) - touch
        expect = max(1, int(cand.expected_raw) - touch)
        empty = above < max(6.0, 0.5 * expect)
        chk = {"mode": "pinch", "pos_raw": g.get("pos_raw"), "hold_goal_raw": g.get("hold_goal_raw"),
               "load": g.get("load"), "above_touch_ticks": above, "expected_raw": int(cand.expected_raw),
               "gap_mm": round(float(GM.width_for_q(arm.gripper_q_deg(int(g["pos_raw"])))), 1), "empty": bool(empty)}
        if above > expect + 30:
            chk["warning"] = ("the jaws stopped further apart than the wall is thick: they may hold the rim "
                              "band or something else")
        load_ref = g.get("load")
        why_empty = "the jaws closed on nothing (no wall between them)"
    else:
        g = arm.grasp_fast(expected_raw=cand.expected_raw)
        stuck = g.get("moved_ticks") is not None and int(g["moved_ticks"]) < 15  # "contact" without having moved
        empty = bool(g.get("empty") or not g.get("contact") or stuck)
        chk = {"mode": "two_sided", **{k: g.get(k) for k in ("contact", "empty", "pos_raw", "width_mm", "duration_s",
                                                             "load")}, "expected_raw": int(cand.expected_raw)}
        if not empty and abs(int(g["pos_raw"]) - int(cand.expected_raw)) > opts.stall_tol_ticks:
            chk["warning"] = "stall position far from the expected width"
        why_empty = "the jaws did not move when closing (were they open?)" if stuck else "the jaws closed on nothing"
    res["checks"]["grasp"] = chk
    tm.mark("grasp")
    if empty:
        # straight up with the jaws as they are, back to the stand-by pose; the caller may look again
        # (not when the touch already showed that heights do not agree here: it would miss again)
        _back_out(arm, sp, res, plan, corr, box_ko)
        off = (res["checks"].get("probe") or {}).get("warning")
        raise StageError("grasp", why_empty + (f"; note: {off}" if off else ""),
                         {"pos_raw": g.get("pos_raw"), "retry": not off})

    # ---------------------------------------------------------------- 7. lift a little, check; lift clear, look
    res["stage"] = "lift"
    res["arm_state"] = "holding_object"
    try:  # the way out again, from where the jaws really closed
        way = _after_grasp(arm, sc, target, container, cand, plan.place, opts, arm.read_q(), roll)
    except SafetyError as e:
        arm.set_gripper(raw=cand.release_raw, wait=True)  # it still stands where it was: let go of it
        res["arm_state"] = "stopped"
        _back_out(arm, sp, res, plan, corr, box_ko)
        raise StageError("lift", f"no way out from the measured grasp pose: {e}")
    arm.run_path([arm.read_q()] + way["lift"], sp["approach"], settle=False, ignore_keepouts=box_ko)
    held, info = _held(g, _jaws(arm), load_ref)
    res["checks"]["lift"] = {"after_mm": round(opts.lift_check * 1000), **info, "held": held}
    if not held:
        res["arm_state"] = "stopped"
        tm.mark("lift")
        _back_out(arm, sp, res, plan, corr, box_ko)
        raise StageError("lift", f"the object slipped out of the jaws while it was lifted (jaws at raw {info['raw']}, "
                                 f"they stopped at {g['pos_raw']} when closing)", {"gripper_raw": info["raw"], "retry": True})
    arm.run_path([arm.read_q()] + way["rise"], sp["transit"], settle=0.3, ignore_keepouts=box_ko)
    held, info = _held(g, _jaws(arm), load_ref)
    res["checks"]["lift"].update(top={**info, "held": held})
    if not held:
        res["arm_state"] = "stopped"
        tm.mark("lift")
        _retreat(arm, sp, res, ignore=box_ko)
        raise StageError("lift", "the object fell out of the jaws while it was lifted clear of the rim "
                                 f"(jaws at raw {info['raw']})", {"gripper_raw": info["raw"], "retry": True})
    gone_at_lift = None
    if opts.lift_look and opts.verify:
        # the lifted object no longer covers its old place in the image: one frame tells whether
        # what the jaws hold is the object (and not, say, a neighbour)
        try:
            img_l = sess.frame(fresh=True)
            pres = _presence(S, img_l, sess, target, arm.read_q(), arm.gripper_q_deg(_gripper_raw(arm)))
        except MOTION_ERRORS:
            raise
        except Exception as e:  # camera or perception trouble: the final check will have to decide
            img_l, pres = None, {"present": None, "color_now": None, "error": f"{type(e).__name__}: {e}"}
        gone_at_lift = None if pres["present"] is None else not pres["present"]
        res["checks"]["lift"]["camera"] = {**pres, "gone": gone_at_lift}
        still = (pres["present"] is True and pres.get("color_now") in _color_family(target.color)
                 and not _alike_neighbour(sc, target))
        if still:
            if opts.save_images and img_l is not None:
                sess.save("lift", img_l)
            tm.mark("lift")
            raise StageError("lift", "the jaws hold something, but the camera still sees the object at its place "
                                     "in the container; stopped above the container, holding")
    tm.mark("lift")

    # ---------------------------------------------------------------- 8. carry out and put down (one motion)
    res["stage"] = "transfer"
    try:
        arm.run_path([arm.read_q()] + way["carry"], sp["transit"], settle=0.6, ignore_keepouts=box_ko)
    except MotionAborted as e:
        # met the table before the planned height (the table is higher than calibrated): on the last
        # stretch above the place spot that is "put down", anywhere else it is a fault
        now = K.fk(arm.read_q(), tool)[:3, 3]
        tp = way["tip_place"]
        if not (math.hypot(now[0] - tp[0], now[1] - tp[1]) < 0.006 and now[2] < tp[2] + opts.place_contact_window):
            raise
        res["checks"]["place_contact"] = {"above_planned_mm": round(float(now[2] - tp[2]) * 1000, 1),
                                          "note": f"the descent stopped on contact: {e}"}
    tm.mark("transfer")
    held, info = _held(g, _jaws(arm), load_ref)
    res["checks"]["held_before_release"] = {**info, "held": held}
    if not held:
        res["arm_state"] = "stopped"
        lost = {"gripper_raw": info["raw"]}
        if _retreat(arm, sp, res, via=[(way["above_place"], 90.0, roll)]) and opts.verify:
            try:  # where did it go?
                ver = _verify_take(sess, S, sc, target, container, plan.place["xy"], before_inside, gone_at_lift,
                                   opts, "lost")
                lost["camera"] = ver
            except StageError:
                pass
        where = (lost.get("camera") or {}).get("on_table")
        raise StageError("transfer", "the object was lost on the way out (the jaws are closed)"
                         + (f"; it lies on the table at ({where['center'][0]:.3f}, {where['center'][1]:.3f})" if where
                            else ""), lost)

    # ---------------------------------------------------------------- 9. open, straight up, stand-by pose
    res["stage"] = "release"
    _emit(sess, "release")
    arm.set_gripper(raw=cand.release_raw, wait=False)
    arm_sleep(arm, opts.release_wait_s)
    res["arm_state"] = "stopped"
    q_now = arm.read_q()
    up = _tip_path(arm, [(way["above_place"], 90.0, roll)], q_now)
    arm.run_path([q_now] + up + plan_transit(arm, up[-1], standby_q(arm, up[-1]), way["z_clear"]), sp["transit"],
                 settle=0.3)
    res["arm_state"] = "ready"
    tm.mark("release")

    # ---------------------------------------------------------------- 10. verify
    res["stage"] = "verify"
    if opts.verify:
        ver = _verify_take(sess, S, sc, target, container, plan.place["xy"], before_inside, gone_at_lift, opts,
                           "final")
        res["checks"]["verify"] = ver
        tm.mark("verify")
        if not ver["confirmed"]:
            detail = dict(ver)
            on = ver["on_table"]
            where = "" if on is None else f"({on['center'][0]:.3f}, {on['center'][1]:.3f})"
            if ver.get("wrong_object"):
                reason = (f"the wrong object was taken out: the {on['color']} {on['kind']} that lay in the container "
                          f"too is on the table at {where}; the {target.color} {target.kind} asked for is still in "
                          "the container")
                detail["next"] = ("tell the user which object came out. The one asked for is still in the container and "
                                  "its look-alike is out of the way now: repeat the command to take it. `fd pick "
                                  f"--nearest {on['center'][0]:.3f},{on['center'][1]:.3f}` puts the other one back "
                                  "into the container, if it should not have left it")
            elif ver.get("look_alikes_in_container"):
                reason = (f"a {on['color']} {on['kind']} stands on the table at {where}, but the object asked for is "
                          f"{target.color} and a {on['color']} object lay in the container too: the camera cannot "
                          "tell which of the two was taken out")
                detail["next"] = ("look at the saved \"final\" image (or `fd scene`): if the object asked for is "
                                  "still in the container, repeat the command")
            elif on is None and ver["left_container"]:
                reason = ("the object left the container but the camera does not see it on the table near "
                          f"({plan.place['xy'][0]:.3f}, {plan.place['xy'][1]:.3f})")
            elif on is None:
                reason = "the object is not seen on the table and still seems to be in the container"
            else:
                reason = (f"an object of that colour stands on the table at {where}, but its place in the container "
                          "still looks occupied (was the right object taken?)")
            raise StageError("verify", reason, detail)

    # ---------------------------------------------------------------- 11. end state
    res["stage"] = "done"
    if opts.end == "park":
        park(arm, sp)
        res["arm_state"] = "parked"
        tm.mark("park")


def sim_take_hooks(sess: Session) -> None:
    """Simulation only (`fd --sim take`, tests on SimBus): give the simulated arm the two things the
    take-out skill feels. When it closes, the jaws stall where the planner expects them to hold the
    object. While it touches (opts.probe, off by default), a surface stops the fingertip at the
    height the skill expects (cfg["take"]["sim_surface_offset"] metres higher or lower, to try a
    scene whose heights are wrong). The camera cannot see a simulated arm, so the caller also sets
    opts.visual = "off" (take_out accepts that for a simulated arm only)."""
    arm = sess.arm
    bus = arm.bus
    tool = arm.tip_tool()
    world = {"floor": None, "stall": None}
    offset = float((arm.cfg.get("take") or {}).get("sim_surface_offset", 0.0))

    def blocker(pos):
        if world["stall"] is not None and pos["gripper"] < world["stall"]:
            return True
        if world["floor"] is not None:
            return float(K.fk(arm.raw_to_q(pos), tool)[2, 3]) < world["floor"]
        return False

    def on_event(name, info):
        if name == "probe":
            world["floor"] = float(info["expected_tip_z"]) + offset
        elif name == "probe_done":
            world["floor"] = None
        elif name == "grasp":
            world["stall"] = int(info["expected_raw"])
        elif name == "release":
            world["stall"] = None

    bus.blocker = blocker
    sess.on_event = on_event
