# TAZUNA Robotics Harness (formerly JetHarness / fable_direct v2) — fast, hardened pick-and-place (design spec)

Status: implementation spec for the night of 2026-11-11. The robot arm is powered OFF / unattended tonight:
**nothing may command motion on the real arm.** Everything is validated in simulation and on saved frames; a
hardware check list is produced for the morning.

Addendum of 2026-11-12 (a tool on the gripper, drawing with a pen): "Tools and drawing (2026-11-12)" at the end of
this file. Sections 1 to 8 are the specification of the night before and are kept as they were written.

## 1. Why (measured baseline)

Two logged pick-and-place runs with v1 (one LLM tool call per primitive, 23-28 calls per object):

| run | wall time | command execution | LLM thinking between calls |
|---|---|---|---|
| white cap | 267 s | about 105 s | about 160 s (60 %) |
| blue block | 424 s | about 115 s | about 310 s (73 %) |

Inside "execution": `grasp-plan` alone took 12-14 s (brute-force roll search), every move 7-8 s (8-25 deg/s
plus up to 2.5 s settle), camera open 1.2 s per frame, `park` 12+ s, box fit 10+ s.

Goal: **one tool call per object** (`fd pick ...`) that perceives, plans, executes with built-in checks and
verifies, in roughly 15 s of robot time at the already-validated speeds, less once faster speed tiers have
been validated on hardware. The LLM is consulted only when a check fails.

## 2. Hard rules for everyone working on this tonight

- NEVER move the real arm. Do not run `fd.py` commands that open the serial bus (`state`, `torque`, `tip`, ...),
  do not instantiate `servo_bus.Bus().open()` against `/dev/ttyACM0`. Use `sim_bus.SimBus`, `phys_bus.PhysBus`
  or the pty fake bus.
- The camera may be used only by the scene work (`camera.grab()` / `fastcam`), and `camera.restore_auto()` must
  be called afterwards. Prefer the saved frames in `testdata/`.
- Do not write `config.json`, `camera_model.json`, `calib_points.json`, `probes.json` (live calibration).
  Tests build their config from `robot.DEFAULT_CONFIG` or `robot.load_config()` copies in memory.
- Do not edit files you do not own (section 4). If you need a change in someone else's file, say so in your
  report; do not make it.
- numpy + OpenCV only (no scipy, no pip installs). Python: `/home/jetson/miniconda/envs/lerobot/bin/python`.
- Never edit anything under `/home/jetson/src`.
- Keep every public name and signature listed in section 5 working (other modules are being written against
  them at the same time).
- Other people are editing other files in this directory right now. An import error from a file you do not own
  can be transient: wait and retry; do not "fix" it.

## 3. Rig facts (measured tonight on the real arm; see also `refs/`)

- SO-101 follower, 6x STS3215 (7.4 V type) on a 5 V supply, bus `/dev/ttyACM0`, 1 Mbps. Loop about 45 Hz.
- Raw-tick mapping: `q = sign*(raw - zero)*2pi/4096`, all signs +1, zero 2047 except wrist_roll 1944.6 (fitted).
  +pan turns toward -y. Servo limits (ticks): pan 864..3418, lift 931..3356, elbow 775..3064,
  wrist_flex 835..3266, roll 0..4095, gripper 1998..3648. Rest (parked) raw: 1999/955/3025/2820/1983/2184.
- Open question (kinematics review, finding 0): camera data prefer a shoulder_lift zero of about 2115
  (-6 deg) but table probes do not. Not resolved tonight; the code must make zero ticks easy to re-fit.
- Table plane z = -0.0235 m (base stands on feet). Probes: (0.170, 0.000) -> -0.0220, (0.140, -0.100) -> -0.0244,
  (0.200, -0.169) -> -0.0241.
- Gripper: `gripper_frame_link` origin is ON the fixed jaw's inner pad, 6.3 mm short of its nose. Tip tool
  offset used by the harness: (-0.001, 0, 0.0068) m in tool axes. Moving jaw opens toward base azimuth
  `psi = roll - pan - 2.789 deg` when the gripper points straight down. Tips touch at raw about 2050;
  `q_jaw_deg = -11.41 + (raw - 2050)*360/4096`. Closed-form geometry: `gripper_model.py` (mm / deg).
  Measured stalls: 20-23 mm wide blocks -> raw 2194; 30 mm cap -> raw 2263.
- Servo loop: P=16 at boot (dead band about 3 deg on shoulder_lift, 5 deg sag at reach). Harness sets P=32 in
  RAM (Lock=1) and adds a software integral term. `Present_Load` is about 8.2 permille per tick of
  (goal - present) at P=32 (saturates near 122 ticks); 4.3 at P=16 (gripper stays at 16).
  Gripper overload protection: Overload_Torque 25 %, Protection_Time 2 s, then torque drops to 20 %.
  A 60-tick squeeze sits on that threshold: use about 40 ticks.
- Validated on hardware so far: joint speeds up to 25 deg/s (transit), 8-12 deg/s (descents), in-servo
  Goal_Velocity cap 1100 steps/s, Acceleration 40. Peak holding load about 32 % on shoulder_lift.
  Community guidance (refs/research/sts3215-behavior.md): transit 30-45 deg/s, near contact 10-15 deg/s.
- Camera: C920 1920x1080, fixed about 0.55 m in front of the base, 0.23 m up, looking back at the robot,
  about 33 deg down. Model `camera_model.json` (pinhole, f pinned 1430, fitted on 15 on-roll-axis fingertip
  points, in-sample rms 13 px = 4.5 mm; leave-one-out 8.7 mm rms). Local scale at the work area about
  3.7 px/mm lateral. Locked settings: focus_absolute 20, exposure 333, gain 29, WB 4100.
- Scene geometry (all through that camera model): see `testdata/frames.json` ("truth").
  The black box is about 100 x 111 mm outer, rim about 30 mm above the table (NOT 50 mm as v1 assumed; that
  bias put the release point 24 mm too close to the camera).
- The glossy table produces a bright low-saturation glare patch (image x 260..620, y 390..860) and mirror
  reflections of objects just below them (towards the camera).

## 4. Files and owners

| file | owner | notes |
|---|---|---|
| `fdtypes.py`, `DESIGN.md`, `fd.py`, `skills.py`, `README.md`, `config.json` | integrator (main session) | do not edit |
| `robot.py`, `sim_bus.py`, `phys_bus.py`, `test_sim.py`, `test_motion_phys.py` | MOTION | motion core, safety, speed tiers |
| `kinematics.py`, `gripper_model.py`, `test_kinematics.py` | KIN | IK fixes, gripper geometry API |
| `scene.py`, `vision.py`, `camera.py`, `fastcam.py`, `blockfit.py`, `capfit.py`, `test_scene.py`, `testdata/` | SCENE | perception |
| `grasp.py`, `test_grasp.py` | GRASP | grasp + release planning |
| `servo_bus.py`, `fakebus.py`, `test_bus_fake.py` | BUS | protocol layer tests + remaining fixes |

`refs/` is read-only reference material: `refs/reviews/*.json` (four adversarial reviews + verifier verdicts,
with code-level fixes), `refs/patches/` (the reviewers' validated patch files and simulators),
`refs/research/*.md` (research reports), `refs/gripper/`, `refs/urdf/`, `refs/fakebus/`.

## 5. Interfaces (must hold when you finish)

### 5.1 `robot.Arm` (MOTION) — existing names keep working

`open/close/__enter__/__exit__`, `raw_to_q`, `q_to_raw`, `q_limits()`, `tip_tool()`, `read_raw()`, `read_q()`,
`state()`, `check_q(q, allow_contact=False, ignore_keepouts=(), what="target", q_start=None)`,
`health_check()`, `torque_on()`, `torque_off()`, `apply_gains()`, `restore_gains()`, `is_torqued()`,
`solve(xyz, pitch_down_deg, roll_deg, q_seed=None, use_tip=True)`,
`move_joints(q_target, speed_deg_s=35.0, **kw)`, `move_tip(xyz, pitch_down_deg, roll_deg=None,
speed_deg_s=30.0, straight=False, step_m=0.005, **kw)`, `run_path(waypoints, speed_deg_s, allow_contact=False,
ignore_keepouts=(), settle=True, hz=50.0, monitor=None)`, `set_gripper(frac, wait_s=1.2)`,
`grasp(min_frac=0.0, ...)`, `probe_down(max_drop_m=0.06, speed_m_s=0.006, ...)`, `gripper_frac(raw)`,
`gripper_raw_for(frac)`. Exceptions `SafetyError`, `MotionAborted`. Module names `ARM`, `DEFAULT_CONFIG`,
`load_config`, `save_config`, `TICKS_PER_RAD`, `min_jerk`.

New (MOTION must add; skills.py is written against these):

```python
Arm.speeds(tier: str | None = None) -> dict
    # {"tier": name, "transit": deg/s, "approach": deg/s, "precise": deg/s, "validated": bool}
    # tier None -> cfg["speed_tier"]. Tiers live in cfg["speed_tiers"]; defaults in DEFAULT_CONFIG:
    #   "slow":   transit 25, approach 12, precise 8   (validated on hardware 2026-11-10)
    #   "normal": transit 45, approach 20, precise 10  (NOT yet validated)
    #   "fast":   transit 70, approach 30, precise 12  (NOT yet validated)
    # cfg["speed_tier"] defaults to "slow"; cfg["speed_tiers_validated"] = ["slow"].
    # Requesting a tier that is not in speed_tiers_validated raises SafetyError unless allow_unvalidated=True
    # is passed (used only by the benchmark).

Arm.run_path(..., settle=True | False | float, tol_deg=0.25, gripper_goal_raw=None, contact_depth=None) -> dict
    # settle: True = up to cfg settle time, False = none, float = max seconds. Early exit when inside tol.
    # gripper_goal_raw: written once when the motion starts (gripper moves concurrently, no waiting).
    # returns additionally: "duration_s", "settle_s", "peak_err_deg", "aborted": False
    # Raises MotionAborted (after releasing the push, see finding MS-0) when the target is not reached.

Arm.plan_tip_path(points, q_start=None) -> list[np.ndarray]
    # points: [(xyz, pitch_down_deg, roll_deg), ...]; straight Cartesian segments between consecutive points
    # starting from q_start (default: measured pose); pitch/roll interpolated; >= 1 waypoint per 5 mm and per
    # 4 deg of orientation change; raises SafetyError on IK failure / branch jump. No motion.

Arm.move_tip_path(points, speed_deg_s, **run_path_kwargs) -> dict
    # plan_tip_path + run_path as ONE continuous motion (no stop at intermediate points).

Arm.set_gripper(frac=None, wait_s=1.2, raw=None, wait=True) -> dict   # raw overrides frac; wait=False returns at once
Arm.grasp_fast(expected_raw=None, squeeze_ticks=40, timeout_s=1.5, empty_margin=45) -> dict
    # Close in ONE command and watch the position at loop rate; stall = object. Returns
    # {"contact": bool, "pos_raw": int, "hold_goal_raw": int, "q_jaw_deg": float, "width_mm": float,
    #  "empty": bool, "duration_s": float, "load": int}. "empty" when the jaws reach touch_raw + empty_margin
    # or less. Must leave a holding goal of pos - squeeze_ticks on contact (never below the closed clamp).
Arm.gripper_q_deg(raw) -> float ; Arm.gripper_raw_for_q(q_deg) -> int      # via cfg["gripper_touch_raw"] (default 2050)
Arm.ready_q() -> np.ndarray       # the stand-by pose (cfg["ready"]: tip xyz (0.20, 0, 0.12), pitch 70, roll 0)
Arm.rest_q() -> np.ndarray        # parked pose clamped into the limits (as cmd_park does today)
Arm.bench_motion(tier, moves=None) -> dict   # free-space benchmark used tomorrow on hardware; sim-tested tonight
```

`sim_bus.SimBus` must mirror the current `servo_bus.Bus` API (`torque_on` idempotent and returning
`{"newly_enabled": [...]}`, `snapshot()` keys incl. `acc`, `vel_cap`, `lock`, `feedback()`, `faults()`,
`status`, `stats`, `set_p_gain`, `read/write` of `Lock`, `P_Coefficient`, `Torque_Limit`, `torque_off`
restoring the profile registers). `phys_bus.PhysBus(SimBus)` = the verifier's torque-level model
(`refs/patches/verify/blocked-windup/phys_bus.py`, calibrated against the real log) made importable from the
project.

### 5.2 `kinematics` / `gripper_model` (KIN)

Existing: `ARM_JOINTS`, `URDF_LIMITS`, `fk(q, tool=None)`, `link_frames(q)`, `gripper_link_pose`,
`tool_offset(dx, dy, dz)`, `pose_summary`, `pitch_down_of`, `approach_axis`, `jaw_open_axis`,
`ik(target_xyz, pitch_down_deg, wrist_roll=0.0, q_init=None, tool=None, limits=None) -> (q, info)`,
`pitch_range(...) -> (min, max) | None`, `_X_PAN`, `_A1`, `_A2`.

New:

```python
kinematics.feasible_pitches(xyz, wrist_roll=0.0, tool=None, limits=None, lo=-20, hi=120, step=2) -> list[tuple[int, int]]
    # contiguous feasible intervals (pitch_range keeps returning the overall (min, max))
kinematics.best_pitch(xyz, want=90.0, wrist_roll=0.0, tool=None, limits=None) -> (pitch | None, degraded: bool)
kinematics.jaw_tilt_deg(q) -> float                    # elevation of the jaw-open axis above horizontal
kinematics.roll_for_azimuth(psi_deg, pan_deg, pitch_down_deg=90.0) -> float   # closed form (vertical gripper), normalised to (-180, 180]
kinematics.jaw_points(q, jaw_q_deg, tip_tool=None) -> dict
    # base-frame 3D points (metres): {"fixed_nose", "moving_nose", "moving_pad", "pivot", "palm"}
gripper_model (mm / deg, tool frame): keep everything in the copied file; add
    nose_points(q_jaw_deg) -> {"fixed": (x, y, z), "moving": (x, y, z)}    # distal ends of both jaws
    finger_back_x(z) (alias of fixed_back_face_x), moving_outer_x(q, z)     # envelope for clearance checks
    q_for_width(w_mm) -> q_contact, raw helpers q_from_raw(raw, touch_raw=2050), raw_from_q(q, touch_raw=2050)
```

### 5.3 `scene` / `fastcam` (SCENE)

```python
scene.perceive(img_bgr, cam, cfg, arm_q=None, jaw_q_deg=None, debug=None) -> fdtypes.Scene
    # cam: camera.CameraModel; cfg: robot config dict (uses cfg["table_z"], optional cfg["scene"] overrides);
    # arm_q: 5 joint angles (rad) so the arm can be masked out of the image (None = arm not in view).
    # Finds every loose object standing on the table (boxes, cylinders) and containers (the black box, with
    # "inner" filled from the measured rim height), marks objects lying inside a container (in_container),
    # never reports the robot, cables or glare as objects. Target <= 0.35 s per frame on this Jetson.
scene.annotate(img_bgr, scene, cam) -> img        # wireframes + ids/labels for a human/LLM glance
scene.find_object(scene, query: dict) -> SceneObject | raises LookupError
    # query keys (all optional): "color", "kind", "id", "nearest_to": (x, y), "largest": bool
scene.object_present(img_bgr, cam, cfg, obj: SceneObject, arm_q=None) -> bool   # is it still where it was?
scene.objects_in_container(img_bgr, cam, cfg, container: SceneObject) -> list[dict]  # colour blobs inside the rim
scene.detect_jaws(img_bgr, ref_bgr, cam, arm_q, jaw_q_deg, cfg) -> dict
    # {"ok": bool, "fixed": {"uv", "pred_uv", "visible"}, "moving": {...}, "dxy_base": (dx, dy) | None,
    #  "confidence": 0..1}: measured vs FK-predicted jaw-nose pixels and the implied base-frame xy offset of the
    #  gripper at nose height (camera truth minus FK). ref_bgr = a frame of the same scene without the arm there.
fastcam.FastCamera(lock=True, restore=True)   # context manager; background capture thread, MJPG 1920x1080
    .latest(n_avg=1, max_age_s=0.2) -> img    # newest frame(s), < 50 ms once warm
    .wait_fresh(after_time) -> img            # first frame captured after a given time.time()
```

### 5.4 `grasp` (GRASP)

```python
grasp.plan_grasp(target: SceneObject, scene: Scene, arm, cfg=None) -> list[GraspCandidate]   # best first; may be all not-ok
grasp.plan_release(target: SceneObject, container: SceneObject, cand: GraspCandidate, arm, cfg=None) -> ReleasePlan
```

`arm` is a `robot.Arm` (tests use `Arm(bus=SimBus(), cfg=...)`.open()); the planner uses `arm.solve`,
`arm.check_q`, `arm.q_limits`, `arm.tip_tool`, `arm.cfg`, `kinematics`, `gripper_model`. Must run in < 50 ms.

### 5.5 `skills` / `fd.py` (integrator)

`skills.pick_place(session, target_query, container_query, opts)` strings the above together:
perceive -> plan -> (torque on) -> open gripper while moving to hover -> optional one-frame visual check ->
descend -> `grasp_fast` -> width check -> lift/transfer/lower as one path -> release -> retreat -> verify ->
ready or park. CLI: `fd pick --color blue --kind box --into black` etc.

## 6. Review findings to fold in (ids: MS = motion-safety, SP = servo-protocol, KI = kinematics, VM = vision-math)

Full text with code-level fixes: `refs/reviews/wfres_5a-740_review_<lens>.json`; verifier verdicts with
corrected fixes: `refs/reviews/wfres_5a-740_verify_*.json`. Where a verifier disagrees with the reviewer, the
verifier's version wins.

- MOTION: MS-0 blocked target (gravity-aware integral bounds, release to `q_meas + integ_start`, raise
  MotionAborted; allowance about 1.5 deg while tracking, 0.5 deg when inheriting an offset), MS-1 inherited
  contact offset, MS-2 probe minimum duration / inconclusive result / per-joint push sign (also KI-8),
  MS-3 (SimBus side) torque_on idempotent, MS-5 check_q relaxations (verifier variant "mine": capped floor
  relaxation, keep-out depth rule, bounded `--contact` depth with probe exemption), MS-6 gain ordering (Lock :=1
  and gains before torque; refuse out-of-range p_gain with nothing changed; ramp or refuse gain steps on a
  loaded arm), MS-7 plan clock + NaN speed, MS-8 `_require_torque` checks all six joints and the profile
  registers, `_hold_here` keeps the compensation, health check per motion, MS-9 gripper (`set_gripper` blocked
  handling, `grasp(min_frac)` guard, closed clamp at touch raw), KI-2 floor check for the open moving jaw when
  pitch < 90, KI-3 wrist_roll limits from zero_ticks (no special case) and no silent clamping in `_write_q`,
  KI-6 straight-line interpolation by orientation change, lower `track_err_abort_deg` to about 10,
  squeeze 40 ticks, servo fault bits (SP-3) abort a motion (overload 32, overheat 4, overcurrent 8).
- KIN: KI-4 pitch intervals, KI-5 angle wrap before the limit test, jaw tilt reporting (KI-2), gripper API.
- SCENE: VM-0 container rim height measured (not assumed), VM-1 reflection margin, VM-2 block dimensions
  normalised (length >= width, yaw = long axis) and not biased, VM-4 `fit_camera` pinned f (no refinement when
  `f_range[0] == f_range[1]`), `f_at_bound`, leave-one-out error helper, VM-5 `find_tip` (jaw bridging, static
  background check), VM-6 k1 acceptance rules, VM-7 ROI clipping in `fit_box`, VM-8 grid labels after crop.
- BUS: SP-2 `set_p_gain` no-op when already equal, SP-5 Operating_Mode check (done in `limits()`), SP-6
  retry counters surfaced, SP-9 fake-bus tests in the project; everything else in SP already landed in the
  23:58 rewrite and needs tests, not new code.
- integrator (fd.py): MS-4 torque-off guard, MS-5 `cmd_joints` single read + table-fit floor, KI-1 directed
  jaw azimuth, KI-7 `check` roll default, VM-3 remove `calib-point`, calib-fit plausibility gate, VM-8 catch-all
  exception logging, SP status in `brief_state`.

## 7. Speed levers (who implements what)

1. One command per object (integrator).
2. Closed-form planning: no brute-force roll search (KIN `roll_for_azimuth`, GRASP).
3. Perception in <= 0.35 s on a cropped / down-scaled image instead of 10+ s full-frame IoU fits (SCENE).
4. Camera thread instead of re-opening the device for every frame (SCENE `fastcam`).
5. Continuous multi-waypoint paths (lift -> transfer -> lower without stopping), gripper moving while the arm
   moves, settle only where precision matters and only as long as needed (MOTION).
6. `grasp_fast` (about 0.5 s instead of 1.8 s) (MOTION).
7. Stand-by at the ready pose between picks instead of park/unpark (integrator; `park` stays explicit).
8. Speed tiers above the validated 25 deg/s, enabled only after `bench_motion` has passed on hardware (MOTION).

## 8. Reporting

Finish with: files changed; public API added/changed; tests added and their results (exact command lines and
pass counts); every review finding you handled (id -> what you did) or deliberately did not (why); numbers
(timings, accuracies, simulated tracking errors); anything you need from another owner; what must be checked
on the real arm tomorrow.

---

# Addendum: Tools and drawing (2026-11-12)

Written at the end of the night of 2026-11-11 / 12 from the code in this tree: `tooling.py`, `ink.py`, `strokes.py`,
`simpaper.py`, `tracking.py`, `penskill.py`, and the `tool`, `snap`, `track-test` and `draw` commands of `fd.py`.

**None of it has run on the real arm.** It was built and tested on the simulated servo bus, the simulated sheet of
paper and the saved photos of 2026-11-11 only (suites and counts: README chapter 8). Every number below that is not
marked "real arm" or "photo" comes from that simulation and says nothing about the hardware. The first run on the
real arm is the check list "朝の実機確認（道具と描画）" in README chapter 5.

## A1. Why

On the evening of 2026-11-11 a pen was taped to the gripper and a cat was drawn. The harness had nothing for that.
It was done with ad-hoc scripts (`tools/pen_draw/`, kept as the prototype) and took 75 minutes. A calibration script
believed a detector that could not see, pressed the pen 3 mm deeper per round and scribbled the sheet at eight
places; a lift that was out of reach left the pen 8 mm above the paper and the script died there. The update turns
what that evening taught into commands of the harness: `fd tool attach / status / set / calibrate / rest / detach`,
`fd snap --exposure`, `fd track-test`, `fd draw`.

## A2. Facts of 2026-11-11 (real arm, one evening, two pens)

Source: `testdata/pen_20261111/manifest.json` with its photos and traces, and the specification of the update.

- **Camera.** The locked exposure (`camera.LOCKED`: exposure_time_absolute 333, gain 29) is for the dark desk. White
  paper clips to 255 with it and a thin ink line disappears. With exposure 110 / gain 0 the paper reads 155 to 175
  and ink shows.
- **The detector lied and the script believed it.** A detector answered "no ink" on clipped photos; the script went
  deeper on that answer.
- **The camera model is only good on the table plane** (it was fitted to fingertip touches on the table). A pen tip
  measured 5 to 7 cm above the table was about 1 cm off along the viewing ray. Ink on the paper is measured well
  (the edges of the A4 sheet come out square to about 3 %).
- **Where the ink lands differs from where the model puts the tool tip, and differently left and right.** Nine marks
  at gripper pitch 75 (`grid_pitch75`), ink minus commanded place: 1.2 to 5.5 mm on the robot's right (y = -0.057),
  7.6 to 11.2 mm in the middle, 12.2 to 16.6 mm on its left (y = +0.063), mostly towards the robot. The cause is not
  known. An affine map commanded -> actual fitted to the nine marks (rms 1.96 mm, max 3.26 mm) was good enough to
  draw with.
- **Control.** `Arm._track` integrates the joint error whenever the planned joint speed is below 5 deg/s
  (`integ_speed_gate_deg_s`); at drawing speed (6 mm/s) that is always. With the integral frozen the path error in
  the air was mean 3.35 / max 7.8 mm; with it on, mean 0.6 to 0.8 / max 2.5 to 3.6 mm, and the height wandered by
  -3.9 to +2.5 mm. The joints stick and slip: a joint stands until its standing command is about 17 ticks (lift),
  16 (elbow) or 7 to 8 (pan, wrist flex) away and then jumps 5 to 11 ticks (`phys_bus.STICK_SLIP_20261111`), which
  makes diagonals stair-stepped. A pen-down path must run with `settle=False`: a settle while pressing ends in the
  "pinned" abort.
- **Contact height.** In plan terms (height of the MODELLED tip) the pen first left ink at about -1.5 mm on the right
  and around -0.7 mm on the left at pitch 75. A drawing height of -3.8 mm gave ink almost everywhere. At pitch 80 two
  of seven marks at -3.6 mm left nothing (both on the left).
- **A tool changes what is safe.** `fd park` opens the gripper to 10 % and folds the arm until the FINGERTIP is 28 mm
  above the table: a 6 cm tool would be driven 18 mm into the table. `fd ready` put the fingertip, not the tool tip,
  at the stand-by point. What worked: fold to (pan 20, lift -100.2, elbow 85.2, wrist_flex 42, roll 0) deg, where the
  tool tip was 56 mm above the table, then torque off; nothing moved in 3 s.
- **The one drawing:** 18 strokes, 862 mm of ink, 6 mm/s, 234.6 s, every stroke completed, a recognisable cat.
  Measured afterwards on the photo: 92 % of the line is on the paper, 0.8 mm beside where it was laid out (mean).

### A2.1 Correction: which direction of travel loses ink

**The first belief was wrong.** The notes of the night, the first version of the specification, the manifest's
"truth" text and `tools/pen_draw/README.md` said: pushing chatters; lines drawn away from the robot (the tip leading)
came out dashed, lines drawn towards the robot or sideways were whole. A "pull strokes only" rule was built on that.
The sentence came from a look at the photo, without checking in which direction each line had been drawn.

What the photo says (`ink.stroke_coverage` on `photos/225720_cat1.jpg` against `225024_grid_after.jpg`; the strokes
laid out and directed as `draw.py` drew them, which ran without `--pull`; the pen leaned 15 deg with its tip pointing
away from the robot; direction = within 60 deg of the line from the pan axis; measured again for this document):

| direction of travel | missing ink, whole drawing | far half (x > 0.2575) | robot's right half | left half |
|---|---|---|---|---|
| away from the robot (the tip leads) | 4.5 % | 5.5 % | 3.2 % | 5.9 % |
| across | 7.5 % | 11.0 % | 8.8 % | 6.4 % |
| towards the robot (the tip trails) | 11.7 % | 18.8 % | 17.2 % | 5.6 % |

The leg drawn towards the robot (stroke 3) is 52 % missing; the two legs drawn away from it (strokes 2 and 4) are
4 % and 0 % missing. (The specification's table sorts by +-x instead of the radial line and reads 4.1 / 7.9 / 11.4 %:
the same picture.) So in the one drawing that exists, the lines that the first belief called safe are the ones that
broke.

What this does not show: it is one drawing with one pen; breaks finer than about 0.5 mm are blurred shut by the
camera; the loss sits far out on the robot's right (17 to 19 %) and is about 6 % elsewhere, so place and direction
cannot be told apart here. It does not prove the opposite rule either.
Consequence: no direction is built in (decision 5 below).

## A3. Design decisions

1. **The tool tip is the tool point** (`tooling.py`, `robot.Arm`). One file says what is on the gripper
   (`tool_active.json` next to `config.json`; no file = no tool = the old behaviour; a file that cannot be read is an
   error, never "no tool"). `Arm` reads it when it is built; `Arm.tip_tool()` is then the tool tip, so planning and
   the floor and keep-out checks work on the tool tip, and `tooling.check_pose` also tests five points along the
   tool's body (the arm itself tests one point). While a tool is attached the gripper never receives a goal (every
   gripper method of `Arm` refuses); `pick`, `take`, `do`, `repl`, `grip`, `grasp`, `probe`, `bench`, `park`,
   `calib-at`, `servo-tip` and `tip` / `joints` with `--contact` refuse before anything moves; `ready` puts the TOOL
   tip at the stand-by point; `fd tool rest` folds the arm into the pose that held that night (generalised: the wrist
   is raised until the tool tip is 5 cm above the table) and torque off is only possible there. A free move with a tool never brings its
   tip lower than 2 cm above the table: a low tip is lifted straight up first, the way is checked every 1.5 deg
   before anything moves, and the measured height is watched while it moves. Why not a second config file as on the
   night (`FD_CONFIG` trick): the safety rules have to follow the tool, not a list of commands "never to use".
2. **Seeing is tri-state, and the exposure follows the scene** (`ink.py`, `camera.py`). Every answer about ink is
   True (seen), False (not there, AND the photo could have shown it) or None (cannot judge, with the reason).
   `ink.can_show_ink` decides whether a "no" can be earned: the place in view, at most 2 % of it clipped (250 or
   above), paper level 90 to 242, at least 90 % of it visible paper, noise low enough for a 0.4 mm line. Ink is
   found with a line filter (black-hat), new ink as the difference to a photo taken just before, thresholds from the
   noise of the photo at hand. `camera.EXPOSURES` has "desk" (the locked values) and "paper" (110 / 0);
   `camera.grab(exposure="paper")` puts the desk values back before it returns; `fd snap --exposure paper` reports
   `clipped` and `ink_visible`.
3. **The stop rule** (specification section 6; `penskill.py`). (1) Before a descent whose result is read from a
   photo, `can_show_ink` at that place must be ok, else stop (`visual`). (2) A search for the contact height goes
   down in ladders of at most 4 dashes and 1.5 mm per dash, then a photo; it goes lower only when the readout is
   judgeable and says "no ink" for every rung, each ladder on untouched paper, never further than 6 mm under the
   height where the model expects the paper. (3) "Cannot judge" on a rung: stop, nothing more is drawn. (4) Once ink
   was seen, calibration presses at most 3 mm below that height, a drawing at most `press` (2.3 mm, at most 4 mm)
   below the contact plane. (5) A command that stops lifts the pen and says where it is. The limits are constants
   (`MAX_RUNGS`, `MAX_RUNG_DZ_M`, `MAX_SEARCH_BELOW_M`, `MAX_AFTER_INK_M`, `MAX_PRESS_M`); options can only be
   stricter (`PenOptions.check`), a second guard refuses any pen-down height below the limit whatever the code above
   it computed (`_Run.guard`), and every pen-down move hands `run_path` a `contact_depth` of its own. When the top
   rung of a ladder already left ink the search looks HIGHER, never deeper. Every exit path lifts the pen
   (`tooling.lift`, which tries other pitches), then goes aside; a lift that fails is reported, not hidden
   (`pen.where` is "stopped" and `lift_error` says why).
4. **Tip, contact height and position are measured on the paper, with ink** (`penskill.calibrate`). Not in the air:
   the camera model is only good on the table plane. `tip`: the first inked rung of one ladder gives all three
   components of the tip offset (its height: along the tool; its place: sideways and radial). `contact`: ladders at
   four places (three or more) give the plane "plan z of first ink"; a plane with a residual above 1.5 mm rms or a
   tilt above 6 mm per 10 cm is refused. `map`: eight 3 mm marks around the drawing box give an affine map
   commanded -> actual; fewer than six found marks or an rms above 3 mm is no map; a mark that left no ink is
   reported, never guessed. The map is affine on purpose (the night's quadratic fit was better on its nine marks,
   0.94 against 1.96 mm rms, but six coefficients per axis follow the noise of a few marks). Test marks go outside
   the box `fd draw` would use, 3 cm from older ink, each ladder after a fresh photo. A drawing is cut in ink
   coordinates, sent through the inverse of the map and drawn `press` below the plane.
5. **The stroke direction is measured, not assumed** (A2.1; `penskill._stage_direction`, `strokes.direction_lean`).
   The direction test draws two 20 mm lines away from the robot, two towards it and two across, and reads them with
   `ink.stroke_coverage`. A direction is preferred only when the other loses clearly more ink (more than 10 points
   and more than twice as much); otherwise the result is "any". It is stored in the tool file
   (`calibrated["direction"]`). `fd draw --strokes auto` follows it; without a measurement, or with "any", the
   strokes are drawn as the file has them and the result says that no direction was chosen from a measurement.
   `simpaper.SimPaper` has no skipping rule built in: which direction skips is a hidden truth that a test sets
   (default none). The cutter keeps its old name (`strokes.pull_pieces`) and some doc comments of stage 1 still
   carry the first belief (`tooling.lean_xy`, the module texts of `strokes.py` and `simpaper.py`): the behaviour
   follows the measurement, the comments do not.
6. **The better slow motion is opt-in** (`run_path(..., fine=True)`, `tracking.py`, `fd track-test`). The fine mode
   adds a friction feed-forward at reversals, inside the same correction bounds as the default mode. With `fine` off
   every goal write is bit for bit what it was (golden digests in `test_tracking.py`). `fd track-test` flies a figure
   in the air (refused below 2 cm) and reports path error, height wander, lag and what the error consists of;
   `--fine --repeat 3 --accept` on the REAL arm records a passed comparison in `config.json`
   (`fine_motion.accepted`); from two recorded passes on `fd draw` and `fd tool calibrate` use the fine mode for
   their strokes (`tracking.fine_accepted`). A simulated pass is never recorded. On the torque-level simulation with
   the stick-slip fit the fine mode is about 10 % better in the mean path error (one run for this document, square
   24 mm, three flights each: mean 0.45 -> 0.40 mm, p95 1.19 -> 0.96 mm, height range unchanged); on the ideal
   simulated servos it is not better. That simulation reproduces the night's path error only within a factor of two
   (`test_simulation_against_the_real_traces`), so this is a reason to try the mode on the arm, not a result.
7. **A simulated world with hidden truths** (`simpaper.SimPaper`, `fd --sim ... --truth FILE`). The sheet, the pen
   and the camera are simulated with truths the harness is not told (the real tool offset, a position error that
   depends on the place, a tilted contact surface, height wander, photo noise, which direction skips, the desk
   exposure), so that the logic above, most of all the stop rule, can be exercised without hardware. It models no
   contact force, no ink flow, no pen sliding in its tape and no paper that moves: it is a test bench for logic, not
   evidence that a drawing will succeed.

In that simulation, with truths like the night's (tool offset wrong by 15 mm sideways and 8 mm in length, position
sheared by 10 %, contact tilted by 1.5 mm across the box, height wander 1.5 mm, photo noise 0.6 grey levels):
calibration takes 315 s of robot time and 19 photos (tip corrected by 15.8 mm, plane rms 0.38 mm at 4 places, map
rms 1.22 mm from 8 marks); the cat is then drawn in 225 s with 99.9 % of the line on the paper and a mean offset of
0.32 mm. With the desk exposure both commands stop at `visual` before the pen touches anything; with a detector
forced to say "no ink" the search ends 6 mm under the expected paper.

## A4. What is not solved

- **The cause of the position error that differs left and right** (A2). The map absorbs it where it was measured;
  nothing explains it. Not separated: a sideways error of the tool tip (it turns with the pan angle), the zeros of
  elbow, wrist and wrist roll (README chapter 9: only the shoulder_lift zero was refitted, and camera and arm
  disagree by a few mm on the robot's left side also without a tool), flex of pen and tape under load. Until it is
  understood, every new pen position needs its own map and the map is only trusted near its marks.
- **Stick-slip.** The size of a slip is set inside the servo (5 to 11 ticks, 0.25 to 0.4 mm of tip travel per
  tick). The fine mode can only help a joint that reverses; a steadily creeping joint still stands and jumps, so
  slow diagonals stay stair-stepped. Not tried: drawing faster where the joints run in long moves (5 to 6 deg/s).
- **Height wander.** -3.9 to +2.5 mm in the air at 6 mm/s on the real arm. The drawing presses 2.3 mm below a
  measured plane and hopes the pen and its tape take up the rest; there is no feedback of the contact while a
  stroke is drawn. The contact heights themselves are only known to one rung (1.5 mm).
- **Which direction of travel keeps its ink** on the real pen (A2.1): measured per tool, never yet on hardware.
- **The stop rule counts in plan terms.** Its reference is the height where the MODEL expects the paper, so it is
  only as good as the recorded tool length. A photo is read after a ladder, not after a dash: within one ladder the
  pen is sent up to 4.5 mm deeper after its first inked dash. Found in simulation while writing this: a first search
  starts 12 mm above the expected paper, so a pen recorded 15 mm too short was already down on its first dash and
  the ladder sent its real tip 7 mm below the paper surface before the first photo (14 mm for a pen recorded 22 mm
  too short; the simulated pen sinks in, a real one would be pressed against the table); a pen recorded 10 mm too
  long never reached the paper and the stage stopped cleanly without a mark. The length has to be measured to about
  5 mm, and the first dash of the first ladder has to be watched: it must be in the air.
- **Gaps of this implementation** (state of 2026-11-12, 04:30): `tool.region` is recorded but `fd draw` does not
  keep the drawing inside it (on a used sheet the box moved beside the calibrated area and plane and map were
  extrapolated; simulation: mean offset 1.0 instead of 0.3 mm); there is no rough tip measurement from photos in the
  air (the specification allowed it as optional); `fd tool calibrate --start-above` accepts up to 30 mm, but a
  search has five ladders, which from about 25 mm on may end above the paper (at 30 mm they end 1.5 mm above the
  height where the model expects it), and the stage then still reports "no ink down to" the depth limit; Ctrl-C is
  not caught (the servos hold the last goal, possibly with the pen on the paper; `fd tool rest` lifts first).
- **Everything that needs the hardware:** the tool poses and free moves, the paper exposure through `camera.grab`,
  the readouts on live photos, pressing depths, the fine mode, `fd draw` as a whole.

## A5. Where it lives

| file | what |
|---|---|
| `tooling.py` | the tool file, tool poses (stand-by, rest), checks with a tool on, free moves, lift, refusals |
| `ink.py` | ink on paper from photos: `can_show_ink`, `find_ink`, `ladder_readout`, `marks_readout`, `stroke_coverage` |
| `camera.py`, `fastcam.py` | exposure presets "desk" / "paper" |
| `strokes.py` | drawing files, layout on the sheet, cutting by direction, corner splitting, the map and its inverse |
| `simpaper.py` | the simulated sheet, pen and camera with hidden truths |
| `tracking.py` | recording and analysing how the tip follows a path; `fine_accepted` |
| `penskill.py` | `calibrate` (tip, contact, map, direction) and `draw`, with the stop rule |
| `fd.py` | `tool ...`, `snap --exposure`, `track-test`, `draw`; guards for commands that must not run with a tool |
| `drawings/cat.json` | the drawing of 2026-11-11 |
| `tools/pen_draw/` | the prototype scripts of 2026-11-11 (superseded; kept for reference) |
| `test_ink.py`, `test_tooling.py`, `test_strokes.py`, `test_tracking.py`, `test_integration_pen.py`, `test_penskill.py` | tests, all without hardware |
