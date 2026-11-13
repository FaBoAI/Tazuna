# Integration notes: TAZUNA (then "JetHarness") update, stage 1 (night of 2026-11-12)

> **Note (2026-11-13).** This record was written before the repository was reorganised. The file names it uses are
> still right, but the files now live in `tools/<feature>/`, `tests/`, `config/` and `docs/` (the README.md of that time
> is `docs/MANUAL.md`; `testdata/` and `refs/` are under `tests/`; `tools/pen_draw/` is `tools/prototypes/pen_draw/`).
> The closed-form IK of `grasp.py` (`_ik_fast` and its helpers) and the vectorised FK of `tracking.py` now live in
> `tools/ik/kinfast.py`; both modules still offer them under the old names. Where the record says a file lies
> "next to `config.json`" (the tool file `tool_active.json`), it means the project folder itself, NOT `config/`.

> **Addendum, 2026-11-12 09:30.** This file is the record of stage 1 (written about 02:00) and was not rewritten.
> What came after it, in this order: stage 2 (`penskill.py`, `fd tool calibrate`, `fd draw`, `test_penskill.py`,
> the rewritten README chapters 4, 5, 9); of the three independent reviews of stage 2 only "use" ran (the other
> two and the fix pass were stopped by the spend limit); two of its findings were fixed by hand (an interrupted
> `fd draw` / `fd tool calibrate` lifts the pen and still answers; a contact plane or map measured with another
> tip offset or pitch is refused by `fd draw`); the project was renamed TAZUNA; the comments that still stated
> the pull-strokes rule (open problem 1 below) were corrected; the tree was merged into the live harness.
> Open problem 3 (`fd --sim park` touching the real camera) was fixed in stage 2. For what holds now, read
> README chapters 8 (tests) and 9 (known problems), not sections 3 and 8 here.

For stage 2 (DRAW: `penskill.py`, `fd draw`, `fd tool calibrate`). Written from the code in this tree, not from
the package reports. **Nothing in this tree has run on the real arm.** Everything below was checked on the
simulated servo bus, saved photos and the simulated paper only.

## 1. What was merged from where

Base: `jetharness_work/base` (copied with `cp -a`, `.git` removed). Changed files found with `diff -rq` of each
copy against base.

| File | From | How |
|---|---|---|
| `ink.py`, `test_ink.py` | A (INK) | new, copied |
| `camera.py`, `fastcam.py` | A | changed by A only: A's version |
| `tooling.py`, `test_tooling.py` | B (TOOL) | new, copied |
| `skills.py` | B | changed by B only: B's version |
| `strokes.py`, `simpaper.py`, `test_strokes.py`, `drawings/cat.json` | C1 (STROKES) | new, copied |
| `tracking.py`, `test_tracking.py` | D (MOTION) | new, copied |
| `phys_bus.py` | D | changed by D only: D's version |
| `fd.py` | A (+47 -3), B (+428 -3), D (+211 -0) | `git merge-file` against base, A then B then D |
| `robot.py` | B (+56 -6), D (+77 -5) | `git merge-file` against base, B then D |

No package added test data. No `runs/`, `__pycache__`, tool file or scratch file was copied (none existed).

**Conflicts: none.** Both 3-way merges were clean. Checked line by line: every added and every removed line of
every package is in the merged file, and nothing else (fd.py +686 -6, robot.py +133 -11). The two places where
packages touch neighbouring lines (`COMMANDS` in fd.py: `tool` from B, `track-test` from D) merged correctly.

## 2. Changes made by the integrator (after the merge)

1. `fd.py`, `cmd_track_test`: the `--phys` arm is built with `tool=arm.tool`. As D wrote it (`Arm(bus=..., cfg=...)`)
   the second arm read the REAL arm's tool file (B's default `tool="active"`), so a simulated tool was lost.
2. `tracking.py`: new `tool_frame(tool)`; `fk_points`, `Trace.tip / analyse / breakdown / joint_stats / save` take a
   `tooling.Tool` (anything with `.tip_offset`) as well as a 4x4 frame or None. Before, a `Tool` raised TypeError.
3. `test_sim.py`, `test_skills.py`, `test_grasp.py`, `test_motion_phys.py`, `test_bus_fake.py`, `test_tracking.py`:
   one line each, `os.environ["FD_TOOL"] = <testdata/no_tool_attached.json>` (a file that never exists). Reason:
   with a tool file in the tree (the pen really attached) these suites failed (test_sim 54 of 92, test_grasp 28
   of 37, test_motion_phys 13 of 16, test_skills 12 of 117, test_tracking 3 of 13), and the only way to get
   them through was `fd tool detach` with the pen still on the gripper. No assertion was changed
   (`test_motion_phys.py` also got the `import os` the line needs).
4. `test_tooling.py`, last test: "no tool file exists in the tree" became "the tree's tool files are what they
   were before the tests" (same reason; creating, changing or removing one still fails).
5. NEW `test_integration_pen.py` (15 tests, 32 s): the seams, see section 5. Each of fixes 1 to 3 was checked by
   undoing it in a scratch overlay: the tests fail without it.
6. NEW `INTEGRATION_NOTES.md` (this file).

Nothing else differs from the package copies (`ink.py`, `tooling.py`, `strokes.py`, `simpaper.py`, `camera.py`,
`fastcam.py`, `skills.py`, `phys_bus.py`, `robot.py`, `test_ink.py`, `test_strokes.py` are byte-identical to the
copy or the clean merge).

## 3. Test results in this tree

Interpreter `/home/jetson/miniconda/envs/lerobot/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, each suite alone, one
after the other, under an extra audit hook that refuses `/dev/ttyACM*`, `/dev/video*` and `v4l2-ctl`.

| Suite | Run 1: the merge as it came out | Run 2: with the changes of section 2 |
|---|---|---|
| `python nlq.py` | self-test passed | self-test passed |
| `test_skills.py` | 117 / 117 (38 s) | 117 / 117 |
| `test_sim.py` | 92 / 92 (14 s) | 92 / 92 |
| `test_grasp.py` | 37 / 37 (122 s) | 37 / 37 |
| `test_kinematics.py` | 30 / 30 (78 s) | 30 / 30 |
| `test_scene.py` | 50 / 50 (45 s) | 50 / 50 |
| `test_bus_fake.py` | 76 passed, 1 skipped (70 s) | 76 passed, 1 skipped |
| `test_motion_phys.py` | 16 / 16 (218 s) | import error (see below); after the fix, alone: 16 / 16 (217 s) |
| `test_ink.py` | 28 / 28 (16 s) | 28 / 28 |
| `test_tooling.py` | 33 / 33 (27 s) | 33 / 33 |
| `test_strokes.py` | 30 / 30 (49 s) | 30 / 30 |
| `test_tracking.py` | 13 / 13 (108 s) | 13 / 13 |
| `test_integration_pen.py` | (did not exist) | 15 / 15 (32 s) |

One failure, made by the integrator: in run 2 `test_motion_phys.py` stopped at import with a NameError, because
the `FD_TOOL` line added to it uses `os` and that file did not import it. Fixed (`import os`), rerun alone.
No other failure in either run. The only hardware attempt the extra hook saw is the deliberate
`open("/dev/ttyACM0")` with which test_strokes and test_integration_pen check their own guard. After the runs
the tree has no `runs/` entry, no `__pycache__`, no tool file.

With a pen in `$FD_TOOL` set from outside (what a tool file in the tree does): before the pins of section 2
test_sim 38 / 92, test_skills 105 / 117, test_grasp 9 / 37, test_motion_phys 3 / 16, test_tracking 10 / 13 (test_ink,
test_scene, test_kinematics, test_bus_fake passed); with the pins all six read no tool at import, and test_sim
92 / 92 (inside test_integration_pen), test_skills 117 / 117, test_grasp 37 / 37 were run that way.

Default motion is bit for bit what it was: `JH_TRACK_TREE=<base> python test_tracking.py --golden` and the same
without the variable print the same four digests of all goal writes (and they equal `GOLDEN` in the file).

## 4. Old commands, base against int (no tool file, no new option)

45 `fd --sim ...` command lines run in base and in int through `fd.main` with the run directory redirected
(neither tree was written to): `scene` x5, `selftest`, `pick` x5, `do` x4, `take` x3 (all with `--frame` images
of `testdata/`), `ready`, `home`, `ready --tier fast`, `park`, `state`, `torque` x3, `gains`, `joints`, `tip` x2,
`check` x3, `grip`, `grasp` x3, `probe`, `bench` x2, `grasp-plan` x2, `calib-at`, `config`, `report`.
**All 45: same exit code, same JSON (2386 values that are not times), same files written.** Time values differ
by at most 10 ms. `fd snap` cannot be compared (base has no `--frame` for it and would open the camera); in
int `fd --sim snap --frame IMG` keeps the old keys (`tip_xyz`, `q_deg`, `raw`, `view`, `crop`, `scale`).

## 5. How the packages fit

`test_integration_pen.py` runs this chain on the simulated arm and paper, except `tooling.contact_z`,
`tooling.lift` and the two real-camera calls (those are covered by test_tooling and test_ink only).

```python
tool = tooling.active(sim)                                  # = arm.tool when the Arm was built after the attach
lean = functools.partial(tooling.lean_xy, arm, tool)        # the `lean` callable of strokes
polys = strokes.layout(strokes.load("cat"), centre_xy, half_m, rot_deg)
plan = strokes.plan_pieces(polys, lean, tool.corr_map)      # pieces: {"stroke", "ink", "cmd", "parts"}; 1.8 s for the cat
z = tooling.contact_z(tool, xy) - press                     # plan height of the MODELLED tip
wps = arm.plan_tip_path([(np.array([x, y, z]), tool.pitch_deg, 0.0), ...], q_start=arm.read_q(), step_m=0.0015)
speed = tracking.joint_speed_for(wps, length_m, v_m_s)
trace = tracking.record(arm, wps, speed, allow_contact=True, settle=False)   # pen down: contact allowed, NO settle
tooling.lift(arm); tooling.free_move(arm, q_aside, 30.0); tooling.go_ready(arm); tooling.rest(arm)
photo = camera.grab(exposure="paper")        # | FastCamera.paper_frame() | SimPaper.render("paper")
ink.can_show_ink(photo, cam, table_z, region)["ok"]         # False: every "no ink" there is unknown
ink.ladder_readout(photo, cam, table_z, rungs, before=before)
ink.marks_readout(photo, cam, table_z, commanded_xy, before=before)
tool.corr_map = strokes.fit_map(commanded_xy, measured_xy); tooling.save(tool, sim); arm.tool = tooling.active(sim)
ink.stroke_coverage(photo, cam, table_z, [pc["ink"] for pc in plan["pieces"]], before=before)
paper = simpaper.SimPaper(arm, cam, **simpaper.night_truths())   # attaches itself to arm.run_path
```

Measured in simulation with these pieces (the first with a scratch script, the others in the test file):
- Whole cat, the night's hidden truths (shear 10 %, contact tilted 1.5 mm, wander 1.5 mm), tool tip and map
  assumed known, press 2.3 mm, 6 mm/s: 34 pieces, 309 s of robot time (8 s wall), `ink.stroke_coverage` 0.996
  to 1.000, offset mean 0.27 mm, deepest press 3.8 mm. (Calibration itself is stage 2 and was not simulated.)
- Ladder of four dashes with the tool file wrong by 15 mm / 8 mm: readout `[False, False, True, True]`, judgeable,
  offset read (7.3, -13.5) mm against the true (5.9, -14.1) mm. Same ladder with the desk exposure: no rung is False.
- Nine marks -> `fit_map`: read within 0.32 mm of the truth, map rms 0.12 mm, shear coefficient -0.099 (truth
  -0.10); a square drawn with the stored map: coverage 1.0, offset 0.15 mm where the hidden error is 7.2 mm.
- `tooling.lean_xy` is within 3 deg of `strokes.radial_lean()` for the night's pen; for a tool across the
  fingers it points the other way (towards the robot), and pull_pieces turns the pieces accordingly.

Things to know when coding against them:
- `allow_contact=True` lets the tool tip go to table minus `contact_depth_m` (6 mm; `contact_depth=` per call).
  Without it the tool tip stops at the table height (`floor_z + floor_clearance` = `table_z` in this config).
- `Arm` keeps the tool it was built with: after `tooling.save` assign `arm.tool = tooling.active(sim)`.
- `tooling.lean_xy`, `work_q` raise `SafetyError` out of reach; `strokes` raises `ValueError`. Catch both.
- `SimPaper.render(noise=...)`: keep the white noise at 0.8 grey levels or less (or average 4 frames and
  JPEG them, as the real pipeline does). From 1.2 on, `ink` with a `before` photo reads blank paper as "old ink"
  and answers "cannot judge" instead of "no" (safe, but calibration would stop at `visual`). Default is 0.
- `ink.ladder_readout` searches 25 mm around the ladder and takes ANY new ink there for rungs: take the `before`
  photo right before each ladder and keep a ladder 3 cm from other fresh ink (stop rule 2 asks for untouched
  paper anyway).
- SimPaper's default pen axis runs from the gripper-frame origin to the true tip; `tooling.tool_axis` takes a
  tool along the fingers to be parallel to them. The leans differ by 26 deg (night's pen) to 40 deg (night
  truths). Effect on the cat drawn as pull pieces: 0.4 % of the line (7 extra breaks). Pass
  `pen_axis=tooling.tool_axis(cfg, tool)` to SimPaper if the two shall agree.
- In simulation the arm's links are drawn pale (150 against paper 170): `ink` takes them for paper. Only the
  dark pen hides paper. Photos must be taken with the arm aside, as on the real arm.
- `fd --sim tool attach` without `$FD_TOOL` writes `tool_active.sim.json` INTO THE TREE and every later
  `fd --sim` command carries that tool. Tests use `$FD_TOOL`; detach before the tree is delivered.
- `fd --sim park` and quitting `fd --sim repl` run `v4l2-ctl` on the real camera (camera back to auto) also
  with `--sim`: behaviour of base, not changed. With a tool attached `park` is refused before that. Do not run
  them unguarded (ground rule 0.1). Guard used here, as `usercustomize.py` on `PYTHONPATH`: an audit hook that
  raises on `open` of `/dev/ttyACM*`, `/dev/video*` and on `subprocess.Popen` with `v4l2-ctl` (a scratch file of
  that night; it is not part of the repository).
- Run `fd` only as `fd.main(argv)` with `fd.RUNS` pointed at a scratch directory, or `runs/` fills up.

## 6. Actual public signatures

### camera.py / fastcam.py (INK)
```python
EXPOSURES = {"desk": {"auto_exposure": 1, "exposure_dynamic_framerate": 0, "exposure_time_absolute": 333, "gain": 29},
             "paper": {"auto_exposure": 1, "exposure_dynamic_framerate": 0, "exposure_time_absolute": 110, "gain": 0}}
EXPOSURE_WARM_FRAMES = 10
set_exposure(name: str) -> None                    # ValueError for an unknown name
exposure_name(vals: dict | None = None) -> str | None        # "desk" | "paper" | "auto" | None
grab(n_avg=4, warm=5, lock=True, exposure: str | None = None) -> np.ndarray   # exposure given: desk put back in a finally
FastCamera.exposure                                 # "desk"
FastCamera.set_exposure(name: str, settle_s: float = 0.5) -> None
FastCamera.paper_frame(n_avg: int = 4, exposure: str = "paper", settle_s: float = 0.5) -> np.ndarray
```

### ink.py (INK): pure functions; `cam` is a camera.CameraModel; `z` the table height
```python
@dataclass Mark(xy, length_m, width_m, angle_deg, area_px, bbox_px, darkness, confidence, pieces=1, extent_xy=())
    .to_dict() -> dict
can_show_ink(img, cam, z, region=None) -> {"ok", "clipped", "paper_level", "reason", "in_view", "paper_share", "noise"}
ink_map(img, line_px=None) -> np.ndarray (int16)
find_ink(after, cam, z, before=None, region=None, min_area_px=8) -> list[Mark]        # largest first
match_marks(expected_xy, marks, max_dist_m=0.03, merge_m=None) -> list[Mark | None]
marks_readout(after, cam, z, expected_xy, before=None, max_dist_m=0.03, min_len_m=0.0)
    -> {"marks": [{"inked": True|False|None, "mark", "why", "xy", "faint", "old_ink_here"}], "found", "missing", "unknown", "judgeable"}
ladder_readout(after, cam, z, rungs, before=None, max_offset_m=0.025, ink_darkness=None)
    -> {"rungs": [{"i", "inked": True|False|None, "mark", "why", "speck"}], "first_inked", "judgeable", "notes",
        "untouched": bool|None, "offset_mm": [dx, dy]|None, "can"}
stroke_coverage(after, cam, z, polylines_xy, before=None, tol_m=0.004, step_m=0.0005, min_gap_m=0.0015, detail=False)
    -> {"drawn_fraction", "gaps": [{"stroke", "from_m", "to_m"}], "offset_mm": {"mean", "p95"}, "judgeable",
        "unknown_fraction", "ink_m", "missing_m", "can", "per_stroke"} (+ "samples" with detail=True)
merge_marks(marks) -> Mark | None;  clipped_share(img, cam=None, z=0.0, region=TABLE_AREA) -> float
table_xy(cam, uv, z) -> (N, 2);  pixels(cam, xy, z) -> (N, 2);  as_json(obj)        # Marks -> dicts for fd output
constants: CLIP_LEVEL 250, MAX_CLIPPED 0.02, PAPER_LEVEL_MIN 90, PAPER_LEVEL_MAX 242, MIN_PAPER_SHARE 0.90,
           MAX_OFFSET_M 0.025, TABLE_AREA (0.09, -0.24, 0.45, 0.24)
```
`region` = (x0, y0, x1, y1) on the table. `rungs` = [((x, ya), (x, yb)), ...] in drawing order.

### tooling.py (TOOL)
```python
TOOL_FILE = <tree>/tool_active.json;  SIM_TOOL_FILE = <tree>/tool_active.sim.json;  $FD_TOOL overrides both
MOUNTS = ("along_fingers", "across_fingers", "other");  DEFAULT_PITCH = {"along_fingers": 75, "across_fingers": 30, "other": 75}
MIN_FREE_HEIGHT 0.02, PLAN_MARGIN 0.005, LIFT_HEIGHT 0.035, REST_CLEARANCE 0.05, REST_PAN_DEG 20, REST_TOL_DEG 10,
REST_MIN_HEIGHT 0.03, MAX_OFFSET_M 0.30, NEXT_LOCKED (text)
@dataclass Tool(name, tip_offset, mount="other", pitch_deg=75.0, gripper_raw=None, contact=None, corr_map=None,
                region=None, calibrated={}, notes=[], attached_at=None, extra={})
    .to_dict() -> dict;  Tool.from_dict(d) -> Tool            # SafetyError for a bad description
check_offset(v) -> list;  check_pitch(v) -> float
tool_path(sim=False) -> str;  load(path) -> Tool;  active(sim=False) -> Tool | None;  source(sim=False) -> str | None
save(tool, sim=False) -> None;  clear(sim=False) -> None
config_with_tool(cfg, tool) -> dict                           # deep copy, "tip_offset" replaced, "tool_name" added
offset_for(cfg, mount, length_m) -> list;  infer_mount(cfg, tip_offset) -> str
tool_axis(cfg, tool) -> np.ndarray (3,);  length_of(cfg, tool) -> float
table_z(arm) -> float;  tip_height(arm, q, tool) -> float;  body_points(arm, q, tool, n=5) -> (n, 3)
check_pose(arm, tool, q, q_start=None, min_height=None, what="tool pose") -> float      # SafetyError
rest_q(arm, tool, clearance_m=0.05, pan_deg=20.0) -> np.ndarray                         # SafetyError if impossible
ready_pose(arm, tool) -> {"q", "xyz", "pitch", "moved_m"};  ready_q(arm, tool) -> np.ndarray
work_q(arm, tool, xy, height_m=0.0) -> (q, pitch)                                      # kinematics only
lean_xy(arm, tool, xy) -> np.ndarray (2,) unit
contact_z(tool, xy) -> float | None;  calibration_state(tool) -> {"tip", "contact", "map"}: bool
lift_path(arm, tool, q, to_height=0.035) -> list;  lift(arm, to_height=0.035, speed_deg_s=12.0, settle=0.3) -> dict
plan_free_move(arm, tool, q_from, q_to, min_height=0.02) -> {"lift", "path", "via", "min_height_m"}
free_move(arm, q_to, speed_deg_s, settle=0.5, min_height=0.02, lift_speed_deg_s=None, plan=None) -> dict
go_ready(arm, tier=None) -> dict
rest(arm, tier=None, clearance_m=0.05, pan_deg=20.0, torque_off=True, dry=False, watch_s=1.5) -> dict
in_rest_pose(arm, tool, q=None) -> (bool, why);  torque_off_block(arm) -> str | None
lock_message(tool, what) -> str;  refusal(tool, what) -> dict;  sim_start(arm) -> bool
```
`active()` raises SafetyError when the file exists and cannot be used (never "no tool").
`calibrated["tip"]` is what `calibration_state` calls "tip": nothing in stage 1 sets it.

### robot.Arm and skills (TOOL, MOTION)
```python
Arm(bus=None, cfg=None, sleep=None, log=None, tool="active")   # "active" = tooling.active(); None; or a Tool
arm.tool                                                       # Tool | None
Arm.tip_tool() -> 4x4                                          # the TOOL tip when arm.tool is set
Arm.torque_off(force=False)                                    # with a tool: SafetyError outside the tool rest pose
Arm.ready_q() / Arm.rest_q()                                   # the tool's poses with a tool
Arm.set_gripper / grasp / pinch / grasp_fast, _gripper_clamp(raw)   # SafetyError with a tool
Arm.run_path(waypoints, speed_deg_s, allow_contact=False, ignore_keepouts=(), settle=True, hz=50.0, monitor=None,
             tol_deg=0.25, gripper_goal_raw=None, contact_depth=None, time_law=None, fine=False)
    # fine=True: result also has "fine": True, "fine_ff_deg"
Arm.FINE_DEFAULTS, Arm._fine_params(), Arm._fine_corr(fine, integ, qd, lo, hi), cfg["fine_motion"]
skills.go_ready(arm, tier=None)                                # hands over to tooling.go_ready with a tool
skills.park(arm, sp=None)                                      # SafetyError with a tool
PhysBus(..., tau_static=None, stick_speed_deg_s=0.0);  phys_bus.STICK_SLIP_20261111
```

### strokes.py (STROKES): metres on the table, mm inside a drawing, errors are ValueError
```python
load(path_or_name) -> {"name", "path", "box_mm": [w, h], "strokes": [np.ndarray (N, 2) mm]};  names() -> list
layout(drawing, centre_xy, half_m, rot_deg=0.0) -> list[np.ndarray]
box_corners(centre_xy, half_m, rot_deg=0.0, box_mm=None) -> (4, 2);  sheet_rot_deg(sheet) -> float
fit_box(want_half_m, sheet=None, ok_xy=None, visible_xy=None, margin_m=0.01, rot_deg=None, *, box_mm=None,
        region=None, prefer_xy=None, grid_m=0.004, min_half_m=0.015)
    -> {"ok", "centre", "half_m", "rot_deg", "reason", "limited_by", "want_half_m", "corners"}
length(poly) -> float;  resample(poly, step_m, corner_deg=25.0) -> np.ndarray;  split_corners(poly, deg=55.0) -> list
radial_lean(axis_xy=PAN_AXIS_XY, outward=True) -> callable;  constant_lean(vec) -> callable
advance(poly, lean) -> np.ndarray;  push_travel(poly, lean) -> float;  push_excursion(poly, lean) -> float
pull_pieces(poly, lean, hyst_m=0.003, min_push_m=0.006, overlap_m=0.001, merge=False) -> list[np.ndarray]
stats(pieces, lean=None) -> {"count", "points", "ink_m", "penup_m", "push_m", "push_max_m"}
order_pieces(pieces, start_xy=None) -> list[int]
apply_map(points, corr_map) -> np.ndarray;  invert_map(points, corr_map) -> np.ndarray       # None = identity
fit_map(commanded, actual, centre=None, kind=None) -> {"centre", "kind", "coef", "rms_mm", "max_mm", "n", "fits"}
plan_pieces(polys_xy, lean=None, corr_map=None, step_m=0.0012, corner_deg=55.0, order=False, start_xy=None, **pull_kw)
    -> {"pieces": [{"stroke", "ink", "cmd", "parts"}], "stats", "strokes"}
constants: DRAWINGS_DIR, PAN_AXIS_XY = (0.0388, 0.0), DEFAULT_REGION = (0.10, -0.20, 0.40, 0.20)
```
`lean` is a callable `lean(p) -> unit 2-vector`. Maps: `{"centre", "kind": "affine"|"quad", "coef"}` (Tool.corr_map)
or the night's `{"centre", "use", "affine": {...}, "quad": {...}}`.

### simpaper.py (STROKES)
```python
sheet_polygon(centre_xy=(0.225, 0.0), size_m=A4_M, rot_deg=6.0) -> (4, 2)
shear(centre_xy=(0.223, 0.003), kxy=-0.10, kyx=0.0, offset=(0.0, 0.0)) -> dict
night_truths(box_centre=(0.2575, 0.003), box_half_m=0.0525, table_z=-0.0015) -> dict     # kwargs for SimPaper
SimPaper(arm=None, cam=None, *, true_tip_offset=None, tip_error=None, contact=None, pos_error=None, wander_mm=0.0,
         seed=0, sheet=None, pen_width_m=0.001, pen_axis=None, push_skips=True, old_ink=None, noise=0.0,
         blur_px=0.45, shading=0.08, show_arm=True, attach=True)
    .monitor(inner=None) -> callback(t, q_plan, q_meas);  .attach(arm=None);  .detach()
    .feed(q_meas, t=None) -> bool;  .feed_tip(xyz, t=None, lean=None) -> bool;  .pen_up()
    .add_ink(polyline_xy, width_m=None, darkness=1.0, old=True);  .ink(new_only=False) -> list[dict]
    .ink_length(new_only=True) -> float;  .clear(old_too=False)
    .coverage(polylines_xy, tol_m=0.004, step_m=0.001, new_only=True) -> {"drawn_fraction", "offset_mm", "samples"}
    .state() -> dict;  .load_state(state)
    .render(exposure="paper", noise=None, blur_px=None, show_arm=None) -> (1080, 1920, 3) uint8     # "paper" | "desk"
    .snap(path=None, exposure="paper", **kw) -> np.ndarray
    .surface_z(x, y);  .wander(t);  .true_tip(q, t=None);  .lean_xy(q);  .truth()
    attributes: max_depth_m, samples, true_tip_offset, model_tip_offset, pen_axis, sheet
```
Paper reads 170 with "paper" (SPEC says about 175), 255 with "desk". With a tool on the arm the model tip is
`arm.tip_tool()`, i.e. the tool tip; `tip_error` is added to that.

### tracking.py (MOTION)
```python
Trace(t, q_plan=None, q_meas=None, tip_plan=None, tip_meas=None, meta=None)
    attributes: t, q_plan, q_meas, tip_plan, tip_meas, meta, result;  property has_joints
    Trace.from_tip_array(a, meta=None);  Trace.load(path);  .save(path, tool=None) -> str (.npz)
    .tip(tool=None) -> (plan_xyz, meas_xyz)
    .analyse(tool=None) -> {"path_err_mm": {"mean", "p95", "max"}, "z_err_mm": {"min", "max"}, "lag_s",
                            "err_after_lag_mm", "end_err_mm", "duration_s", "n", "rate_hz"}
    .breakdown(tool=None, cfg=None) -> {"tool_frame_mm", "spectrum", "stick_slip", "joints"}
    .joint_stats(cfg=None, tool=None) -> dict
record(arm, waypoints, speed_deg_s, **run_path_kw) -> Trace        # on an exception: exc.trace = the part recorded
tool_frame(tool=None) -> 4x4;  fk_points(q, tool=None) -> (N, 3)
fine_accepted(cfg, min_runs=2) -> bool
shape_points(shape, size_m, centre_xy, z) -> list;  joint_speed_for(wps, length_m, v_m_s, lo=1.5, hi=35.0) -> float
slip_events(t, plan_ticks, meas_ticks, min_stuck_s=0.12, min_slip=3) -> list
joints_from_tips(trace, cfg, tip_offset, pitch_deg, roll_tick=None, first_range=(2, 22, 22, 22, 2)) -> (ticks, n_bad) | None
```
`tool` = None (gripper_frame_link), a 4x4 frame (`arm.tip_tool()`) or a `tooling.Tool`.

### fd commands (new or changed)
```
fd snap [--name N] [--grid G] [--crop x0,y0,x1,y1] [--scale S] [--plain] [--mark ...] [--no-robot]
        [--exposure desk|paper] [--frame IMG]
    adds "exposure", "clipped", "ink_visible", "paper": {"level", "reason"}, "frame"; "next" + "say" when clipped
fd tool attach --name N (--tip x,y,z | --along-fingers LEN | --across-fingers LEN) [--pitch P]
               [--mount along_fingers|across_fingers|other] [--note T] [--replace] [--no-robot] [--dry]
fd tool status
fd tool set [--name N] [--tip x,y,z | --along-fingers LEN | --across-fingers LEN] [--pitch P] [--mount M]
            [--region x0,y0,x1,y1|none] [--note T] [--keep-calibration] [--no-robot] [--dry]
fd tool rest [--tier T] [--clearance 0.05] [--pan 20] [--keep-torque] [--dry]
fd tool detach [--dry]
fd tool calibrate                 placeholder: `_tool_calibrate(a)` in fd.py, `TOOL_COMMANDS["calibrate"]`, a
                                  sub-parser without options (a comment marks where they go). Returns ok False.
fd track-test [--shape circle|square|line] [--size 20] [--v 6] [--height 0.03] [--at 0.22,0.0] [--pitch P]
              [--fine] [--repeat 1] [--accept] [--phys] [--save] [--dry]
    -> "ok", "stage", "reason", plan values ("z", "pitch", "tool", "length_mm", "joint_speed_deg_s", "expected_s", ...),
       "runs": {"default": {...}, "fine": {...}}, "traces", "fine_verdict", "accepted", "next", "say"
fd ready            with a tool adds "tool", "tool_tip_xyz", "standby_xyz", "standby_pitch_deg", "standby_moved_m",
                    "via", "lifted_first", "min_tip_height_m"
```
`COMMANDS["tool"] = (cmd_tool, False)`: the group opens the arm itself (`with make_arm(a.sim) as arm:`).
`COMMANDS["track-test"] = (cmd_track_test, True)`. With a tool attached these return the refusal
`{"ok": False, "error": "SafetyError", "stage": "tool", "reason", "message", "tool", "next", "say"}` before
anything moves: `pick`, `take`, `do` (after the sentence was read), `repl`, `park`, `bench`, `grip`, `grasp`,
`probe`, `calib-at`, `servo-tip`; `tip` / `joints` only with `--contact`; `torque off` outside the tool rest
pose (`--force` still releases). `make_arm(sim=True)` gives the simulated arm the tool of
`tooling.active(sim=True)` and starts it limp in the tool rest pose.

## 7. Deviations from SPEC section 5

Everything listed in section 5 exists with the meaning given there. Differences:

- 5.1: `set_exposure` and `grab` raise `ValueError` (not `SafetyError`) for an unknown exposure name.
  `EXPOSURES["desk"]` holds the four exposure controls of `LOCKED`, not focus and white balance.
  Additions: `exposure_name`, `EXPOSURE_WARM_FRAMES`, `FastCamera.set_exposure / paper_frame`, `fd snap --frame`.
  `fd snap --frame` without `--sim` does not open the servo bus (one changed line in `main()`).
- 5.2: additions only (extra arguments and result keys as listed above, `marks_readout`, `merge_marks`,
  `clipped_share`, `table_xy`, `pixels`, `as_json`). `match_marks` merges pieces closer than 7 mm into one mark
  first. Each rung of `ladder_readout` also has `"i"` and `"speck"`. Acceptance not met as written: mark 0 of
  `grid_pitch75` is 3.0 mm from the manifest value (limit 1.5 mm); the test asserts 3.5 mm. A's finding: the
  manifest value is the centre of a remnant of the mark.
- 5.3: `active`, `save`, `clear` take `sim=False`; `config_with_tool` also adds `"tool_name"`; `Tool` has
  `attached_at` and `extra`; `rest_q` accepts a clearance of 0.03 to 0.15 m only. More commands are refused
  than listed (`repl`, `calib-at`, `servo-tip`, `tip` / `joints --contact`).
- 5.4: `pull_pieces` draws every piece from the end that pushes it less (the prototype kept pushes up to
  `min_push_m`); `overlap_m` is the total shared length at a cut; `resample` also adds points on long segments;
  `fit_box` returns a smaller box as ok True with a reason. Errors are `ValueError`. SimPaper paper level 170.
- 5.5: `Trace.tip(tool)` / `analyse(tool)` default to `tool=None`; the tool is a 4x4 frame or (integration) a
  `tooling.Tool`. A trace loaded from the night's `.npy` files ignores `tool` (it has tips only).
  `fd track-test` has `--pitch`, `--repeat`, `--phys`, `--save`, `--dry` in addition; `--accept` needs `--repeat 3`.

## 8. Open problems, most important first

1. **The photo of the cat contradicts the pull-strokes rule.** SPEC section 2 and the manifest say that lines
   drawn away from the robot (pen pushed, tip first) came out dashed. A measured the opposite on
   `225720_cat1.jpg` and the integrator measured it again (`ink.stroke_coverage`, before `225024`, strokes in
   file order: the night's `draw.py` ran without `--pull`, which was only ever simulated). Missing ink, % of the
   line, by direction against the pen's lean (pushed / pulled: within 60 deg of it) and by half of the drawing box:

   | | near (x < 0.2575) | far | robot's right (y < 0.003) | left | all |
   |---|---|---|---|---|---|
   | pushed (tip first, away from the robot) | 3.2 | 5.5 | 3.2 | 5.9 | 4.4 |
   | across | 4.4 | 11.0 | 8.8 | 6.4 | 7.5 |
   | pulled (tip last, towards the robot) | 5.7 | 18.8 | 17.2 | 5.6 | 11.7 |

   The two legs drawn away from the robot are whole (96 %, 100 %); the leg drawn towards it is 48 % drawn. So in
   the one drawing that exists, the PULLED lines broke, most of all far out on the right. Chatter finer than
   0.5 mm would not show in the photo, and one evening is little data, but nothing in the test data supports
   "pull only". `strokes.pull_pieces` with `tooling.lean_xy`, and SimPaper's skipping (which was built from the
   SPEC sentence, so the simulation always agrees with it), implement the SPEC rule. With the opposite sign
   (`lean = lambda p: -tooling.lean_xy(arm, tool, p)`) the same cutter gives 34 pieces that all travel tip
   first. Stage 2 should keep the rule a parameter of `fd draw`, and the first thing on the real arm should be
   two test lines, one each way.
2. **Not run on the real arm.** Tool poses (`rest_q` gives wrist 43.9 deg / tip 50 mm for the night's pen; 42 deg
   / 56 mm is what held), free moves, the paper exposure through `camera.grab`, `track-test`, the fine mode.
3. **`fd --sim park` and `fd --sim repl` touch the real camera** (`v4l2-ctl`, see section 5). Base behaviour.
4. **`ink` and photo noise**: see section 5 (white noise from 1.2 grey levels: "cannot judge" on blank paper
   with a `before` photo). The real photos of the night have a noise figure of 0.5; a camera setting with more
   noise would stop the calibration at `visual`. Not changed.
5. `ink.ladder_readout` reads any new ink within 25 mm as rungs (section 5).
6. SimPaper's pen axis differs from tooling's tool axis (section 5): 0.4 % on the cat.
7. The fine mode: about 10 % less mean path error on the physics simulation (D's report), not better on the
   ideal simulated servos (`fd --sim track-test --fine` reports "did not pass"). Opt-in; `fine_accepted(cfg)` is
   False until `fd track-test --fine --repeat 3 --accept` passed twice on the real arm.
8. `grid_pitch75` mark 0: 3.0 mm from the manifest (section 7).
9. `sim_fuzz.py` (not a suite) still reads the tool file of the tree. `test_ink.py` does too, and passes with
   a tool attached.
10. The reports of B, C1 and D reached the integrator cut off; "needs from others" and "risks" of those
    packages were not seen. The code was the source for everything here.
11. Disk: 3.1 GB free.
