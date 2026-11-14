"""`fd calib-air`: where is the tip of the tool on the gripper (and, when the camera was moved, where is the
camera)? Measured from photos of the tool tip held in the air (aircal.py).

    fd [--sim] calib-air [--tip-only | --sets air,low] [--dry] [--save [--force]] [--from DIR] [--end rest|stay] [--no-look]

Why a command (2026-11-13): that day the air calibration ran five times on the real arm, every time by
throw-away scripts and a tool file edited by hand, and one of those times twenty poses were flown with a
registered tip nobody had looked at (17 cm of brush registered as 8). What the scripts did by hand is here in
this order, with the checks that were missing:

1. LOOK FIRST. The arm goes to a pose in which the camera sees the gripper and the tool and which keeps a
   tool of unknown length off the table (aircal.look_pose: fingertip >= 12 cm up, fingers <= 30 deg down; the
   module text of aircal says which tools that covers). Wherever the way there turns the base (for the first
   photo it always does), the tool is first lifted 10 cm straight up, and the arm comes straight down afterwards
   (aircal.look_way, 2026-11-14 evening: until then the look pose was turned at its own height, and a tool longer
   than registered passed 12 .. 47 mm above the table over the rig's water dish). What has to be clear is the
   place where the arm comes down for the first photo: the plan, the look's record and a refused look name it
   (_side_note). One photo of the scene without the tool there, one in the
   pose; when no tip candidate lies within 60 px of where the REGISTERED tip would be seen, the command stops
   there: the arm goes nowhere else. The first photo is taken with the look pose turned aside about the base
   (aircal.look_aside), wherever the arm stood, and the arm turns back for the second; it is taken where the
   arm stands only when the camera model sees neither the arm nor anything a tool of up to 30 cm could reach
   from the gripper at the place of the look (aircal.look_hidden, silhouette): what arm or tool hide in the
   first photo would be "new" in the second, and paint among it passed for the tip (reviews of 2026-11-14: first
   behind the arm, then behind a handle that lay beside the finger line, outside the two lines the tool was
   drawn as until round 3). For the same reason nothing that lies where the arm stood for the first photo can
   make the look pass, no look is taken over a background that shows bristle-coloured paint near the place of
   the look (aircal.look_background: the arm holds where the first photo was taken, stage "look", and the sheet
   is covered with a blank one first), and where no turn clears that place the look is refused before anything
   moves. A look never passes by colour alone, and only --no-look skips it: that word is the
   operator's statement that the registered tip was checked against a photo, and the first set is then flown on
   it (a tool 9 cm longer than registered reaches the table there: nothing but the look can know). One photo
   sees two of the three numbers: a registration ON the tip's line of sight passes, however far along it.
2. The first set of poses is flown high (tip-only: 8 and 13 cm, full: 7 .. 17 cm, heights of the MODELLED tip).
   Its photos are solved at once, and only when that fit passes the gates below, and its tip lies within 3 cm of
   the registered one, is the low set (3 .. 3.5 cm over the sheet) planned, with the tip just solved, not the
   registered one. That is what was done by hand in the morning and in the evening; with the second mount of
   the evening the first set gave 8 of 20 photos and no low set was flown. No option flies a low set with a tip
   that was neither measured before (the tool file's tip_measured) nor solved in this run. Once a fit of this
   run has passed the gates, everything that still moves is planned with ITS tip, also when the jump rule kept
   the low set on the ground: the fold at the end above all (5.).
3. Two solves. --tip-only (the default when a camera model exists): the model is kept, the three numbers of the
   tip are solved (aircal.solve_tip: vote, then least squares). The full solve (--sets air,low; the default
   without a model): camera pose and tip together (aircal.solve), for a camera that has been moved.
4. Nothing of the rig is written unless --save, and --save only stores a fit that passes the gates: the tip
   goes into the tool file (the tool keeps its name, pitch and joint limits; a changed tip drops the pen
   calibration, as `fd tool set --tip` does) with a record tip_measured (when the photos were taken, when it
   was stored, where the jaw stood; a tool file that had no jaw position gets the one the servo reported
   during the photos, so that its jaw is compared before a move from then on); the full solve also replaces
   the camera model, the old one copied to backups/ first. A stored tip leads to `fd paint status` when a
   painting set-up is on record (the contact plane holds for one brush state), never to the pen calibration.
   --force (with --save) stores a fit that fails the gates, and photos whose records name no tool: nothing
   else. It moves nothing and skips no check of the arm (until the review of 2026-11-13 the same
   word also went on after a refused look and flew the low set with an unmeasured tip: a tool 9 cm longer than
   registered was at or under the table at 3 of 27 poses in simulation). Never stored, with or without
   --force: photos of another tool or another taping of it, and photos taken with the jaw elsewhere than the
   tool file has it now (the same brush had its tip 2.2 cm elsewhere after the jaw was closed). The capture
   writes where the servo said the jaw stood into every record, and a capture that is asked to --save is
   refused before it moves when the jaw does not stand where the tool file has it.
5. The arm ends folded with the torque ON (--end rest; never off here: a wet brush may hang over a sheet) or
   stays where the last photo was taken (--end stay). With a painting set-up on record the fold is the one
   `fd paint rest` chooses (paintskill.rest_choice: the brush, and the place it would sink to, off the canvas),
   not the tool's default rest pose, which had the brush 4 cm INSIDE the sheet; and every hint after a stop
   names `fd paint rest`. Without a set-up: the tool's rest pose, and `fd tool rest --pan=A --keep-torque`
   with A the base angle the arm stands at (a hint is a command that works as typed).
   The fold is planned with the tip the photos gave whenever a fit of them passed the gates (a single set is
   solved before the fold for that), not with the registered one: a tool taped across the fingers, registered
   on its line of sight 93 mm from its tip, was solved to 0.2 mm and then folded 17 mm INTO the table with the
   registered tip, result ok (simulated, 2026-11-14). Where no fold is safe with the solved tip the arm is not
   folded: it holds where the last photo was taken, torque on, and the result (stage "end", not ok) says why.
   While the solved tip is further than LOW_MAX_JUMP_M from the tool file's, every hint that folds the arm
   starts with storing it: `fd paint rest` and `fd tool rest` plan with the tip the tool file has.
   The arm is folded only with a CONFIRMED tip (round 3, 2026-11-14): the tip of a fit of this run that passed
   the gates, or the registered tip when the tool file has it on record as measured and the photos of this run
   do not speak against it (no photo, or a fit within LOW_MAX_JUMP_M of it). In every other case, also after a
   look that passed and with --no-look, the arm is not folded: it holds where the last photo was taken, torque
   on, the result is not ok (stage "end"), the measurement (if any) is reported and nothing is stored; `next`
   says to LOOK at arm and tool before anything moves, and what works then. (Until then such an arm was folded
   with the registered tip: registered on the line of sight 93 mm from its tip, 9 of 21 photos without the
   brush, the real tip ended 17 mm under the table, result ok.) While the tool file has no tip on record as
   measured and the run stored none, the hint of a folded arm names the storing first as well: `fd paint rest`
   folds such a tool again, as high as the arm folds, before it switches the torque off.

--dry plans only (how many poses can be flown, the lowest modelled tip height). --from DIR solves the photos of
an earlier capture again, without arm or camera (and stores the result with --save). --sim stands before the
command (`fd --sim calib-air ...`; among the words after it, it is refused); it has no camera: it is for
--dry and --from, and it never writes a tool file inside the project folder.

A tip-only solve is only as good as the camera model: a camera that has moved is absorbed by the tip (2 cm of
camera = 1 .. 2 cm of tip, simulated with the evening's poses) while the residuals hardly grow. Two warnings
watch that (the result stays ok, nothing is stored without --force): the residuals against the model's own
(RMS_OVER_MODEL) and the camera pose the same photos give when it is solved too (aircal.camera_check). A
camera model file that cannot be read is a refusal (stage "camera") for the tip-only solve; the full solve
goes on without a prior model.

Results are plain dicts: ok, stage, reason, next, say. Every `fd ...` between backticks that this module
writes into one is a complete command that works as it is typed (tests/test_aircal.py parses them all).
Verified on the real arm on 2026-11-13: capture and both solves, through scripts. This command itself has only
run on the simulated arm and on those photos.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import signal
import sys
import threading
import time

import numpy as np

MODE_NAME = {"tip": "tip-only", "full": "full"}
DEFAULT_SETS = {"tip": ("high", "low"), "full": ("air", "low")}

# ----------------------------------------------------------------------------- gates
# What a fit must show before it is stored (--save) or trusted to plan the low set. All numbers are from the
# real arm on 2026-11-13.
#
# Tip only (camera not moved; four runs): 20 .. 34 photos used = 91 .. 100 % of the photos, rms 6.6 .. 9.3 px
# (the camera model's own fit: 8.9 px), formal error 0.2 .. 0.5 mm. The run that must NOT pass: the second mount,
# 8 of 20 photos (in 11 of the others the tip was outside the picture: the registered tip was 9 cm short).
TIP_MIN_USED = 12            # "about a dozen poses" is the least the quick mode was meant for; 3 unknowns
TIP_MIN_SHARE = 0.6          # of the photos: between the 40 % that failed and the 91 % of the worst good run,
#                              with room for a hand in the picture or a few tips at its edge
TIP_MAX_SIGMA_M = 0.001      # twice the largest formal error of the good runs
# The camera has moved when the tip-only rms is above 1.3 x the rms the model file carries: the four runs with
# the camera unmoved gave 0.74 .. 1.05 x; the rms of 30 photos scatters by 9 % (1 / sqrt(2 * 57)), so 1.3 is
# three standard errors above the model's own figure and 2.3 px above the worst unmoved run (9.3 px). A model
# with a small rms of its own must not make every run a warning: not below 10 px (what the detector and the
# arm leave on this rig whatever the model says).
RMS_OVER_MODEL = 1.3
RMS_FLOOR_PX = 10.0
# ... or when the same photos, solved for the camera too, put it elsewhere (aircal.camera_check). On the four
# real runs with the camera unmoved: 0.6 .. 14.7 mm, 0.4 .. 1.2 deg, d = 1.8 .. 3.5 formal standard errors;
# simulated unmoved: d <= 3.9; simulated 2 cm in any direction: d >= 8; the morning's photos against the model
# of 2026-11-11 (the camera really had been moved, by 52 mm): 52 mm, 3.5 deg, d = 122.
CHECK_D = 6.0
CHECK_MM = 5.0
CHECK_DEG = 0.5
CHECK_MIN_USED = 15          # nine numbers from fewer photos accuse nobody (8 photos: +- 7 .. 9 mm, formally)
#
# Full (camera pose + tip). The morning: 55 of 61 photos, rms 8.9 px, tip +- 1.0 .. 1.6 mm, camera centre
# +- 2 .. 3 mm; from its first 31 photos alone: 29 used, 8.5 px, tip +- 1.6 .. 2.7 mm, centre +- 3.4 .. 5.4 mm
# (the low set was planned with that). The same solve on the evening's photos (camera not moved): 33 / 34
# photos put the camera 2.0 / 0.6 mm from the model, 22 photos 5.9 mm (centre +- 4 .. 7 mm), 9 photos 17 mm.
FULL_MIN_USED = 25
FULL_MIN_SHARE = 0.6
FULL_MAX_TIP_SIGMA_M = 0.003
FULL_MAX_CENTER_SIGMA_M = 0.006
RMS_ABS_MAX_PX = 13.0        # 1.5 x the 8.9 px of the model the pictures of that day were painted with; every fit
#                              of that day that was right had 6.0 .. 9.3 px (`fd calib-fit` lets 20 px pass)
# The low set is planned with the tip the high set gave. A tip that came out further than this from the
# registered one is reported, not flown: after a look that passed (60 px, about 14 mm in the picture) the two
# can only be far apart along the look's line of sight, or one of them is wrong, and 3.5 cm over the sheet is no
# place to find out which. The day's registrations were 8 .. 27 mm from what was then measured (27: the
# morning's "deliberately long" guess, which today's look would have refused).
LOW_MAX_JUMP_M = 0.030
FEW_SHARE = 0.5              # "most photos have no candidate near the solution": the warning of the second mount
LOOK_SAME_DEG = 12.0         # an arm this near the look pose has the tool where the look photo will show it: a
#                              photo from there is no photo "before", so the arm is first turned aside
PHOTO_JAW_TOL_TICKS = 8      # the jaw as READ during the photos against the tool file's gripper_raw: what
#                              tooling.hold_jaw asks of a held jaw (JAW_END_TOL_TICKS, 0.7 deg; pressed on the
#                              fixed jaw it creeps 2 .. 3 ticks). Two RECORDS are compared with JAW_SAME_TICKS (3)
SQUEEZE_ON_TICKS = 5         # a goal this far beyond the position is a squeeze (tooling.jaw_state's rule)
SECONDS_PER_POSE = 4.2       # 24 poses in 90 s, 12 in 55 s (2026-11-13 evening, 25 deg/s)


class _BadOptions(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):           # one JSON result instead of a usage text on stderr and exit code 2
        raise _BadOptions(message)


def build_parser():
    """The words after `fd calib-air`. --sim is not one of them: it stands before the command, where fd reads it
    (and keeps the simulated tool file out of the project folder); main() takes it from the front of what fd
    hands over and refuses it anywhere else."""
    # allow_abbrev=False (round 3, T17): argparse would take `--no` for --no-look and `--sa` for --save. --no-look is
    # the operator's statement that the registered tip was checked against a photo, and --save writes the tool file:
    # each of them counts only as the whole word.
    p = _Parser(prog="fd [--sim] calib-air", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                allow_abbrev=False)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tip-only", dest="tip_only", action="store_true",
                   help="keep the camera model, solve the tool tip (the default when a camera model exists)")
    g.add_argument("--sets", default=None, metavar="air,low",
                   help="the full solve (camera pose + tip) with these pose sets, flown in this order: air (7 .. 17 cm "
                        "up), low (3 .. 6 cm; planned with the tip the air set gave)")
    p.add_argument("--dry", action="store_true", help="plan only: poses per set, how many can be flown, lowest height")
    p.add_argument("--save", action="store_true", help="store the result (tool file; full solve: camera model too) "
                                                      "when the fit passes the gates")
    p.add_argument("--force", action="store_true", help="with --save: store a fit that fails the gates (and photos "
                                                       "whose records name no tool). Nothing else: it moves nothing "
                                                       "and does not skip the first look")
    p.add_argument("--from", dest="from_dir", default=None, metavar="DIR",
                   help="no arm, no camera: solve the photos of an earlier capture (every *records*.json in DIR)")
    p.add_argument("--end", choices=("rest", "stay"), default="rest",
                   help="afterwards: folded with the torque ON (default; with a painting set-up: where `fd paint rest` "
                        "folds), or stay at the last pose")
    p.add_argument("--no-look", dest="no_look", action="store_true",
                   help="skip the first look: the operator has checked the registered tip against a photo of gripper and "
                        "tool. The first set is then flown on that word alone")
    return p


# ----------------------------------------------------------------------------- seams (the tests replace these)
def _make_arm(sim: bool):
    from robot import Arm, load_config
    if sim:
        import tooling
        from sim_bus import SimBus
        arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=tooling.active(sim=True))
        arm.open()
        tooling.sim_start(arm)
        return arm
    return Arm().open()


def _photo_fn():
    import camera as C
    return lambda: C.grab(exposure="paper")


def _run_dir() -> str:
    """Today's run directory, named as fd.py names it (runs/<date_time>/, runs/current points at it)."""
    fn = getattr(sys.modules.get("__main__"), "run_dir", None)
    if not callable(fn):
        import fd
        fn = fd.run_dir
    return fn()


def _backup_dir(model_path: str) -> str:
    """Where the camera model that is being replaced is kept: backups/ of the project for the rig's own model,
    else (a model named by $FD_CAMERA_MODEL: tests, experiments) a backups/ folder next to that file."""
    import camera as C
    import tazuna_paths
    if os.path.realpath(model_path) == os.path.realpath(C.CAM_MODEL_PATH):
        return os.path.join(tazuna_paths.ROOT, "backups")
    return os.path.join(os.path.dirname(os.path.abspath(model_path)), "backups")


# ----------------------------------------------------------------------------- small helpers
def _mm(v, digits: int = 1) -> list:
    """Metres as millimetres for a result (None for a number that is not finite: the result must stay JSON)."""
    return [round(float(x) * 1000.0, digits) if math.isfinite(float(x)) else None for x in v]


def _refusal(stage: str, reason: str, next_: str, say: str, **more) -> dict:
    return {"ok": False, "stage": stage, "reason": reason, "next": next_, "say": say, **more}


def _tip_record(tool) -> dict | None:
    """The record of how the tool's tip was measured (Tool.tip_measured; a tooling without that field keeps it
    among the file's unknown keys), or None: the tip is a guess."""
    rec = getattr(tool, "tip_measured", None)
    if rec is None:
        rec = (getattr(tool, "extra", None) or {}).get("tip_measured")
    return rec if isinstance(rec, dict) else None


def _with_tip(tool, tip):
    """A copy of the tool with another tip (in memory: for planning the next set)."""
    import tooling as TL
    other = TL.Tool.from_dict(tool.to_dict())
    other.tip_offset = [round(float(v), 5) for v in TL.check_offset(tip)]
    return other


def _where(arm, tool, at: str) -> dict:
    """Where the arm is, for the operator."""
    import tooling as TL
    q = arm.read_q()
    return {"at": at, "q_deg": [round(math.degrees(float(v)), 1) for v in q],
            "tip_height_mm": round(TL.tip_height(arm, q, tool) * 1000.0, 1), "torque": "on" if arm.is_torqued() else "off"}


def _term_as_interrupt():
    """A cancelled or timed-out tool call ends the process with SIGTERM: turned into KeyboardInterrupt, so that
    the result still says where the arm is (as fd does for `draw`)."""
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, signal.default_int_handler)


class _JawNotHeld(Exception):
    pass


def _power_rigid(arm):
    """Torque on, with the tool as rigid as it will work. A tool that is taped to the moving jaw and held closed by
    a squeeze (tooling: Tool.gripper_goal_raw) loses the squeeze whenever the torque was off and then floats
    several millimetres in the jaw's backlash: tooling.hold_jaw puts it back. When the jaw stands elsewhere than
    the tool file says, the tip would be measured where it will not be while painting: _JawNotHeld, before the arm
    moves. Returns what hold_jaw answered (None: this tool, or this tooling, has no jaw hold)."""
    import tooling as TL
    from robot import SafetyError
    TL._power(arm, None)
    hold = getattr(TL, "hold_jaw", None)
    if hold is None or getattr(arm.tool, "gripper_goal_raw", None) is None:
        return None
    try:
        return hold(arm)
    except SafetyError as e:
        raise _JawNotHeld(str(e)) from e


def _waypoints(plan: dict) -> list:
    import aircal as AC
    return AC.plan_waypoints(plan)


class _PhotoNotStored(Exception):
    """A photo was taken and could not be written (a full disk, a folder that cannot be written)."""


def _in_project(path: str) -> bool:
    """Does this file lie in the project folder (where the rig's own state files are)?"""
    import tazuna_paths
    return (os.path.realpath(path) + os.sep).startswith(os.path.realpath(tazuna_paths.ROOT) + os.sep)


# ----------------------------------------------------------------------------- folding the arm away
def _painting() -> dict:
    """What the painting feature knows about the table, for folding the arm away: {"PS": the paintskill module,
    "path": the set-up file, "setup": the set-up, "error": why the file cannot be used} (each None when it does
    not apply).

    Why (2026-11-13): the tool's default rest pose (pan 20) has a brush 4 cm INSIDE the sheet; in the morning the
    wrist sank there with the torque off until the brush stood on the paper. `fd paint rest` chooses a fold that
    keeps the brush, and the place it would sink to, off the canvas (paintskill.rest_choice); this command
    folded into the default pose and named `fd tool rest` in every hint.

    No painting set-up, and everything is as it was: the paint modules are not there or have no rest_choice
    (they belong to tools/paint: asked for through getattr), there is no set-up file, or the tool file is named
    by $FD_TOOL while the set-up is the rig's own (tests and experiments with a scratch tool: that set-up
    describes another table; `fd tool rest` draws the same line)."""
    from robot import SafetyError
    out = {"PS": None, "path": None, "setup": None, "error": None}
    try:
        import paintskill as PS
    except Exception:           # the painting feature is not there, or does not load: calibration must still work
        return out
    path_of, load, choice = (getattr(PS, n, None) for n in ("setup_path", "load_setup", "rest_choice"))
    if not (callable(path_of) and callable(load) and callable(choice)):
        return out
    if os.environ.get("FD_TOOL") and not os.environ.get("FD_PAINT_SETUP"):
        return out
    path = path_of()
    if not os.path.isfile(path):
        return out
    out.update(PS=PS, path=path)
    try:
        out["setup"] = load(path)
    except SafetyError as e:    # (paintskill.Refusal: a set-up that cannot be read or used)
        out["error"] = str(e)
    return out


def _pan_of(arm_report) -> float | None:
    """The base angle (deg) of an arm as a result reports it ({"q_deg": [...]}; None when it could not be read)."""
    q = (arm_report or {}).get("q_deg") if isinstance(arm_report, dict) else None
    return float(q[0]) if isinstance(q, (list, tuple)) and len(q) == 5 else None


def _fold_hint(paint: dict, sim: bool = False, keep: bool = False, pan_deg: float | None = None) -> str:
    """The command that folds the arm away, for a `next`: `fd paint rest` when a painting set-up is on record (it
    folds where the brush is off the canvas), else `fd tool rest --pan=A --keep-torque` with A the base angle
    the arm stands at (pan_deg), and the reason for choosing another. keep: the hint is for folding with the
    torque left on.

    A command in a `next` works as it is typed. Until 2026-11-14 this one read `fd tool rest --pan P
    --keep-torque`, which fd refuses; without the placeholder it would be the tool's default pose (pan 20), over
    whatever lies in front of the robot. Where the arm points now, its tool is over nothing it has not been
    over already. pan_deg None (the arm could not be read): words, no command."""
    fd = "fd --sim" if sim else "fd"
    if paint.get("path"):
        broken = " (once the painting set-up can be read again: `fd paint status` says what is wrong with it)" \
            if paint.get("error") else ""
        if keep:
            return (f"`{fd} paint rest --keep-torque` folds the arm away where the brush is off the canvas, torque on"
                    + broken)
        return (f"`{fd} paint rest` folds the arm away where the brush is off the canvas and switches the torque off "
                f"(--keep-torque leaves it on); not `fd tool rest`, whose default pose has the brush over the picture"
                + broken)
    why = ("a base angle at which nothing lies under the tool that it must not touch: with the torque off the wrist can "
           "sink until the tool stands on what is under it, as on 2026-11-13")
    if pan_deg is None:
        return (f"fold the arm away with {fd} tool rest, --keep-torque and --pan with {why} (`{fd} state` reads where the "
                "arm stands); the same command without --keep-torque then switches the torque off")
    pan = math.trunc(float(pan_deg) * 10.0) / 10.0          # (towards 0: never beyond a pan limit the arm stands at)
    return (f"`{fd} tool rest --pan={pan:g} --keep-torque` folds the arm away where it points now, torque on (give --pan "
            f"another angle when that is not {why}); `{fd} tool rest --pan={pan:g}` then switches the torque off")


def _store_first(stores: str, far_m: float | None) -> str:
    """The front of a fold hint after this run solved a tip that the TOOL FILE does not have: `fd paint rest` and
    `fd tool rest` plan with the tip on file and would move the arm again. stores: the command that stores the
    solved tip (no motion).

    far_m: how far the solved tip lies from the registered one when that is more than LOW_MAX_JUMP_M (2026-11-14,
    simulated: 93 mm apart, and the fold with the registered tip ended 17 mm under the table). None: the two are
    near, and the tool file has no tip on record as measured: `fd paint rest` then does not switch the torque off
    where the arm lies, it first folds the arm again, as high as it folds (the wrist from 44 to 32 deg, the tip
    from 40 to 80 mm: a pose in which torque-off was never tried on the real arm; round 3, T4)."""
    if far_m is None:
        return (f"FIRST `{stores}` (no motion): it stores the tip solved in this run and puts it on record as measured. Until "
                "then the tool file has a tip that was never measured, and what folds the arm or switches it off next does "
                "not leave it where it lies: it folds it again, by the registered tip and as high as the arm folds. THEN ")
    return (f"FIRST `{stores}` (no motion): it stores the tip solved in this run, which is {far_m * 1000:.0f} mm from the "
            "registered one and is what the arm was planned with since. Until it is stored every other command folds "
            "with the registered tip, which the photos contradict. THEN ")


def _rise_mm(flown: dict):
    """What aircal.fly_look_way measured before the base turned: the least any point of a tool had gained (mm)."""
    v = flown.get("rise_at_turn_m")
    return None if v is None else round(float(v) * 1000.0, 1)


def _side_note(aside: dict | None) -> str | None:
    """Where the first look takes the arm for the photo "before", and how it gets there (round 3b, U2): said in the
    --dry plan, in the look's record and after a refused look, because it happens BEFORE the look has judged the
    registered tip. Since the evening of 2026-11-14 the base turns only with the tool lifted (aircal.look_way): what
    the tool passes over on the way is no longer what has to be cleared, the place where it comes down still is."""
    if not aside:
        return None
    import aircal as AC
    lift = AC.LOOK_LIFT_M * 100.0
    at = aside.get("fingertip_xy")
    where = f" (the fingertip 12 cm above x {at[0]:.2f}, y {at[1]:.2f} m, the fingers pointing on outwards)" if at else ""
    return (f"for the photo before, the arm goes to the look pose turned {aside['turn_deg']:+g} deg about its base (pan "
            f"{aside['pan_deg']:.0f} deg) and back. Its base turns only after the gripper was lifted {lift:.0f} cm straight up, "
            f"and the arm comes straight down afterwards: whatever is taped to the gripper, of any length, passes over that "
            f"side of the table at least {lift:.0f} cm above the table ({lift:.0f} cm above where it hung before the lift). "
            f"Where the arm comes down for the photo{where}, the tool hangs as it does at the look pose itself, and one "
            "longer than registered hangs lower. Before a tool that was not measured is run, that place and the 20 cm "
            "beyond it must be clear of anything higher than about 2 cm; what the tool only passes over on the way need "
            "not be moved unless it is higher than about 6 cm")


def _hold_next(paint: dict, again: str, pan_deg: float | None, measured: bool) -> str:
    """`next` for an arm this command did NOT fold because nothing confirms where the tip of its tool is (round 3,
    T1): look first, then what works. again: the command that was typed, without --save / --force. measured: the
    tool file has the tip on record as measured, and the photos of this run speak against it.

    NO fold is named here, with or without a painting set-up (round 3b, U1): `fd paint rest` and `fd tool rest`
    plan with the registered tip, which is the one nothing confirms. Typed after a held result, they ended ok with
    the real tip 2 to 22 mm under the table in simulation. What is named: measure and register the tool and run
    the command again, or take the tool off by hand. (paint and pan_deg are kept for the callers; not used.)"""
    ruler = ("measure the tool with a ruler (where its tip is from the gripper: along the fingers, towards the fixed jaw, "
             "sideways) and register what you measured with fd tool set and --tip=X,Y,Z (metres, gripper frame; "
             "`fd tool status` shows what is registered"
             + ("; a changed tip is no longer on record as measured" if measured else "")
             + f"), then `{again}` again: it starts from where the arm stands")
    away = ("type NO fold while the tip is not confirmed (the folds of the paint and tool commands plan with the "
            "registered tip" + (", the one the photos speak against: register the tool anew first" if measured else "")
            + "; a tool that stands elsewhere than registered can end in the table that way). If the arm has to be put "
              "away before that: support the arm by hand, take the tool off, then `fd tool detach`")
    return ("LOOK at the arm and at the tool before another command moves it: the arm holds where the last photo was taken "
            "(torque on) and was NOT folded. What works next: " + ruler + ". To put the arm away: " + away)


def _torque_on_next(paint: dict, fold: dict | None, first: str = "", pan_deg: float | None = None) -> str:
    """`next` for an arm this command leaves with the torque on (the result's "arm"). fold: how --end rest folded
    it (_fold), None: it stands at its last pose (--end stay). first: _store_first, when it applies. pan_deg:
    the base angle it stands at (for _fold_hint)."""
    if fold is None:
        return "the torque is ON and the arm stands at its last pose; " + first + _fold_hint(paint, pan_deg=pan_deg)
    if paint.get("path"):
        return ("the torque is ON; " + first + "`fd paint rest` switches it off where the arm is folded now (the brush "
                "off the canvas). Not `fd tool rest`: its default pose has the brush over the picture")
    return (f"the torque is ON and the arm is folded in the tool's rest pose (pan {fold['pan_deg']:g}); " + first
            + _fold_hint(paint, pan_deg=fold["pan_deg"] if pan_deg is None else pan_deg))


def _fold_plan(arm, tool, paint: dict) -> dict:
    """How --end rest will fold the arm with this tool, decided before anything moves: {"by": "paint" | "tool",
    "pan_deg", "clearance_m", ...}. SafetyError (a paintskill.Refusal with its own stage and next) when the
    painting set-up cannot be used or leaves no folded pose off the canvas, or when the arm cannot rest with
    this tool at all. No motion."""
    import tooling as TL
    from robot import SafetyError
    if paint.get("error"):
        raise SafetyError(f"a painting set-up is on record and cannot be used ({paint['error']}): where the canvas "
                          "lies is not known, so the arm cannot be folded away from it")
    if paint.get("setup") is not None:
        carried = arm.tool
        arm.tool = tool                                     # (rest_choice folds the tool the arm carries)
        try:
            choice = dict(paint["PS"].rest_choice(arm, paint["setup"]))
        finally:
            arm.tool = carried
        choice.pop("q", None)
        return {"by": "paint", "pan_deg": float(choice["pan_deg"]), "clearance_m": float(choice["clearance_m"]),
                "clear_of_canvas_mm": choice.get("clear_of_canvas_mm"), "near": choice.get("near"),
                "chosen": choice.get("chosen"), "note": choice.get("note"), "setup": paint["path"]}
    TL.rest_q(arm, tool)                                    # (SafetyError: no rest pose with this tool)
    return {"by": "tool", "pan_deg": float(TL.REST_PAN_DEG), "clearance_m": float(TL.REST_CLEARANCE)}


def _fold(arm, paint: dict) -> tuple:
    """--end rest: fold the arm away with the torque ON, for the tool the arm carries now (in a run: the tool
    with the tip its photos gave, see _run). With a painting set-up: the pose `fd paint rest` would choose
    (paintskill.rest_choice); else the tool's rest pose.
    -> (tooling.rest's result, the fold as _fold_plan describes it, with the tip it was planned with, plus what
    paintskill.rest_report says of it when that function exists). SafetyError before it moves (no folded pose
    for this tip, or no safe way into it: the arm then holds where it is); MotionAborted on the way."""
    import tooling as TL
    from robot import SafetyError
    fold = _fold_plan(arm, arm.tool, paint)
    fold["planned_with_tip_mm"] = _mm(arm.tool.tip_offset)
    if fold["by"] == "paint":
        r = TL.rest(arm, clearance_m=fold["clearance_m"], pan_deg=fold["pan_deg"], torque_off=False)
        report = getattr(paint["PS"], "rest_report", None)
        if callable(report):    # (tools/paint: lands with the same update; its dict has "ok" = clear of the canvas)
            try:
                rep = report(arm, arm.tool, paint["setup"], fold["pan_deg"], fold["clearance_m"])
                if isinstance(rep, dict):
                    fold["canvas"] = {k: v for k, v in rep.items() if k != "q"}
            except (SafetyError, TypeError, ValueError, KeyError):
                pass
    else:
        r = TL.rest(arm, torque_off=False)
    return r, fold


# ----------------------------------------------------------------------------- the jaw the photos were taken with
def _photo_times(records) -> tuple:
    """(first, last) time of the photos as "YYYY-MM-DDTHH:MM:SS", or (None, None). A record of this command
    carries "at"; the captures of 2026-11-13 (scripts) only a time of day: for those the photo files say when."""
    times = []
    for r in records:
        at = r.get("at")
        if isinstance(at, str) and len(at) >= 19:
            times.append(at[:19])
            continue
        try:
            times.append(time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(str(r["image"])))))
        except (OSError, KeyError):
            pass
    return (min(times), max(times)) if times else (None, None)


def _jaw_seen(records) -> dict | None:
    """Where the jaw stood while the photos were taken, as the servo said (aircal.jaw_reading in every record):
    {"raw": [lowest, highest], "squeezed": True | False | None (None: the records do not say), "photos"}; None
    for records without it (captures from before 2026-11-13 evening)."""
    def number(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    read = [r for r in records if number(r.get("gripper_raw"))]
    if not read:
        return None
    raws = [int(r["gripper_raw"]) for r in read]
    # (the goal: a number, or None for a gripper whose torque was off; a record without the key does not say)
    beyond = sorted(int(r["gripper_raw"]) - int(r["gripper_goal_raw"]) if r["gripper_goal_raw"] is not None else 0
                    for r in read if "gripper_goal_raw" in r and (r["gripper_goal_raw"] is None or number(r["gripper_goal_raw"])))
    squeezed = bool(beyond[len(beyond) // 2] >= SQUEEZE_ON_TICKS) if beyond else None
    return {"raw": [min(raws), max(raws)], "squeezed": squeezed, "photos": len(raws)}


def _jaw_median(records) -> int | None:
    """The middle one of the jaw positions the servo reported during the photos (None: the records carry none)."""
    raws = sorted(int(r["gripper_raw"]) for r in records
                  if isinstance(r.get("gripper_raw"), (int, float)) and not isinstance(r.get("gripper_raw"), bool)
                  and math.isfinite(r["gripper_raw"]))
    return raws[len(raws) // 2] if raws else None


def jaw_mismatch(taken_with: dict | None, records, tool) -> list:
    """Why these photos do NOT show the tool as its file describes it now (empty: nothing speaks against it).
    A tip belongs to one jaw state: the brush of 2026-11-13 had its tip at (-5.1, -2.5, 60.4) mm with the jaw
    at raw 2182 and at (16.0, -1.1, 55.8) mm closed and squeezed at raw 2041, 2.2 cm apart, under the same
    name and attach time; loose in the jaw's backlash it first touched the paper 4 mm higher than squeezed.

    What is compared, in this order of trust:
    - where the servo said the jaw stood during the photos (the records of this command), against the tool
      file's gripper_raw (PHOTO_JAW_TOL_TICKS) and, where the records say it, whether it was squeezed against
      whether the file records a hold (not for a tool recorded as held BETWEEN the jaws or taped to the fixed
      one: the first is squeezed by its nature, the second does not move with the jaw, and neither can have a
      hold on record);
    - for records without that (the captures of 2026-11-13): what the tool file said then (the records' "tool")
      against what it says now: gripper_raw and the holding goal, within tooling.JAW_SAME_TICKS, a value on one
      side only counting as a difference;
    - on_jaw, when both say on which jaw the tool is and disagree ("not said" contradicts nothing: it is a label,
      `fd tool jaw mount` writes it without moving anything).
    A tool that is recorded on the FIXED jaw does not go with the moving jaw: where that stood is not compared
    (tooling.jaw_tolerance draws the same line)."""
    import tooling as TL
    same = int(getattr(TL, "JAW_SAME_TICKS", 3))
    now_raw = getattr(tool, "gripper_raw", None)
    now_goal = getattr(tool, "gripper_goal_raw", None)
    now_on = getattr(tool, "on_jaw", None)
    why = []
    seen = _jaw_seen(records)
    if now_on == "fixed":
        pass
    elif seen is not None:
        lo, hi = seen["raw"]
        where = f"raw {lo}" if lo == hi else f"raw {lo} .. {hi}"
        if now_raw is not None and max(abs(lo - int(now_raw)), abs(hi - int(now_raw))) > PHOTO_JAW_TOL_TICKS:
            why.append(f"the jaw stood at {where} while the photos were taken (read from the servo); the tool file now "
                       f"has it at raw {int(now_raw)}")
        if seen["squeezed"] is not None and now_on not in ("between", "fixed") and seen["squeezed"] != (now_goal is not None):
            why.append("the jaw was " + ("held shut with a squeeze" if seen["squeezed"] else "NOT held with a squeeze")
                       + " while the photos were taken (read from the servo); the tool file now records "
                       + (f"a hold (goal raw {int(now_goal)})" if now_goal is not None else "no hold"))
    elif taken_with is not None:
        t_raw, t_goal = taken_with.get("gripper_raw"), taken_with.get("gripper_goal_raw")
        if (t_raw is None) != (now_raw is None) or (t_raw is not None and abs(int(t_raw) - int(now_raw)) > same):
            why.append(f"the photos were taken with the jaw at raw {t_raw} (the tool file of that moment); the tool file "
                       f"now has it at raw {now_raw}")
        if (t_goal is None) != (now_goal is None) or (t_goal is not None and abs(int(t_goal) - int(now_goal)) > same):
            why.append("the photos were taken with " + ("no jaw hold on record" if t_goal is None else
                                                         f"the jaw held by a squeeze (goal raw {t_goal})")
                       + "; the tool file now records " + ("no hold" if now_goal is None else f"a hold (goal raw {int(now_goal)})"))
    t_on = None if taken_with is None else taken_with.get("on_jaw")
    if t_on is not None and now_on is not None and t_on != now_on:
        why.append(f"the photos were taken with the tool recorded on_jaw = {t_on}; the tool file now says {now_on}")
    return why


def other_tool(taken_with: dict | None, tool) -> str | None:
    """Why these are photos of another tool, or of another taping of it (name and attach time of the records'
    "tool" against the tool file now), or None."""
    if taken_with and (taken_with.get("name") != tool.name or taken_with.get("attached_at") != tool.attached_at):
        return (f"the photos were taken with tool '{taken_with.get('name')}' attached at {taken_with.get('attached_at')}; "
                f"the tool registered now is '{tool.name}' attached at {tool.attached_at}")
    return None


def _jaw_elsewhere(arm, tool) -> str | None:
    """Before a capture, reads only: the jaw of a tool that is NOT held by a squeeze stands elsewhere than the
    tool file has it (more than PHOTO_JAW_TOL_TICKS). Whatever is measured then is the tip of a jaw state the
    file does not describe, and could not be stored (not_this_tool): better said before three minutes of poses
    than after. None: it stands where the file says, the file records no jaw position, or the tool is held by a
    squeeze (tooling.hold_jaw puts that back, or refuses, when the torque comes on)."""
    import aircal as AC
    raw = getattr(tool, "gripper_raw", None)
    if raw is None or getattr(tool, "gripper_goal_raw", None) is not None or getattr(tool, "on_jaw", None) == "fixed":
        return None
    try:
        pos = int(AC.jaw_reading(arm)["gripper_raw"])
    except Exception:           # (an arm that cannot be read fails where it is used, with the text for that)
        return None
    if abs(pos - int(raw)) <= PHOTO_JAW_TOL_TICKS:
        return None
    return f"the jaw stands at raw {pos}, {abs(pos - int(raw))} ticks from where the tool file has it (raw {int(raw)})"


def not_this_tool(taken_with: dict | None, records, tool) -> list:
    """Why a tip from these photos must not be stored into `tool` (the tool file as it is now), whatever the
    options: they are photos of another tool or taping (other_tool), or they were taken with the jaw in another
    state (jaw_mismatch). Empty: nothing speaks against it."""
    other = other_tool(taken_with, tool)
    return [other] if other else jaw_mismatch(taken_with, records, tool)


def _anew(mode: str) -> str:
    """The command that measures the tool as it is registered now, and stores it."""
    return "fd calib-air --tip-only --save" if mode == "tip" else "fd calib-air --sets air,low --save"


def hold_recorded_since(taken_with: dict | None, records, tool) -> bool:
    """Is the ONE thing that keeps these photos out of the tool file a jaw hold that went on record after they
    were taken? True when their records carry no reading of the jaw (the captures of 2026-11-13: scripts), the
    tool file of that moment had the jaw where the file has it now (tooling.JAW_SAME_TICKS) and recorded no
    hold, and the file records one now.

    That is the rig's own case (review of 2026-11-14): the evening's last capture was taken with the jaw closed
    at raw 2041 and squeezed by a script, which nothing recorded; `fd tool jaw close` then put the hold on
    record, and `fd calib-air --from DIR --save` refused the photos, as it must: what a tool file said is no
    proof of the jaw. The comparison stays as it is. What this answers is whether the refusal can name a way
    back (_way_back): for records in which the SERVO said that the jaw was not squeezed there is none."""
    import tooling as TL
    if taken_with is None or other_tool(taken_with, tool) is not None or _jaw_seen(records) is not None:
        return False
    t_raw, t_goal = taken_with.get("gripper_raw"), taken_with.get("gripper_goal_raw")
    now_raw, now_goal = getattr(tool, "gripper_raw", None), getattr(tool, "gripper_goal_raw", None)
    if t_raw is None or now_raw is None or t_goal is not None or now_goal is None:
        return False
    if abs(int(t_raw) - int(now_raw)) > int(getattr(TL, "JAW_SAME_TICKS", 3)):
        return False
    return len(jaw_mismatch(taken_with, records, tool)) == 1        # (nothing else speaks against them)


def _way_back(folder: str, mode: str) -> str:
    """For hold_recorded_since: how photos taken BEFORE the hold went on record are stored after all, by commands
    that work as typed. The order is the one `fd paint status` leads through for such a tool file: the tip first."""
    stores = _from_words(folder, *([] if mode == "tip" else ["--sets", "air,low"]), "--save")
    return ("The way back, ONLY when these photos were taken with the jaw closed and squeezed as it is now (the capture of "
            "2026-11-13 evening: the squeeze was put on by a script, which their records could not know): `fd tool jaw close "
            f"--squeeze 0` takes the hold off the record (the jaw stays closed), then `{stores}` stores the tip, then `fd "
            "tool jaw close` holds the jaw again (a close that ends where the file has the jaw keeps the tip and its record)")


def _not_storable(res: dict, mode: str, why: list, back: str | None = None) -> None:
    """A measurement that was NOT asked to be stored, from photos that could not be: say so instead of naming
    `--save` as the next step (it would be refused). back: _way_back, when there is one."""
    res["not_this_tool"] = list(why)
    res["next"] = ("these photos do not show the tool as its file describes it now (" + "; ".join(why) + "): --save would "
                   f"refuse them, with or without --force. `fd tool jaw status` shows the jaw; `{_anew(mode)}` measures "
                   "the tool as it is registered now" + (". " + back if back else ""))
    res["say"] = ((res.get("say") or "").replace(SAY_NOT_SAVED, "").replace(SAY_FORCE, "")
                  + "この写真は、いま登録されているツールの状態とは違う状態で撮ったものなので、この値は保存できません。"
                  + (SAY_WAY_BACK if back else ""))


# ----------------------------------------------------------------------------- photos -> fit -> verdict
def _load_images(records) -> tuple:
    """(images, records that could be read, names of those that could not)."""
    import cv2
    images, good, missing = [], [], []
    for r in records:
        img = cv2.imread(r["image"]) if os.path.isfile(r["image"]) else None
        if img is None:
            missing.append(os.path.basename(str(r["image"])))
            continue
        images.append(img)
        good.append(r)
    return images, good, missing


def _solve(mode: str, images, records, cam, guess) -> dict:
    """The fit of the photos (aircal.measure_tip / measure_full; ValueError when there is none)."""
    import aircal as AC
    import camera as C
    qs = np.array([r["q"] for r in records], dtype=float)
    if mode == "tip":
        fit = AC.measure_tip(images, qs, cam, tip_guess=guess)
        fit["check"] = AC.camera_check(qs, fit.pop("points"), cam, fit["tip_offset"])
        return fit
    fit = AC.measure_full(images, qs, guess if guess is not None else (0.0, 0.0, 0.06), C.F_PINNED, C.W / 2.0, C.H / 2.0,
                          prior=cam)
    if cam is not None:
        import cv2
        rmat, _ = cv2.Rodrigues(np.asarray(fit["rvec"], dtype=float))
        fit["moved_mm"] = round(float(np.linalg.norm(np.asarray(fit["center"]) - cam.center)) * 1000.0, 1)
        fit["turned_deg"] = round(math.degrees(float(np.linalg.norm(cv2.Rodrigues(rmat @ cam.R.T)[0]))), 2)
    return fit


def _camera_moved(fit: dict, cam) -> str | None:
    """Why the camera seems to have moved since its model was fitted (tip-only solve), or None."""
    why = []
    limit = max(RMS_OVER_MODEL * float(cam.rms), RMS_FLOOR_PX) if cam.rms else RMS_ABS_MAX_PX
    if fit["rms_px"] > limit:
        why.append(f"the residuals are {fit['rms_px']:.1f} px rms, above {limit:.1f} px ("
                   + (f"{RMS_OVER_MODEL} x the {float(cam.rms):.1f} px the camera model's own fit left" if cam.rms
                      else "the model file carries no rms of its own") + ")")
    chk = fit.get("check")
    if (chk and chk.get("d") is not None and chk.get("used", 0) >= CHECK_MIN_USED and chk["d"] >= CHECK_D
            and (chk["moved_mm"] >= CHECK_MM or chk["turned_deg"] >= CHECK_DEG)):
        why.append(f"the same photos, solved for the camera too, put it {chk['moved_mm']:.0f} mm and {chk['turned_deg']:.1f} deg "
                   f"from the model ({chk['d']:.0f} formal standard errors)")
    return "; ".join(why) if why else None


def judge(mode: str, fit: dict, cam) -> dict:
    """What the numbers of a fit say. -> {"problems": [...] (why it must not be stored, or relied on to plan the
    low set; empty = it passes the gates), "moved": text | None (tip-only: the camera seems to have moved),
    "few": bool (most photos have no candidate near the solution)}"""
    import tooling as TL
    from robot import SafetyError
    used, photos = int(fit["used"]), int(fit["photos"])
    share = used / max(1, photos)
    problems, moved = [], None
    if mode == "tip":
        sigma = np.asarray(fit["sigma"], dtype=float)
        if used < TIP_MIN_USED:
            problems.append(f"{used} photos carry the fit (at least {TIP_MIN_USED} are wanted)")
        if share < TIP_MIN_SHARE:
            problems.append(f"only {used} of the {photos} photos have a candidate near the solution (at least "
                            f"{TIP_MIN_SHARE:.0%} are wanted)")
        if not (np.all(np.isfinite(sigma)) and float(sigma.max()) <= TIP_MAX_SIGMA_M):
            problems.append(f"the formal error of the tip is {_mm(sigma)} mm (at most {TIP_MAX_SIGMA_M * 1000:.0f} mm)")
        moved = _camera_moved(fit, cam)
        if moved:
            problems.append("the camera seems to have moved: " + moved)
    else:
        sigma = np.asarray(fit["sigma"], dtype=float)[6:9]
        csig = np.asarray(fit["center_sigma"], dtype=float)
        if used < FULL_MIN_USED:
            problems.append(f"{used} photos carry the fit (at least {FULL_MIN_USED} are wanted for camera pose + tip)")
        if share < FULL_MIN_SHARE:
            problems.append(f"only {used} of the {photos} photos agree with the solution (at least {FULL_MIN_SHARE:.0%} "
                            "are wanted)")
        if fit["rms_px"] > RMS_ABS_MAX_PX:
            problems.append(f"rms {fit['rms_px']:.1f} px is above {RMS_ABS_MAX_PX:.0f} px")
        if not (np.all(np.isfinite(sigma)) and float(sigma.max()) <= FULL_MAX_TIP_SIGMA_M):
            problems.append(f"the formal error of the tip is {_mm(sigma)} mm (at most {FULL_MAX_TIP_SIGMA_M * 1000:.0f} mm)")
        if not (np.all(np.isfinite(csig)) and float(csig.max()) <= FULL_MAX_CENTER_SIGMA_M):
            problems.append(f"the formal error of the camera centre is {_mm(csig)} mm (at most "
                            f"{FULL_MAX_CENTER_SIGMA_M * 1000:.0f} mm)")
        if fit["center"][2] < 0.05:
            problems.append(f"the camera centre would be {fit['center'][2] * 1000:.0f} mm above the robot's base plane")
    try:
        TL.check_offset(fit["tip_offset"])
    except SafetyError as e:
        problems.append(str(e))
    return {"problems": problems, "moved": moved, "few": share < FEW_SHARE}


def _report(res: dict, mode: str, fit: dict, records, registered, cam, verdict: dict) -> None:
    """The numbers of a fit, into the result (no files, no motion)."""
    tip = np.asarray(fit["tip_offset"], dtype=float)
    names = [os.path.basename(str(r["image"])) for r in records]
    sigma = np.asarray(fit["sigma"], dtype=float)
    res.update(poses=len(records), photos=int(fit["photos"]), with_candidate=int(fit["n"]), used=int(fit["used"]),
               rms_px=round(float(fit["rms_px"]), 2), max_px=round(float(fit["max_px"]), 1),
               tip_mm=_mm(tip), tip_sigma_mm=_mm(sigma if mode == "tip" else sigma[6:9], 2),
               tip_offset=[round(float(v), 5) for v in tip], detector=fit.get("detector"),
               unused=[{"photo": names[u["i"]], "why": u["why"]} for u in fit["unused"]],
               gates={"passed": not verdict["problems"], "problems": verdict["problems"]})
    if registered is not None:
        res.update(registered_tip_mm=_mm(registered),
                   from_registered_mm=round(float(np.linalg.norm(tip - np.asarray(registered, dtype=float))) * 1000.0, 1))
    if mode == "tip":
        res["camera_model"] = {"rms_px": None if cam.rms is None else round(float(cam.rms), 2), "check": fit.get("check")}
    else:
        res["camera"] = {"center": [round(float(v), 4) for v in fit["center"]],
                         "center_sigma_mm": _mm(fit["center_sigma"]),
                         "moved_mm": fit.get("moved_mm"), "turned_deg": fit.get("turned_deg"),
                         "heads": fit.get("heads"), "cond": round(float(fit["cond"]), 1) if math.isfinite(fit["cond"]) else None}
    warnings = []
    if verdict["moved"]:
        warnings.append("the camera seems to have moved since its model was fitted: " + verdict["moved"]
                        + ". A tip-only solve then puts the camera's move into the tip: this tip is not to be trusted")
    if verdict["few"]:
        outside = sum(1 for u in fit["unused"] if u["why"].startswith(("tip outside", "tip at the edge")))
        far = "" if res.get("from_registered_mm") is None else \
            f" The poses were planned with a registered tip {res['from_registered_mm']:.0f} mm from the solved one."
        warnings.append(f"only {fit['used']} of the {fit['photos']} photos have a candidate near the solution"
                        + (f" (in {outside} of the others the tip is outside the picture or at its edge)" if outside else "")
                        + "." + far)
    if warnings:
        res["warnings"] = warnings
        res["warning"] = warnings[0]


# what `say` advises about storing (taken out again by _save once the result IS stored)
SAY_NOT_SAVED = "保存はしていません（保存するときは --save）。"
SAY_FORCE = "測り直してください（それでも保存するときは fd calib-air --from 写真のフォルダ --save --force。アームは動きません）。"
SAY_WAY_BACK = ("写真を撮ったときも、いまと同じようにあごを閉じて押し付けていたと分かっている場合だけ、測り直さずに戻せます: "
                "fd tool jaw close --squeeze 0 → この写真を --save → fd tool jaw close の順です（結果の next のとおり）。")
SAY_STORE_FIRST = ("測った先端は登録値と大きく違うため、アームは測った先端の位置で計画して動かしました。fd paint rest や "
                   "fd tool rest の前に、必ず先にこの測定結果を保存してください（保存する前は、登録値の先端で畳み直してしまいます）。")
SAY_STORE_UNMEASURED = ("道具のファイルの先端は、この測定結果を保存するまで「未測定」のままです。fd paint rest や fd tool rest の前に、"
                        "先にこの測定結果を保存してください（保存する前は、畳み直してからトルクを切ります）。")
SAY_HOLD = ("ツール先端の位置を確かめられなかったので、アームを畳んでいません（最後に写真を撮った姿勢で、トルク ON のまま保持して"
            "います）。次に何かを動かす前に、アームと道具を目で見て確かめてください。道具を定規で測って fd tool set で登録し直してから、"
            "もう一度 fd calib-air を実行してください。先端が確かめられるまで、畳むコマンド（fd paint rest / fd tool rest）は"
            "打たないでください（登録された先端で計画するので、道具が机に当たることがあります）。")


def _from_words(folder: str, *more: str) -> str:
    """`fd calib-air --from FOLDER ...` as it is typed (no arm, no camera: the same words with and without --sim)."""
    return " ".join(["fd calib-air --from", folder, *more])


def _verdict_text(res: dict, mode: str, verdict: dict, folder: str) -> None:
    """`next` and `say` of a finished measurement that was not stored. folder: where its photos and records are."""
    t = res["tip_offset"]
    tip_arg = f"{t[0]:.4f},{t[1]:.4f},{t[2]:.4f}"     # (typed as --tip=...: fd reads `--tip -0.12,...` as an option without a value)
    sets = "" if mode == "tip" else " --sets air,low"
    head = (f"ツール先端を測りました: ({res['tip_mm'][0]}, {res['tip_mm'][1]}, {res['tip_mm'][2]}) mm"
            f"（写真 {res['photos']} 枚中 {res['used']} 枚、残差 {res['rms_px']} px）。")
    if mode == "full":
        c = res["camera"]["center"]
        head += f"カメラの位置は ({c[0]}, {c[1]}, {c[2]}) m です。"
    if verdict["moved"]:
        res["next"] = ("`fd calib-air --sets air,low` (camera pose and tip together, about 5 minutes): the camera seems to "
                       "have moved, so this tip is not stored")
        res["say"] = head + "ただし、カメラが動いたようです（この先端の値は信用できません）。カメラごと測り直す fd calib-air --sets air,low が必要です。"
    elif verdict["few"]:
        res["next"] = (f"`fd tool set --tip={tip_arg}` registers this tip as the better guess (the poses are then planned "
                       f"with it), then `fd calib-air{sets}` again")
        outside = sum(1 for u in res["unused"] if u["why"].startswith(("tip outside", "tip at the edge")))
        res["say"] = (head + ("ただし、登録した先端の位置が実際と大きく違うため、多くの写真で先端が写真の外でした。"
                              if 2 * outside >= len(res["unused"]) else
                              "ただし、多くの写真で先端が見つかっていません（隠れている、または写真の外にあります）。")
                      + "この値を fd tool set で登録してから、もう一度測ってください。")
    elif verdict["problems"]:
        res["next"] = (f"the fit does not pass the gates (see \"gates\"): look at the photos in {folder} and measure again "
                       f"(`fd calib-air{sets}`); `{_from_words(folder, *sets.split(), '--save', '--force')}` stores it "
                       "anyway (no motion: --force only stores)")
        res["say"] = head + "ただし、精度の基準を満たしていません。" + SAY_FORCE
    else:
        res["next"] = f"`{_from_words(folder, *sets.split(), '--save')}` stores it without moving the arm"
        res["say"] = head + SAY_NOT_SAVED


# ----------------------------------------------------------------------------- storing
def _store_tip(tool, tip, record: dict) -> list:
    """Give the tool the measured tip (in memory) and say what that dropped. tooling.set_measured_tip does it;
    a tooling without it (before the evening of 2026-11-13) gets the same steps: a changed tip drops the pen
    calibration exactly as `fd tool set --tip` does, and the record is kept among the file's other keys."""
    import tooling as TL
    setter = getattr(TL, "set_measured_tip", None)
    if setter is not None:
        return list(setter(tool, tip, record))
    new = [round(float(v), 5) for v in TL.check_offset(tip)]
    dropped = []
    if new != [round(float(v), 5) for v in tool.tip_offset]:
        dropped = [k for k, v in (("contact", tool.contact), ("map", tool.corr_map), ("calibrated", tool.calibrated)) if v]
        tool.contact, tool.corr_map, tool.calibrated = None, None, {}
    tool.tip_offset = new
    if "tip_measured" in getattr(TL.Tool, "__dataclass_fields__", {}):
        tool.tip_measured = dict(record)
    else:
        tool.extra["tip_measured"] = dict(record)
    return dropped


def _save(res: dict, mode: str, fit: dict, verdict: dict, tool, sim: bool, force: bool, taken_with: dict | None,
          records, folder: str) -> None:
    """--save: the tip into the tool file, for the full solve the camera model too (the old one to backups/
    first). Changes `res` (ok, stage, saved, files, next, say).

    tool: the tool as its file has it NOW (the one that is written); taken_with: the tool the photos were taken
    with (the "tool" of their records, None when they name none); records: the photos' records; folder: where
    they are.

    Never stored, whatever the options: photos of another tool or another taping of it (name, attach time), and
    photos taken with the jaw elsewhere than the tool file has it now (jaw_mismatch): such a tip is a
    measurement of something else (where the one difference is a hold that went on record after the photos, the
    refusal names the way back: hold_recorded_since). Stored only with --force: a fit that fails the gates, and
    photos whose records name no tool (nothing says what they show).

    With the tip goes the jaw position the servo reported during the photos, into a tool file that had none
    (_jaw_median). `next` afterwards: `fd paint status` when a painting set-up is on record, else `fd tool
    status`."""
    import camera as C
    import tooling as TL
    from robot import SafetyError
    problems = list(verdict["problems"])
    if tool is None:
        res.update(ok=False, stage="save", saved=False, reason="no tool is registered: there is no tool file to store the tip in",
                   next="`fd tool status` shows what is registered. Register the tool with fd tool attach (--name, and "
                        "--along-fingers with its length from the fingertip in metres), then `fd calib-air` again",
                   say="ツールが登録されていないので、保存できませんでした。")
        return
    sets = "" if mode == "tip" else " --sets air,low"
    anew = _anew(mode)
    wrong = not_this_tool(taken_with, records, tool)
    jaw = bool(wrong) and other_tool(taken_with, tool) is None
    if wrong:
        back = _way_back(folder, mode) if hold_recorded_since(taken_with, records, tool) else None
        res.update(ok=False, stage="save", saved=False, not_this_tool=wrong,
                   reason="not stored (--force does not change that): " + "; ".join(wrong)
                          + ". A tip measured that way is not the tip of the tool as it is registered now",
                   next=("`fd tool jaw status` shows where the jaw stands and what the tool file records. With the jaw as it "
                         f"is to be used (`fd tool jaw close` holds it shut): `{anew}` measures the tip anew; these photos "
                         "belong to the tool file only while it describes the jaw as they were taken"
                         + (". " + back if back else "")
                         if jaw else
                         f"these photos are of another tool, or of another taping of it: `{anew}` measures the tool that is "
                         "registered now"),
                   say=("この写真は、いま登録されているツールの状態（" + ("あごの位置・押さえ" if jaw else "ツール名・取り付け時刻")
                        + "）とは違う状態で撮ったものなので、保存していません（--force でも保存しません）。"
                          f"いまの状態で {anew} を実行して測り直してください。" + (SAY_WAY_BACK if back else "")))
        if back:
            res["way_back"] = True
        return
    if taken_with is None:
        problems.append("the records name no tool: nothing says that these photos show the tool that is registered now")
    model_path = C.model_path()
    if mode == "full" and sim and os.path.realpath(model_path) == os.path.realpath(C.CAM_MODEL_PATH):
        res.update(ok=False, stage="save", saved=False,
                   reason="a simulated call with --save and the full solve would replace the rig's real camera model "
                          f"({C.CAM_MODEL_PATH}): --sim does not protect that",
                   next="run it without --sim if that is what you want",
                   say="--sim では実機のカメラモデルを書き換えません。保存していません。")
        return
    files = {"tool": TL.tool_path(sim)}
    if sim and not os.environ.get("FD_TOOL") and _in_project(files["tool"]):
        # (main refuses such a call at the door; this is the last line before the write, as tooljaw._save has it)
        res.update(ok=False, stage="save", saved=False,
                   reason=f"the simulated tool file would be written into the project folder ({files['tool']}): a simulated "
                          "command never does that",
                   next="run it through fd, with --sim right after fd (fd keeps the simulated tool outside the project "
                        "folder), or name a scratch tool file with $FD_TOOL",
                   say="シミュレーションのツールのファイルをプロジェクトのフォルダに書くことになるため、保存していません。")
        return
    if problems and not force:
        only_no_tool = not verdict["problems"]
        res.update(ok=False, stage="save", saved=False, reason="not stored: " + "; ".join(problems),
                   next=((f"`{_from_words(folder, *sets.split(), '--save', '--force')}` stores it when these ARE photos of "
                          "the tool that is registered now, with its jaw as it is now (--force only stores; no motion)")
                         if only_no_tool else
                         (res.get("next") or f"measure again (`fd calib-air{sets}`); "
                                             f"`{_from_words(folder, *sets.split(), '--save', '--force')}` stores it anyway")))
        res["say"] = ((res.get("say") or "") + ("記録にツールの情報がないので、保存していません（いまのツールの写真だと分かっているときは "
                                                "--save --force）。" if only_no_tool else "基準を満たさないので、保存していません。"))
        return
    _first, last = _photo_times(records)
    seen = _jaw_seen(records)
    record = {"at": last or time.strftime("%Y-%m-%dT%H:%M:%S"), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "by": "calib-air", "solve": MODE_NAME[mode], "rms_px": round(float(fit["rms_px"]), 2),
              "used": int(fit["used"]), "of": int(fit["photos"]), "sigma_mm": res["tip_sigma_mm"], "photos": folder}
    if last is None:
        record["at_note"] = "the time of the photos is not known: \"at\" is the time of saving"
    if seen is not None:
        record["jaw_raw"] = seen["raw"]
        if seen["squeezed"] is not None:
            record["jaw_squeezed"] = seen["squeezed"]
    if problems:
        record["forced"] = problems
    new_tool = TL.Tool.from_dict(tool.to_dict())
    before = [round(float(v), 5) for v in tool.tip_offset]
    # A tool file that never said where its jaw stood (attached with --no-robot, or with the arm off) is not
    # compared with the jaw before a move (tooling.jaw_tolerance is None for it), whatever happens to the jaw
    # afterwards. Where the servo reported the jaw during the photos is the position this tip belongs to: it
    # goes on record with the tip. Not when the readings are further apart than a held jaw may stand from its
    # record: then the photos name no ONE position (said in the result; the file stays without one).
    jaw_recorded, jaw_spread = None, None
    if getattr(new_tool, "gripper_raw", None) is None and seen is not None:
        if seen["raw"][1] - seen["raw"][0] <= PHOTO_JAW_TOL_TICKS:
            jaw_recorded = _jaw_median(records)
        else:
            jaw_spread = seen["raw"]
    try:
        dropped = _store_tip(new_tool, fit["tip_offset"], record)
        if jaw_recorded is not None:
            new_tool.gripper_raw = int(jaw_recorded)
        tmp = None
        if mode == "full":
            model = C.CameraModel(C.F_PINNED, C.W / 2.0, C.H / 2.0, fit["rvec"], fit["tvec"], 0.0, float(fit["rms_px"]),
                                  int(fit["used"]))
            text = model.to_dict()
            stamp = time.strftime("%Y%m%d_%H%M%S")
            if os.path.exists(model_path):
                bdir = _backup_dir(model_path)
                os.makedirs(bdir, exist_ok=True)
                files["camera_model_backup"] = os.path.join(bdir, f"camera_model.before_aircal_{stamp}.json")
                shutil.copy(model_path, files["camera_model_backup"])
            text["source"] = (f"air calibration (fd calib-air), {time.strftime('%Y-%m-%d %H:%M')}: {fit['used']} of "
                              f"{fit['photos']} tool-tip photos, rms {fit['rms_px']:.1f} px, tool '{tool.name}'"
                              + (f"; the model before it is in {files['camera_model_backup']}" if "camera_model_backup" in files
                                 else "") + ("; stored with --force: " + "; ".join(problems) if problems else ""))
            tmp = model_path + ".tmp"
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            with open(tmp, "w") as fh:
                json.dump(text, fh, indent=2)
            files["camera_model"] = model_path
        try:
            TL.save(new_tool, sim)
        except Exception:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
            raise
        if tmp:
            os.replace(tmp, model_path)
    except (SafetyError, OSError) as e:
        res.update(ok=False, stage="save", saved=False, reason=f"could not store the result: {e}",
                   next="neither the tool file nor the camera model was changed; repair the cause and run "
                        f"`{_from_words(folder, *sets.split(), '--save')}`",
                   say="結果を保存できませんでした。")
        return
    moved = float(np.linalg.norm(np.asarray(new_tool.tip_offset) - np.asarray(before))) * 1000.0
    res.update(saved=True, files=files, tip_changed_mm=round(moved, 1), dropped=dropped, tip_measured=record)
    if problems:
        res["forced"] = problems
    if jaw_recorded is not None:
        res["jaw_recorded_raw"] = int(jaw_recorded)
        res.setdefault("notes", []).append(
            f"the tool file had no jaw position on record: raw {int(jaw_recorded)}, where the servo reported the jaw during "
            "the photos, was stored with the tip (gripper_raw). From now on the jaw is compared with it before a move; "
            "`fd tool jaw status` shows both")
    elif jaw_spread is not None:
        res.setdefault("notes", []).append(
            f"the tool file has no jaw position on record, and the jaw stood at raw {jaw_spread[0]} .. {jaw_spread[1]} "
            "during the photos: no single position was stored with the tip, so the jaw of this tool is still not "
            "compared before a move. `fd tool jaw status` shows where it stands")
    # What comes next. A tool with a measured tip is a painting tool (fd's rule): with a painting set-up on record
    # the next step is whatever `fd paint status` finds missing, and the first thing it finds after a stored tip is
    # the contact plane, which holds for ONE brush state and is older than the tool file now. Never the pen
    # calibration: until 2026-11-14 this result led a brush to `fd tool status` and from there to `fd tool
    # calibrate`, whose tip stage lowered the brush 9 .. 15 mm below first touch inside the canvas (simulated).
    setup_path = _painting().get("path")
    pen_dropped = any(k in dropped for k in ("contact", "map", "calibrated"))
    if setup_path:
        res["next"] = ("`fd paint status` says what the painting set-up still needs with this tip. Its contact plane holds "
                       "for one brush state, and the tool file is newer than it now: "
                       + (f"the tip moved by {moved:.1f} mm, so a plane that was measured with the old tip does not hold for "
                          "it and is measured again, not adopted" if moved > 0.5 else
                          "`fd paint contact --adopt` when brush, jaw, sheet and underlay are as they were when it was "
                          "measured, else a new measurement"))
    else:
        res["next"] = ("`fd tool status` shows the tool"
                       + ("; its pen calibration (contact plane, map) was dropped: for a PEN, `fd tool calibrate --pen` before "
                          "the next `fd draw cat` (--pen: a tool with a measured tip counts as a painting tool, and the tip "
                          "stage of that calibration replaces the measured tip)" if pen_dropped else "")
                       + (f"; the tip moved by {moved:.1f} mm: a contact height that was measured with the old tip (a paint "
                          "set-up) no longer holds, measure it again before painting" if moved > 0.5 else ""))
    res["say"] = ((res.get("say") or "").replace(SAY_NOT_SAVED, "").replace(SAY_FORCE, "").replace(SAY_STORE_FIRST, "")
                  .replace(SAY_STORE_UNMEASURED, "")
                  + f"ツール「{tool.name}」の先端を保存しました（登録値から {moved:.1f} mm 変更）。"
                  + ("カメラモデルも保存し、前のモデルは backups に残しました。" if mode == "full" else "")
                  + ("基準を満たしていませんが、--force で保存しました。" if problems else "")
                  + ("次は fd paint status で、絵のセットアップに足りないものを確かめてください（紙の高さは、"
                     + ("先端が変わったので測り直しが必要です）。" if moved > 0.5 else
                        "いまの筆の状態のものとして記録し直すか、測り直す必要があります）。") if setup_path else ""))
    if sim and files["tool"] != TL.tool_path(False):
        # (the simulation mirrored the real tool until now. A copy of the real tool's record is read until the real
        # tool file changes: tooling._stale_copy; a tool attached in simulation is the simulation's own)
        res["simulated_tool"] = True
        state = getattr(TL, "sim_tool_state", None)
        copy = bool(state()["copy_of_real"]) if callable(state) else True
        res["next"] += (f"; this is the SIMULATED tool file ({files['tool']}), the real one is as it was: "
                        + ("the simulated arm carries this copy until the real tool file changes" if copy else
                           "the simulated arm carries a tool of its own, not the real arm's")
                        + "; `fd --sim tool detach` returns to the real tool")
        res["say"] += ("保存先は、シミュレーション用の道具のファイルです（実機の道具のファイルは変えていません。"
                       + ("実機のファイルが変わるか " if copy else "") + "fd --sim tool detach を実行するまで、fd --sim はこの"
                       "ファイルを使います）。")


# ----------------------------------------------------------------------------- the first look
def _look_plan(arm, tool, cam, q0) -> dict:
    """The first look, planned from where the arm stands (no motion): {"pose" (aircal.look_pose), "aside": None
    or aircal.look_aside (the arm is turned aside about its base for the photo "before"), "plan": the way to
    the look pose (from the aside pose when there is one; an aircal.look_way result: where the base turns, the
    tool is lifted first), "near": the arm stands within LOOK_SAME_DEG of the
    look pose, "hides": a photo taken where it stands would not show the place of the look without the arm (it
    is near, or the camera model sees arm or tool there: aircal.look_hidden)}. SafetyError when there is no pose,
    no turn or no safe way.

    Until 2026-11-14 only an arm within LOOK_SAME_DEG of the look pose was turned aside. An arm 13 .. 25 deg
    from it (lift, elbow, wrist) stood with its tool over the place of the look all the same, and paint that the
    tool hid in the photo before passed for the tip in the look photo (aircal: silhouette)."""
    import aircal as AC
    from robot import SafetyError
    pose = AC.look_pose(arm, tool, cam)
    near = float(np.max(np.abs(np.asarray(q0, dtype=float) - pose["q"]))) <= math.radians(LOOK_SAME_DEG)
    hides = cam is not None and (near or AC.look_hidden(cam, q0, pose["q"], tool.tip_offset) is not None)
    aside = AC.look_aside(arm, tool, cam, q0, pose["q"]) if hides else None
    # (the way to the look pose is aircal.look_way: wherever it turns the base, the tool is lifted first)
    plan = AC.look_way(arm, tool, q0, pose["q"]) if aside is None else aside["back"]
    dip = AC.unknown_tool_dip(arm, tool, _waypoints(plan))
    if dip:
        raise SafetyError(f"the way to the look pose would bring a tool of unknown length lower than it is now: a tip "
                          f"{dip['length_m'] * 100:.0f} cm out {dip['direction']} would pass {dip['lowest_m'] * 1000:.0f} mm "
                          f"above the table (it is {dip['ends_m'][0] * 1000:.0f} mm up now). Nothing moved")
    return {"pose": pose, "aside": aside, "plan": plan, "near": near, "hides": hides}


def _keep_photo(path: str, img) -> str:
    """Store a look photo; _PhotoNotStored when it is not on the disk afterwards."""
    import aircal as AC
    why = AC.store_photo(path, img)
    if why:
        raise _PhotoNotStored(f"the photo was taken but could not be stored as {path} ({why})")
    return path


def _look(arm, tool, cam, photo, out_dir: str, mode: str, log) -> dict:
    """A photo of the scene without the tool at the look pose, then to the look pose, one photo: is the
    registered tip where a tip candidate is? SafetyError before anything moves when there is no pose or no way;
    MotionAborted when a way is stopped. -> the look's report ("ok": the registered tip can be relied on for
    the high set; "checked": the photo could be judged at all).

    The photo "before" is what keeps paint on the sheet from passing for the tool tip. It is taken where the
    arm stands, before anything moves; an arm that stands at the look pose, or where the camera model sees arm
    or tool at the place of the look, is turned aside for it with the torque on (aircal.look_aside) and comes
    back: what the arm hides in the photo before would be "new" in the look photo. Both ways are flown by
    aircal.fly_look_way: the base turns only with the tool lifted 10 cm straight up (asked of the measured arm),
    and the arm comes straight down onto the pose. For the same reason nothing
    that the look photo shows where the arm stood for the photo before can make the look pass
    (aircal.look_check, q_bg). Without a photo before (no camera model: nothing to compare with) the look is
    not checked."""
    import aircal as AC
    from robot import MotionAborted, SafetyError
    q0 = arm.read_q()
    lp = _look_plan(arm, tool, cam, q0)
    pose, aside, plan = lp["pose"], lp["aside"], lp["plan"]
    out = {"pose": {"fingertip": pose["fingertip"], "pitch": pose["pitch"],
                    "q_deg": [round(math.degrees(float(v)), 1) for v in pose["q"]]},
           "photo": None, "photo_before": None, "before": None, "checked": False}
    before, q_before, powered, moved = None, None, False, False
    if aside is not None:
        jaw = _power_rigid(arm)
        powered = True
        if jaw is not None:
            out["jaw"] = jaw
        flown = AC.fly_look_way(arm, aside["q"], way=aside["plan"])     # (lift, turn the base, straight down)
        moved = True
        arm._sleep(0.35)
        q_before = arm.read_q()
        before = photo()                                    # the scene without the tool at the look pose
        stood = ("it stood at the look pose" if lp["near"] else
                 "where it stood the camera could have seen it, or a tool taped to it, at the place of the look")
        out["before"] = {"taken": f"at the look pose turned aside about the base ({stood})", "turn_deg": aside["turn_deg"],
                         "pan_deg": aside["pan_deg"], "tip_in_picture": aside["tip_in_picture"], "over": _side_note(aside),
                         "way": flown["via"], "rise_at_turn_mm": _rise_mm(flown)}
        out["stands"] = "aside"
        log(f"look: {stood}; the look pose turned {aside['turn_deg']:+g} deg aside (pan {aside['pan_deg']:.0f} deg) for the "
            f"photo before: {flown['via']}")
    elif not lp["near"]:
        q_before = q0
        before = photo()                                    # ... taken before anything moves
        out["before"] = {"taken": "where the arm stood, before it moved"}
        out["stands"] = "start"
    if before is not None:
        out["photo_before"] = _keep_photo(os.path.join(out_dir, "look_before.jpg"), before)
        if cam is not None:
            # a clean background (round 3): no look where the photo before shows bristle-coloured paint near the
            # place where the registered tip will be seen. Refused here: the arm does not go on to the look pose
            ground = AC.look_background(before, cam, pose["q"], tool.tip_offset)
            out["background"] = ground
            if not ground["clean"]:
                out.update(ok=False, dirty=True,
                           why=(f"the photo of the scene without the tool ({out['photo_before']}) shows bristle-coloured paint "
                                f"where the look would judge: {ground['paint_px']} pixels within {ground['radius_px']:.0f} px of "
                                f"the place where the registered tip of '{tool.name}' will be seen (the nearest "
                                f"{ground['nearest_px']:.0f} px from it). The look cannot tell a brush from paint behind it"))
                return out
    if not powered:
        jaw = _power_rigid(arm)
        if jaw is not None:
            out["jaw"] = jaw
    try:
        flown = AC.fly_look_way(arm, pose["q"], way=plan)   # (where the base turns: lifted first, then straight down)
    except SafetyError as e:
        if not moved:
            raise
        # (the arm stands turned aside: this is no refusal "before anything moved")
        raise MotionAborted(f"the way back to the look pose could not be flown from where the arm stands ({e}); the arm "
                            "holds there") from e
    out["way"] = {"via": flown["via"], "rise_at_turn_mm": _rise_mm(flown)}
    arm._sleep(0.35)
    q = arm.read_q()
    img = photo()
    out["photo"] = _keep_photo(os.path.join(out_dir, "look.jpg"), img)
    log(f"look: photo {out['photo']}")
    if cam is None:
        out.update(ok=False, why="there is no camera model: the program cannot say where the registered tip would be seen")
        return out
    chk = AC.look_check(img, cam, q, tool.tip_offset, bg=before, q_bg=q_before)
    out.update(ok=chk["ok"], checked=chk["checked"], registered_px=chk["registered_px"], nearest_px=chk["nearest_px"],
               nearest_mm=chk["nearest_mm"], limit_px=chk["limit_px"], candidates=chk["candidates"],
               hidden_before=chk["hidden_before"], nearest_clear_px=chk["nearest_clear_px"], suggest_tip=chk["suggest_tip"])
    if not chk["checked"]:
        out["why"] = ("there is no photo of the scene without the tool at the look pose: by colour alone paint on the "
                      "sheet passes for a tool tip, so the look photo was not judged")
    elif not chk["ok"]:
        if chk.get("background") and not chk["background"]["clean"]:    # (at the pose as measured; planned: clean)
            out.update(background=chk["background"],
                       why="the photo of the scene without the tool shows bristle-coloured paint where the look judges ("
                           f"{chk['background']['paint_px']} pixels within {chk['background']['radius_px']:.0f} px of the "
                           "registered tip's place): the look cannot tell a brush from paint behind it. Cover the sheet "
                           "with a blank one, or take it away")
        elif chk["nearest_px"] is None:
            out["why"] = ("the look photo shows no tip candidate at all (the tip is hidden, outside the picture, or not "
                          "bristle-coloured)")
        elif chk["nearest_px"] <= chk["limit_px"]:      # (near, but where the arm stood for the photo before)
            out["why"] = (f"the only tip candidate near where the registered tip of '{tool.name}' would be seen "
                          f"({chk['nearest_px']:.0f} px from it) lies where the arm stood for the photo before: that photo "
                          "does not show the scene there, so it may be paint the arm hid, and was not taken for the tip")
        else:
            out["why"] = (f"the nearest tip candidate in the look photo is {chk['nearest_px']:.0f} px (about "
                          f"{chk['nearest_mm']:.0f} mm in the picture) from where the registered tip of '{tool.name}' "
                          f"would be seen (limit {chk['limit_px']:.0f} px)")
        if mode == "full":
            out["why"] += "; or the camera has moved (the look uses the camera model the rig has)"
    return out


def _look_refusal(res: dict, look: dict, tool, mode: str, again: str, paint: dict) -> dict:
    """The result of a look that did not find the registered tip (the arm holds at the look pose). again: the
    command that was typed, without --save / --force (`fd calib-air`, `fd calib-air --sets air,low`)."""
    aside = look.get("dirty") and look.get("stands") == "aside"
    over = (look.get("before") or {}).get("over")
    stay = ("The arm holds at the look pose" + (" turned aside about its base" if aside else "") + " (torque on). That pose "
            "keeps a tool of up to 10 cm in any direction (20 cm along the fingers) off the table; on the way to it the "
            "base turns only with the tool lifted 10 cm straight up, and the arm comes straight down onto the pose"
            + (" (" + over + ")" if over else "") + ". A fold is planned with the registered tip: "
            "fold only when the photo shows the tool no longer than "
            "registered (" + _fold_hint(paint, pan_deg=_pan_of(res.get("arm"))) + ")")
    if look.get("dirty"):
        # (round 3: no clean background. The photo before was taken, the look photo was not, nothing was flown)
        place = (look.get("background") or {}).get("place_px")
        res.update(ok=False, stage="look", reason=look["why"] + ". No look photo was taken and the capture was not started",
                   next=f"cover the sheet with a blank one, or take it away, so that nothing bristle-coloured (red, orange) "
                        f"lies behind the place of the look (pixel {place} of {look['photo_before']}), then `{again}` again. "
                        + (stay if aside else "The arm stands where it stood: nothing moved"),
                   say="確認の場所の背景に、筆先と同じ色（赤・橙）の絵の具が写っています。この背景では、筆先と絵の具を見分けられないので、"
                       "確認の写真を撮っていません（撮影も始めていません）。絵を白い紙で覆うか、紙をどけてから、もう一度 fd calib-air を"
                       "実行してください。" + ("アームは、確認の姿勢から根元を横へ向けた所で止めてあります（トルク ON）。" if aside else
                                          "アームは動かしていません。"))
        return res
    if not look.get("checked"):
        res.update(ok=False, stage="look", reason=look["why"] + ". The capture was not started",
                   next=f"look at {look['photo']}: when the tool tip is where the registration of '{tool.name}' says "
                        f"(`fd tool status`), `{again} --no-look` flies the first set on that word. {stay}",
                   say="登録された先端の位置を写真で確かめられませんでした（カメラモデルがない、または比べる写真がありません）。"
                       "撮影は始めていません。アームは確認の姿勢で止めてあります（トルク ON）。写真を見て確かめてから "
                       "--no-look で進めてください。")
        return res
    s = look.get("suggest_tip")
    hint = ("a better guess is registered with fd tool set --tip=X,Y,Z (metres, gripper frame; with \"=\": the numbers may "
            "be negative)"
            + (f": `fd tool set --tip={s[0]:.4f},{s[1]:.4f},{s[2]:.4f}` puts it on the line of sight of that candidate. "
               "How far along that line the tip is, one photo does not say, and the next look passes anywhere on it: "
               "measure the tool with a ruler and type what you measured. Where the real tip is further out than the "
               "registration, it is lower than the plan while the first set is flown (simulated 2026-11-14: registered as "
               "printed here, 9 to 12 cm from the real tip, the tip was 4 to 5 cm lower than planned at the photo poses)"
               if s else
               " (`fd tool status` shows the registered one)"))
    # a tool that IS where it is registered, and still no candidate there: the camera model is what is wrong
    moved = ("`fd calib-air --sets air,low --no-look` (the camera has moved: solve it too)" if mode == "tip" else
             f"`{again} --no-look` (the camera has moved, which is what the full solve is for)")
    res.update(ok=False, stage="look", reason=look["why"] + ". The capture was not started",
               next=f"look at {look['photo']}. The tool tip is not where it is registered: {hint}, then `{again}` again (it "
                    f"turns the arm aside for a new photo by itself). The photo shows the tip where it is registered "
                    f"(relative to the gripper): {moved}. Only --no-look skips the look; --force does not. {stay}",
               say=("最初の確認写真で、登録されているツール先端の位置に筆先が見つかりませんでした"
                    + (f"（いちばん近い候補まで 約 {look['nearest_px']:.0f} px・約 {look['nearest_mm']:.0f} mm）"
                       if look.get("nearest_px") is not None else "（候補がありません）")
                    + "。撮影は始めていません。アームは確認の姿勢で止めてあります（トルク ON）。写真を見て、先端の登録が"
                      "違っていれば fd tool set で直し（next に出る値は写真の面の中だけを合わせたものです。奥行きは定規で"
                      "測って確かめてください）、登録が合っていればカメラが動いています"
                    + ("（fd calib-air --sets air,low --no-look）。" if mode == "tip" else "（同じコマンドに --no-look を付けて進めます）。")
                    + "登録より長い道具が付いているあいだは、収納の動きをさせないでください。"))
    return res


# ----------------------------------------------------------------------------- --dry
def _heights_note(tool) -> str:
    rec = _tip_record(tool)
    if rec:
        return f"heights are of the tool tip as measured on {rec.get('at')} ({rec.get('by')}, rms {rec.get('rms_px')} px)"
    return ("heights are of the MODELLED tool tip: the registered tip has NOT been measured, the real tip may be "
            "lower by as much as the registration is wrong")


def _plan_sets(arm, tool, mode: str, names, q_from) -> tuple:
    """([{"name", "poses", "plannable", "refused", "lowest_tip_mm", "low"}], [the plannable poses per set], q_end)"""
    import aircal as AC
    q = np.asarray(q_from, dtype=float)
    reports, kept = [], []
    for name in names:
        spec = AC.POSE_SETS[mode][name]
        poses = AC.plan_set(arm, tool, spec)
        chain = AC.plannable(arm, tool, poses, q)
        q = chain["q_end"]
        kept.append(chain["poses"])
        reports.append({"name": name, "poses": len(poses), "plannable": len(chain["poses"]),
                        "refused": [f"{r['xyz']}: {r['why'][:120]}" for r in chain["refused"][:3]],
                        "lowest_tip_mm": None if chain["lowest_m"] is None else round(chain["lowest_m"] * 1000.0, 1),
                        "low": AC.is_low(spec)})
    return reports, kept, q


def _fold_refusal(res: dict, e, a, paint: dict) -> dict:
    """--end rest cannot be done as things are: the refusal, before anything moves."""
    stage = getattr(e, "stage", None) or ("setup" if paint.get("error") else "end")
    nxt = getattr(e, "next", "") or ""
    cmd = "fd --sim calib-air" if getattr(a, "sim", False) else "fd calib-air"
    # (a hint works as it is typed: for a plan it is the plan. Until round 3 a simulated --dry was answered with
    # `fd --sim calib-air --end stay`, which is refused: the simulated arm has no camera)
    stay = (f"`{cmd} --dry --end stay` plans the run that leaves the arm at its last pose with the torque on instead of "
            "folding it" if getattr(a, "dry", False) else
            f"`{cmd} --end stay` leaves the arm at its last pose with the torque on instead of folding it")
    res.update(ok=False, stage=stage, reason=f"the arm could not be folded away afterwards: {e}. Nothing moved",
               next=(("`fd paint status` says what is wrong with the painting set-up; " if paint.get("error") else
                      (nxt + "; " if nxt else ""))
                     + stay),
               say="終わったあとにアームを安全に畳める姿勢がないため、始めていません（アームは動かしていません）。"
                   "--end stay を付けると、最後の姿勢でトルク ON のまま止まります。")
    return res


def _dry(arm, tool, cam, mode: str, names, a, paint: dict) -> dict:
    """--dry: what the run would do, from where the arm stands: the fold at the end, the first look, the poses of
    every set and which of them the run would fly. Nothing moves, nothing is written to the servo bus."""
    import aircal as AC
    from robot import SafetyError
    res = {"ok": True, "stage": "planned", "mode": MODE_NAME[mode], "tool": tool.name,
           "registered_tip_mm": _mm(tool.tip_offset), "tip_measured": _tip_record(tool) is not None,
           "heights": _heights_note(tool)}
    if a.end == "rest":
        try:
            res["end"] = {"fold": _fold_plan(arm, tool, paint), "torque": "on",
                          "note": "planned here with the registered tip; the run plans the fold again with the tip its photos "
                                  "give (when their fit passes the gates). It does not fold when no fold is safe with that "
                                  "tip, nor when no tip is confirmed (no fit passes the gates and the registered tip "
                                  + ("is on record as measured: then only when the photos speak against it"
                                     if _tip_record(tool) is not None else "was never measured")
                                  + "): the arm then holds where the last photo was taken, torque on"}
        except SafetyError as e:
            return _fold_refusal(res, e, a, paint)
    else:
        res["end"] = {"fold": None, "torque": "on", "note": "--end stay: the arm stays where the last photo is taken"}
    # the jaw (reads only), as the run asks before it moves: a plan for a run that would be refused is no plan
    try:
        import tooling as TL
        in_place = getattr(TL, "jaw_in_place", None)
        if callable(in_place):
            in_place(arm, tool)
        elsewhere = _jaw_elsewhere(arm, tool)
    except SafetyError as e:
        res.update(ok=False, stage="tool", reason=f"the jaw is not where the tool file has it: {e}. Nothing moved",
                   next="`fd tool jaw status` shows the jaw and what the tool file records. A tool on the outer face of the "
                        "moving jaw: `fd tool jaw mount moving_outer`, then `fd tool jaw close` (closes the jaw on the fixed "
                        "jaw and holds it), then the same command again",
                   say="ツールを留めたあごが、ツールのファイルの記録と違う位置にあります。fd tool jaw status で確かめ、"
                       "fd tool jaw close のあとでやり直してください。アームは動かしていません。")
        return res
    if elsewhere:
        res["notes"] = [elsewhere + ": a run with --save would be refused before it moves (what it measures could not be "
                        "stored into the tool file as it is)"]
    q = arm.read_q()
    if a.no_look:
        res["look"] = {"skipped": "--no-look"}
    else:
        try:
            lp = _look_plan(arm, tool, cam, q)
            pose, plan = lp["pose"], lp["plan"]
            res["look"] = {"fingertip": pose["fingertip"], "pitch": pose["pitch"],
                           "q_deg": [round(math.degrees(float(v)), 1) for v in pose["q"]], "via": plan["via"],
                           "tip_height_mm": round(pose["tip_height_m"] * 1000.0, 1), "registered_px": pose["tip_px"],
                           "way_safe_for_unknown_length": True,
                           "turn_over": _side_note(lp["aside"]),
                           "photo_before": ("none: without a camera model the look cannot be checked" if cam is None else
                                            "where the arm stands, before it moves" if lp["aside"] is None else
                                            f"at the look pose turned {lp['aside']['turn_deg']:+g} deg aside about its base "
                                            + ("(it stands at the look pose)" if lp["near"] else
                                               "(where it stands the camera could see it, or a tool taped to it, at the "
                                               "place of the look)")),
                           "background": (None if cam is None else
                                          f"the photo before must show nothing bristle-coloured (red, orange) within "
                                          f"{AC.LOOK_CLEAN_PX:.0f} px of pixel {pose['tip_px']}: a picture there is covered "
                                          "with a blank sheet, or taken away, before the run")}
            q = pose["q"]
        except SafetyError as e:
            res.update(ok=False, stage="look", reason=f"no first look is possible: {e}",
                       next=_fold_hint(paint, a.sim, keep=True, pan_deg=math.degrees(float(q[0])))
                            + " (the look is planned from where the arm stands); or check "
                            "the registered tip by hand (`fd snap --exposure paper`) and use --no-look",
                       say="最初の確認の姿勢へ安全に動けないため、計画を止めました（アームは動かしていません）。")
            return res
    reports, _kept, _q = _plan_sets(arm, tool, mode, names, q)
    res["sets"] = reports
    measured = res["tip_measured"]
    for k, r in enumerate(reports):
        r["flown"] = bool(r["plannable"])
        if r["low"] and k == 0 and not measured:
            # (the run's rule: a set 3 .. 6 cm over the table is never flown on a tip that is only registered)
            r["flown"] = False
            r["note"] = ("NOT flown in the run: the registered tip has not been measured, and no set before it gives "
                         "one (fly a high set first)")
        elif r["low"] and k > 0:
            r["note"] = "planned here with the registered tip; in the run it is planned again with the tip the first set gives"
    total, ok = sum(r["poses"] for r in reports), sum(r["plannable"] for r in reports if r["flown"])
    lows = [r["lowest_tip_mm"] for r in reports if r["lowest_tip_mm"] is not None and r["flown"]]
    res.update(poses=total, plannable=ok, lowest_tip_mm=min(lows) if lows else None,
               minutes=round((ok * SECONDS_PER_POSE + 45.0) / 60.0, 1))
    if not ok:
        unmeasured_low = any(r["plannable"] and not r["flown"] for r in reports)
        res.update(ok=False, stage="plan",
                   reason=("the set flies 3 .. 6 cm over the table and the registered tip has not been measured: no "
                           "option flies it with a tip that is only registered" if unmeasured_low else
                           "none of the calibration poses can be flown with this tool"),
                   next=("`fd calib-air --sets air,low --dry`: the high set first, the low set is then planned with the tip "
                         "it gives" if unmeasured_low else
                         "`fd tool status`: check the registered tip and the tool's joint limits"),
                   say=("先端をまだ測っていないので、低い姿勢のセットだけを飛ぶことはできません（高いセットを先に）。" if unmeasured_low
                        else "この道具では、校正の姿勢を 1 つも取れません。"))
        return res
    res["next"] = (f"`fd calib-air{'' if mode == 'tip' else ' --sets ' + ','.join(names)}` flies them (about "
                   f"{res['minutes']} minutes; give the Bash call a timeout of 600000 ms)")
    res["say"] = (f"空中校正の計画です（{MODE_NAME[mode]}）: {total} 姿勢のうち {ok} 姿勢を取れます。"
                  f"ツール先端（モデル上）のいちばん低い高さは机から {res['lowest_tip_mm']} mm です。"
                  + ("先端はまだ測っていないので、実際の高さは登録値のずれの分だけ違います。" if not res["tip_measured"] else "")
                  + "アームは動かしていません。")
    return res


# ----------------------------------------------------------------------------- the run on the arm
def _run(a, mode: str, names, arm, cam, photo, out_dir: str, log, paint: dict | None = None) -> dict:
    """Look, capture set by set, solve, fold away. Stores nothing of the rig (main does, with --save).

    What decides whether a pose is flown is the arm and the photos, never an option for storing: --force is not
    looked at here. A look that did not pass ends the command; a low set is flown only with a tip that was
    measured before or solved in this run from a set that passed the gates and lies within LOW_MAX_JUMP_M of
    the registration.

    Whatever moves after a fit of this run's photos has passed the gates is planned with the tip THAT fit gave
    (`cur`), wherever it lies: the next set (when the jump rule lets it fly) and the fold at the end. Until
    2026-11-14 a tip further than LOW_MAX_JUMP_M from the registration was reported and the arm then folded
    with the REGISTERED one: a tool taped across the fingers (tip really at (85, 0, 45) mm), registered on its
    line of sight as the refused look had suggested (93 mm off), was solved to 0.2 mm and then folded 17 mm
    INTO the table, result ok (simulated; verifier's case E). When no fold is safe with the solved tip the arm
    is not folded: it holds where the last photo was taken, torque on, and the result says so (stage "end").
    A later command folds with the tip the TOOL FILE has: after a solved tip that is far from it, every hint
    of this run names the storing first (fold_hint).

    Round 3 (2026-11-14): the fold needs a CONFIRMED tip (unconfirmed, below): `cur` after a fit that passed the
    gates, or a registered tip that is on record as measured and that this run's photos do not contradict. An
    arm whose tip nothing confirms is not folded at all: state["held"], stage "end", `next` from _hold_next."""
    import aircal as AC
    import tooling as TL
    from robot import MotionAborted, SafetyError
    paint = _painting() if paint is None else paint
    tool = arm.tool
    registered = [float(v) for v in tool.tip_offset]
    sets = "" if mode == "tip" else " --sets " + ",".join(names)
    again = "fd calib-air" + sets
    so_far = f"the photos taken so far: `{_from_words(out_dir, *sets.split())}`"
    stores = _from_words(out_dir, *([] if mode == "tip" else ["--sets", "air,low"]), "--save")     # (as _verdict_text spells it)
    res = {"ok": True, "stage": "done", "mode": MODE_NAME[mode], "tool": tool.name, "dir": out_dir, "sets": [],
           "heights": _heights_note(tool), "saved": False}
    state = {"stage": "look", "at": "where it was", "ja": "動かす前の姿勢", "touched": False}
    records, fit, verdict, cur = [], None, None, tool
    notes, no_fit = [], None

    def solved_far() -> float | None:
        """How far (m) the tip the arm is planned with since a fit of this run passed the gates lies from the
        tool file's, when that is more than LOW_MAX_JUMP_M; else None (no such fit, or the two are near)."""
        if cur is tool:
            return None
        d = float(np.linalg.norm(np.asarray(cur.tip_offset, dtype=float) - np.asarray(registered)))
        return d if d > LOW_MAX_JUMP_M else None

    def unconfirmed() -> str | None:
        """Why the tip a fold would be planned with is NOT confirmed (None: it is). Round 3, T1: the arm is folded
        only with a confirmed tip. Confirmed is
        - the tip of a fit of this run's photos that passed the gates (`cur`: the fold is planned with it), or
        - the registered tip, when the tool file has it on record as measured AND the photos of this run do not
          speak against it: their fit lies within LOW_MAX_JUMP_M of it (whatever its gates), or no photo was
          taken at all.
        Everything else is a guess, and a fold planned with a guess ended under the table (simulated, 2026-11-14:
        a tool across the fingers, registered on its line of sight 93 mm from its tip: the first set's fit failed
        the 60 % gate, the arm was folded with the registered tip, the real tip ended 17 mm under the table and
        the result was ok): a registered tip that was never measured, whatever the look said (one photo does not
        see along its line of sight) and whatever the operator said (--no-look); a recorded tip whose photos give
        no fit, or a fit that lies elsewhere."""
        if cur is not tool:
            return None
        measured = _tip_record(tool) is not None
        head = "the tool file has the tip on record as measured, but " if measured else "the registered tip was never measured, and "
        if no_fit is not None:
            return head + f"the photos of this run give no fit ({no_fit})"
        if fit is not None:
            jump = float(np.linalg.norm(np.asarray(fit["tip_offset"], dtype=float) - np.asarray(registered)))
            if measured and jump <= LOW_MAX_JUMP_M:
                return None
            return head + (f"the fit of this run's photos, which lies {jump * 1000:.0f} mm from it, does not pass the gates ("
                           + "; ".join((verdict or {}).get("problems") or ["see \"gates\""]) + ")")
        if measured:
            return None                 # (no photo of this run: nothing speaks against the record)
        return head + "no photo of this run shows it (" + ("; ".join(notes) or "no pose was flown") + ")"

    def fold_hint(keep: bool = False) -> str:
        """_fold_hint, for a result of this run. The command it names plans with the tip the TOOL FILE has: when
        this run solved a tip far from that one, storing it comes first."""
        far = solved_far()
        guess = ("LOOK at the arm and at the tool first (the registered tip was never measured, and what folds the arm "
                 "plans with it): " if cur is tool and _tip_record(tool) is None else "")
        return (guess + ("" if far is None else _store_first(stores, far))
                + _fold_hint(paint, keep=keep, pan_deg=_pan_of(res.get("arm"))))

    def arm_is(at: str, ja: str) -> None:
        """Where the arm is, into the result ("arm") and, in Japanese, for `say`. An arm that cannot be read just
        now (an interrupt in the middle of a bus transaction) must not cost the operator the rest of the result."""
        try:
            res["arm"] = _where(arm, cur, at)
            state["arm_ja"] = (f"アームは{ja}（トルク {res['arm']['torque'].upper()}、ツール先端はモデル上で机から "
                               f"{res['arm']['tip_height_mm']} mm）。")
        except Exception as e:
            res["arm"] = {"at": at, "unread": f"{type(e).__name__}: {e}", "next": "`fd state` reads the arm"}
            state["arm_ja"] = f"アームは{ja}（状態は読めませんでした。fd state で確かめてください）。"
        if solved_far() is not None:        # (the arm is planned with a tip the tool file does not have: see fold_hint)
            state["arm_ja"] += SAY_STORE_FIRST
        elif cur is not tool and _tip_record(tool) is None:
            state["arm_ja"] += SAY_STORE_UNMEASURED

    def solve_now():
        images, good, missing = _load_images(records)
        if missing:
            notes.append(f"{len(missing)} photos could not be read back: {missing[:3]}")
        f = _solve(mode, images, good, cam, cur.tip_offset)
        return f, good, judge(mode, f, cam)

    if a.end == "rest":             # where the arm will be folded is settled before it leaves where it is
        try:
            res["end"] = {"fold": _fold_plan(arm, tool, paint)}
        except SafetyError as e:
            arm_is("where it was (nothing moved)", "動かしていません")
            return _fold_refusal(res, e, a, paint)
    # the jaw, before anything moves (reads only). tooling.jaw_in_place: the jaw does not stand where the tool
    # file has it (since the evening of 2026-11-13 for every tool that records a jaw position; before that only
    # for a jaw held by a squeeze): the tool is not where its registered tip is, and no pose planned with that tip
    # is flown. Where that check does not reach (an older tooling), a capture that is asked to --save is refused
    # all the same: what it measured could not be stored.
    jaw_note, refused = None, False
    try:
        in_place = getattr(TL, "jaw_in_place", None)
        if callable(in_place):
            in_place(arm, tool)
        jaw_note = _jaw_elsewhere(arm, tool)
    except SafetyError as e:
        jaw_note, refused = f"the jaw is not where the tool file has it: {e}", True
    except Exception:               # (a bus that does not answer fails below, where the result says where the arm is)
        pass
    if jaw_note and (refused or getattr(a, "save", False)):
        arm_is("where it was (nothing moved)", "動かしていません")
        res.update(ok=False, stage="tool",
                   reason=(jaw_note if refused else jaw_note + ": a tip measured now is the tip of a jaw state the tool "
                           "file does not describe, and --save could not store it") + ". Nothing moved",
                   next=("`fd tool jaw status` shows the jaw and what the tool file records. A tool on the outer face of the "
                         "moving jaw: `fd tool jaw mount moving_outer`, then `fd tool jaw close` (closes the jaw on the fixed "
                         f"jaw and holds it), then `{again}" + (" --save" if getattr(a, "save", False) else "") + "` again"
                         + ("" if refused else f"; without --save the capture runs and only reports (`{again}`)")),
                   say=("ツールを留めたあごが、ツールのファイルの記録と違う位置にあります（このまま測ると、記録と違う状態の先端を"
                        "測ることになります）。fd tool jaw status で確かめ、fd tool jaw close のあとでやり直してください。"
                        "アームは動かしていません。"))
        return res
    if jaw_note:
        notes.append(jaw_note + ": what this run measures is the tip with the jaw as it stands, and cannot be stored into "
                     "the tool file as it is")
    try:
        try:
            os.makedirs(out_dir, exist_ok=True)
            if a.no_look:
                res["look"] = {"skipped": "--no-look"}
            else:
                state["touched"] = True
                look = _look(arm, tool, cam, photo, out_dir, mode, log)
                state.update(at="the look pose", ja="確認の姿勢")
                res["look"] = look
                if look.get("dirty"):   # (no clean background: the arm did not go on to the look pose)
                    if look.get("stands") == "aside":
                        state.update(at="the look pose turned aside about the base", ja="確認の姿勢から根元を横へ向けた所")
                        arm_is("the look pose turned aside about the base (no look photo was taken, the capture was not started)",
                               "確認の姿勢から根元を横へ向けた所で止めてあります")
                    else:
                        state.update(at="where it was", ja="動かす前の姿勢")
                        arm_is("where it was (nothing moved)", "動かしていません")
                    return _look_refusal(res, look, tool, mode, again, paint)
                if not look["ok"]:      # (no option goes on from here: --no-look is the operator's own look)
                    arm_is("the look pose (the capture was not started)", "確認の姿勢で止めてあります")
                    return _look_refusal(res, look, tool, mode, again, paint)
            state["stage"] = "capture"
            good = []
            for k, name in enumerate(names):
                spec = AC.POSE_SETS[mode][name]
                if AC.is_low(spec) and fit is None and _tip_record(tool) is None:
                    notes.append(f"the set '{name}' flies 3 .. 6 cm over the table and was not flown: the registered tip has "
                                 "not been measured and no set before it gave one (no option flies it so: a high set first)")
                    continue
                arm.tool = cur
                poses = AC.plan_set(arm, cur, spec)
                chain = AC.plannable(arm, cur, poses, arm.read_q())
                rep = {"name": name, "poses": len(poses), "plannable": len(chain["poses"]), "photos": 0,
                       "lowest_tip_mm": None if chain["lowest_m"] is None else round(chain["lowest_m"] * 1000.0, 1),
                       "planned_with_tip_mm": _mm(cur.tip_offset)}
                res["sets"].append(rep)
                if not chain["poses"]:
                    notes.append(f"none of the {len(poses)} poses of the set '{name}' can be flown")
                    continue
                t_set = time.time()
                state["touched"] = True
                jaw = _power_rigid(arm)
                if jaw is not None:
                    rep["jaw"] = jaw
                cap = AC.capture(arm, chain["poses"], photo, out_dir, log=log, prefix=name)
                rep.update(photos=len(cap["records"]), skipped=len(cap["skipped"]), seconds=round(time.time() - t_set, 1),
                           records=AC.save_records(out_dir, cur, cap, f"records_{k}_{name}.json"))
                records += cap["records"]
                state.update(at=f"pose {len(cap['records'])} of the set '{name}'", ja=f"「{name}」セットの {len(cap['records'])} 枚目の姿勢")
                if cap.get("interrupted"):
                    raise KeyboardInterrupt
                if cap.get("photo_error"):
                    raise _PhotoNotStored(cap["photo_error"])
                if cap["stopped"]:
                    raise MotionAborted(cap["stopped"])
                more = k + 1 < len(names)
                if more or (cur is tool and a.end == "rest"):
                    # What moves next is planned with what these photos say, and only when they say it well: the
                    # next set, and the fold at the end. (A last set is solved here, before the fold, only when
                    # no set of this run has given a tip yet: `--sets air`.)
                    try:
                        fit, good, verdict = solve_now()
                    except ValueError as e:
                        no_fit = e
                        if more:
                            notes.append(f"the photos of '{name}' give no fit ({e}): the next set was not flown")
                        break
                    if verdict["problems"]:
                        if more:
                            notes.append(f"the fit of '{name}' does not pass the gates (" + "; ".join(verdict["problems"])
                                         + f"): the set '{names[k + 1]}' was not flown")
                        break
                    rep["tip_mm"] = _mm(fit["tip_offset"])
                    log(f"set {name}: tip {rep['tip_mm']} mm from {fit['used']} of {fit['photos']} photos")
                    # From here on the arm is planned with this tip, however far it is from the registered one: the
                    # jump rule below keeps the LOW SET from being flown, it does not make the registration right
                    cur = _with_tip(tool, fit["tip_offset"])
                    jump = float(np.linalg.norm(np.asarray(fit["tip_offset"]) - np.asarray(registered)))
                    if more and jump > LOW_MAX_JUMP_M and AC.is_low(AC.POSE_SETS[mode][names[k + 1]]):
                        notes.append(f"the tip solved from '{name}' is {jump * 1000:.0f} mm from the registered one (more than "
                                     f"{LOW_MAX_JUMP_M * 1000:.0f} mm): the set '{names[k + 1]}', 3 .. 6 cm over the table, was not "
                                     "flown with a tip that no look has confirmed (no option flies it so). Store or register "
                                     "this tip and run again")
                        break
                else:
                    fit = None                              # (solved below, on every photo)
            # fold away before the last solve: the arm must not hold a wet brush over the sheet meanwhile. The fold
            # is planned with `cur`: the tip the photos gave when a fit of them passed the gates, else the registered
            # one, and that one only when it is confirmed (unconfirmed)
            state["stage"] = "end"
            arm.tool = cur
            why_not = unconfirmed() if state["touched"] and a.end == "rest" else None
            if not state["touched"]:
                # no look, no pose that could be flown: this command neither powered nor moved the arm, and it does
                # not start now by folding it (with --no-look and nothing measured, on the registered tip alone)
                arm_is("where it was (this command did not move it)", "動かしていません")
            elif why_not:
                # Nothing confirms the tip a fold would be planned with: the arm is NOT folded (round 3, T1). It holds
                # where the last photo was taken; the measurement, when there is one, is still reported below.
                state["held"] = True
                arm_is(state["at"] + " (not folded: no tip is confirmed)", f"{state['ja']}で保持しています")
                res["end"] = {"fold": None, "not_folded": why_not, "unconfirmed": True,
                              "registered_tip_mm": _mm(registered), "tip_measured": _tip_record(tool) is not None}
                res.update(ok=False, stage="end",
                           reason=f"the arm was not folded away: nothing confirms where the tip of tool '{tool.name}' is "
                                  f"({why_not}). It holds where the last photo was taken, torque on",
                           next=_hold_next(paint, again, _pan_of(res.get("arm")), _tip_record(tool) is not None))
            elif a.end == "rest":
                r, fold = _fold(arm, paint)     # (SafetyError: no fold is safe with this tip. The arm holds; see below)
                res["end"] = {"fold": fold}
                state["folded"] = bool(r.get("ok")) and r.get("moved") is not False
                if r.get("ok") and r.get("moved") is False:
                    arm_is("where it lay folded (it was not moved)", "動かしていません（畳まれたままです）")
                elif r.get("ok") and fold["by"] == "paint":
                    arm_is(f"the rest pose `fd paint rest` chooses (folded at pan {fold['pan_deg']:g}, the brush off the canvas)",
                           "絵にかからない収納姿勢にあります")
                elif r.get("ok"):
                    arm_is("the tool's rest pose (folded)", "収納姿勢にあります")
                else:
                    arm_is("NOT in the rest pose", "収納姿勢に入っていません")
                    res.update(ok=False, stage="end", reason=str(r.get("reason")),
                               next="the torque is still ON and the arm holds; `fd state`, then " + fold_hint())
            else:
                arm_is(state["at"] + " (--end stay)", f"{state['ja']}のままです")
            res["arm"]["tip"] = "modelled with the tip solved in this run" if cur is not tool else "modelled with the registered tip"
            if state["touched"] and res["arm"].get("torque") == "on":
                pan_now = _pan_of(res["arm"])
                unmeasured = _tip_record(tool) is None
                if state.get("held"):
                    res["arm"]["next"] = res["next"]                    # (look first: the same words)
                else:
                    fold_done = res["end"]["fold"] if a.end == "rest" else None
                    plain = _torque_on_next(paint, fold_done, pan_deg=pan_now)
                    res["arm"]["next"] = plain
                    if cur is not tool and (solved_far() is not None or unmeasured):
                        # the arm stands (or is folded) as the SOLVED tip allows, and the tool file has another one,
                        # or one that is not on record as measured: whatever folds the arm or switches it off next
                        # would first move it again (round 2: with the registered tip, into the table; round 3, T4:
                        # `fd paint rest` folds an unmeasured tool again, higher, and switches off there)
                        res["arm_next_stored"] = plain                  # (for main: once --save has stored the tip)
                        res["arm"]["next"] = _torque_on_next(paint, fold_done, _store_first(stores, solved_far()), pan_now)
                    elif cur is tool and unmeasured:
                        # (--end stay.) No fit of this run has passed the gates so far (the last set is solved below)
                        # and the registered tip was never measured: nothing confirms it yet
                        state["stay_unconfirmed"] = (pan_now, plain)
                        res["arm"]["next"] = _hold_next(paint, again, pan_now, False)
        except SafetyError as e:
            arm_is(state["at"], f"{state['ja']}にあります")
            if state["stage"] == "end":
                # the fold was refused (no folded pose for this tip, or no safe way into one): the arm is not folded
                with_tip = "the tip solved in this run" if cur is not tool else "the registered tip"
                res["end"] = {"fold": None, "not_folded": str(e), "planned_with_tip_mm": _mm(cur.tip_offset)}
                res["arm"]["tip"] = "modelled with " + with_tip
                reason = (f"the arm was not folded away: with {with_tip} {_mm(cur.tip_offset)} mm no fold is safe ({e}). It "
                          "holds where it is, torque on")
            else:
                reason = f"SafetyError: {e}"
            res.update(ok=False, stage=state["stage"], reason=reason,
                       next=("nothing moved for the look. For this registration no photo of the scene without the tool can "
                             "be taken: no turn of the look pose keeps the arm and what a tool can reach clear of the place "
                             "of the look and can be flown back from with the tool lifted (the reason says which), from any "
                             "start pose, so folding the arm first does not help. Either look "
                             "yourself (`fd snap --exposure paper` shows gripper and tool; `fd tool status` what is "
                             "registered) and then run the command with --no-look, or measure the tool with a ruler and "
                             "register it anew"
                             if state["stage"] == "look" and "cannot be turned aside" in str(e) else
                             "nothing moved for the look. It is planned from where the arm stands: " + fold_hint(keep=True)
                             + ", then the same command (fold only a tool that is no longer than registered: a fold is planned "
                               "with the registered tip); or check the registered tip by hand (`fd snap --exposure paper`) "
                               "and use --no-look"
                             if state["stage"] == "look" else
                             "the arm holds (torque on) and was not folded away: `fd state`; " + fold_hint()
                             + ". Run that command with --dry first: it says whether, and how, the arm can be folded with "
                               "the tip the tool file has"
                             if state["stage"] == "end" else
                             "nothing moved for the refused step; `fd tool status` / `fd state`, then `fd calib-air --dry`"),
                       say=f"空中校正を止めました（{state['stage']}: 安全チェックで拒否）。" + state["arm_ja"])
        except _JawNotHeld as e:
            arm_is(state["at"], f"{state['ja']}にあります")
            res.update(ok=False, stage="tool", reason=f"the tool is not held as its file says: {e}",
                       next=f"`fd tool jaw close` closes the jaw against the fixed jaw and holds it, then `{again}` again. "
                            "The arm did not move for this step" + (f"; {so_far}" if records else ""),
                       say="ツールを留めたあごが、登録どおりに閉じて保持されていません（このまま測ると、描くときと違う先端の"
                           "位置を測ってしまいます）。fd tool jaw close のあとでやり直してください。" + state["arm_ja"])
            state["failed"] = True
        except _PhotoNotStored as e:
            arm_is("where the photo could not be stored (after " + state["at"] + "); it holds there",
                   f"{state['ja']}のあと、写真を保存できなかった場所で保持しています")
            res.update(ok=False, stage=state["stage"], reason=str(e), error="PhotoNotStored",
                       next=f"is the disk full (`df -h`)? Make room, then `{again}` again. The arm holds (torque on): "
                            + fold_hint() + (f". {so_far[0].upper() + so_far[1:]}" if records else ""),
                       say="写真を保存できなかったので、撮影を止めました（ディスクの空きを確かめてください）。" + state["arm_ja"])
            state["failed"] = True
        except MotionAborted as e:
            arm_is("where the motion stopped (after " + state["at"] + "); it holds there", f"{state['ja']}を出たあと、止まった場所で保持しています")
            res.update(ok=False, stage=state["stage"], reason=f"a motion was stopped: {e}",
                       next="`fd state`; when the way is clear, " + fold_hint() + f". {so_far[0].upper() + so_far[1:]}",
                       say="動作が途中で止まりました。" + state["arm_ja"])
        except KeyboardInterrupt:
            arm_is("where it was interrupted (after " + state["at"] + "); it holds there", f"{state['ja']}のあと、中断された場所で保持しています")
            res.update(ok=False, stage=state["stage"], reason="interrupted", interrupted=True,
                       next="the arm holds where it was interrupted (torque on): " + fold_hint() + f". {so_far[0].upper() + so_far[1:]}",
                       say="空中校正が途中で中断されました。" + state["arm_ja"])
        except Exception as e:          # the camera failed, a file could not be written ...: still say where the arm is
            arm_is("where it was when the error came (after " + state["at"] + "); it holds there",
                   f"{state['ja']}のあと、その場で保持しています")
            res.update(ok=False, stage=state["stage"], reason=f"{type(e).__name__}: {e}", error=type(e).__name__,
                       next="`fd state`; " + fold_hint() + f". {so_far[0].upper() + so_far[1:]}",
                       say=f"空中校正の途中でエラーが起きました（{type(e).__name__}）。" + state["arm_ja"])
            state["failed"] = True
        if notes:
            res["notes"] = notes
        if not records or res.get("interrupted") or state.get("failed") or res["stage"] in ("look", "capture"):
            if res["ok"]:
                res.update(ok=False, stage="capture", reason="no photo was taken: " + "; ".join(notes or ["no pose could be flown"]),
                           next=f"`{again} --dry` shows what can be flown",
                           say="校正の姿勢を取れなかったため、写真を撮っていません。")
            return res
        # the measurement: every photo of this run
        if fit is None or len(good) != len(records):
            try:
                if no_fit is not None:                      # (the same photos as when the solve failed)
                    raise no_fit
                fit, good, verdict = solve_now()
            except ValueError as e:
                moved_hint = " or the camera has moved (`fd calib-air --sets air,low`)" if mode == "tip" else ""
                photos = f"look at the photos in {out_dir}: is the tool tip in the picture and bristle-coloured?" + moved_hint
                if state.get("held"):
                    # (not folded, stage "end": its reason says that the photos give no fit, its next what to do first)
                    res["next"] += ". The photos: " + photos
                    res["say"] = "写真から先端の位置を求められませんでした。" + SAY_HOLD + state.get("arm_ja", "")
                    return res
                res.update(ok=False, stage="solve", reason=f"the photos give no fit: {e}", next=photos,
                           say="写真から先端の位置を求められませんでした。" + state.get("arm_ja", ""))
                if state.get("stay_unconfirmed"):       # (--end stay: the arm's own hint says to look first)
                    res["say"] += "登録された先端は未測定です。次に動かす前に、アームと道具を目で確かめてください。"
                return res
            except KeyboardInterrupt:
                res.update(ok=False, stage="solve", reason="interrupted while the photos were being solved", interrupted=True,
                           next=f"`{_from_words(out_dir, *sets.split())}` solves the photos again (no motion)",
                           say="写真の解析中に中断されました。撮影は終わっています。" + state.get("arm_ja", ""))
                return res
        res["fit"], res["fit_records"], res["verdict"] = fit, good, verdict         # (for main: --save; removed there)
        _report(res, mode, fit, good, registered, cam, verdict)
        if res["ok"] and state.get("stay_unconfirmed") and not verdict["problems"]:
            # (--end stay, an unmeasured registration, and the photos of this run give a fit that passes the gates:
            # that tip can be stored, and storing it comes before anything that folds the arm)
            pan_now, plain = state.pop("stay_unconfirmed")
            jump = float(np.linalg.norm(np.asarray(fit["tip_offset"], dtype=float) - np.asarray(registered)))
            res["arm_next_stored"] = plain
            res["arm"]["next"] = _torque_on_next(paint, None, _store_first(stores, jump if jump > LOW_MAX_JUMP_M else None),
                                                 pan_now)
            state["arm_ja"] = state.get("arm_ja", "") + SAY_STORE_UNMEASURED
        if res["ok"]:
            _verdict_text(res, mode, verdict, out_dir)
            wrong = [] if getattr(a, "save", False) else jaw_mismatch(tool.to_dict(), good, tool)
            if wrong:                                       # (main's --save says so itself)
                _not_storable(res, mode, wrong)
            res["say"] += state.get("arm_ja", "")
            if state.get("stay_unconfirmed"):
                res["say"] += "登録された先端は未測定で、この実行でも確かめられていません。次に動かす前に、アームと道具を目で確かめてください。"
        elif (res.get("end") or {}).get("unconfirmed"):
            res["say"] = (f"ツール先端の測定値は ({res['tip_mm'][0]}, {res['tip_mm'][1]}, {res['tip_mm'][2]}) mm でしたが"
                          f"（写真 {res['photos']} 枚中 {res['used']} 枚）、基準を満たしていないか、記録と合わないため、確かな値では"
                          "ありません。" + SAY_HOLD + state.get("arm_ja", ""))
        elif (res.get("end") or {}).get("not_folded"):
            res["say"] = ("測定はできましたが、安全に畳める姿勢がないため、アームを畳んでいません（その場でトルク ON のまま保持しています）。"
                          + state.get("arm_ja", ""))
        else:
            res["say"] = "測定はできましたが、収納の動きが完了していません。" + state.get("arm_ja", "")
        return res
    finally:
        arm.tool = tool


# ----------------------------------------------------------------------------- --from DIR
def read_records(folder: str, tools_out: list | None = None) -> tuple:
    """(records, the tool they were taken with (the first file's) or None, file names) of every *records*.json in
    the folder, in the order of the file names. A record needs "q" (5 joint angles, rad) and "image"; an image
    that is not where the record says is looked for in the folder itself. tools_out: a list that receives the
    "tool" of every record file (None for a file that names none)."""
    records, tool, names = [], None, []
    for path in sorted(glob.glob(os.path.join(folder, "*records*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        recs = data.get("records") if isinstance(data, dict) else data
        if not isinstance(recs, list):
            continue
        names.append(os.path.basename(path))
        named = data["tool"] if isinstance(data, dict) and isinstance(data.get("tool"), dict) else None
        if tools_out is not None:
            tools_out.append(named)
        if tool is None:
            tool = named
        for r in recs:
            if not (isinstance(r, dict) and "image" in r and len(r.get("q") or ()) == 5):
                continue
            r = dict(r)
            if not os.path.isfile(str(r["image"])):
                r["image"] = os.path.join(folder, os.path.basename(str(r["image"])))
            records.append(r)
    return records, tool, names


def _one_tool(tools: list) -> str | None:
    """Why the record files of one folder do not describe ONE tool in one jaw state (None: they do)."""
    import tooling as TL
    same = int(getattr(TL, "JAW_SAME_TICKS", 3))
    named = [t for t in tools if t is not None]
    if not named:
        return None
    if len(named) != len(tools):
        return "some of the record files name no tool"
    first = named[0]
    for t in named[1:]:
        for key in ("name", "attached_at", "on_jaw"):
            if t.get(key) != first.get(key):
                return f"the record files differ in the tool's {key} ({first.get(key)!r}, {t.get(key)!r})"
        for key in ("gripper_raw", "gripper_goal_raw"):
            x, y = first.get(key), t.get(key)
            if (x is None) != (y is None) or (x is not None and abs(int(x) - int(y)) > same):
                return f"the record files differ in the tool's {key} ({x}, {y})"
    return None


def _from_dir(a, mode: str, cam) -> dict:
    import tooling as TL
    from robot import SafetyError
    folder = os.path.abspath(a.from_dir)
    res = {"ok": True, "stage": "done", "mode": MODE_NAME[mode], "dir": folder, "saved": False}
    if not os.path.isdir(folder):
        return _refusal("load", f"{folder} is not a folder", "give the folder of an earlier capture (runs/<date>/aircal_...)",
                        "指定したフォルダがありません。", mode=MODE_NAME[mode])
    tools = []
    try:
        records, taken_with, names = read_records(folder, tools)
    except (OSError, ValueError) as e:
        return _refusal("load", f"the records in {folder} cannot be read: {e}", "check the *records*.json files",
                        "記録ファイルを読めません。", mode=MODE_NAME[mode])
    images, records, missing = _load_images(records)
    res["records"] = names
    if missing:
        res["notes"] = [f"{len(missing)} photos named in the records are missing: {missing[:5]}"]
    if not records:
        return _refusal("load", f"no photo with a joint pose in {folder} (*records*.json with \"q\" and \"image\")",
                        "give the folder of an earlier capture", "写真と関節角の記録が見つかりません。", mode=MODE_NAME[mode], dir=folder)
    tool, tool_error = None, None
    try:
        tool = TL.active(a.sim)
    except SafetyError as e:
        tool_error = str(e)
    guess = (taken_with or {}).get("tip_offset") or (None if tool is None else tool.tip_offset)
    try:
        fit = _solve(mode, images, records, cam, guess)
    except ValueError as e:
        res.update(ok=False, stage="solve", reason=f"the photos give no fit: {e}",
                   next="is the tool tip in the pictures and bristle-coloured?"
                        + (f" Or the camera has moved: `{_from_words(folder, '--sets', 'air,low')}` solves it too"
                           if mode == "tip" else ""),
                   say="写真から先端の位置を求められませんでした。")
        return res
    verdict = judge(mode, fit, cam)
    res["tool"] = (taken_with or {}).get("name") or (None if tool is None else tool.name)
    first, last = _photo_times(records)
    res["taken"] = {"first": first, "last": last, "jaw": _jaw_seen(records),
                    "tool": None if taken_with is None else {k: taken_with.get(k) for k in
                                                             ("name", "attached_at", "gripper_raw", "on_jaw", "gripper_goal_raw")}}
    _report(res, mode, fit, records, guess, cam, verdict)
    _verdict_text(res, mode, verdict, folder)
    if not a.save:
        wrong = [] if tool is None else not_this_tool(taken_with, records, tool)
        if wrong:
            _not_storable(res, mode, wrong, _way_back(folder, mode) if hold_recorded_since(taken_with, records, tool) else None)
        return res
    mixed = _one_tool(tools)
    if tool_error or mixed:
        res.update(ok=False, stage="save", saved=False,
                   reason=("not stored: " + (tool_error or f"{mixed}: the photos of this folder do not show one tool in one "
                                                           "jaw state (--force does not change that)")),
                   next=("repair the tool file, or take the tool off by hand and run `fd tool detach`" if tool_error else
                         "`fd calib-air --tip-only --save` measures the tool that is registered now"),
                   say="保存していません（" + ("ツールのファイルが読めません" if tool_error else "このフォルダの記録は 1 つの道具の状態のものではありません") + "）。")
        return res
    _save(res, mode, fit, verdict, tool, a.sim, a.force, taken_with, records, folder)
    return res


# ----------------------------------------------------------------------------- the command
def _load_camera() -> tuple:
    """(the camera model of the rig, None) or (None, why its file cannot be used)."""
    import camera as C
    try:
        cam = C.CameraModel.load()
        numbers = [cam.f, cam.cx, cam.cy, *cam.rvec.tolist(), *cam.tvec.tolist()]
        if not all(math.isfinite(float(v)) for v in numbers) or cam.f <= 0.0:
            raise ValueError("its numbers are not a camera (focal length, pose)")
        return cam, None
    except Exception as e:      # not JSON, a key missing, a list where numbers belong, cv2 refusing the pose ...
        return None, f"{type(e).__name__}: {e}"


def main(argv=None, sim=None) -> dict:
    """`fd [--sim] calib-air ...` (fd.py prints the dict this returns as the JSON result of the command).

    argv: the words after `calib-air`, with "--sim" in FRONT when fd itself was called with --sim (that is how
    fd.py hands it over; a caller that knows may say sim=True / False instead). --sim anywhere else is refused:
    fd read the command as one for the REAL arm then, and has not moved the simulated tool file out of the
    project folder. One leading "--" is dropped (`fd calib-air -- --dry`, the spelling fd needed until
    2026-11-13). Every simulated result carries "sim": True."""
    words = list(sys.argv[1:] if argv is None else argv)
    lead = words[:1] == ["--sim"]
    if lead:
        words = words[1:]
    if words[:1] == ["--"]:
        words = words[1:]
    if "--sim" in words or (lead and sim is not None and not sim):
        typed = " ".join(w for w in words if w != "--sim")
        # (the same words in the spelling that works; where they have neither --dry nor --from, that spelling would be
        # refused too, the simulated arm has no camera: then the one that is not)
        works = "--dry" in words or "--from" in words
        return _refusal("options", "--sim must stand before the command: right after fd (fd --sim calib-air, then the "
                                   "other words). Nothing moved",
                        f"`fd --sim calib-air {typed}`" if works else
                        "`fd --sim calib-air --dry` plans a capture on the simulated arm (it has no camera: a simulated "
                        "call needs --dry, or --from and the folder of an earlier capture)",
                        "--sim は fd のすぐ後ろに書いてください（fd --sim calib-air ...）。何も動かしていません。")
    try:
        a = build_parser().parse_args(words)
    except _BadOptions as e:
        return _refusal("options", str(e), "`fd calib-air --help` lists the options",
                        f"指定を受け付けられませんでした（{e}）。アームは動かしていません。")
    a.sim = lead if sim is None else bool(sim)
    res = _command(a)
    if a.sim:
        res["sim"] = True
    return res


def _command(a) -> dict:
    import aircal as AC
    import camera as C
    import tooling as TL
    t0 = time.time()
    fd = "fd --sim calib-air" if a.sim else "fd calib-air"
    if a.sim and not os.environ.get("FD_TOOL") and _in_project(TL.tool_path(True)):
        # fd points the simulated tool file out of the project folder for every `fd --sim ...`: it has not, so it
        # did not read this call as a simulated one (`fd calib-air --sim ...`), or this is not fd calling
        return _refusal("options", "--sim must stand before the command: right after fd (fd --sim calib-air, then the "
                                   "other words). This call was not prepared as a simulated one: its tool file would be "
                                   f"{TL.tool_path(True)}, inside the project folder. Nothing moved",
                        "`fd --sim calib-air --dry` is how a simulated call is spelled (from a script: name a scratch tool "
                        "file with $FD_TOOL)",
                        "--sim は fd のすぐ後ろに書いてください（fd --sim calib-air ...）。何も動かしていません。")
    if a.force and not a.save:
        return _refusal("options", "--force only stores a fit that fails the gates: it needs --save. It does not skip the "
                                   "first look (--no-look does) and flies nothing",
                        f"`{fd} --dry` shows what a new capture would fly; --save --force go with --from and the folder of an "
                        "earlier capture, and store what that one measured (no motion)",
                        "--force は「基準を満たさない結果も保存する」という意味だけです（--save と一緒に使います）。"
                        "最初の確認を省くのは --no-look です。アームは動かしていません。")
    model_path = C.model_path()
    have_model = os.path.exists(model_path)
    if a.sets is not None:
        mode, names = "full", [n.strip() for n in a.sets.split(",") if n.strip()]
        known = AC.POSE_SETS["full"]
        if not names or any(n not in known for n in names) or len(set(names)) != len(names):
            return _refusal("options", f"--sets {a.sets!r}: a list of {', '.join(known)} without repeats, e.g. air,low",
                            "`fd calib-air --sets air,low`", "--sets の指定が読めません。アームは動かしていません。")
    else:
        mode = "tip" if (a.tip_only or have_model) else "full"
        names = list(DEFAULT_SETS[mode])
    if mode == "tip" and not have_model:
        return _refusal("options", f"--tip-only needs a camera model and {model_path} does not exist",
                        "`fd calib-air --sets air,low` solves the camera pose and the tip together",
                        "カメラモデルがないので、先端だけの校正はできません。fd calib-air --sets air,low を使ってください。")
    if a.dry and (a.save or a.from_dir):
        return _refusal("options", "--dry plans the capture: it has nothing to store and no photos to read (--save, --from)",
                        "`fd calib-air --dry` plans a capture; --from with the folder of an earlier capture solves its photos, "
                        "and stores the result with --save", "--dry は計画だけです。--save や --from とは一緒に使えません。")
    if a.sim and not (a.dry or a.from_dir):
        return _refusal("options", "the simulated arm has no camera: with --sim, calib-air works only with --dry (the plan) "
                                   "or with --from and the folder of an earlier capture (its photos)",
                        "`fd --sim calib-air --dry` plans a capture; with --from and the folder of an earlier capture the "
                        "simulated command solves that capture's photos",
                        "シミュレーションにはカメラがないので、--dry か --from と一緒に使ってください。")
    cam, notes = None, []
    if have_model:
        cam, why = _load_camera()
        if why and mode == "tip":
            # (the tip-only solve keeps the camera model: without one it has nothing to keep)
            full = (_from_words(os.path.abspath(a.from_dir), "--sets", "air,low") if a.from_dir else
                    f"{fd} --sets air,low" + (" --dry" if a.dry else ""))
            return _refusal("camera", f"the camera model {model_path} cannot be read ({why})",
                            f"`{full}` solves the camera pose and the tip together and needs no model to start from (with "
                            "--save the damaged file is copied to backups/ before the new model replaces it)",
                            "カメラモデルのファイルが読めません。カメラごと測り直す fd calib-air --sets air,low を使ってください"
                            "（アームは動かしていません）。", camera_model=model_path)
        if why:
            notes.append(f"the camera model {model_path} cannot be read ({why}): the full solve runs without a prior "
                         "model, and the first look has no model to check the registered tip with")
    if a.from_dir:
        res = _from_dir(a, mode, cam)
    else:
        res = _on_arm(a, mode, names, cam)
    if notes:
        res["notes"] = notes + list(res.get("notes") or [])
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _on_arm(a, mode: str, names, cam) -> dict:
    """--dry, or the capture: everything that opens the arm (the real one is refused while the hardware is locked)."""
    import tooling as TL
    from robot import SafetyError
    if not a.sim:
        import hwlock
        msg = hwlock.message("The servo bus and the camera")
        if msg:
            return {"ok": False, "error": "HardwareLocked", "stage": "hardware", "reason": msg, "message": msg,
                    "next": "`fd hw status` says why it is locked; `fd hw unlock` unlocks. `fd --sim calib-air --dry` needs "
                            "no hardware, nor does fd calib-air with --from and the folder of an earlier capture",
                    "say": "実機はロック中です（サーボもカメラも開いていません）。シミュレーション（fd --sim）とテストは使えます。"}
    paint = _painting()
    try:
        arm = _make_arm(a.sim)
    except SafetyError as e:
        return _refusal("start", str(e), "`fd tool status` / the rig's configuration; nothing moved",
                        "アームを開けませんでした（何も動かしていません）。", error="SafetyError", message=str(e))
    try:
        if arm.tool is None:
            return _refusal("tool", "no tool is registered: the air calibration measures the tip of a tool that is taped "
                                    "to the gripper", "register it with fd tool attach: --name, and --along-fingers with the "
                                    "length from the fingertip to the tool tip in metres (a guess that the first look will "
                                    "check). `fd tool status` shows what is registered",
                            "ツールが登録されていません。先に fd tool attach を実行してください。")
        if a.dry:
            return _dry(arm, arm.tool, cam, mode, names, a, paint)
        _term_as_interrupt()
        log = lambda *m: print(*m, file=sys.stderr, flush=True)      # noqa: E731
        out_dir = os.path.join(_run_dir(), "aircal_" + time.strftime("%H%M%S"))
        res = _run(a, mode, names, arm, cam, _photo_fn(), out_dir, log, paint)
        fit, verdict, used = res.pop("fit", None), res.pop("verdict", None), res.pop("fit_records", None)
        arm_next_stored = res.pop("arm_next_stored", None)
        stores = _from_words(out_dir, *([] if mode == "tip" else ["--sets", "air,low"]), "--save")
        if a.save and fit is not None and res["ok"]:
            try:
                now = TL.active(a.sim)      # the tool file as it is NOW: what is written, and what the photos must fit
            except SafetyError as e:
                now = None
                res.update(ok=False, stage="save", saved=False, reason=f"not stored: {e}", next=f"`{stores}` once the tool "
                           "file can be read again", say=(res.get("say") or "") + "ツールのファイルが読めないので、保存していません。")
            else:
                _save(res, mode, fit, verdict, now, a.sim, a.force, arm.tool.to_dict(), used or [], out_dir)
                if res.get("saved") and arm_next_stored and isinstance(res.get("arm"), dict):
                    res["arm"]["next"] = arm_next_stored    # (the tool file has the tip the arm was folded with)
        elif a.save and fit is not None:
            res.setdefault("notes", []).append(
                "nothing was stored because the command did not end as planned"
                + (": the fit does not pass the gates either (see \"gates\"), so there is nothing to store: measure again"
                   if (verdict or {}).get("problems") else f": `{stores}` stores this measurement"))
            res["say"] = (res.get("say") or "") + "保存はしていません。"
        try:
            st = arm.state()
            res.update(temps=st.get("temp_c"), volt=st.get("volt"))
        except Exception:
            pass
        return res
    finally:
        arm.close()
