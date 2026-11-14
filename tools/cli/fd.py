#!/usr/bin/env python
"""fd — the command line of TAZUNA (TAZUNA Robotics Harness; formerly "JetHarness" / "Fable Direct"), the harness for the
SO-101 follower (v2). "Give AI the reins of robotics."

Each invocation is ONE tool call made by the LLM operator. It prints one JSON result, appends
it to the run log and exits; the servos keep holding their last goal.

FAST PATH (use these first; one call does the whole job with built-in checks):
  fd scene                       what is on the table (objects, sizes, positions, containers)
  fd pick --color blue           pick the blue object and drop it into the (only) container
  fd pick --id 3 --into-color black --end park
  fd pick --color red --dry      plan only: shows the chosen grasp, moves nothing
  fd take --color white          take the white object OUT of the container and put it on the table
  fd take --id 4 --to 0.23,-0.03 --dry
  fd do "青いキューブを箱に入れて"   same thing from a plain sentence, read by rules (no LLM at all);
                                 「箱の中の白いキャップを取り出して」 runs `take`
  fd ready / fd park             stand-by pose (torque on) / folded rest pose (torque off)
  fd repl                        interactive: type requests one per line (camera and bus stay open)

`pick` / `take` stop at the first failed check and say which stage failed and why ("stage", "reason",
"arm_state", "next"); only then are the low-level commands below needed.

LOW LEVEL (one primitive per call): state, torque, joints, tip, check, grip, grasp, probe, snap.
TOOL ON THE GRIPPER (a pen, a brush taped to it): tool attach / status / set / rest / detach / calibrate / jaw.
  While a tool is attached "tip" is the TOOL tip, the gripper is locked, pick / take / do / probe / bench / park are
  refused, `ready` puts the tool tip at the stand-by point and `tool rest` is what folds the arm away.
  fd tool jaw status|mount|close|open   the jaw of a tool taped to the OUTER face of the moving jaw: close it on the
                                 fixed jaw and hold it there, so that the tool has no play (`fd tool jaw help`)
  fd calib-air [--tip-only] [--dry]     measure the tip of the tool from photos of it held in the air (and the camera
                                 pose, when the camera was moved): after a tool is attached, re-taped or its jaw moved
  fd tool calibrate              measure the PEN on the paper (tip, contact height, map, stroke direction). Refused for a
                                 painting tool (a brush: `fd paint status` leads through its steps) unless --pen says
                                 that a pen is on the gripper and this calibration is meant
  fd draw cat [--dry]            draw tools/draw/drawings/cat.json with it: photo, layout, preview, drawing, photo check
PAINTING (a brush on the gripper, a palette, water, a sheet): paint status / brush / check / look / locate / test /
  contact / preview / run / rinse / rest (`fd paint --help`). With a wet brush over a picture, `fd paint rest` folds
  the arm away, not `fd tool rest`.
HARDWARE LOCK: hw lock / unlock / status. While it is set, this copy of the harness opens neither the real servo bus
  nor the real camera (the simulated arm, --sim right after fd, and the tests still work).
CALIBRATION / SERVICE: ref, calib-at, calib-fit, calib-roll, calib-air, table-fit, bench, selftest, config, camera,
  report.

The words after `paint`, `calib-air` and `tool jaw` are read by those commands themselves (their own --help lists
them); fd's own option --sim stands BEFORE the command, as everywhere: `fd --sim calib-air --dry` (a --sim among
the words after one of these commands is refused).

Coordinates: robot base frame, metres (+x forward towards the camera, +y left, +z up).
Angles: degrees. "pitch" = how far the gripper's approach axis points below horizontal
(90 = straight down). "tip" = physical fingertip (gripper_frame_link + configured offset).
"jaw azimuth" = base-frame direction from the fixed jaw to the moving jaw (the side it opens to).
Add --sim before the command to run against a simulated arm: the servo bus is never opened. The
camera is still the real one (opened, exposure locked) unless saved images are given with --frame;
`--sim ... --frame IMG` touches no hardware at all. --frame without --sim is for dry runs (--dry) only.
To de-energise the arm use `park` (never `torque off` with the arm raised: it would fall).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tools/

try:
    import importlib.util

    import numpy as np

    if importlib.util.find_spec("cv2") is None:      # OpenCV is imported later, by the commands that use pictures
        raise ModuleNotFoundError("No module named 'cv2'", name="cv2")
    import tazuna_paths  # puts every tools/<feature>/ folder on sys.path, says where the files are
    import kinematics as K
    from hwlock import HardwareLocked
    from robot import ARM, Arm, MotionAborted, SafetyError, load_config, require_rig_config
    from servo_bus import BusError
except ImportError as _e:               # still answer with one JSON
    _name = (getattr(_e, "name", None) or "?").split(".")[0]
    _hint = ("TAZUNA needs numpy, OpenCV (cv2), feetech-servo-sdk (scservo_sdk) and pyserial: set TAZUNA_PYTHON to an "
             "interpreter that has them, or write its path into config/python")
    if not isinstance(_e, ModuleNotFoundError):                 # installed, but it does not load
        _msg = f"this Python ({sys.executable}) cannot load a module TAZUNA needs ({_e}). {_hint}"
    elif _name in ("numpy", "cv2", "scservo_sdk", "serial"):    # a Python without the packages TAZUNA needs
        _msg = f"this Python ({sys.executable}) has no module '{_name}'. {_hint}"
    else:                                                       # a file of the harness itself is missing
        _msg = (f"the TAZUNA module '{_name}' was not found under {os.path.dirname(HERE)}: the tools/ folder of this "
                "copy is incomplete")
    print(json.dumps({"ok": False, "error": type(_e).__name__, "message": _msg}, ensure_ascii=False))
    sys.exit(1)

ROOT = tazuna_paths.ROOT                                   # the project folder (runs/, backups/, the tool file)
RUNS = os.environ.get("FD_RUNS") or os.path.join(ROOT, "runs")      # $FD_RUNS: tests that start fd as a process keep the rig's run log clean
CALIB_POINTS = os.path.join(tazuna_paths.CONFIG_DIR, "calib_points.json")
REF_PATH = os.path.join(tazuna_paths.CONFIG_DIR, "ref_table.jpg")
PROBES = os.path.join(tazuna_paths.CONFIG_DIR, "probes.json")
AXIS_TOOL = (-0.0079, 0.000218121, 0.0068)  # point on the wrist-roll axis at fingertip depth (tool frame)
ROLL_SKEW_DEG = 2.789  # jaw azimuth = roll - pan - ROLL_SKEW_DEG for a vertical gripper

# Pose in which the arm was found (servo ticks), used for the torque-off guard.
REST_RAW = {"shoulder_pan": 1999, "shoulder_lift": 955, "elbow_flex": 3025, "wrist_flex": 2820,
            "wrist_roll": 1983}


# ------------------------------------------------------------------ run log
def run_dir() -> str:
    os.makedirs(RUNS, exist_ok=True)
    ptr = os.path.join(RUNS, "current")
    if os.path.exists(ptr):
        d = open(ptr).read().strip()
        # one run directory per day, and only one of THIS folder: the pointer holds a full path, so in a copy of the
        # project folder it still names the original's runs/ (the copy would log into the original)
        if (os.path.isdir(d) and os.path.basename(d).startswith(time.strftime("%Y%m%d"))
                and os.path.realpath(os.path.dirname(d)) == os.path.realpath(RUNS)):
            return d
    d = os.path.join(RUNS, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    with open(ptr, "w") as f:
        f.write(d)
    return d


def log_event(cmd: str, args: dict, result: dict, t_start: float | None = None):
    d = run_dir()
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd, "args": args, "result": result,
           "python": sys.executable}
    if t_start is not None:
        rec["duration_s"] = round(time.time() - t_start, 2)
    with open(os.path.join(d, "events.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=_jsonable, ensure_ascii=False) + "\n")


def next_image_path(tag: str) -> str:
    d = run_dir()
    n = len([f for f in os.listdir(d) if f.endswith(".jpg") and f[:4].isdigit()])
    return os.path.join(d, f"{n:04d}_{tag}.jpg")


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def out(result: dict):
    print(json.dumps(result, default=_jsonable, indent=None, ensure_ascii=False))


# ------------------------------------------------------------------ helpers
def brief_state(arm: Arm) -> dict:
    st = arm.state()
    keys = ("q_deg", "tip_xyz", "pitch_down_deg", "jaw_yaw_deg", "gripper_open_frac", "load", "temp_c", "volt",
            "goal_minus_pos_ticks", "status", "faults")
    out = {k: st[k] for k in keys if k in st}
    stats = getattr(arm.bus, "stats", None)
    if stats:
        out["bus_stats"] = dict(stats)  # retries / fallbacks / failures / requests of this command
    return out


def roll_for_azimuth(arm: Arm, xyz, pitch: float, azimuth_deg: float) -> float:
    """Wrist roll (deg) that puts the MOVING jaw on the `azimuth_deg` side of the fixed jaw
    (directed, modulo 360). Closed form (exact for the horizontal projection of the jaw line at
    any pitch), iterated because the pan angle depends slightly on the roll, verified by FK."""
    tool, lim = arm.tip_tool(), arm.q_limits()
    pan = -math.degrees(math.atan2(xyz[1], xyz[0] - K._X_PAN))
    roll = None
    exact = getattr(K, "roll_for_azimuth", None)  # exact at any pitch (kinematics.py)
    for _ in range(3):
        if exact is not None and 1.0 < pitch < 179.0:
            roll = float(exact(azimuth_deg, pan, pitch))
        else:
            roll = (azimuth_deg + pan + ROLL_SKEW_DEG + 180.0) % 360.0 - 180.0
        q, info = K.ik(xyz, pitch, math.radians(roll), tool=tool, limits=lim)
        if not info["ok"]:
            raise SafetyError(f"jaw azimuth {azimuth_deg} deg needs wrist roll {roll:.1f} deg, which is not "
                              f"feasible here: {info['reason']}")
        pan = math.degrees(float(q[0]))
    jaw = K.jaw_open_axis(K.fk(q))
    horiz = math.hypot(jaw[0], jaw[1])
    if horiz < 0.5:
        raise SafetyError("the gripper is too far from vertical for a jaw azimuth to make sense; give --roll")
    d = (math.degrees(math.atan2(jaw[1], jaw[0])) - azimuth_deg + 180.0) % 360.0 - 180.0
    if abs(d) > 2.0:
        raise SafetyError(f"jaw azimuth {azimuth_deg} deg cannot be reached at pitch {pitch} (off by {d:.1f} deg)")
    return float(roll)


def _best_pitch(arm, xyz, roll_deg, want=90.0):
    """Feasible pitch closest to `want` at this fingertip position (None if unreachable)."""
    best = getattr(K, "best_pitch", None)
    if best is not None:
        p, _deg = best(xyz, want=want, wrist_roll=math.radians(roll_deg), tool=arm.tip_tool(), limits=arm.q_limits())
        return None if p is None else float(p)
    rng = K.pitch_range(xyz, math.radians(roll_deg), tool=arm.tip_tool(), limits=arm.q_limits())
    if rng is None:
        return None
    lo, hi = rng[0] + 1.0, rng[1] - 1.0
    return max(lo, min(hi, want)) if lo <= hi else float(rng[0] + rng[1]) / 2.0


def _append_json(path, rec):
    data = json.load(open(path)) if os.path.exists(path) else []
    data.append(rec)
    json.dump(data, open(path, "w"), indent=1, default=_jsonable)
    return len(data)


def edit_config_file(mutate) -> dict:
    """Change config.json itself (the overrides), not the merged configuration: saving the merged
    one would freeze every built-in default into the file. The previous file is kept in backups/.
    mutate(raw_dict) edits in place. Returns the merged configuration afterwards."""
    from robot import CONFIG_PATH
    if os.environ.get("FD_CONFIG"):
        # reads follow $FD_CONFIG, this would write config/config.json: the answer would not show what was written
        raise SafetyError("FD_CONFIG is set: the configuration is read from that file, and this command would change "
                          f"{CONFIG_PATH} instead. Unset FD_CONFIG to change the rig's file")
    if not os.path.isfile(CONFIG_PATH):
        # a file made here would hold only the keys of this command and none of the rig's measurements
        raise SafetyError(f"{CONFIG_PATH} is missing: this command changes the rig's configuration file, it does not "
                          "create one (restore the file, e.g. from backups/ or from the repository)")
    raw = json.load(open(CONFIG_PATH))
    os.makedirs(os.path.join(ROOT, "backups"), exist_ok=True)
    shutil.copy(CONFIG_PATH, os.path.join(ROOT, "backups", f"config.{time.strftime('%m%d_%H%M%S')}.json"))
    mutate(raw)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(raw, f, indent=2, sort_keys=True, default=_jsonable)
    os.replace(tmp, CONFIG_PATH)
    return load_config()


def _not_with_sim(a, what: str, instead: str = ""):
    """`--sim` promises that nothing of the rig is touched: commands that write its files or change the real camera
    refuse it (they do not use the arm, so the flag would only look safe)."""
    if a.sim:
        raise SafetyError(f"\"fd --sim {a.cmd}\" would {what}: --sim does not protect that. "
                          + (instead or "Run the command without --sim if that is what you want"))


def _ref_photo() -> dict:
    """What a fingertip-detecting command says about the reference photo of the empty table (`fd ref`)."""
    if os.path.exists(REF_PATH):
        return {"reference_photo": True}
    old = os.path.join(ROOT, "ref_table.jpg")          # where it was kept until the reorganisation of 2026-11-13
    return {"reference_photo": False,
            "note": f"no reference photo ({REF_PATH}): the fingertip was looked for without it; "
                    + (f"an older one lies at {old}: move it into config/, or " if os.path.exists(old) else "")
                    + "take one with `fd ref` (arm clear of the table)"}


def _cam_model():
    import camera as C
    if not os.path.exists(C.model_path()):
        raise SafetyError("camera_model.json is missing: run the camera calibration first")
    return C.CameraModel.load()


def _camera_to_auto(a) -> bool:
    """`park` and leaving `repl` hand the camera back to its automatic exposure (v4l2-ctl on the REAL
    camera). Under --sim that must not happen: a simulated command has no business with the real camera's
    settings (it ran on the real camera until 2026-11-12). True when the call was made."""
    if getattr(a, "sim", False):
        return False
    try:
        import camera as C
        C.restore_auto()
    except Exception:
        return False
    return True


def sim_state_dir() -> str:
    """Where `fd --sim` keeps what its simulated world remembers between commands (the simulated tool, the
    simulated sheet of paper): a directory OUTSIDE the tree, one per tree. A tool attached in simulation
    used to be written into the tree (tool_active.sim.json) and travelled with every copy of it."""
    import hashlib
    import tempfile
    return os.path.join(tempfile.gettempdir(), f"tazuna_sim_{os.getuid()}", hashlib.sha1(ROOT.encode()).hexdigest()[:10])


def _sim_tool_out_of_tree():
    """Guard of `fd --sim`: the simulated tool file never lies in the tree. Unless $FD_TOOL names the file
    (tests, experiments), tooling.SIM_TOOL_FILE is pointed at sim_state_dir() for this process; a file that an
    older version left in the tree is moved there. Returns the value to put back afterwards (None: untouched)."""
    import tooling
    if os.environ.get("FD_TOOL"):
        return None
    old = tooling.SIM_TOOL_FILE
    tree = os.path.realpath(ROOT) + os.sep
    if not (os.path.realpath(old) + os.sep).startswith(tree):
        return None                      # somebody already put it elsewhere (a test)
    new = os.path.join(sim_state_dir(), os.path.basename(old))
    if os.path.exists(old) and not os.path.exists(new):
        os.makedirs(os.path.dirname(new), exist_ok=True)
        shutil.move(old, new)
    tooling.SIM_TOOL_FILE = new
    return old


def _fd(a) -> str:
    """`fd`, or `fd --sim` when this call was simulated: the first words of a command a result names, so that
    what is typed after a rehearsal stays a rehearsal."""
    return "fd --sim" if getattr(a, "sim", False) else "fd"


def _sim_tool_said(res: dict, a, wrote: bool = False, was: dict | None = None) -> dict:
    """What fd's own simulated tool commands (attach / set / calibrate / detach) say about the tool file they
    work on, added to the result; nothing without --sim. The same keys as `fd --sim tool jaw` gives
    (tooljaw._sim_said): "simulated_tool" (the simulation reads a tool file of its own, not the real arm's),
    "sim_note" (which file that is, and how to get back to the real tool), "sim_copy_dropped" (a simulated copy
    that is not read any more because the real tool was changed or replaced since), and a sentence in "say".

    Review of 2026-11-14: `fd --sim tool set --pitch 60` on the tool the simulation was mirroring stored a copy of
    the real tool's record as the simulated tool and said nothing of it; every later `fd --sim` then rehearsed
    with a record the real arm does not have. wrote: this command stored the tool; was: tooling.sim_tool_state()
    at the start of the command (a copy it wrote over, or removed, is still reported as left behind)."""
    if not getattr(a, "sim", False):
        return res
    import tooling
    st = tooling.sim_tool_state()
    res["simulated_tool"] = bool(st["simulated_tool"])
    dropped = st["dropped"] or (was or {}).get("dropped")
    if dropped:
        res["sim_copy_dropped"] = dropped
        res["say"] = (res.get("say") or "") + ("（以前のシミュレーション用のツール記録は、そのあと実機のツール記録が変わったため"
                                               "使っていません。）")
    if st["simulated_tool"]:
        res["sim_note"] = ("this command stored the tool in the SIMULATED tool file: " if wrote else "") + (st["note"] or "")
        if wrote:
            res["say"] = (res.get("say") or "") + (
                "（シミュレーション用のツール記録に保存しました。実機のツール記録は変えていません。"
                + ("実機の記録が変わるか fd --sim tool detach を実行するまで、" if st["copy_of_real"] else
                   "fd --sim tool detach を実行するまで、") + "fd --sim はこの記録を使います。）")
    return res


# A PAINTING tool is not a pen. The Japanese half of why a tool counts as one (_painting_tool).
PAINTING_JA = {"tip": "筆先を空中で測った記録（fd calib-air）がある",
               "jaw": "可動爪の外側に付けた、または爪の間に挟んだという記録（fd tool jaw）がある",
               "setup": "絵のセットアップが登録されている"}


def _paint_setup_on_record() -> str | None:
    """Path of the painting set-up that counts for the tool file in force (paintskill.setup_path()), or None: no
    painting module, no set-up file, or a tool file named by $FD_TOOL while the set-up is the rig's own. The last
    is the rule of the canvas gate of `fd tool rest` (_rest_canvas): a scratch tool file is a test or an experiment
    and the rig's set-up describes another table (the pen suites would otherwise pass or fail by what was painted
    on this desk last); with $FD_PAINT_SETUP pinned too, the set-up counts."""
    try:
        import paintskill
    except Exception:           # the painting feature is not there, or does not load
        return None
    setup_path = getattr(paintskill, "setup_path", None)
    if not callable(setup_path):
        return None
    path = setup_path()
    if not os.path.isfile(path) or (os.environ.get("FD_TOOL") and not os.environ.get("FD_PAINT_SETUP")):
        return None
    return path


def _painting_tool(tool) -> tuple | None:
    """(kind, why) when `tool` is a PAINTING tool (a brush: `fd tool jaw`, `fd calib-air` and the `fd paint` commands
    prepare it), None for a pen. kind: "tip" | "jaw" | "setup" (PAINTING_JA); why: one English clause. Files only.

    Why (review of 2026-11-14, on copies of the rig's files): `fd tool status` answered a brush with "not
    calibrated (tip, contact, map): run `fd tool calibrate` before `fd draw`", and nothing refused that command.
    It is the PEN calibration: it lowers the tool tip onto the paper until ink shows, as far as 6 mm below where the
    paper is expected, and replaces the tip offset by the one it reads from the ink. Its tip stage ran on the brush
    whose tip had been measured in the air: down to plan z -6 mm at a place inside the canvas, 9 to 15 mm below
    first touch (the bristles are 15 mm long; the paint commands allow 7.5), and the measured tip was replaced.

    A painting tool is one whose file has a measured tip (tip_measured: `fd calib-air --save`) or records a jaw
    held shut for it (gripper_goal_raw: `fd tool jaw close`; since round 3b a jaw record WITHOUT a hold, on_jaw
    fixed / moving_outer / between alone, does not make one), and ANY tool while a painting set-up is on record
    (_paint_setup_on_record): a
    sheet with a picture may lie where a pen calibration would put its test marks. For such a tool `fd tool
    status`, `attach` and `set` name `fd paint status`, never the pen calibration, and `fd tool calibrate` is
    refused unless --pen declares that a pen is on the gripper (_tool_calibrate).

    A tool recorded on the FIXED jaw is not one by that record alone (round 3, 2026-11-14). The next of a refused
    `fd ready` sends a PEN whose limp jaw moved to `fd tool jaw mount fixed` (the moving jaw does not carry it, so
    it is not compared any more). Any jaw record counted until then: the pen became a painting tool, `fd tool
    status` and a refused `fd draw` sent it to `fd paint status`, which is refused on a rig without a painting
    set-up, and its own calibration was refused without --pen."""
    if tool is None:
        return None
    # (the commands are named in words: inside backticks stands only a command that is the next step as it is typed)
    if getattr(tool, "tip_measured", None):
        return "tip", "its tip was measured in the air (by fd calib-air)"
    on = getattr(tool, "on_jaw", None)
    if getattr(tool, "gripper_goal_raw", None) is not None:
        # (round 3b, U6: only a HOLD marks a painting tool. on_jaw moving_outer / between alone does not: the next of
        # a refused `fd ready` sends a pen to `fd tool jaw mount ...`, and a pen recorded that way on a rig without a
        # painting set-up was sent to `fd paint status`, which is refused there)
        where = {"moving_outer": "on the outer face of the moving jaw, ", "between": "between the jaws, "}.get(on, "")
        return "jaw", f"its file records it {where}with a jaw held shut for it (a record of fd tool jaw)"
    path = _paint_setup_on_record()
    return None if path is None else ("setup", f"a painting set-up is on record ({path})")


def _fold_words(a, tool) -> str:
    """The command that folds the arm away with `tool` on, for a refusal that names it: `fd tool rest`, or, for a
    painting tool with a painting set-up on record, `fd paint rest` in the spelling of this call (`fd --sim paint
    rest` after a simulated one), as `fd tool status` names it since round 2.

    Why (round 3, 2026-11-14, copies of the rig's files): `fd --sim park` answered "`fd tool rest` folds the arm
    away with the tool on", and `fd --sim tool rest` was then refused at stage canvas (the tool's default fold has
    the brush over the sheet: _rest_canvas) with `fd --sim paint rest` as its next: two refusals before the
    command that folds the arm off the picture, the first without --sim after a simulated call."""
    if tool is not None and _painting_tool(tool) and _paint_setup_on_record():
        return f"{_fd(a)} paint rest"
    return "fd tool rest"


def _tool_refusal(arm, what: str, a=None):
    """The guard at the top of every command that would move the gripper, fold the arm as if nothing were
    taped to it, or poke the fingertip at something: with a tool attached (arm.tool) the complete refusal
    result ("stage": "tool", "next" names `fd tool rest` / `fd tool detach`; for a painting tool with a painting
    set-up on record `fd paint rest`: _fold_words), else None. Nothing is written to the servo bus before it."""
    tool = getattr(arm, "tool", None)
    if tool is None:
        return None
    import tooling
    fold = _fold_words(a, tool)
    if fold == "fd tool rest":
        return tooling.refusal(tool, what)
    return tooling.refusal(tool, what, fold=fold, detach=f"{_fd(a)} tool detach")


def _tool_no_contact(arm, a, what: str):
    """Guard of the low-level motion commands: with a tool attached, --contact (the tip up to contact_depth_m
    below the table height) is refused. That is how a pen was pressed into the sheet by hand-driven steps on
    2026-11-11; lowering a tool onto the paper belongs to the commands that stop by what the camera sees."""
    tool = getattr(arm, "tool", None)
    if tool is None or not getattr(a, "contact", False):
        return None
    if _painting_tool(tool):         # a brush is never sent to the pen commands (_painting_tool)
        left_to = "the painting commands, which press a brush by the contact plane of the painting set-up"
        instead = f"; `{_fd(a)} paint status` says how a brush is brought onto the paper"
    else:
        left_to = "the pen commands (fd draw, fd tool calibrate), which stop by what the camera sees"
        instead = ", or use the pen commands (`fd tool status` says what the pen still needs)"
    why = (f"tool '{tool.name}' is attached: `{what} --contact` would let its tip go up to "
           f"{float(arm._c('contact_depth_m')) * 1000:.0f} mm below the table height by hand. Nothing moved. Without "
           f"--contact the tip can be brought down to the table height; pressing a tool onto the paper is left to {left_to}")
    return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why, "tool": tool.name,
            "next": f"omit --contact (the tool tip then stops at the table height){instead}",
            "say": f"ツール「{tool.name}」が付いているので、--contact（机の高さより下へ押し込む動き）は実行しませんでした"
                   "（アームは動かしていません）。"}


# ------------------------------------------------------------------ fast path
CAMERA_COMMANDS = ("scene", "pick", "take", "do", "selftest", "repl")
# commands that move the arm by what the camera shows: with --frame (saved images) they are for the
# simulated arm (--sim) or a dry run (--dry) only
MOVING_CAMERA_COMMANDS = ("pick", "take", "do", "repl")


def _open_camera(a):
    """Live camera thread, or a replay of saved frames (--frame, repeatable). main() calls
    this before the arm is opened, so the camera warms up while the bus is being set up."""
    import skills
    if getattr(a, "frame", None):
        import cv2
        frames = [cv2.imread(p) for p in a.frame]
        if any(f is None for f in frames):
            raise SafetyError("cannot read one of the --frame images")
        return skills.ReplayCamera(frames)
    import fastcam
    cam = fastcam.FastCamera(lock=True, restore=False)
    cam.__enter__()
    return cam


def _close_camera(cam):
    try:
        if hasattr(cam, "__exit__"):
            cam.__exit__(None, None, None)
        elif hasattr(cam, "close"):
            cam.close()
    except Exception:
        pass


def _query(a, prefix="") -> dict:
    q = {}
    for key in ("color", "kind", "id"):
        v = getattr(a, prefix + key, None)
        if v is not None:
            q[key] = v
    near = getattr(a, prefix + "nearest", None)
    if near:
        try:
            xy = tuple(float(v) for v in str(near).split(","))
        except ValueError:
            xy = ()
        if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise SafetyError("--nearest needs x,y in metres (base frame), e.g. --nearest 0.23,-0.03")
        q["nearest_to"] = xy
    if getattr(a, prefix + "largest", False):
        q["largest"] = True
    return q


def cmd_scene(arm, a):
    """Perceive the table: every object with kind, colour, position, size; containers."""
    import cv2

    import scene as S
    import skills
    cam_model = _cam_model()
    camera = a._camera
    t0 = time.time()
    img = camera.latest(n_avg=2)
    t1 = time.time()
    q = arm.read_q()
    jaw = arm.gripper_q_deg(int(arm.bus.read("gripper", "Present_Position")))
    sc = S.perceive(img, cam_model, arm.cfg, arm_q=q, jaw_q_deg=jaw)
    t2 = time.time()
    raw_path = next_image_path("scene_raw")
    cv2.imwrite(raw_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    view_path = raw_path.replace("_raw.jpg", "_view.jpg")
    try:
        import vision as V
        V.save_view(S.annotate(img, sc, cam_model), view_path, scale=0.6667)
    except Exception as e:
        view_path = f"(annotation failed: {e})"
    return {"objects": skills._summarise(sc), "warnings": list(sc.warnings), "table_z": sc.table_z,
            "timings": {"frame_s": round(t1 - t0, 2), "perceive_s": round(t2 - t1, 2)},
            "raw": raw_path, "view": view_path}


def cmd_pick(arm, a):
    """Pick an object and drop it into a container: perceive, plan, move, grasp, verify."""
    blocked = _tool_refusal(arm, "pick", a)
    if blocked:
        return blocked
    import skills
    cam_model = _cam_model()
    # the arm waits at the stand-by pose between objects; `--end park` folds it away once, at the end
    opts = skills.options_from_config(arm.cfg, tier=a.tier, visual=a.visual, retries=a.retries)
    opts.end, opts.dry, opts.any_match = "ready", bool(a.dry), bool(a.all)
    if a.no_verify:
        opts.verify = False
    notes = []
    if a.sim and not a.dry:
        _sim_object(arm)
        opts.visual = "off"  # the camera cannot see a simulated arm
        if a.all:
            notes.append("simulation: --all carries one object only (the picture does not change when the simulated "
                         "arm moves, so the next look would find the same object again)")
        if len(a.frame or []) < 2:
            opts.verify = False  # ... nor a simulated result
            notes.append("simulation: the look at the jaws and the final verification are skipped "
                         "(give two --frame images, before and after, to exercise the verification)")
        else:
            notes.append("simulation: the look at the jaws is skipped")
    sess = skills.Session(arm, a._camera, cam_model, image_dir=run_dir())
    results = []
    n = 0
    while True:
        n += 1
        r = skills.pick_place(sess, _query(a), _query(a, "into_") or None, opts)
        results.append(r)
        if not (a.all and r["ok"]) or a.dry or n >= a.max_objects or (a.sim and not a.dry):
            break
    last = results[-1]
    if (a.all and len(results) > 1 and not last["ok"] and last.get("stage") == "perceive"
            and "target not found" in last.get("reason", "")):
        # "--all" ends when no matching object is left: that is success, not a failure
        results.pop()
        last = results[-1]
    if a.end == "park" and not a.dry and last.get("arm_state") in ("ready", "standby"):
        skills.park(arm, arm.speeds(a.tier))
        last["arm_state"] = "parked"
    # short by default (less for the operator to read); everything with --verbose or on failure
    keep = ("ok", "stage", "reason", "arm_state", "attempts", "target", "checks", "timings", "images")
    res = {k: last[k] for k in keep if k in last}
    if "container" in last:
        res["container"] = {k: last["container"].get(k) for k in ("id", "color")}
    if "speed_tier" in last:
        res["tier"] = last["speed_tier"].get("tier")
    if last.get("scene_warnings"):
        res["scene_warnings"] = last["scene_warnings"]
    if a.verbose or not last["ok"] or a.dry:
        for k in ("scene", "grasp", "release", "plan", "grasp_candidates", "speed_tier"):
            if k in last:
                res[k] = last[k]
        if "container" in last:
            res["container"] = last["container"]
    else:
        g = last.get("grasp", {})
        res["grasp"] = {"jaw_azimuth_deg": round(float(g.get("jaw_azimuth_deg", 0.0)), 1),
                        "width_mm": round(float(g.get("width", 0.0)) * 1000, 1),
                        "tip_grasp": [round(float(v), 4) for v in g.get("tip_grasp", ())]}
    if not last["ok"]:
        res["next"] = _advice(last)
    if notes:
        res["notes"] = notes
    stats = getattr(arm.bus, "stats", None)
    if stats:
        res["bus_stats"] = dict(stats)
    if len(results) > 1:
        res["picked"] = [{"target": r.get("target"), "ok": r["ok"], "total_s": r.get("timings", {}).get("total")}
                         for r in results]
    res["ok"] = all(r["ok"] for r in results)
    return res


def cmd_take(arm, a):
    """Take an object OUT of a container and put it down on the table: perceive, plan the grasp from
    the free space around it, look at the jaws above the rim, straight down to a height taken from
    geometry (table + container floor + the object's own bottom), close, lift, carry out, put down,
    verify. One call, no LLM in the loop."""
    blocked = _tool_refusal(arm, "take", a)
    if blocked:
        return blocked
    import skills
    cam_model = _cam_model()
    to = None
    if getattr(a, "to", None):
        try:
            to = tuple(float(v) for v in str(a.to).split(","))
        except ValueError:
            to = ()
        if len(to) != 2:
            raise SafetyError("--to needs x,y in metres (base frame), e.g. --to 0.23,-0.03")
    opts = skills.take_options_from_config(arm.cfg, tier=a.tier, retries=getattr(a, "retries", None),
                                           probe=getattr(a, "probe", None))
    opts.end, opts.dry, opts.to = "ready", bool(a.dry), to
    if getattr(a, "hollow", False):
        opts.hollow = True
    elif getattr(a, "solid", False):
        opts.hollow = False
    if getattr(a, "assume_hollow", False):  # `fd do`: the sentence calls it a cap, a cup, a lid
        opts.assume_hollow = True
    if getattr(a, "no_verify", False):
        opts.verify = False
    notes = []
    sess = skills.Session(arm, a._camera, cam_model, image_dir=run_dir())
    if a.sim and not a.dry:
        skills.sim_take_hooks(sess)  # an object between the jaws (and a surface under the optional touch)
        opts.visual = "off"  # the camera cannot see a simulated arm
        if len(a.frame or []) < 2:
            opts.verify = False  # ... nor a simulated result
            notes.append("simulation: the look at the jaws and the camera checks of the result are skipped "
                         "(give two --frame images, before and after, to exercise the verification)")
        else:
            notes.append("simulation: the look at the jaws is skipped")
    last = skills.take_out(sess, _query(a), _query(a, "from_") or None, opts)
    if a.end == "park" and not a.dry and last.get("arm_state") in ("ready", "standby"):
        skills.park(arm, arm.speeds(a.tier))
        last["arm_state"] = "parked"
    # short by default (less for the operator to read); everything with --verbose or on failure
    keep = ("ok", "skill", "stage", "reason", "arm_state", "attempts", "target", "checks", "timings", "images")
    res = {k: last[k] for k in keep if k in last}
    if "container" in last:
        res["container"] = {k: last["container"].get(k) for k in ("id", "color")}
    if "speed_tier" in last:
        res["tier"] = last["speed_tier"].get("tier")
    if last.get("scene_warnings"):
        res["scene_warnings"] = last["scene_warnings"]
    if "place" in last:
        res["place"] = last["place"]["xy"]
    if a.verbose or not last["ok"] or a.dry:
        for k in ("scene", "grasp", "plan", "grasp_candidates", "speed_tier"):
            if k in last:
                res[k] = last[k]
        if "container" in last:
            res["container"] = last["container"]
    elif "grasp" in last:
        g = last["grasp"]
        res["grasp"] = {"mode": g.get("mode"), "jaw_azimuth_deg": round(float(g.get("jaw_azimuth_deg", 0.0)), 1),
                        "tip_grasp": [round(float(v), 4) for v in g.get("tip_grasp", ())]}
    if not last["ok"]:
        res["next"] = last.get("next") or _advice_take(last)
    if notes:
        res["notes"] = notes
    stats = getattr(arm.bus, "stats", None)
    if stats:
        res["bus_stats"] = dict(stats)
    return res


def _advice_take(r: dict) -> str:
    """One line for the operator: what to do after a failed take-out (saves a round of thinking)."""
    stage, state = r.get("stage"), r.get("arm_state")
    reason = r.get("reason", "")
    # (round 3: a command with a value the program does not know stands in words, not in backticks: `fd take --id N`,
    # typed as it was printed, is refused)
    if stage == "start":
        return ("nothing moved. Use a validated speed tier (omit --tier), or validate this one on the real arm with "
                "fd bench, --tier and its name, --accept")
    if stage == "perceive" and reason.startswith("container not found"):
        return ("nothing was taken. The camera does not recognise a container here (it must be dark, open at the top "
                "and not hidden): `fd scene` shows what it sees")
    if stage == "perceive" and reason.startswith("not in a container"):
        return "nothing was taken. The object is not inside a container: fd pick moves things that stand on the table"
    if stage == "perceive" and "objects inside the container match" in reason:
        return "several objects fit the description: take one with fd take --id N (N: its id in \"scene\")"
    if stage == "perceive":
        return ("nothing was taken. Check \"scene\" for what the camera sees inside the container; take by id (fd take "
                "--id N, N: the id in \"scene\")")
    if stage == "plan" and any(k in reason for k in ("cannot point straight down", "IK failed", "out of reach")):
        return ("nothing was taken: the arm cannot hold the gripper straight down over the object (reach). Move the "
                "container to straight in front of the robot, well inside its reach, and repeat; more room around the "
                "object or another place spot (--to) does not help")
    if stage == "plan":
        return ("nothing was taken: no way to hold the object was found (see \"grasp_candidates\"); make room around it "
                "or give the place with --to x,y")
    if stage == "visual" and any(k in reason for k in ("calibration?", "not confirmed", "disagree")):
        return ("the arm went back to the stand-by pose WITHOUT descending: what the camera measured of the jaws is "
                "larger than a look can be trusted with, or two looks did not agree (see checks.visual.steps). Do not "
                "repeat blindly: compare jaws and object by hand in ONE photo (hover with fd tip, then fd snap with "
                "--crop x0,y0,x1,y1 --scale 4 --plain); a steady offset of more than 1 cm means the calibration is off")
    if stage == "visual":
        return ("the arm went back to the stand-by pose WITHOUT descending: the camera could not measure where the jaws "
                "are (something pale behind them, glare, or a jaw hidden). Change that and repeat; by hand: hover with "
                "fd tip, compare jaws and object in ONE photo (fd snap with --crop x0,y0,x1,y1 --scale 4 --plain)")
    if state == "holding_object":
        return ("the arm stopped while HOLDING the object: inspect with `fd state` / `fd snap`, then move it by hand "
                "with the low-level commands (fd tip and its x y z, `fd grip 0.5`) or ask a person; do not use `park` yet")
    if state == "stopped":
        return "the arm stopped away from the stand-by pose: run `fd state`, then `fd ready` if the way is clear"
    if stage == "probe":
        return ("the optional touch (--probe) stopped the take-out (see checks.probe); the arm is back at the stand-by "
                "pose. Either something lies higher under the jaws than the scene says (the object's rim, a closed "
                "top, a neighbour) or the touch misread: look at the saved scene image before repeating")
    if stage in ("grasp", "lift"):
        return ("the object is still in the container (it may have moved) and the arm is at the stand-by pose: repeat "
                "the command; if it slips again, look at the saved scene image and choose with --hollow / --solid")
    if stage == "transfer":
        return "the object was dropped on the way: `fd scene` shows where it lies now (then fd pick or fd take)"
    if stage == "verify":
        return "the motion finished but the camera could not confirm the result: look at the saved \"final\" image"
    return "the arm is waiting (torque on); `fd scene` shows the table as it is now, then try again or `fd park`"


def _sim_object(arm):
    """Simulation only: make the simulated jaws stall where the planner expects the object, so
    that a simulated pick runs through the whole sequence."""
    stall = {"raw": None}
    arm.bus.blocker = lambda pos: stall["raw"] is not None and pos["gripper"] < stall["raw"]
    real = arm.grasp_fast

    def grasp_fast(expected_raw=None, **kw):
        stall["raw"] = expected_raw
        return real(expected_raw=expected_raw, **kw)

    arm.grasp_fast = grasp_fast


def _advice(r: dict) -> str:
    """One line for the operator: what to do after a failed pick (saves a round of thinking)."""
    stage, state = r.get("stage"), r.get("arm_state")
    reason = r.get("reason", "")
    if stage == "start":
        return ("nothing moved. Use a validated speed tier (omit --tier), or validate this one on the real arm with "
                "fd bench, --tier and its name, --accept")
    if stage == "perceive" and "objects match" in reason:
        return "several objects fit the description: pick one with fd pick --id N (N: its id in \"scene\"), or add --all"
    if stage == "perceive":
        return ("check \"scene\" for what the camera sees; pick by id (fd pick --id N, N: the id in \"scene\") or look at "
                "the saved scene image")
    if stage == "plan":
        return "no safe grasp was found (see \"grasp_candidates\"); the object may be out of reach or boxed in"
    if stage == "visual":
        return ("the arm went back to the ready pose. The jaw position could not be confirmed by the camera for a grasp "
                "with little room: move the neighbouring objects apart (or the object away from the box), or repeat "
                "with `--visual off` to go by the joint angles alone (it may touch the neighbour)")
    if state == "holding_object":
        return ("the arm stopped while HOLDING the object: inspect with `fd state` / `fd snap`, then move it by hand "
                "with the low-level commands (fd tip and its x y z, `fd grip 0.5`) or ask a person; do not use `park` yet")
    if state == "stopped":
        return "the arm stopped away from the stand-by pose: run `fd state`, then `fd ready` if the way is clear"
    if stage == "verify":
        return "the motion finished but the camera could not confirm the result: look at the saved \"final\" image"
    if state in ("parked", "limp"):
        return "nothing moved (torque is off); `fd scene` shows what the camera sees"
    return "the arm is waiting (torque on); `fd scene` shows the table as it is now, then try again or `fd park`"


def cmd_do(arm, a):
    """Run a plain-language request such as 「青いキューブを箱に入れて」 without any LLM: a small
    rule-based reader (nlq.py) turns it into a `pick` or a `take`. Only a plain positive command is
    run: if the request cannot be read with confidence, or is something else (「〜しないで」, a
    question, a condition, a request with a second part), nothing moves and "need_llm" is returned."""
    import nlq
    r = nlq.parse(a.text)
    if not r["ok"]:
        # not readable by rules: hand the decision to the operator together with what the camera
        # sees, so that one more call (`fd pick --id N ...`) is enough
        # (in words, without backticks: the ids are the operator's to choose from "objects", and `fd pick --id N`, typed
        # as it was printed, is refused)
        res = {"ok": False, "need_llm": True, "reason": r["reason"], "text": a.text,
               "next": "choose the object yourself from \"objects\" and call fd pick --id N with its id "
                       "(add --into-id and the container's id if there are several containers)"}
        if r.get("action") == "take_out":
            res["next"] = ("choose the object yourself from \"objects\" (those with \"in_container\") and call "
                           "fd take --id N with its id (it is put on a free spot of the table; --to x,y names the spot)")
        if "one container into another" in r["reason"]:
            res["next"] = ("two containers are named: that is two steps for the operator. fd take --id N takes the "
                           "object out of the first one (onto the table), then fd pick --id M --into-id K puts it "
                           "into the second (N, M, K: ids in \"objects\"; they are counted anew after every command)")
        if r.get("not_plain"):
            # a prohibition, an exclusion, a condition, a question, a statement, a request with a second part:
            # the object and container named in it say nothing about what is wanted
            res["not_plain"] = r["not_plain"]
            res["next"] = ("nothing was moved: the sentence is not a plain command (" + str(r["not_plain"]) + "). Read it "
                           "yourself. If it asks for no motion, answer the user; if it does, choose the object from "
                           "\"objects\" and use fd pick --id N / fd take --id N (N: its id) for exactly the part that is "
                           "wanted")
        try:
            sc = cmd_scene(arm, a)
            res.update(objects=sc["objects"], warnings=sc["warnings"], view=sc["view"])
        except Exception as e:  # the scene is a convenience here
            res["scene_error"] = f"{type(e).__name__}: {e}"
        return res
    blocked = _tool_refusal(arm, "do", a)  # (a sentence that is not run at all was answered above)
    if blocked:
        blocked["request"] = {"text": a.text, "action": r.get("action", "put_in"), "targets": r["targets"]}
        return blocked
    con = r["container"] or {}
    a.id, a.nearest, a.largest = None, None, False
    take = r.get("action") == "take_out"
    if take:  # 「箱の中の〜を取り出して」: the container is where it comes FROM
        a.from_color, a.from_id, a.to = con.get("color"), None, None
        # a cap, a cup, a lid: hollow and open at the top unless the camera says otherwise (its wall may be pinched)
        a.hollow, a.solid, a.assume_hollow, a.probe = False, False, bool(r.get("hollow")), None
        run = cmd_take
    else:
        a.into_color, a.into_id = con.get("color"), None
        run = cmd_pick
    a.all = bool(r["all"])
    end, notes, runs = a.end, [], []
    for i, tgt in enumerate(r["targets"]):  # several colours in one sentence = several picks, in that order
        a.color, a.kind = tgt.get("color"), tgt.get("kind")
        a.end = end if i == len(r["targets"]) - 1 else "ready"
        res = run(arm, a)
        if (not res["ok"] and res.get("stage") == "perceive" and "target not found" in res.get("reason", "")
                and a.kind and a.color):
            # the shape word may not match how the camera classified the object: retry on colour alone
            a.kind = None
            res = run(arm, a)
            notes.append(f"{a.color}: shape word ignored, matched on colour only")
        runs.append(res)
        if not res["ok"]:
            break
    a.end = end
    res = runs[-1]
    if len(r["targets"]) > 1:
        res["picked"] = [{"target": x.get("target"), "ok": x["ok"], "total_s": x.get("timings", {}).get("total")}
                         for x in runs]
        res["ok"] = len(runs) == len(r["targets"]) and all(x["ok"] for x in runs)
    res["request"] = {"text": a.text, "action": r.get("action", "put_in"), "targets": r["targets"], "container": con,
                      "all": a.all}
    if notes:
        res["request"]["notes"] = notes
    res["say"] = _say(res)
    return res


def cmd_repl(arm, a):
    """Interactive session: type one request per line (「青いキューブを箱に入れて」). The camera and
    the servo bus stay open, so there is no start-up time, and no LLM is involved at all.
    Words: scene (見る) / ready (待機) / park (収納) / quit (終了; parks the arm first)."""
    blocked = _tool_refusal(arm, "repl", a)  # (its requests pick or take, and it parks on quit)
    if blocked:
        return blocked
    words = {"scene": "scene", "見る": "scene", "みる": "scene", "ready": "ready", "待機": "ready",
             "park": "park", "収納": "park", "quit": "quit", "exit": "quit", "q": "quit", "終了": "quit", "おわり": "quit"}
    base = (["--sim"] if a.sim else [])
    print("fd repl: 依頼を1行で入力してください（例: 青いキューブを箱に入れて / scene / park / quit）", flush=True)
    n_ok = n = 0
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            word = words.get(line.lower())
            if word == "quit":
                break
            t0 = time.time()
            try:
                if word in ("scene", "ready", "park"):
                    ns = build_parser().parse_args(base + [word])
                    ns._camera = a._camera
                    res = COMMANDS[word][0](arm, ns)
                    res = {"ok": True, **res} if "ok" not in res else res
                    if word == "scene":
                        for o in res["objects"]:
                            where = f"（箱 {o['in_container']} の中）" if o["in_container"] is not None else ""
                            print(f"  [{o['id']}] {o['color']} {o['kind']}  位置 ({o['center'][0]:.3f}, {o['center'][1]:.3f})"
                                  f"  寸法 {o['size_mm']} mm{where}")
                    else:
                        print("  待機姿勢です（トルクON）。" if word == "ready" else "  収納しました（トルクOFF）。")
                else:
                    ns = build_parser().parse_args(base + ["do", line] + (["--tier", a.tier] if a.tier else [])
                                                   + (["--visual", a.visual] if a.visual else []))
                    ns._camera = a._camera
                    res = cmd_do(arm, ns)
                    n += 1
                    n_ok += bool(res.get("ok"))
                    print("  " + (res.get("say") or f"読み取れませんでした: {res.get('reason')}"), flush=True)
                    if not res.get("ok") and res.get("next"):
                        print("  -> " + res["next"])
            except (SafetyError, MotionAborted, BusError) as e:
                res = {"ok": False, "error": type(e).__name__, "message": str(e)}
                print(f"  エラー: {type(e).__name__}: {e}")
            res["wall_s"] = round(time.time() - t0, 2)
            log_event("repl:" + (word or "do"), {"line": line}, res, t0)
    except KeyboardInterrupt:
        print("\n中断しました。アームはその場で保持しています。", flush=True)
        return {"ok": False, "interrupted": True, "requests": n, "succeeded": n_ok,
                "next": "run `fd state`, then `fd park` when the way is clear"}
    parked = False
    if arm.is_torqued() and not a.no_park:
        import skills
        print("収納します…", flush=True)
        skills.park(arm, arm.speeds(a.tier))
        parked = True
    _camera_to_auto(a)
    return {"requests": n, "succeeded": n_ok, "parked": parked}


_JA_COLOR = {"red": "赤い", "blue": "青い", "green": "緑の", "yellow": "黄色い", "white": "白い", "black": "黒い",
             "ivory": "アイボリーの", "orange": "オレンジの", "purple": "紫の", "pink": "ピンクの", "gray": "グレーの"}
_JA_KIND = {"box": "ブロック", "cylinder": "円柱", "container": "箱", "unknown": "物体"}


def _say(res: dict) -> str:
    """One Japanese sentence the operator can pass on as it is."""
    def name(t):
        t = t or {}
        return _JA_COLOR.get(t.get("color"), "") + _JA_KIND.get(t.get("kind"), "物体")

    take = res.get("skill") == "take_out"
    if res.get("dry") or res.get("stage") == "planned":
        if take:
            return f"{name(res.get('target'))}を箱から取り出す計画ができました（アームは動かしていません）。"
        return f"{name(res.get('target'))}をつかむ計画ができました（アームは動かしていません）。"
    done = [p for p in res.get("picked", []) if p.get("ok")] if res.get("picked") else None
    secs = res.get("timings", {}).get("total")
    if res.get("ok"):
        if take:
            if done is not None:
                return f"{len(done)}個を箱から取り出して机に置きました。"
            return f"{name(res.get('target'))}を箱から取り出して机に置きました" + (f"（{secs:.0f}秒）。" if secs else "。")
        if done is not None:
            return f"{len(done)}個を箱に入れました。"
        return f"{name(res.get('target'))}を箱に入れました" + (f"（{secs:.0f}秒）。" if secs else "。")
    state = {"ready": "アームは待機姿勢です", "standby": "アームは箱の上で待機しています", "parked": "アームは収納姿勢です",
             "stopped": "アームは途中で停止しています",
             "holding_object": "アームは物体をつかんだまま停止しています"}.get(res.get("arm_state"), "")
    head = (f"{len(done)}個は取り出しましたが、" if take else f"{len(done)}個は入れましたが、") if done else ""
    return f"{head}完了できませんでした（{res.get('stage')}: {res.get('reason')}）。{state}。"


def cmd_ready(arm, a):
    """Torque on (if needed) and go to the stand-by pose. With a tool attached: the TOOL tip goes to the
    stand-by point, lifted first when it is low and never lower than 2 cm above the table on the way; what
    keeps it from going is answered as a complete result (stage "plan": nothing moved; "move": it stopped on
    the way and holds), and the result of a move says what was done about a jaw that is held shut ("jaw")."""
    import skills
    tool = getattr(arm, "tool", None)
    q0 = arm.read_q() if tool is not None else None
    try:
        r = skills.go_ready(arm, a.tier)
    except (SafetyError, MotionAborted) as e:
        if tool is None:
            raise                   # (without a tool: as it always was)
        # With a tool the way to the stand-by pose is planned and checked before the torque goes on, and a refusal
        # is the normal answer to a record that does not fit the arm (joint limits that leave no stand-by pose, a
        # held jaw that stands elsewhere than the tool file says): a complete result, as `fd tool rest` gives.
        why = str(e)
        moved = bool(float(np.max(np.abs(arm.read_q() - q0))) > math.radians(0.5))
        torque = "on" if arm.is_torqued() else "off"
        fold = _fold_words(a, tool)      # (`fd paint rest` for a painting tool beside a painting set-up)
        if moved:
            nxt = ("the arm stopped on the way and holds where it is (torque on): `fd state`, then `fd ready` again, or "
                   f"`{fold}`")
            say = f"待機姿勢への移動が途中で止まりました（{why}）。アームはその場で保持しています。"
        else:
            if "fd tool jaw" in why:
                # The way out depends on the tool, and the refusal names it for THAT tool in its last sentence
                # (tooling._jaw_elsewhere: "... Next: ..."). Until round 2 this named `fd tool jaw close` for every jaw
                # refusal: for a tool whose file does not say the jaw (the rig's own file of 2026-11-13, a pen attached
                # with `fd tool attach`) that command is refused, and the right word stood only in the reason.
                way = why.partition("Next: ")[2].strip()
                nxt = "nothing moved. " + (way[0].upper() + way[1:] if way else "`fd tool jaw status` shows where the jaw "
                                           "stands and what the tool file records")
            elif "--joint-limit" in why:
                # (words, no `fd tool set --joint-limit JOINT=LO:HI`: which joint and which range is the operator's to say,
                # and typed as it was printed that command is refused)
                nxt = ("nothing moved. `fd tool status` shows the tool's joint limits; fd tool set with --joint-limit changes "
                       f"them ({JOINT_LIMIT_WORDS}); `{fold}` folds the arm away meanwhile")
            else:
                nxt = f"nothing moved. `fd tool status` shows the tool, `fd state` the arm; `{fold}` folds the arm away"
            say = f"待機姿勢へ移動できませんでした（{why}）。アームは動かしていません。"
        return {"ok": False, "error": type(e).__name__, "stage": "move" if moved else "plan", "reason": why, "message": why,
                "tool": tool.name, "moved": moved, "torque": torque, "next": nxt, "say": say, **brief_state(arm)}
    res = {"joint_err_deg": r["joint_err_deg"], "duration_s": r.get("duration_s"), **brief_state(arm)}
    if r.get("tool"):
        # "jaw": what was done about the squeeze of a tool held on the moving jaw; "lift_via": how a low tool was raised
        res.update({k: r[k] for k in ("tool", "tool_tip_xyz", "standby_xyz", "standby_pitch_deg", "standby_moved_m",
                                      "via", "lifted_first", "lift_via", "min_tip_height_m", "jaw") if k in r})
    return res


def cmd_park(arm, a):
    """Return to the folded rest pose, switch torque off, put the camera back to auto."""
    blocked = _tool_refusal(arm, "park", a)
    if blocked:
        return blocked
    import skills
    already = not arm.is_torqued()
    if not already:
        skills.park(arm, arm.speeds(a.tier))
    _camera_to_auto(a)
    out_ = {"parked": True, **brief_state(arm)}
    if already:
        folded = skills._is_parked(arm.read_q())
        out_.update(parked=bool(folded), note="torque was already off; nothing moved" if folded else
                    "torque is off but the arm is NOT in its folded pose: support it by hand, then "
                    "`fd torque on` and `fd park`")
    return out_


# ------------------------------------------------------------------ tool on the gripper
# (words, not a command in backticks: the name and the tip are the operator's to give, and `fd tool attach --name N
# --tip x,y,z`, typed as it was printed, is refused. The one command that works as it is typed lists the words)
ATTACH_NEXT = ("fd tool attach with --name and the tip registers it: --tip and its x,y,z in metres, or for a straight "
               "tool --along-fingers / --across-fingers and its length beyond the fingertip (`fd tool attach --help` lists "
               "the words)")


def _xyz_arg(text, what: str, n: int = 3) -> list:
    try:
        v = [float(x) for x in str(text).split(",")]
    except ValueError:
        v = []
    if len(v) != n or not all(math.isfinite(x) for x in v):
        raise SafetyError(f"{what} needs {n} numbers separated by commas (metres), e.g. "
                          + ("0.0015,-0.0081,0.0621" if n == 3 else "0.17,-0.06,0.31,0.07"))
    return v


def _tool_brief(tool, cfg) -> dict:
    """What the operator needs to see of a tool (the measured map and plane only as their quality numbers)."""
    import tooling
    d = {"name": tool.name, "tip_offset": [round(float(v), 5) for v in tool.tip_offset], "mount": tool.mount,
         "length_mm": round(tooling.length_of(cfg, tool) * 1000.0, 1), "pitch_deg": tool.pitch_deg,
         "gripper_raw": tool.gripper_raw, "attached_at": tool.attached_at, "region": tool.region,
         "calibrated": tooling.calibration_state(tool), "notes": list(tool.notes)}
    if tool.joint_limits_deg:
        d["joint_limits_deg"] = tool.joint_limits_deg
    # a tool taped to a jaw (2026-11-13): on which jaw, the goal that holds the jaw shut with a squeeze, and whether
    # the tip was MEASURED (`fd calib-air`) or is a first guess / an estimate: that evening a guess of 8 cm was 9 cm short
    for k in ("on_jaw", "gripper_goal_raw"):
        if getattr(tool, k, None) is not None:
            d[k] = getattr(tool, k)
    d["tip_measured"] = bool(getattr(tool, "tip_measured", None))
    if tool.contact:
        d["contact"] = {k: tool.contact.get(k) for k in ("a", "bx", "by", "rms_mm") if k in tool.contact}
    if tool.corr_map:
        d["corr_map"] = {k: tool.corr_map.get(k) for k in ("kind", "rms_mm", "max_mm", "n") if k in tool.corr_map}
    return d


def _tool_spec(a, cfg):
    """(tip_offset, mount) asked for on the command line (attach / set), or (None, None) when no tip option
    was given. --along-fingers / --across-fingers LEN: a straight tool that sticks out LEN beyond the
    fingertip; --tip: the tip measured by hand, in gripper_frame_link axes."""
    import tooling
    tip = mount = None
    if getattr(a, "along_fingers", None) is not None:
        tip, mount = tooling.offset_for(cfg, "along_fingers", a.along_fingers), "along_fingers"
    elif getattr(a, "across_fingers", None) is not None:
        tip, mount = tooling.offset_for(cfg, "across_fingers", a.across_fingers), "across_fingers"
    elif getattr(a, "tip", None) is not None:
        tip = tooling.check_offset(_xyz_arg(a.tip, "--tip"))
        mount = tooling.infer_mount(cfg, tip)
    if getattr(a, "mount", None):
        mount = a.mount
    return tip, mount


JOINT_LIMIT_WORDS = ("JOINT=LO:HI in degrees (e.g. wrist_flex=20:97), JOINT=none (remove the limit of that joint) or "
                     "none (remove them all)")


def _tool_joint_limits(a, current=None):
    """The tool's joint limits after the words of --joint-limit (repeatable, read in the order given):
    `JOINT=LO:HI` sets the limit of that joint and leaves the other joints as they are, `JOINT=none` removes the
    limit of one joint, `none` removes them all. -> {joint: [lo, hi]} ({} = no limit), or None when the option
    was not given. `current`: the limits the tool has now (None for a new tool). SafetyError for a word that is
    none of the three, an unknown joint, a range that is not lo < hi.

    Until 2026-11-13 the option REPLACED the whole set: `fd tool set --joint-limit wrist_flex=20:95`, the one-joint
    form every refusal recommends, silently removed the roll limit of a brush whose handle hits the forearm when the
    gripper rolls."""
    import tooling
    given = getattr(a, "joint_limit", None)
    if not given:
        return None
    out = {j: [float(v[0]), float(v[1])] for j, v in (current or {}).items()}
    for text in given:
        word = str(text).strip()
        if word.lower() == "none":
            out = {}
            continue
        j, eq, rng = word.partition("=")
        j, rng = j.strip(), rng.strip()
        if not eq or not j or not rng:
            raise SafetyError(f"--joint-limit {text!r}: {JOINT_LIMIT_WORDS}")
        if rng.lower() == "none":
            if j not in ARM:
                raise SafetyError(f"--joint-limit {text!r}: unknown joint {j!r} (one of {', '.join(ARM)})")
            out.pop(j, None)
            continue
        try:
            lo, hi = (float(x) for x in rng.split(":"))
        except ValueError:
            raise SafetyError(f"--joint-limit {text!r}: {JOINT_LIMIT_WORDS}") from None
        out.update(tooling.check_joint_limits({j: [lo, hi]}) or {})
    return out


def _joint_limits_problem(tool, arm=None) -> str | None:
    """Why the arm cannot apply the joint limits of `tool`, or None when it can: Arm.q_limits() refuses a range
    that leaves nothing of what the arm itself allows for that joint (2026-11-13: `fd tool set --joint-limit
    wrist_flex=200:300` was stored with ok true, and every command that moves refused from then on). Asked of the
    servo limits of an opened `arm` when one is given, else of the simulated arm's, which are the servo limits of
    this arm type; nothing is sent to a servo bus either way."""
    if not getattr(tool, "joint_limits_deg", None):
        return None
    if arm is None:
        from sim_bus import SimBus
        probe = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=tool)
        probe.raw_limits = probe.bus.limits()
    else:
        probe = Arm(bus=arm.bus, cfg=arm.cfg, tool=tool)
        probe.raw_limits = arm.raw_limits
    try:
        probe.q_limits()
    except SafetyError as e:
        return str(e)
    return None


def _limits_text(limits) -> str:
    """Joint limits as one line: "wrist_flex 20..97, wrist_roll -5..5 deg" ("none" when there are none)."""
    if not limits:
        return "none"
    return ", ".join(f"{j} {lo:g}..{hi:g}" for j, (lo, hi) in limits.items()) + " deg"


def _joint_limit_refused(why: str, name=None, limits=None) -> dict:
    """The complete refusal for --joint-limit words that cannot be used, or limits the arm cannot apply:
    nothing was stored. `limits`: what the tool file keeps (None: there is no tool file yet)."""
    res = {"ok": False, "error": "SafetyError", "stage": "options", "reason": why, "message": why, "saved": False,
           "next": "fd tool set with --joint-limit JOINT=LO:HI sets the limit of one joint in degrees and keeps the others "
                   "(e.g. wrist_flex=20:97; repeatable), JOINT=none removes one, none all; `fd tool status` shows what is "
                   "stored",
           "say": f"--joint-limit の指定を受け付けられませんでした（{why}）。ツールの記録は変えていません。"}
    if name:
        res["tool"] = name
    if limits is not None:
        res["joint_limits_deg"] = limits
        res["say"] += f"いまの関節制限: {_limits_text(limits) if limits else 'なし'}。"
    return res


def _tool_arm_facts(a, tool, notes: list) -> dict:
    """Read-only look at the arm for `tool attach` / `tool set`: the gripper position, where this tool's tip
    is now, whether the arm has a rest and a stand-by pose with it, and ("limits_error") why this arm cannot
    apply the tool's joint limits, if it cannot. The servo bus may be unavailable (arm off, somebody
    teleoperating): the tool is recorded all the same, since that only adds refusals."""
    import tooling
    if getattr(a, "no_robot", False):
        return {}
    try:
        with make_arm(a.sim) as arm:
            facts = {"gripper_raw": int(arm.bus.read("gripper", "Present_Position"))}
            why = _joint_limits_problem(tool, arm)       # (with the servo limits THIS arm has stored)
            if why:
                facts["limits_error"] = why
            q = arm.read_q()
            h = tooling.tip_height(arm, q, tool)
            facts["tip_height_mm"] = round(h * 1000.0, 1)
            if h < -0.005 and not a.sim:
                notes.append(f"in the pose the arm is in now, this tip would be {-h * 1000:.0f} mm BELOW the table: "
                             "the tip offset is wrong, or the arm is not where its encoders say. Check before moving")
            for key, fn in (("rest", lambda: tooling.rest_q(arm, tool)), ("standby", lambda: tooling.ready_pose(arm, tool)["q"])):
                try:
                    qq = fn()
                    facts[key] = {"q_deg": [round(math.degrees(float(v)), 1) for v in qq],
                                  "tip_height_mm": round(tooling.tip_height(arm, qq, tool) * 1000.0, 1)}
                except SafetyError as e:
                    facts[key] = {"error": str(e)}
                    notes.append(f"no {key} pose with this tool: {e}")
            return facts
    except Exception as e:  # the look is optional
        notes.append(f"the arm could not be read ({type(e).__name__}: {e}); the gripper position was not recorded")
        return {}


def _tool_attach(a):
    """Record what is taped to the gripper. From now on "tip" is the tool tip and the gripper is locked."""
    import tooling
    if not a.sim:
        require_rig_config()      # the tool tip is the rig's fingertip plus the tool: never from built-in numbers
    cfg = load_config()
    path = tooling.tool_path(a.sim)
    was = tooling.sim_tool_state() if a.sim else None
    # "already attached": the file this command writes is the one the tool is READ from. A simulated tool file that
    # is left behind (a copy of the real tool as it was before the real tool was changed or replaced:
    # tooling._stale_copy) is not read any more: it does not stand in the way, the new simulated tool is written
    # over it. (Until 2026-11-14 the file's mere existence counted, and `fd --sim tool attach` answered "tool
    # 'brush_thin' is already attached" with the name of a copy the simulation no longer used.)
    if tooling.source(a.sim) == path and not a.replace:
        try:
            name = tooling.load(path).name
        except SafetyError:
            name = "?"
        why = f"tool '{name}' is already attached"
        return _sim_tool_said(
            {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why, "tool": name,
             "next": f"`{_fd(a)} tool set --help` lists what fd tool set changes of it; `{_fd(a)} tool detach` (after taking "
                     "it off) removes it; the same fd tool attach command with --replace replaces it (its calibration "
                     "is lost)",
             "say": f"すでにツール「{name}」が登録されています。変更は fd tool set、付け替えは --replace を使ってください。"}, a, was=was)
    tip, mount = _tool_spec(a, cfg)
    if tip is None:
        raise SafetyError("tool attach needs the tip: --tip x,y,z | --along-fingers LEN | --across-fingers LEN (metres)")
    name = str(a.name).strip()
    if not name:
        raise SafetyError("tool attach needs a name (--name pen)")
    pitch = tooling.check_pitch(a.pitch if a.pitch is not None else tooling.DEFAULT_PITCH[mount])
    try:
        limits = _tool_joint_limits(a)
    except SafetyError as e:
        return _joint_limit_refused(str(e), name)
    tool = tooling.Tool(name=name, tip_offset=tip, mount=mount, pitch_deg=pitch,
                        attached_at=time.strftime("%Y-%m-%dT%H:%M:%S"), notes=[str(n) for n in (a.note or [])],
                        joint_limits_deg=limits or None)
    why = _joint_limits_problem(tool)       # limits the arm cannot apply are not recorded: every move would refuse
    if why:
        return _joint_limit_refused(why, name)
    notes = []
    facts = _tool_arm_facts(a, tool, notes)
    if facts.get("limits_error"):
        return _joint_limit_refused(facts["limits_error"], name)
    tool.gripper_raw = facts.pop("gripper_raw", None)
    if not a.dry:
        tooling.save(tool, a.sim)
    res = {"ok": True, "stage": "planned" if a.dry else "attached", "tool": _tool_brief(tool, cfg), "file": path,
           "saved": not a.dry, "arm": facts, "notes": notes,
           "locked": ["grip", "grasp", "pick", "take", "do", "repl", "probe", "bench", "park", "calib-at", "servo-tip",
                      "torque off (outside the tool's rest pose)"],
           "next": "`fd ready` puts the TOOL tip at the stand-by point; `fd tool calibrate` measures the tip on paper "
                   "(needed before drawing with fd draw); `fd tool rest` folds the arm away with the tool on",
           "say": f"ツール「{name}」を登録しました（先端はグリッパーから {tooling.length_of(cfg, tool) * 1000:.0f} mm）。"
                  "グリッパーはロックされ、park の代わりに fd tool rest を使います。"}
    if a.dry:
        res["say"] = f"ツール「{name}」を登録する内容を確認しました（保存していません）。"
    paints = _painting_tool(tool)
    if paints:      # (a newly attached tool: a painting set-up is on record. Never sent to the pen calibration)
        res["painting_tool"] = paints[1]
        res["next"] = (f"{paints[1]}: `{_fd(a)} paint status` says what a brush needs before it paints (the jaw, the tip, "
                       f"which brush, the paper height) and names the next command; `{_fd(a)} paint rest` folds the arm away "
                       "with the tool on, off the canvas")
        res["say"] = (f"ツール「{name}」を登録する内容を確認しました（保存していません）。" if a.dry else
                      f"ツール「{name}」を登録しました（先端はグリッパーから {tooling.length_of(cfg, tool) * 1000:.0f} mm）。"
                      "グリッパーはロックされました。") + ("絵のセットアップが登録されているので、筆の準備は fd paint status の next に"
                                              "従ってください。片付けは fd paint rest を使います。")
    return _sim_tool_said(res, a, wrote=not a.dry, was=was)


def _tool_status(a):
    """What the harness believes is on the gripper. Reads the tool file only (never the servo bus)."""
    import tooling
    cfg = load_config()
    try:
        tool = tooling.active(a.sim)
    except SafetyError as e:
        return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": str(e), "message": str(e),
                "attached": None, "file": tooling.source(a.sim),
                "next": "repair the file, or take the tool off by hand and run `fd tool detach`",
                "say": "ツールの記録ファイルが読めません。グリッパーに何が付いているか分からないため、動作は拒否されます。"}
    if tool is None:
        return _sim_tool_said(
            {"ok": True, "stage": "status", "attached": False, "tool": None, "file": tooling.tool_path(a.sim),
             "next": "nothing is attached: every command works as usual. With something taped to the gripper: "
                     + ATTACH_NEXT,
             "say": "ツールは登録されていません（通常どおり動きます）。"}, a)
    cal = tooling.calibration_state(tool)
    res = {"ok": True, "stage": "status", "attached": True, "tool": _tool_brief(tool, cfg), "file": tooling.source(a.sim),
           "locked": ["grip", "grasp", "pick", "take", "do", "repl", "probe", "bench", "park", "calib-at", "servo-tip",
                      "torque off (outside the tool's rest pose)"]}
    paints = _painting_tool(tool)
    if paints:
        res["painting_tool"] = paints[1]
    if all(cal.values()):
        # (words, no `fd draw FILE`: which drawing is the operator's to say; the fold is `fd paint rest` beside a
        # painting set-up, where `fd tool rest` is refused over the canvas: _fold_words)
        res["next"] = ("calibrated: fd draw with the name or the file of a drawing draws it (`fd draw cat --dry` shows the "
                       f"layout of the bundled one); `fd ready` stands by; `{_fold_words(a, tool)}` folds away")
        res["say"] = f"ツール「{tool.name}」が付いています（校正済み）。"
    elif paints:
        # A brush is never sent to the pen calibration (_painting_tool): what it needs is what `fd paint status` lists
        fold = "paint rest" if _paint_setup_on_record() else "tool rest"       # (`fd paint rest` needs the set-up)
        res["next"] = (f"a painting tool ({paints[1]}): `{_fd(a)} paint status` says what it still needs before it paints "
                       f"and names the next command; `{_fd(a)} ready` stands by; `{_fd(a)} {fold}` folds away")
        res["say"] = (f"ツール「{tool.name}」が付いています（絵の具用の道具として扱います: {PAINTING_JA[paints[0]]}。筆先は"
                      f"{'測定済み' if getattr(tool, 'tip_measured', None) else '未測定'}）。描く前に足りないものと次のコマンドは "
                      "fd paint status が示します。")
    else:
        missing = ", ".join(k for k, v in cal.items() if not v)
        res["next"] = (f"not calibrated ({missing}): run `fd tool calibrate` before drawing (fd draw); `fd ready` stands by; "
                       "`fd tool rest` folds away")
        res["say"] = f"ツール「{tool.name}」が付いています（未校正: {missing}）。描く前に fd tool calibrate が必要です。"
    # --sim: whether the simulation reads a tool file of its own ("simulated_tool"), and a copy that was left behind
    return _sim_tool_said(res, a)


def _tool_set(a):
    """Change what is recorded about the attached tool. A new tip, mount or working pitch makes the
    measured contact plane and correction map worthless: they are dropped unless --keep-calibration.
    --joint-limit changes the joints it names and keeps the limits of the others (_tool_joint_limits); limits the
    arm cannot apply are refused before anything is stored, and the result lists the limits that remain."""
    import tooling
    if not a.sim:
        require_rig_config()      # (as in _tool_attach)
    cfg = load_config()
    tool = tooling.active(a.sim)
    if tool is None:
        why = "no tool is attached"
        return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why,
                "next": ATTACH_NEXT,
                "say": "ツールが登録されていません。先に fd tool attach を実行してください。"}
    before = _tool_brief(tool, cfg)
    was = tooling.sim_tool_state() if a.sim else None
    paints = _painting_tool(tool)       # (asked before the change: a new tip takes the record of a measured tip away)
    changed, geometry = [], False
    tip, mount = _tool_spec(a, cfg)
    if tip is not None and [round(float(v), 5) for v in tip] != [round(float(v), 5) for v in tool.tip_offset]:
        tool.tip_offset, geometry = tip, True
        changed.append("tip_offset")
    if mount is not None and mount != tool.mount:
        tool.mount, geometry = mount, True
        changed.append("mount")
    if a.pitch is not None and tooling.check_pitch(a.pitch) != tool.pitch_deg:
        tool.pitch_deg, geometry = tooling.check_pitch(a.pitch), True
        changed.append("pitch_deg")
    if a.name and str(a.name).strip() != tool.name:
        tool.name = str(a.name).strip()
        changed.append("name")
    kept = dict(tool.joint_limits_deg or {})         # what the file has; what a refusal below leaves in it
    try:
        limits = _tool_joint_limits(a, kept)         # merged joint by joint with what the tool has
    except SafetyError as e:
        return _joint_limit_refused(str(e), tool.name, kept)
    new_limits = limits is not None and limits != kept
    if new_limits:
        tool.joint_limits_deg = limits or None      # (not geometry: the measured contact plane and map stay)
        changed.append("joint_limits_deg")
        why = _joint_limits_problem(tool)           # before anything is stored
        if why:
            return _joint_limit_refused(why, tool.name, kept)
    if a.region is not None:
        region = None if str(a.region).lower() in ("none", "") else _xyz_arg(a.region, "--region", 4)
        if region != tool.region:
            tool.region = region
            changed.append("region")
    for n in a.note or []:
        tool.notes.append(str(n))
        changed.append("notes")
    dropped = False
    if geometry and not a.keep_calibration and (tool.contact or tool.corr_map or tool.calibrated):
        tool.contact, tool.corr_map, tool.calibrated, dropped = None, None, {}, True
    notes = []
    # a look at the arm with the tool as it will be: new limits can leave it without a rest or a stand-by pose
    facts = _tool_arm_facts(a, tool, notes) if (geometry or new_limits) else {}
    if new_limits and facts.get("limits_error"):
        return _joint_limit_refused(facts["limits_error"], tool.name, kept)
    facts.pop("gripper_raw", None)  # the position at ATTACH time stays the reference
    if changed and not a.dry:
        tooling.save(tool, a.sim)
    now = dict(tool.joint_limits_deg or {})
    paints = paints or _painting_tool(tool)
    if paints:      # a brush is never sent to the pen calibration (_painting_tool)
        nxt = (("the measured contact height and correction map (the pen calibration) were dropped. " if dropped else "")
               + f"`{_fd(a)} paint status` says what this painting tool still needs before it paints; `{_fd(a)} tool status` "
                 "shows the tool")
        again_ja = ("ペン用の校正の記録は消しました。" if dropped else "") + "筆の準備の状況は fd paint status で確認できます。"
    else:
        nxt = ("the measured contact height and correction map were dropped: run `fd tool calibrate` before drawing (fd draw)"
               if dropped else "`fd tool status` shows the tool")
        again_ja = "校正はやり直しが必要です。" if dropped else ""
    res = {"ok": True, "stage": "planned" if a.dry else "set", "changed": changed, "saved": bool(changed) and not a.dry,
           "calibration_dropped": dropped, "tool": _tool_brief(tool, cfg), "before": before, "arm": facts,
           "joint_limits_deg": now, "notes": notes, "file": tooling.tool_path(a.sim), "next": nxt,
           "say": ((f"ツール「{tool.name}」の設定を変更する内容を確認しました（{', '.join(changed)}。保存していません）。" if a.dry else
                    f"ツール「{tool.name}」の設定を変更しました（{', '.join(changed)}）。") if changed else
                   f"ツール「{tool.name}」の設定は変わっていません。") + again_ja}
    if paints:
        res["painting_tool"] = paints[1]
    if limits is not None:
        # what remains is said in words: an operator who corrects one joint must see that the others are still there
        res["say"] += f"関節制限: {_limits_text(now)}。" if now else "関節制限はありません。"
    no_pose = [k for k in ("rest", "standby") if "error" in (facts.get(k) or {})]
    if no_pose:
        poses = " and ".join({"rest": "rest pose (`fd tool rest`)", "standby": "stand-by pose (`fd ready`)"}[k] for k in no_pose)
        res["next"] = (f"with the tool as it is recorded now the arm has no {poses}: \"notes\" say why; fd tool set "
                       "changes the record again. " + res["next"][0].upper() + res["next"][1:])
        res["say"] += ("注意: この記録では" + "・".join({"rest": "収納姿勢", "standby": "待機姿勢"}[k] for k in no_pose)
                       + "が作れません（notes に理由）。")
    return _sim_tool_said(res, a, wrote=res["saved"], was=was)


def _rest_canvas(arm, a, q0):
    """What the painting set-up says about the folded pose `fd tool rest` would leave the arm in: None when there
    is nothing to ask (see below), else the report of paintskill.rest_report (a dict with "ok": the tool tip, and
    the place it comes down on when the wrist sinks with the torque off, lie clear of the canvas; "why") plus
    "pan_deg", "clearance_mm", "judged" and "setup". SafetyError (paintskill.Refusal) when a set-up file exists
    and cannot be used.

    Why (2026-11-13): the tool's default rest pose (pan 20) has a brush 4 cm INSIDE the sheet. In the morning the
    wrist sank there with the torque off until the brush stood on the paper; in the evening every hint after a
    stopped command still named `fd tool rest`. `fd paint rest` chooses a fold off the canvas; this is the gate
    on the other door.

    Nothing to ask, and the command is what it always was: no painting module or no rest_report in it (called
    through getattr: the function belongs to tools/paint), no set-up file, or no such fold with this tool
    (tooling.rest then refuses in its own words). Not asked either, and the result says so ("ok": None): a tool
    file named by $FD_TOOL while the set-up is the rig's own. That is a test or an experiment with a scratch tool,
    and the rig's set-up describes another table (the suites that fold a simulated pen away would otherwise pass
    or fail by what was painted on this desk last).
    An arm that already lies limp in a rest pose is left alone by tooling.rest whatever pan was asked for: the
    pose that is judged is then the one it lies in."""
    import tooling
    try:
        import paintskill
    except Exception:           # the painting feature is not there, or does not load: folding away must still work
        return None
    report, setup_path = getattr(paintskill, "rest_report", None), getattr(paintskill, "setup_path", None)
    if not callable(report) or not callable(setup_path):
        return None
    path = setup_path()
    if not os.path.isfile(path):
        return None
    if os.environ.get("FD_TOOL") and not os.environ.get("FD_PAINT_SETUP"):
        return {"ok": None, "why": "not checked against the canvas: the tool file is named by $FD_TOOL and the painting "
                                   f"set-up ({path}) is the rig's own"}
    setup = paintskill.load_setup(path)
    tool = arm.tool
    pan, clearance, judged = float(a.pan), float(a.clearance), "the pose asked for"
    if not arm.is_torqued() and tooling.in_rest_pose(arm, tool, q0)[0]:
        pan = round(math.degrees(float(q0[0])), 1)
        clearance = round(min(0.15, max(0.03, tooling.tip_height(arm, q0, tool))), 4)      # (the range a rest pose has)
        judged = "where the arm lies folded, torque off"
    try:
        tooling.rest_q(arm, tool, clearance, pan)
        rep = report(arm, tool, setup, pan, clearance)
    except SafetyError:
        return None
    if not isinstance(rep, dict):
        return None
    return {**rep, "ok": bool(rep.get("ok")), "why": str(rep.get("why") or ""), "pan_deg": round(pan, 1),
            "clearance_mm": round(clearance * 1000.0), "judged": judged, "setup": path}


def _rest_words(a, over_canvas=None, dry=None) -> str:
    """`fd [--sim] tool rest` with the options this call was given (for a `next` that works as it is typed);
    over_canvas / dry: with or without that option, whatever the call had."""
    words = ["fd"] + (["--sim"] if a.sim else []) + ["tool", "rest"]
    if a.pan != 20.0:
        words += ["--pan", f"{a.pan:g}"]
    if a.clearance != 0.05:
        words += ["--clearance", f"{a.clearance:g}"]
    if a.tier:
        words += ["--tier", str(a.tier)]
    for flag, on in (("--keep-torque", a.keep_torque), ("--over-canvas", a.over_canvas if over_canvas is None else over_canvas),
                     ("--dry", a.dry if dry is None else dry)):
        if on:
            words.append(flag)
    return " ".join(words)


def _tool_rest(a):
    """Fold the arm away with the tool on and switch the torque off (what `park` is without a tool). With a
    painting set-up on record, a pose that has the tool over the canvas is refused unless --over-canvas says that
    it is meant (_rest_canvas): `fd paint rest` is the command that keeps a brush off the picture."""
    import tooling
    with make_arm(a.sim) as arm:
        if arm.tool is None:
            why = "no tool is attached"
            return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why,
                    "next": "`fd park` folds the arm away", "say": "ツールが登録されていません。収納は fd park です。"}
        q0 = arm.read_q()
        paint_rest = "fd --sim paint rest" if a.sim else "fd paint rest"
        canvas = blocked = None
        try:
            canvas = _rest_canvas(arm, a, q0)
        except SafetyError as e:        # a painting set-up that cannot be read: where its canvas lies is not known
            canvas = {"ok": None, "why": f"the painting set-up cannot be used ({e}): not checked"}
            if not a.over_canvas:
                why = (f"a painting set-up is on record and cannot be used ({e}): whether this folded pose has tool "
                       f"'{arm.tool.name}' over the canvas is not known. Nothing moved")
                blocked = {"stage": "setup", "reason": why,
                           "next": f"`fd paint status` says what is wrong with the set-up; `{_rest_words(a, over_canvas=True)}` "
                                   "folds the arm without that check (name a pan that has the tool off the sheet: --pan DEG)",
                           "say": "絵のセットアップのファイルが読めないため、収納姿勢がキャンバスの上かどうか確かめられません。"
                                  "収納していません（アームは動かしていません）。紙の上でないと分かっているときは "
                                  "--over-canvas を付けてください。"}
        except Exception as e:          # a fault in the painting module: the way to fold the arm away must stay open
            canvas = {"ok": None, "why": f"the painting module failed ({type(e).__name__}: {e}): not checked"}
            if not a.over_canvas:
                why = (f"the painting module failed while this folded pose was checked against its canvas "
                       f"({type(e).__name__}: {e}). Nothing moved")
                blocked = {"stage": "internal", "reason": why,
                           "next": f"`{_rest_words(a, over_canvas=True)}` folds the arm without that check (name a pan that "
                                   "has the tool off the sheet: --pan DEG)",
                           "say": "絵のセットアップとの照合中にエラーが起きたため、収納していません（アームは動かしていません）。"
                                  "紙の上でないと分かっているときは --over-canvas を付けてください。"}
        if canvas is not None and canvas["ok"] is False and not a.over_canvas:
            lies = canvas["judged"] != "the pose asked for"
            why = ((f"the arm lies folded at pan {canvas['pan_deg']:g} with the torque off (a rest pose: `fd tool rest` leaves "
                    f"it there), and there tool '{arm.tool.name}' is" if lies else
                    f"folded at pan {canvas['pan_deg']:g} with its tip {canvas['clearance_mm']:.0f} mm up, tool "
                    f"'{arm.tool.name}' would be")
                   + " over the canvas of the painting set-up"
                   + (f": {canvas['why']}" if canvas["why"] else
                      ": with the torque off the wrist can sink until the tool stands on the sheet (2026-11-13)")
                   + ". Nothing moved")
            blocked = {"stage": "canvas", "reason": why, "canvas": canvas,
                       "next": f"`{paint_rest}` folds the arm where the tool is off the canvas; when no sheet lies there any "
                               f"more, or this pose is meant: `{_rest_words(a, over_canvas=True)}`",
                       "say": (f"アームはいま、トルクOFFのまま、ツール「{arm.tool.name}」の先端がキャンバスの上に来る姿勢で畳まれています。"
                               "手首が下がると絵に触れるおそれがあります（今回は何も動かしていません）。fd paint rest を使うと、"
                               "絵にかからない位置へ畳み直します。紙がもう無いときは --over-canvas を付けてください。" if lies else
                               f"この収納姿勢では、ツール「{arm.tool.name}」の先端がキャンバスの上に来ます。トルクを切ると手首が下がって"
                               "絵に触れるおそれがあるため、収納していません（アームは動かしていません）。fd paint rest を使うと、"
                               "絵にかからない位置に畳みます。紙がもう無いときは --over-canvas を付けてください。")}
        if blocked:
            return {"ok": False, "error": "SafetyError", "message": blocked["reason"], "tool": arm.tool.name, "moved": False,
                    "torque": "on" if arm.is_torqued() else "off", **blocked, **brief_state(arm)}
        try:
            res = tooling.rest(arm, a.tier, clearance_m=a.clearance, pan_deg=a.pan, torque_off=not a.keep_torque,
                               dry=a.dry)
        except (SafetyError, MotionAborted) as e:
            # refused before anything moved (no rest pose with this tool, no clear way to it), or stopped on the way
            moved = bool(float(np.max(np.abs(arm.read_q() - q0))) > math.radians(0.5))
            res = {"ok": False, "error": type(e).__name__, "stage": "move" if moved else "plan", "reason": str(e),
                   "message": str(e), "tool": arm.tool.name, "moved": moved,
                   "torque": "on" if arm.is_torqued() else "off",
                   "next": ("the arm stopped on the way and holds where it is: `fd state`, then `fd tool rest` again or "
                            "`fd ready`" if moved else
                            "nothing moved. `fd tool status` shows the tool; if the arm cannot rest with it, support the "
                            "arm by hand, take the tool off and run `fd tool detach`")}
        res.update(brief_state(arm))
    over = a.over_canvas and canvas is not None and canvas["ok"] is not True      # folded although the check did not pass
    if canvas is not None:
        res["canvas"] = {**canvas, "overridden": True} if over else canvas
    if res.get("stage") == "planned":
        res["next"] = f"`{_rest_words(a, dry=False)}` does it"
        res["say"] = (f"ツール「{res['tool']}」を付けたまま収納する経路を確認しました（アームは動かしていません）。"
                      f"収納姿勢ではツール先端が机の {res['rest_tip_height_mm']:.0f} mm 上になります。")
    elif res["ok"]:
        res["next"] = "`fd ready` raises the arm again; after taking the tool off: `fd tool detach`"
        res["say"] = (f"ツール「{res['tool']}」を付けたまま収納しました" if res.get("moved") else
                      f"ツール「{res['tool']}」を付けたまま、すでに収納姿勢です（何も動かしていません）")
        res["say"] += ("（トルクOFF）。" if res.get("torque") == "off" else "（トルクONのまま）。")
        res["say"] += f"ツール先端は机の {res['tip_height_mm']:.0f} mm 上です。"
    elif res.get("error") and res.get("moved"):
        res["say"] = f"収納を完了できませんでした（{res.get('reason')}）。アームは途中で止まっています。"
    elif res.get("error"):
        res["say"] = f"収納できませんでした（{res.get('reason')}）。アームは動かしていません。"
    else:
        res["say"] = f"収納を完了できませんでした（{res.get('reason')}）。"
    if over and res["ok"]:
        res["say"] += ("（--over-canvas の指定により、キャンバスの上かどうかの確認を通さずに畳む姿勢です。"
                       "筆が濡れているときは fd paint rest を使ってください。）")
    return res


def _tool_detach(a):
    """Forget the tool: the gripper, `park` and the fingertip as "tip" are back. Run it only when the tool
    is really off the gripper: nothing here can see that."""
    import tooling
    path = tooling.tool_path(a.sim)
    was = tooling.sim_tool_state() if a.sim else None      # (before the file goes: a copy that was left behind is told)
    if not os.path.exists(path):
        res = {"ok": True, "stage": "detached", "attached": False, "removed": False, "file": path,
               "next": "nothing was attached: the gripper and `park` work as usual",
               "say": "ツールは登録されていませんでした。"}
        real = tooling.source(True) if a.sim else None
        if real:
            # No simulated tool file, and the simulation mirrors the real arm's tool: the simulated arm DOES carry a
            # tool. Until round 3 (2026-11-14) this answered "attached": false and 「ツールは登録されていませんでした」,
            # with the truth in a key "note" only, while `fd --sim tool status` right after showed the tool and
            # `fd --sim park` was refused because of it.
            try:
                mirrors = tooling.load(real).name
            except SafetyError:
                mirrors = "?"
            res.update(attached=True, mirrors=mirrors,
                       next=f"there is no SIMULATED tool file to remove: the simulation mirrors the real arm's tool '{mirrors}', "
                            "so the gripper and `park` stay locked in simulation as they are on the real arm (`fd --sim tool "
                            "status` shows it). The real tool is detached without --sim, once it is off the gripper",
                       say="シミュレーション用のツール記録はありません（何も外していません）。シミュレーションは実機のツール"
                           f"「{mirrors}」の記録を映しているので、グリッパーと park はロックされたままです。実機のツールは、"
                           "グリッパーから外したあとに、--sim を付けない fd tool detach で登録を外します。")
        return _sim_tool_said(res, a, was=was)
    try:
        name = tooling.load(path).name
    except SafetyError:
        name = "?"
    # Without its own tool file the simulation mirrors the real arm's tool. When the real arm carries one, removing
    # the simulated file unlocks nothing: until 2026-11-14 the result said "the gripper and park are unlocked" all
    # the same, also for a simulated copy that had been left behind and was not even read any more.
    mirrors, real = None, tooling.tool_path(False)
    if a.sim and real != path and os.path.exists(real):
        try:
            mirrors = tooling.load(real).name
        except SafetyError:
            mirrors = "?"
    if not a.dry:
        tooling.clear(a.sim)
    res = {"ok": True, "stage": "planned" if a.dry else "detached", "attached": bool(a.dry), "removed": not a.dry,
           "tool": name, "file": path,
           "next": "the gripper and `park` are unlocked and \"tip\" is the fingertip again: make sure the tool is "
                   "really off the gripper before anything moves",
           "say": (f"ツール「{name}」の登録を外す内容を確認しました（まだ外していません）。" if a.dry else
                   f"ツール「{name}」の登録を外しました。グリッパーと park のロックは解除されています。"
                   "ツールが実際に外れていることを確認してから動かしてください。")}
    if mirrors is not None:
        res.update(attached=True, mirrors=mirrors,
                   next=f"the SIMULATED tool file {'would be' if a.dry else 'was'} removed; the simulation then mirrors the real "
                        f"arm's tool '{mirrors}' again, so the gripper and `park` stay locked in simulation as they are on the "
                        "real arm (`fd --sim tool status` shows it). The real tool is detached without --sim, once it is off "
                        "the gripper",
                   say=(f"シミュレーション用のツール記録（「{name}」）を外す内容を確認しました（まだ外していません）。外すと、" if a.dry else
                        f"シミュレーション用のツール記録（「{name}」）を外しました。")
                       + f"シミュレーションは実機のツール「{mirrors}」の記録に戻るので、グリッパーと park はロックされたままです。")
    return _sim_tool_said(res, a, was=was)


def _pen_options(a, **over):
    """penskill.PenOptions from the command line of `fd draw` / `fd tool calibrate`."""
    import penskill
    kw = dict(over)
    kw["dry"] = bool(getattr(a, "dry", False))
    if getattr(a, "at", None):
        kw["at"] = tuple(_xyz_arg(a.at, "--at", 2))
    if getattr(a, "size", None) is not None:
        kw["size_mm"] = float(a.size)
    for name, key, scale in (("press", "press_m", 0.001), ("v", "v_mm_s", 1.0), ("min_coverage", "min_coverage", 1.0),
                             ("places", "places", 1), ("start_above", "start_above_m", 0.001)):
        if getattr(a, name, None) is not None:
            kw[key] = getattr(a, name) * scale
    for name in ("strokes", "end", "marks_on"):
        if getattr(a, name, None):
            kw[name] = getattr(a, name)
    rot = getattr(a, "rot", None)
    if rot not in (None, "auto"):
        try:
            kw["rot_deg"] = float(rot)
        except ValueError:
            raise SafetyError(f"--rot {rot!r}: a number of degrees or 'auto' (follow the sheet)") from None
    return penskill.PenOptions(**kw).check()


def _pen_session(arm, a):
    """(penskill.PenSession, close()) for `fd draw` / `fd tool calibrate`.

    Real arm: photos come from the camera with the exposure asked for (camera.grab puts the desk values back).
    --sim: the sheet, the pen and the camera are simpaper.SimPaper; no camera is opened. Its hidden truths come
    from --truth FILE (penskill.load_truth), and the sheet with its ink and the REAL pen tip are kept in
    sim_paper.json next to the simulated tool file, so that the morning's commands can be rehearsed one after
    the other (`--new-sheet`: a blank sheet). A new truth file or a newly attached tool starts a new world."""
    import penskill
    import tooling
    cam = _cam_model()
    if not a.sim:
        if getattr(a, "truth", None) or getattr(a, "new_sheet", False):
            raise SafetyError("--truth and --new-sheet describe the SIMULATED sheet and pen: use them with --sim")
        import camera as C
        return penskill.PenSession(arm, cam, lambda exposure="paper": C.grab(exposure=exposure), image_dir=run_dir), (lambda: None)
    path = os.path.join(os.path.dirname(os.path.abspath(tooling.tool_path(True))), "sim_paper.json")
    state = None
    if os.path.exists(path):
        try:
            with open(path) as f:
                state = json.load(f)
        except (OSError, ValueError):
            state = None
    tool = arm.tool
    key = None if tool is None else f"{tool.name}|{tool.attached_at}"
    truth = penskill.load_truth(a.truth) if getattr(a, "truth", None) else None
    if state is not None and (state.get("tool_key") != key or (truth is not None and truth != state.get("truth"))):
        state = None
    if state is not None and truth is None:
        truth = state.get("truth") or None
    if state is not None and getattr(a, "new_sheet", False):
        state = {**state, "paper": None}
    paper, photo = penskill.sim_world(arm, cam, truth, state)
    sess = penskill.PenSession(arm, cam, photo, image_dir=run_dir, sim=True, paper=paper,   # (made at the first photo)
                               save_tool=lambda t: tooling.save(t, True))

    def close():
        st = penskill.sim_state(paper, truth)
        st["tool_key"] = key
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(st, f)
        os.replace(path + ".tmp", path)

    return sess, close


def _pen_refused(e, skill: str) -> dict:
    """An option or a file that cannot be used: the complete refusal (nothing moved)."""
    return {"ok": False, "error": "SafetyError", "skill": skill, "stage": "options", "reason": str(e), "message": str(e),
            "next": f"`fd {'draw' if skill == 'draw' else 'tool calibrate'} --help` lists the options",
            "say": f"指定を受け付けられませんでした（{e}）。アームは動かしていません。"}


def _calibrate_words(a, pen: bool = True) -> str:
    """`fd [--sim] tool calibrate [--pen] ...` with the options this call was given (for a `next` that works as it
    is typed)."""
    import shlex
    words = _fd(a).split() + ["tool", "calibrate"] + (["--pen"] if pen else [])
    if a.stage != "all":
        words += ["--stage", a.stage]
    for flag, value in (("--at", a.at), ("--size", a.size), ("--marks-on", None if a.marks_on == "outside" else a.marks_on),
                        ("--places", a.places), ("--start-above", a.start_above), ("--truth", getattr(a, "truth", None))):
        if value is not None:
            text = f"{value:g}" if isinstance(value, float) else str(value)
            words += [f"{flag}={text}"] if text.startswith("-") else [flag, text]      # (`--at -0.1,0.2` reads as an option)
    words += [flag for flag, on in (("--new-sheet", getattr(a, "new_sheet", False)), ("--dry", a.dry)) if on]
    return shlex.join(words)


def _pen_calibration_refused(a, tool, paints) -> dict:
    """The complete refusal of `fd tool calibrate` for a painting tool without the word --pen (_painting_tool has
    the reason and what happened). Nothing was opened, nothing moved."""
    import penskill
    deep = penskill.PenOptions().max_search_below_m * 1000.0
    # (the commands this text only NAMES stand in words: `fd calib-air` and `fd paint contact` in backticks read as the
    # next step, and typed as printed the first flies a measurement and the second is refused)
    why = (f"tool '{tool.name}' is a painting tool ({paints[1]}), and fd tool calibrate is the PEN calibration: it lowers "
           f"the tool tip onto the paper until ink shows (as far as {deep:.0f} mm below where the paper is expected) and "
           "replaces the tip offset by the one it reads from the ink. For a brush that is the wrong measurement: its tip "
           "is measured in the air (fd calib-air) and the paper height on a scrap sheet (fd paint contact). Nothing "
           "moved")
    return {"ok": False, "error": "SafetyError", "skill": "calibrate", "stage": "tool", "reason": why, "message": why,
            "tool": tool.name, "painting_tool": paints[1],
            "next": f"`{_fd(a)} paint status` says what this tool still needs before it paints and names the next command. "
                    f"Only when the thing on the gripper is a PEN and the pen calibration is meant: `{_calibrate_words(a)}`",
            "say": f"ツール「{tool.name}」は絵の具用の道具として扱っています（{PAINTING_JA[paints[0]]}）。fd tool calibrate は"
                   "ペンの校正で、先端を紙に下ろして試し書きをし、先端の値をインクから読んだ値に置き換えるため、実行しません"
                   "でした（何も動かしていません）。筆の準備は fd paint status の next に従ってください。グリッパーに付いて"
                   "いるのがペンで、ペンの校正をするときだけ --pen を付けます。"}


def _nexts_with_pen(res: dict) -> None:
    """After `fd tool calibrate --pen`: every `fd tool calibrate ...` the result names carries the word too, so
    that the next command works as it is typed (without it a painting tool is refused again)."""
    def fix(text):
        return text.replace("`fd tool calibrate --pen", "`fd tool calibrate").replace("`fd tool calibrate", "`fd tool calibrate --pen")

    if isinstance(res.get("next"), str):
        res["next"] = fix(res["next"])
    for st in (res.get("stages") or {}).values():
        if isinstance(st, dict) and isinstance(st.get("next"), str):
            st["next"] = fix(st["next"])
    if isinstance(res.get("notes"), list):
        res["notes"] = [fix(n) if isinstance(n, str) else n for n in res["notes"]]


def _tool_calibrate(a):
    """`fd tool calibrate [--pen] [--stage tip|contact|map|direction|all]`: measure the attached PEN on the paper
    (penskill.calibrate): the tip from the first ink of a ladder, the contact plane, the map commanded ->
    actual, the direction of travel that keeps its ink. Every stage stores its result in the tool file. It
    lowers the tool onto the paper by the stop rule only: no step deeper without a photo that earned it.

    A painting tool (_painting_tool: a measured tip, a record on the moving jaw or between the jaws, or a painting
    set-up on record; a pen recorded on the fixed jaw is none) is refused
    before the arm or the camera is opened, unless --pen declares that the thing on the gripper is a pen and this
    calibration is meant. With --pen on a tool whose tip was measured in the air, the result says that the stage
    `tip` replaces that tip."""
    import penskill
    import tooling
    try:
        tool = tooling.active(a.sim)
    except SafetyError:
        tool = None                 # (a tool file that cannot be read: the arm says so below, in its own words)
    paints = _painting_tool(tool)
    if paints and not a.pen:
        return _pen_calibration_refused(a, tool, paints)
    try:
        opts = _pen_options(a)
    except SafetyError as e:
        return _pen_refused(e, "calibrate")
    stages = penskill.STAGES if a.stage == "all" else (a.stage,)
    was = tooling.sim_tool_state() if a.sim else None
    measured = None if tool is None or not tool.tip_measured else [round(float(v) * 1000.0, 1) for v in tool.tip_offset]
    with make_arm(a.sim) as arm:
        try:
            sess, close = _pen_session(arm, a)
        except SafetyError as e:
            return _pen_refused(e, "calibrate")
        stored = json.dumps(arm.tool.to_dict(), sort_keys=True, default=str) if arm.tool is not None else None
        try:
            _term_as_interrupt()
            res = penskill.calibrate(sess, stages, opts)
        finally:
            close()
        res["file"] = tooling.source(a.sim)
        if arm.tool is not None:
            res["calibrated"] = tooling.calibration_state(arm.tool)
        wrote = arm.tool is not None and json.dumps(arm.tool.to_dict(), sort_keys=True, default=str) != stored
        gone = measured is not None and arm.tool is not None and not arm.tool.tip_measured
    if paints:
        res["painting_tool"] = paints[1]
    if a.pen:
        res["pen_declared"] = True
        _nexts_with_pen(res)
    if measured is not None and "tip" in stages:
        # the tip stage puts the tip it reads from the ink in the place of the one measured in the air
        tip = "(" + ", ".join(f"{v:g}" for v in measured) + ") mm"
        note = (f"the tip of this tool was measured in the air ({tip}, by fd calib-air). The stage `tip` of the pen calibration "
                + ("has REPLACED it by the tip it read from the ink, and the record of the air measurement is gone: the "
                   "painting commands refuse this tool until `fd calib-air --save` has measured it again" if gone else
                   "REPLACES it by the tip it reads from the ink and drops the record of the air measurement (the painting "
                   "commands then refuse this tool until `fd calib-air --save` has measured it again)"
                   + ("" if a.dry else "; this run did not get that far")))
        res.setdefault("notes", [])
        res["notes"] = list(res["notes"]) + [note]
        res["measured_tip_replaced"] = bool(gone)
        res["say"] = (res.get("say") or "") + (
            f"（注意: 空中で測った筆先 {tip} は、この校正の tip の段階でインクから読んだ値に置き換わりました。"
            "絵のコマンドで使う前に fd calib-air --save で測り直してください。）" if gone else
            f"（注意: この校正の tip の段階は、空中で測った筆先 {tip} をインクから読んだ値に置き換えます。）")
    return _sim_tool_said(res, a, wrote=wrote, was=was)


def _term_as_interrupt():
    """A cancelled or timed-out tool call ends this process with SIGTERM. For the two commands that hold a pen on
    the paper for minutes that must not kill the process on the spot: turned into KeyboardInterrupt, penskill lifts
    the pen and the result is still printed (reviewer finding, 2026-11-12)."""
    import signal
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, signal.default_int_handler)


def cmd_draw(arm, a):
    """`fd draw FILE|NAME`: check the tool -> photo (paper exposure) -> layout on the sheet -> preview ->
    draw -> photo check (penskill.draw). Needs a tool with a measured contact plane and map
    (`fd tool calibrate`). --dry stops after the preview and moves nothing."""
    import penskill
    try:
        opts = _pen_options(a)
        sess, close = _pen_session(arm, a)
    except SafetyError as e:
        return _pen_refused(e, "draw")
    try:
        _term_as_interrupt()
        res = penskill.draw(sess, a.drawing, opts)
    finally:
        close()
    paints = _painting_tool(getattr(arm, "tool", None))
    if paints and res.get("stage") == "tool" and not res.get("ok"):
        # `fd draw` is closed to a brush by the missing pen calibration, and its refusal must not send the brush there
        # (penskill's own text does: "`fd tool calibrate` (all stages) ..."; _painting_tool says what that did)
        name = arm.tool.name
        res["painting_tool"] = paints[1]
        res["reason"] = (f"{res.get('reason') or 'the tool cannot draw'}. fd draw draws with a PEN that carries the pen "
                         f"calibration; tool '{name}' is a painting tool ({paints[1]}). Nothing moved")
        res["next"] = (f"`{_fd(a)} paint status` says what this tool still needs before it paints, and names the next command "
                       "(a plan is painted layer by layer with the painting commands, not with fd draw)")
        res["say"] = (f"ツール「{name}」は絵の具用の道具として扱っています（{PAINTING_JA[paints[0]]}）。fd draw はペンの校正を済ませた"
                      "ペンで描くコマンドなので、実行しませんでした（アームは動かしていません）。筆で描く準備は fd paint status が"
                      "示します。")
    return res


# sub-command -> function(args). The arm is opened only where it is needed (`rest`; a read-only look in
# `attach` / `set`), so `status` and `detach` work with the arm switched off.
def _tool_jaw(a):
    """`fd tool jaw ...`: move the gripper jaw although a tool is registered, for a tool taped to the OUTER face
    of the moving jaw (2026-11-13: a brush taped there had play until the jaw was closed against the fixed jaw
    and held with a light squeeze). The words after `jaw` go to tools/draw/tooljaw.py."""
    import tooljaw
    return tooljaw.main((["--sim"] if a.sim else []) + list(a.jaw_args))


TOOL_COMMANDS = {"attach": _tool_attach, "status": _tool_status, "set": _tool_set, "rest": _tool_rest,
                 "detach": _tool_detach, "calibrate": _tool_calibrate, "jaw": _tool_jaw}


def cmd_tool(arm, a):
    """A tool taped to the gripper: attach / status / set / rest / detach / calibrate / jaw."""
    return TOOL_COMMANDS[a.tool_cmd](a)


def cmd_calib_air(arm, a):
    """`fd calib-air ...`: after the camera was moved (or a new tool was taped on): hold the tool tip at places in
    the air, photograph it, and solve the camera pose and the tool tip together (tools/calib/aircal_cli.py has the
    options). It opens the arm and the camera itself."""
    import aircal_cli
    return aircal_cli.main((["--sim"] if a.sim else []) + list(a.calib_air_args))


# (after `fd hw lock` in a next: the reason is a text the operator writes, so it stands in words, not as
# `fd hw lock --reason "..."`, which typed as it was printed locks with the reason "...")
HW_REASON = "; --reason and a text behind it record why"


def _hw_copies(hwlock) -> tuple:
    """(paths, an English sentence for `next`, a Japanese one for `say`) about the copies of the harness next to
    this folder, and the older harness code inside it (snapshots under tests/refs/, backups/), whose servo bus does
    not look at the lock (hwlock.copies_without_lock); ([], "", "") without any."""
    paths = hwlock.copies_without_lock(ROOT)
    if not paths:
        return [], "", ""
    n = len(paths)
    inside = sum((os.path.realpath(p) + os.sep).startswith(os.path.realpath(ROOT) + os.sep) for p in paths)
    where = ("next to this folder" if not inside else "inside this folder" if inside == n else
             f"next to this folder or inside it ({inside} inside: old snapshots and backups)")
    en = (f"{n} older cop{'y' if n == 1 else 'ies'} of the harness {where} {'does' if n == 1 else 'do'} NOT look "
          "at the lock (\"copies_without_lock\"): an fd or a script started from there still opens the servos and the camera")
    ja = (f"ただし、このフォルダの近く{'や中' if inside else ''}に、ロックを知らない古いコピーが {n} 個あります（copies_without_lock）。"
          "そこから fd やスクリプトを実行すると、ロック中でもサーボとカメラが開きます。")
    return paths, en, ja


def cmd_hw(arm, a):
    """`fd hw lock|unlock|status`: the hardware lock (tools/motion/hwlock.py). While it is set, this copy of the
    harness, and every copy whose servo bus module looks at the lock, opens neither the real servo bus nor the
    real camera. Copies from before the lock existed do not look: `lock` and `status` list the ones next to this
    folder, and the older harness code inside it (snapshots under tests/refs/, backups/), as "copies_without_lock",
    and the texts no longer promise more than that.

    `lock` and `unlock` change the lock of the REAL hardware whether --sim is given or not, so with --sim they are
    refused, like the other commands that change the rig (2026-11-13: `fd --sim hw unlock` took the lock off, and
    "`fd --sim ...` only" is exactly what a program is allowed while the arm is locked). `status` only reads.
    `lock` on a rig that is locked keeps the first reason and time, and says so."""
    import hwlock
    was = hwlock.info()
    since = lambda d: f"{d['since']} から。" if d.get("since") else ""      # noqa: E731
    if a.sim and a.hw_cmd != "status":
        does = "set" if a.hw_cmd == "lock" else "remove"
        # (what was typed is quoted, not put in backticks, and `fd hw lock` is the whole command: its reason is a text
        # the operator writes, and a placeholder inside backticks is typed as it is printed)
        why = (f"\"fd --sim hw {a.hw_cmd}\" would {does} the lock of the REAL hardware ({hwlock.path()}): --sim does not "
               "protect that, and a simulated command must not decide whether the real arm and camera can be opened. "
               f"Nothing was changed (the hardware is {'locked' if was else 'NOT locked'})")
        return {"ok": False, "error": "SafetyError", "stage": "options", "reason": why, "message": why,
                "locked": was is not None, "file": hwlock.path(),
                "next": "`fd --sim hw status` shows the lock; "
                        + (f"`fd hw lock` (without --sim) sets it{HW_REASON}" if a.hw_cmd == "lock" else
                           "`fd hw unlock` (without --sim) removes it: only when the user has said that the real arm may be used"),
                "say": f"--sim を付けた fd hw {a.hw_cmd} は、実機のロックそのものを変えてしまうので実行しませんでした。"
                       f"ロックは変わっていません（いまは{'ロック中' if was else 'ロックされていません'}）。"}
    try:
        if a.hw_cmd == "lock":
            d = hwlock.lock(a.reason or "")
            others, en, ja = _hw_copies(hwlock)
            res = {"ok": True, "stage": "locked", "locked": True, **d, "copies_without_lock": others}
            if d.get("already"):
                asked = str(a.reason or "")
                if asked and asked != d["reason"]:
                    res["asked_reason"] = asked
                res["next"] = ("it was locked already: the first reason and time were kept"
                               + (", the reason given now was NOT written" if "asked_reason" in res else "")
                               + ". To change the text: `fd hw unlock`, then fd hw lock with --reason and the new text (the "
                                 "hardware is open in between)")
                res["say"] = (f"実機はすでにロックされていました（{since(d)}理由: {d['reason'] or '記載なし'}）。"
                              "最初の理由と時刻をそのまま残しています"
                              + ("（今回の理由は記録していません）。" if "asked_reason" in res else "。"))
            else:
                res["next"] = ("the simulated arm (--sim right after fd, before the command) and the tests still work; "
                               "`fd hw unlock` (never with --sim) unlocks")
                res["say"] = ("実機をロックしました。このフォルダのハーネスは、ロックを外すまでサーボもカメラも開きません。"
                              "アームの姿勢とトルクはそのままです。")
            res["next"] += f"; {en}" if en else ""
            res["say"] += ja
            return res
        if a.hw_cmd == "unlock":
            removed = hwlock.unlock()
            res = {"ok": True, "stage": "unlocked", "locked": False, "removed": removed, "file": hwlock.path()}
            if removed:
                res["was"] = {k: was.get(k) for k in ("reason", "since", "by")} if was else None
                res["next"] = ("every command without --sim opens the real servo bus and camera again: move the arm only "
                               f"when the user asks for it; `fd hw lock` locks again{HW_REASON}")
                res["say"] = "実機のロックを外しました。--sim を付けないコマンドは、実機のサーボとカメラを開くようになります。"
            else:
                res["next"] = f"nothing was locked; `fd hw lock` locks the real arm and camera{HW_REASON}"
                res["say"] = "実機はロックされていませんでした。"
            return res
    except OSError as e:
        why = f"the lock file {hwlock.path()} could not be {'written' if a.hw_cmd == 'lock' else 'removed'} ({type(e).__name__}: {e})"
        now = hwlock.info()
        return {"ok": False, "error": type(e).__name__, "stage": "lock", "reason": why, "message": why,
                "locked": now is not None, "file": hwlock.path(),
                "next": "`fd hw status` shows the lock as it is; check who may write that folder",
                "say": f"ロックのファイルを{'書けません' if a.hw_cmd == 'lock' else '消せません'}でした（{e}）。"
                       f"いまは{'ロック中' if now else 'ロックされていません'}。"}
    others, en, ja = _hw_copies(hwlock)
    res = {"ok": True, "stage": "status", "locked": was is not None, **(was or {"file": hwlock.path()}),
           "copies_without_lock": others}
    if was:
        res["next"] = ("the simulated arm (--sim right after fd, before the command) and the tests work while it is locked; "
                       "`fd hw unlock` (never with --sim) unlocks, "
                       "when the user says that the real arm may be used") + (f"; {en}" if en else "")
        res["say"] = f"実機はロック中です（{since(was)}理由: {was['reason'] or '記載なし'}）。" + ja
    else:
        res["next"] = f"`fd hw lock` locks the real arm and camera (nothing moves){HW_REASON}"
        res["say"] = "実機はロックされていません。"
    return res


def cmd_paint(arm, a):
    """`fd paint ...`: watercolour with a brush taped to the gripper (tools/paint/paintcli.py has the
    sub-commands: status, brush, check, look, locate, test, contact, preview, run, rinse, rest). It opens the
    arm itself."""
    import paintcli
    return paintcli.main((["--sim"] if a.sim else []) + list(a.paint_args))


def cmd_bench(arm, a):
    """Free-space motion benchmark of a speed tier (hardware validation): moves to the ready pose
    at the default (validated) tier first, then runs the benchmark moves at the tier under test.
    Keep the space in front of the arm free (about 30 cm) and watch it."""
    blocked = _tool_refusal(arm, "bench", a)
    if blocked:
        return blocked
    import skills
    skills.go_ready(arm, None)
    r = arm.bench_motion(a.tier)
    if a.accept and a.sim:
        # a simulated pass says nothing about the real servos: never record it
        r["accepted"] = False
        r["note"] = "simulation: a speed tier can only be accepted by a benchmark on the real arm"
    elif r.get("passed") and a.accept:
        val = list(load_config().get("speed_tiers_validated", ["slow"]))
        if a.tier not in val:
            val.append(a.tier)

        def mutate(raw):
            raw["speed_tiers_validated"] = val
            if a.make_default:
                raw["speed_tier"] = a.tier

        cfg = edit_config_file(mutate)
        r["accepted"] = True
        r["speed_tier_default"] = cfg.get("speed_tier", "slow")
    elif a.accept:
        r["accepted"] = False
    return r


def cmd_report(arm, a):
    """Summary of a run log (default: today's): how long the picks took per stage and per speed
    tier, what failed and why, benchmark results. Reads files only."""
    d = a.run or run_dir()
    path = os.path.join(d, "events.jsonl")
    if not os.path.exists(path):
        return {"ok": False, "reason": f"no log at {path}"}
    picks, fails, benches = [], [], []
    for line in open(path):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        cmd, res = rec.get("cmd", ""), rec.get("result") or {}
        if res.get("sim") and not a.sim:
            continue  # simulated runs are listed only with --sim
        if cmd == "bench":
            benches.append({"t": rec.get("t"), "tier": res.get("tier"), "pass": res.get("pass"),
                            "accepted": res.get("accepted"), "peak_err_deg": res.get("peak_err_deg"),
                            "fails": res.get("fails")})
        if (cmd not in ("pick", "take", "do", "repl:do") or "timings" not in res
                or res.get("stage") in ("planned", "start")):
            continue
        tier = res.get("tier") or (res.get("speed_tier") or {}).get("tier")
        if res.get("skill") == "take_out":
            tier = f"{tier or '?'} (take-out)"
        if res.get("ok"):
            picks.append({"tier": tier or "?", "t": res["timings"]})
        else:
            fails.append({"t": rec.get("t"), "stage": res.get("stage"), "reason": str(res.get("reason"))[:160],
                          "arm_state": res.get("arm_state")})
    by_tier = {}
    for p in picks:
        by_tier.setdefault(p["tier"], []).append(p["t"])
    summary = {}
    for tier, ts in by_tier.items():
        stages = sorted({k for t in ts for k in t if k != "total"})
        summary[tier] = {"picks": len(ts), "total_mean_s": round(float(np.mean([t.get("total", 0.0) for t in ts])), 1),
                         "total_min_s": round(float(np.min([t.get("total", 0.0) for t in ts])), 1),
                         "stage_mean_s": {k: round(float(np.mean([t.get(k, 0.0) for t in ts])), 2) for k in stages}}
    return {"run": d, "picks_ok": len(picks), "picks_failed": len(fails), "by_tier": summary, "failures": fails[-10:],
            "bench": benches[-10:]}


# ------------------------------------------------------------------ tracking test (slow, accurate motion)
TRACK_MIN_HEIGHT_M = 0.02  # the test figure is flown in the AIR: never nearer to the table than this
# The fine mode passes a comparison when, on the same figure flown in turns with the default mode, its path
# error is no larger in the mean and neither its 95th percentile nor its height wander is more than 10 %
# larger. Stick-slip makes single runs scatter by about +-20 %, more than the gain to be shown: a comparison
# is only RECORDED (--accept) when every mode was flown TRACK_ACCEPT_REPEAT times or more (--repeat), and
# the accepted comparisons are kept as a list (one figure at one place is not the whole drawing area).
TRACK_PASS = {"mean_ratio": 1.0, "p95_ratio": 1.1, "z_range_ratio": 1.1}
TRACK_ACCEPT_REPEAT = 3


def _track_mean(each: list) -> dict:
    """One result for several flights of the same figure in the same mode: the mean of the mean and of
    the 95th-percentile path error, the largest maximum, the envelope of the height error; "each" keeps
    the single flights, "breakdown" is that of the last one. A single flight is returned as it is."""
    if len(each) == 1:
        return each[0]
    avg = lambda f: round(float(np.mean([f(r) for r in each])), 2)  # noqa: E731
    return {"path_err_mm": {"mean": avg(lambda r: r["path_err_mm"]["mean"]), "p95": avg(lambda r: r["path_err_mm"]["p95"]),
                            "max": max(r["path_err_mm"]["max"] for r in each)},
            "z_err_mm": {"min": min(r["z_err_mm"]["min"] for r in each), "max": max(r["z_err_mm"]["max"] for r in each)},
            "lag_s": avg(lambda r: r["lag_s"]), "end_err_mm": avg(lambda r: r["end_err_mm"]),
            "duration_s": avg(lambda r: r["duration_s"]), "n": int(sum(r["n"] for r in each)), "rate_hz": each[-1]["rate_hz"],
            "peak_err_deg": max(r["peak_err_deg"] for r in each), "loop_hz": each[-1]["loop_hz"], "flights": len(each),
            "each": [{"path_err_mm": r["path_err_mm"], "z_err_mm": r["z_err_mm"]} for r in each],
            "breakdown": each[-1]["breakdown"]}


def _track_verdict(base: dict, fine: dict) -> dict:
    """Compare two Trace.analyse results of the same figure (default mode, fine mode)."""
    zr = lambda r: r["z_err_mm"]["max"] - r["z_err_mm"]["min"]  # noqa: E731
    ratios = {"mean_ratio": fine["path_err_mm"]["mean"] / max(base["path_err_mm"]["mean"], 1e-6),
              "p95_ratio": fine["path_err_mm"]["p95"] / max(base["path_err_mm"]["p95"], 1e-6),
              "z_range_ratio": zr(fine) / max(zr(base), 1e-6)}
    fails = [f"{k} {v:.2f} above {TRACK_PASS[k]}" for k, v in ratios.items() if v > TRACK_PASS[k]]
    return {"passed": not fails, "fails": fails, **{k: round(v, 2) for k, v in ratios.items()}}


def cmd_track_test(arm, a):
    """Fly a small figure (circle / square / line) with the tool tip in the AIR at drawing speed and
    report how well the tip followed it: path error (mean / p95 / max), height wander, lag, and what
    the error consists of (along the path, across it, height, per joint, stick-slip). Nothing is
    touched: the figure is refused below 2 cm above the table, and the way to it stays above it.

    --fine flies the figure in turns in the default mode and in the opt-in slow-motion mode
    (run_path(fine=True)), --repeat N times each, and says whether the fine mode passed the
    comparison. --accept (real arm only, never --sim; needs --repeat 3 or more because single
    flights scatter) appends a passed comparison to config.json ("fine_motion"."accepted"): that
    list is what may make the fine mode the default for drawing. --phys (with --sim only) runs on
    the torque-level servo model with the stick-slip friction fitted to the traces of 2026-11-11
    instead of the ideal simulated servos (which follow any path almost perfectly)."""
    import skills
    import tracking as TR
    h = float(a.height)
    if not (math.isfinite(h) and h >= TRACK_MIN_HEIGHT_M):
        raise SafetyError(f"track-test flies its figure in the air: --height {a.height} m above the table is "
                          f"refused (minimum {TRACK_MIN_HEIGHT_M} m; never negative, never touching)")
    if not (math.isfinite(a.size) and 5.0 <= a.size <= 80.0):
        raise SafetyError(f"--size {a.size} mm outside 5 .. 80")
    if not (math.isfinite(a.v) and 1.0 <= a.v <= 30.0):
        raise SafetyError(f"--v {a.v} mm/s outside 1 .. 30")
    if not 1 <= a.repeat <= 5:
        raise SafetyError(f"--repeat {a.repeat} outside 1 .. 5")
    if a.accept and not a.fine:
        raise SafetyError("--accept records a passed comparison of the fine mode: use it together with --fine")
    if a.accept and a.repeat < TRACK_ACCEPT_REPEAT:
        raise SafetyError(f"--accept needs --repeat {TRACK_ACCEPT_REPEAT} or more: single flights scatter by about "
                          f"20 %, more than the difference between the modes")
    if a.phys and not a.sim:
        raise SafetyError("--phys chooses the servo model of the simulation: use it together with --sim")
    try:
        cx, cy = (float(v) for v in str(a.at).split(","))
        if not (math.isfinite(cx) and math.isfinite(cy)):
            raise ValueError
    except ValueError:
        raise SafetyError("--at needs x,y in metres (base frame), e.g. --at 0.22,0.0") from None
    if a.phys:
        # same configuration and pose, torque-level servos with stick-slip (simulation only)
        import phys_bus as PB
        bus = PB.PhysBus(arm.cfg, io_s=0.003, **PB.STICK_SLIP_20261111)
        for j in bus.pos:
            bus.pos[j] = bus.goal[j] = bus.sp[j] = float(arm.bus.pos[j])
        # (the tool of the simulated arm goes with it: Arm() by itself would read the REAL arm's tool file)
        arm = type(arm)(bus=bus, cfg=arm.cfg, tool=getattr(arm, "tool", None)).open()
    cfg = arm.cfg
    plane = cfg.get("table_plane") or [float(cfg.get("table_z", 0.0)), 0.0, 0.0]
    probe = TR.shape_points(a.shape, a.size / 1000.0, (cx, cy), 0.0)
    z_table = max(float(plane[0] + plane[1] * p[0] + plane[2] * p[1]) for p in probe)  # highest table point under it
    z = z_table + h
    pts = TR.shape_points(a.shape, a.size / 1000.0, (cx, cy), z)
    tool_rec = getattr(arm, "tool", None)
    pitch = a.pitch if a.pitch is not None else float(getattr(tool_rec, "pitch_deg", None) or 75.0)
    roll = 0.0
    length = float(sum(math.dist(p, q) for p, q in zip(pts[:-1], pts[1:])))
    plan = {"shape": a.shape, "size_mm": a.size, "v_mm_s": a.v, "centre_xy": [cx, cy], "height_m": round(h, 4),
            "z": round(z, 4), "pitch": pitch, "length_mm": round(length * 1000.0, 1),
            "tool": getattr(tool_rec, "name", None), "fine": bool(a.fine), "repeat": a.repeat}
    # the whole plan is made and checked from the stand-by pose before anything moves
    tip = arm.tip_tool()
    q_ready = arm.ready_q()
    above = np.array([pts[0][0], pts[0][1], max(z, z_table + 0.06)])
    q_above = arm.solve(above, pitch, roll, q_seed=q_ready)
    down = arm.plan_tip_path([(np.array(pts[0]), pitch, roll)], q_start=q_above, step_m=0.003)
    wps = arm.plan_tip_path([(np.array(p), pitch, roll) for p in pts[1:]], q_start=down[-1], step_m=0.0015)
    low = min(float(K.fk(q, tip)[2, 3]) for q in down + wps)
    if low < z_table + TRACK_MIN_HEIGHT_M - 0.001:
        raise SafetyError(f"the planned figure would bring the tool tip down to {1000 * (low - z_table):.1f} mm above "
                          f"the table (minimum {1000 * TRACK_MIN_HEIGHT_M:.0f} mm)")
    transit = skills.plan_transit(arm, q_ready, q_above, z_table + 0.06)
    speed = TR.joint_speed_for(wps, length, a.v / 1000.0)
    arm.check_path([q_ready] + transit + down[1:], 25.0)
    chk = arm.check_path(wps, speed)
    modes = ("default", "fine") if a.fine else ("default",)
    plan.update(joint_speed_deg_s=round(speed, 2), expected_s=round(chk["duration_s"], 1),
                flights=a.repeat * len(modes))
    if a.dry:
        return {"ok": True, "stage": "planned", "dry": True, "reason": "", **plan,
                "next": "run it without --dry (real arm: someone watching; simulation: add --sim before the command)",
                "say": "追従テストの計画ができました（アームは動かしていません）。"}
    skills.go_ready(arm, None)
    tier = arm.speeds(None)

    def to_start():
        q = arm.read_q()
        arm.run_path([q] + skills.plan_transit(arm, q, q_above, z_table + 0.06) + down[1:], tier["approach"], settle=True)
        return arm.plan_tip_path([(np.array(p), pitch, roll) for p in pts[1:]], q_start=arm.read_q(), step_m=0.0015)

    each, runs, stage, saved = {m: [] for m in modes}, {}, "start", []
    try:
        for k in range(a.repeat):
            for mode in modes:
                stage = f"approach ({mode} {k + 1})"
                path = to_start()
                stage = f"track ({mode} {k + 1})"
                tr = TR.record(arm, path, TR.joint_speed_for(path, length, a.v / 1000.0), settle=0.8,
                               **({"fine": True} if mode == "fine" else {}))
                each[mode].append({**tr.analyse(tip), "breakdown": tr.breakdown(tip, cfg),
                                   "peak_err_deg": tr.result["peak_err_deg"], "loop_hz": tr.result["loop_hz"]})
                if a.save or not a.sim:
                    tr.meta.update(plan, mode=mode, flight=k + 1, sim=bool(a.sim))
                    saved.append(tr.save(os.path.join(
                        run_dir(), time.strftime("%H%M%S") + f"_track_{a.shape}_{mode}{k + 1}"), tip))
        stage = "return"
        skills.go_ready(arm, None)
        runs = {m: _track_mean(each[m]) for m in modes}
    except (SafetyError, MotionAborted, BusError) as e:
        st = arm.state()
        runs = {m: _track_mean(each[m]) for m in modes if each[m]}
        return {"ok": False, "stage": stage, "reason": f"{type(e).__name__}: {e}", **plan, "runs": runs,
                "tip_xyz": st["tip_xyz"], "tip_above_table_mm": round(1000 * (st["tip_xyz"][2] - z_table), 1),
                "next": "the arm holds where it stopped, in the air: `fd ready` to bring it back, then look at the reason",
                "say": f"追従テストは途中で止まりました（{stage}）。アームは空中で停止しています。"}
    res = {"ok": True, "stage": "done", "reason": "", **plan, "runs": runs, "traces": saved}
    d = runs["default"]
    say = (f"追従テスト（{a.shape}、{a.v:g} mm/秒）: 経路誤差 平均 {d['path_err_mm']['mean']} mm・最大 {d['path_err_mm']['max']} mm、"
           f"高さのふらつき {d['z_err_mm']['min']}〜{d['z_err_mm']['max']} mm。")
    res["next"] = "compare with the fine mode: add --fine" if not a.fine else ""
    if a.fine:
        f = runs["fine"]
        v = _track_verdict(d, f)
        res["fine_verdict"] = v
        say += (f"ファインモード: 平均 {f['path_err_mm']['mean']} mm・最大 {f['path_err_mm']['max']} mm、"
                f"高さ {f['z_err_mm']['min']}〜{f['z_err_mm']['max']} mm（{'合格' if v['passed'] else '不合格'}）。")
        if a.accept and a.sim:
            # a simulated pass says nothing about the real servos: never record it
            res["accepted"] = False
            res["note"] = "simulation: the fine mode can only be accepted by a comparison flown on the real arm"
        elif a.accept and v["passed"]:
            rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "shape": a.shape, "size_mm": a.size, "v_mm_s": a.v,
                   "repeat": a.repeat,
                   "centre_xy": [cx, cy], "height_m": h, "pitch": pitch, "tool": plan["tool"],
                   "default": {k: d[k] for k in ("path_err_mm", "z_err_mm")},
                   "fine": {k: f[k] for k in ("path_err_mm", "z_err_mm")}, "verdict": v}

            def mutate(raw):
                raw.setdefault("fine_motion", {}).setdefault("accepted", []).append(rec)

            cfg2 = edit_config_file(mutate)
            res["accepted"] = True
            res["accepted_runs"] = len(cfg2.get("fine_motion", {}).get("accepted", []))
        elif a.accept:
            res["accepted"] = False
        res["next"] = ((f"fine mode passed here: fly other shapes / places; on the real arm `--repeat "
                        f"{TRACK_ACCEPT_REPEAT} --accept` records a pass") if v["passed"]
                       else "fine mode did not pass here: keep the default mode (see fine_verdict.fails)")
        if a.repeat < TRACK_ACCEPT_REPEAT:
            res["next"] += f"; single flights scatter by about 20 %: use --repeat {TRACK_ACCEPT_REPEAT} before believing it"
    if a.sim:
        say += "（シミュレーションです。実機の値ではありません。）"
    res["say"] = say
    return res


def cmd_selftest(arm, a):
    """Read-only health check: bus, servo registers, calibration, camera, perception. No motion."""
    rep = {"checks": {}}
    st = arm.state()
    snap = arm.bus.snapshot()
    rep["checks"]["bus"] = {"volt": st["volt"], "temp_c_max": max(st["temp_c"].values()),
                            "status": st["status"], "torque": st["torque"]}
    exp = {"shoulder_pan": (864, 3418), "shoulder_lift": (931, 3356), "elbow_flex": (775, 3064),
           "wrist_flex": (835, 3266), "wrist_roll": (0, 4095), "gripper": (1998, 3648)}
    lim = arm.raw_limits
    rep["checks"]["calibration_unchanged"] = all(tuple(lim[j]) == exp[j] for j in exp)
    rep["checks"]["servo_limits"] = {j: list(lim[j]) for j in lim}
    rep["checks"]["p_gain"] = {j: arm.bus.read(j, "P_Coefficient") for j in ARM}
    rep["checks"]["profile"] = {j: [snap[j].get("acc"), snap[j].get("vel_cap"), snap[j].get("lock")] for j in snap}
    rep["checks"]["speed_tier"] = arm.speeds(None)
    try:
        r = cmd_scene(arm, a)
        rep["checks"]["scene"] = {"n_objects": len(r["objects"]), "timings": r["timings"], "view": r["view"],
                                  "objects": [(o["id"], o["kind"], o["color"]) for o in r["objects"]]}
    except Exception as e:
        rep["checks"]["scene"] = {"error": f"{type(e).__name__}: {e}"}
    rep["ok"] = bool(rep["checks"]["calibration_unchanged"]) and "error" not in rep["checks"]["scene"]
    return rep


def cmd_camera(arm, a):
    _not_with_sim(a, "change the settings of the real camera")
    import camera as C
    if a.mode == "lock":
        C.lock_settings()
    else:
        C.restore_auto()
    return {"camera": a.mode}


# ------------------------------------------------------------------ low-level commands
def cmd_state(arm, a):
    return arm.state()


def cmd_torque(arm, a):
    if a.mode == "on":
        r = arm.torque_on()
        extra = r if isinstance(r, dict) else {}
    else:
        tool = getattr(arm, "tool", None)
        if tool is not None:
            # the rest pose that counts is the TOOL's (the plain one would have the tool in the table)
            import tooling
            fold = _fold_words(a, tool)      # (`fd paint rest` for a painting tool beside a painting set-up)
            why = None if a.force else tooling.torque_off_block(arm, fold)
            if why:
                return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why,
                        "tool": tool.name, "next": f"`{fold}` (folds the arm away with the tool, then torque off)",
                        "say": f"ツール「{tool.name}」が付いたままこの姿勢でトルクを切ると腕が落ちてツールが机に当たるため、"
                               f"実行しませんでした。{fold.replace('--sim ', '')} で収納してください。"}
            arm.torque_off(force=bool(a.force))
            return {"torque": a.mode, "tool": tool.name, **brief_state(arm)}
        if not a.force:
            grav = ("shoulder_lift", "elbow_flex", "wrist_flex")
            holding = [j for j in grav if arm.bus.read(j, "Torque_Enable") == 1]
            off = np.degrees(np.abs(arm.read_q() - arm.raw_to_q(REST_RAW)))[1:4]
            if holding and float(off.max()) > 10.0:
                raise SafetyError(
                    f"refusing torque off: {', '.join(holding)} hold the arm {off.max():.0f} deg away from the "
                    "rest pose and it would fall; use `park`, or `torque off --force` as an emergency stop")
        arm.torque_off()
        extra = {}
    return {"torque": a.mode, **extra, **brief_state(arm)}


def cmd_gains(arm, a):
    """Apply (or restore) the position-loop P gain of the arm joints (RAM only)."""
    if not a.force and arm.is_torqued():
        snap = arm.bus.snapshot()
        loaded = [j for j in ARM if abs(snap[j]["goal"] - snap[j]["pos"]) > 10]
        if loaded:
            raise SafetyError(f"refusing to step the gain while {', '.join(loaded)} carry load "
                              "(the arm would jerk); park first or pass --force")
    if a.restore:
        arm.restore_gains(force=a.force)
    else:
        if a.p is not None:
            arm.cfg["p_gain"] = a.p
        arm.apply_gains(force=a.force)
    return {"p_gain": {j: arm.bus.read(j, "P_Coefficient") for j in ARM}, **brief_state(arm)}


def cmd_joints(arm, a):
    blocked = _tool_no_contact(arm, a, "joints")
    if blocked:
        return blocked
    q = arm.read_q()
    tgt = q.copy()
    for item in a.set:
        name, val = item.split("=")
        idx = ARM.index(name)
        v = math.radians(float(val))
        tgt[idx] = q[idx] + v if a.rel else v
    # one reading for both the start of the path and its safety relaxation
    res = arm.run_path([q, tgt], a.speed, allow_contact=a.contact, ignore_keepouts=a.ignore_keepout or ())
    return {"joint_err_deg": res["joint_err_deg"], "corr_deg": res["corr_deg"], **brief_state(arm)}


def _resolve_tip_args(arm, a):
    xyz = np.array([a.x, a.y, a.z], dtype=float)
    q_now = arm.read_q()
    if getattr(a, "rel", False):
        xyz = K.fk(q_now, arm.tip_tool())[:3, 3] + xyz
    pitch = a.pitch
    if pitch is None:
        pitch = K.pose_summary(q_now, arm.tip_tool())["pitch_down_deg"]
    roll = a.roll if a.roll is not None else math.degrees(float(q_now[4]))
    if a.jaw_azimuth is not None:
        roll = roll_for_azimuth(arm, xyz, pitch, a.jaw_azimuth)
    return xyz, float(pitch), float(roll)


def cmd_tip(arm, a):
    blocked = _tool_no_contact(arm, a, "tip")
    if blocked:
        return blocked
    xyz, pitch, roll = _resolve_tip_args(arm, a)
    res = arm.move_tip(xyz, pitch, roll, a.speed, straight=a.straight, allow_contact=a.contact,
                       ignore_keepouts=a.ignore_keepout or ())
    return {"target": xyz.round(4).tolist(), "pitch": pitch, "roll": roll,
            "joint_err_deg": res["joint_err_deg"], "corr_deg": res["corr_deg"], **brief_state(arm)}


def cmd_check(arm, a):
    """Dry-run: is this tip pose reachable / safe? No motion. Uses the CURRENT roll when --roll
    is not given (as `tip` does)."""
    xyz, pitch, roll = _resolve_tip_args(arm, a)
    lim, tool = arm.q_limits(), arm.tip_tool()
    res = {"xyz": xyz.tolist(), "roll_used": round(roll, 1),
           "feasible_pitch_range": K.pitch_range(xyz, math.radians(roll), tool=tool, limits=lim)}
    fp = getattr(K, "feasible_pitches", None)
    if fp is not None:
        res["feasible_pitch_intervals"] = fp(xyz, math.radians(roll), tool=tool, limits=lim)
    if a.pitch is not None:
        try:
            q = arm.solve(xyz, pitch, roll)
            arm.check_q(q, allow_contact=a.contact)
            jaw = K.jaw_open_axis(K.fk(q))
            res.update(ok=True, q_deg=np.degrees(q).round(2).tolist(),
                       jaw_azimuth_deg=round(math.degrees(math.atan2(jaw[1], jaw[0])), 1),
                       jaw_tilt_deg=round(math.degrees(math.asin(max(-1.0, min(1.0, float(jaw[2]))))), 1))
        except SafetyError as e:
            res.update(ok=False, reason=str(e))
    return res


def cmd_grip(arm, a):
    blocked = _tool_refusal(arm, "grip", a)
    if blocked:
        return blocked
    return arm.set_gripper(a.frac)


def cmd_grasp(arm, a):
    blocked = _tool_refusal(arm, "grasp", a)
    if blocked:
        return blocked
    if getattr(a, "thin", False):
        return arm.pinch()
    if a.fast:
        return arm.grasp_fast()
    return arm.grasp(min_frac=a.min_frac)


def cmd_probe(arm, a):
    # probe_down lowers the "tip" until it lands and reads 3-18 mm deep: with a tool that is the tool, pushed in
    blocked = _tool_refusal(arm, "probe", a)
    if blocked:
        return blocked
    res = arm.probe_down(max_drop_m=a.max_drop, speed_m_s=a.speed)
    return {**res, **brief_state(arm)}


def cmd_snap(arm, a):
    """One photo. --exposure paper: the short exposure for white paper (ink shows, the paper is not clipped);
    the desk values are back before this returns. --frame IMG: a saved image instead of the camera (nothing
    is opened). The result says with which exposure, how much of the table area is clipped to white, and
    whether thin ink could be seen at all in this photo ("ink_visible")."""
    import cv2

    import camera as C
    import vision as V
    exposure = getattr(a, "exposure", None) or "desk"
    frame = getattr(a, "frame", None)
    if frame:
        img = cv2.imread(frame) if os.path.isfile(frame) else None
        if img is None:
            raise SafetyError(f"cannot read the --frame image {frame}")
    elif exposure == "desk":
        img = C.grab()
    else:
        img = C.grab(exposure=exposure)
    vis = img.copy()
    info = {}
    cam = None
    if os.path.exists(C.model_path()):
        cam = C.CameraModel.load()
    if arm is not None:
        st = arm.state()
        info["tip_xyz"] = st["tip_xyz"]
        info["q_deg"] = st["q_deg"]
        if cam is not None and not a.plain:
            uv = cam.project([st["tip_xyz"]])[0]
            V.draw_points(vis, {"tip(FK)": uv}, color=(255, 0, 255))
    if cam is not None and not a.plain:
        V.draw_axes(vis, cam, (0.15, 0.0, arm.cfg.get("table_z", 0.0) if arm else 0.0), 0.05)
        for m in a.mark or []:
            coords, _, label = m.partition(":")
            x, y, z = (float(v) for v in coords.split(","))
            V.draw_points(vis, {label or coords: cam.project([[x, y, z]])[0]}, color=(0, 255, 255))
    if a.grid:
        vis = V.draw_grid(vis, a.grid)
    raw_path = next_image_path(a.name + "_raw")
    cv2.imwrite(raw_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    view_path = raw_path.replace("_raw.jpg", "_view.jpg")
    V.save_view(vis, view_path, crop=crop, scale=a.scale)
    info.update(raw=raw_path, view=view_path, crop=crop, scale=a.scale)
    try:
        info.update(_snap_exposure_report(img, cam, exposure, frame, arm))
    except Exception as e:  # the photo is saved: a fault in the report must not cost the operator the picture
        info.update(exposure=exposure, clipped=None, exposure_report_error=f"{type(e).__name__}: {e}")
    return info


def _snap_exposure_report(img, cam, exposure, frame, arm) -> dict:
    """What `fd snap` says about the exposure of its photo: "exposure", "clipped" (share of the table area at
    250 or above) and whether thin ink could show on the paper in it (ink.can_show_ink of the whole picture)."""
    import ink
    try:
        z = float((arm.cfg if arm is not None else load_config()).get("table_z", 0.0))
    except Exception:
        z = 0.0
    out = {"exposure": exposure, "clipped": round(ink.clipped_share(img, cam, z), 4)}
    if frame:
        out["frame"] = frame
    if cam is not None:
        can = ink.can_show_ink(img, cam, z)
        out["ink_visible"] = bool(can["ok"])
        out["paper"] = {"level": can["paper_level"], "reason": can["reason"]}
        if not can["ok"] and out["clipped"] > ink.MAX_CLIPPED:
            out["next"] = "fd snap --exposure paper   (this photo cannot show ink: the paper is clipped)"
            out["say"] = "紙が白飛びしていて線は写りません。線を見るときは `fd snap --exposure paper` で撮ってください。"
    return out


# ------------------------------------------------------------------ calibration / service
def cmd_calib_fit(arm, a):
    """Fit the camera model to calib_points.json. The focal length is pinned (the data cannot
    identify it); a model that fails the plausibility gate is not saved without --force."""
    if not a.dry:
        _not_with_sim(a, "replace the rig's real camera model (config/camera_model.json)",
                      "Use --dry to see the fit without saving, or run it without --sim")
    import camera as C
    pts = json.load(open(CALIB_POINTS))
    if a.drop:
        pts = [p for i, p in enumerate(pts) if i not in set(a.drop)]
    xyz = np.array([p["xyz"] for p in pts])
    uv = np.array([p["uv"] for p in pts])
    if a.free_f:
        cam = C.fit_camera(xyz, uv, fit_k1=a.k1)
    else:
        cam = C.fit_camera(xyz, uv, f_range=(a.f, a.f), fit_k1=a.k1)
    res = C.residuals(cam, xyz, uv)
    c = cam.center
    problems = []
    if cam.rms is None or cam.rms > 20.0:
        problems.append(f"rms {cam.rms} px > 20")
    if a.free_f and not (1160.0 < cam.f < 1640.0):
        problems.append(f"focal length {cam.f} at the edge of the search range")
    if c[2] < float(xyz[:, 2].max()) + 0.05:
        problems.append("camera centre is not above the calibration points")
    if c[0] < float(xyz[:, 0].max()):
        problems.append("camera centre is not in front of the calibration points")
    saved = False
    if not a.dry and (not problems or a.force):
        if os.environ.get("FD_CAMERA_MODEL"):
            raise SafetyError("FD_CAMERA_MODEL is set: the camera model is read from that file, and this command would "
                              f"replace {C.CAM_MODEL_PATH} instead. Unset FD_CAMERA_MODEL to save the rig's model")
        if os.path.exists(C.CAM_MODEL_PATH):
            shutil.copy(C.CAM_MODEL_PATH, C.CAM_MODEL_PATH + ".bak")
        cam.save()
        saved = True
    return {"ok": not problems or a.force, "n": len(pts), "f": cam.f, "k1": cam.k1, "rms_px": cam.rms,
            "max_px": float(res.max()), "residuals_px": res.round(1).tolist(),
            "camera_center_base": c.round(4).tolist(), "problems": problems, "saved": saved}


def cmd_ref(arm, a):
    """Reference frame of the table area (arm must be clear of it, e.g. at the ready pose)."""
    _not_with_sim(a, "photograph with the real camera and replace the rig's reference photo (config/ref_table.jpg)")
    import cv2

    import camera as C
    img = C.grab(n_avg=8)
    cv2.imwrite(REF_PATH, img, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return {"ref": REF_PATH, "tip_xyz": arm.state()["tip_xyz"]}


def _detect_tip(arm, tag, expect=None, max_dist=120.0):
    import cv2

    import camera as C
    import vision as V
    img = C.grab()
    ref = cv2.imread(REF_PATH) if os.path.exists(REF_PATH) else None
    roi = tuple(arm.cfg.get("tip_roi", (0, 420, 1920, 1075)))
    exclude = [tuple(r) for r in arm.cfg.get("tip_exclude", [(740, 380, 1060, 490)])]
    if expect is None and os.path.exists(C.model_path()):
        try:
            tip_now = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
            expect, max_dist = tuple(C.CameraModel.load().project([tip_now])[0]), 150.0
        except Exception:
            expect = None
    det = V.find_tip(img, ref, roi=roi, exclude=exclude, expect=expect, max_dist=max_dist)
    raw_path = next_image_path(tag + "_raw")
    cv2.imwrite(raw_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if det is None:
        return None, raw_path, None
    u, v = det["uv"]
    bx, by, bw, bh = det["bbox"]
    if bx <= roi[0] + 1 or bx + bw >= roi[2] - 1 or v >= roi[3] - 2:
        return None, raw_path, None  # silhouette cut by the search window: unreliable
    vis = img.copy()
    m = det["mask"]
    vis[m > 0] = (0.65 * vis[m > 0] + 0.35 * np.array([255, 128, 0])).astype(np.uint8)
    V.draw_points(vis, {"": (u, v)}, color=(0, 0, 255), size=36)
    view_path = raw_path.replace("_raw.jpg", "_view.jpg")
    x0, y0 = int(max(0, u - 150)), int(max(0, v - 210))
    V.save_view(vis, view_path, crop=(x0, y0, x0 + 300, y0 + 260), scale=1.5)
    return (u, v), raw_path, view_path


def cmd_calib_at(arm, a):
    """At table position (x, y): probe the table height, then photograph the closed
    fingertip at two wrist rolls 180 deg apart (their mean pixel is the projection of a
    point ON the roll axis, so the result does not depend on the roll zero) at one or more
    heights above the table. Appends to probes.json / calib_points.json (raw joint angles
    are stored so the points can be recomputed if the zero ticks are re-fitted)."""
    blocked = _tool_refusal(arm, "calib-at", a)  # (it probes the table and photographs the closed fingertip)
    if blocked:
        return blocked
    if a.sim:
        raise SafetyError("calib-at records calibration data (probes.json, calib_points.json); it is not "
                          "available on the simulated arm")
    tool = arm.tip_tool()
    axis_tool = K.tool_offset(*AXIS_TOOL)
    out_pts = []
    q = arm.read_q()
    tip = K.fk(q, tool)[:3, 3]
    if tip[2] < a.travel_z - 0.003:
        up = np.array([tip[0], tip[1], a.travel_z])
        arm.move_tip(up, _best_pitch(arm, up, a.roll0), a.roll0, 15.0, straight=True)
    target = np.array([a.x, a.y, a.travel_z])
    p_t = _best_pitch(arm, target, a.roll0)
    if p_t is None:
        raise SafetyError(f"({a.x}, {a.y}, {a.travel_z}) is out of reach")
    arm.move_tip(target, p_t, a.roll0, a.speed, straight=True)
    if a.table_z is None:
        pr = arm.probe_down(max_drop_m=a.travel_z - a.min_z)
        if not pr["contact"]:
            return {"ok": False, "reason": pr.get("reason") or "no table contact",
                    "probe": {k: pr.get(k) for k in ("xy", "start_z", "lowest_tip_z")}}
        z_table = pr["contact_tip_z"]
        n_probes = _append_json(PROBES, {"xy": pr["xy"], "z": z_table, "stall_z": pr.get("stall_tip_z"),
                                         "pitch": round(p_t, 1), "t": time.strftime("%H:%M:%S")})
    else:
        z_table, n_probes = a.table_z, None
    for h in a.heights:
        target = np.array([a.x, a.y, z_table + h])
        p_h = _best_pitch(arm, target, a.roll0)
        try:
            if p_h is None:
                raise SafetyError("out of reach")
            arm.move_tip(target, p_h, a.roll0, 12.0, straight=True)
        except SafetyError as e:
            out_pts.append({"h": h, "skipped": str(e)[:120]})
            continue
        uvs, axes, views, qs = [], [], [], []
        expect = None
        for roll in (a.roll0 - 90.0, a.roll0 + 90.0):
            qn = arm.read_q()
            qt = qn.copy()
            qt[4] = math.radians(roll)
            arm.run_path([qn, qt], 25.0)  # roll only: 25 x joint_speed_scale 1.6 = the 40 deg/s used on 2026-11-10
            time.sleep(0.35)
            qm = arm.read_q()
            uv, raw_path, view_path = _detect_tip(arm, f"cal_h{int(h * 1000)}_r{int(roll)}", expect=expect,
                                                  max_dist=110.0)
            if uv is None:
                break
            expect = uv
            uvs.append(uv)
            axes.append(K.fk(qm, axis_tool)[:3, 3])
            qs.append(qm)
            views.append(view_path)
        if len(uvs) == 2:
            xyz = np.mean(axes, axis=0)
            uv = np.mean(np.array(uvs), axis=0)
            rec = {"xyz": xyz.round(5).tolist(), "uv": uv.round(2).tolist(), "tag": f"axis h={h}",
                   "uv_rolls": [list(u) for u in uvs], "rolls": [a.roll0 - 90.0, a.roll0 + 90.0],
                   "pitch": round(p_h, 1), "q": [[round(float(v), 6) for v in qq] for qq in qs],
                   "raw": [arm.q_to_raw(qq) for qq in qs],
                   "axis_spread_mm": round(float(np.linalg.norm(axes[0] - axes[1]) * 1000), 2), "views": views}
            if not a.dry:
                rec["n_points"] = _append_json(CALIB_POINTS, rec)
            out_pts.append(rec)
        else:
            out_pts.append({"h": h, "skipped": "fingertip not detected"})
        qn = arm.read_q()
        qt = qn.copy()
        qt[4] = math.radians(a.roll0)
        arm.run_path([qn, qt], 25.0)
    return {"table_z": z_table, "n_probes": n_probes, "points": out_pts, **_ref_photo(), **brief_state(arm)}


def cmd_table_fit(arm, a):
    """Fit the table plane z = a + b*x + c*y (base frame) to the probe results. The fingertip
    floor limit is put at the HIGHEST probed point, so no ordinary command can go below a
    surface that was actually touched.
    Saved only with --force: the probed heights read 3-18 mm too deep on this arm (the fingertip slides outward
    after the first touch), so the fit puts the table and the fingertip floor limit that much too low."""
    if not a.dry:
        _not_with_sim(a, "write table_z, floor_z and table_plane into the rig's real configuration file",
                      "Use --dry to see the fit without saving, or run it without --sim")
        if not a.force:
            raise SafetyError("table-fit would replace table_z, floor_z and table_plane by the probed heights, which read "
                              "3-18 mm too deep on this arm (docs/MANUAL.md chapter 9): nothing was written. --dry shows "
                              "the fit; --force saves it")
    pts = json.load(open(PROBES))
    if a.drop:
        pts = [p for i, p in enumerate(pts) if i not in set(a.drop)]
    A = np.array([[1.0, p["xy"][0], p["xy"][1]] for p in pts])
    z = np.array([p["z"] for p in pts])
    if len(pts) >= 4 and not a.flat:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    else:
        coef = np.array([float(np.mean(z)), 0.0, 0.0])
    res = z - A @ coef
    cfg = load_config()
    floor = float(np.max(z)) - float(cfg.get("floor_clearance", 0.004))
    if not a.dry:
        edit_config_file(lambda raw: raw.update(table_plane=[float(c) for c in coef], table_z=float(np.mean(z)),
                                                floor_z=floor))
    return {"n": len(pts), "plane_abc": coef.round(5).tolist(), "residual_mm": (res * 1000).round(2).tolist(),
            "points": [[*p["xy"], round(p["z"], 4)] for p in pts],
            "fingertip_limit_z": round(floor + float(cfg.get("floor_clearance", 0.004)), 4), "saved": not a.dry}


def cmd_calib_roll(arm, a):
    """Estimate the wrist-roll zero from the paired fingertip detections of calib-at.

    The closed fingertips sit r_eff off the roll axis on the fixed-jaw (+x_gf) side, so
    they move on a circle when the wrist rolls. With the camera model known (it was fitted
    from on-axis points only) the direction of that displacement in the image tells which
    roll reading corresponds to URDF roll 0."""
    if not a.dry:
        _not_with_sim(a, "write the wrist-roll zero into the rig's real configuration file",
                      "Use --dry to see the estimate without saving, or run it without --sim")
    cam = _cam_model()
    pts = [p for p in json.load(open(CALIB_POINTS)) if p.get("uv_rolls")]
    zero_now = load_config()["zero_ticks"]["wrist_roll"]
    best = None
    for delta in np.arange(-90.0, 90.01, 0.5):       # urdf_roll = reported_roll - delta
        for r_eff in np.arange(0.003, 0.0131, 0.0005):
            err = 0.0
            for p in pts:
                axis = np.array(p["xyz"])
                pan = -math.atan2(axis[1], axis[0] - K._X_PAN)
                pred = []
                for roll in p["rolls"]:
                    psi = math.radians(roll - delta) - pan - math.radians(ROLL_SKEW_DEG)
                    d = -np.array([math.cos(psi), math.sin(psi), 0.0])   # +x_gf (fixed-jaw side)
                    pred.append(cam.project([axis + r_eff * d])[0])
                obs = np.array(p["uv_rolls"])
                err += float(np.sum(((obs[0] - obs[1]) - (pred[0] - pred[1])) ** 2))
            if best is None or err < best[0]:
                best = (err, float(delta), float(r_eff))
    err, delta, r_eff = best
    rms = math.sqrt(err / max(1, len(pts)))
    new_zero = zero_now + delta * 4096.0 / 360.0
    if not a.dry:
        edit_config_file(lambda raw: raw.setdefault("zero_ticks", dict(load_config()["zero_ticks"])).update(
            wrist_roll=new_zero))
    return {"n": len(pts), "delta_deg": delta, "r_eff_mm": r_eff * 1000, "rms_px": rms,
            "zero_tick_old": zero_now, "zero_tick_new": new_zero, "saved": not a.dry}


def cmd_grasp_plan(arm, a):
    """Plan a top-down grasp of an object described by hand (no camera). No motion."""
    import fdtypes as T
    import grasp as G
    cfg = arm.cfg
    z_table = cfg.get("table_z")
    if z_table is None:
        raise SafetyError("table_z unknown")
    yaw = math.radians(a.yaw)
    length, width = max(a.length, a.width), min(a.length, a.width)
    c, s = math.cos(yaw), math.sin(yaw)
    if a.kind == "cylinder":
        fp = [(a.cx + length / 2 * math.cos(t), a.cy + length / 2 * math.sin(t))
              for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    else:
        fp = [(a.cx + sx * length / 2 * c - sy * width / 2 * s, a.cy + sx * length / 2 * s + sy * width / 2 * c)
              for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    obj = T.SceneObject(id=1, kind=a.kind, color="other", center=(a.cx, a.cy), yaw=yaw,
                        size=(length, width, a.height), z_bottom=z_table, footprint=fp,
                        pixel_bbox=(0, 0, 0, 0), confidence=1.0)
    sc = T.Scene(objects=[obj], table_z=z_table)
    cands = G.plan_grasp(obj, sc, arm, cfg)
    return {"candidates": [c.to_dict() for c in cands[:6]]}


def cmd_servo_tip(arm, a):
    """LEGACY: bring the CLOSED fingertip to (x, y, z) in camera-consistent coordinates by
    repeated detection of the lowest gripper pixel. Known bias: with the jaws closed the lowest
    pixel is the nose of the jaw nearer the camera (usually the moving jaw), about 5 mm from the
    fingertip point along the jaw azimuth. `pick` uses scene.detect_jaws instead."""
    blocked = _tool_refusal(arm, "servo-tip", a)  # (it steers by the picture of the closed fingertip)
    if blocked:
        return blocked
    cam = _cam_model()
    tool = arm.tip_tool()
    target = np.array([a.x, a.y, a.z], dtype=float)
    roll = a.roll if a.roll is not None else math.degrees(arm.read_q()[4])
    pitch = _best_pitch(arm, target, roll)
    if pitch is None:
        raise SafetyError("target out of reach")
    arm.move_tip(target, pitch, roll, a.speed, straight=a.straight, ignore_keepouts=a.ignore_keepout or ())
    steps = []
    total = 0.0
    ok = False
    for it in range(a.iters):
        time.sleep(0.4)
        p_fk = K.fk(arm.read_q(), tool)[:3, 3]
        uv, raw_path, view_path = _detect_tip(arm, f"servo{it}", expect=tuple(cam.project([p_fk])[0]),
                                              max_dist=a.max_px)
        if uv is None:
            return {"ok": False, "reason": "fingertip not found near its expected pixel", "steps": steps,
                    "raw": raw_path, **_ref_photo()}
        p_cam = cam.to_plane(uv, float(p_fk[2]))
        d = target[:2] - p_cam[:2]
        dist = float(np.linalg.norm(d))
        steps.append({"fk_tip": p_fk.round(4).tolist(), "uv": [round(float(v), 1) for v in uv],
                      "cam_tip_xy": p_cam[:2].round(4).tolist(), "error_mm": [round(float(v) * 1000, 1) for v in d],
                      "view": view_path})
        if dist < a.tol:
            ok = True
            break
        if total + dist > a.max_corr:
            return {"ok": False, "reason": f"correction would exceed {a.max_corr * 1000:.0f} mm", "steps": steps}
        total += dist
        nxt = np.array([p_fk[0] + d[0], p_fk[1] + d[1], target[2]])
        p_n = _best_pitch(arm, nxt, roll)
        arm.move_tip(nxt, p_n, roll, 8.0, straight=True, ignore_keepouts=a.ignore_keepout or ())
    p_fk = K.fk(arm.read_q(), tool)[:3, 3]
    return {"ok": ok, "converged": ok, "fk_minus_cam_mm": [round(float(v) * 1000, 1) for v in (p_fk[:2] - target[:2])],
            "total_correction_mm": round(total * 1000, 1), "steps": steps, **_ref_photo(), **brief_state(arm)}


def cmd_config(arm, a):
    if not a.set:
        return load_config()
    _not_with_sim(a, "change the rig's real configuration file (config/config.json)")

    def mutate(raw):
        for item in a.set:
            key, _, val = item.partition("=")
            node = raw
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = json.loads(val)

    return edit_config_file(mutate)


# name -> (function, needs the arm)
COMMANDS = {
    "scene": (cmd_scene, True), "pick": (cmd_pick, True), "take": (cmd_take, True), "do": (cmd_do, True),
    "repl": (cmd_repl, True), "ready": (cmd_ready, True), "home": (cmd_ready, True),
    "park": (cmd_park, True), "bench": (cmd_bench, True), "selftest": (cmd_selftest, True),
    "tool": (cmd_tool, False),  # opens the arm itself where it needs it (`tool status` / `detach` work with the arm off)
    "draw": (cmd_draw, True),
    "paint": (cmd_paint, False),  # (opens the arm itself: `paint status` works with the arm off)
    "hw": (cmd_hw, False),
    "calib-air": (cmd_calib_air, False),
    "camera": (cmd_camera, False), "report": (cmd_report, False),
    "track-test": (cmd_track_test, True),
    "state": (cmd_state, True), "torque": (cmd_torque, True), "gains": (cmd_gains, True),
    "joints": (cmd_joints, True), "tip": (cmd_tip, True), "check": (cmd_check, True),
    "grip": (cmd_grip, True), "grasp": (cmd_grasp, True), "probe": (cmd_probe, True), "snap": (cmd_snap, True),
    "calib-fit": (cmd_calib_fit, False), "config": (cmd_config, False), "ref": (cmd_ref, True),
    "calib-at": (cmd_calib_at, True), "table-fit": (cmd_table_fit, False), "calib-roll": (cmd_calib_roll, False),
    "grasp-plan": (cmd_grasp_plan, True), "servo-tip": (cmd_servo_tip, True),
}


def build_parser():
    p = argparse.ArgumentParser(prog="fd", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true",
                   help="simulated arm (the servo bus is not opened; the camera still is, unless --frame gives saved "
                        "images); state starts at the rest pose")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_frames(s):
        s.add_argument("--frame", action="append",
                       help="use saved image(s) instead of the live camera (pick / take / do / repl: only with --sim "
                            "or --dry, never to move the real arm)")

    s = sub.add_parser("scene", help="what is on the table")
    add_frames(s)
    s = sub.add_parser("pick", help="pick an object and drop it into a container (one call does everything)")
    s.add_argument("--color", help="red, blue, green, yellow, white, ivory, black, ...")
    s.add_argument("--kind", choices=("box", "cylinder", "unknown"))
    s.add_argument("--id", type=int, help="object id from `scene`")
    s.add_argument("--nearest", help="x,y: the object nearest to this point")
    s.add_argument("--largest", action="store_true")
    s.add_argument("--into-color", dest="into_color")
    s.add_argument("--into-id", dest="into_id", type=int)
    s.add_argument("--all", action="store_true", help="repeat until no matching object is left")
    s.add_argument("--max-objects", type=int, default=6)
    s.add_argument("--tier", default=None, help="speed tier (default: config speed_tier)")
    s.add_argument("--visual", choices=("auto", "on", "off"), default=None,
                   help="look at the jaws before descending: auto (default) = record, correct only tight grasps; "
                        "on = always correct; off = skip")
    s.add_argument("--end", choices=("ready", "park"), default="ready",
                   help="afterwards: stay raised and ready for the next request (torque on) or fold away (torque off)")
    s.add_argument("--retries", type=int, default=None)
    s.add_argument("--dry", action="store_true", help="perceive and plan only")
    s.add_argument("--no-verify", action="store_true")
    s.add_argument("--verbose", action="store_true")
    add_frames(s)
    s = sub.add_parser("take", help="take an object OUT of a container and put it on the table (one call does everything)")
    s.add_argument("--color", help="red, blue, green, yellow, white, ivory, black, ...")
    s.add_argument("--kind", choices=("box", "cylinder", "unknown"))
    s.add_argument("--id", type=int, help="object id from `scene` (an object with \"in_container\")")
    s.add_argument("--nearest", help="x,y: the object in the container nearest to this point")
    s.add_argument("--largest", action="store_true")
    s.add_argument("--from-color", dest="from_color", help="colour of the container it is taken out of")
    s.add_argument("--from-id", dest="from_id", type=int)
    s.add_argument("--to", default=None, help="x,y on the table where it is put down (default: the nearest free spot)")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--hollow", action="store_true",
                   help="the object is hollow and open at the top (a cap, a cup lying open side up): its wall may be "
                        "pinched, fixed jaw outside, moving jaw inside. A wall is pinched only when this option, the "
                        "camera or the request (`fd do \"...キャップ...\"`) says so")
    g.add_argument("--solid", action="store_true", help="the object is not hollow: grasp across it only")
    s.add_argument("--probe", choices=("auto", "on", "off"), default=None,
                   help="touch check before the last stretch down: off (default; heights come from geometry), on, "
                        "auto = when the grasp leaves < 3 mm. It can only STOP the take-out (something higher than "
                        "expected under the jaws); it never sets the grasp height (probe_down reads 3-18 mm deep)")
    s.add_argument("--tier", default=None, help="speed tier (default: config speed_tier)")
    s.add_argument("--end", choices=("ready", "park"), default="ready")
    s.add_argument("--retries", type=int, default=None)
    s.add_argument("--dry", action="store_true", help="perceive and plan only")
    s.add_argument("--no-verify", action="store_true")
    s.add_argument("--verbose", action="store_true")
    add_frames(s)
    s = sub.add_parser("do", help="plain-language request, e.g. fd do '青いキューブを箱に入れて' (no LLM needed)")
    s.add_argument("text")
    s.add_argument("--max-objects", type=int, default=6)
    s.add_argument("--tier", default=None)
    s.add_argument("--visual", choices=("auto", "on", "off"), default=None)
    s.add_argument("--end", choices=("ready", "park"), default="ready",
                   help="afterwards: stay raised and ready for the next request (torque on) or fold away (torque off)")
    s.add_argument("--retries", type=int, default=None)
    s.add_argument("--dry", action="store_true")
    s.add_argument("--no-verify", action="store_true")
    s.add_argument("--verbose", action="store_true")
    add_frames(s)
    for name in ("ready", "home", "park"):
        s = sub.add_parser(name)
        s.add_argument("--tier", default=None)
    s = sub.add_parser("tool", help="a tool taped to the gripper (a pen, a brush): attach / status / set / rest / detach / "
                                    "calibrate / jaw (the jaw of a tool taped to the moving jaw); `fd calib-air` measures "
                                    "the tip of a newly attached tool in the air")
    tsub = s.add_subparsers(dest="tool_cmd", required=True)

    def add_tip(t, required):
        g = t.add_mutually_exclusive_group(required=required)
        g.add_argument("--tip", help="x,y,z of the tool tip in gripper_frame_link axes, metres (x: towards the fixed "
                                     "jaw, y: sideways, z: along the fingers), e.g. 0.0015,-0.0081,0.0621")
        g.add_argument("--along-fingers", dest="along_fingers", type=float, metavar="LEN",
                       help="a straight tool taped along the fingers whose tip is LEN metres beyond the fingertip")
        g.add_argument("--across-fingers", dest="across_fingers", type=float, metavar="LEN",
                       help="... taped across the fingers, tip LEN metres out on the fixed-jaw side")
        t.add_argument("--pitch", type=float, default=None,
                       help="gripper pitch for working with it (default: 75 along the fingers, 30 across)")
        t.add_argument("--mount", choices=("along_fingers", "across_fingers", "other"), default=None,
                       help="how it is taped on (default: from the tip option)")
        t.add_argument("--note", action="append", help="free text kept with the tool")
        t.add_argument("--joint-limit", dest="joint_limit", action="append", metavar="JOINT=LO:HI",
                       help="extra joint limit in degrees while this tool is on, e.g. wrist_flex=20:97 for a long "
                            "handle that lies over the forearm (repeatable, read in the order given). `tool set` "
                            "changes the joints that are named and keeps the limits of the others; JOINT=none "
                            "removes the limit of one joint, 'none' removes them all. A range the arm cannot apply "
                            "is refused")
        t.add_argument("--no-robot", action="store_true", help="do not read the arm (it is off, or being teleoperated)")
        t.add_argument("--dry", action="store_true", help="show what would be recorded; write nothing")

    t = tsub.add_parser("attach", help="record what is taped to the gripper (locks the gripper and park)")
    t.add_argument("--name", required=True)
    add_tip(t, True)
    t.add_argument("--replace", action="store_true", help="replace the tool that is recorded (its calibration is lost)")
    tsub.add_parser("status", help="what is attached, whether it is calibrated, what to do next (file only)")
    t = tsub.add_parser("set", help="change the recorded tool (tip, pitch, name, region, note)")
    t.add_argument("--name", default=None)
    add_tip(t, False)
    t.add_argument("--region", default=None, help="x0,y0,x1,y1 on the table where the calibration holds, or 'none'")
    t.add_argument("--keep-calibration", action="store_true",
                   help="keep the measured contact plane / map on record although the tip, mount or pitch changes "
                        "(`fd draw` refuses them when the tip moved more than 0.5 mm or the pitch more than 0.5 deg)")
    t = tsub.add_parser("rest", help="fold the arm away with the tool on, then torque off (instead of `park`); with a "
                                     "brush and a picture on the table use `fd paint rest`")
    t.add_argument("--tier", default=None)
    t.add_argument("--clearance", type=float, default=0.05, help="tool tip above the table in the rest pose (m)")
    t.add_argument("--pan", type=float, default=20.0, help="base angle of the rest pose (deg)")
    t.add_argument("--keep-torque", action="store_true", help="stay folded with the torque on")
    t.add_argument("--over-canvas", dest="over_canvas", action="store_true",
                   help="fold although the painting set-up has its canvas under this pose (refused otherwise: with the "
                        "torque off the wrist can sink until a brush stands on the picture; `fd paint rest` folds "
                        "off the canvas)")
    t.add_argument("--dry", action="store_true", help="plan and check only")
    t = tsub.add_parser("detach", help="forget the tool (ONLY when it is off the gripper): unlocks gripper and park")
    t.add_argument("--dry", action="store_true")
    t = tsub.add_parser("jaw", help="a tool taped to the OUTER face of the moving jaw: close the jaw against the fixed "
                                    "jaw and hold it (no play), or show where it stands (`fd tool jaw --help`)",
                        add_help=False)
    t.add_argument("jaw_args", nargs=argparse.REMAINDER)
    t = tsub.add_parser("calibrate", help="measure a PEN on the paper: tip, contact height, correction map, stroke "
                                          "direction (stops by what the camera sees; needed before `fd draw`). Not for "
                                          "a brush: `fd paint status` leads through what a painting tool needs")
    t.add_argument("--pen", action="store_true",
                   help="a declaration: the thing on the gripper is a PEN, and this calibration (ink on the paper) is "
                        "meant. Needed for a tool that counts as a painting tool (its tip was measured with `fd "
                        "calib-air`, its file records a jaw, or a painting set-up is on record): without the word "
                        "such a tool is refused before anything moves. The stage `tip` replaces a tip that was "
                        "measured in the air")
    t.add_argument("--stage", choices=("tip", "contact", "map", "direction", "all"), default="all",
                   help="one stage (each reads and writes the tool file, so a stage can be repeated) or all, in this order")
    t.add_argument("--at", default=None, help="x,y: centre of the drawing box the test marks are laid out around")
    t.add_argument("--size", type=float, default=None, help="size (mm) of the drawing box that is kept clean (default 105)")
    t.add_argument("--marks-on", dest="marks_on", choices=("outside", "scrap"), default="outside",
                   help="outside (default): test marks stay out of the box `fd draw` would use. scrap: anywhere in "
                        "reach, for a sheet that will be thrown away")
    t.add_argument("--places", type=int, default=None, help="contact ladders (3 .. 8, default 4)")
    t.add_argument("--start-above", dest="start_above", type=float, default=None,
                   help="mm above the expected paper height at which a first search starts (default 12)")
    t.add_argument("--dry", action="store_true", help="photo and plan only (where the marks would go); nothing moves")
    t.add_argument("--truth", default=None, help="--sim only: JSON file with the hidden truths of the simulated pen and sheet")
    t.add_argument("--new-sheet", dest="new_sheet", action="store_true", help="--sim only: a blank simulated sheet")
    s = sub.add_parser("calib-air", help="camera moved, or a new tool: photograph the tool tip in the air and solve the "
                                         "camera pose and the tool tip together (`fd calib-air --help`)", add_help=False)
    s.add_argument("calib_air_args", nargs=argparse.REMAINDER)
    s = sub.add_parser("hw", help="the hardware lock: while it is set, this copy of the harness opens neither the real "
                                  "servo bus nor the camera (for work by programs or agents that must stay in simulation). "
                                  "Older copies of the harness do not know the lock: `fd hw status` lists them")
    hsub = s.add_subparsers(dest="hw_cmd", required=True)
    t = hsub.add_parser("lock", help="lock the real arm and camera (nothing moves; the torque stays as it is). Not with "
                                     "--sim. A rig that is locked keeps its first reason and time")
    t.add_argument("--reason", default="", help="why, for whoever meets the refusal")
    hsub.add_parser("unlock", help="remove the lock (not with --sim: it is the lock of the REAL hardware)")
    hsub.add_parser("status", help="locked? since when, why; and the copies of the harness next to this folder, and the "
                                   "old harness code inside it, that do not know the lock")
    s = sub.add_parser("paint", help="watercolour with a brush taped to the gripper: status / brush / check / look / "
                                     "locate / test / contact / preview / run / rinse / rest (`fd paint --help` lists "
                                     "them, `fd paint run --help` etc. the options of one)", add_help=False)
    s.add_argument("paint_args", nargs=argparse.REMAINDER)
    s = sub.add_parser("draw", help="draw a stroke file with the attached, calibrated tool (one call: photo, layout, "
                                    "preview, drawing, photo check)")
    s.add_argument("drawing", help="a drawing file, or the name of one in tools/draw/drawings/ (cat)")
    s.add_argument("--dry", action="store_true", help="photo, layout, plan and preview image only; nothing moves")
    s.add_argument("--size", type=float, default=None, help="longer side of the drawing box in mm (default 105)")
    s.add_argument("--at", default=None, help="x,y: where the centre of the box should be (default: mid-sheet)")
    s.add_argument("--rot", default="auto", help="turn of the box in degrees, or auto = follow the sheet")
    s.add_argument("--strokes", choices=("as-is", "away", "toward", "auto"), default="auto",
                   help="direction the pieces travel: auto (default) = what the tool's direction test measured, "
                        "as-is when it was not measured; away / toward = away from / towards the robot")
    s.add_argument("--press", type=float, default=None, help="mm below the contact plane (default 2.3, at most 4)")
    s.add_argument("--v", type=float, default=None, help="pen speed on the paper in mm/s (default 6)")
    s.add_argument("--end", choices=("aside", "rest"), default="aside",
                   help="afterwards: beside the sheet with the torque on, or the tool rest pose with the torque off")
    s.add_argument("--min-coverage", dest="min_coverage", type=float, default=None,
                   help="share of the line the final photo must show for ok (default 0.85)")
    s.add_argument("--truth", default=None, help="--sim only: JSON file with the hidden truths of the simulated pen and sheet")
    s.add_argument("--new-sheet", dest="new_sheet", action="store_true", help="--sim only: a blank simulated sheet")
    s = sub.add_parser("repl", help="interactive: type requests one per line; no start-up time, no LLM")
    s.add_argument("--tier", default=None)
    s.add_argument("--visual", choices=("auto", "on", "off"), default=None)
    s.add_argument("--no-park", action="store_true", help="leave the arm at the stand-by pose on quit")
    add_frames(s)
    s = sub.add_parser("bench", help="validate a speed tier on the real arm (free-space moves from the ready pose)")
    s.add_argument("--tier", required=True)
    s.add_argument("--accept", action="store_true", help="mark the tier validated in config.json if it passes")
    s.add_argument("--make-default", action="store_true")
    s = sub.add_parser("selftest")
    add_frames(s)
    s = sub.add_parser("camera")
    s.add_argument("mode", choices=("lock", "auto"))
    s = sub.add_parser("report", help="summary of a run log: pick timings per stage and tier, failures, benchmarks")
    s.add_argument("--run", default=None, help="run directory (default: today's)")
    s = sub.add_parser("track-test", help="fly a small figure in the AIR at drawing speed and measure how well the "
                                          "tool tip follows it (never below 2 cm above the table)")
    s.add_argument("--shape", choices=("circle", "square", "line"), default="circle")
    s.add_argument("--size", type=float, default=20.0, help="diameter / side / length in mm (5 .. 80)")
    s.add_argument("--v", type=float, default=6.0, help="tool-tip speed in mm/s (1 .. 30)")
    s.add_argument("--height", type=float, default=0.03, help="height of the figure above the table in m (>= 0.02)")
    s.add_argument("--at", default="0.22,0.0", help="x,y of the figure's centre (base frame, metres)")
    s.add_argument("--pitch", type=float, default=None, help="gripper pitch (default: the attached tool's, else 75)")
    s.add_argument("--fine", action="store_true",
                   help="fly it in turns in the default mode and in the opt-in slow-motion mode, and compare")
    s.add_argument("--repeat", type=int, default=1, help="flights per mode (1 .. 5); single flights scatter by 20 %%")
    s.add_argument("--accept", action="store_true",
                   help="with --fine --repeat 3 on the REAL arm: record a passed comparison in config.json "
                        "(never with --sim)")
    s.add_argument("--phys", action="store_true",
                   help="with --sim: servo model with the stick-slip friction measured on 2026-11-11")
    s.add_argument("--save", action="store_true", help="keep the recorded trace (.npz) also for a simulated run")
    s.add_argument("--dry", action="store_true", help="plan and check only")

    sub.add_parser("state")
    s = sub.add_parser("torque", help="on / off. To de-energise a raised arm use `park`")
    s.add_argument("mode", choices=("on", "off"))
    s.add_argument("--force", action="store_true", help="release even though the arm is raised (it WILL fall)")
    s = sub.add_parser("gains")
    s.add_argument("--p", type=int, default=None)
    s.add_argument("--restore", action="store_true")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("joints")
    s.add_argument("set", nargs="+", help="name=deg ...")
    s.add_argument("--rel", action="store_true")
    s.add_argument("--speed", type=float, default=30.0)
    s.add_argument("--contact", action="store_true", help="allow the fingertip a few mm below the floor limit")
    s.add_argument("--ignore-keepout", action="append")
    for name in ("tip", "check"):
        s = sub.add_parser(name)
        s.add_argument("x", type=float)
        s.add_argument("y", type=float)
        s.add_argument("z", type=float)
        s.add_argument("--pitch", type=float, default=None)
        s.add_argument("--roll", type=float, default=None)
        s.add_argument("--jaw-azimuth", "--jaw-yaw", dest="jaw_azimuth", type=float, default=None,
                       help="direction fixed jaw -> moving jaw in the base frame (directed)")
        s.add_argument("--contact", action="store_true")
        s.add_argument("--rel", action="store_true", help="xyz is an offset from the current tip")
        if name == "tip":
            s.add_argument("--straight", action="store_true")
            s.add_argument("--speed", type=float, default=25.0)
            s.add_argument("--ignore-keepout", action="append")
    s = sub.add_parser("grip")
    s.add_argument("frac", type=float)
    s = sub.add_parser("grasp")
    s.add_argument("--min-frac", type=float, default=0.0)
    s.add_argument("--fast", action="store_true")
    s.add_argument("--thin", action="store_true",
                   help="hold something thin (a sheet, a cup wall): press as hard as a normal hold, no contact detection")
    s = sub.add_parser("probe")
    s.add_argument("--max-drop", type=float, default=0.05)
    s.add_argument("--speed", type=float, default=0.006, help="descent speed in m/s")
    s = sub.add_parser("snap")
    s.add_argument("--name", default="snap")
    s.add_argument("--grid", type=int, default=0)
    s.add_argument("--crop", default=None, help="x0,y0,x1,y1 in full-resolution pixels")
    s.add_argument("--scale", type=float, default=0.6667)
    s.add_argument("--plain", action="store_true")
    s.add_argument("--mark", action="append", help="x,y,z:label (base frame) to draw")
    s.add_argument("--no-robot", action="store_true")
    s.add_argument("--exposure", choices=("desk", "paper"), default="desk",
                   help="desk (default): the locked exposure everything else uses. paper: short exposure for white "
                        "paper, so that ink shows (the desk values are put back before the command returns)")
    s.add_argument("--frame", default=None,
                   help="a saved image instead of the camera (the camera is not opened; without --sim the arm is "
                        "not opened either)")
    s = sub.add_parser("calib-fit")
    s.add_argument("--f", type=float, default=1430.0, help="pinned focal length (px)")
    s.add_argument("--free-f", action="store_true", help="search the focal length instead of pinning it")
    s.add_argument("--drop", type=int, nargs="*")
    s.add_argument("--k1", action="store_true")
    s.add_argument("--dry", action="store_true")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("config")
    s.add_argument("--set", nargs="*")
    sub.add_parser("ref")
    s = sub.add_parser("calib-at")
    s.add_argument("x", type=float)
    s.add_argument("y", type=float)
    s.add_argument("--travel-z", type=float, default=0.04)
    s.add_argument("--min-z", type=float, default=-0.05, help="lowest height the probe may reach")
    s.add_argument("--table-z", type=float, default=None, help="skip probing and use this table height")
    s.add_argument("--heights", type=float, nargs="+", default=[0.012, 0.06])
    s.add_argument("--roll0", type=float, default=0.0)
    s.add_argument("--speed", type=float, default=20.0)
    s.add_argument("--dry", action="store_true")
    s = sub.add_parser("calib-roll")
    s.add_argument("--dry", action="store_true")
    s = sub.add_parser("grasp-plan", help="plan a grasp for an object described by hand")
    s.add_argument("cx", type=float)
    s.add_argument("cy", type=float)
    s.add_argument("--kind", choices=("box", "cylinder"), default="box")
    s.add_argument("--length", type=float, required=True, help="long side / diameter (m)")
    s.add_argument("--width", type=float, required=True, help="short side / diameter (m)")
    s.add_argument("--height", type=float, required=True)
    s.add_argument("--yaw", type=float, default=0.0, help="direction of the long side in the base frame (deg)")
    s = sub.add_parser("servo-tip", help="legacy closed-gripper visual servo")
    s.add_argument("x", type=float)
    s.add_argument("y", type=float)
    s.add_argument("z", type=float)
    s.add_argument("--roll", type=float, default=None)
    s.add_argument("--speed", type=float, default=20.0)
    s.add_argument("--straight", action="store_true")
    s.add_argument("--iters", type=int, default=4)
    s.add_argument("--tol", type=float, default=0.0015)
    s.add_argument("--max-corr", type=float, default=0.025)
    s.add_argument("--max-px", type=float, default=90.0)
    s.add_argument("--ignore-keepout", action="append")
    s = sub.add_parser("table-fit")
    s.add_argument("--drop", type=int, nargs="*")
    s.add_argument("--flat", action="store_true")
    s.add_argument("--dry", action="store_true")
    s.add_argument("--force", action="store_true", help="save the fit although probed heights read 3-18 mm too deep")
    return p


def make_arm(sim: bool) -> Arm:
    if sim:
        import tooling
        from sim_bus import SimBus
        # the simulated arm carries the simulated tool (`fd --sim tool attach`) or mirrors the real one, and
        # with a tool it starts in the tool's rest pose (the plain rest pose would have the tool in the table)
        arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=tooling.active(sim=True))
        tooling.sim_start(arm)
        return arm
    return Arm()        # refuses (SafetyError) without the rig's configuration file: robot.require_rig_config()


# sub-commands that read their own words (their modules have their own parsers)
_OWN_WORDS = {"paint": "paint_args", "calib-air": "calib_air_args"}


def _split_own_words(argv):
    """fd's parser gets the words up to and including a sub-command that reads its own words (`paint`,
    `calib-air`, `tool jaw`); the rest goes to that sub-command untouched, without one leading `--`.
    argparse.REMAINDER cannot do this when the first of those words is an option: `fd calib-air --dry` ended in
    "unrecognized arguments", exit code 2 and no JSON result, although every `next` of that command is spelled
    that way (found by the review of 2026-11-13). -> (words for the parser, attribute name | None, words)"""
    argv = list(argv)
    for i, w in enumerate(argv):
        if w.startswith("-"):
            continue                       # fd's own options stand before the command (--sim)
        if w in _OWN_WORDS:
            return argv[:i + 1], _OWN_WORDS[w], argv[i + 1:]
        if w == "tool" and argv[i + 1:i + 2] == ["jaw"]:
            return argv[:i + 2], "jaw_args", argv[i + 2:]
        break
    return argv, None, []


def _sim_misplaced(a, own: str) -> dict:
    """The refusal for a `--sim` among the words of a sub-command that reads its own words. fd hands those
    commands "--sim" in FRONT of their words when fd itself was called with --sim, so a `--sim` typed after the
    command (`fd tool jaw --sim status`) looks to them exactly like fd's own: they would work on the simulated arm
    while fd takes the call for a real one (no "sim" in the result or in the run log, the simulated tool file not
    moved out of the project folder). Which was meant is not guessed: refused, with the spelling that works."""
    import shlex
    name = {"paint_args": "paint", "calib_air_args": "calib-air", "jaw_args": "tool jaw"}[own]
    typed = shlex.join(w for w in getattr(a, own) if w != "--sim")
    why = f"--sim must stand before the command (fd --sim {name}, then its words). Nothing moved"
    return {"ok": False, "error": "SafetyError", "stage": "options", "reason": why, "message": why,
            "next": f"`fd --sim {name} {typed}".rstrip() + "`",
            "say": f"--sim は fd のすぐ後ろに書いてください（fd --sim {name} ...）。何も動かしていません。"}


def main(argv=None):
    argv, own, words = _split_own_words(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if own is not None:
        setattr(a, own, words[1:] if words[:1] == ["--"] else words)
    fn, needs_arm = COMMANDS[a.cmd]
    if own is not None and "--sim" in getattr(a, own):
        fn, needs_arm = (lambda _arm, a_: _sim_misplaced(a_, own)), False
    if a.cmd == "snap" and (a.no_robot or (a.frame and not a.sim)):
        needs_arm = False       # a saved image says nothing about where the real arm is now: do not open the bus for it
    args = {k: v for k, v in vars(a).items() if k != "cmd"}
    t0 = time.time()
    a._camera = None
    sim_tool_was = _sim_tool_out_of_tree() if a.sim else None
    try:
        if (getattr(a, "frame", None) and not a.sim and not getattr(a, "dry", False)
                and a.cmd in MOVING_CAMERA_COMMANDS):
            # before the camera, the bus or anything else is opened: the REAL arm would be driven by a
            # picture of some other moment
            raise SafetyError(f"fd {a.cmd} with --frame replays saved images, and without --sim that would move the "
                              "REAL arm by them: use --frame together with --sim (simulated arm) or with --dry "
                              "(plan only; pick / take / do)")
        if needs_arm and not a.sim:
            require_rig_config()      # before the camera is touched; Arm() asks again when it is built
        if a.cmd in CAMERA_COMMANDS:
            a._camera = _open_camera(a)
        if needs_arm:
            with make_arm(a.sim) as arm:
                res = fn(arm, a)
        else:
            res = fn(None, a)
        res = {"ok": True, **res} if "ok" not in res else res
    except (SafetyError, MotionAborted, BusError, HardwareLocked) as e:
        res = {"ok": False, "error": type(e).__name__, "message": str(e)}
        if isinstance(e, HardwareLocked) or "hardware is locked" in str(e):
            res.update(stage="hardware", reason=str(e), next="`fd hw status` says why it is locked; `fd hw unlock` unlocks",
                       say="実機はロック中です（サーボもカメラも開いていません）。シミュレーション（fd --sim）とテストは使えます。")
    except Exception as e:  # anything else must still reach the operator and the log
        import traceback
        res = {"ok": False, "error": type(e).__name__, "message": str(e),
               "trace": traceback.format_exc().strip().splitlines()[-4:]}
    finally:
        if a._camera is not None:
            _close_camera(a._camera)
        if sim_tool_was is not None:
            import tooling
            tooling.SIM_TOOL_FILE = sim_tool_was
    if a.sim:
        res["sim"] = True
    res["wall_s"] = round(time.time() - t0, 2)
    try:
        log_event(a.cmd, args, res, t0)
    except OSError as e:        # a full disk or a read-only folder must not swallow the answer of a finished command
        res["log_error"] = f"the run record could not be written ({type(e).__name__}: {e})"
    out(res)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
