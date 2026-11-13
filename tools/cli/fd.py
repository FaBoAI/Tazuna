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
TOOL ON THE GRIPPER (a pen taped to it): tool attach / status / set / rest / detach / calibrate. While a tool is
  attached "tip" is the TOOL tip, the gripper is locked, pick / take / do / probe / bench / park are refused,
  `ready` puts the tool tip at the stand-by point and `tool rest` is what folds the arm away.
  fd tool calibrate              measure the pen on the paper (tip, contact height, map, stroke direction)
  fd draw cat [--dry]            draw tools/draw/drawings/cat.json with it: photo, layout, preview, drawing, photo check
CALIBRATION / SERVICE: ref, calib-at, calib-fit, calib-roll, table-fit, bench, selftest, config, camera, report.

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
RUNS = os.path.join(ROOT, "runs")
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
        raise SafetyError(f"`fd --sim {a.cmd}` would {what}: --sim does not protect that. "
                          + (instead or f"Run `fd {a.cmd}` without --sim if that is what you want"))


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


def _tool_refusal(arm, what: str):
    """The guard at the top of every command that would move the gripper, fold the arm as if nothing were
    taped to it, or poke the fingertip at something: with a tool attached (arm.tool) the complete refusal
    result ("stage": "tool", "next" names `fd tool rest` / `fd tool detach`), else None. Nothing is written
    to the servo bus before it."""
    tool = getattr(arm, "tool", None)
    if tool is None:
        return None
    import tooling
    return tooling.refusal(tool, what)


def _tool_no_contact(arm, a, what: str):
    """Guard of the low-level motion commands: with a tool attached, --contact (the tip up to contact_depth_m
    below the table height) is refused. That is how a pen was pressed into the sheet by hand-driven steps on
    2026-11-11; lowering a tool onto the paper belongs to the commands that stop by what the camera sees."""
    tool = getattr(arm, "tool", None)
    if tool is None or not getattr(a, "contact", False):
        return None
    why = (f"tool '{tool.name}' is attached: `{what} --contact` would let its tip go up to "
           f"{float(arm._c('contact_depth_m')) * 1000:.0f} mm below the table height by hand. Nothing moved. Without "
           "--contact the tip can be brought down to the table height; pressing a tool onto the paper is left to "
           "`fd draw` / `fd tool calibrate`, which stop by what the camera sees")
    return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why, "tool": tool.name,
            "next": "omit --contact (the tool tip then stops at the table height), or use `fd draw` / `fd tool calibrate`",
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
    blocked = _tool_refusal(arm, "pick")
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
    blocked = _tool_refusal(arm, "take")
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
    if stage == "start":
        return ("nothing moved. Use a validated speed tier (omit --tier), or validate this one on the real arm with "
                "`fd bench --tier T --accept`")
    if stage == "perceive" and reason.startswith("container not found"):
        return ("nothing was taken. The camera does not recognise a container here (it must be dark, open at the top "
                "and not hidden): `fd scene` shows what it sees")
    if stage == "perceive" and reason.startswith("not in a container"):
        return "nothing was taken. The object is not inside a container: `fd pick` moves things that stand on the table"
    if stage == "perceive" and "objects inside the container match" in reason:
        return "several objects fit the description: take one with `fd take --id N` (ids in \"scene\")"
    if stage == "perceive":
        return "nothing was taken. Check \"scene\" for what the camera sees inside the container; take by id (`fd take --id N`)"
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
                "repeat blindly: compare jaws and object by hand in ONE photo (hover with `fd tip`, then `fd snap --crop "
                "x0,y0,x1,y1 --scale 4 --plain`); a steady offset of more than 1 cm means the calibration is off")
    if stage == "visual":
        return ("the arm went back to the stand-by pose WITHOUT descending: the camera could not measure where the jaws "
                "are (something pale behind them, glare, or a jaw hidden). Change that and repeat; by hand: hover with "
                "`fd tip`, compare jaws and object in ONE photo (`fd snap --crop x0,y0,x1,y1 --scale 4 --plain`)")
    if state == "holding_object":
        return ("the arm stopped while HOLDING the object: inspect with `fd state` / `fd snap`, then move it by hand "
                "with the low-level commands (`fd tip`, `fd grip 0.5`) or ask a person; do not use `park` yet")
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
        return "the object was dropped on the way: `fd scene` shows where it lies now (then `fd pick` or `fd take`)"
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
                "`fd bench --tier T --accept`")
    if stage == "perceive" and "objects match" in reason:
        return "several objects fit the description: pick one with `fd pick --id N` (ids in \"scene\"), or add --all"
    if stage == "perceive":
        return "check \"scene\" for what the camera sees; pick by id (`fd pick --id N`) or look at the saved scene image"
    if stage == "plan":
        return "no safe grasp was found (see \"grasp_candidates\"); the object may be out of reach or boxed in"
    if stage == "visual":
        return ("the arm went back to the ready pose. The jaw position could not be confirmed by the camera for a grasp "
                "with little room: move the neighbouring objects apart (or the object away from the box), or repeat "
                "with `--visual off` to go by the joint angles alone (it may touch the neighbour)")
    if state == "holding_object":
        return ("the arm stopped while HOLDING the object: inspect with `fd state` / `fd snap`, then move it by hand "
                "with the low-level commands (`fd tip`, `fd grip 0.5`) or ask a person; do not use `park` yet")
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
        res = {"ok": False, "need_llm": True, "reason": r["reason"], "text": a.text,
               "next": "choose the object yourself from \"objects\" and call `fd pick --id N` "
                       "(add `--into-id M` if there are several containers)"}
        if r.get("action") == "take_out":
            res["next"] = ("choose the object yourself from \"objects\" (those with \"in_container\") and call "
                           "`fd take --id N` (it is put on a free spot of the table; `--to x,y` names the spot)")
        if "one container into another" in r["reason"]:
            res["next"] = ("two containers are named: that is two steps for the operator. `fd take --id N` takes the "
                           "object out of the first one (onto the table), then `fd pick --id M --into-id K` puts it "
                           "into the second (ids in \"objects\"; they are counted anew after every command)")
        if r.get("not_plain"):
            # a prohibition, an exclusion, a condition, a question, a statement, a request with a second part:
            # the object and container named in it say nothing about what is wanted
            res["not_plain"] = r["not_plain"]
            res["next"] = ("nothing was moved: the sentence is not a plain command (" + str(r["not_plain"]) + "). Read it "
                           "yourself. If it asks for no motion, answer the user; if it does, choose the object from "
                           "\"objects\" and use `fd pick --id N` / `fd take --id N` for exactly the part that is wanted")
        try:
            sc = cmd_scene(arm, a)
            res.update(objects=sc["objects"], warnings=sc["warnings"], view=sc["view"])
        except Exception as e:  # the scene is a convenience here
            res["scene_error"] = f"{type(e).__name__}: {e}"
        return res
    blocked = _tool_refusal(arm, "do")  # (a sentence that is not run at all was answered above)
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
    blocked = _tool_refusal(arm, "repl")  # (its requests pick or take, and it parks on quit)
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
    stand-by point, lifted first when it is low and never lower than 2 cm above the table on the way."""
    import skills
    r = skills.go_ready(arm, a.tier)
    res = {"joint_err_deg": r["joint_err_deg"], "duration_s": r.get("duration_s"), **brief_state(arm)}
    if r.get("tool"):
        res.update({k: r[k] for k in ("tool", "tool_tip_xyz", "standby_xyz", "standby_pitch_deg", "standby_moved_m",
                                      "via", "lifted_first", "min_tip_height_m") if k in r})
    return res


def cmd_park(arm, a):
    """Return to the folded rest pose, switch torque off, put the camera back to auto."""
    blocked = _tool_refusal(arm, "park")
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


def _tool_arm_facts(a, tool, notes: list) -> dict:
    """Read-only look at the arm for `tool attach` / `tool set`: the gripper position, where this tool's tip
    is now, and whether the arm has a rest and a stand-by pose with it. The servo bus may be unavailable
    (arm off, somebody teleoperating): the tool is recorded all the same, since that only adds refusals."""
    import tooling
    if getattr(a, "no_robot", False):
        return {}
    try:
        with make_arm(a.sim) as arm:
            facts = {"gripper_raw": int(arm.bus.read("gripper", "Present_Position"))}
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
    if os.path.exists(path) and not a.replace:
        try:
            name = tooling.load(path).name
        except SafetyError:
            name = "?"
        why = f"tool '{name}' is already attached"
        return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why, "tool": name,
                "next": "`fd tool set ...` changes it; `fd tool detach` (after taking it off) removes it; "
                        "`fd tool attach --replace ...` replaces it (its calibration is lost)",
                "say": f"すでにツール「{name}」が登録されています。変更は fd tool set、付け替えは --replace を使ってください。"}
    tip, mount = _tool_spec(a, cfg)
    if tip is None:
        raise SafetyError("tool attach needs the tip: --tip x,y,z | --along-fingers LEN | --across-fingers LEN (metres)")
    name = str(a.name).strip()
    if not name:
        raise SafetyError("tool attach needs a name (--name pen)")
    pitch = tooling.check_pitch(a.pitch if a.pitch is not None else tooling.DEFAULT_PITCH[mount])
    tool = tooling.Tool(name=name, tip_offset=tip, mount=mount, pitch_deg=pitch,
                        attached_at=time.strftime("%Y-%m-%dT%H:%M:%S"), notes=[str(n) for n in (a.note or [])])
    notes = []
    facts = _tool_arm_facts(a, tool, notes)
    tool.gripper_raw = facts.pop("gripper_raw", None)
    if not a.dry:
        tooling.save(tool, a.sim)
    res = {"ok": True, "stage": "planned" if a.dry else "attached", "tool": _tool_brief(tool, cfg), "file": path,
           "saved": not a.dry, "arm": facts, "notes": notes,
           "locked": ["grip", "grasp", "pick", "take", "do", "repl", "probe", "bench", "park", "calib-at", "servo-tip",
                      "torque off (outside the tool's rest pose)"],
           "next": "`fd ready` puts the TOOL tip at the stand-by point; `fd tool calibrate` measures the tip on paper "
                   "(needed before `fd draw`); `fd tool rest` folds the arm away with the tool on",
           "say": f"ツール「{name}」を登録しました（先端はグリッパーから {tooling.length_of(cfg, tool) * 1000:.0f} mm）。"
                  "グリッパーはロックされ、park の代わりに fd tool rest を使います。"}
    if a.dry:
        res["say"] = f"ツール「{name}」を登録する内容を確認しました（保存していません）。"
    return res


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
        return {"ok": True, "stage": "status", "attached": False, "tool": None, "file": tooling.tool_path(a.sim),
                "next": "nothing is attached: every command works as usual. With something taped to the gripper: "
                        "`fd tool attach --name N (--tip x,y,z | --along-fingers LEN | --across-fingers LEN)`",
                "say": "ツールは登録されていません（通常どおり動きます）。"}
    cal = tooling.calibration_state(tool)
    res = {"ok": True, "stage": "status", "attached": True, "tool": _tool_brief(tool, cfg), "file": tooling.source(a.sim),
           "locked": ["grip", "grasp", "pick", "take", "do", "repl", "probe", "bench", "park", "calib-at", "servo-tip",
                      "torque off (outside the tool's rest pose)"]}
    if a.sim:
        res["simulated_tool"] = tooling.source(True) == tooling.tool_path(True) and not os.environ.get("FD_TOOL")
    if all(cal.values()):
        res["next"] = "calibrated: `fd draw FILE` draws; `fd ready` stands by; `fd tool rest` folds away"
        res["say"] = f"ツール「{tool.name}」が付いています（校正済み）。"
    else:
        missing = ", ".join(k for k, v in cal.items() if not v)
        res["next"] = (f"not calibrated ({missing}): run `fd tool calibrate` before `fd draw`; `fd ready` stands by; "
                       "`fd tool rest` folds away")
        res["say"] = f"ツール「{tool.name}」が付いています（未校正: {missing}）。描く前に fd tool calibrate が必要です。"
    return res


def _tool_set(a):
    """Change what is recorded about the attached tool. A new tip, mount or working pitch makes the
    measured contact plane and correction map worthless: they are dropped unless --keep-calibration."""
    import tooling
    if not a.sim:
        require_rig_config()      # (as in _tool_attach)
    cfg = load_config()
    tool = tooling.active(a.sim)
    if tool is None:
        why = "no tool is attached"
        return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why,
                "next": "`fd tool attach --name N (--tip x,y,z | --along-fingers LEN | --across-fingers LEN)`",
                "say": "ツールが登録されていません。先に fd tool attach を実行してください。"}
    before = _tool_brief(tool, cfg)
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
    facts = _tool_arm_facts(a, tool, notes) if geometry else {}
    facts.pop("gripper_raw", None)  # the position at ATTACH time stays the reference
    if changed and not a.dry:
        tooling.save(tool, a.sim)
    return {"ok": True, "stage": "planned" if a.dry else "set", "changed": changed, "saved": bool(changed) and not a.dry,
            "calibration_dropped": dropped, "tool": _tool_brief(tool, cfg), "before": before, "arm": facts,
            "notes": notes, "file": tooling.tool_path(a.sim),
            "next": ("the measured contact height and correction map were dropped: run `fd tool calibrate` before "
                     "`fd draw`" if dropped else "`fd tool status` shows the tool"),
            "say": (f"ツール「{tool.name}」の設定を変更しました（{', '.join(changed)}）。" if changed else
                    f"ツール「{tool.name}」の設定は変わっていません。")
                   + ("校正はやり直しが必要です。" if dropped else "")}


def _tool_rest(a):
    """Fold the arm away with the tool on and switch the torque off (what `park` is without a tool)."""
    import tooling
    with make_arm(a.sim) as arm:
        if arm.tool is None:
            why = "no tool is attached"
            return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why,
                    "next": "`fd park` folds the arm away", "say": "ツールが登録されていません。収納は fd park です。"}
        q0 = arm.read_q()
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
    if res.get("stage") == "planned":
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
    return res


def _tool_detach(a):
    """Forget the tool: the gripper, `park` and the fingertip as "tip" are back. Run it only when the tool
    is really off the gripper: nothing here can see that."""
    import tooling
    path = tooling.tool_path(a.sim)
    if not os.path.exists(path):
        res = {"ok": True, "stage": "detached", "attached": False, "removed": False, "file": path,
               "say": "ツールは登録されていませんでした。"}
        if a.sim and tooling.source(True):
            res["note"] = ("no simulated tool was attached; the simulation mirrors the real arm, which carries one "
                           "(`fd tool status`): detach that without --sim once it is off the gripper")
        return res
    try:
        name = tooling.load(path).name
    except SafetyError:
        name = "?"
    if not a.dry:
        tooling.clear(a.sim)
    return {"ok": True, "stage": "planned" if a.dry else "detached", "attached": bool(a.dry), "removed": not a.dry,
            "tool": name, "file": path,
            "next": "the gripper and `park` are unlocked and \"tip\" is the fingertip again: make sure the tool is "
                    "really off the gripper before anything moves",
            "say": (f"ツール「{name}」の登録を外す内容を確認しました（まだ外していません）。" if a.dry else
                    f"ツール「{name}」の登録を外しました。グリッパーと park のロックは解除されています。"
                    "ツールが実際に外れていることを確認してから動かしてください。")}


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


def _tool_calibrate(a):
    """`fd tool calibrate [--stage tip|contact|map|direction|all]`: measure the attached tool on the paper
    (penskill.calibrate): the tip from the first ink of a ladder, the contact plane, the map commanded ->
    actual, the direction of travel that keeps its ink. Every stage stores its result in the tool file. It
    lowers the tool onto the paper by the stop rule only: no step deeper without a photo that earned it."""
    import penskill
    try:
        opts = _pen_options(a)
    except SafetyError as e:
        return _pen_refused(e, "calibrate")
    stages = penskill.STAGES if a.stage == "all" else (a.stage,)
    with make_arm(a.sim) as arm:
        try:
            sess, close = _pen_session(arm, a)
        except SafetyError as e:
            return _pen_refused(e, "calibrate")
        try:
            _term_as_interrupt()
            res = penskill.calibrate(sess, stages, opts)
        finally:
            close()
        res["file"] = __import__("tooling").source(a.sim)
        if arm.tool is not None:
            res["calibrated"] = __import__("tooling").calibration_state(arm.tool)
    return res


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
        return penskill.draw(sess, a.drawing, opts)
    finally:
        close()


# sub-command -> function(args). The arm is opened only where it is needed (`rest`; a read-only look in
# `attach` / `set`), so `status` and `detach` work with the arm switched off.
TOOL_COMMANDS = {"attach": _tool_attach, "status": _tool_status, "set": _tool_set, "rest": _tool_rest,
                 "detach": _tool_detach, "calibrate": _tool_calibrate}


def cmd_tool(arm, a):
    """A tool taped to the gripper: attach / status / set / rest / detach / calibrate."""
    return TOOL_COMMANDS[a.tool_cmd](a)


def cmd_bench(arm, a):
    """Free-space motion benchmark of a speed tier (hardware validation): moves to the ready pose
    at the default (validated) tier first, then runs the benchmark moves at the tier under test.
    Keep the space in front of the arm free (about 30 cm) and watch it."""
    blocked = _tool_refusal(arm, "bench")
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
            why = None if a.force else tooling.torque_off_block(arm)
            if why:
                return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": why, "message": why,
                        "tool": tool.name, "next": "`fd tool rest` (folds the arm away with the tool, then torque off)",
                        "say": f"ツール「{tool.name}」が付いたままこの姿勢でトルクを切ると腕が落ちてツールが机に当たるため、"
                               "実行しませんでした。fd tool rest で収納してください。"}
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
    blocked = _tool_refusal(arm, "grip")
    if blocked:
        return blocked
    return arm.set_gripper(a.frac)


def cmd_grasp(arm, a):
    blocked = _tool_refusal(arm, "grasp")
    if blocked:
        return blocked
    if getattr(a, "thin", False):
        return arm.pinch()
    if a.fast:
        return arm.grasp_fast()
    return arm.grasp(min_frac=a.min_frac)


def cmd_probe(arm, a):
    # probe_down lowers the "tip" until it lands and reads 3-18 mm deep: with a tool that is the tool, pushed in
    blocked = _tool_refusal(arm, "probe")
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
    blocked = _tool_refusal(arm, "calib-at")  # (it probes the table and photographs the closed fingertip)
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
    surface that was actually touched."""
    if not a.dry:
        _not_with_sim(a, "write table_z, floor_z and table_plane into the rig's real configuration file",
                      "Use --dry to see the fit without saving, or run it without --sim")
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
    blocked = _tool_refusal(arm, "servo-tip")  # (it steers by the picture of the closed fingertip)
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
    s = sub.add_parser("tool", help="a tool taped to the gripper (a pen): attach / status / set / rest / detach")
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
    t = tsub.add_parser("rest", help="fold the arm away with the tool on, then torque off (instead of `park`)")
    t.add_argument("--tier", default=None)
    t.add_argument("--clearance", type=float, default=0.05, help="tool tip above the table in the rest pose (m)")
    t.add_argument("--pan", type=float, default=20.0, help="base angle of the rest pose (deg)")
    t.add_argument("--keep-torque", action="store_true", help="stay folded with the torque on")
    t.add_argument("--dry", action="store_true", help="plan and check only")
    t = tsub.add_parser("detach", help="forget the tool (ONLY when it is off the gripper): unlocks gripper and park")
    t.add_argument("--dry", action="store_true")
    t = tsub.add_parser("calibrate", help="measure the tool on the paper: tip, contact height, correction map, "
                                          "stroke direction (stops by what the camera sees; needed before `fd draw`)")
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


def main(argv=None):
    a = build_parser().parse_args(argv)
    fn, needs_arm = COMMANDS[a.cmd]
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
            raise SafetyError(f"`fd {a.cmd} --frame` replays saved images, and without --sim that would move the "
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
    except (SafetyError, MotionAborted, BusError) as e:
        res = {"ok": False, "error": type(e).__name__, "message": str(e)}
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
