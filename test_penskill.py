"""Tests of the pen skill (penskill.py: `fd tool calibrate`, `fd draw`) and of the two guards of stage 2
(`fd --sim park` / `repl` leave the real camera alone; the simulated tool file never lies in the tree).

Run: python test_penskill.py [name-part ...]        (no hardware: simulated servo bus, simulated paper)

What they show (SPEC section 7 "DRAW" and 9.2; everything on simpaper.SimPaper with hidden truths):
 - attach -> calibrate (tip, contact, map, direction) -> `draw cat` in a world like the night's (tool offset
   wrong by 15 mm sideways and 8 mm in length, position sheared by 10 %, contact tilted by 1.5 mm across the
   box, height wander 1.5 mm): coverage >= 0.9, mean offset <= 2 mm, test marks outside the drawing box;
 - with the desk exposure calibration and drawing stop at "visual" before the pen touches anything;
 - with a detector forced to answer "no ink" the search stops within the depth limit of the stop rule;
 - --dry moves nothing and returns the plan / the preview image;
 - a tool without calibration is refused with the command to run;
 - the direction test recovers a hidden "tip-trailing skips" and a hidden "tip-leading skips" truth, and
   `--strokes auto` then cuts and turns the pieces accordingly;
 - calibration works with photo noise of 0.8 grey levels and stops at "visual" with more (it never guesses);
 - every stop leaves the pen lifted and says where it is; a lift that fails is reported as such;
 - the commands as the operator types them (`fd --sim tool attach / calibrate --truth / draw`).

Every test uses $FD_TOOL (a scratch file) and a scratch run directory: nothing is written into the tree.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "pen_20261111", "camera_model.json")
_TMP = tempfile.mkdtemp(prefix="jh_penskill_")
TOOL_PATH = os.path.join(_TMP, "tool_active.json")
os.environ["FD_TOOL"] = TOOL_PATH

# ---- nothing here may touch the arm or the camera (SPEC 0.1): fail loudly if anything tries
_TRIPPED = []


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith(("/dev/ttyACM", "/dev/ttyUSB", "/dev/video")):
            _TRIPPED.append((event, p))
            raise RuntimeError(f"a test opened {p}")
    elif event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"):
        s = repr(args)
        if "v4l2-ctl" in s or "/dev/video" in s or "/dev/ttyACM" in s:
            _TRIPPED.append((event, s))
            raise RuntimeError(f"a test ran {s}")


sys.addaudithook(_audit)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _no_capture(*a, **k):
    _TRIPPED.append(("cv2.VideoCapture", a))
    raise RuntimeError("a test opened the camera")


cv2.VideoCapture = _no_capture

import camera  # noqa: E402
import fd  # noqa: E402
import ink as INK  # noqa: E402
import kinematics as K  # noqa: E402
import penskill as PS  # noqa: E402
import simpaper as SP  # noqa: E402
import strokes as ST  # noqa: E402
import tooling as TL  # noqa: E402
from robot import Arm, MotionAborted, SafetyError, load_config  # noqa: E402
from sim_bus import SimBus  # noqa: E402

fd.log_event = lambda *a, **k: None        # no run log from tests
fd.RUNS = os.path.join(_TMP, "runs")       # photos of `fd` go to the scratch directory, not into the tree
_TREE_BEFORE = {d: sorted(os.listdir(os.path.join(HERE, d))) if os.path.isdir(os.path.join(HERE, d)) else None
                for d in ("runs", "backups", ".")}
CFG = load_config()
CAM = camera.CameraModel.load()
TZ = float(CFG["table_z"])
IMG = os.path.join(_TMP, "img")
NIGHT = {"night": True, "seed": 3, "noise": 0.6}      # SPEC section 7 "DRAW": the hidden truths of that evening
_CACHE = {}


# ----------------------------------------------------------------------------- helpers
def pen0():
    """The pen as the operator would attach it: "about 5.5 cm beyond the fingertip, along the fingers"."""
    return TL.Tool(name="pen", tip_offset=TL.offset_for(CFG, "along_fingers", 0.055), mount="along_fingers",
                   pitch_deg=75.0, gripper_raw=2061, attached_at="2026-11-12T03:00:00")


def world(truth=None, tool=None, state=None):
    """A simulated arm as fd.make_arm(True) builds it (limp, in the tool's rest pose) with `tool` in the
    tool file, the simulated sheet / pen / camera with hidden truths, and the session of a command."""
    TL.clear()
    TL.save(tool if tool is not None else pen0())
    arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=TL.active())
    assert TL.sim_start(arm)
    arm.open()
    paper, photo = PS.sim_world(arm, CAM, truth, state)
    sess = PS.PenSession(arm, CAM, photo, image_dir=IMG, sim=True, paper=paper)
    return arm, paper, sess


def check_result(r, skill):
    """What every result of the pen skill has (SPEC 0.7): ok, stage, reason, next, a Japanese say, timings,
    photos, and where the pen was left."""
    for k in ("ok", "skill", "stage", "reason", "next", "say", "timings", "photos", "tool"):
        assert k in r, (k, sorted(r))
    assert r["skill"] == skill and isinstance(r["say"], str) and r["say"]
    assert "シミュレーション" in r["say"] or "実機" in r["say"] or r["stage"] == "tool", r["say"]
    if r["stage"] != "tool":
        assert {"tip_xyz", "height_mm", "lifted", "where", "torque"} <= set(r["pen"]), r.get("pen")
    json.dumps(r, default=fd._jsonable)                      # what fd prints and logs


def calibrated():
    """One full calibration in the night's world, shared by the tests that start from a calibrated pen."""
    if "cal" not in _CACHE:
        arm, paper, sess = world(NIGHT)
        io0 = dict(arm.bus.io)
        t0 = time.time()
        r = PS.calibrate(sess, PS.STAGES, PS.PenOptions())
        _CACHE["cal"] = {"res": r, "tool": json.loads(json.dumps(arm.tool.to_dict())), "state": PS.sim_state(paper, NIGHT),
                         "truth_tip": paper.true_tip_offset.copy(), "max_depth": paper.max_depth_m, "arm": arm, "paper": paper,
                         "wall": time.time() - t0, "io0": io0}
    return _CACHE["cal"]


def calibrated_world(truth, sheet="blank"):
    """A fresh arm carrying the calibrated pen of calibrated(), in a world with the same REAL pen."""
    c = calibrated()
    assert c["res"]["ok"], c["res"]
    state = {"true_tip_offset": [float(v) for v in c["truth_tip"]], "paper": c["state"]["paper"] if sheet == "used" else None}
    return world(truth, TL.Tool.from_dict(c["tool"]), state)


def fd_run(argv):
    assert argv[0] == "--sim", "tests run the simulated arm only"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fd.main(list(argv))
    return json.loads(buf.getvalue().strip().splitlines()[-1]), rc


def inside(xy, corners, margin=0.0):
    """Is a table point inside the (convex) box, `margin` away from its edges?"""
    c = np.asarray(corners, float)
    for a, b in zip(c, np.roll(c, -1, axis=0)):
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        if float((np.asarray(xy) - a) @ n) * np.sign(float((c.mean(axis=0) - a) @ n)) < margin:
            return False
    return True


# ============================================================================= 1. refusals without motion
def test_01_no_tool_and_uncalibrated_tool_are_refused_with_the_command_to_run():
    TL.clear()
    arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=TL.active()).open()
    sess = PS.PenSession(arm, CAM, lambda e="paper": (_ for _ in ()).throw(AssertionError("a photo was taken")), sim=True)
    for fn, args in ((PS.draw, ("cat",)), (PS.calibrate, ())):
        r = fn(sess, *args)
        assert r["ok"] is False and r["stage"] == "tool" and "fd tool attach" in r["next"], r
    arm, paper, sess = world(NIGHT)
    io0 = dict(arm.bus.io)
    r = PS.draw(sess, "cat")
    assert r["ok"] is False and r["stage"] == "tool" and "fd tool calibrate" in r["next"], r
    assert "contact" in r["reason"] and "map" in r["reason"] and "fd tool calibrate" in r["say"]
    assert arm.bus.io["writes"] == io0["writes"] and not arm.is_torqued() and paper.samples == 0
    # contact alone is not enough either
    arm.tool.contact = {"x0": 0.22, "y0": 0.0, "a": TZ, "bx": 0.0, "by": 0.0}
    r = PS.draw(sess, "cat")
    assert r["stage"] == "tool" and "map" in r["reason"] and "contact" not in r["reason"].split("(")[1], r
    check_result(r, "draw")


def test_02_options_cannot_loosen_the_stop_rule():
    PS.PenOptions().check()
    for kw in ({"press_m": 0.0041}, {"rung_dz_m": 0.002}, {"rungs": 5}, {"max_search_below_m": 0.008},
               {"after_ink_max_m": 0.004}, {"mark_press_m": 0.0035}, {"strokes": "pull"}, {"end": "park"},
               {"marks_on": "box"}, {"places": 2}, {"at": ("a", 1)}, {"press_m": float("nan")}):
        try:
            PS.PenOptions(**kw).check()
        except SafetyError:
            continue
        raise AssertionError(f"accepted {kw}")
    assert PS.MAX_RUNGS == 4 and PS.MAX_RUNG_DZ_M == 0.0015 and PS.MAX_SEARCH_BELOW_M == 0.006
    assert PS.MAX_AFTER_INK_M == 0.003 and PS.MAX_PRESS_M == 0.004 and PS.PenOptions().press_m == 0.0023
    # the internal guard: whatever a stage computes, a height below the limit is refused
    arm, paper, sess = world(None)
    run = PS._Run(sess, PS.PenOptions(), INK)
    run.guard(-0.003, -0.003, "x")
    try:
        run.guard(-0.0031, -0.003, "x")
    except SafetyError as e:
        assert "depth limit" in str(e)
    else:
        raise AssertionError("the depth guard let a height below the limit pass")


# ============================================================================= 2. the night's world, end to end
def test_03_attach_calibrate_in_the_nights_world():
    """Tool offset wrong by 15 mm sideways and 8 mm in length, a sheared position error of 10 %, contact
    tilted by 1.5 mm across the box, height wander 1.5 mm: all four stages pass and the tool file holds
    what they measured."""
    c = calibrated()
    r, arm, paper = c["res"], c["arm"], c["paper"]
    check_result(r, "calibrate")
    assert r["ok"] and r["stage"] == "done", (r["stage"], r["reason"])
    st = r["stages"]
    assert all(st[k]["ok"] for k in PS.STAGES), {k: st[k].get("reason") for k in st}
    tool = TL.active()
    assert TL.calibration_state(tool) == {"tip": True, "contact": True, "map": True}
    assert set(tool.calibrated) >= {"tip", "contact", "map", "direction"} and all("at" in tool.calibrated[k] for k in PS.STAGES)
    # tip: the ink was 12 to 16 mm from the aim; height and sideways error of the tool are found. (What is
    # left is the shear at the place of the ladder, which the tip correction cannot tell from an offset.)
    before, after = np.array(pen0().tip_offset), np.array(tool.tip_offset)
    assert 0.010 < st["tip"]["shift_mm"] / 1000.0 < 0.022, st["tip"]
    assert abs((after - before)[2] - 0.008) < 0.003, (after - before)                 # 8 mm longer than recorded
    assert abs(after[2] - c["truth_tip"][2]) < 0.0025 and abs(before[2] - c["truth_tip"][2]) > 0.0075   # the length is found
    # the pen touched while the model had it 7.7 mm up (less where the sheet lies lower, minus one ladder step)
    assert 2.5 <= st["tip"]["first_ink_plan_z_mm"] - TZ * 1000.0 <= 9.5, st["tip"]
    # contact: four places, a plane within the limit; with the corrected tip it lies near the table height
    con = tool.contact
    assert st["contact"]["n"] >= 3 and con["rms_mm"] <= 1.5 and abs(con["a"] - TZ) < 0.004, con
    assert len(con["points"]) == st["contact"]["n"] and tool.region is not None
    # map: at least 6 marks, the 10 % shear is in it
    m = tool.corr_map
    assert m["kind"] == "affine" and m["n"] >= 6 and m["rms_mm"] <= 3.0, m
    assert st["map"]["offset_mm"]["max"] > 5.0                                        # the marks landed far from their aims
    # ... and the map knows where the ink lands inside the box (against the truth of the simulated pen)
    bc, bh = np.array(r["box"]["centre"]), r["box"]["size_mm"] / 2000.0
    worst = 0.0
    for dx in (-0.8, 0.0, 0.8):
        for dy in (-0.8, 0.0, 0.8):
            cmd = bc + bh * np.array([dx, dy])
            q = arm.solve(np.array([cmd[0], cmd[1], TL.contact_z(tool, cmd)]), 75.0, 0.0)
            worst = max(worst, float(np.linalg.norm(paper.true_tip(q, 0.0)[:2] - ST.apply_map([cmd], m)[0])))
    # (an affine map cannot follow the tool's residual offset as it turns with the base: 1 mm at the marks,
    # more in the corners of the box; the drawing's mean offset is what the acceptance asks for, test_04)
    assert worst < 0.006, worst
    # direction: no skipping in this world -> no preference, and that is what is stored
    d = tool.calibrated["direction"]
    assert d["prefer"] == "any" and set(d["missing"]) == {"away", "toward", "across"} and max(d["missing"].values()) < 0.15, d
    # the stop rule in numbers: nothing was sent lower than 6 mm under the expected paper before ink was
    # seen, and the real pen never pressed deeper than the wander plus the allowed 3 mm (+ one ladder step)
    assert r["pen"]["lowest_plan_z_mm"] >= TZ * 1000.0 - 9.0, r["pen"]
    assert c["max_depth"] < 0.0075, c["max_depth"]
    # the pen is up and aside, torque on, and the result says so
    assert r["pen"]["where"] == "aside" and r["pen"]["lifted"] and r["pen"]["height_mm"] > 30 and arm.is_torqued()
    assert "紙の横" in r["say"] and "シミュレーション" in r["say"]
    # every test mark lies outside the box `fd draw` would use (the truth of the paper, not the photo)
    box = r["box"]
    corners = ST.box_corners(box["centre"], box["size_mm"] / 2000.0, box["rot_deg"])
    lines = paper.ink(new_only=True)
    assert len(lines) >= 12 and not any(inside(p, corners) for ln in lines for p in ln["xy"][::4]), "ink inside the box"
    assert len(r["photos"]) >= 12 and all(os.path.isfile(p) for p in r["photos"])
    assert arm.bus.pos["gripper"] == 2061.0                                          # the gripper was never commanded
    print(f"    calibrate: {c['wall']:.0f} s wall, {r['timings']['total']:.0f} s of robot time, {len(r['photos'])} photos; "
          f"tip shift {st['tip']['shift_mm']} mm, plane rms {con['rms_mm']} mm (n {con['n']}), map rms {m['rms_mm']} mm "
          f"(n {m['n']}; predicts the ink within {worst * 1000:.1f} mm), direction {d['prefer']} {d['missing']}, deepest real "
          f"press {c['max_depth'] * 1000:.1f} mm")


def test_04_draw_cat_after_calibration_meets_the_acceptance():
    """... and `draw cat` on the same sheet: coverage >= 0.9, mean offset <= 2 mm (photo AND truth)."""
    arm, paper, sess = calibrated_world(NIGHT, sheet="used")
    t0 = time.time()
    r = PS.draw(sess, "cat")
    check_result(r, "draw")
    assert r["ok"] and r["stage"] == "done", (r["stage"], r["reason"])
    cov = r["coverage"]
    assert cov["judgeable"] and cov["drawn_fraction"] >= 0.9 and cov["offset_mm"]["mean"] <= 2.0, cov
    assert r["drawn"]["failed"] == [] and r["plan"]["pieces"] == r["drawn"]["pieces"] >= 18
    lay = r["layout"]
    polys = ST.layout(ST.load("cat"), lay["centre"], lay["size_mm"] / 2000.0, lay["rot_deg"])
    truth = paper.coverage(polys, tol_m=0.004)
    assert truth["drawn_fraction"] >= 0.9 and truth["offset_mm"]["mean"] <= 2.0, truth
    assert abs(lay["size_mm"] - 105.0) < 0.5 and lay["sheet_found"] is True and lay["older_ink_pieces"] >= 12
    # the box is the one the calibration kept clean
    assert np.allclose(lay["centre"], calibrated()["res"]["box"]["centre"], atol=0.006), (lay, calibrated()["res"]["box"])
    # stop rule 4: the drawing was planned press (2.3 mm) below the contact plane, nowhere deeper
    tool = arm.tool
    zs = r["plan"]["plan_z_mm"]
    pts = np.concatenate(polys)
    zc = np.array([TL.contact_z(tool, p) for p in pts]) * 1000.0
    assert zs[0] >= zc.min() - 2.3 - 0.5 and zs[1] <= zc.max() - 2.3 + 0.5 and r["plan"]["press_mm"] == 2.3, (zs, zc.min(), zc.max())
    assert r["pen"]["lowest_plan_z_mm"] >= zs[0] - 0.01
    # no direction was measured as better: drawn as the file has it, and the result says why
    assert r["plan"]["strokes"] == "as-is" and r["plan"]["strokes_source"] == "measured" and any("direction test" in n for n in r["notes"])
    assert r["tracking"]["path_err_mm"]["max"] < 2.0 and r["plan"]["fine_mode"] is False
    assert os.path.isfile(r["preview"]) and r["pen"]["where"] == "aside" and r["pen"]["lifted"]
    print(f"    draw cat: {time.time() - t0:.0f} s wall, {r['timings']['total']:.0f} s of robot time; photo: drawn "
          f"{cov['drawn_fraction']:.3f}, offset mean {cov['offset_mm']['mean']} mm; truth: drawn {truth['drawn_fraction']:.3f}, "
          f"offset {truth['offset_mm']['mean']:.2f} mm; deepest real press {paper.max_depth_m * 1000:.1f} mm")
    # --end rest folds the arm away with the tool on
    arm, paper, sess = calibrated_world(NIGHT)
    r = PS.draw(sess, "cat", PS.PenOptions(size_mm=60.0, end="rest"))
    assert r["ok"] and r["pen"]["where"] == "rest" and r["pen"]["torque"] == "off" and not arm.is_torqued(), r["pen"]
    assert abs(r["layout"]["size_mm"] - 60.0) < 0.5 and "収納" in r["say"]


def test_05_dry_runs_move_nothing_and_return_the_plan():
    arm, paper, sess = world(NIGHT)
    io0, n0 = dict(arm.bus.io), paper.samples
    r = PS.calibrate(sess, PS.STAGES, PS.PenOptions(dry=True))
    check_result(r, "calibrate")
    assert r["ok"] and r["stage"] == "planned" and os.path.isfile(r["preview"]), r
    assert arm.bus.io["writes"] == io0["writes"] and not arm.is_torqued() and paper.samples == n0
    assert r["pen"]["where"] == "not moved" and "動かしていません" in r["say"]
    # (the plan of a dry run is laid out on a photo taken with the arm where it stands: indicative, and it says so)
    assert list(r["plan"]) == list(PS.STAGES) and all(v["at"] for v in r["plan"].values()) and len(r["plan"]["map"]["at"]) >= 6, r["plan"]
    assert any("not from the side pose" in n for n in r["notes"]), r.get("notes")
    corners = ST.box_corners(r["box"]["centre"], r["box"]["size_mm"] / 2000.0, r["box"]["rot_deg"])
    assert not any(inside(p, corners) for v in r["plan"].values() for p in v["at"]), r["plan"]
    assert TL.active().contact is None and TL.active().tip_offset == pen0().tip_offset           # nothing stored
    # drawing
    arm, paper, sess = calibrated_world(NIGHT, sheet="used")
    io0, n0 = dict(arm.bus.io), paper.samples
    r = PS.draw(sess, "cat", PS.PenOptions(dry=True))
    check_result(r, "draw")
    assert r["ok"] and r["stage"] == "planned" and r["dry"] is True and os.path.isfile(r["preview"]), r
    assert arm.bus.io["writes"] == io0["writes"] and not arm.is_torqued() and paper.samples == n0
    assert r["plan"]["pieces"] >= 18 and 550 < r["plan"]["ink_mm"] < 950 and "coverage" not in r and "drawn" not in r
    # (the folded arm hides part of the sheet from the camera: the box of this dry run is smaller, and it says why)
    assert any("not from the side pose" in n for n in r["notes"]) and r["pen"]["where"] == "not moved"
    a, b = cv2.imread(r["preview"]), cv2.imread(r["photos"][0])
    assert a.shape == b.shape and np.count_nonzero(np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 120) > 1500   # the plan is drawn in
    # with the arm at the side pose (where `fd tool calibrate` leaves the real arm) the dry run shows the
    # layout of the real run: the full box, the place the calibration kept clean
    PS._Run(sess, PS.PenOptions(), INK).go_aside()
    io1, n1 = dict(arm.bus.io), paper.samples
    r = PS.draw(sess, "cat", PS.PenOptions(dry=True))
    assert r["ok"] and r["stage"] == "planned" and arm.bus.io["writes"] == io1["writes"] and paper.samples == n1
    assert abs(r["layout"]["size_mm"] - 105.0) < 0.5 and 800 < r["plan"]["ink_mm"] < 950, (r["layout"], r["plan"])
    assert np.allclose(r["layout"]["centre"], calibrated()["res"]["box"]["centre"], atol=0.006)
    assert not any("not from the side pose" in n for n in r["notes"]), r["notes"]


# ============================================================================= 3. when you cannot see, stop
def test_06_desk_exposure_stops_at_visual_before_touching():
    truth = dict(NIGHT, exposure="desk")
    arm, paper, sess = world(truth)
    before = TL.active().to_dict()
    r = PS.calibrate(sess, PS.STAGES, PS.PenOptions())
    check_result(r, "calibrate")
    assert r["ok"] is False and r["stage"] == "visual" and "clipped" in r["reason"], r
    assert os.path.isfile(r["photo"]) and "fd snap --exposure paper" in r["next"]
    assert paper.ink_length() == 0.0 and paper.max_depth_m == 0.0 and "lowest_plan_z_mm" not in r["pen"]
    assert r["pen"]["lifted"] and r["pen"]["height_mm"] > 30, r["pen"]             # (it went aside for the photo only)
    assert TL.active().to_dict() == before and r["stages"] == {}
    assert "判定できない" in r["say"]
    # a calibrated pen that is asked to draw under that light
    arm, paper, sess = calibrated_world(truth)
    r = PS.draw(sess, "cat")
    check_result(r, "draw")
    assert r["ok"] is False and r["stage"] == "visual" and paper.ink_length() == 0.0 and paper.max_depth_m == 0.0, r
    assert "coverage" not in r and r["pen"]["lifted"]


class NoInk:
    """ink.py with a ladder readout that always answers a confident "no ink" (what the detector of
    2026-11-11 did on clipped photos)."""

    def __init__(self):
        self.calls, self.untouched = [], []

    def __getattr__(self, name):
        return getattr(INK, name)

    def ladder_readout(self, after, cam, z, rungs, before=None, **kw):
        self.calls.append(rungs)
        self.untouched.append(INK.ladder_readout(after, cam, z, rungs, before=before)["untouched"])   # what was really there
        return {"rungs": [{"i": i, "inked": False, "mark": None, "why": "forced", "speck": False} for i in range(len(rungs))],
                "first_inked": None, "judgeable": True, "notes": [], "untouched": True, "offset_mm": None, "can": {"ok": True}}


def test_07_a_detector_forced_to_say_no_ink_stops_within_the_depth_limit():
    arm, paper, sess = world(NIGHT)
    liar = NoInk()
    r = PS.calibrate(sess, PS.STAGES, PS.PenOptions(), modules=(liar,))
    check_result(r, "calibrate")
    assert r["ok"] is False and r["stage"] == "tip" and "does not go deeper" in r["reason"], r
    lad = r["stages"]["tip"]["ladders"]
    zs = [z for ld in lad for z in ld["plan_z_mm"]]
    # ladders of at most 4 dashes, 1.5 mm apart, from 12 mm above the expected paper down to 6 mm below it
    assert all(len(ld["plan_z_mm"]) <= 4 for ld in lad) and len(lad) == len(liar.calls) == 4, lad
    assert np.allclose(np.diff(zs), -1.5) and abs(zs[0] - (TZ * 1000.0 + 12.0)) < 1e-6, zs
    assert min(zs) >= TZ * 1000.0 - 6.0 - 1e-6 and abs(r["pen"]["lowest_plan_z_mm"] - min(zs)) < 0.01, (zs, r["pen"])
    # each ladder on untouched paper (the ink of the earlier ones is really there): the true readout found
    # no older ink in any ladder's search window, and no two ladders share a place
    at = np.array([ld["at"] for ld in lad])
    assert liar.untouched == [True] * 4, liar.untouched
    assert len({tuple(a) for a in at.round(3).tolist()}) >= 3, at    # (a ladder that left no ink may be followed on its spot)
    assert paper.ink_length() > 0.02                                 # the pen DID write: the detector lied
    assert TL.active().contact is None and TL.active().tip_offset == pen0().tip_offset   # nothing was stored
    assert r["pen"]["lifted"] and r["pen"]["where"] == "aside" and set(r["stages"]) == {"tip"}
    # after ink was seen the limit is 3 mm below it: a liar cannot push the contact stage further either
    arm, paper, sess = calibrated_world(NIGHT)
    liar = NoInk()
    r = PS.calibrate(sess, ("contact",), PS.PenOptions(), modules=(liar,))
    assert r["ok"] is False and r["stage"] == "contact", r
    for pl in r["stages"]["contact"]["places"]:
        ref = TL.contact_z(TL.Tool.from_dict(calibrated()["tool"]), pl["wanted"]) * 1000.0
        low = min(z for ld in pl["ladders"] for z in ld["plan_z_mm"])
        assert pl["found"] is False and low >= ref - 3.0 - 0.006, (pl["wanted"], low, ref)      # (rounded to 0.01 mm)
    assert TL.active().contact == calibrated()["tool"]["contact"]    # the old plane was not replaced by a guess


def test_08_cannot_judge_on_a_rung_stops_everything():
    """Stop rule 3: one rung that cannot be judged ends the search: nothing more is drawn."""
    class Blind(NoInk):
        def ladder_readout(self, after, cam, z, rungs, before=None, **kw):
            out = NoInk.ladder_readout(self, after, cam, z, rungs, before)
            out["rungs"][1].update(inked=None, why="cannot judge: forced")
            out["judgeable"] = False
            return out

    arm, paper, sess = world(NIGHT)
    blind = Blind()
    r = PS.calibrate(sess, PS.STAGES, PS.PenOptions(), modules=(blind,))
    assert r["ok"] is False and r["stage"] == "visual" and len(blind.calls) == 1 and "cannot judge" in r["reason"], r
    assert os.path.isfile(r["photo"]) and r["pen"]["lifted"] and r["ladder"]["inked"][1] is None
    assert r["pen"]["lowest_plan_z_mm"] >= TZ * 1000.0 + 12.0 - 4.5 - 1e-6       # one ladder, from the top


def test_08b_a_lone_faint_trace_is_the_first_touch_and_nothing_goes_deeper():
    """A ladder whose only ink is a faint trace at one rung (the pen grazed the paper there; no proper ink
    to compare it with): the readout says "cannot judge" whether it is a graze or a faint line. Either way
    the pen touched at that height: the search ends there, and no ladder goes deeper on it."""
    class Graze(NoInk):
        def ladder_readout(self, after, cam, z, rungs, before=None, **kw):
            out = NoInk.ladder_readout(self, after, cam, z, rungs, before)
            aim = np.mean(np.array(rungs[3]), axis=0)
            mark = INK.Mark(xy=(float(aim[0]) + 0.001, float(aim[1]) - 0.012), length_m=0.0019, width_m=0.0005, angle_deg=90.0,
                            area_px=10, bbox_px=(0, 0, 1, 1), darkness=0.25, confidence=0.3)
            out["rungs"][3].update(inked=None, mark=mark, speck=True, why="cannot judge: a faint speck at rung 3's place")
            out["judgeable"] = False
            return out

    arm, paper, sess = world(NIGHT)
    graze = Graze()
    r = PS.calibrate(sess, ("tip",), PS.PenOptions(), modules=(graze,))
    check_result(r, "calibrate")
    tip = r["stages"]["tip"]
    assert r["ok"] and tip["ok"] and len(graze.calls) == 1 and tip["ladders"][0]["graze"] is True, (r["stage"], r["reason"])
    assert tip["ladders"][0]["inked"] == [False, False, False, None]
    assert abs(tip["first_ink_plan_z_mm"] - tip["ladders"][0]["plan_z_mm"][3]) < 1e-6
    assert abs(r["pen"]["lowest_plan_z_mm"] - tip["ladders"][0]["plan_z_mm"][3]) < 0.01          # nothing went deeper
    assert any("faint trace" in n for n in r["notes"]) and 11.0 < tip["shift_mm"] < 20.0, (r["notes"], tip["shift_mm"])
    # a trace on paper that was not untouched, or in a photo that cannot show ink, is still "cannot judge"
    class OnOldInk(Graze):
        def ladder_readout(self, after, cam, z, rungs, before=None, **kw):
            out = Graze.ladder_readout(self, after, cam, z, rungs, before)
            out["untouched"] = False
            return out

    arm, paper, sess = world(NIGHT)
    r = PS.calibrate(sess, ("tip",), PS.PenOptions(), modules=(OnOldInk(),))
    assert r["ok"] is False and r["stage"] == "visual" and TL.active().tip_offset == pen0().tip_offset, r


def test_09_photo_noise_up_to_0p8_works_and_more_stops_at_visual():
    arm, paper, sess = world(dict(NIGHT, noise=0.8))
    r = PS.calibrate(sess, ("tip", "contact", "map"), PS.PenOptions())
    assert r["ok"] and all(r["stages"][k]["ok"] for k in ("tip", "contact", "map")), (r["stage"], r["reason"])
    assert TL.active().corr_map["rms_mm"] <= 3.0 and TL.active().contact["rms_mm"] <= 1.5
    out = {}
    for noise in (1.6, 3.0):
        arm, paper, sess = world(dict(NIGHT, noise=noise))
        before = TL.active().to_dict()
        r = PS.calibrate(sess, PS.STAGES, PS.PenOptions())
        check_result(r, "calibrate")
        out[noise] = (r["stage"], sorted(r["stages"]), r["reason"][:80])
        assert r["ok"] is False and r["stage"] == "visual", (noise, r["stage"], r["reason"])
        # never a guess: what is in the tool file comes from stages whose photos could be read (they report
        # ok); the stage that could not be read stored nothing, and neither did anything after it
        now, done = TL.active().to_dict(), [k for k, v in r["stages"].items() if v.get("ok")]
        assert set(now["calibrated"]) == set(done) and all(v.get("ok") for v in r["stages"].values()), (noise, done, now["calibrated"])
        assert (now["tip_offset"] != before["tip_offset"]) == ("tip" in done)
        assert (now["contact"] is not None) == ("contact" in done) and (now["corr_map"] is not None) == ("map" in done)
        assert "map" not in done and "direction" not in done
        assert r["pen"]["lifted"] and r["pen"]["where"] == "aside"
    print(f"    noise 0.8: calibrated; {out}")


# ============================================================================= 4. the stroke direction is measured
def _direction_case(skips, want):
    truth = dict(NIGHT, skips=skips)
    arm, paper, sess = calibrated_world(truth)
    assert paper.skips == skips
    r = PS.calibrate(sess, ("direction",), PS.PenOptions())
    check_result(r, "calibrate")
    d = r["stages"]["direction"]
    assert r["ok"] and d["ok"] and d["prefer"] == want, (skips, d)
    other = "toward" if want == "away" else "away"
    assert d["missing"][other] > 0.2 and d["missing"][want] < 0.08 and d["missing"]["across"] < 0.15, d
    stored = TL.active().calibrated["direction"]
    assert stored["prefer"] == want and stored["missing"] == d["missing"] and "at" in stored
    assert set(TL.active().calibrated) >= {"tip", "contact", "map", "direction"}       # the other stages are kept
    # `--strokes auto` now cuts and turns the pieces that way
    pol = PS.stroke_policy(TL.active(), "auto")
    assert pol["policy"] == want and pol["source"] == "measured", pol
    r = PS.draw(sess, "cat")
    assert r["ok"] and r["plan"]["strokes"] == want and r["plan"]["strokes_source"] == "measured", (r["stage"], r["reason"])
    assert r["plan"]["pieces"] > 25 and r["coverage"]["drawn_fraction"] >= 0.9, (r["plan"], r["coverage"])
    lay = r["layout"]
    polys = ST.layout(ST.load("cat"), lay["centre"], lay["size_mm"] / 2000.0, lay["rot_deg"])
    plan = ST.plan_pieces(polys, ST.direction_lean(want), TL.active().corr_map)
    bad = ST.radial_lean(outward=(want == "toward"))        # the direction of travel that loses ink
    assert max(ST.push_excursion(pc["ink"], bad) for pc in plan["pieces"]) <= 0.006 + 1e-9
    assert plan["stats"]["count"] == r["plan"]["pieces"]
    good_cov, good_ink = r["coverage"]["drawn_fraction"], paper.ink_length()
    # the same drawing forced the other way loses ink (that is what the test measured)
    arm, paper, sess = calibrated_world(truth)
    tool = TL.active()
    tool.calibrated["direction"] = stored
    TL.save(tool)
    arm.tool = TL.active()
    r2 = PS.draw(sess, "cat", PS.PenOptions(strokes=other))
    assert r2["plan"]["strokes"] == other and r2["plan"]["strokes_source"] == "option"
    assert r2["coverage"]["drawn_fraction"] < good_cov - 0.05 and paper.ink_length() < good_ink - 0.05, (r2["coverage"], good_cov)
    assert r2["coverage"]["gaps_n"] > 10 and (r2["ok"] is False) == (r2["coverage"]["drawn_fraction"] < 0.85)
    print(f"    {skips}: missing {d['missing']} -> prefer {want}; cat drawn {want}: {good_cov:.3f}, forced {other}: "
          f"{r2['coverage']['drawn_fraction']:.3f} ({r2['coverage']['gaps_n']} gaps)")


def test_10_direction_test_recovers_tip_trailing_skips():
    _direction_case("tip-trailing", "away")


def test_11_direction_test_recovers_tip_leading_skips():
    _direction_case("tip-leading", "toward")


def test_12_without_a_direction_test_the_result_says_so():
    c = calibrated()
    tool = TL.Tool.from_dict(c["tool"])
    tool.calibrated.pop("direction")
    pol = PS.stroke_policy(tool, "auto")
    assert pol["policy"] == "as-is" and pol["source"] == "not measured" and "NOT chosen from a measurement" in pol["note"]
    assert PS.stroke_policy(tool, "away") == {"policy": "away", "source": "option", "note": "stroke direction 'away' was asked for"}
    # strokes.direction_lean: the policy as a lean for pull_pieces (SPEC 9.1: pass the negated lean for "away")
    assert ST.direction_lean("as-is") is None and ST.direction_lean("any") is None and ST.direction_lean(None) is None
    p = (0.25, 0.05)
    assert np.allclose(ST.direction_lean("toward")(p), ST.radial_lean()(p)) and np.allclose(ST.direction_lean("away")(p), -ST.radial_lean()(p))
    line = ST.resample(np.array([[0.20, 0.0], [0.26, 0.0]]), 0.0012)        # away from the robot
    assert ST.pull_pieces(line, ST.direction_lean("away"))[0][-1][0] > 0.25      # kept
    assert ST.pull_pieces(line, ST.direction_lean("toward"))[0][-1][0] < 0.21    # turned round
    try:
        ST.direction_lean("pull")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown policy was accepted")


def test_13_simpaper_has_no_skipping_rule_built_in():
    """SPEC 9.1: which direction of travel loses ink is a hidden truth that a test sets; default none."""
    lean = np.array([0.26, 0.0])                                    # the pen's tip points to +x

    def laid(skips_kw, a, b):
        p = SP.SimPaper(cam=CAM, seed=3, **skips_kw)
        for k in range(801):
            q = np.asarray(a) + (np.asarray(b) - np.asarray(a)) * (k / 800.0)
            p.feed_tip((q[0], q[1], p.surface_z(q[0], q[1]) - 0.002), lean=lean)
        p.pen_up()
        return p.ink_length() / 0.08, len(p.ink())

    lead, trail = ((0.20, 0.0), (0.28, 0.0)), ((0.28, 0.0), (0.20, 0.0))
    assert SP.SimPaper(cam=CAM).skips == "none" and SP.SimPaper(cam=CAM).truth()["skips"] == "none"
    for kw in ({}, {"skips": "none"}, {"push_skips": False}):
        assert laid(kw, *lead)[1] == 1 and laid(kw, *trail)[1] == 1 and laid(kw, *lead)[0] > 0.995, kw
    for kw in ({"skips": "tip-leading"}, {"push_skips": True}):
        assert 0.3 < laid(kw, *lead)[0] < 0.8 and laid(kw, *lead)[1] >= 8 and laid(kw, *trail) == (laid({}, *trail)[0], 1), kw
    assert 0.3 < laid({"skips": "tip-trailing"}, *trail)[0] < 0.8 and laid({"skips": "tip-trailing"}, *lead)[1] == 1
    assert laid({"skips": "tip-trailing"}, (0.24, -0.04), (0.24, 0.04))[1] == 1          # across: whole either way
    try:
        SP.SimPaper(cam=CAM, skips="pull")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown skip truth was accepted")
    assert "skips" not in SP.night_truths()                         # the night's world does not claim one


# ============================================================================= 5. every stop lifts and reports
def _break_stroke(arm, n, exc):
    """The n-th pen-down motion of this arm raises `exc` instead of moving (the pen is on the paper then)."""
    orig, seen = arm.run_path, {"n": 0}

    def run_path(*a, **k):
        if k.get("allow_contact") and k.get("settle") is False:
            seen["n"] += 1
            if seen["n"] == n:
                raise exc
        return orig(*a, **k)

    arm.run_path = run_path
    return seen


def test_14_an_abort_while_drawing_lifts_the_pen_and_goes_on_or_stops():
    arm, paper, sess = calibrated_world(NIGHT)
    _break_stroke(arm, 6, MotionAborted("target not reached: test", info={"reason": "target_not_reached"}))
    r = PS.draw(sess, "cat", PS.PenOptions(size_mm=70.0))
    check_result(r, "draw")
    # one piece failed: the pen was lifted, the rest was drawn, and the result is not ok
    assert r["ok"] is False and r["stage"] == "verify" and len(r["drawn"]["failed"]) == 1 and "1 piece(s) failed" in r["reason"], r
    assert r["drawn"]["failed"][0]["error"] == "MotionAborted" and r["drawn"]["pieces"] == r["plan"]["pieces"] - 1
    assert r["coverage"]["drawn_fraction"] > 0.6 and r["pen"]["lifted"] and r["pen"]["where"] == "aside"
    # no failed piece is tolerated: the drawing stops there, pen up, aside
    arm, paper, sess = calibrated_world(NIGHT)
    _break_stroke(arm, 2, MotionAborted("target not reached: test", info={"reason": "target_not_reached"}))
    r = PS.draw(sess, "cat", PS.PenOptions(size_mm=70.0, max_failed=0))
    assert r["ok"] is False and r["stage"] == "draw" and len(r["failed"]) == 1 and "coverage" not in r, r
    assert r["pen"]["lifted"] and r["pen"]["where"] == "aside" and r["pen"]["height_mm"] > 30 and arm.is_torqued()
    assert "ペン先は机の" in r["say"]
    # during calibration: the same
    arm, paper, sess = world(NIGHT)
    _break_stroke(arm, 4, SafetyError("refused: test"))
    r = PS.calibrate(sess, PS.STAGES, PS.PenOptions())
    check_result(r, "calibrate")
    assert r["ok"] is False and r["stage"] == "tip" and "SafetyError" in r["reason"], r
    assert r["pen"]["lifted"] and r["pen"]["where"] == "aside" and TL.active().tip_offset == pen0().tip_offset


def test_15_a_lift_that_fails_is_reported_not_hidden():
    """2026-11-11: the lift failed and the script died with the pen 8 mm above the paper. A lift that fails
    here ends the command with the pen where it is, torque on, and the result says exactly that."""
    arm, paper, sess = calibrated_world(NIGHT)
    _break_stroke(arm, 2, MotionAborted("target not reached: test", info={"reason": "target_not_reached"}))
    real_lift = TL.lift
    TL.lift = lambda *a, **k: (_ for _ in ()).throw(SafetyError("cannot be lifted straight up from this pose: test"))
    try:
        r = PS.draw(sess, "cat", PS.PenOptions(size_mm=70.0))
    finally:
        TL.lift = real_lift
    check_result(r, "draw")
    assert r["ok"] is False and r["stage"] == "draw" and "cannot be lifted" in r["reason"], r
    pen = r["pen"]
    assert pen["where"] == "stopped" and pen["lifted"] is False and "lift_error" in pen and pen["torque"] == "on", pen
    assert pen["height_mm"] < 20 and "持ち上げに失敗" in r["say"] and arm.is_torqued()
    # the reach test of the layout is what keeps this from happening: a place from which the pen cannot be
    # lifted straight up at the working pitch is not "in reach" (near the robot the wrist hits its limit)
    run = PS._Run(sess, PS.PenOptions(), INK)
    assert run.reach_ok(0.23, 0.0) and not run.reach_ok(0.144, 0.03) and not run.reach_ok(0.45, 0.0)
    q = arm.solve(np.array([0.144, 0.03, TZ + 0.002]), 75.0, 0.0)                  # the paper itself is reachable there
    try:
        arm.solve(np.array([0.144, 0.03, TZ + 0.038]), 75.0, 0.0, q_seed=q)
    except SafetyError:
        pass
    else:
        raise AssertionError("the place the test relies on can be lifted from after all")


# ============================================================================= 6. the commands
def test_16_fd_commands_rehearse_the_morning():
    """The commands as typed, one fd call each, sharing one simulated sheet (sim_paper.json next to the
    simulated tool file)."""
    TL.clear()
    truth = os.path.join(_TMP, "night.json")
    with open(truth, "w") as f:
        json.dump(NIGHT, f)
    sheet = os.path.join(_TMP, "sim_paper.json")
    if os.path.exists(sheet):
        os.remove(sheet)
    log = []

    def run(*argv):
        r, rc = fd_run(["--sim", *argv])
        log.append((" ".join(argv), rc, r.get("stage"), (r.get("say") or "")[:70]))
        return r, rc

    r, rc = run("draw", "cat")
    assert rc == 1 and r["stage"] == "tool" and "fd tool attach" in r["next"], r
    r, rc = run("tool", "calibrate")
    assert rc == 1 and r["stage"] == "tool" and "fd tool attach" in r["next"], r
    assert not os.path.exists(fd.RUNS), "a refused command made a run directory"
    r, rc = run("tool", "attach", "--name", "pen", "--along-fingers", "0.055")
    assert rc == 0 and r["file"] == TOOL_PATH, r
    r, rc = run("draw", "cat")
    assert rc == 1 and r["stage"] == "tool" and "fd tool calibrate" in r["next"] and r["sim"] is True, r
    r, rc = run("tool", "calibrate", "--dry", "--truth", truth)
    assert rc == 0 and r["stage"] == "planned" and os.path.isfile(r["preview"]) and r["calibrated"] == {"tip": False, "contact": False, "map": False}, r
    r, rc = run("tool", "calibrate", "--truth", truth)
    assert rc == 0 and r["ok"] and r["stage"] == "done" and r["calibrated"] == {"tip": True, "contact": True, "map": True}, (r["stage"], r["reason"])
    assert r["file"] == TOOL_PATH and os.path.isfile(sheet) and r["photos"][0].startswith(fd.RUNS)
    r, rc = run("tool", "status")
    assert rc == 0 and r["tool"]["calibrated"] == {"tip": True, "contact": True, "map": True} and "fd draw" in r["next"], r
    r, rc = run("draw", "cat", "--dry")                               # no --truth: the world of the last command
    assert rc == 0 and r["stage"] == "planned" and os.path.isfile(r["preview"]) and r["layout"]["older_ink_pieces"] >= 12, r
    r, rc = run("draw", "cat", "--end", "rest")
    assert rc == 0 and r["ok"] and r["coverage"]["drawn_fraction"] >= 0.9 and r["coverage"]["offset_mm"]["mean"] <= 2.0, (r["stage"], r["reason"])
    assert r["pen"]["where"] == "rest" and r["wall_s"] > 0
    cat = r
    r, rc = run("tool", "calibrate", "--stage", "map", "--dry")       # one stage can be planned / repeated alone
    assert rc == 0 and list(r["plan"]) == ["map"], r
    r, rc = run("draw", "cat", "--press", "5")
    assert rc == 1 and r["stage"] == "options" and "press_m" in r["reason"] and "動かしていません" in r["say"], r
    r, rc = run("draw", "dog")
    assert rc == 1 and r["stage"] == "load" and "cat" in r["next"], r
    r, rc = run("draw", "cat", "--dry", "--new-sheet", "--size", "60", "--strokes", "toward", "--rot", "0")
    assert rc == 0 and r["layout"]["older_ink_pieces"] == 0 and abs(r["layout"]["size_mm"] - 60) < 0.5 and r["layout"]["rot_deg"] == 0.0, r
    assert r["plan"]["strokes"] == "toward" and r["plan"]["pieces"] > 25
    r, rc = run("tool", "rest")
    assert rc == 0 and r["ok"], r
    r, rc = run("tool", "detach")
    assert rc == 0 and not os.path.exists(TOOL_PATH)
    # --truth / --new-sheet are for the simulation: the session of a real command refuses them before the
    # camera or the arm is touched (the real command itself is never run here)
    ns = fd.build_parser().parse_args(["draw", "cat", "--truth", truth])
    assert ns.sim is False
    try:
        fd._pen_session(None, ns)
    except SafetyError as e:
        assert "--sim" in str(e)
    else:
        raise AssertionError("--truth was accepted without --sim")
    try:
        PS.load_truth(os.path.join(_TMP, "missing.json"))
    except SafetyError:
        pass
    else:
        raise AssertionError("a missing truth file was accepted")
    with open(os.path.join(_TMP, "bad.json"), "w") as f:
        json.dump({"nigth": True}, f)
    r, rc = run("draw", "cat", "--truth", os.path.join(_TMP, "bad.json"))
    assert rc == 1 and r["stage"] == "options" and "unknown keys" in r["reason"], r
    for line in log:
        print("    fd --sim", line[0].replace(_TMP, "<tmp>"), "->", "exit", line[1], line[2], "|", line[3])
    print(f"    fd --sim draw cat: drawn {cat['coverage']['drawn_fraction']}, offset {cat['coverage']['offset_mm']}, "
          f"{cat['timings']['total']:.0f} s of robot time, {cat['wall_s']} s wall")


def test_17_sim_park_and_repl_leave_the_real_camera_alone():
    """SPEC 9.2 (d): `fd --sim park` and leaving `fd --sim repl` ran v4l2-ctl on the real camera."""
    TL.clear()
    calls = []
    real = camera.restore_auto
    camera.restore_auto = lambda: calls.append("auto")
    import builtins
    real_input = builtins.input
    try:
        r, rc = fd_run(["--sim", "park"])
        assert rc == 0 and r["parked"] is True and calls == [], (r, calls)
        r, rc = fd_run(["--sim", "ready"])
        assert rc == 0
        builtins.input = lambda *a: (_ for _ in ()).throw(EOFError())
        buf = io.StringIO()
        ns = fd.build_parser().parse_args(["--sim", "repl", "--frame", os.path.join(HERE, "testdata", "0004_ready_raw.jpg")])
        with contextlib.redirect_stdout(buf):
            with fd.make_arm(True) as arm:
                out = fd.cmd_repl(arm, ns)
        assert out["requests"] == 0 and calls == [], (out, calls)
        # without --sim the camera IS handed back (the call is made; here it is the stand-in)
        ns = fd.build_parser().parse_args(["park"])
        assert ns.sim is False and fd._camera_to_auto(ns) is True and calls == ["auto"]
        with fd.make_arm(True) as arm:                               # a simulated arm, the real command's arguments
            fd.cmd_park(arm, ns)
        assert calls == ["auto", "auto"]
    finally:
        camera.restore_auto = real
        builtins.input = real_input
    assert _TRIPPED == [], _TRIPPED


def test_18_the_simulated_tool_file_never_lies_in_the_tree():
    """SPEC 9.2 (e): `fd --sim tool attach` without $FD_TOOL wrote tool_active.sim.json into the tree."""
    in_tree = os.path.join(HERE, "tool_active.sim.json")
    assert TL.SIM_TOOL_FILE == in_tree and not os.path.exists(in_tree)
    saved, real_dir = os.environ.pop("FD_TOOL"), fd.sim_state_dir
    scratch = os.path.join(_TMP, "simstate")
    fd.sim_state_dir = lambda: scratch
    try:
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "sim_pen", "--along-fingers", "0.06", "--no-robot"])
        assert rc == 0 and r["file"] == os.path.join(scratch, "tool_active.sim.json") and os.path.isfile(r["file"]), r
        assert not os.path.exists(in_tree) and TL.SIM_TOOL_FILE == in_tree           # (put back after the command)
        assert TL.source(sim=True) != r["file"]                                      # (only fd --sim reads it)
        r, rc = fd_run(["--sim", "tool", "status"])
        assert rc == 0 and r["attached"] and r["tool"]["name"] == "sim_pen" and r["simulated_tool"] is True, r
        r, rc = fd_run(["--sim", "state"])                                           # the simulated arm carries it
        assert rc == 0
        r, rc = fd_run(["--sim", "grip", "0.5"])
        assert rc == 1 and r["stage"] == "tool" and r["tool"] == "sim_pen", r
        r, rc = fd_run(["--sim", "tool", "detach"])
        assert rc == 0 and r["removed"] and not os.path.exists(os.path.join(scratch, "tool_active.sim.json"))
    finally:
        os.environ["FD_TOOL"] = saved
        fd.sim_state_dir = real_dir
    # the real directory is outside the tree and one per tree
    d = fd.sim_state_dir()
    assert not os.path.realpath(d).startswith(os.path.realpath(HERE) + os.sep) and os.path.isabs(d)
    assert TL.SIM_TOOL_FILE == in_tree and not os.path.exists(in_tree)
    # with $FD_TOOL the guard keeps its hands off
    assert fd._sim_tool_out_of_tree() is None


def test_19_layout_on_a_real_photo_of_the_night():
    """The photos of 2026-11-11 (paper exposure, arm at the side pose): what the photo offers is found as on
    the simulated sheet. The real sheet is larger than A4 and cut by the picture, two strips of tape lie on
    it and it carries the ink of that evening. No motion: only the layout is asked."""
    photo = os.path.join(HERE, "testdata", "pen_20261111", "photos", "224912_grid_before.jpg")
    img = cv2.imread(photo)
    assert img is not None and img.shape == (1080, 1920, 3)
    TL.clear()
    TL.save(TL.Tool(name="pen2", tip_offset=[0.0015, -0.0081, 0.0621], mount="along_fingers", pitch_deg=75.0))
    arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=TL.active())
    assert TL.sim_start(arm)
    arm.open()
    sess = PS.PenSession(arm, CAM, lambda e="paper": img, sim=True)
    run = PS._Run(sess, PS.PenOptions(), INK)
    run.q_photo = np.radians(PS.ASIDE_Q_DEG)                 # where the arm stood in that photo
    assert INK.can_show_ink(img, CAM, TZ)["ok"]
    ws = run.workspace(img)
    assert ws.sheet is not None and ws.sheet["rect"] is False and ws.sheet["cut_by_picture"] is True, ws.sheet
    assert 380 <= ws.sheet["size_mm"][0] <= 480 and abs(ws.sheet["rot_deg"] - 6.0) < 2.0 and 150 < ws.level < 190, (ws.sheet, ws.level)
    assert ws.paper(0.22, 0.0, 0.01) and not ws.paper(0.22, 0.30) and not ws.paper(0.03, 0.0)       # sheet / desk / robot base
    for side in (-1.0, 1.0):                                                                    # the two strips of tape
        row = [ws.paper(0.22, side * y) for y in np.arange(0.12, 0.20, 0.004)]
        assert row[0] and not all(row) and row[-1], (side, row)                                   # paper, tape, paper again
    assert 15 <= ws.old_ink <= 60, ws.old_ink                                                     # scribbles, ladders, strokes
    box = run.draw_box(ws, avoid_ink=False)
    # in the middle of the paper the pen can work on: near where the cat was drawn ((0.2575, 0.003) and (0.223, 0.003))
    assert box["ok"] and abs(box["half_m"] - 0.0525) < 1e-6 and box["sheet_found"], box
    assert 0.19 < box["centre"][0] < 0.27 and abs(box["centre"][1]) < 0.03, box["centre"]
    clean = run.draw_box(ws, avoid_ink=True)                 # what `fd draw` would get on this used sheet: less
    assert clean["ok"] and 0.015 <= clean["half_m"] < 0.0525 and "instead of" in clean["reason"], clean
    corners = np.array(clean["corners"])
    assert all(ws.ink_free((x, y, x, y), 0.003) for x, y in corners) and not arm.is_torqued()
    print(f"    real photo: sheet {ws.sheet['size_mm']} mm (cut by the picture), turned {ws.sheet['rot_deg']} deg, {ws.old_ink} "
          f"pieces of ink; box {np.round(box['centre'], 3).tolist()} 105 mm, clear of ink: {np.round(clean['centre'], 3).tolist()} "
          f"{clean['half_m'] * 2000:.0f} mm")


def test_20_an_interrupt_lifts_the_pen_and_still_answers():
    """Reviewer finding (2026-11-12): Ctrl-C, or a tool call that was cancelled or timed out, left the pen pressed on
    the paper with the torque on and printed nothing."""
    arm, paper, sess = calibrated_world(NIGHT)
    _break_stroke(arm, 3, KeyboardInterrupt())
    r = PS.draw(sess, "cat", PS.PenOptions(size_mm=70.0))
    check_result(r, "draw")
    assert r["ok"] is False and r.get("interrupted") is True and r["stage"] == "draw" and "interrupted" in r["reason"], r
    assert r["pen"]["lifted"] and r["pen"]["height_mm"] > 5 and "fd tool rest" in r["next"] and "中断" in r["say"], r
    arm, paper, sess = world(NIGHT)
    _break_stroke(arm, 2, KeyboardInterrupt())
    r = PS.calibrate(sess, PS.STAGES, PS.PenOptions())
    check_result(r, "calibrate")
    assert r["ok"] is False and r.get("interrupted") is True and r["pen"]["lifted"] and "中断" in r["say"], r


def test_21_a_calibration_made_with_another_tip_is_refused():
    """Reviewer finding (2026-11-12): after `fd tool set ... --keep-calibration` had changed the tip, `fd draw` used a
    contact plane measured for another tip and pressed 15 mm into the simulated paper."""
    arm, paper, sess = calibrated_world(NIGHT)
    assert not PS._stale(arm.tool, "contact") and not PS._stale(arm.tool, "map")
    io0 = dict(arm.bus.io)
    t = arm.tool.tip_offset
    arm.tool.tip_offset = [t[0], t[1], t[2] - 0.005]
    r = PS.draw(sess, "cat")
    assert r["ok"] is False and r["stage"] == "tool" and "fd tool calibrate" in r["next"] and "another tip" in r["reason"], r
    assert arm.bus.io["writes"] == io0["writes"] and paper.samples == 0
    arm, paper, sess = calibrated_world(NIGHT)                       # the pitch counts too
    arm.tool.pitch_deg = arm.tool.pitch_deg - 2.0
    assert PS.draw(sess, "cat")["stage"] == "tool"
    arm, paper, sess = calibrated_world(NIGHT)                       # a measurement that does not say what it was made with
    arm.tool.calibrated["map"].pop("tip_offset")
    r = PS.draw(sess, "cat")
    assert r["stage"] == "tool" and paper.samples == 0, r


def test_zz_no_hardware_and_nothing_written_into_the_tree():
    """Runs last (zz): the audit hook saw no serial port, no video device, no v4l2-ctl; the tree has no new
    file, run directory, backup or tool file from these tests."""
    assert _TRIPPED == [], _TRIPPED
    try:
        open("/dev/ttyACM0")
    except RuntimeError:
        _TRIPPED.clear()                                   # the guard works (and nothing was opened)
    except OSError:
        raise AssertionError("the audit hook is not armed")
    else:
        raise AssertionError("the audit hook is not armed")
    now = {d: sorted(os.listdir(os.path.join(HERE, d))) if os.path.isdir(os.path.join(HERE, d)) else None for d in _TREE_BEFORE}
    assert now == _TREE_BEFORE, {d: sorted(set(now[d] or []) ^ set(_TREE_BEFORE[d] or [])) for d in now}
    assert os.environ["FD_TOOL"] == TOOL_PATH and TL.TOOL_FILE == os.path.join(HERE, "tool_active.json")


if __name__ == "__main__":
    import shutil
    import traceback
    only = sys.argv[1:]
    fails = 0
    n = 0
    t_all = time.time()
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn) and (not only or any(o in name for o in only)):
                n += 1
                t0 = time.time()
                try:
                    fn()
                    print(f"PASS {name} ({time.time() - t0:.1f} s)", flush=True)
                except Exception:
                    fails += 1
                    print("FAIL", name, flush=True)
                    traceback.print_exc()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print(f"tests: {n}  passed: {n - fails}  failures: {fails}  ({time.time() - t_all:.0f} s)")
    sys.exit(1 if fails else 0)
