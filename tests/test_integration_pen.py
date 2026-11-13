"""Integration tests of the pen packages as ONE tree (stage 1 of the pen update of TAZUNA, then called JetHarness, night of 2026-11-12):
the seams between INK (camera exposures, ink.py), TOOL (tooling.py and the guards), STROKES (strokes.py,
simpaper.py) and MOTION (tracking.py, run_path's fine mode, `fd track-test`). Each package was built and
tested in its own copy of the tree, where the other three did not exist; these tests run what only exists
after the merge.

Run: python tests/test_integration_pen.py [name-part ...]      (no hardware: simulated servo bus, simulated paper)
     JH_QUICK=1 skips the one test that runs test_sim.py as a subprocess (15 s).

What they show:
 - the tool tip is ONE thing: tooling.config_with_tool, Arm(tool=...).tip_tool(), SimPaper's model of the
   tip, tracking's FK and PhysBus's contact point agree;
 - tooling.lean_xy is the `lean` of strokes.pull_pieces / plan_pieces: the pull invariant holds with it for
   the cat, and the direction of pulling follows the tool's mount (not "+x");
 - SimPaper watches an arm that carries a tooling.Tool: the free moves of tooling (ready, aside, rest) are
   seen and lay no ink, a pen-down run_path lays ink where the TOOL tip was sent;
 - ink.py reads SimPaper's photos: right with the paper exposure, "cannot judge" (never "no ink") with the
   desk exposure, also for a thin pen whose line the desk exposure wipes out completely;
 - a ladder of dashes drawn by the simulated arm whose real pen is NOT where the model has it (the night's
   hidden truths) is read as the paper really is;
 - tracking.record runs over SimPaper (both monitors, a caller's monitor, the fine mode), and a Trace takes
   a tooling.Tool wherever it takes a tool frame;
 - `fd track-test` flies the TOOL tip when a tool is attached, and its --phys arm carries the SIMULATED tool;
 - `fd snap --frame` judges SimPaper's photos with a tool attached;
 - marks read by ink -> strokes.fit_map -> Tool.corr_map in the tool file -> strokes.plan_pieces: the ink
   lands on the drawing although the world is sheared (what `fd tool calibrate` of stage 2 has to do);
 - the suites of the tree do not depend on what is attached to the real arm.

Every test uses $FD_TOOL (a scratch file) and a scratch run directory: nothing is written into the tree.
"""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import contextlib
import functools
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "pen_20261111", "camera_model.json")
_TMP = tempfile.mkdtemp(prefix="jh_integration_")
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
import phys_bus as PB  # noqa: E402
import simpaper as SP  # noqa: E402
import strokes as ST  # noqa: E402
import tooling as TL  # noqa: E402
import tracking as TR  # noqa: E402
from robot import ARM, Arm, MotionAborted, SafetyError, load_config  # noqa: E402
from sim_bus import SimBus  # noqa: E402

fd.log_event = lambda *a, **k: None        # no run log from tests
fd.RUNS = os.path.join(_TMP, "runs")       # photos and traces of `fd` go to the scratch directory, not into the tree
_TREE_BEFORE = {d: sorted(os.listdir(os.path.join(tazuna_paths.ROOT, d))) if os.path.isdir(os.path.join(tazuna_paths.ROOT, d)) else None
                for d in ("runs", "backups")}
np.set_printoptions(precision=4, suppress=True)
CFG = load_config()
CAM = camera.CameraModel.load()
TZ = float(CFG["table_z"])
PEN_TIP = [0.0015, -0.0081, 0.0621]                       # the marker of 2026-11-11, measured on the paper
ASIDE_Q = np.radians([43.15, -20.31, 10.46, 63.28, -0.05])  # the side pose the night's photos were taken from
NOISE = 0.6                                              # grey levels (sd) of the simulated photos, see the notes
SQ = np.array([[0.225, -0.015], [0.225, 0.015], [0.255, 0.015], [0.255, -0.015], [0.225, -0.015]])
_CACHE = {}


class Skip(Exception):
    pass


# ----------------------------------------------------------------------------- helpers
def pen(**over):
    kw = dict(name="pen2", tip_offset=list(PEN_TIP), mount="along_fingers", pitch_deg=75.0, gripper_raw=2061)
    kw.update(over)
    return TL.Tool(**kw)


@contextlib.contextmanager
def tool_file(tool):
    """The tool file of the test ($FD_TOOL) holds `tool` (None: no file) for the duration."""
    TL.clear()
    if tool is not None:
        TL.save(tool)
    try:
        yield
    finally:
        TL.clear()


def fd_run(argv):
    """fd.main(["--sim", ...]) as the command line runs it (fresh simulated arm). -> (result, exit code)."""
    assert argv[0] == "--sim", "tests run the simulated arm only"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fd.main(list(argv))
    return json.loads(buf.getvalue().strip().splitlines()[-1]), rc


def pen_arm(tool=None, ready=True):
    """A simulated arm as fd.make_arm(True) builds it with a tool attached: limp in the tool's rest pose;
    ready=True: then at the tool's stand-by pose (torque on)."""
    arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=tool if tool is not None else pen())
    assert TL.sim_start(arm)
    arm.open()
    if ready:
        TL.go_ready(arm)
    return arm


def tip_path(arm, pts, v_mm_s=6.0, record=False, **kw):
    """Straight segments of the TOOL tip at about v_mm_s, as a pen-down path must run: contact allowed, no
    settle (a settle while pressing ends in the "pinned" abort, SPEC section 2)."""
    q0 = arm.read_q()
    p0 = K.fk(q0, arm.tip_tool())[:3, 3]
    wps = arm.plan_tip_path([(np.asarray(p, float), arm.tool.pitch_deg, 0.0) for p in pts], q_start=q0, step_m=0.0015)
    ln = float(sum(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)) for a, b in zip([p0, *pts[:-1]], pts)))
    speed = TR.joint_speed_for(wps, ln, v_mm_s / 1000.0)
    if record:
        return TR.record(arm, wps, speed, allow_contact=True, settle=False, **kw)
    return arm.run_path(wps, speed, allow_contact=True, settle=False, **kw)


def hover(arm, xy, h=0.02):
    q = arm.read_q()
    qt, _ = TL.work_q(arm, arm.tool, xy, h)
    return arm.run_path([q, qt], 25.0, settle=0.3)


def aside(arm):
    """To the side pose, by tooling's free move (the tool tip never lower than 2 cm)."""
    return TL.free_move(arm, ASIDE_Q, 30.0)


def draw(arm, pieces, z_of, v=6.0, record=False):
    """strokes.plan_pieces pieces with the pen: hover, pen down at cmd[0], the parts one after the other
    without lifting, pen up. z_of(xy) -> plan height of the tool tip. Returns the Traces of the parts."""
    out = []
    for pc in pieces:
        c = pc["cmd"]
        hover(arm, c[0])
        tip_path(arm, [(c[0][0], c[0][1], z_of(c[0]))], 10.0)
        for part in pc["parts"]:
            out.append(tip_path(arm, [(p[0], p[1], z_of(p)) for p in part[1:]], v, record=record))
        tip_path(arm, [(c[-1][0], c[-1][1], TZ + 0.02)], 20.0)
    return out


def dash(arm, a, b, z):
    """One pen-down dash from a to b at plan height z. Returns the metres of ink it really left."""
    paper = arm._simpaper
    hover(arm, a)
    tip_path(arm, [(a[0], a[1], z)], 10.0)
    n0 = paper.ink_length()
    tip_path(arm, [(b[0], b[1], z)], 6.0)
    tip_path(arm, [(b[0], b[1], TZ + 0.02)], 20.0)
    return paper.ink_length() - n0


def feed_line(paper, a, b, dz):
    """Move the REAL tip straight from a to b at dz above (+) or below (-) the true surface; then pen up."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(round(float(np.linalg.norm(b - a)) / 0.0001)) + 1)
    for k in range(n):
        p = a + (b - a) * (k / (n - 1))
        paper.feed_tip((p[0], p[1], paper.surface_z(p[0], p[1]) + dz))
    paper.pen_up()


def region_of(poly, margin=0.01):
    p = np.asarray(poly, float).reshape(-1, 2)
    return (p[:, 0].min() - margin, p[:, 1].min() - margin, p[:, 0].max() + margin, p[:, 1].max() + margin)


def square_run():
    """One drawing shared by several tests: a simulated arm with the pen attached (truths = the model) goes
    to stand-by, aside (photo), draws a 3 cm square as pull pieces, goes aside (photos), rests."""
    if "sq" in _CACHE:
        return _CACHE["sq"]
    arm = pen_arm(ready=False)
    paper = SP.SimPaper(arm, CAM, seed=1)                 # attaches itself to arm.run_path
    s = {"arm": arm, "paper": paper, "gripper0": float(arm.bus.pos["gripper"])}
    s["ready"] = TL.go_ready(arm)
    s["after_ready"] = (paper.samples, paper.ink_length())
    aside(arm)
    s["before"], s["before_desk"] = paper.render("paper", noise=NOISE), paper.render("desk", noise=NOISE)
    lean = functools.partial(TL.lean_xy, arm, arm.tool)
    s["plan"] = ST.plan_pieces([SQ], lean)
    s["traces"] = draw(arm, s["plan"]["pieces"], lambda p: TZ - 0.001, record=True)
    s["ink_m"], s["samples_drawn"] = paper.ink_length(), paper.samples
    s["above"] = paper.render("paper", noise=NOISE)        # the arm is still over the sheet
    aside(arm)
    s["after"], s["after_desk"] = paper.render("paper", noise=NOISE), paper.render("desk", noise=NOISE)
    s["rest"] = TL.rest(arm)
    s["end"] = (paper.samples, paper.ink_length(), float(arm.bus.pos["gripper"]))
    _CACHE["sq"] = s
    return s


# ============================================================================= 1. one tool tip
def test_the_tool_tip_is_one_thing_in_every_package():
    """tooling.config_with_tool (for code that takes a config), Arm(tool=...) (for code that takes an arm),
    SimPaper's model of the tip, tracking's FK and PhysBus's contact point are the same point."""
    tool = pen()
    cfg = load_config()
    finger = list(cfg["tip_offset"])
    cfg2 = TL.config_with_tool(cfg, tool)
    assert cfg["tip_offset"] == finger and cfg2["tip_offset"] == PEN_TIP and cfg2 is not cfg     # a copy
    assert set(cfg2) - set(cfg) == {"tool_name"} and cfg2["tool_name"] == "pen2"
    assert TL.config_with_tool(cfg, None)["tip_offset"] == finger
    arm_t = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=tool).open()
    arm_c = Arm(bus=SimBus(realtime=False), cfg=cfg2, tool=None).open()                     # the FD_CONFIG trick
    arm_0 = Arm(bus=SimBus(realtime=False), cfg=load_config()).open()                         # "active": no file
    assert arm_0.tool is None and arm_c.tool is None and arm_t.tool is tool
    assert np.allclose(arm_t.tip_tool(), arm_c.tip_tool()) and np.allclose(arm_t.tip_tool(), K.tool_offset(*PEN_TIP))
    assert not np.allclose(arm_t.tip_tool(), arm_0.tip_tool())
    assert list(arm_t.cfg["tip_offset"]) == finger          # the arm's own config keeps the fingertip
    p = np.array([0.24, 0.03, TZ + 0.03])
    q_t, q_c = arm_t.solve(p, 75.0, 0.0), arm_c.solve(p, 75.0, 0.0)
    assert np.allclose(q_t, q_c, atol=1e-9)
    assert np.allclose(K.fk(q_t, arm_t.tip_tool())[:3, 3], p, atol=1e-5)
    # SimPaper: what the arm believes is the tool tip; a tip error is added to THAT
    paper = SP.SimPaper(arm_t, CAM, attach=False)
    assert np.allclose(paper.model_tip_offset, PEN_TIP) and np.allclose(paper.true_tip_offset, PEN_TIP)
    assert np.allclose(paper.true_tip(q_t), p, atol=1e-5)
    err = (0.0, -0.015, 0.008)
    paper = SP.SimPaper(arm_t, CAM, tip_error=err, attach=False)
    assert np.allclose(paper.true_tip_offset, np.array(PEN_TIP) + err) and np.allclose(paper.model_tip_offset, PEN_TIP)
    # tracking: a tooling.Tool, the arm's 4x4 frame and kinematics.fk give the same tip
    q = np.stack([q_t, arm_t.solve(p + (0.01, -0.02, 0.0), 75.0, 0.0)])
    a, b = TR.fk_points(q, tool), TR.fk_points(q, arm_t.tip_tool())
    assert np.allclose(a, b, atol=1e-12) and np.allclose(a[0], p, atol=1e-5)
    assert np.allclose(TR.tool_frame(tool), arm_t.tip_tool()) and np.allclose(TR.tool_frame(None), K._T_TOOL)
    # the torque-level servo model takes its contact point from the config: with the tool config it is the pen tip
    assert np.allclose(PB.PhysBus(cfg2).tool, arm_t.tip_tool()) and np.allclose(PB.PhysBus(cfg).tool, arm_0.tip_tool())


# ============================================================================= 2. lean and pull strokes
def test_lean_xy_is_the_lean_of_pull_pieces():
    """strokes.pull_pieces wants lean(p) -> unit vector; tooling.lean_xy(arm, tool, xy) is that with the
    arm and the tool bound. With it the cat obeys the pull invariant, as with the model lean."""
    arm = pen_arm(ready=False)                              # kinematics only: the arm stays limp
    io0 = dict(arm.bus.io)
    lean = functools.partial(TL.lean_xy, arm, arm.tool)
    radial = ST.radial_lean()
    for xy in ((0.2575, 0.003), (0.21, -0.05), (0.21, 0.056), (0.305, -0.05), (0.305, 0.056)):
        v = lean(xy)
        assert v.shape == (2,) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-9
        # the pen leans in the arm's plane, tip away from the robot: within a few degrees of the model lean
        assert float(v @ radial(xy)) > math.cos(math.radians(5.0)), (xy, v, radial(xy))
    cat = ST.layout(ST.load("cat"), (0.2575, 0.003), 0.0525, 6.0)
    total = sum(ST.length(s) for s in cat)
    pushed = sum(ST.push_travel(ST.resample(s, 0.0012), lean) for s in cat)
    t0 = time.perf_counter()
    plan = ST.plan_pieces(cat, lean)
    wall = time.perf_counter() - t0
    st = plan["stats"]
    worst = max(ST.push_excursion(pc["ink"], lean) for pc in plan["pieces"])
    print(f"    cat with tooling.lean_xy: push travel {pushed * 1000:.0f} -> {st['push_m'] * 1000:.1f} mm, {st['count']} pieces, "
          f"worst piece {worst * 1000:.2f} mm, ink {total * 1000:.0f} -> {st['ink_m'] * 1000:.0f} mm, planned in {wall:.1f} s")
    assert pushed > 0.25 and st["push_m"] <= 0.045 and 30 <= st["count"] <= 38, st     # SPEC 7: 275 mm -> at most 45, about 34
    assert worst <= 0.006 + 1e-9 and abs(st["push_max_m"] - worst) < 1e-12
    assert abs(st["ink_m"] / total - 1.0) <= 0.03
    assert wall < 8.0, wall                                  # about 3900 calls of lean_xy
    assert sum(len(pc["parts"]) for pc in plan["pieces"]) >= st["count"]
    assert all(np.array_equal(pc["cmd"], pc["ink"]) for pc in plan["pieces"])   # no correction map: commanded = ink
    assert arm.bus.io == io0 and not arm.is_torqued()        # planning read and wrote nothing
    # out of reach is tooling's refusal (strokes itself only knows ValueError)
    try:
        ST.pull_pieces(np.array([[0.60, 0.0], [0.62, 0.0]]), lean)
    except SafetyError as e:
        assert "out of reach" in str(e)
    else:
        raise AssertionError("a stroke out of reach was planned")


def test_the_pull_direction_follows_the_tool_not_plus_x():
    """A pen along the fingers (pitch 75) points away from the robot, a tool across the fingers (pitch 30)
    towards it: the same line is drawn in opposite directions, each time pulled."""
    arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=None).open()
    cfg = arm.cfg
    along = TL.Tool(name="along6", tip_offset=TL.offset_for(cfg, "along_fingers", 0.06), mount="along_fingers", pitch_deg=75.0)
    across = TL.Tool(name="across6", tip_offset=TL.offset_for(cfg, "across_fingers", 0.06), mount="across_fingers",
                     pitch_deg=TL.DEFAULT_PITCH["across_fingers"])
    line = ST.resample(np.array([[0.20, 0.0], [0.26, 0.0]]), 0.0012)           # drawn away from the robot
    out = {}
    for t in (along, across):
        lean = functools.partial(TL.lean_xy, arm, t)
        out[t.name] = float(lean((0.23, 0.0)) @ ST.radial_lean()((0.23, 0.0)))
        pcs = ST.pull_pieces(line, lean)
        assert len(pcs) == 1 and ST.push_excursion(pcs[0], lean) < 1e-9, (t.name, len(pcs))
        out[t.name + " runs"] = float(pcs[0][-1][0] - pcs[0][0][0])
    assert out["along6"] > 0.99 and out["across6"] < -0.95, out
    assert out["along6 runs"] < -0.059 and out["across6 runs"] > 0.059, out     # turned round / left as drawn


# ============================================================================= 3. SimPaper and the tool
def test_simpaper_watches_an_arm_that_carries_a_tool():
    """Free moves of tooling are seen by the paper and lay no ink; a pen-down path lays ink where the
    TOOL tip was sent; the gripper is never moved."""
    s = square_run()
    arm, paper = s["arm"], s["paper"]
    assert arm.tool.name == "pen2" and np.allclose(paper.model_tip_offset, PEN_TIP)
    n_ready, ink_ready = s["after_ready"]
    assert n_ready > 100 and ink_ready == 0.0                                  # stand-by: seen, in the air
    assert s["ready"]["tool"] == "pen2" and s["ready"]["min_tip_height_m"] >= TL.MIN_FREE_HEIGHT
    assert s["plan"]["stats"]["count"] >= 2 and s["plan"]["stats"]["push_max_m"] <= 0.006
    assert 0.114 < s["ink_m"] < 0.128, s["ink_m"]                              # the square is 120 mm
    truth = paper.coverage([SQ], tol_m=0.001)
    assert truth["drawn_fraction"] >= 0.99 and truth["offset_mm"]["mean"] < 0.4, truth
    assert 0.0005 < paper.max_depth_m < 0.002, paper.max_depth_m               # commanded 1 mm below the paper
    # aside and rest after the drawing: more poses seen, not a millimetre of ink more
    n_end, ink_end, gripper_end = s["end"]
    assert n_end > s["samples_drawn"] + 100 and ink_end == s["ink_m"]
    assert s["rest"]["ok"] and s["rest"]["torque"] == "off" and not arm.is_torqued()
    assert 40.0 <= s["rest"]["tip_height_mm"] <= 70.0, s["rest"]
    assert gripper_end == s["gripper0"] == 2061.0                              # the gripper stayed where it was taped
    assert {v for (_, j, v) in arm.bus.log if j == "gripper"} == {2061}        # (its only goal: hold that position)
    assert getattr(arm, "_simpaper", None) is paper and "run_path" in arm.__dict__
    print(f"    square by the arm with the pen attached: ink {s['ink_m'] * 1000:.1f} mm, on the line within "
          f"{truth['offset_mm']['p95']:.2f} mm (p95), deepest {paper.max_depth_m * 1000:.2f} mm, {n_end} poses seen")


# ============================================================================= 4. ink on SimPaper's photos
def test_ink_reads_the_simulated_photos_with_the_paper_exposure():
    s = square_run()
    paper, before, after = s["paper"], s["before"], s["after"]
    region = region_of(SQ)
    can = INK.can_show_ink(after, CAM, TZ, region)
    assert can["ok"] and can["clipped"] == 0.0 and 150 <= can["paper_level"] <= 185, can
    assert INK.can_show_ink(after, CAM, TZ)["ok"]                              # the whole picture
    assert INK.clipped_share(after, CAM, TZ) < 0.01
    t0 = time.perf_counter()
    marks = INK.find_ink(after, CAM, TZ, before=before, region=region)
    t_find = time.perf_counter() - t0
    assert marks and INK.find_ink(before, CAM, TZ, region=region) == []         # new ink only; a blank sheet before
    ext = np.array([m.extent_xy for m in marks])
    box = np.array([ext[:, 0].min(), ext[:, 1].min(), ext[:, 2].max(), ext[:, 3].max()])
    assert np.abs(box - np.array([SQ[:, 0].min(), SQ[:, 1].min(), SQ[:, 0].max(), SQ[:, 1].max()])).max() < 0.0015, box
    # coverage: the square is there; a stroke that was never drawn is reported missing, and that "no" is earned
    ghost = np.array([[0.270, -0.015], [0.270, 0.015]])
    t0 = time.perf_counter()
    cov = INK.stroke_coverage(after, CAM, TZ, [SQ, ghost], before=before)
    t_cov = time.perf_counter() - t0
    truth = paper.coverage([SQ], tol_m=0.004)
    assert cov["judgeable"] and cov["unknown_fraction"] < 0.02, cov
    assert cov["per_stroke"][0] >= 0.97 and abs(cov["per_stroke"][0] - truth["drawn_fraction"]) <= 0.03, (cov["per_stroke"], truth)
    assert cov["per_stroke"][1] == 0.0 and any(g["stroke"] == 1 and g["to_m"] - g["from_m"] > 0.025 for g in cov["gaps"]), cov["gaps"]
    assert not any(g["stroke"] == 0 for g in cov["gaps"]), cov["gaps"]
    assert cov["offset_mm"]["mean"] <= 1.0, cov["offset_mm"]
    only = INK.stroke_coverage(after, CAM, TZ, [SQ], before=before)
    assert only["drawn_fraction"] >= 0.97 and only["judgeable"]
    assert max(t_find, t_cov) < 0.4, (t_find, t_cov)                           # SPEC 7: each call under 0.4 s
    # the photo taken while the pen still stands on the sheet differs (the silhouette is in the picture)
    assert np.abs(s["above"].astype(int) - after.astype(int)).max() > 80
    print(f"    paper exposure: {len(marks)} mark(s), coverage {only['drawn_fraction']:.3f} (truth {truth['drawn_fraction']:.3f}), "
          f"offset {only['offset_mm']['mean']} mm; find_ink {t_find * 1000:.0f} ms, stroke_coverage {t_cov * 1000:.0f} ms")


def test_the_desk_exposure_is_cannot_judge_never_no_ink():
    """The lie of 2026-11-11 cannot be told on the simulated sheet either: with the locked desk exposure
    the paper is clipped, and every readout says "cannot judge" where it saw nothing."""
    s = square_run()
    before, after = s["before_desk"], s["after_desk"]
    region = region_of(SQ)
    can = INK.can_show_ink(after, CAM, TZ, region)
    assert not can["ok"] and can["clipped"] > 0.5 and "clipped" in can["reason"], can
    assert INK.clipped_share(after, CAM, TZ) > 0.3
    cov = INK.stroke_coverage(after, CAM, TZ, [SQ], before=before)
    assert cov["judgeable"] is False and cov["can"]["ok"] is False, cov
    # blank paper 5 cm beside the square (so that its fresh line is outside the 25 mm these readouts search),
    # clipped: a ladder and two marks that "left nothing"
    rungs = [((0.235 + 0.004 * i, -0.073), (0.235 + 0.004 * i, -0.067)) for i in range(4)]
    places = [(0.235, -0.07), (0.235, 0.07)]
    rd = INK.ladder_readout(after, CAM, TZ, rungs, before=before)
    assert [r["inked"] for r in rd["rungs"]] == [None] * 4 and rd["judgeable"] is False and rd["first_inked"] is None, rd["rungs"]
    assert all("cannot judge" in r["why"] for r in rd["rungs"])
    mr = INK.marks_readout(after, CAM, TZ, places, before=before)
    assert [m["inked"] for m in mr["marks"]] == [None, None] and mr["missing"] == [] and mr["judgeable"] is False
    # the same questions on the paper-exposure photo are answered "no", and that is earned there
    rd = INK.ladder_readout(s["after"], CAM, TZ, rungs, before=s["before"])
    assert [r["inked"] for r in rd["rungs"]] == [False] * 4 and rd["judgeable"] is True and rd["untouched"] is True, rd["rungs"]
    mr = INK.marks_readout(s["after"], CAM, TZ, places, before=s["before"])
    assert [m["inked"] for m in mr["marks"]] == [False, False] and mr["judgeable"] is True
    # a bold line that still shows through the clipping is ink (what is SEEN counts); only "nothing" means nothing
    bold = INK.find_ink(after, CAM, TZ, before=before, region=region)
    assert bold and max(m.length_m for m in bold) > 0.02


def test_a_thin_pen_vanishes_with_the_desk_exposure_and_the_readout_says_so():
    """The extra-fine pen (0.4 mm): four dashes that all left ink. The paper exposure shows them; the desk
    exposure wipes them out, and then no rung may be called "no ink"."""
    paper = SP.SimPaper(cam=CAM, pen_width_m=0.0004, seed=4)
    before = {e: paper.render(e, noise=NOISE) for e in ("paper", "desk")}
    rungs = [((0.235 + 0.004 * i, -0.093), (0.235 + 0.004 * i, -0.087)) for i in range(4)]
    for a, b in rungs:
        feed_line(paper, a, b, -0.0004)                                        # a light touch: a faint, narrow line
    after = {e: paper.render(e, noise=NOISE) for e in ("paper", "desk")}
    rd = INK.ladder_readout(after["paper"], CAM, TZ, rungs, before=before["paper"])
    assert [r["inked"] for r in rd["rungs"]] == [True] * 4 and rd["first_inked"] == 0 and rd["judgeable"], rd["rungs"]
    assert max(abs(v) for v in rd["offset_mm"]) < 1.0, rd["offset_mm"]          # nothing hidden here: where aimed
    region = region_of(np.array(rungs).reshape(-1, 2))
    assert len(INK.find_ink(after["paper"], CAM, TZ, before=before["paper"], region=region)) == 4
    assert INK.find_ink(after["desk"], CAM, TZ, before=before["desk"], region=region) == []   # gone
    rd = INK.ladder_readout(after["desk"], CAM, TZ, rungs, before=before["desk"])
    assert [r["inked"] for r in rd["rungs"]] == [None] * 4 and not rd["judgeable"], rd["rungs"]
    cov = INK.stroke_coverage(after["desk"], CAM, TZ, [np.array(r) for r in rungs], before=before["desk"])
    assert cov["judgeable"] is False
    cov = INK.stroke_coverage(after["paper"], CAM, TZ, [np.array(r) for r in rungs], before=before["paper"])
    assert cov["judgeable"] and cov["drawn_fraction"] >= 0.9, cov


def test_a_ladder_drawn_by_the_arm_with_hidden_truths_is_read_as_it_is():
    """The night's world: the real pen 15 mm beside and 8 mm longer than the tool file says, the position
    sheared, the height wandering. A ladder of dashes, each 1.5 mm lower: the readout must say which rungs
    left ink (they lie 1.5 cm from where they were aimed), and nothing may be read from the desk exposure."""
    cfg = load_config()
    tool = TL.Tool(name="pen", tip_offset=TL.offset_for(cfg, "along_fingers", 0.055), mount="along_fingers", pitch_deg=75.0,
                   gripper_raw=2061)
    arm = pen_arm(tool, ready=False)
    paper = SP.SimPaper(arm, CAM, seed=3, **SP.night_truths())
    TL.go_ready(arm)
    aside(arm)
    before = paper.render("paper", noise=NOISE)
    x0, y0, z_top, dz = 0.235, -0.09, TZ + 0.0085, 0.0015                       # at most 1.5 mm per dash (stop rule 2)
    rungs = [((x0 + 0.004 * i, y0 - 0.003), (x0 + 0.004 * i, y0 + 0.003)) for i in range(4)]
    laid = [dash(arm, a, b, z_top - dz * i) for i, (a, b) in enumerate(rungs)]
    aside(arm)
    after = paper.render("paper", noise=NOISE)
    assert any(v == 0.0 for v in laid) and any(v > 0.004 for v in laid), laid    # the test needs both kinds
    rd = INK.ladder_readout(after, CAM, TZ, rungs, before=before)
    got = [r["inked"] for r in rd["rungs"]]
    for i, v in enumerate(laid):
        if v == 0.0:
            assert got[i] is False, (i, laid, rd["rungs"][i]["why"])            # an earned "no"
        elif v > 0.003:
            assert got[i] is True, (i, laid, rd["rungs"][i]["why"])
    first_true = min(i for i, v in enumerate(laid) if v > 0.003)
    assert rd["first_inked"] == first_true and rd["untouched"] is True, rd
    if all(v == 0.0 or v > 0.003 for v in laid):
        assert rd["judgeable"] is True, rd["rungs"]
    # where the ink lies against where it was aimed: far (the hidden tool error), and the readout knows how far
    lines = [ln["xy"].mean(axis=0) for ln in paper.ink(new_only=True)]
    aimed = [np.mean(rungs[i], axis=0) for i, v in enumerate(laid) if v > 0.0]
    true_off = np.mean([ln - a for ln, a in zip(lines, aimed)], axis=0) * 1000.0
    assert 10.0 < float(np.linalg.norm(true_off)) < 25.0, true_off
    assert np.abs(np.array(rd["offset_mm"]) - true_off).max() < 3.0, (rd["offset_mm"], true_off)
    # the model saw the pen 4 to 8 mm ABOVE the table when it touched: the real pen is longer than the file says
    assert paper.max_depth_m > 0.001 and z_top - dz * first_true > TZ + 0.003
    desk = INK.ladder_readout(paper.render("desk", noise=NOISE), CAM, TZ, rungs)
    got_desk = [r["inked"] for r in desk["rungs"]]
    assert False not in got_desk and not desk["judgeable"], desk["rungs"]       # (the marker's bold rungs may still show)
    assert all(got_desk[i] is None for i, v in enumerate(laid) if v == 0.0), (got_desk, laid)
    print(f"    ladder with hidden truths: ink laid {[round(v * 1000, 1) for v in laid]} mm, read {got}, offset read "
          f"{rd['offset_mm']} mm (true {np.round(true_off, 1).tolist()})")


# ============================================================================= 5. tracking over the paper
def test_tracking_records_what_the_paper_sees():
    """tracking.record and SimPaper both hang on run_path's monitor: both see every control cycle, a
    caller's monitor still runs and still stops the motion, and the fine mode passes through."""
    s = square_run()
    arm0, paper0 = s["arm"], s["paper"]
    assert len(s["traces"]) >= 4 and all(tr.has_joints and tr.result["aborted"] is False for tr in s["traces"])
    # the measured tool-tip path of the traces is where the ink is (the paper takes the pose 100 ms smoothed)
    pm = np.concatenate([tr.tip(arm0.tool)[1] for tr in s["traces"]])
    ink = np.concatenate([ln["xy"] for ln in paper0.ink(new_only=True)])
    d = np.sqrt(((ink[:, None, :] - pm[None, :, :2]) ** 2).sum(axis=2)).min(axis=1)
    assert d.max() < 0.0006, d.max()
    an = s["traces"][0].analyse(arm0.tool)
    assert an == s["traces"][0].analyse(arm0.tip_tool()) and an["path_err_mm"]["max"] < 1.0, an
    bd = s["traces"][0].breakdown(arm0.tool, arm0.cfg)
    assert set(bd["joints"]) == set(ARM) and "err_rms_ticks" in bd["joints"]["elbow_flex"]
    # a fresh arm in the air: three watchers on one motion
    arm = pen_arm()
    paper = SP.SimPaper(arm, CAM)
    q0 = arm.read_q()
    q1, _ = TL.work_q(arm, arm.tool, (0.24, 0.02), 0.03)
    calls = []
    n0 = paper.samples
    tr = TR.record(arm, [q0, q1], 20.0, monitor=lambda t, qp, qm: calls.append(t) and False)
    assert len(tr) == len(calls) == paper.samples - n0 > 30
    assert np.allclose(tr.tip(arm.tool)[0], TR.fk_points(tr.q_plan, arm.tip_tool()))
    assert abs(float(tr.tip(arm.tool)[1][-1][2]) - (TZ + 0.03)) < 0.002 and paper.ink_length() == 0.0
    path = tr.save(os.path.join(_TMP, "trace_air"), arm.tool)                   # a Tool where a tool frame was asked
    back = TR.Trace.load(path)
    assert np.allclose(back.tip()[1], tr.tip(arm.tool)[1]) and back.meta["fine"] is False
    try:
        tr.tip(PEN_TIP)                                                        # a bare offset is not a tool
    except ValueError as e:
        assert "4x4" in str(e)
    else:
        raise AssertionError("a 3-vector was taken for a tool frame")
    # the caller's monitor stops the motion; the partial trace and the paper hold the same cycles
    q2, _ = TL.work_q(arm, arm.tool, (0.24, -0.03), 0.03)
    n0, seen = paper.samples, []
    try:
        TR.record(arm, [arm.read_q(), q2], 20.0, monitor=lambda t, qp, qm: seen.append(t) or len(seen) >= 25)
    except MotionAborted as e:
        assert e.info.get("reason") == "monitor" and len(e.trace) == 25 == paper.samples - n0, (len(e.trace), paper.samples - n0)
    else:
        raise AssertionError("the caller's monitor did not stop the motion")
    # the opt-in fine mode through all three layers (SimPaper's wrapper, record, run_path)
    n0 = paper.samples
    tr = TR.record(arm, [arm.read_q(), q2], 6.0, settle=False, fine=True)
    assert tr.result["fine"] is True and tr.meta["fine"] is True and paper.samples - n0 == len(tr)
    tr = TR.record(arm, [arm.read_q(), q1], 6.0, settle=False)
    assert "fine" not in tr.result and tr.meta["fine"] is False
    assert TR.fine_accepted(arm.cfg) is False                                   # never accepted on the real arm


# ============================================================================= 6. fd track-test with a tool
def test_fd_track_test_flies_the_attached_tool():
    seen = []
    real_record = TR.record

    def spy(arm, *a, **k):
        tr = real_record(arm, *a, **k)
        seen.append((arm, tr))
        return tr

    TR.record = spy
    try:
        with tool_file(pen()):
            r, rc = fd_run(["--sim", "track-test", "--shape", "line", "--size", "20"])
            assert rc == 0 and r["ok"] and r["stage"] == "done", r
            assert r["tool"] == "pen2" and r["pitch"] == 75.0 and abs(r["z"] - (TZ + 0.03)) < 1e-9, r
            assert r["runs"]["default"]["path_err_mm"]["max"] < 1.5, r["runs"]
            arm, tr = seen[-1]
            assert arm.tool.name == "pen2"
            plan_tip, meas_tip = tr.tip(arm.tool)
            assert np.abs(plan_tip[:, 2] - (TZ + 0.03)).max() < 0.001            # the TOOL tip flew the line, 3 cm up
            assert abs(plan_tip[:, 1].min() + 0.01) < 0.0015 and abs(plan_tip[:, 1].max() - 0.01) < 0.0015
            assert np.abs(plan_tip[:, 0] - 0.22).max() < 0.001
            assert meas_tip[:, 2].min() - TZ >= TL.MIN_FREE_HEIGHT
            finger = tr.tip(None)[0]                                           # gripper_frame_link: somewhere else
            assert np.linalg.norm(finger - plan_tip, axis=1).min() > 0.04
            # --phys: the torque-level servo model, same tool, still in the air
            n = len(seen)
            r, rc = fd_run(["--sim", "track-test", "--phys", "--shape", "line", "--size", "10"])
            assert rc == 0 and r["ok"] and r["tool"] == "pen2", r
            arm, tr = seen[-1]
            assert len(seen) == n + 1 and isinstance(arm.bus, PB.PhysBus) and arm.tool.name == "pen2"
            assert tr.tip(arm.tool)[1][:, 2].min() - TZ >= TL.MIN_FREE_HEIGHT
            r, rc = fd_run(["--sim", "track-test", "--height", "0.015"])
            assert rc == 1 and r["error"] == "SafetyError", r                    # in the air only
        with tool_file(None):
            r, rc = fd_run(["--sim", "track-test", "--dry"])
            assert rc == 0 and r["stage"] == "planned" and r["tool"] is None and r["pitch"] == 75.0, r
    finally:
        TR.record = real_record


def test_fd_track_test_phys_carries_the_simulated_tool():
    """`fd --sim track-test --phys` builds a second Arm on the physics servo model. Arm() by itself reads
    the REAL arm's tool file: the figure would be planned for another tool than the simulation carries."""
    real, simf = os.path.join(_TMP, "real_tool.json"), os.path.join(_TMP, "sim_tool.json")
    saved = (TL.TOOL_FILE, TL.SIM_TOOL_FILE, os.environ.pop("FD_TOOL"))
    TL.TOOL_FILE, TL.SIM_TOOL_FILE = real, simf
    keys = ("tool", "pitch", "z", "length_mm", "joint_speed_deg_s", "expected_s")
    try:
        TL.save(pen(name="sim_pen", pitch_deg=70.0), sim=True)                 # only the simulation has a tool
        assert TL.active() is None and TL.active(sim=True).name == "sim_pen"
        plain, rc = fd_run(["--sim", "track-test", "--dry"])
        assert rc == 0 and plain["tool"] == "sim_pen" and plain["pitch"] == 70.0, plain
        phys, rc = fd_run(["--sim", "track-test", "--phys", "--dry"])
        assert rc == 0 and {k: phys[k] for k in keys} == {k: plain[k] for k in keys}, (phys, plain)
        TL.save(pen(name="real_pen", pitch_deg=75.0))                          # the real arm carries another one
        phys, rc = fd_run(["--sim", "track-test", "--phys", "--dry"])
        assert rc == 0 and {k: phys[k] for k in keys} == {k: plain[k] for k in keys}, (phys, plain)
        TL.clear(sim=True)                                                     # no simulated tool: the real one is mirrored
        phys, rc = fd_run(["--sim", "track-test", "--phys", "--dry"])
        assert rc == 0 and phys["tool"] == "real_pen" and phys["pitch"] == 75.0, phys
    finally:
        TL.TOOL_FILE, TL.SIM_TOOL_FILE = saved[0], saved[1]
        os.environ["FD_TOOL"] = saved[2]
        for p in (real, simf):
            if os.path.exists(p):
                os.remove(p)


# ============================================================================= 7. fd snap
def test_fd_snap_judges_the_simulated_photos_with_a_tool_attached():
    s = square_run()
    files = {}
    for name in ("after", "after_desk"):
        files[name] = os.path.join(_TMP, name + ".jpg")
        assert cv2.imwrite(files[name], s[name], [cv2.IMWRITE_JPEG_QUALITY, 95])
    with tool_file(pen()):
        r, rc = fd_run(["--sim", "snap", "--frame", files["after"], "--exposure", "paper"])
        assert rc == 0 and r["ok"] and r["exposure"] == "paper" and r["frame"] == files["after"], r
        assert r["clipped"] < 0.02 and r["ink_visible"] is True and "next" not in r, r
        for k in ("tip_xyz", "q_deg", "raw", "view", "crop", "scale"):          # the keys snap always had
            assert k in r, k
        # "tip" is the TOOL tip: the simulated arm starts limp in the tool's rest pose, the tip 5 cm above the table
        assert abs(r["tip_xyz"][2] - (TZ + TL.REST_CLEARANCE)) < 0.003, r["tip_xyz"]
        assert r["raw"].startswith(fd.RUNS) and os.path.isfile(r["raw"]) and os.path.isfile(r["view"])
        r, rc = fd_run(["--sim", "snap", "--frame", files["after_desk"]])
        assert rc == 0 and r["exposure"] == "desk" and r["clipped"] > 0.3 and r["ink_visible"] is False, r
        assert "--exposure paper" in r["next"] and r["say"], r
    with tool_file(None):
        r, rc = fd_run(["--sim", "snap", "--frame", files["after"]])
        assert rc == 0 and r["exposure"] == "desk" and 0.0 < r["tip_xyz"][2] - TZ < 0.04, r   # the fingertip, parked


# ============================================================================= 8. marks -> map -> tool file -> drawing
def test_marks_make_a_map_the_tool_file_keeps_it_and_the_drawing_lands():
    """What `fd tool calibrate` (stage 2) has to do, with the pieces of stage 1: marks at commanded places,
    ink.marks_readout, strokes.fit_map, Tool.corr_map through the tool file, strokes.plan_pieces with it.
    The world is sheared by 10 % and shifted; the drawing must land on its lines all the same."""
    with tool_file(pen()):
        arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=TL.active())
        assert TL.sim_start(arm)
        arm.open()
        paper = SP.SimPaper(arm, CAM, pos_error=SP.shear((0.2575, 0.003), kxy=-0.10, offset=(-0.004, 0.002)), seed=2)
        TL.go_ready(arm)
        lean = functools.partial(TL.lean_xy, arm, arm.tool)
        aside(arm)
        before = paper.render("paper", noise=NOISE)
        grid = np.array([(x, y) for x in (0.21, 0.2575, 0.305) for y in (-0.05, 0.003, 0.056)])
        for g in grid:
            side = np.array([-lean(g)[1], lean(g)[0]])                         # across the lean: neither pushed nor pulled
            assert dash(arm, g - 0.0015 * side, g + 0.0015 * side, TZ - 0.001) > 0.002
        aside(arm)
        marked = paper.render("paper", noise=NOISE)
        mr = INK.marks_readout(marked, CAM, TZ, grid, before=before)
        assert mr["found"] == 9 and mr["judgeable"] and mr["missing"] == [] and mr["unknown"] == [], [m["why"] for m in mr["marks"]]
        meas = np.array([m["xy"] for m in mr["marks"]])
        true = np.array([ln["xy"].mean(axis=0) for ln in paper.ink(new_only=True)])
        assert np.abs(meas - true).max() < 0.0008, np.abs(meas - true).max()    # the camera model holds on the paper
        assert np.linalg.norm(meas - grid, axis=1).max() > 0.006                # ... and the ink is NOT where it was aimed
        m = ST.fit_map(grid, meas)
        assert m["kind"] == "affine" and m["n"] == 9 and m["rms_mm"] < 0.6, m
        assert abs(np.array(m["coef"])[2][0] + 0.10) < 0.02, m["coef"]           # d actual_x / d commanded_y: the shear
        # into the tool file and back
        tool = TL.active()
        tool.corr_map = m
        tool.calibrated["map"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rms_mm": m["rms_mm"], "n": m["n"]}
        TL.save(tool)
        again = TL.active()
        assert again.corr_map == json.loads(json.dumps(m)) and TL.calibration_state(again) == {"tip": False, "contact": False, "map": True}
        assert np.abs(ST.apply_map(ST.invert_map(meas, again.corr_map), again.corr_map) - meas).max() < 5e-8   # 0.05 um
        arm.tool = again                                                       # (an Arm keeps the tool it was built with)
        # a square inside the grid, pre-distorted with the stored map
        sq = np.array([[0.225, 0.02], [0.225, 0.044], [0.249, 0.044], [0.249, 0.02], [0.225, 0.02]])
        hidden = np.array(paper._pos_error(0.237, 0.032)) * 1000.0
        assert float(np.linalg.norm(hidden)) > 5.0, hidden                      # without the map: 7 mm off here
        plan = ST.plan_pieces([sq], lean, arm.tool.corr_map)
        assert not np.allclose(plan["pieces"][0]["cmd"], plan["pieces"][0]["ink"], atol=0.003)
        draw(arm, plan["pieces"], lambda p: TZ - 0.001)
        aside(arm)
        drawn = paper.render("paper", noise=NOISE)
        cov = INK.stroke_coverage(drawn, CAM, TZ, [sq], before=marked)
        truth = paper.coverage([sq], tol_m=0.0015)
        assert cov["judgeable"] and cov["drawn_fraction"] >= 0.95 and cov["offset_mm"]["mean"] <= 1.0, cov
        assert truth["drawn_fraction"] >= 0.97 and truth["offset_mm"]["mean"] < 0.6, truth
        print(f"    9 marks read within {np.abs(meas - true).max() * 1000:.2f} mm, map rms {m['rms_mm']} mm; square drawn with the "
              f"map: coverage {cov['drawn_fraction']:.3f}, offset {cov['offset_mm']['mean']} mm (hidden error there "
              f"{np.linalg.norm(hidden):.1f} mm)")


# ============================================================================= 9. suites and the tool on the real arm
def test_the_suites_do_not_depend_on_the_tool_on_the_arm():
    """A tool file in the tree (the pen is really attached) must not change a regression suite: otherwise
    people run `fd tool detach` to get the tests through, with the pen still on the gripper. Every suite that
    builds an Arm pins $FD_TOOL; test_sim.py passes while $FD_TOOL holds a pen."""
    for name in sorted(os.listdir(HERE)):
        if name.startswith("test_") and name.endswith(".py"):
            with open(os.path.join(HERE, name)) as fh:
                src = fh.read()
            if "Arm(" in src:
                assert 'os.environ["FD_TOOL"] = ' in src, f"{name} builds an Arm and does not pin FD_TOOL"
    if os.environ.get("JH_QUICK"):
        raise Skip("JH_QUICK set")
    with tool_file(pen()):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")                    # FD_TOOL: a file that holds the pen
        env.pop("FD_CONFIG", None)
        env.pop("FD_CAMERA_MODEL", None)
        p = subprocess.run([sys.executable, os.path.join(HERE, "test_sim.py")], capture_output=True, text=True,
                           timeout=600, env=env, cwd=HERE)
        tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-400:]
        assert p.returncode == 0 and "failures: 0" in tail and "tests: 97" in tail, tail
        assert TL.active().name == "pen2"                                      # and the suite left the tool file alone


def test_zz_no_hardware_and_nothing_written_into_the_tree():
    """Runs last (zz): the audit hook saw no serial port, no video device, no v4l2-ctl; the tree has no new
    run directory, no backup and no tool file from these tests."""
    assert _TRIPPED == [], _TRIPPED
    try:
        open("/dev/ttyACM0")
    except RuntimeError:
        _TRIPPED.clear()                                   # the guard works (and nothing was opened)
    except OSError:
        raise AssertionError("the audit hook is not armed")
    else:
        raise AssertionError("the audit hook is not armed")
    now = {d: sorted(os.listdir(os.path.join(tazuna_paths.ROOT, d))) if os.path.isdir(os.path.join(tazuna_paths.ROOT, d)) else None
           for d in _TREE_BEFORE}
    assert now == _TREE_BEFORE, (now, _TREE_BEFORE)
    assert os.environ["FD_TOOL"] == TOOL_PATH and TL.TOOL_FILE == os.path.join(tazuna_paths.ROOT, "tool_active.json")
    assert not os.path.exists(TOOL_PATH)


if __name__ == "__main__":
    import shutil
    import traceback
    only = sys.argv[1:]
    fails = skipped = 0
    n = 0
    t_all = time.time()
    try:
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
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print(f"tests: {n}  passed: {n - fails - skipped}  skipped: {skipped}  failures: {fails}  ({time.time() - t_all:.0f} s)")
    sys.exit(1 if fails else 0)
