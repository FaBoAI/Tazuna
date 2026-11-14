"""Run every suite, one after another, each in its own process; print one line per suite.

    python tests/run_all.py            all 20 (about 30 minutes on a Jetson Orin Nano)
    python tests/run_all.py ink tool   only the suites whose file name contains one of the words

The suites run with the Python that `fd` uses ($TAZUNA_PYTHON, else the first line of config/python, else the one
that runs this script), so `python` here may be any Python 3.
No hardware is used: the suites work on the simulated arm, simulated servos and saved photos.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUITES = [os.path.join("tools", "cli", "nlq.py")] + [os.path.join("tests", n) for n in (
    "test_ink.py", "test_tooling.py", "test_strokes.py", "test_tracking.py", "test_integration_pen.py", "test_penskill.py",
    "test_skills.py", "test_sim.py", "test_grasp.py", "test_kinematics.py", "test_scene.py", "test_bus_fake.py",
    "test_motion_phys.py", "test_hwlock.py", "test_cli_words.py", "test_tooljaw.py", "test_paint_model.py", "test_paint.py",
    "test_aircal.py")]


def harness_python() -> str:
    """The interpreter the launcher `fd` would take (same rule), else the one running this script."""
    py = os.environ.get("TAZUNA_PYTHON", "").strip()
    if not py:
        try:
            with open(os.path.join(ROOT, "config", "python"), encoding="utf-8") as f:
                py = next((ln.strip() for ln in f if ln.strip()), "")
        except OSError:
            py = ""
    return os.path.expanduser(py) if py else sys.executable


def main(argv) -> int:
    words = [os.path.basename(w) for w in argv[1:]]        # a path such as tests/test_ink.py selects that suite
    py = harness_python()
    print(f"python: {py}", flush=True)
    failed = ran = 0
    for rel in SUITES:
        if words and not any(w in os.path.basename(rel) for w in words):
            continue
        ran += 1
        t0 = time.time()
        p = subprocess.run([py, os.path.join(ROOT, rel)], capture_output=True, text=True, cwd=ROOT,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        lines = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()]
        failed += p.returncode != 0
        print(f"{'ok  ' if p.returncode == 0 else 'FAIL'} {rel:34s} {time.time() - t0:5.0f} s  {lines[-1][:200] if lines else ''}", flush=True)
    if not ran:
        print(f"no suite has {' or '.join(repr(w) for w in words)} in its file name: nothing was run")
        return 2
    print(("all suites passed" if ran == len(SUITES) else f"the {ran} selected suite(s) passed") if not failed else f"{failed} suite(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
