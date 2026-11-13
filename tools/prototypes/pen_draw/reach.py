"""Where can the pen tip go? (simulation only: no hardware)"""
import sys, math
import numpy as np
import penlib as P
from penlib import K, SafetyError

off = [float(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.056, 0.0135, 0.0068]
P.write_pen_config(off)
home_raw = {"shoulder_pan": 1956, "shoulder_lift": 1145, "elbow_flex": 3029, "wrist_flex": 2118, "wrist_roll": 2048, "gripper": 2054}
P.save_state(home_raw=home_raw, tip_offset=off)
arm = P.make_arm(True).open()
lim = arm.q_limits()
print("limits deg:", {j: (round(math.degrees(a), 1), round(math.degrees(b), 1)) for j, (a, b) in lim.items()})
q0 = arm.read_q()
print("home q deg", np.degrees(q0).round(2), "pen tip", K.fk(q0, arm.tip_tool())[:3, 3].round(4))
ZT = arm.cfg["table_z"]


def info(x, y, z, pitch, roll=0.0):
    try:
        q = arm.solve(np.array([x, y, z]), pitch, roll)
        arm.check_q(q, allow_contact=True, contact_depth=0.003)
    except SafetyError as e:
        return None, str(e)[:60]
    fr = K.link_frames(q)
    marg = min(min(q[i] - lim[j][0], lim[j][1] - q[i]) for i, j in enumerate(P.ARM))
    jaws = arm._jaw_samples(None, None)
    jz = arm._jaw_low_z(fr[4] @ arm._tool_gf, jaws)
    return (q, math.degrees(marg), fr[3][2, 3], fr[4][2, 3], jz), ""


for pitch in (-10, 0, 10, 20, 30, 40, 50):
    print("== pitch %d" % pitch)
    for y in (0.0, 0.08, -0.08):
        ok = []
        for xi in range(12, 40):
            x = xi / 100.0
            r, why = info(x, y, ZT, pitch)
            r2, _ = info(x, y, ZT + 0.03, pitch)
            if r and r2:
                ok.append((x, r))
        if ok:
            xs = [o[0] for o in ok]
            print("  y=%+.2f: x %.2f..%.2f  | margin min %.1f deg | wrist z min %.3f | grip-body z min %.3f | jaw low z min %.3f" % (
                y, min(xs), max(xs), min(o[1][1] for o in ok), min(o[1][2] for o in ok), min(o[1][3] for o in ok), min(o[1][4] for o in ok)))
        else:
            print("  y=%+.2f: none (%s)" % (y, why))
