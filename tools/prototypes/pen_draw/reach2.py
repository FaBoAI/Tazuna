import sys, math
import numpy as np
import penlib as P
from penlib import K, SafetyError
arm = P.make_arm(True).open()
lim = arm.q_limits()
ZT = arm.cfg["table_z"]
def ok(x, y, z, pitch):
    try:
        q = arm.solve(np.array([x, y, z]), pitch, 0.0)
        arm.check_q(q, allow_contact=True, contact_depth=0.003)
    except SafetyError:
        return None
    return min(min(q[i] - lim[j][0], lim[j][1] - q[i]) for i, j in enumerate(P.ARM)), q
for pitch in (15, 20, 25, 30, 35, 40):
    # x-range at |y| = 0, 0.04, 0.07 for z = table and table+0.03, requiring margin >= 3 deg
    out = []
    for y in (0.0, 0.04, 0.07, 0.09):
        xs = []
        for xi in range(150, 460, 5):
            x = xi / 1000.0
            a, b = ok(x, y, ZT - 0.001, pitch), ok(x, y, ZT + 0.03, pitch)
            if a and b and min(a[0], b[0]) >= math.radians(3.0):
                xs.append(x)
        out.append("y=%.2f: %s" % (y, ("%.3f..%.3f" % (min(xs), max(xs))) if xs else "none"))
    print("pitch %d (3 deg margin): " % pitch + " | ".join(out))
# which joint limits bind at pitch 30?
for x in (0.22, 0.24, 0.26, 0.30, 0.34, 0.38, 0.40, 0.42):
    r = ok(x, 0.0, ZT, 30)
    if r:
        print(" pitch30 x=%.2f q=%s margin %.1f" % (x, np.degrees(r[1]).round(1), math.degrees(r[0])))
    else:
        print(" pitch30 x=%.2f infeasible" % x)
