import sys, math
import numpy as np
import penlib as P
from penlib import K, SafetyError, camera
arm = P.make_arm(True).open()
lim = arm.q_limits()
ZT = arm.cfg["table_z"]
cam = camera.CameraModel.load()
print("image bottom row on the table:", [cam.to_plane(np.array([u, 1079.0]), ZT).round(3).tolist() for u in (0, 480, 960, 1440, 1919)])
print("row 1000:", [cam.to_plane(np.array([u, 1000.0]), ZT).round(3).tolist() for u in (0, 960, 1919)])
print("paper far corners (view px 180,330 / 915,318):", cam.to_plane(np.array([270.0, 495.0]), ZT).round(3), cam.to_plane(np.array([1372.0, 477.0]), ZT).round(3))
def marg(x, y, z, pitch):
    try:
        q = arm.solve(np.array([x, y, z]), pitch, 0.0)
        arm.check_q(q, allow_contact=True, contact_depth=0.003)
    except SafetyError:
        return None
    m = [(min(q[i] - lim[j][0], lim[j][1] - q[i]), j) for i, j in enumerate(P.ARM)]
    return min(m), q
for pitch in (40, 45, 50):
    print("== pitch", pitch)
    for xi in range(180, 330, 10):
        x = xi / 1000.0
        row = []
        for y in (0.0, 0.035, 0.07):
            a, b = marg(x, y, ZT - 0.001, pitch), marg(x, y, ZT + 0.03, pitch)
            if a and b:
                m = min(a[0], b[0])
                row.append("%5.1f(%s)" % (math.degrees(m[0]), m[1][:5]))
            else:
                row.append("   -      ")
        q = marg(x, 0.0, ZT, pitch)
        print("  x=%.2f  margins y=0/.035/.07: %s   q(y=0)=%s" % (x, " ".join(row), None if not q else np.degrees(q[1]).round(1)))
