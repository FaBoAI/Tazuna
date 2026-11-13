#!/home/jetson/miniconda/envs/lerobot/bin/python
"""One ramp stroke, then a photo and a zoomed crop of it:  stroke_test.py NAME X YA YB Z_HI Z_LO  (metres, model heights)"""
import json
import sys

import penlib as P
from penlib import K, camera, cv2, np

name, x, ya, yb, z_hi, z_lo = sys.argv[1], *[float(v) for v in sys.argv[2:7]]
cam = camera.CameraModel.load()
arm = P.make_arm(False).open()
try:
    if not arm.is_torqued():
        arm.torque_on()
    P.tip_path(arm, [(x, ya, z_hi + 0.012)], 45.0, settle=0.15)
    P.tip_path(arm, [(x, ya, z_hi)], 15.0, settle=0.4)
    q1 = arm.read_q()
    P.tip_path(arm, [(x, yb, z_lo)], 12.0, settle=0.4, step=0.002)
    q2 = arm.read_q()
    tool = arm.tip_tool()
    print("measured tip at start", K.fk(q1, tool)[:3, 3].round(4), "at end", K.fk(q2, tool)[:3, 3].round(4))
    P.tip_path(arm, [(x, yb, z_lo + 0.012)], 25.0, settle=0.1)
    P.go_aside(arm)
finally:
    arm.close()
path, img = P.snap_paper("st_" + name)
zt = -0.0015
pa, pb = cam.project([[x, ya, zt], [x, yb, zt]])
cx, cy = int((pa[0] + pb[0]) / 2), int((pa[1] + pb[1]) / 2)
x0, y0 = max(0, cx - 170), max(0, min(1080 - 200, cy - 100))
c = img[y0:y0 + 200, x0:x0 + 340]
g = c.min(axis=2).astype(np.float32)
lo, hi = np.percentile(g, 0.3), np.percentile(g, 99.5)
t = np.clip((g - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
t = cv2.cvtColor(cv2.resize(t, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
cv2.line(t, (int((pa[0] - x0) * 3), int((pa[1] - y0) * 3)), (int((pb[0] - x0) * 3), int((pb[1] - y0) * 3)), (0, 0, 255), 1)
out = "/tmp/tazuna_scratch/pen/st_%s.png" % name
cv2.imwrite(out, t)
print(json.dumps({"photo": path, "crop": out, "pred_px": [pa.round(0).tolist(), pb.round(0).tolist()], "crop_origin": [x0, y0]}))
