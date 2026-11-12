#!/home/jetson/miniconda/envs/lerobot/bin/python
"""A "ladder" of short dashes at falling heights:  ladder.py NAME X0 Y Z_TOP [N=4] [DZ=0.001] [--sim]

Rung i is a 6 mm dash along y at x = X0 + 4 mm * i, drawn at the planned height Z_TOP - DZ * i exactly as a
drawing stroke is (settled hover, then the descent, the dash and the lift with the correction frozen). The
first rung that leaves ink tells the planned height at which the pen touches the paper. Then a photo (arm at
the side) and an enlarged crop for a look.
"""
import json
import sys
import time

import penlib as P
from penlib import K, camera, cv2, np

args = [v for v in sys.argv[1:] if not v.startswith("--")]
sim = "--sim" in sys.argv
name, x0, y, z_top = args[0], float(args[1]), float(args[2]), float(args[3])
n = int(args[4]) if len(args) > 4 else 4
dz = float(args[5]) if len(args) > 5 else 0.001
DASH, GAP = 0.006, 0.004
cam = camera.CameraModel.load()
arm = P.make_arm(sim).open()
zt = float(arm.cfg["table_z"])
rungs = []
try:
    if not arm.is_torqued():
        arm.torque_on()
    before = None if sim else P.snap_paper("lad_%s_before" % name)[1] if False else None
    for i in range(n):
        x, z = x0 + GAP * i, z_top - dz * i
        P.tip_path(arm, [(x, y - DASH / 2, z + P.LIFT)], 50.0, settle=True)          # settled hover above the start
        P.tip_path(arm, [(x, y - DASH / 2, z)], 12.0, settle=False)                   # down
        time.sleep(0.0 if sim else 0.4)
        q1 = arm.read_q()
        P.tip_path(arm, [(x, y + DASH / 2, z)], 6.0, settle=False, step=0.002)        # the dash
        time.sleep(0.0 if sim else 0.3)
        q2 = arm.read_q()
        P.tip_path(arm, [(x, y + DASH / 2, z + P.LIFT + 0.004)], 30.0, settle=0.3)    # up
        tool = arm.tip_tool()
        rungs.append({"i": i, "x": x, "z_plan": round(z, 5), "tip_start": K.fk(q1, tool)[:3, 3].round(4).tolist(),
                      "tip_end": K.fk(q2, tool)[:3, 3].round(4).tolist()})
        print(json.dumps(rungs[-1]), flush=True)
    P.go_aside(arm)
finally:
    arm.close()
if sim:
    print("sim ok")
    sys.exit(0)
path, img = P.snap_paper("lad_" + name)
pm = cam.project([[x0 + GAP * (n - 1) / 2, y, zt]])[0]
t, cx0, cy0 = P.stretch_crop(img, pm[0], pm[1], 150, 90, 4)
for r in rungs:
    pa, pb = cam.project([[r["x"], y - DASH / 2, zt], [r["x"], y + DASH / 2, zt]])
    cv2.line(t, (int((pa[0] - cx0) * 4), int((pa[1] - cy0) * 4)), (int((pb[0] - cx0) * 4), int((pb[1] - cy0) * 4)), (0, 0, 255), 1)
out = "/tmp/tazuna_scratch/pen/lad_%s.png" % name
cv2.imwrite(out, t)
P.save_state(**{"ladder_" + name: {"x0": x0, "y": y, "z_top": z_top, "n": n, "dz": dz, "rungs": rungs, "photo": path}})
print(json.dumps({"photo": path, "crop": out, "crop_origin": [cx0, cy0]}))
