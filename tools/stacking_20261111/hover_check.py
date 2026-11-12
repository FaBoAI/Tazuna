"""Hover the opened jaws above a planned grasp and draw where the model says the pads and the object are."""
import copy, json, math, os, subprocess, sys, time
HERE = '/home/jetson/lerobot/fable_direct'
sys.path.insert(0, HERE)
import numpy as np, cv2, camera
FD = os.path.join(HERE, 'fd')
cam = camera.CameraModel.load()
def fd(*a):
    p = subprocess.run([FD, *a], capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(p.stdout)
    except Exception:
        d = {'ok': False, 'message': (p.stdout + p.stderr)[:300]}
    print('%-6s ok=%s %s' % (a[0], d.get('ok'), (d.get('message') or '')[:200]), flush=True)
    return d
cx, cy, R, ztop, tx, ty, tz, psi, pitch, open_raw, tag = [float(v) for v in sys.argv[1:11]] + [sys.argv[11]]
fd('grip', '%.4f' % ((open_raw - 1998.0) / 1650.0))
r = fd('tip', str(tx), str(ty), str(tz), '--pitch', str(pitch), '--jaw-azimuth', str(psi), '--speed', '25')
if not r.get('ok'):
    sys.exit(1)
s = fd('snap', '--name', tag, '--plain')
tip = np.array(s['tip_xyz']); img = cv2.imread(s['raw'])
u0, v0 = cam.project([[cx, cy, ztop]])[0]
X0, Y0, SC = int(u0) - 200, int(v0) - 190, 3
big = cv2.resize(img[Y0:Y0 + 340, X0:X0 + 400], None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
def px(p):
    a, b = cam.project([p])[0]
    return int(round((a - X0) * SC)), int(round((b - Y0) * SC))
pts = np.array([px([cx + R * math.cos(t), cy + R * math.sin(t), ztop]) for t in np.linspace(0, 2 * math.pi, 73)], np.int32)
cv2.polylines(big, [pts], True, (0, 255, 255), 1, cv2.LINE_AA); cv2.drawMarker(big, px([cx, cy, ztop]), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
u = np.array([math.cos(math.radians(psi)), math.sin(math.radians(psi)), 0.0])
opening = (open_raw - 2090.0) / 8.9 / 1000.0
for p, col, nm in ((tip, (0, 0, 255), 'F'), (tip + opening * u, (255, 0, 255), 'M')):
    cv2.drawMarker(big, px(p), col, cv2.MARKER_TILTED_CROSS, 22, 2); cv2.putText(big, nm, (px(p)[0] + 10, px(p)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.circle(big, px([p[0], p[1], ztop]), 5, col, 1, cv2.LINE_AA)
out = os.path.join(HERE, 'runs/20261111_030203/%s_marked.jpg' % tag)
cv2.imwrite(out, big, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out, 'tip', tip.round(4))
