"""Stage A: hover the opened jaws above the planned pinch place and photograph them with the predicted positions drawn."""
import json, subprocess, sys, math, time
sys.path.insert(0, '/home/jetson/lerobot/fable_direct')
import numpy as np, cv2, camera
FD = '/home/jetson/lerobot/fable_direct/fd'
cam = camera.CameraModel.load()
def fd(*a):
    t0 = time.time()
    p = subprocess.run([FD, *a], capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(p.stdout)
    except Exception:
        d = {'ok': False, 'message': (p.stdout + p.stderr)[:300]}
    print('%-7s %5.1fs ok=%s %s' % (a[0], time.time() - t0, d.get('ok'), (d.get('message') or '')[:160]), flush=True)
    return d
cx, cy, x, y, z, psi, tag = [float(v) for v in sys.argv[1:7]] + [sys.argv[7]]
ZR = 0.020
if '--stay' not in sys.argv:
    fd('ready'); fd('grip', '0.1012')
    r = fd('tip', str(x), str(y), '0.045', '--pitch', '90', '--jaw-azimuth', str(psi), '--speed', '20')
    if not r.get('ok'):
        sys.exit(1)
r = fd('tip', str(x), str(y), str(z), '--pitch', '90', '--jaw-azimuth', str(psi), '--straight', '--speed', '8')
s = fd('snap', '--name', tag, '--plain')
tip = np.array(s['tip_xyz']); print('tip', tip.round(4), 'q', s.get('q_deg'), 'jaw_yaw', s.get('jaw_yaw_deg'))
img = cv2.imread(s['raw'])
X0, Y0, X1, Y1, SC = 980, 640, 1380, 980, 3
big = cv2.resize(img[Y0:Y1, X0:X1], None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
def px(p):
    u, v = cam.project([p])[0]
    return int(round((u - X0) * SC)), int(round((v - Y0) * SC))
u = np.array([math.cos(math.radians(psi)), math.sin(math.radians(psi)), 0.0])
# cap rim circle (outer R 15.5, inner 13.5) on the rim plane
for R, col in ((0.0155, (0, 255, 255)), (0.0135, (0, 200, 255))):
    pts = np.array([px([cx + R * math.cos(a), cy + R * math.sin(a), ZR]) for a in np.linspace(0, 2 * math.pi, 73)], np.int32)
    cv2.polylines(big, [pts], True, col, 1, cv2.LINE_AA)
cv2.drawMarker(big, px([cx, cy, ZR]), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
# predicted fixed pad end (the commanded fingertip) and moving pad end, at the fingertip height
cv2.drawMarker(big, px(tip), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2); cv2.putText(big, 'F', (px(tip)[0] + 10, px(tip)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
mv = tip + 0.014 * u
cv2.drawMarker(big, px(mv), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2); cv2.putText(big, 'M', (px(mv)[0] + 10, px(mv)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
# where those two ends will be at rim height when lowered straight down
for p, col in ((tip, (0, 0, 255)), (mv, (255, 0, 255))):
    cv2.circle(big, px([p[0], p[1], ZR]), 5, col, 1, cv2.LINE_AA)
out = '/home/jetson/lerobot/fable_direct/runs/20261111_030203/%s_marked.jpg' % tag
cv2.imwrite(out, big, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out, big.shape)
