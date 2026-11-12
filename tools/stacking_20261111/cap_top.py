"""Put the cap on top of the tower: pinch the wall FAR from the tower (fixed finger inside the cavity, moving jaw outside,
pointing away from the tower), carry, set it on the hub.   cap_top.py hover | go"""
import json, math, os, subprocess, sys, time
HERE = '/home/jetson/lerobot/fable_direct'
sys.path.insert(0, HERE)
import numpy as np, cv2, camera
FD = os.path.join(HERE, 'fd'); cam = camera.CameraModel.load(); T0 = time.time()
def fd(*a):
    t0 = time.time()
    p = subprocess.run([FD, *a], capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(p.stdout)
    except Exception:
        d = {'ok': False, 'message': (p.stdout + p.stderr)[:300]}
    extra = {k: d.get(k) for k in ('pos_raw', 'load', 'above_touch_ticks') if k in d}
    print('[%5.1fs] %-6s %4.1fs ok=%s %s %s' % (time.time() - T0, a[0], time.time() - t0, d.get('ok'), extra or '', (d.get('message') or '')[:240]), flush=True)
    return d
TABLE = -0.0015
C = np.array([float(sys.argv[2]), float(sys.argv[3])])            # cap centre
Hc = np.array([float(sys.argv[4]), float(sys.argv[5])])           # hub top centre on the tower
ZHUB = TABLE + 0.0843
d = (C - Hc) / np.hypot(*(C - Hc)); psi = math.degrees(math.atan2(d[1], d[0]))
F = C + 0.0095 * d                                               # fixed pad 4 mm inside the far inner wall
ZG, ZHOV = TABLE + 0.002 + 0.0025, 0.050
psi_r = psi + (math.degrees(math.atan2(Hc[1], Hc[0])) - math.degrees(math.atan2(F[1], F[0])))
dr = np.array([math.cos(math.radians(psi_r)), math.sin(math.radians(psi_r))])
R = Hc + 0.0135 * dr                                             # fingertip when the cap's centre is over the hub
ZREL, ZAB = ZHUB + 0.0065, ZHUB + 0.0215
A = ('--pitch', '90', '--jaw-azimuth', '%.2f' % psi)
B = ('--pitch', '80', '--jaw-azimuth', '%.2f' % psi_r)
print('cap (%.4f, %.4f) hub (%.4f, %.4f): away direction %.1f deg; fixed pad at (%.4f, %.4f); release tip (%.4f, %.4f, %.4f) az %.1f' % (*C, *Hc, psi, *F, *R, ZREL, psi_r), flush=True)
if sys.argv[1] == 'hover':
    fd('grip', '0.1012')
    if not fd('tip', '%.4f' % F[0], '%.4f' % F[1], '%.4f' % ZHOV, *A, '--speed', '22').get('ok'):
        sys.exit(1)
    s = fd('snap', '--name', 'hov_cap', '--plain')
    tip = np.array(s['tip_xyz']); img = cv2.imread(s['raw'])
    u0, v0 = cam.project([[C[0], C[1], 0.015]])[0]
    X0, Y0, SC = int(u0) - 190, int(v0) - 230, 3
    big = cv2.resize(img[Y0:Y0 + 340, X0:X0 + 400], None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
    def px(p):
        a, b = cam.project([p])[0]
        return int(round((a - X0) * SC)), int(round((b - Y0) * SC))
    for Rr, col in ((0.0155, (0, 255, 255)), (0.0135, (0, 200, 255))):
        pts = np.array([px([C[0] + Rr * math.cos(t), C[1] + Rr * math.sin(t), 0.015]) for t in np.linspace(0, 2 * math.pi, 73)], np.int32)
        cv2.polylines(big, [pts], True, col, 1, cv2.LINE_AA)
    u = np.array([d[0], d[1], 0.0])
    for p, col, nm in ((tip, (0, 0, 255), 'F'), (tip + 0.014 * u, (255, 0, 255), 'M')):
        cv2.drawMarker(big, px(p), col, cv2.MARKER_TILTED_CROSS, 22, 2); cv2.putText(big, nm, (px(p)[0] + 10, px(p)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.circle(big, px([p[0], p[1], 0.015]), 5, col, 1, cv2.LINE_AA)
    out = os.path.join(HERE, 'runs/20261111_030203/hov_cap_marked.jpg')
    cv2.imwrite(out, big, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out, 'tip', tip.round(4))
else:
    dx, dy = (float(sys.argv[6]), float(sys.argv[7])) if len(sys.argv) > 7 else (0.0, 0.0)
    Fx, Fy = F[0] + dx, F[1] + dy
    if not fd('tip', '%.4f' % Fx, '%.4f' % Fy, '%.4f' % ZHOV, *A, '--speed', '15').get('ok'):
        sys.exit(1)
    if not fd('tip', '%.4f' % Fx, '%.4f' % Fy, '%.4f' % ZG, *A, '--straight', '--speed', '8').get('ok'):
        sys.exit(1)
    g = fd('grasp', '--thin')
    fd('tip', '%.4f' % Fx, '%.4f' % Fy, '%.4f' % ZHOV, *A, '--straight', '--speed', '8')
    s = fd('snap', '--name', 'cap_lifted', '--plain')
    img = cv2.imread(s['raw']).min(axis=2).astype(np.float32)
    u0, v0 = cam.project([[C[0], C[1], TABLE + 0.004]])[0]
    b = float(img[int(v0) - 6:int(v0) + 7, int(u0) - 10:int(u0) + 11].mean())
    print('brightness where the cap stood: %.0f (white cap = 230+, table = under 120)' % b, flush=True)
    if b > 170:
        print('ABORT: the cap is still on the table'); fd('grip', '0.1012'); fd('ready'); sys.exit(2)
    if not fd('tip', '%.4f' % R[0], '%.4f' % R[1], '%.4f' % ZAB, *B, '--speed', '18').get('ok'):
        print('carry refused: the cap is still held'); sys.exit(3)
    fd('tip', '%.4f' % R[0], '%.4f' % R[1], '%.4f' % ZREL, *B, '--straight', '--speed', '6')
    fd('grip', '0.1012')
    fd('tip', '%.4f' % R[0], '%.4f' % R[1], '%.4f' % ZAB, *B, '--straight', '--speed', '8')
    fd('ready')
    s = fd('snap', '--name', 'tower_done', '--plain')
    print('[%5.1fs] photo %s' % (time.time() - T0, s.get('raw')))
