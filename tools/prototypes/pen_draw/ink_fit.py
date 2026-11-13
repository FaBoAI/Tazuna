#!/home/jetson/miniconda/envs/lerobot/bin/python
"""Correct the pen-tip point from the ink of a ladder:  ink_fit.py LADDER_NAME [--apply]

Each inked rung is found in the ladder's photo (dark blobs, ordered along x), its centre on the table is compared
with where the model had the pen tip while the dash was drawn, and the difference (base frame) is turned into the
gripper frame. The first inked rung also gives the height: the paper is where the pen first left ink."""
import json
import sys

import penlib as P
from penlib import K, camera, cv2, np

name = [v for v in sys.argv[1:] if not v.startswith("--")][0]
cam = camera.CameraModel.load()
st = P.load_state()
L = st["ladder_" + name]
cfg = P.load_config()
p_old = np.array(cfg["tip_offset"], float)
zt = float(cfg["table_z"])
img = cv2.imread(L["photo"])
bh = P.ink_map(img)
rungs = L["rungs"]
y = L["y"]
# search window: around the planned ladder, generous (the ink may be 2 cm off)
pm = cam.project([[L["x0"] + 0.006, y, zt]])[0]
x0, y0 = int(pm[0]) - 130, int(pm[1]) - 70
sub = (bh[y0:y0 + 170, x0:x0 + 260] > 28).astype(np.uint8)
n, lab, stt, cen = cv2.connectedComponentsWithStats(sub, 8)
blobs = []
for k in range(1, n):
    if stt[k, cv2.CC_STAT_AREA] < 6:
        continue
    ys, xs = np.where(lab == k)
    pts = np.array([cam.to_plane(np.array([float(a + x0), float(b + y0)]), zt)[:2] for a, b in zip(xs, ys)])
    blobs.append({"area": int(stt[k, cv2.CC_STAT_AREA]), "centre": pts.mean(axis=0), "x": [pts[:, 0].min(), pts[:, 0].max()],
                  "y": [pts[:, 1].min(), pts[:, 1].max()]})
blobs.sort(key=lambda b: b["centre"][0])
for b in blobs:
    print("ink blob: area %4d px  centre (%.4f, %.4f)  x %.4f..%.4f  y %.4f..%.4f" % (b["area"], *b["centre"], *b["x"], *b["y"]))
# dashes are at least 3 mm long in y; the LAST len(dashes) rungs are the inked ones (deeper = later)
dashes = [b for b in blobs if b["y"][1] - b["y"][0] > 0.003]
if not dashes:
    print(json.dumps({"ok": False, "message": "no inked dash found"}))
    sys.exit(1)
inked = rungs[len(rungs) - len(dashes):]
ds = []
for r, b in zip(inked, dashes):
    q_mid = None
    model = (np.array(r["tip_start"]) + np.array(r["tip_end"])) / 2.0     # where the model had the tip during the dash
    # the ink's own centre in x; in y the dash runs from the planned start to the planned end, so use the centre too
    E = np.array([b["centre"][0] - model[0], b["centre"][1] - model[1], 0.0])
    ds.append((r, b, model, E))
    print("rung %d (plan z %+.1f mm): model tip (%.4f, %.4f, %.4f)  ink centre (%.4f, %.4f)  E = (%+.1f, %+.1f) mm" % (
        r["i"], r["z_plan"] * 1000, *model, *b["centre"], E[0] * 1000, E[1] * 1000))
first = ds[0]
# height: at the first inked rung the real tip is on the paper (table + paper thickness)
E_first = first[3].copy()
E_first[2] = (zt + 0.0001) - first[2][2] - 0.0004      # just in contact: the tip is about 0.4 mm into its travel
arm = P.make_arm(True).open()
q = arm.solve(np.array([first[0]["x"], y, first[0]["z_plan"]]), P.PITCH, 0.0)
R = K.fk(q)[:3, :3]
d_loc = R.T @ E_first
p_new = p_old + d_loc
out = {"ok": True, "ladder": name, "first_inked_rung": first[0]["i"], "contact_plan_z_mm": round(first[0]["z_plan"] * 1000, 2),
       "E_base_mm": (E_first * 1000).round(2).tolist(), "delta_local_mm": (d_loc * 1000).round(2).tolist(),
       "tip_offset_old": p_old.round(5).tolist(), "tip_offset_new": p_new.round(5).tolist()}
if "--apply" in sys.argv:
    P.write_pen_config(p_new)
    P.save_state(tip_offset=p_new.round(5).tolist(), ink_fit=out)
    out["applied"] = True
print(json.dumps(out))
