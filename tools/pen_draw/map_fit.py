#!/home/jetson/miniconda/envs/lerobot/bin/python
"""Fit the map commanded -> actual from the grid marks saved by grid_cal.py and store it (pen_state.json "map")."""
import json
import sys

import penlib as P
from penlib import np

st = P.load_state()
g = st["grid"]
blobs = [(np.array(b["xy"]), b["area"]) for b in g["blobs"]]
# merge pieces of one mark (closer than 7 mm)
merged = []
used = [False] * len(blobs)
for i, (xy, ar) in enumerate(blobs):
    if used[i]:
        continue
    grp = [(xy, ar)]
    used[i] = True
    for j in range(i + 1, len(blobs)):
        if not used[j] and np.linalg.norm(blobs[j][0] - xy) < 0.007:
            grp.append(blobs[j])
            used[j] = True
    w = np.array([a for _, a in grp], float)
    merged.append((np.sum([p * a for p, a in grp], axis=0) / w.sum(), int(w.sum())))
cmd = np.array([m["cmd"] for m in g["marks"]])
# one-to-one assignment, nearest first
pairs = sorted(((float(np.linalg.norm(c - b[0])), i, j) for i, c in enumerate(cmd) for j, b in enumerate(merged)))
ci, bj, match = set(), set(), {}
for d, i, j in pairs:
    if i in ci or j in bj or d > 0.03:
        continue
    match[i] = j
    ci.add(i)
    bj.add(j)
act = np.array([merged[match[i]][0] for i in sorted(match)])
c = cmd[sorted(match)]
for a_, c_, i in zip(act, c, sorted(match)):
    print("cmd (%.4f, %+.4f) -> ink (%.4f, %+.4f)  E = (%+5.1f, %+5.1f) mm  area %d" % (*c_, *a_, *((a_ - c_) * 1000), merged[match[i]][1]))


def feats(xy, quad):
    x, y = xy[:, 0] - P.XC, xy[:, 1] - P.YC
    cols = [np.ones_like(x), x, y]
    if quad:
        cols += [x * x, x * y, y * y]
    return np.stack(cols, axis=1)


out = {"n": int(len(c)), "centre": [P.XC, P.YC]}
for quad in (False, True):
    A = feats(c, quad)
    coef, *_ = np.linalg.lstsq(A, act, rcond=None)
    res = act - A @ coef
    rms = float(np.sqrt((res ** 2).sum(axis=1).mean()) * 1000)
    mx = float(np.sqrt((res ** 2).sum(axis=1)).max() * 1000)
    print("%s fit: rms %.2f mm, max %.2f mm" % ("quadratic" if quad else "affine", rms, mx))
    out["quad" if quad else "affine"] = {"coef": coef.tolist(), "rms_mm": round(rms, 2), "max_mm": round(mx, 2)}
aff = np.array(out["affine"]["coef"])
print("affine matrix d(actual)/d(cmd):\n", aff[1:].T.round(3), "\noffset at the centre (mm):", ((aff[0] - np.array([P.XC, P.YC])) * 1000).round(1))
out["use"] = "affine" if len(c) < 9 or out["affine"]["rms_mm"] < 1.3 * out["quad"]["rms_mm"] + 0.8 else "quad"
out["z_marks"] = g["z"]
P.save_state(map=out)
print(json.dumps({"use": out["use"], "affine_rms": out["affine"]["rms_mm"], "quad_rms": out["quad"]["rms_mm"], "n": out["n"]}))
