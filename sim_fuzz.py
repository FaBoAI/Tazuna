"""Workspace sweep of the whole pick sequence on the simulated arm (no hardware, no camera).

Random scenes (one target, the container, up to two bystanders) are planned with the real
grasp planner and executed with the real skill and motion code on the simulator. Reports how
often a plan exists, whether a planned pick then runs through, and how long the robot would
take per stage at each speed tier.

Run:  python sim_fuzz.py [-n 60] [--tiers slow,normal,fast] [--seed 1] [--phys] [--json out.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time

import numpy as np

import fdtypes as T
import skills
from robot import Arm, load_config
from sim_bus import SimBus

TABLE_Z = None  # filled from the config


def make_box(i, color, cx, cy, size, yaw, z_bottom):
    length, width, _h = size
    c, s = math.cos(yaw), math.sin(yaw)
    fp = [(cx + sx * length / 2 * c - sy * width / 2 * s, cy + sx * length / 2 * s + sy * width / 2 * c)
          for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return T.SceneObject(id=i, kind="box", color=color, center=(cx, cy), yaw=yaw, size=size, z_bottom=z_bottom,
                         footprint=fp, pixel_bbox=(0, 0, 10, 10), confidence=1.0)


def make_cyl(i, color, cx, cy, d, h, z_bottom):
    fp = [(cx + d / 2 * math.cos(t), cy + d / 2 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    return T.SceneObject(id=i, kind="cylinder", color=color, center=(cx, cy), yaw=0.0, size=(d, d, h),
                         z_bottom=z_bottom, footprint=fp, pixel_bbox=(0, 0, 10, 10), confidence=1.0)


def make_container(i, cx, cy, yaw_deg=4.5, outer=(0.121, 0.110), rim=0.030, wall=0.004, floor=0.003):
    o = make_box(i, "black", cx, cy, (outer[0], outer[1], rim), math.radians(yaw_deg), TABLE_Z)
    o.kind = "container"
    o.inner = {"center": (cx, cy), "yaw": o.yaw, "size": (outer[0] - 2 * wall, outer[1] - 2 * wall),
               "floor_z": TABLE_Z + floor, "rim_z": TABLE_Z + rim}
    return o


def random_scene(rng, n_extra):
    con = make_container(9, 0.262 + rng.uniform(-0.01, 0.01), 0.085 + rng.uniform(-0.01, 0.01), rng.uniform(-10, 10))
    objs, tries = [], 0
    while len(objs) < 1 + n_extra and tries < 200:
        tries += 1
        x, y = rng.uniform(0.15, 0.34), rng.uniform(-0.17, 0.02)
        if any(math.hypot(x - o.center[0], y - o.center[1]) < 0.05 for o in objs):
            continue
        i = len(objs) + 1
        color = ["blue", "red", "green"][len(objs)]
        if rng.random() < 0.6:
            w = rng.uniform(0.015, 0.035)
            objs.append(make_box(i, color, x, y, (rng.uniform(w, 0.06), w, rng.uniform(0.012, 0.03)),
                                 rng.uniform(-math.pi / 2, math.pi / 2), TABLE_Z))
        else:
            objs.append(make_cyl(i, color, x, y, rng.uniform(0.018, 0.036), rng.uniform(0.012, 0.03), TABLE_Z))
    return T.Scene(objects=objs + [con], table_z=TABLE_Z)


class SynthScene:
    """Perception stand-in: the scene is known; the object arrives when the jaws open over the box."""

    def __init__(self, sc, arm):
        self.sc, self.arm, self.released = sc, arm, False

    def perceive(self, img, cam, cfg, arm_q=None, jaw_q_deg=None, debug=None):
        return self.sc

    def annotate(self, img, sc, cam):
        return img

    def objects_in_container(self, img, cam, cfg, con, arm_q=None, jaw_q_deg=None):
        return [{"color": "blue", "area_px": 900.0}] if self.released else []

    def object_present(self, img, cam, cfg, obj, arm_q=None):
        return not self.released

    def detect_jaws(self, *a, **k):
        return {"ok": False, "confidence": 0.0, "dxy_base": None}


def one_run(sc, tier, G, phys=False, dry=False, start="standby"):
    cfg = copy.deepcopy(load_config())
    cfg["speed_tiers_validated"] = list(cfg["speed_tiers"])  # simulation only (never written to disk)
    if phys:
        from phys_bus import PhysBus
        bus = PhysBus(cfg)
    else:
        bus = SimBus(realtime=False)
    with Arm(bus=bus, cfg=cfg) as arm:
        S = SynthScene(sc, arm)
        stall = {"raw": None}
        bus.blocker = lambda pos: stall["raw"] is not None and pos["gripper"] < stall["raw"]
        real_grasp, real_grip = arm.grasp_fast, arm.set_gripper

        def grasp_fast(expected_raw=None, **kw):
            stall["raw"] = expected_raw
            return real_grasp(expected_raw=expected_raw, **kw)

        def set_gripper(*a, **kw):
            S.released = True  # the only set_gripper call of the sequence is the release
            return real_grip(*a, **kw)

        arm.grasp_fast, arm.set_gripper = grasp_fast, set_gripper
        sess = skills.Session(arm, skills.ReplayCamera([np.zeros((4, 4, 3), np.uint8)]), None, image_dir=None)
        if start != "parked" and not dry:
            skills.go_ready(arm, tier)
            if start == "standby":  # steady state: the arm waits above the container after the previous pick
                con = sc.containers()[0]
                top = np.array([con.inner["center"][0], con.inner["center"][1], con.inner["rim_z"] + 0.065])
                q = arm.read_q()
                arm.run_path([q, arm.solve(top, skills._feasible_pitch(arm, top, 0.0, 90.0), 0.0, q_seed=q)], 25.0,
                             settle=False)
        opts = skills.PickOptions(tier=tier, visual="off", dry=dry)
        r = skills.pick_place(sess, {"color": "blue"}, None, opts, modules=(G, S))
        r["end_tip"] = arm.state()["tip_xyz"] if not dry else None
    return r


def main(argv=None):
    global TABLE_Z
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--tiers", default="slow,normal,fast")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--phys", action="store_true", help="torque-level simulator instead of the kinematic one")
    ap.add_argument("--start", choices=("standby", "ready", "parked"), default="standby",
                    help="where the arm is when the pick starts (standby = above the container, as after a previous pick)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    import grasp as G
    TABLE_Z = float(load_config()["table_z"])
    rng = random.Random(a.seed)
    tiers = a.tiers.split(",")
    rows, t_start = [], time.time()
    for k in range(a.n):
        sc = random_scene(rng, rng.choice([0, 0, 1, 2]))
        tgt = sc.objects[0]
        row = {"k": k, "kind": tgt.kind, "xy": [round(v, 3) for v in tgt.center],
               "size_mm": [round(v * 1000) for v in tgt.size], "yaw": round(math.degrees(tgt.yaw)),
               "n_objects": len(sc.objects) - 1}
        r = one_run(sc, tiers[0], G, dry=True)
        row["planned"] = bool(r["ok"])
        if not r["ok"]:
            row["plan_reason"] = r["reason"][:160]
            rows.append(row)
            continue
        for tier in tiers:
            try:
                r = one_run(sc, tier, G, phys=a.phys, start=a.start)
                row[tier] = {"ok": bool(r["ok"]), "stage": r["stage"], "reason": r["reason"][:200],
                             "arm_state": r.get("arm_state"), "t": r["timings"]}
            except Exception as e:  # a crash is a finding, not a reason to stop the sweep
                row[tier] = {"ok": False, "stage": "crash", "reason": f"{type(e).__name__}: {e}"[:200], "t": {}}
        rows.append(row)
    planned = [r for r in rows if r["planned"]]
    print(f"{len(rows)} scenes, {len(planned)} with a plan ({100 * len(planned) / max(1, len(rows)):.0f} %), "
          f"{time.time() - t_start:.0f} s of computing")
    why = {}
    for r in rows:
        if not r["planned"]:
            key = r["plan_reason"].split(":")[0][:70] + " | " + r["plan_reason"].split(":")[-1][:70]
            why[key] = why.get(key, 0) + 1
    for key, n in sorted(why.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   no plan x{n}: {key}")
    summary = {"n": len(rows), "planned": len(planned), "tiers": {}}
    for tier in tiers:
        res = [r[tier] for r in planned if tier in r]
        good = [x for x in res if x["ok"]]
        line = f"tier {tier}: {len(good)}/{len(res)} planned picks ran through"
        if good:
            tot = np.array([x["t"]["total"] for x in good])
            stages = {s: float(np.mean([x["t"].get(s, 0.0) for x in good]))
                      for s in ("perceive", "plan", "approach", "visual", "descend", "grasp", "transfer", "release", "verify")}
            line += (f"; robot time per pick: mean {tot.mean():.1f} s, median {np.median(tot):.1f} s, "
                     f"range {tot.min():.1f}-{tot.max():.1f} s")
            summary["tiers"][tier] = {"ok": len(good), "of": len(res), "mean_s": round(float(tot.mean()), 2),
                                      "median_s": round(float(np.median(tot)), 2), "min_s": round(float(tot.min()), 2),
                                      "max_s": round(float(tot.max()), 2),
                                      "stages_mean_s": {k: round(v, 2) for k, v in stages.items()}}
            print(line)
            print("   mean per stage: " + ", ".join(f"{k} {v:.2f}" for k, v in stages.items() if v > 0.005))
        else:
            print(line)
        bad = {}
        for x in res:
            if not x["ok"]:
                key = f"{x['stage']}: {x['reason'][:110]}"
                bad[key] = bad.get(key, 0) + 1
        for key, n in sorted(bad.items(), key=lambda kv: -kv[1])[:8]:
            print(f"   failed x{n}: {key}")
    if a.json:
        json.dump({"summary": summary, "rows": rows}, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
