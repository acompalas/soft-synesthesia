"""Why is a Park pad "on" (reading > 0, drawn green in the viewer)? Breaks the on-steps of a recorded
Park rollout down by the pad's normal force at that step.

  no force now   : reading held from an earlier step (sensor latency) or dropout bookkeeping
  < 0.1 N        : grazing / solver-level contact
  0.1 - 1 N      : light contact, below the binary threshold (tau 1-2 N)
  1 - 2 N        : around the binary threshold
  >= 2 N         : clear contact, binary would fire

Usage: python diagnose_park_firing.py rollouts/park_fig7
"""
import os
import sys

import numpy as np

d = np.load(os.path.join(sys.argv[1], "touch_rollout.npz"))
s = d["sensed"]                                            # (T, 16) what the policy got
fn = d["cell_fn"].reshape(len(s), 16, 4).sum(-1)           # (T, 16) pad normal force now, N
pads = [str(p).replace("_fsr", "") for p in d["sensor_names"]]
print(f"object {d['object']}, {len(s)} steps, latency {float(d['latency'])}, dropout {float(d['sensor_noise'])}")
edges = [(-1, 1e-6, "no force now"), (1e-6, 0.1, "<0.1 N"), (0.1, 1.0, "0.1-1 N"), (1.0, 2.0, "1-2 N"), (2.0, 1e9, ">=2 N")]
print(f"{'pad':14s} {'on-steps':>8s} " + " ".join(f"{e[2]:>13s}" for e in edges) + "   median reading when force < 1 N")
tot = np.zeros(len(edges))
for i, p in enumerate(pads):
    on = s[:, i] > 0
    if not on.any():
        print(f"{p:14s} {0:8d}"); continue
    f = fn[on, i]
    cnt = np.array([((f > lo) & (f <= hi)).sum() if lo >= 0 else (f <= hi).sum() for lo, hi, _ in edges])
    tot += cnt
    weak = on & (fn[:, i] < 1.0)
    print(f"{p:14s} {on.sum():8d} " + " ".join(f"{c / on.sum():13.0%}" for c in cnt) +
          f"   {np.median(s[weak, i]) if weak.any() else float('nan'):.3f}")
print(f"{'all pads':14s} {int(tot.sum()):8d} " + " ".join(f"{c / tot.sum():13.0%}" for c in tot))
