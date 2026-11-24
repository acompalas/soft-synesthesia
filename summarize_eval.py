"""Table of the checkpoint evaluations made by eval_checkpoints.sh (rollouts/eval/<run>_ep<E>_<split>/eval.npz).

Per run and epoch, on the train objects and the held-out objects: mean time to fall [s] (50 s = never
dropped), mean |rotation| [turns] and the share of episodes that ended in a drop. Per-object means are taken
first, so every object counts equally. The best epoch per run is the one with the most rotation per episode
on the TRAIN objects (rotation within the 50 s episode, so it rewards both turning and not dropping; time to
fall breaks ties), so the held-out objects stay untouched for the final comparison.

Usage: python summarize_eval.py
"""
import glob
import os
import re

import numpy as np

rows = {}
for f in sorted(glob.glob("rollouts/eval/*/eval.npz")):
    m = re.match(r"(\w+)_ep(\d+)_(train|test)$", os.path.basename(os.path.dirname(f)))
    if not m:
        continue
    run, ep, split = m.group(1), int(m.group(2)), m.group(3)
    d = np.load(f)
    objs = np.unique(d["object"])
    per_obj = lambda x: np.mean([x[d["object"] == o].mean() for o in objs])
    rows.setdefault((run, ep), {})[split] = (per_obj(d["ttf_s"]), per_obj(np.abs(d["rotation_rad"])) / (2 * np.pi),
                                             per_obj(d["dropped"].astype(float)), len(d["object"]))

print(f"{'run':10s} {'epoch':>5s} | {'train TTF':>9s} {'turns':>6s} {'drop':>5s} | {'test TTF':>8s} {'turns':>6s} {'drop':>5s}")
best = {}
for (run, ep), v in sorted(rows.items()):
    cells = []
    for sp in ("train", "test"):
        cells.append(f"{v[sp][0]:9.1f} {v[sp][1]:6.2f} {v[sp][2]:5.0%}" if sp in v else f"{'-':>9s} {'-':>6s} {'-':>5s}")
    print(f"{run:10s} {ep:5d} | {cells[0]} | {cells[1]}")
    key = lambda r: (round(rows[r]["train"][1], 2), rows[r]["train"][0])          # turns, then time to fall
    if "train" in v and (run not in best or key((run, ep)) > key((run, best[run]))):
        best[run] = ep
print("\nbest epoch per run, by rotation per episode on the TRAIN objects (held-out objects kept for the final comparison):")
for run, ep in best.items():
    print(f"  {run}: epoch {ep}")
