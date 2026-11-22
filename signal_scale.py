"""How visible is the Park touch signal to the policy after its input normalisation?

rl_games normalises each policy input by its running mean and std (then clips to +-5 std), so what
matters is how far a touch moves an input in units of that input's std. Binary touch on the same
rollout (threshold 1.5 N on the pad force) is the reference.

Input: a touch_rollout.npz recorded with touch_model=park (see verify_park_sensors.py). Prints the
current readout and candidate log readouts r = max(0, log(g / eps)) / log(g_max / eps), g = pad dR/R0
(0 below the readout resolution eps).

Usage:
  python signal_scale.py rollouts/park_check/touch_rollout.npz
"""
import sys

import numpy as np

G_MAX = 9.0
d = np.load(sys.argv[1])
s, f = d["sensed"], d["force"]                     # (T, 16) policy readings 0..1, pad force norms N
eps_rec = float(d["park_log_eps"]) if "park_log_eps" in d.files else 0.0
g = s * G_MAX if eps_rec <= 0 else np.where(s > 0, eps_rec * np.exp(s * np.log(G_MAX / eps_rec)), 0.0)   # pad dR/R0
# (with a log readout, readings below eps were already 0 when recorded, so candidates are approximate)
touched = f >= 0.1


def describe(name, r):
    z = r / (r.std(0) + 1e-9)                      # shift from "untouched" in std units, per pad
    cap = r >= 0.999
    light = touched & (f < 2.0)
    print(f"{name:26s} median touch {np.median(z[touched]):5.2f} std, p90 {np.percentile(z[touched], 90):5.2f}, "
          f"capped {np.median(z[cap]) if cap.any() else float('nan'):5.2f}; touches under 2 N "
          f"{np.median(z[light]) if light.any() else float('nan'):5.2f} std")


print(f"touched pad-steps (> 0.1 N) {touched.mean():.1%}; pad readings at the cap {(g >= G_MAX - 1e-3).mean():.2%}; "
      f"recorded readout: {'linear' if eps_rec <= 0 else f'log, eps {eps_rec}'}")
b = (f >= 1.5).astype(float)
describe("binary (tau 1.5 N)", b)
describe("Park, linear readout", g / G_MAX)
for eps in (0.03, 0.01, 0.003):
    describe(f"Park, log readout eps {eps}", np.clip(np.log(np.maximum(g, 1e-12) / eps), 0, None) / np.log(G_MAX / eps))
