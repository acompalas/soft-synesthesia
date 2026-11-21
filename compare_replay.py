"""Compare pad forces from state replays of the same recorded rollout on different robots.

  A = original robot, replay 1        B = original robot, replay 2 (noise floor: same robot twice)
  S = split robot (sum of each pad's 4 cells)

For every pad and step, all three start from the identical recorded state, so A vs B shows how much
the simulator alone varies, and A vs S shows what the split changes. Prints per-pad and overall
statistics and the binary agreement at fixed thresholds, and saves compare_replay.png.

Usage:
  python compare_replay.py rollouts/replay/orig_a.npz rollouts/replay/orig_b.npz rollouts/replay/split.npz
"""
import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

TOUCH_N = 0.1
THRESHOLDS = [1.0, 1.5, 2.0]


def stats(x, y):
    """x, y: (T, 16) pad forces. Returns dict over pad-steps where either is touched."""
    m = (x > TOUCH_N) | (y > TOUCH_N)
    if not m.any():
        return None
    a, b = x[m], y[m]
    rel = np.abs(a - b) / np.maximum(np.maximum(a, b), TOUCH_N)
    out = {"n": int(m.sum()),
           "med_rel": float(np.median(rel)),
           "p90_rel": float(np.percentile(rel, 90)),
           "corr_log": float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]) if m.sum() > 2 else float("nan")}
    for th in THRESHOLDS:
        out[f"agree@{th}"] = float(((x >= th) == (y >= th))[m].mean())
    return out


def fmt(s):
    if s is None:
        return "  (no contact)"
    ag = " ".join(f"{100 * s[f'agree@{th}']:5.1f}" for th in THRESHOLDS)
    return f"{s['n']:6d} {100 * s['med_rel']:8.1f} {100 * s['p90_rel']:8.1f} {s['corr_log']:7.3f}   {ag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig_a"); ap.add_argument("orig_b"); ap.add_argument("split")
    ap.add_argument("--no-show", dest="show", action="store_false")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    A, B, S = (np.load(p) for p in (args.orig_a, args.orig_b, args.split))
    fa, fb, fs = A["pad_force"], B["pad_force"], S["pad_force"]
    names = [str(n) for n in A["pad_names"]]
    print(f"steps {len(fa)}, substeps {int(A['substeps'])}; sensors: A={A['sensor']} B={B['sensor']} S={S['sensor']}")
    hdr = f"  {'':18s} {'n':>6s} {'med rel%':>8s} {'p90 rel%':>8s} {'r(log)':>7s}   agree% at " + " ".join(f"{t}N" for t in THRESHOLDS)
    for label, y in (("A vs B (noise floor)", fb), ("A vs S (split effect)", fs)):
        print(f"\n{label}\n{hdr}")
        for p in range(fa.shape[1]):
            print(f"  {names[p]:18s} {fmt(stats(fa[:, [p]], y[:, [p]]))}")
        print(f"  {'ALL PADS':18s} {fmt(stats(fa, y))}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for ax, y, title in ((axs[0], fb, "original vs original (noise floor)"),
                         (axs[1], fs, "original vs split (sum of cells)")):
        m = (fa > TOUCH_N) | (y > TOUCH_N)
        ax.scatter(np.maximum(fa[m], 1e-2), np.maximum(y[m], 1e-2), s=5, alpha=0.4)
        lim = [1e-2, max(fa.max(), y.max(), 1.0) * 1.5]
        ax.plot(lim, lim, "k--", lw=0.8)
        for th in THRESHOLDS:
            ax.axvline(th, color="r", lw=0.4, alpha=0.5); ax.axhline(th, color="r", lw=0.4, alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("original pad force (N)"); ax.set_ylabel("comparison pad force (N)"); ax.set_title(title, fontsize=10)
    fig.suptitle("State replay: pad force from identical states (red lines: 1, 1.5, 2 N)", fontsize=11)
    out = os.path.join(os.path.dirname(os.path.abspath(args.split)), "compare_replay.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
