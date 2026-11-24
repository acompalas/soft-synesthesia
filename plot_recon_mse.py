"""Held-out reconstruction error for every rollout set, over seeds.

Reads models/shape_recon/<policy>_<pair>/seed*/results.json (shape_recon.py output) for the four sets
    baseline latest (ep5000), Park latest (ep5000), baseline best (ep4500), Park best (ep3600)
each trained with no touch and with the touch signal of its own policy (baseline: binary, Park: Park).

Figure (two square panels):
  (a) held-out MSE [mm^2] per set: no touch vs own sensor; bar = mean over seeds, dots = seeds, line = SD
  (b) reduction in MSE from adding touch, paired within each seed: 100 (none - touch) / none [%]
A set without results yet is marked "not yet". The size-only reference (mean outline x the true object
scale) is printed in the note under the figure; it is about 5x larger than the CNN errors, so it is not drawn.
Also prints the table behind the figure.

Usage (repo root):
    python plot_recon_mse.py
    python plot_recon_mse.py --root models/shape_recon --out deliverables/3_shape_reconstruction/recon_mse.png
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETS = [("baseline", "latest", "binary"), ("park", "latest", "park"),
        ("baseline", "best", "binary"), ("park", "best", "park")]
NAME = {("baseline", "latest"): "Baseline\nlatest", ("park", "latest"): "Park\nlatest",
        ("baseline", "best"): "Baseline\nbest", ("park", "best"): "Park\nbest"}
COLOR = {"none": "#8c8c8c", "binary": "#1f77b4", "park": "#d62728"}
LABEL = {"none": "No touch", "binary": "Binary touch", "park": "Park skin"}


def load(root, policy, pair):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "%s_%s" % (policy, pair), "seed*", "results.json"))):
        seed = os.path.basename(os.path.dirname(f))
        for r in json.load(open(f))["results"]:
            out.setdefault(r["signal"], {})[seed] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="models/shape_recon")
    ap.add_argument("--out", default="deliverables/3_shape_reconstruction/recon_mse.png")
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6.0))
    rng = np.random.default_rng(0)
    w = 0.36
    sized, rows = [], []
    for i, (pol, pair, sig) in enumerate(SETS):
        d = load(args.root, pol, pair)
        seeds = sorted(set(d.get("none", {})) & set(d.get(sig, {})))
        if not seeds:
            for ax in (ax1, ax2):
                ax.text(i, 0.02, "not yet", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="0.55")
            rows.append("%-16s not yet" % NAME[(pol, pair)].replace("\n", " "))
            continue
        none = np.array([d["none"][s]["heldout_mse_mm2"] for s in seeds])
        touch = np.array([d[sig][s]["heldout_mse_mm2"] for s in seeds])
        sized += [d["none"][s]["sized_mean_outline_mse_mm2"] for s in seeds]
        for x, v, key in ((i - w / 2, none, "none"), (i + w / 2, touch, sig)):
            ax1.bar(x, v.mean(), w * 0.92, color=COLOR[key], alpha=0.85, zorder=2)
            ax1.errorbar(x, v.mean(), v.std(ddof=1) if len(v) > 1 else 0, color="0.15", capsize=3, lw=1, zorder=3)
            ax1.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=12, color="white", edgecolors="0.15",
                        linewidths=0.8, zorder=4)
        red = 100 * (none - touch) / none
        ax2.bar(i, red.mean(), 0.55, color=COLOR[sig], alpha=0.85, zorder=2)
        ax2.errorbar(i, red.mean(), red.std(ddof=1) if len(red) > 1 else 0, color="0.15", capsize=3, lw=1, zorder=3)
        ax2.scatter(i + rng.uniform(-0.08, 0.08, len(red)), red, s=12, color="white", edgecolors="0.15",
                    linewidths=0.8, zorder=4)
        ax2.text(i, max(red.max(), 0) + 0.4, "%.1f%%" % red.mean(), ha="center", va="bottom", fontsize=8)
        rows.append("%-16s n=%d  none %.2f +- %.2f   %-6s %.2f +- %.2f   touch vs none %+.1f%% +- %.1f%%"
                    % (NAME[(pol, pair)].replace("\n", " "), len(seeds), none.mean(), none.std(ddof=1),
                       sig, touch.mean(), touch.std(ddof=1), -red.mean(), red.std(ddof=1)))

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(SETS)))
        ax.set_xticklabels([NAME[(p, q)] for p, q, _ in SETS], fontsize=9)
        ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_box_aspect(1)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("held-out MSE [mm$^2$], rotation-aligned")
    ax1.set_title("(a) Reconstruction error on the 25 held-out objects", fontsize=10)
    ax2.axhline(0, color="0.3", lw=0.8)
    ax2.set_ylabel("MSE reduction from touch [%] (same seed)")
    ax2.set_title("(b) What touch adds over the joints alone", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k], alpha=0.85, label=LABEL[k]) for k in ("none", "binary", "park")]
    handles.append(plt.Line2D([], [], ls="", marker="o", mfc="white", mec="0.15", label="one seed"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    note = "bars: mean over seeds, lines: 1 SD"
    if sized:
        note += ";  size-only reference (mean outline x true scale): %.1f mm$^2$" % np.mean(sized)
    fig.text(0.5, 0.06, note, ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("\n".join(rows))
    print("saved", args.out)


if __name__ == "__main__":
    main()
