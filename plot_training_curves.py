"""Training curves in the style of Yin et al. Fig. 6: reward and time-to-fall (TTF) against training epochs.

Reads the TensorBoard event files rl_games wrote for each run (one seed per run, so there is no shaded
band across seeds as in the paper: the faint line is the raw per-epoch value, the solid line is an
exponential moving average, the same smoothing TensorBoard shows).

  reward : rewards/iter          mean episode reward of the training batch
  TTF    : episode_lengths/iter  x control period (0.1 s): mean episode length in seconds

Usage (repo root):
    python plot_training_curves.py --list cluster_runs/baseline          # print the tags in a run
    python plot_training_curves.py \\
        --run "Binary (baseline)=cluster_runs/baseline" \\
        --run "Park skin=cluster_runs/park_fixed" \\
        --out deliverables/1_training_curves/training_curves.png
Each --run is LABEL=FOLDER; every events.out.tfevents* file under FOLDER is read (a resumed run's files
are merged by epoch). Colours: the first run blue, the second red, the third orange. --mark adds a dot at a
chosen epoch of a run, e.g. --mark "Park skin=3600" for the checkpoint used later.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

COLORS = ["#1f77b4", "#d62728", "#ff7f0e"]
CTRL_DT = 0.1                                                     # s per control step (10 Hz)


def read_scalars(folder, tag):
    files = sorted(glob.glob(os.path.join(folder, "**", "events.out.tfevents*"), recursive=True))
    if not files:
        raise SystemExit("no TensorBoard event files under %s" % folder)
    steps, vals = {}, {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            continue
        for e in ea.Scalars(tag):
            steps[e.step] = e.value                               # later files overwrite repeated epochs
    if not steps:
        raise SystemExit("tag %s not found under %s (try --list)" % (tag, folder))
    x = np.array(sorted(steps))
    return x, np.array([steps[s] for s in x])


def ema(v, w):
    out, m = np.empty_like(v, dtype=float), v[0]
    for i, x in enumerate(v):
        m = w * m + (1 - w) * x
        out[i] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="LABEL=FOLDER (repeat)")
    ap.add_argument("--list", metavar="FOLDER", help="print the scalar tags found under FOLDER and exit")
    ap.add_argument("--reward-tag", default="rewards/iter")
    ap.add_argument("--length-tag", default="episode_lengths/iter")
    ap.add_argument("--smooth", type=float, default=0.95, help="EMA weight (TensorBoard's smoothing slider)")
    ap.add_argument("--max-epoch", type=int, default=5000)
    ap.add_argument("--mark", action="append", default=[], help="LABEL=EPOCH: dot on that run's curves")
    ap.add_argument("--title", default="Multi-object training (z-axis, 100 train objects, 1 seed)")
    ap.add_argument("--out", default="deliverables/1_training_curves/training_curves.png")
    args = ap.parse_args()

    if args.list:
        files = sorted(glob.glob(os.path.join(args.list, "**", "events.out.tfevents*"), recursive=True))
        tags = set()
        for f in files:
            ea = EventAccumulator(f, size_guidance={"scalars": 1})
            ea.Reload()
            tags |= set(ea.Tags()["scalars"])
        print("%d event files; scalar tags:\n  %s" % (len(files), "\n  ".join(sorted(tags))))
        return
    if not args.run:
        raise SystemExit("give at least one --run LABEL=FOLDER")
    marks = dict(m.split("=", 1) for m in args.mark)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5.0))
    panels = [(args.reward_tag, 1.0, "Cumulative reward (training batch)", "Reward"),
              (args.length_tag, CTRL_DT, "TTF [s]", "Time to fall (TTF)")]
    for k, spec in enumerate(args.run):
        label, folder = spec.split("=", 1)
        c = COLORS[k % len(COLORS)]
        for ax, (tag, scale, ylabel, title) in zip(axs, panels):
            x, v = read_scalars(folder, tag)
            keep = x <= args.max_epoch
            x, v = x[keep], v[keep] * scale
            s = ema(v, args.smooth)
            ax.plot(x, v, color=c, lw=0.6, alpha=0.18)
            ax.plot(x, s, color=c, lw=2.0, label=label)
            if label in marks:
                e = int(marks[label])
                i = int(np.argmin(np.abs(x - e)))
                ax.plot(x[i], s[i], "o", ms=8, mfc=c, mec="white", mew=2, zorder=5)
                ax.annotate("ep%d" % e, (x[i], s[i]), xytext=(0, 10), textcoords="offset points", ha="center",
                            fontsize=8, color="0.25")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            print("%-20s %-22s final (EMA) %.1f, max (EMA) %.1f at epoch %d"
                  % (label, tag, s[-1], s.max(), x[int(np.argmax(s))]))
    for ax in axs:
        ax.set_xlabel("Training epochs")
        ax.set_xlim(0, args.max_epoch)
        ax.set_xticks(np.arange(0, args.max_epoch + 1, 1000))
        ax.set_xticklabels(["0"] + ["%dK" % (t // 1000) for t in range(1000, args.max_epoch + 1, 1000)])
        ax.grid(color="0.9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_box_aspect(1)
    axs[1].legend(fontsize=9, frameon=False, loc="lower right")
    fig.suptitle(args.title, fontsize=11)
    fig.text(0.5, 0.01, "faint: per-epoch value; solid: moving average (weight %.2f); one seed per run"
             % args.smooth, ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()
