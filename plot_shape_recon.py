"""Figures for the shape-reconstruction results written by shape_recon.py.

Reads <results>/predictions_<signal>.npz and results.json, plus the true polygons from columns.json,
and writes three figures to --out (default results/shape_recon/):

  shape_recon_outlines.png  held-out objects (best, median and worst by error): the true cross-section
                            and the reconstruction from each touch signal, overlaid in the object frame
  shape_recon_3d.png        the same objects extruded to 3D: true shape and each signal's reconstruction
  shape_recon_error.png     held-out RMSE per signal with a 95% interval (bootstrap over the held-out
                            objects), against always guessing the mean training outline

How a reconstruction is drawn: each window's prediction is rotated by the best cyclic shift found by
the loss (so it is shown in the object's own frame), divided by that window's object scale (0.95-1.1),
and the predictions of all windows of the object are averaged. The shaded band is the 10-90th
percentile over windows for the binary signal.

Usage (from the repo root):
    python plot_shape_recon.py
    python plot_shape_recon.py --results rollouts/shape_recon_results --objects col_012 col_092
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.dirname(os.path.abspath(__file__))
COLUMNS = os.path.join(ROOT, "assets/urdf/objects/meshes/columns.json")
STYLE = {"true": ("black", "True shape"),
         "none": ("#7f7f7f", "No touch"),
         "binary": ("#1f77b4", "Binary touch"),
         "park": ("#d62728", "Park skin")}


def load(results_dir, signals):
    reg = json.load(open(COLUMNS))
    labels = {n: np.asarray(r) for n, r in zip(reg["names"], reg["outline_radii_m"])}
    polys = {n: np.asarray(p) for n, p in zip(reg["names"], reg["polygon_xy_m"])}
    summary = {r["signal"]: r for r in json.load(open(os.path.join(results_dir, "results.json")))["results"]}
    P = {}
    for s in signals:
        d = np.load(os.path.join(results_dir, "predictions_%s.npz" % s))
        pred, true, shift, obj = d["pred_m"], d["true_m"], d["shift"].astype(int), d["object"].astype(str)
        R = pred.shape[1]
        # rotate each prediction into the object frame: pred[k] ~ true[k + shift]  ->  np.roll(pred, shift)
        idx = (np.arange(R)[None, :] - shift[:, None]) % R
        aligned = pred[np.arange(len(pred))[:, None], idx]
        scale = true.mean(1) / np.array([labels[o].mean() for o in obj])
        P[s] = {"aligned": aligned / scale[:, None], "obj": obj, "mse": d["mse_mm2"]}
    return reg, labels, polys, summary, P


def per_object_rmse(P, s, objs):
    return np.array([np.sqrt(P[s]["mse"][P[s]["obj"] == o].mean()) for o in objs])


def outline_xy(r):
    th = 2 * np.pi * np.arange(len(r) + 1) / len(r)
    r = np.append(r, r[0])
    return r * np.cos(th), r * np.sin(th)


def pick_objects(P, signals, objs, n):
    err = np.mean([per_object_rmse(P, s, objs) for s in signals], axis=0)
    order = objs[np.argsort(err)]
    k = n // 3
    picks = list(order[:k]) + list(order[len(order) // 2 - k // 2: len(order) // 2 - k // 2 + k]) + list(order[-(n - 2 * k):])
    return list(dict.fromkeys(picks))  # keep order, drop repeats


def fig_outlines(picks, polys, P, signals, out):
    cols = 3
    rows = int(np.ceil(len(picks) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, o in zip(np.ravel(axs), picks):
        poly = np.vstack([polys[o], polys[o][:1]]) * 1e3
        ax.fill(poly[:, 0], poly[:, 1], color="0.88", zorder=0)
        ax.plot(poly[:, 0], poly[:, 1], color=STYLE["true"][0], lw=1.6, label=STYLE["true"][1], zorder=3)
        if "binary" in P:
            a = P["binary"]["aligned"][P["binary"]["obj"] == o] * 1e3
            lo, hi = np.percentile(a, 10, axis=0), np.percentile(a, 90, axis=0)
            th = 2 * np.pi * np.arange(a.shape[1] + 1) / a.shape[1]
            lo, hi = np.append(lo, lo[0]), np.append(hi, hi[0])
            ax.fill(np.r_[hi * np.cos(th), (lo * np.cos(th))[::-1]], np.r_[hi * np.sin(th), (lo * np.sin(th))[::-1]],
                    color=STYLE["binary"][0], alpha=0.15, lw=0, zorder=1)
        errs = []
        for s in signals:
            m = P[s]["obj"] == o
            x, y = outline_xy(P[s]["aligned"][m].mean(0) * 1e3)
            ax.plot(x, y, color=STYLE[s][0], lw=1.3, label=STYLE[s][1], zorder=4)
            errs.append("%s %.2f" % (s, np.sqrt(P[s]["mse"][m].mean())))
        ax.set_aspect("equal")
        ax.set_xlim(-42, 42); ax.set_ylim(-42, 42)
        ax.axis("on"); ax.set_xticks([-30, 0, 30]); ax.set_yticks([-30, 0, 30]); ax.tick_params(labelsize=7)
        ax.set_title("%s\nRMSE mm: %s" % (o, ", ".join(errs)), fontsize=8)
    handles, names = np.ravel(axs)[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=8, frameon=False)
    fig.suptitle("Held-out objects: true cross-section and mean reconstruction (mm, object frame)\n"
                 "shaded: 10-90% of binary-touch reconstructions over windows", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def extrude(ax, xy, height, color):
    xy = np.vstack([xy, xy[:1]])
    sides = [[(*xy[i], 0), (*xy[i + 1], 0), (*xy[i + 1], height), (*xy[i], height)] for i in range(len(xy) - 1)]
    top = [[(x, y, height) for x, y in xy[:-1]]]
    # Sides (4 vertices each) and top (N vertices) go in separate collections, each as one array:
    # older matplotlib (python 3.8 env) cannot compute shading normals for a ragged list of polygons.
    for faces in (np.array(sides, dtype=float), np.array(top, dtype=float)):
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=color, linewidths=0.2,
                                             alpha=0.85, shade=True))


def fig_3d(picks, polys, P, signals, height_mm, out):
    cols = ["true"] + list(signals)
    fig = plt.figure(figsize=(2.3 * len(cols), 2.3 * len(picks)))
    for i, o in enumerate(picks):
        for j, s in enumerate(cols):
            ax = fig.add_subplot(len(picks), len(cols), i * len(cols) + j + 1, projection="3d")
            if s == "true":
                xy = polys[o] * 1e3
            else:
                x, y = outline_xy(P[s]["aligned"][P[s]["obj"] == o].mean(0) * 1e3)
                xy = np.column_stack([x[:-1], y[:-1]])
            extrude(ax, xy, height_mm, STYLE[s][0] if s != "true" else "0.45")
            ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_zlim(0, 80)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=28, azim=-60)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(STYLE[s][1], fontsize=9)
            if j == 0:
                ax.text2D(-0.05, 0.5, o, transform=ax.transAxes, rotation=90, va="center", fontsize=8)
    fig.suptitle("Held-out objects extruded to %d mm: true shape and mean reconstruction" % height_mm, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_error(P, signals, objs, summary, out, n_boot=20000, seed=0):
    # RMSE over all held-out windows, as shape_recon.py prints it; the interval resamples whole objects
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(objs), (n_boot, len(objs)))
    means, cis = [], []
    for s in signals:
        sums = np.array([P[s]["mse"][P[s]["obj"] == o].sum() for o in objs])
        counts = np.array([(P[s]["obj"] == o).sum() for o in objs])
        means.append(np.sqrt(sums.sum() / counts.sum()))
        cis.append(np.percentile(np.sqrt(sums[idx].sum(1) / counts[idx].sum(1)), [2.5, 97.5]))
    means, cis = np.array(means), np.array(cis)
    ref = np.sqrt(summary[signals[0]]["mean_outline_mse_mm2"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    x = np.arange(len(signals))
    ax.bar(x, means, color=[STYLE[s][0] for s in signals], width=0.6)
    ax.errorbar(x, means, yerr=[means - cis[:, 0], cis[:, 1] - means], fmt="none", ecolor="black", capsize=4, lw=1)
    ax.axhline(ref, color="black", ls="--", lw=1)
    ax.text(-0.4, ref, "guess the mean outline (%.2f mm)" % ref, ha="left", va="bottom", fontsize=8)
    for xi, m in zip(x, means):
        ax.text(xi, m / 2, "%.2f" % m, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([STYLE[s][1] for s in signals])
    ax.set_ylabel("Held-out outline RMSE (mm)")
    ax.set_ylim(0, ref * 1.15)
    ax.set_title("Shape reconstruction on %d held-out objects\n(error bars: 95%% interval, bootstrap over objects)"
                 % len(objs), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return means, cis, ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="rollouts/shape_recon_results", help="output folder of shape_recon.py")
    ap.add_argument("--out", default="results/shape_recon")
    ap.add_argument("--signals", nargs="+", default=["none", "binary", "park"])
    ap.add_argument("--objects", nargs="*", help="held-out objects to draw (default: best, median and worst 2 each)")
    ap.add_argument("--n-objects", type=int, default=6)
    args = ap.parse_args()

    reg, labels, polys, summary, P = load(args.results, args.signals)
    objs = np.unique(P[args.signals[0]]["obj"])
    picks = args.objects or pick_objects(P, args.signals, objs, args.n_objects)
    os.makedirs(args.out, exist_ok=True)

    fig_outlines(picks, polys, P, args.signals, os.path.join(args.out, "shape_recon_outlines.png"))
    fig_3d(picks, polys, P, args.signals, reg["height_m"] * 1e3, os.path.join(args.out, "shape_recon_3d.png"))
    means, cis, ref = fig_error(P, args.signals, objs, summary, os.path.join(args.out, "shape_recon_error.png"))

    print("Objects drawn: %s" % ", ".join(picks))
    for s, m, (lo, hi) in zip(args.signals, means, cis):
        print("  %-7s RMSE %.2f mm  (95%% interval %.2f-%.2f)" % (s, m, lo, hi))
    print("  mean outline guess %.2f mm" % ref)
    print("Saved to %s/" % args.out)


if __name__ == "__main__":
    main()
