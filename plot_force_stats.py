"""Typical pad forces in policy rollouts vs the working range of each touch sensor.

Input: force_stats.npz from the task's force-stats hook (touched pad-steps, 3 steps after a reset
excluded), e.g.
  python isaacgymenvs/train.py task=AllegroArmMOAR test=True checkpoint=<ckpt> headless=True \\
    task.env.objSet=columns_train task.env.axis=z task.env.sensor=split4 \\
    task.env.observationType=partial_stack task.env.ablation_mode=no-pc task.env.legacy_obs=True \\
    task.env.numEnvs=1024 +task.env.force_stats=30

Per sensor, a histogram of the pad force (log axis) with each sensor's range marked:
  binary touch: threshold tau ~ U(1, 2) N (shaded);
  Park skin of modulus E (p* = E for Park's channels): a single loaded cell closes at
  alpha_cell * E * A_cell (dotted), a uniformly loaded pad saturates at alpha * E * A_pad (solid).
Prints, per skin, the share of touches below the one-cell closure force and below the whole-pad
saturation force (a real contact lies between these two cases).

Outputs next to the input: force_stats_pads.png (4 x 4), force_stats_all.png.
Usage:
  python plot_force_stats.py force_stats.npz
  python plot_force_stats.py force_stats.npz --E 69e3 338e3
"""
import argparse
import importlib.util
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(ROOT, "assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_split4.urdf")
SKINS = {69e3: ("EcoFlex 00-30 (Park)", "tab:red"), 338e3: ("Dragon Skin 20", "tab:blue")}


def load_park():
    spec = importlib.util.spec_from_file_location("park_skin", os.path.join(ROOT, "isaacgymenvs/utils/park_skin.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--E", type=float, nargs="+", default=[69e3, 338e3], help="skin moduli to mark, Pa")
    args = ap.parse_args()
    d = np.load(args.npz)
    pads = [str(p) for p in d["pads"]]
    forces = [d[f"pad{i}"] for i in range(16)]
    ps = load_park()
    cells = [p if k == 0 else f"{p}_c{k}" for p in pads for k in range(4)]
    areas = np.array(ps.cell_areas_from_urdf(URDF, cells)).reshape(16, 4)

    def ranges(E):
        skin = ps.ParkSkin(areas.ravel(), n_cells=4, E=E)
        return skin.alpha_cell * E * areas.mean(1), skin.alpha * E * areas.sum(1)   # one cell closes, pad saturates

    allf = np.concatenate(forces)
    print(f"{int(d['num_envs'])} envs x {int(d['steps'])} steps; touched pad-steps {allf.size} "
          f"({float(d['touched_share']):.1%} of settled pad-steps); median {np.median(allf):.1f} N, "
          f"p90 {np.percentile(allf, 90):.1f} N, p99 {np.percentile(allf, 99):.1f} N")
    print(f"binary touch (tau 1-2 N): {np.mean(allf >= 1):.0%}-{np.mean(allf >= 2):.0%} of touches fire")
    for E in args.E:
        one, whole = ranges(E)
        below_one = np.mean(np.concatenate([f < one[i] for i, f in enumerate(forces)]))
        below_whole = np.mean(np.concatenate([f < whole[i] for i, f in enumerate(forces)]))
        name = SKINS.get(E, (f"E = {E / 1e3:.0f} kPa", None))[0]
        print(f"{name:22s} E {E / 1e3:5.0f} kPa: graded (not closed) for {below_one:.0%} of touches if each load "
              f"sits on one cell, {below_whole:.0%} if spread over the whole pad")

    bins = np.logspace(-1, 4, 60)

    def panel(ax, f, i, small):
        ax.hist(f, bins=bins, color="0.6", density=True)
        ax.axvspan(1, 2, color="tab:green", alpha=0.25, label=None if small else "binary threshold 1-2 N")
        for E in args.E:
            one, whole = ranges(E)
            name, c = SKINS.get(E, (f"E = {E / 1e3:.0f} kPa", None))
            a = one[i] if i is not None else np.median(one)
            b = whole[i] if i is not None else np.median(whole)
            ax.axvline(a, color=c, ls=":", lw=1.5, label=None if small else f"{name}: one cell closes")
            ax.axvline(b, color=c, ls="-", lw=1.5, label=None if small else f"{name}: whole pad saturates")
        ax.set_xscale("log"); ax.set_xlim(0.1, 1e4); ax.set_box_aspect(1); ax.set_yticks([])

    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axs.ravel()):
        panel(ax, forces[i], i, small=True)
        ax.set_title(f"{i}: {pads[i]} (median {np.median(forces[i]) if forces[i].size else 0:.1f} N)", fontsize=8)
        ax.tick_params(labelsize=7)
        if i >= 12: ax.set_xlabel("pad force (N)", fontsize=8)
    fig.suptitle("Pad force in rollouts (touched steps) vs sensor ranges: binary threshold (green), "
                 "one cell closes (dotted), whole pad saturates (solid)", fontsize=10)
    fig.tight_layout()
    out = os.path.dirname(os.path.abspath(args.npz))
    fig.savefig(os.path.join(out, "force_stats_pads.png"), dpi=110)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    panel(ax, allf, None, small=False)
    ax.set_xlabel("pad force (N)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"Pad force in rollouts, all sensors (median pad lines for the skins)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "force_stats_all.png"), dpi=130)
    print(f"saved force_stats_pads.png and force_stats_all.png in {out}")


if __name__ == "__main__":
    main()
