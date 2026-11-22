"""Standalone test of the Park skin model (isaacgymenvs/utils/park_skin.py). No Isaac Gym needed.

Each of the 16 sensors (pads) is pressed uniformly at sampled pressures (0 to 80 kPa for Park's
EcoFlex skin, scaled with E): the force on each of its 4 cells is pressure x that cell's area. The
sensor's reading dR/R0 is plotted against the pressure, with Park's Eq. 4 drawn in the background.

  park_sensors.png  4 x 4 grid, one square panel per sensor
  park_model.png    one square chart: all 16 sensors' readings on top of the Eq. 4 curve

Checks (printed, PASS/FAIL):
  [cells]   every cell pressed alone follows Eq. 4 with its own area, and leaves the other cells at 0;
  [sensors] every uniformly pressed sensor follows Eq. 4;
  [area]    the same force on one cell never reads lower than on the whole pad, and a closed cell
            saturates the pad (cells in series, one channel per pad).

Usage (from the repo root):
  python test_park_skin.py                 # Dragon Skin 20 (the skin used for training)
  python test_park_skin.py --E 69e3        # Park's EcoFlex 00-30
"""
import argparse
import importlib.util
import os

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(ROOT, "assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_split4.urdf")
PADS = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr", "link_5.0_fsr",
        "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr", "link_10.0_fsr",
        "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
        "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]   # task order
P_KPA = np.arange(0, 80.1, 5.0)


def load_module():
    # import park_skin.py directly so this test does not import isaacgymenvs (and Isaac Gym)
    spec = importlib.util.spec_from_file_location("park_skin", os.path.join(ROOT, "isaacgymenvs/utils/park_skin.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eq4(skin, p_kpa, alpha=None):
    x = np.minimum(np.asarray(p_kpa) * 1e3 / skin.p_star, skin.alpha if alpha is None else alpha)
    return 1.0 / (1.0 - x) - 1.0


def decorate(ax, skin, small=False):
    ax.set_box_aspect(1)
    lines = [(skin.alpha * skin.p_star / 1e3, "r", "cap")]
    if abs(skin.p_star - 69e3) < 1:                                       # Park's measured range (EcoFlex 00-30)
        lines = [(15, "g", "15 kPa (min. detectable)"), (50, "orange", "50 kPa (max. tested)")] + lines
    for kpa, c, lab in lines:
        ax.axvline(kpa, color=c, lw=0.8, ls="--", label=None if small else lab)
    ax.set_xlim(0, P_KPA[-1]); ax.set_ylim(0, skin.g_max * 1.08)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--E", type=float, default=338e3,
                    help="elastomer modulus, Pa (default Dragon Skin 20, 338e3; Park's EcoFlex 00-30: 69e3)")
    ap.add_argument("--no-show", dest="show", action="store_false")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    ps = load_module()
    cells = [p if k == 0 else f"{p}_c{k}" for p in PADS for k in range(4)]
    areas = np.array(ps.cell_areas_from_urdf(URDF, cells))
    skin = ps.ParkSkin(areas, n_cells=4, E=args.E, alpha=args.alpha)
    global P_KPA
    P_KPA = P_KPA * skin.p_star / 69e3                                   # same range relative to p*
    print(f"p_star = {skin.p_star / 1e3:.2f} kPa, cap = {skin.alpha * skin.p_star / 1e3:.1f} kPa (alpha {skin.alpha}), "
          f"max reading dR/R0 = {skin.g_max:.2f}")
    n = len(P_KPA)

    # checks: each cell alone, then each sensor pressed uniformly
    cell_err = 0.0
    for c in range(64):
        f = torch.zeros(n, 64)
        f[:, c] = torch.as_tensor(P_KPA * 1e3 * areas[c], dtype=torch.float32)
        g = skin.cell_response(f).numpy()
        cell_err = max(cell_err, np.abs(g[:, c] - eq4(skin, P_KPA, skin.alpha_cell)).max(),
                       np.abs(np.delete(g, c, axis=1)).max())
    readings = np.zeros((16, n))
    for i in range(16):
        f = torch.zeros(n, 64)
        for k in range(4):
            f[:, 4 * i + k] = torch.as_tensor(P_KPA * 1e3 * areas[4 * i + k], dtype=torch.float32)
        readings[i] = skin.pad_readings(f)[:, i].numpy() * skin.g_max        # dR/R0
    pad_err = np.abs(readings - eq4(skin, P_KPA)[None]).max()
    print(f"[cells]   every cell alone vs Eq. 4 (other cells stay 0): max error {cell_err:.1e} -> {'PASS' if cell_err < 1e-4 else 'FAIL'}")
    print(f"[sensors] every sensor pressed uniformly vs Eq. 4:        max error {pad_err:.1e} -> {'PASS' if pad_err < 1e-4 else 'FAIL'}")
    # same force on one cell vs on the whole pad: one cell must never read lower, and a closed cell saturates
    worst, sat_ok = np.inf, True
    for i in range(16):
        A = areas[4 * i:4 * i + 4]
        for F in np.linspace(0.05, 40, 400):
            whole = torch.zeros(1, 64); whole[0, 4 * i:4 * i + 4] = torch.as_tensor(F * A / A.sum(), dtype=torch.float32)
            r_whole = skin.pad_readings(whole)[0, i].item()
            for k in range(4):
                one = torch.zeros(1, 64); one[0, 4 * i + k] = F
                r_one = skin.pad_readings(one)[0, i].item()
                worst = min(worst, r_one - r_whole)
                if F / A[k] >= skin.p_star * skin.alpha_cell and r_one < 1.0 - 1e-6:
                    sat_ok = False
    print(f"[area]    same force on one cell vs whole pad: min(one - whole) {worst:+.1e} -> {'PASS' if worst > -1e-6 else 'FAIL'}; "
          f"closed cell saturates the pad -> {'PASS' if sat_ok else 'FAIL'}")

    p_fine = np.linspace(0, P_KPA[-1], 400)

    # 4 x 4 grid of square panels in a square figure
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axs.ravel()):
        ax.plot(p_fine, eq4(skin, p_fine), color="0.75", lw=4, zorder=1)
        ax.scatter(P_KPA, readings[i], s=14, color="tab:blue", zorder=3)
        decorate(ax, skin, small=True)
        ax.set_title(f"{i}: {PADS[i]}", fontsize=8)
        ax.tick_params(labelsize=7)
        if i % 4 == 0:
            ax.set_ylabel("dR/R0", fontsize=8)
        if i >= 12:
            ax.set_xlabel("pressure (kPa)", fontsize=8)
    fig.suptitle(f"Sensor readings (points) vs Park Eq. 4 (grey), p* = {skin.p_star / 1e3:.0f} kPa", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "park_sensors.png"), dpi=110)

    # one square chart, all sensors
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(p_fine, eq4(skin, p_fine), color="0.75", lw=5, zorder=1, label="Park Eq. 4")
    for i in range(16):
        ax.scatter(P_KPA, readings[i], s=10, zorder=3, label="16 sensors" if i == 0 else None, color="tab:blue")
    decorate(ax, skin)
    ax.set_xlabel("pressure (kPa)"); ax.set_ylabel("dR/R0")
    ax.set_title(f"All 16 sensors vs Park model (p* = {skin.p_star / 1e3:.0f} kPa)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "park_model.png"), dpi=130)

    print(f"saved {os.path.join(ROOT, 'park_sensors.png')} and {os.path.join(ROOT, 'park_model.png')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
