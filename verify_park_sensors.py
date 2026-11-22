"""Check the Park touch signal on a rollout recorded in the simulator with task.env.touch_model=park.

Record with latency and dropout off so the check is exact:
  +task.env.record_dir=rollouts/park_check task.env.latency=0 task.env.sensorNoise=0

[1] Recomputes every pad reading offline from the recorded cell normal forces and per-pad
    thresholds with park_skin.py, and compares it with what the policy received.  Must match.
[2] Saturation and reading statistics.
Charts (square), Park-style calibration plots from simulated contacts: one point per touched
pad-step, x = mean pad pressure (sum of cell normal forces / pad area), y = pad dR/R0, colored by
how many cells carry load; Eq. 4 at nominal p* in the background.
  park_sim_all.png   all 16 sensors in one chart
  park_sim_pads.png  4 x 4, one panel per sensor

Usage:
  python verify_park_sensors.py rollouts/park_check
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
TOUCH_N = 0.1      # pad counts as touched above this total normal force
CELL_N = 0.05      # cell counts as loaded above this normal force
COLORS = {1: "tab:red", 2: "tab:orange", 3: "tab:green", 4: "tab:blue"}


def load_module():
    spec = importlib.util.spec_from_file_location("park_skin", os.path.join(ROOT, "isaacgymenvs/utils/park_skin.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec_dir")
    ap.add_argument("--no-show", dest="show", action="store_false")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    d = np.load(os.path.join(args.rec_dir, "touch_rollout.npz"))
    assert str(d["touch_model"]) == "park", "this rollout was not recorded with touch_model=park"
    sensed, fn, thresh = d["sensed"], d["cell_fn"], d["thresh"]          # (T,16), (T,64), (T,16)
    cell_names = [str(n) for n in d["cell_names"]]
    pads = [str(n) for n in d["sensor_names"]]
    T = len(sensed)
    exact = float(d["latency"]) == 0.0 and float(d["sensor_noise"]) == 0.0
    print(f"object {d['object']}, {T} steps, alpha {float(d['park_alpha'])}, calibration scaling {bool(d['park_calib'])}, "
          f"latency {float(d['latency'])}, dropout {float(d['sensor_noise'])}")

    ps = load_module()
    areas = np.array(ps.cell_areas_from_urdf(URDF, cell_names))
    E = float(d["park_E"]) if "park_E" in d.files else 69e3
    eps = float(d["park_log_eps"]) if "park_log_eps" in d.files else 0.0
    skin = ps.ParkSkin(areas, n_cells=4, E=E, alpha=float(d["park_alpha"]), log_eps=eps if eps > 0 else None)
    print(f"skin E = p* = {skin.p_star / 1e3:.0f} kPa, readout {'linear' if eps <= 0 else f'log, eps {eps}'}")
    scale = None
    if bool(d["park_calib"]):
        th = torch.as_tensor(thresh, dtype=torch.float32)
        scale = torch.where(th > 0, th / 1.5, torch.ones_like(th))
    offline = skin.pad_readings(torch.as_tensor(fn, dtype=torch.float32), scale).numpy()

    err = np.abs(offline - sensed).max()
    print(f"\n[1] max |policy reading - offline Park model|: {err:.2e}", end="")
    if exact:
        print(f"  -> {'PASS' if err < 1e-5 else 'FAIL'} (latency and dropout off, must match)")
    else:
        print("  (latency/dropout on: mismatch expected; record with task.env.latency=0 task.env.sensorNoise=0)")

    fn4 = fn.reshape(T, 16, 4)
    a4 = areas.reshape(16, 4)
    pad_force = fn4.sum(-1)
    touched = pad_force > TOUCH_N
    loaded = (fn4 > CELL_N).sum(-1)
    p_mean = pad_force / a4.sum(-1)[None] / 1e3                         # kPa
    cell_p = fn4 / a4[None] / 1e3
    dr = skin.dR_from_reading(sensed)                                   # pad dR/R0
    print(f"[2] touched pad-steps: {int(touched.sum())} of {touched.size}; readings when touched: "
          f"median {np.median(sensed[touched]) if touched.any() else float('nan'):.3f}, "
          f"max {sensed.max():.3f} (1.0 = cap); touched pad-steps at the cap: {(sensed[touched] >= 0.999).mean():.1%}")

    PMAX = 80 * skin.p_star / 69e3                                        # same range relative to p*
    p_fine = np.linspace(0, PMAX, 400)
    curve = 1 / (1 - np.minimum(p_fine * 1e3 / skin.p_star, skin.alpha)) - 1

    def panel(ax, idx_mask, title, small):
        ax.plot(p_fine, curve, color="0.75", lw=4, zorder=1, label=None if small else "Eq. 4 (nominal p*)")
        for k in (1, 2, 3, 4):
            m = idx_mask & (loaded == k)
            if m.any():
                ax.scatter(np.minimum(p_mean[m], PMAX * 0.995), dr[m], s=6 if small else 8, alpha=0.6, color=COLORS[k],
                           zorder=3, label=None if small else f"{k} loaded cell{'s' if k > 1 else ''}")
        if abs(skin.p_star - 69e3) < 1:                                   # Park's measured range (EcoFlex 00-30)
            for kpa, c in ((15, "g"), (50, "orange")):
                ax.axvline(kpa, color=c, lw=0.8, ls="--")
        ax.set_box_aspect(1); ax.set_xlim(0, PMAX); ax.set_ylim(0, skin.g_max * 1.08)
        ax.set_title(title, fontsize=8 if small else 10)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    panel(ax, touched, f"Simulated contacts, all sensors (object {d['object']})", small=False)
    ax.set_xlabel(f"mean pad pressure (kPa), clipped at {PMAX:.0f}"); ax.set_ylabel("pad dR/R0")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(args.rec_dir, "park_sim_all.png"), dpi=130)

    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axs.ravel()):
        m = np.zeros_like(touched); m[:, i] = touched[:, i]
        panel(ax, m, f"{i}: {pads[i]} ({int(touched[:, i].sum())} pts)", small=True)
        ax.tick_params(labelsize=7)
        if i % 4 == 0: ax.set_ylabel("pad dR/R0", fontsize=8)
        if i >= 12: ax.set_xlabel("mean pad pressure (kPa)", fontsize=8)
    fig.suptitle("Simulated contacts per sensor (red 1, orange 2, green 3, blue 4 loaded cells)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(args.rec_dir, "park_sim_pads.png"), dpi=110)
    print(f"\nsaved park_sim_all.png and park_sim_pads.png in {args.rec_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
