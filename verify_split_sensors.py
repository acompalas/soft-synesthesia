"""Check the split-pad sensor logic from a recorded rollout (touch_rollout.npz from the task recorder,
run with task.env.sensor=split4). Prints numerical checks and saves verify_split_sensors.png.

Checks
  1. Pad force used by the policy pipeline == |vector sum of its 4 cell forces|   (summing is correct)
  2. Binary signal == 1[pad force >= threshold]  (exact when latency=0 and sensorNoise=0)
  3. Thresholds lie in [1, 2) N and are constant within an episode
  4. Per-pad table: how often each pad is on, force when on, normal share of the force,
     and how many of its 4 cells carry load when the pad is touched (what Park will see)

Usage:
  python verify_split_sensors.py rollouts/split_check
"""
import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

TOUCH_N = 0.1   # a pad counts as "touched" above this force, N
CELL_N = 0.05   # a cell counts as "loaded" above this compressive normal force, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec_dir")
    ap.add_argument("--no-show", dest="show", action="store_false")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    d = np.load(os.path.join(args.rec_dir, "touch_rollout.npz"))
    C = int(d["n_cells"])
    sensed, force, thresh = d["sensed"], d["force"], d["thresh"]          # (T,16) each
    cell_force, cell_fn = d["cell_force"], d["cell_fn"]                   # (T,16C,3), (T,16C)
    reset = d["reset"]
    names = [str(n) for n in d["sensor_names"]]
    T, P = force.shape
    cf = cell_force.reshape(T, P, C, 3)
    fn = cell_fn.reshape(T, P, C)
    print(f"object {d['object']}, {T} steps, {P} pads x {C} cells, latency={float(d['latency'])}, "
          f"sensorNoise={float(d['sensor_noise'])}, resets={int(reset.sum())}")
    if C == 1:
        print("NOTE: n_cells == 1, this rollout was recorded with the unsplit robot")

    # 1. pad force == |sum of cell vectors|
    from_cells = np.linalg.norm(cf.sum(axis=2), axis=-1)
    err = np.abs(from_cells - force).max()
    rel = (np.abs(from_cells - force) / np.maximum(force, 1.0)).max()  # float32 rounding is ~1e-6 relative
    print(f"\n[1] max |pad force - |sum of cell forces||: {err:.2e} N (relative {rel:.1e})  -> "
          f"{'PASS' if rel < 1e-4 else 'FAIL'}")

    # 2. binary logic
    raw = (force >= thresh).astype(float)
    agree = (raw == sensed).mean()
    exact = float(d["latency"]) == 0.0 and float(d["sensor_noise"]) == 0.0
    print(f"[2] binary == 1[pad force >= threshold]: agreement {100 * agree:.2f}% of pad-steps "
          f"({int((raw != sensed).sum())} mismatches)", end="")
    if exact:
        print(f" -> {'PASS' if agree == 1.0 else 'FAIL'} (noise off, must be exact)")
    else:
        print(" (latency/dropout on, so some mismatch is expected; rerun with latency=0 sensorNoise=0 to check exactly)")

    # 3. thresholds
    ep = np.concatenate([[0], np.cumsum(reset[:-1])])                     # episode id per step
    const = all(np.allclose(thresh[ep == e], thresh[ep == e][0]) for e in np.unique(ep))
    print(f"[3] thresholds in [{thresh.min():.3f}, {thresh.max():.3f}] N, constant within each episode: "
          f"{const} -> {'PASS' if const and thresh.min() >= 1.0 and thresh.max() < 2.0 else 'CHECK'}")

    # 4. per-pad table
    fmag = np.linalg.norm(cf, axis=-1)                                     # (T,P,C) cell force magnitude
    print(f"\n[4] per pad (touched: pad force > {TOUCH_N} N; loaded cell: normal force > {CELL_N} N)")
    print(f"  {'pad':18s} {'on %':>6s} {'F when on':>10s} {'normal share':>12s}   loaded cells when touched (1/2/3/4)")
    loaded_all = []
    for p in range(P):
        on = sensed[:, p] > 0
        touched = force[:, p] > TOUCH_N
        f_on = force[on, p].mean() if on.any() else float("nan")
        share = fn[:, p].sum() / max(fmag[:, p].sum(), 1e-9)
        n_loaded = (fn[touched, p] > CELL_N).sum(axis=1)
        loaded_all.append(n_loaded)
        hist = [int((n_loaded == k).sum()) for k in range(1, C + 1)]
        print(f"  {names[p]:18s} {100 * on.mean():6.1f} {f_on:10.2f} {share:12.2f}   {hist}")
    loaded_all = np.concatenate(loaded_all)

    # figure
    act = np.argsort(-(sensed > 0).sum(axis=0))[:3]
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1], hspace=0.45, wspace=0.3)
    t = np.arange(T)
    for row, p in enumerate(act[:2]):
        ax = fig.add_subplot(gs[row, :])
        on = sensed[:, p] > 0
        ax.fill_between(t, 0, 1, where=on, transform=ax.get_xaxis_transform(), color="#5b5bff", alpha=0.15,
                        step="mid", label="binary on")
        for k in range(C):
            ax.plot(t, fn[:, p, k], lw=0.9, label=f"cell {k} normal")
        ax.plot(t, force[:, p], color="k", lw=1.4, label="pad |sum of cells|")
        ax.step(t, thresh[:, p], color="r", ls="--", lw=1, where="mid", label="threshold")
        for r in np.where(reset)[0]:
            ax.axvline(r + 0.5, color="0.5", ls=":", lw=0.8)
        ax.set_xlim(0, T); ax.set_ylim(bottom=0)
        ax.set_title(f"pad {p}: {names[p]}", fontsize=9); ax.set_ylabel("force (N)")
        if row == 0:
            ax.legend(fontsize=7, ncol=7, loc="upper right")
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(range(1, C + 1), [(loaded_all == k).sum() for k in range(1, C + 1)], color="0.4")
    ax.set_xlabel("loaded cells when pad touched"); ax.set_ylabel("pad-steps")
    ax = fig.add_subplot(gs[2, 1])
    touched = force > TOUCH_N
    ax.scatter(force[touched], fn.sum(axis=2)[touched], s=4, alpha=0.4)
    m = force[touched].max() if touched.any() else 1.0
    ax.plot([0, m], [0, m], "k--", lw=0.8)
    ax.set_xlabel("pad force |sum of cells| (N)"); ax.set_ylabel("sum of cell normal forces (N)")
    ax = fig.add_subplot(gs[2, 2])
    ax.hist(force[touched], bins=40, color="0.4")
    ax.axvspan(1.0, 2.0, color="r", alpha=0.1, label="threshold range")
    ax.set_xlabel("pad force when touched (N)"); ax.legend(fontsize=7)
    fig.suptitle(f"Split-pad sensor check, object {d['object']}", fontsize=11)
    out = os.path.join(args.rec_dir, "verify_split_sensors.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {os.path.abspath(out)}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
