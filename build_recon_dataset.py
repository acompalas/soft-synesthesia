"""Cut shape-reconstruction rollouts into drop-free windows and compute the three touch signals.

Follows Yin et al. (Rotating without Seeing), Ablation Study II, and our project plan: the z-rotation
policy rotates each column object; we keep 200-step windows (20 s) in which the object is not dropped;
the input to the temporal CNN is the hand joint positions plus a touch signal at every step; the label
is the object's 64-radius outline. From the SAME windows we compute three touch signals, so only the
sensor model differs:
  none   : zeros (the paper's ablation)
  binary : 1 if the pad's net force (vector sum of its 4 cells) >= tau, as in the task
  park   : Park skin reading of the cells' compressive normal forces (park_skin.py, fixed p*)
The sensors are ideal by default (fixed tau and p*, no latency or dropout); --noisy applies the task's
per-window threshold / calibration draw, latency 0.2 and dropout 0.1 instead.

Input: one or more recon_rollout.npz files from +task.env.recon_dir (see the task file).
A window is kept if no reset occurs inside it (a reset is a drop or the 500-step episode end), so a
window never spans two episodes. Windows are non-overlapping, taken back to back within each episode.

Output (one .npz):
  dof_pos (W, L, 22) float32 hand + arm joint positions obj_pose (W, L, 7) object pose (diagnostics only)
  touch_binary (W, L, 16) uint8   touch_park (W, L, 16) float16   sensed (W, L, 16) float16 (policy's input)
  radii (W, 64) outline label in metres, scaled to the env's object; object (W,), split (W,), scale (W,)
  src (W, 3) = file index, env, start step

Usage:
  python build_recon_dataset.py rollouts/recon_train/recon_rollout.npz rollouts/recon_test/recon_rollout.npz \
      --out rollouts/recon_dataset.npz
"""
import argparse
import importlib.util
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(ROOT, "assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_split4.urdf")


def load_park():
    spec = importlib.util.spec_from_file_location("park_skin", os.path.join(ROOT, "isaacgymenvs/utils/park_skin.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def windows_of(reset, L):
    """reset: (T,) bool for one env. Returns start steps of back-to-back L-step windows with no reset inside."""
    T, starts, t = len(reset), [], 0
    seg_start = 0
    for t in range(T):
        if reset[t]:                                      # episode ends at t; next episode starts at t + 1
            seg_end = t                                   # last usable step is t - 1 (the reset step itself is excluded)
            starts += list(range(seg_start, seg_end - L + 1, L))
            seg_start = t + 1
    starts += list(range(seg_start, T - L + 1, L))        # episode still running at the end of the recording
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=200, help="window length in control steps (paper: 200 = 20 s)")
    ap.add_argument("--tau", type=float, default=1.5, help="binary threshold, N (task draws U(1, 2) per episode)")
    ap.add_argument("--park-E", type=float, default=338e3)
    ap.add_argument("--park-log-eps", type=float, default=0.003)
    ap.add_argument("--noisy", action="store_true", help="per-window tau / p* draw, latency 0.2, dropout 0.1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    L = args.length

    ps = load_park()
    out = {k: [] for k in ("dof_pos", "obj_pose", "touch_binary", "touch_park", "sensed", "radii", "object", "split",
                           "scale", "src")}
    skin = None
    for fi, path in enumerate(args.inputs):
        with np.load(path) as z:                          # read every array ONCE: indexing an NpzFile re-reads
            d = {k: z[k] for k in z.files}                # and decompresses the whole array on each access
        reset, T, N = d["reset"], d["reset"].shape[0], d["reset"].shape[1]
        if skin is None:
            pads = [str(p) for p in d["pad_names"]]
            cells = [p if k == 0 else f"{p}_c{k}" for p in pads for k in range(4)]
            skin = ps.ParkSkin(ps.cell_areas_from_urdf(URDF, cells), n_cells=4, E=args.park_E,
                               log_eps=args.park_log_eps if args.park_log_eps > 0 else None)
        n_kept = 0
        for e in range(N):
            for t0 in windows_of(reset[:, e], L):
                sl = slice(t0, t0 + L)
                fpad, fn = d["pad_force"][sl, e], d["cell_fn"][sl, e]           # (L, 16), (L, 64)
                if args.noisy:
                    tau = rng.uniform(1.0, 2.0, size=16)
                    scale = torch.as_tensor(np.tile(tau / 1.5, (L, 1)), dtype=torch.float32)
                else:
                    tau, scale = np.full(16, args.tau), None
                b = (fpad >= tau[None]).astype(np.float32)
                p = skin.pad_readings(torch.as_tensor(fn, dtype=torch.float32), scale).numpy()
                if args.noisy:                                                   # latency then dropout, as in the task
                    for x in (b, p):
                        for t in range(1, L):
                            lag = rng.random(16) < 0.2
                            x[t, lag] = x[t - 1, lag]
                        x[rng.random(x.shape) < 0.1] = 0.0
                out["dof_pos"].append(d["dof_pos"][sl, e]); out["obj_pose"].append(d["obj_pose"][sl, e])
                out["touch_binary"].append(b); out["touch_park"].append(p); out["sensed"].append(d["sensed"][sl, e])
                out["radii"].append(d["radii"][e]); out["object"].append(str(d["object"][e]))
                out["split"].append(str(d["split"][e])); out["scale"].append(float(d["obj_scale"][e]))
                out["src"].append((fi, e, t0)); n_kept += 1
        n_drop = int((reset & (d["progress"] < 499)).sum())
        print(f"{path}: {N} envs x {T} steps ({d['touch_model']} policy), {n_drop} drops; kept {n_kept} windows of {L}")

    if not out["object"]:
        print("no windows kept"); return
    res = {k: np.array(v) for k, v in out.items()}
    for k in ("dof_pos", "obj_pose", "radii"):
        res[k] = res[k].astype(np.float32)
    res["touch_binary"] = res["touch_binary"].astype(np.uint8)          # 0/1
    res["touch_park"] = res["touch_park"].astype(np.float16)            # readout in [0, 1]
    res["sensed"] = res["sensed"].astype(np.float16)
    np.savez_compressed(args.out, **res, length=L, tau=args.tau, park_E=args.park_E, park_log_eps=args.park_log_eps,
                        noisy=args.noisy, angle_convention="radius k at angle 2*pi*k/64 from +x towards +y, object frame")
    for sp in ("train", "test"):
        m = res["split"] == sp
        if not m.any():
            continue
        objs, counts = np.unique(res["object"][m], return_counts=True)
        tb, tp = res["touch_binary"][m], res["touch_park"][m]
        print(f"{sp}: {int(m.sum())} windows over {len(objs)} objects (per object min {counts.min()}, "
              f"median {int(np.median(counts))}, max {counts.max()}); binary on {tb.mean():.1%} of pad-steps, "
              f"park > 0 on {(tp > 0).mean():.1%}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
