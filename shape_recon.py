"""Shape reconstruction from touch: temporal CNN + rotation-aligned loss.

Predicts the outline of a column-shaped object (64 radii, see make_column_objects.py) from one
200-step window of the hand rotating it, and compares touch signals computed from the SAME windows:

    none    hand joint positions only
    binary  hand joint positions + 16 on/off pad signals (pad force >= tau)
    park    hand joint positions + 16 Park-skin readings (park_skin.py, Dragon Skin 20, log readout)
    sensed  hand joint positions + the touch signal the policy itself received (optional)

Input: the dataset shards written by the repo's build_recon_dataset.py (collect_recon.sh puts them in
rollouts/recon_shards/). The touch signals, labels and train/test split are all in the shards, so the
sensor models live in one place (the repo) and this script only trains and evaluates.

Usage (from the repo root; results go to rollouts/shape_recon_results/, which git ignores):
    python shape_recon.py --data rollouts/recon_shards
    python shape_recon.py --data shard1.npz shard2.npz --signals none binary park sensed --epochs 60
    python shape_recon.py --data rollouts/fake_shards        # test with shape_fake_rollouts.py output

Fields read from each shard (W windows of L steps):
    dof_pos       (W, L, 22)  arm + hand joint positions [rad]; joints 6-21 are the hand
    touch_binary  (W, L, 16)  0/1
    touch_park    (W, L, 16)  Park readout in [0, 1]
    sensed        (W, L, 16)  the policy's own touch input (only needed for --signals sensed)
    radii         (W, 64)     outline label [m], already scaled to that environment's object
    object, split (W,)        object name and "train"/"test"

Works with Python 3.8+, PyTorch and numpy. Runs on CPU; uses a GPU if one is available.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HAND = slice(6, 22)          # dof_pos columns 0-5 are the xArm, 6-21 the Allegro hand
TOUCH_KEY = {"binary": "touch_binary", "park": "touch_park", "sensed": "sensed"}
JOINT_STD_FLOOR = 1e-3       # rad; a joint that barely moves in training is not blown up at test time


# --------------------------------------------------------------------------- data
def shard_paths(items):
    paths = []
    for it in items:
        paths += sorted(glob.glob(os.path.join(it, "*.npz"))) if os.path.isdir(it) else [it]
    if not paths:
        raise SystemExit("No .npz shards found in %s" % items)
    return paths


def load_shards(paths, signals):
    """Concatenate the fields this run needs from every shard."""
    keys = ["dof_pos", "radii", "object", "split", "scale"] + [TOUCH_KEY[s] for s in signals if s != "none"]
    parts = {k: [] for k in keys}
    for p in paths:
        with np.load(p) as z:
            missing = [k for k in keys if k not in z.files]
            if missing:
                raise SystemExit("%s has no %s" % (p, missing))
            for k in keys:
                v = z[k]
                if k == "dof_pos":
                    v = v[:, :, HAND].astype(np.float32)           # keep only the 16 hand joints
                parts[k].append(v)                                 # touch stays uint8 / float16 (memory)
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["object"] = data["object"].astype(str)
    data["split"] = data["split"].astype(str)
    return data


def build_inputs(data, signal, fit_idx):
    """(W, C, L) float16: 16 hand joints normalised with the statistics of the fit windows, plus 16 touch
    channels unless signal == 'none'. Written in place into one float16 array (no float32 copies of the whole
    dataset), so the full 57k-window set needs about 0.7 GB here. Batches are converted to float32 on the fly.
    Touch channels are already in [0, 1] for every signal, so they are left as they are: the same scale for
    binary and Park, and a pad that is rarely touched in training cannot turn into a huge input at test time."""
    dof = data["dof_pos"]                                              # (W, L, 16) float32
    mean = dof[fit_idx].mean(axis=(0, 1))
    std = np.maximum(dof[fit_idx].std(axis=(0, 1)), JOINT_STD_FLOOR)
    W, L = dof.shape[:2]
    C = 16 if signal == "none" else 32
    x = np.empty((W, C, L), np.float16)
    for i in range(0, W, 4096):                                        # chunked: no full-size temporaries
        x[i:i + 4096, :16] = np.transpose((dof[i:i + 4096] - mean) / std, (0, 2, 1))
        if signal != "none":
            x[i:i + 4096, 16:] = np.transpose(data[TOUCH_KEY[signal]][i:i + 4096], (0, 2, 1))
    return x, mean, std


# --------------------------------------------------------------------------- model and loss
class TemporalCNN(nn.Module):
    """1D convolutions over time steps, then pooling over time, then an MLP to the outline.

    Pooling over the whole rollout makes the output independent of when in the rollout a contact
    happened, and lets the same network accept rollouts of any length.
    """

    def __init__(self, in_channels, n_radii=64, width=64, dropout=0.1):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(nn.Conv1d(cin, cout, kernel_size=5, stride=stride, padding=2),
                                 nn.BatchNorm1d(cout), nn.ReLU())

        self.conv = nn.Sequential(block(in_channels, width, 1),
                                  block(width, width, 2),
                                  block(width, 2 * width, 2),
                                  block(2 * width, 2 * width, 2))
        self.head = nn.Sequential(nn.Linear(4 * width, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, n_radii))

    def forward(self, x):                                   # x: (B, C, T)
        f = self.conv(x)                                    # (B, 2*width, T/8)
        f = torch.cat([f.mean(-1), f.amax(-1)], dim=1)      # average and max over time
        return self.head(f)                                 # (B, n_radii)


def aligned_mse(pred, target):
    """Rotation-aligned mean squared error.

    The outline is 64 radii at evenly spaced angles, so rotating the object about z by one angular
    step is a circular shift of the vector. For each sample this returns the MSE at the shift of the
    target that fits the prediction best, and that shift.

    pred, target: (B, R).  Returns (loss per sample (B,), best shift per sample (B,)).
    """
    R = target.shape[1]
    idx = (torch.arange(R, device=target.device)[None, :] + torch.arange(R, device=target.device)[:, None]) % R
    shifted = target[:, idx]                                # (B, R shifts, R): shifted[b, s, k] = target[b, (k+s) % R]
    err = ((pred[:, None, :] - shifted) ** 2).mean(-1)      # (B, R shifts)
    loss, shift = err.min(dim=1)
    return loss, shift


# --------------------------------------------------------------------------- training
def run_signal(signal, data, args, device):
    """Train on the train objects minus a few validation objects; report the held-out (test) objects once, at
    the end. The validation error is printed during training and is what any tuning should look at."""
    split, objects = data["split"], data["object"]
    train_objs = sorted(set(objects[split == "train"]))
    rng = np.random.default_rng(args.seed)
    val_objs = set(rng.choice(train_objs, size=min(args.val_objects, len(train_objs) - 1), replace=False)) \
        if args.val_objects > 0 else set()
    is_val = np.isin(objects, sorted(val_objs))
    fit_idx = np.where((split == "train") & ~is_val)[0]
    val_idx = np.where(is_val)[0]
    te_idx = np.where(split == "test")[0]
    assert len(fit_idx) and len(te_idx), "need windows from both the train and the test objects"
    y_m = data["radii"].astype(np.float32)                 # outline of each env's (scaled) object [m]

    x, joint_mean, joint_std = build_inputs(data, signal, fit_idx)
    unit = float(y_m[fit_idx].mean())                      # train in units of the mean radius
    y = torch.from_numpy(y_m / unit)

    torch.manual_seed(args.seed)
    model = TemporalCNN(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    gen = torch.Generator().manual_seed(args.seed)

    def batch_x(idx):
        return torch.from_numpy(x[idx]).to(device).float()

    def predict(idx):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(batch_x(idx[i:i + 2048])) for i in range(0, len(idx), 2048)])

    def mean_loss(idx):
        return aligned_mse(predict(idx), y[idx].to(device))[0].mean().item() if len(idx) else float("nan")

    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = fit_idx[torch.randperm(len(fit_idx), generator=gen).numpy()]
        total = 0.0
        for i in range(0, len(perm), args.batch):
            b = np.sort(perm[i:i + args.batch])                # sorted indices read faster from the array
            loss = aligned_mse(model(batch_x(b)), y[b].to(device))[0].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        val = mean_loss(val_idx)
        history.append((total / len(fit_idx), val))
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print("  [%s] epoch %3d  train %.4f  validation %.4f  (units of mean radius^2; %d validation objects)"
                  % (signal, epoch + 1, history[-1][0], val, len(val_objs)))

    # Save the trained network with what is needed to reuse it on new windows (same input normalisation).
    torch.save({"state_dict": model.state_dict(), "in_channels": x.shape[1], "signal": signal,
                "joint_mean": joint_mean, "joint_std": joint_std, "unit_m": unit, "seed": args.seed,
                "val_objects": sorted(val_objs)}, os.path.join(args.out, "model_%s.pt" % signal))

    yv = y[te_idx].to(device)
    pred = predict(te_idx)
    loss, shift = aligned_mse(pred, yv)
    mse_mm2 = loss.cpu().numpy() * (unit * 1e3) ** 2        # per held-out window
    # Reference: always predict the mean training outline (what a network learns with no information).
    mean_outline = torch.from_numpy(y_m[fit_idx].mean(0, keepdims=True) / unit).to(device).expand_as(yv)
    ref_mm2 = aligned_mse(mean_outline, yv)[0].cpu().numpy() * (unit * 1e3) ** 2
    # Size-only reference: the mean training outline at scale 1, multiplied by each window's true object scale.
    # The hand's grip opening already gives away the size, so this is the bar for having learned SHAPE.
    sc = data["scale"].astype(np.float32)
    shape1 = (y_m[fit_idx] / sc[fit_idx, None]).mean(0, keepdims=True)
    sized = torch.from_numpy(shape1 * sc[te_idx, None] / unit).to(device)
    size_mm2 = aligned_mse(sized, yv)[0].cpu().numpy() * (unit * 1e3) ** 2

    obj_te = objects[te_idx]
    per_obj = {o: float(np.sqrt(mse_mm2[obj_te == o].mean())) for o in np.unique(obj_te)}
    return {
        "signal": signal,
        "n_train_windows": int(len(fit_idx)), "n_val_windows": int(len(val_idx)), "n_test_windows": int(len(te_idx)),
        "val_objects": sorted(val_objs),
        "heldout_mse_mm2": float(mse_mm2.mean()),
        "heldout_rmse_mm": float(np.sqrt(mse_mm2.mean())),
        "mean_outline_mse_mm2": float(ref_mm2.mean()),
        "sized_mean_outline_mse_mm2": float(size_mm2.mean()),
        "per_object_rmse_mm": per_obj,
        "history": history,
    }, {
        "pred_m": pred.cpu().numpy() * unit,          # predicted outline [m]
        "true_m": y_m[te_idx],                        # true outline [m]
        "shift": shift.cpu().numpy(),                 # roll the TRUE outline by -shift to overlay: np.roll(true, -shift)
        "object": obj_te,
        "mse_mm2": mse_mm2,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="shard .npz files and/or folders of shards")
    ap.add_argument("--signals", nargs="+", default=["none", "binary", "park"],
                    choices=["none", "binary", "park", "sensed"])
    ap.add_argument("--out", default="models/shape_recon/run",
                    help="output folder: results.json, predictions_<signal>.npz, model_<signal>.pt (git-ignored)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-objects", type=int, default=10,
                    help="train objects held out as validation (for tuning); the test objects are only scored at the end")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = shard_paths(args.data)
    data = load_shards(paths, args.signals)
    W, L = data["dof_pos"].shape[:2]
    train_objs = set(data["object"][data["split"] == "train"])
    test_objs = set(data["object"][data["split"] == "test"])
    assert not train_objs & test_objs, "an object appears in both splits: %s" % sorted(train_objs & test_objs)[:5]
    print("%d windows of %d steps from %d shard(s) on %s: %d train objects, %d held-out objects"
          % (W, L, len(paths), device, len(train_objs), len(test_objs)))

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for s in args.signals:
        t0 = time.time()
        res, pred = run_signal(s, data, args, device)
        res["train_seconds"] = time.time() - t0
        print("  [%s] %.0f s (%.1f s per epoch)" % (s, res["train_seconds"], res["train_seconds"] / args.epochs))
        np.savez(os.path.join(args.out, "predictions_%s.npz" % s), **pred)
        summary.append(res)
    json.dump({"shards": paths, "seed": args.seed, "epochs": args.epochs, "results": summary},
              open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\nHeld-out objects, rotation-aligned error")
    print("  %-8s %12s %10s" % ("signal", "MSE [mm^2]", "RMSE [mm]"))
    for r in summary:
        print("  %-8s %12.2f %10.2f" % (r["signal"], r["heldout_mse_mm2"], r["heldout_rmse_mm"]))
    print("  %-8s %12.2f %10.2f   (always predict the mean training outline)"
          % ("mean", summary[0]["mean_outline_mse_mm2"], np.sqrt(summary[0]["mean_outline_mse_mm2"])))
    print("  %-8s %12.2f %10.2f   (mean outline x the true object scale: knows the size, not the shape)"
          % ("sized", summary[0]["sized_mean_outline_mse_mm2"], np.sqrt(summary[0]["sized_mean_outline_mse_mm2"])))
    print("Saved to %s/" % args.out)


if __name__ == "__main__":
    main()
