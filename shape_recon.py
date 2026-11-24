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
    keys = ["dof_pos", "radii", "object", "split"] + [TOUCH_KEY[s] for s in signals if s != "none"]
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
                elif k in TOUCH_KEY.values():
                    v = v.astype(np.float32)
                parts[k].append(v)
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["object"] = data["object"].astype(str)
    data["split"] = data["split"].astype(str)
    return data


def build_inputs(data, signal):
    """(W, C, L): 16 hand joints, plus 16 touch channels unless signal == 'none'."""
    x = data["dof_pos"] if signal == "none" else np.concatenate([data["dof_pos"], data[TOUCH_KEY[signal]]], -1)
    return np.ascontiguousarray(np.transpose(x, (0, 2, 1)))


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
    tr, te = data["split"] == "train", data["split"] == "test"
    assert tr.any() and te.any(), "need windows from both the train and the test objects"
    y_m = data["radii"].astype(np.float32)                 # outline of each env's (scaled) object [m]

    x = build_inputs(data, signal)
    # Normalise the joint channels with training statistics. Touch channels are already in [0, 1]
    # for every signal, so they are left as they are: the same scale for binary and Park, and a pad
    # that is rarely touched in training cannot turn into a huge input at test time.
    mean = x[tr, :16].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(x[tr, :16].std(axis=(0, 2), keepdims=True), JOINT_STD_FLOOR)
    x[:, :16] = (x[:, :16] - mean) / std

    unit = float(y_m[tr].mean())                           # train in units of the mean radius
    xt, yt = torch.from_numpy(x[tr]), torch.from_numpy(y_m[tr] / unit)
    xv, yv = torch.from_numpy(x[te]), torch.from_numpy(y_m[te] / unit).to(device)

    torch.manual_seed(args.seed)
    model = TemporalCNN(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    gen = torch.Generator().manual_seed(args.seed)

    def predict(xs):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xs[i:i + 4096].to(device)) for i in range(0, len(xs), 4096)])

    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            loss = aligned_mse(model(xt[b].to(device)), yt[b].to(device))[0].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        val = aligned_mse(predict(xv), yv)[0].mean().item()   # printed only; never used to choose anything
        history.append((total / len(xt), val))
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print("  [%s] epoch %3d  train %.4f  held-out %.4f  (units of mean radius^2)"
                  % (signal, epoch + 1, history[-1][0], val))

    pred = predict(xv)
    loss, shift = aligned_mse(pred, yv)
    mse_mm2 = loss.cpu().numpy() * (unit * 1e3) ** 2        # per held-out window
    # Reference: always predict the mean training outline (what a network learns with no information).
    mean_outline = torch.from_numpy(y_m[tr].mean(0, keepdims=True) / unit).to(device).expand_as(yv)
    ref_mm2 = aligned_mse(mean_outline, yv)[0].cpu().numpy() * (unit * 1e3) ** 2

    obj_te = data["object"][te]
    per_obj = {o: float(np.sqrt(mse_mm2[obj_te == o].mean())) for o in np.unique(obj_te)}
    return {
        "signal": signal,
        "n_train_windows": int(tr.sum()), "n_test_windows": int(te.sum()),
        "heldout_mse_mm2": float(mse_mm2.mean()),
        "heldout_rmse_mm": float(np.sqrt(mse_mm2.mean())),
        "mean_outline_mse_mm2": float(ref_mm2.mean()),
        "per_object_rmse_mm": per_obj,
        "history": history,
    }, {
        "pred_m": pred.cpu().numpy() * unit,          # predicted outline [m]
        "true_m": y_m[te],                            # true outline [m]
        "shift": shift.cpu().numpy(),                 # roll the TRUE outline by -shift to overlay: np.roll(true, -shift)
        "object": obj_te,
        "mse_mm2": mse_mm2,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="shard .npz files and/or folders of shards")
    ap.add_argument("--signals", nargs="+", default=["none", "binary", "park"],
                    choices=["none", "binary", "park", "sensed"])
    ap.add_argument("--out", default="rollouts/shape_recon_results", help="output folder (default is git-ignored)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
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
        res, pred = run_signal(s, data, args, device)
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
    print("Saved to %s/" % args.out)


if __name__ == "__main__":
    main()
