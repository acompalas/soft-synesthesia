"""Checks for shape_recon.py and the reconstruction shards before any real training.

Part 1 (no data): the rotation-aligned loss, the network shapes, and that the network can memorise a
handful of windows (if it cannot overfit 64 windows, something in the model or the loss is broken).
Part 2 (real shards): labels, split, touch ranges, and that every kept window really is drop-free and
shows the object turning.

Small and CPU-friendly, so it can run while Isaac Gym uses the GPU:
    CUDA_VISIBLE_DEVICES= python test_shape_recon.py --data rollouts/recon_shards/train_s100.npz rollouts/recon_shards/test_s127.npz
"""
import argparse
import json
import os

import numpy as np
import torch

import shape_recon as sr

ROOT = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  (" + detail + ")") if detail else ""))


def part1(radii_bank):
    print("Part 1: loss and network")
    R = 64
    t = torch.from_numpy(radii_bank[:8].astype(np.float32))
    pred = torch.roll(t, shifts=-3, dims=1)                    # = np.roll(true, -3)
    loss, shift = sr.aligned_mse(pred, t)
    check("aligned loss is 0 for a rotated copy of the outline", float(loss.max()) < 1e-10, "max %.1e" % loss.max())
    check("recovered shift matches the docstring (np.roll(true, -shift) overlays)", bool((shift == 3).all()),
          "shifts %s" % shift.tolist()[:4])
    loss_other, _ = sr.aligned_mse(t[1:2].expand(8, R), t)
    check("aligned loss is > 0 for a different object", float(loss_other[2:].min()) > 0,
          "min %.2e m^2" % loss_other[2:].min())
    plain = ((t[1:2] - t) ** 2).mean(-1)
    check("aligned loss never exceeds the unaligned loss", bool((loss_other <= plain + 1e-12).all()))

    m = sr.TemporalCNN(32)
    n_par = sum(p.numel() for p in m.parameters())
    m.eval()
    out200 = m(torch.zeros(4, 32, 200))
    out150 = m(torch.zeros(4, 32, 150))
    check("network maps (B, 32, L) to (B, 64) for L = 200 and 150", out200.shape == (4, 64) and out150.shape == (4, 64),
          "%d parameters" % n_par)
    rf, jump = 1, 1                                            # receptive field of the conv stack, in control steps
    for layer in m.conv:
        c = layer[0]
        rf += (c.kernel_size[0] - 1) * jump
        jump *= c.stride[0]
    print("  receptive field of the conv stack: %d steps (%.1f s) before pooling over time" % (rf, rf / 10))


def overfit(data):
    print("Part 1b: memorise 64 real windows (binary signal)")
    idx = np.where(data["split"] == "train")[0][:64]
    x = torch.from_numpy(sr.build_inputs(data, "binary", idx)[idx]).float()
    y_m = data["radii"][idx].astype(np.float32)
    unit = float(y_m.mean())
    y = torch.from_numpy(y_m / unit)
    torch.manual_seed(0)
    m = sr.TemporalCNN(x.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ref = sr.aligned_mse(y.mean(0, keepdim=True).expand_as(y), y)[0].mean().item()
    mm = lambda v: np.sqrt(v) * unit * 1e3
    print("  mean outline: %.2f mm RMSE" % mm(ref))
    for step in range(1, 2001):
        loss = sr.aligned_mse(m(x), y)[0].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0:
            m.eval()
            with torch.no_grad():
                ev = sr.aligned_mse(m(x), y)[0].mean().item()
            m.train()
            print("  step %4d: train-mode %.2f mm, eval-mode %.2f mm" % (step, mm(loss.item()), mm(ev)))
    check("training error falls far below the mean-outline error (MSE < 10%)", ev < 0.1 * ref,
          "after 2000 steps %.2f mm" % mm(ev))


def yaw_of(q):                                                  # Isaac Gym quaternions are (x, y, z, w)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def part2(paths):
    print("Part 2: shards")
    reg = json.load(open(os.path.join(ROOT, "assets/urdf/objects/meshes/columns.json")))
    bank = dict(zip(reg["names"], np.asarray(reg["outline_radii_m"], np.float32)))
    test_set = set(reg["test"])
    keys = ["dof_pos", "obj_pose", "touch_binary", "touch_park", "sensed", "radii", "object", "split", "scale"]
    parts = {k: [] for k in keys}
    for p in paths:
        with np.load(p) as z:
            for k in keys:
                parts[k].append(z[k])
    d = {k: np.concatenate(v) for k, v in parts.items()}
    obj, split = d["object"].astype(str), d["split"].astype(str)
    W, L = d["dof_pos"].shape[:2]
    print("  %d windows of %d steps, %d objects" % (W, L, len(set(obj))))

    expect = np.stack([bank[o] for o in obj]) * d["scale"][:, None]
    err = np.abs(d["radii"] - expect).max()
    check("label = columns.json outline x env scale", err < 1e-5, "max error %.1e m" % err)
    check("split agrees with the columns.json test list", all((s == "test") == (o in test_set) for o, s in zip(obj, split)))
    check("no object in both splits", not (set(obj[split == "train"]) & set(obj[split == "test"])))
    check("window length is 200 (paper)", L == 200)
    check("binary touch is 0/1", set(np.unique(d["touch_binary"]).tolist()) <= {0, 1})
    tp = d["touch_park"].astype(np.float32)
    check("Park reading in [0, 1]", tp.min() >= 0 and tp.max() <= 1.0 + 1e-3, "max %.3f" % tp.max())
    check("hand joints are finite", bool(np.isfinite(d["dof_pos"]).all()))

    z = d["obj_pose"][..., 2]
    sag = (z[:, :1] - z.min(1, keepdims=True))[:, 0]
    check("no window contains a drop (object never falls > 3 cm below its start)", float(sag.max()) < 0.03,
          "largest sag %.1f mm, median %.1f mm" % (1e3 * sag.max(), 1e3 * np.median(sag)))
    turns = np.abs(np.diff(np.unwrap(yaw_of(d["obj_pose"][..., 3:7]), axis=1), axis=1).sum(1)) / (2 * np.pi)
    print("  object turns per window: median %.2f, 10th pct %.2f, 90th pct %.2f; %.1f%% under a quarter turn"
          % (np.median(turns), np.percentile(turns, 10), np.percentile(turns, 90), 100 * (turns < 0.25).mean()))
    print("  touch per window: binary on %.1f%% of pad-steps, Park > 0 on %.1f%%; %.1f%% of windows with no binary touch"
          % (100 * d["touch_binary"].mean(), 100 * (tp > 0).mean(), 100 * (d["touch_binary"].sum((1, 2)) == 0).mean()))
    return bank


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="a train shard and a test shard")
    args = ap.parse_args()
    torch.set_num_threads(4)
    bank = part2(sr.shard_paths(args.data))
    part1(np.stack(list(bank.values())))
    overfit(sr.load_shards(sr.shard_paths(args.data), ["binary"]))
    print("\n%d / %d checks passed" % (sum(results), len(results)))


if __name__ == "__main__":
    main()
