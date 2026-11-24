"""Make a FAKE dataset shard for testing shape_recon.py before real rollouts exist.

THIS IS NOT SIMULATION DATA. It is a toy model whose only purpose is to check that the data
format, the network and the loss run end to end and that the network can learn when the touch
signal does depend on shape. Do not report numbers from it.

It writes the same fields as the repo's build_recon_dataset.py, and computes the Park readings with
the repo's own park_skin.py (Dragon Skin 20, log readout), so shape_recon.py reads it unchanged.

Toy model: the object turns about z at a constant rate. Each of the 16 pads sits at a fixed angle
and distance from the axis and is pressed when the outline reaches it. The force grows with how far
the outline pushes past the pad, and it is shared unevenly between the pad's 4 cells depending on
the local slope of the outline (a corner loads fewer cells than a flat face). Joint positions are a
periodic finger motion plus a small push-back from the contact.

Usage (from the repo root; the shard goes to rollouts/fake_shards/, which git ignores):
    python shape_fake_rollouts.py
    python shape_recon.py --data rollouts/fake_shards
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch


def load_park(repo):
    spec = importlib.util.spec_from_file_location("park_skin", os.path.join(repo, "isaacgymenvs/utils/park_skin.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)),
                    help="path to the soft-synesthesia repository (default: the folder this script is in)")
    ap.add_argument("--out", default="rollouts/fake_shards/fake_shard.npz", help="output shard (default is git-ignored)")
    ap.add_argument("--per-object", type=int, default=40, help="windows per object")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tau", type=float, default=1.5, help="binary threshold [N]")
    ap.add_argument("--park-E", type=float, default=338e3)
    ap.add_argument("--park-log-eps", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reg = json.load(open(os.path.join(args.repo, "assets/urdf/objects/meshes/columns.json")))
    names, radii = reg["names"], np.asarray(reg["outline_radii_m"])
    test_set = set(reg["test"])
    R = radii.shape[1]
    rng = np.random.default_rng(args.seed)

    ps = load_park(args.repo)
    pads = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr", "link_5.0_fsr",
            "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr", "link_10.0_fsr",
            "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
            "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]
    cells = [p if k == 0 else "%s_c%d" % (p, k) for p in pads for k in range(4)]
    urdf = os.path.join(args.repo, "assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_split4.urdf")
    skin = ps.ParkSkin(ps.cell_areas_from_urdf(urdf, cells), n_cells=4, E=args.park_E, log_eps=args.park_log_eps)

    pad_angle = rng.uniform(0, 2 * np.pi, 16)            # where each pad sits around the object
    pad_dist = rng.uniform(0.020, 0.030, 16)             # its distance from the axis [m]
    stiffness = 600.0                                    # N per metre of overlap (toy value)
    t = np.arange(args.steps)

    out = {k: [] for k in ("dof_pos", "touch_binary", "touch_park", "radii", "object", "split", "scale")}
    for name, r in zip(names, radii):
        slope = (np.roll(r, -1) - np.roll(r, 1)) / 2.0   # change of radius per angular step
        for _ in range(args.per_object):
            scale = rng.uniform(0.95, 1.10)
            turns = rng.uniform(1.0, 3.0)                # the paper's objects turn a few times in 20 s
            theta = rng.uniform(0, 2 * np.pi) + 2 * np.pi * turns * t / args.steps
            # outline angle facing each pad at each step, as a (fractional) index into the 64 radii
            idx = ((pad_angle[None, :] - theta[:, None]) % (2 * np.pi)) / (2 * np.pi) * R
            i0 = np.floor(idx).astype(int) % R
            frac = idx - np.floor(idx)
            r_here = (r[i0] * (1 - frac) + r[(i0 + 1) % R] * frac) * scale
            s_here = slope[i0] * scale
            overlap = np.clip(r_here - pad_dist[None, :], 0, None)
            force = stiffness * overlap * (1 + 0.1 * rng.standard_normal(overlap.shape))
            force = np.clip(force, 0, None)              # (T, 16) total pad force [N]
            g = np.clip(s_here / 0.002, -1, 1)           # -1..1: which side of the pad is loaded
            w = np.stack([1 + g, 1 + g, 1 - g, 1 - g], axis=-1) / 4.0
            w = w ** 3
            w /= w.sum(-1, keepdims=True)
            cf = (force[..., None] * w).reshape(args.steps, 64)   # (T, 64) cell normal force, pad-major
            gait = 0.3 * np.sin(2 * np.pi * 4 * t[:, None] / args.steps + np.linspace(0, 2 * np.pi, 16)[None, :])
            q = gait - 2.0 * overlap + 0.03 * rng.standard_normal((args.steps, 16))
            dof = np.zeros((args.steps, 22), np.float32)
            dof[:, 6:] = q                               # columns 0-5 are the arm, as in the real data
            out["dof_pos"].append(dof)
            out["touch_binary"].append((force >= args.tau).astype(np.uint8))
            out["touch_park"].append(skin.pad_readings(torch.as_tensor(cf, dtype=torch.float32)).numpy())
            out["radii"].append(r * scale)
            out["object"].append(name)
            out["split"].append("test" if name in test_set else "train")
            out["scale"].append(scale)

    res = {k: np.array(v) for k, v in out.items()}
    res["touch_park"] = res["touch_park"].astype(np.float16)
    res["radii"] = res["radii"].astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **res)
    tp = res["touch_park"].astype(np.float32)
    print("Wrote %d FAKE windows (%d objects x %d) to %s" % (len(res["object"]), len(names), args.per_object, args.out))
    print("  binary on %.0f%% of pad-steps; Park > 0 on %.0f%%, Park saturated (1.0) on %.1f%%"
          % (100 * res["touch_binary"].mean(), 100 * (tp > 0).mean(), 100 * (tp >= 0.999).mean()))


if __name__ == "__main__":
    main()
