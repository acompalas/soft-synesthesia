"""3D preview of a random sample of the generated column objects.

Reads the actual OBJ meshes the URDFs point to (assets/urdf/objects/meshes/generated/),
applies the URDF scale (0.03 m per mesh unit), and draws them in an N x N panel.
Title colour: blue = train, red = held out (from columns.json).

Usage (from the repo root):
  python visualize_columns_3d.py                 # 4x4, seed 0, saves columns_3d_preview.png and shows it
  python visualize_columns_3d.py --grid 3 --seed 5
  python visualize_columns_3d.py --no-show       # save only
"""
import argparse
import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OBJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "urdf", "objects")
MESH_DIR = os.path.join(OBJ_ROOT, "meshes", "generated")
REGISTRY = os.path.join(OBJ_ROOT, "meshes", "columns.json")
URDF_SCALE = 0.03  # must match make_column_objects.py


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                faces.append([int(p.split("/")[0]) - 1 for p in parts[1:]])
    return np.array(verts), faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=4, help="panel is grid x grid (default 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["all", "train", "test"], default="all",
                    help="sample from all objects, only training objects, or only held-out test objects")
    ap.add_argument("--out", default="columns_3d_preview.png")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="only save the PNG, do not open a window")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    with open(REGISTRY) as f:
        reg = json.load(f)
    test = set(reg["test"])
    names = {"all": reg["names"], "train": reg["train"], "test": reg["test"]}[args.split]

    k = min(args.grid * args.grid, len(names))
    rng = np.random.default_rng(args.seed)
    picks = sorted(rng.choice(names, size=k, replace=False))

    fig = plt.figure(figsize=(3 * args.grid, 3 * args.grid))
    lim = 40.0  # mm, half-width of every axis box so sizes are comparable across panels
    for i, name in enumerate(picks):
        v, faces = load_obj(os.path.join(MESH_DIR, f"{name}.obj"))
        v = v * URDF_SCALE * 1e3  # mesh units -> mm
        n_side = len(v) // 2      # vertices: bottom ring then top ring
        held_out = name in test
        color = "tab:red" if held_out else "tab:blue"

        ax = fig.add_subplot(args.grid, args.grid, i + 1, projection="3d")
        tris = [v[fc] for fc in faces]
        ax.add_collection3d(Poly3DCollection(tris, facecolor=color, alpha=0.35,
                                             edgecolor="none"))
        # outline the top and bottom rings and the vertical edges (skip vertical edges
        # on smooth shapes, where they would just be a dense grey wall)
        for ring in (v[:n_side], v[n_side:]):
            closed = np.vstack([ring, ring[:1]])
            ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="k", lw=0.8)
        if n_side <= 12:
            for j in range(n_side):
                ax.plot(*np.column_stack([v[j], v[j + n_side]]), color="k", lw=0.8)

        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=35)
        ax.tick_params(labelsize=6, pad=0)
        ax.set_title(f"{name}  ({'test' if held_out else 'train'}, {n_side} sides)",
                     fontsize=8, color=color)

    fig.suptitle(f"{k} random column objects (seed {args.seed}), axes in mm, "
                 f"blue = train, red = held out", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {os.path.abspath(args.out)}")
    if args.show:
        if "agg" in matplotlib.get_backend().lower():
            # no GUI backend in this env (e.g. no tkinter): open the saved PNG instead
            import subprocess
            subprocess.Popen(["xdg-open", os.path.abspath(args.out)])
        else:
            plt.show()


if __name__ == "__main__":
    main()
