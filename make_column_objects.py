"""Generate column-shaped objects (extruded convex polygons)
Usage (run from anywhere; OUT_DIR defaults to this repo's assets/urdf/objects):
  python make_column_objects.py
  python make_column_objects.py --n 109 --seed 1

Output, matching the layout of the other objects in assets/urdf/objects:
  <OUT_DIR>/<name>.urdf                      one URDF per object, next to the provided ones
  <OUT_DIR>/meshes/generated/<name>.obj      meshes (URDF paths point here)
  <OUT_DIR>/meshes/columns.json              names, 64-radius labels, train/test split
  <OUT_DIR>/meshes/columns_preview.png       picture of every outline

Import numpy required

Conventions copied from the provided objects (e.g. set_obj11_cylinder.urdf):
  * meshes are stored at "unit" size and scaled by 0.03 in the URDF, so 1 mesh unit = 30 mm
  * the provided objects are 60 mm tall (z from -1 to +1 mesh units) and centred on the origin
  * inertia is given as a density; the task overwrites the mass with a random 0.2-0.6 kg anyway

Shapes are CONVEX on purpose. Isaac Gym collides a mesh as its convex hull unless convex
decomposition is switched on, so a non-convex outline would not be what the hand actually feels.
"""
import argparse
import json
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "urdf", "objects")
MESH_SUBDIR = "generated"  # meshes go in OUT_DIR/meshes/<MESH_SUBDIR>/

URDF_SCALE = 0.03          # metres per mesh unit, same as the provided objects
HALF_HEIGHT = 1.0          # mesh units -> 60 mm tall, same as the provided objects
N_RADII = 64               # outline label resolution (the plan's choice)

URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <mesh filename="meshes/{sub}/{name}.obj" scale="{s} {s} {s}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <mesh filename="meshes/{sub}/{name}.obj" scale="{s} {s} {s}"/>
      </geometry>
    </collision>
    <inertial>
      <density value="1000.0"/>
    </inertial>
  </link>
</robot>
"""

# Geometry
def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def polygon_area_centroid(pts):
    nxt = np.roll(pts, -1, axis=0)
    c = cross2(pts, nxt)
    area = c.sum() / 2.0
    centroid = ((pts + nxt) * c[:, None]).sum(axis=0) / (6.0 * area)
    return area, centroid

def is_convex_ccw(pts, tol=1e-9):
    e1 = np.roll(pts, -1, axis=0) - pts
    e2 = np.roll(pts, -2, axis=0) - np.roll(pts, -1, axis=0)
    return bool(np.all(cross2(e1, e2) > tol))

MIN_EDGE_DIST = 0.50       # mesh units (15 mm): closest the outline may come to the axis
MAX_WIDTH = 2.20           # mesh units (66 mm): largest allowed extent in x or y (the provided objects are 48-63 mm across)

def sample_convex_polygon(rng): #Shape selection function
    smooth = rng.random() < 0.2
    n = int(rng.integers(40, 49)) if smooth else int(rng.integers(3, 9))
    while True:
        pts = _sample_convex_polygon(rng, n, smooth)
        nxt = np.roll(pts, -1, axis=0)
        edge_dist = np.abs(cross2(pts, nxt)) / np.linalg.norm(nxt - pts, axis=1)
        if edge_dist.min() >= MIN_EDGE_DIST and np.ptp(pts, axis=0).max() <= MAX_WIDTH:
            return pts


def _sample_convex_polygon(rng, n, smooth):
    step = 2 * np.pi / n
    jitter = 0.0 if smooth else 0.3
    angles = np.arange(n) * step + rng.uniform(-jitter, jitter, n) * step + rng.uniform(0, 2 * np.pi)

    # Size: the provided objects are 48-60 mm across with corners up to ~42 mm from the axis.
    if smooth:
        a = rng.uniform(0.85, 1.10)            # semi-axis: 25.5-33 mm
    else:
        a = rng.uniform(1.00, 1.30)            # corner radius: 30-39 mm
    b = a * rng.uniform(0.75, 1.0)             # aspect ratio
    tilt = rng.uniform(0, np.pi)

    x, y = a * np.cos(angles), b * np.sin(angles)
    pts = np.column_stack([x * np.cos(tilt) - y * np.sin(tilt),
                           x * np.sin(tilt) + y * np.cos(tilt)])
    _, centroid = polygon_area_centroid(pts)
    return pts - centroid


def outline_radii(pts, n_radii=N_RADII):
    theta = 2 * np.pi * np.arange(n_radii) / n_radii
    d = np.column_stack([np.cos(theta), np.sin(theta)])[:, None, :]     # (R, 1, 2)
    p = pts[None, :, :]                                                  # (1, E, 2)
    e = (np.roll(pts, -1, axis=0) - pts)[None, :, :]                     # (1, E, 2)
    denom = cross2(d, e)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cross2(p, e) / denom           # distance along the ray
        s = cross2(p, d) / denom           # position along the edge, 0..1
    hit = (np.abs(denom) > 1e-12) & (r > 0) & (s >= -1e-9) & (s <= 1 + 1e-9)
    r = np.where(hit, r, np.inf)
    out = r.min(axis=1)
    assert np.all(np.isfinite(out)), "a ray missed the polygon (origin not inside?)"
    return out


def extrude_to_obj(pts, half_height, path):
    n = len(pts)
    lines = ["# extruded convex polygon, generated by make_column_objects.py"]
    for z in (-half_height, half_height):
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y in pts]
    bot = lambda i: i % n + 1            # OBJ indices start at 1
    top = lambda i: i % n + 1 + n
    for i in range(n):                   # side walls, outward-facing
        lines.append(f"f {bot(i)} {bot(i + 1)} {top(i + 1)}")
        lines.append(f"f {bot(i)} {top(i + 1)} {top(i)}")
    for i in range(1, n - 1):            # caps (fan triangulation is valid for convex polygons)
        lines.append(f"f {top(0)} {top(i)} {top(i + 1)}")
        lines.append(f"f {bot(0)} {bot(i + 1)} {bot(i)}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=125, help="number of objects to generate (default 125)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-fraction", type=float, default=0.2,
                    help="fraction of objects held out for the shape-reconstruction test set")
    ap.add_argument("--prefix", default="col", help="object name prefix")
    args = ap.parse_args()

    obj_dir = OUT_DIR
    if not os.path.isdir(obj_dir):
        raise SystemExit(f"Not found: {obj_dir}\nSet OUT_DIR at the top of make_column_objects.py.")
    meta_dir = os.path.join(obj_dir, "meshes")                 # columns.json and the preview
    mesh_dir = os.path.join(meta_dir, MESH_SUBDIR)
    os.makedirs(mesh_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    names, radii_m, polys_m = [], [], []
    for i in range(args.n):
        name = f"{args.prefix}_{i:03d}"
        pts = sample_convex_polygon(rng)
        assert is_convex_ccw(pts), f"{name}: polygon is not convex"
        extrude_to_obj(pts, HALF_HEIGHT, os.path.join(mesh_dir, f"{name}.obj"))
        with open(os.path.join(obj_dir, f"{name}.urdf"), "w") as f:
            f.write(URDF_TEMPLATE.format(name=name, sub=MESH_SUBDIR, s=URDF_SCALE))
        names.append(name)
        radii_m.append(outline_radii(pts) * URDF_SCALE)
        polys_m.append((pts * URDF_SCALE).tolist())
    radii_m = np.array(radii_m)

    # Held-out split by object, for the shape-reconstruction study.
    order = rng.permutation(args.n)
    n_test = int(round(args.test_fraction * args.n))
    test = sorted(names[j] for j in order[:n_test])
    train = sorted(names[j] for j in order[n_test:])

    registry = {
        "names": names,
        "outline_radii_m": radii_m.round(6).tolist(),
        "polygon_xy_m": polys_m,
        "height_m": 2 * HALF_HEIGHT * URDF_SCALE,
        "n_radii": N_RADII,
        "angle_convention": "radius k is at angle 2*pi*k/n_radii from +x towards +y, object frame, about z",
        "scale_note": "Values are for object scale 1.0. The task scales each object by a random "
                      "0.95-1.1 per environment; multiply the radii by that scale when labelling a rollout.",
        "train": train,
        "test": test,
        "seed": args.seed,
    }
    reg_path = os.path.join(meta_dir, "columns.json")
    with open(reg_path, "w") as f:
        json.dump(registry, f)

    print(f"Wrote {args.n} objects to {obj_dir}")
    print(f"  outline radius: min {radii_m.min()*1e3:.1f} mm, mean {radii_m.mean()*1e3:.1f} mm, "
          f"max {radii_m.max()*1e3:.1f} mm; height {registry['height_m']*1e3:.0f} mm")
    print(f"  split: {len(train)} train / {len(test)} test objects")
    print(f"  registry: {reg_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = 16
        rows = int(np.ceil(args.n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
        for ax in np.ravel(axs):
            ax.axis("off")
        th = 2 * np.pi * np.arange(N_RADII + 1) / N_RADII
        for ax, name, r in zip(np.ravel(axs), names, radii_m):
            rr = np.append(r, r[0]) * 1e3
            ax.fill(rr * np.cos(th), rr * np.sin(th), color="tab:red" if name in test else "tab:blue")
            ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_aspect("equal")
            ax.set_title(name[-3:], fontsize=6, pad=1)
        fig.suptitle("Column outlines from the 64-radius labels (blue = train, red = held out)", fontsize=9)
        png = os.path.join(meta_dir, "columns_preview.png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        print(f"  preview: {png}")
    except ImportError:
        print("  (matplotlib not installed, skipped the preview image)")


if __name__ == "__main__":
    main()
