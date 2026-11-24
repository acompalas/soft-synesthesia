"""Reconstruction figures comparing the binary baseline policy and the Park policy.

Each rollout set is scored with the touch signal of the policy that produced it (baseline sets: binary,
Park sets: Park), one CNN per set, as trained by shape_recon.py. Two checkpoint pairs:
    latest : baseline ep5000  vs  Park ep5000
    best   : baseline ep4500  vs  Park ep3600   (highest training reward)

Figures (written to --out):
  recon_outlines.png    held-out objects (rows) x pair (columns): true cross-section, no-touch, baseline and
                        Park reconstructions overlaid in the object frame [mm]
  recon_3d_latest.png   layout of the paper's Fig. 9: rows GT | w/o touch | w/ binary | w/ Park, columns =
                        held-out objects, grey meshes on white rendered by Isaac Gym (--renderer mpl: matplotlib)
  recon_3d_best.png     the same for the best-reward pair
A set whose results are not there yet is left blank. "No touch" is the joints-only CNN of the baseline set of
the same pair by default (--none-from park uses the Park set's instead): the reference for what touch adds.

How a reconstruction is made (same as plot_shape_recon.py): each held-out window's 64 predicted radii are
rotated by the shift the loss found (into the object frame) and divided by that window's object scale, and
the windows of each object are averaged. The 3D shape extrudes that outline to the object's 60 mm height.
"True" is the object's own URDF, i.e. exactly the mesh the policy rotated in simulation.

Usage (repo root, soft-syn env; Isaac Gym renders off-screen with camera sensors, no viewer window):
    python plot_recon_compare.py
    python plot_recon_compare.py --objects col_012 col_092 col_118
    python plot_recon_compare.py --no-3d                # 2D outlines only, no Isaac Gym
Default results folders (seed 1 of each set):
    models/shape_recon/{baseline,park}_{latest,best}/seed1
"""
import argparse
import json
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OBJ_DIR = os.path.join(ROOT, "assets/urdf/objects")
COLUMNS = os.path.join(OBJ_DIR, "meshes/columns.json")
SETS = {  # (pair, policy) -> touch signal the CNN was trained on
    ("latest", "baseline"): "binary", ("latest", "park"): "park",
    ("best", "baseline"): "binary", ("best", "park"): "park"}
PAIR_TITLE = {"latest": "Latest (baseline ep5000, Park ep5000)",
              "best": "Best reward (baseline ep4500, Park ep3600)"}
COLOR = {"true": "#222222", "none": "#8c8c8c", "baseline": "#1f77b4", "park": "#d62728"}
LABEL = {"true": "True", "none": "No touch", "baseline": "Baseline (binary)", "park": "Park skin"}
RGB_3D = {"true": (0.85, 0.85, 0.85), "none": (0.45, 0.45, 0.45), "baseline": (0.30, 0.52, 0.80),
          "park": (0.85, 0.33, 0.30)}
RECON = ("none", "baseline", "park")                              # reconstruction columns, in order


# --------------------------------------------------------------------------- data
def load_set(results_dir, signal, labels):
    """Per held-out object: mean reconstructed outline in the object frame at scale 1 [m], and RMSE [mm]."""
    path = os.path.join(results_dir, "predictions_%s.npz" % signal)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    pred, true, shift, obj = d["pred_m"], d["true_m"], d["shift"].astype(int), d["object"].astype(str)
    R = pred.shape[1]
    idx = (np.arange(R)[None, :] - shift[:, None]) % R            # pred[k] ~ true[k + shift] -> object frame
    aligned = pred[np.arange(len(pred))[:, None], idx]
    scale = true.mean(1) / np.array([labels[o].mean() for o in obj])
    aligned = aligned / scale[:, None]
    out = {}
    for o in np.unique(obj):
        m = obj == o
        out[o] = {"outline": aligned[m].mean(0), "rmse_mm": float(np.sqrt(d["mse_mm2"][m].mean()))}
    return out


def pick_objects(sets, n):
    """Best, median and worst held-out objects by mean RMSE over the sets that exist."""
    avail = [s for s in sets.values() if s is not None]
    objs = sorted(set.intersection(*[set(s) for s in avail]))
    err = np.array([np.mean([s[o]["rmse_mm"] for s in avail]) for o in objs])
    order = [objs[i] for i in np.argsort(err)]
    k = max(1, n // 3)
    mid = len(order) // 2 - k // 2
    picks = order[:k] + order[mid:mid + k] + order[len(order) - (n - 2 * k):]
    return list(dict.fromkeys(picks))


def ring(r):
    th = 2 * np.pi * np.arange(len(r)) / len(r)                   # radius k at angle 2*pi*k/64, from +x to +y
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


# --------------------------------------------------------------------------- 2D
def fig_outlines(picks, polys, sets, out):
    pairs = ["latest", "best"]
    fig, axs = plt.subplots(len(picks), 2, figsize=(6.4, 3.2 * len(picks)), squeeze=False)
    for i, o in enumerate(picks):
        for j, pair in enumerate(pairs):
            ax = axs[i, j]
            p = np.vstack([polys[o], polys[o][:1]]) * 1e3
            ax.plot(p[:, 0], p[:, 1], color=COLOR["true"], lw=1.6, ls=(0, (4, 2)), label=LABEL["true"], zorder=1)
            errs = []
            for pol in RECON:
                s = sets[(pair, pol)]
                if s is None or o not in s:
                    errs.append("%s n/a" % pol)
                    continue
                xy = ring(s[o]["outline"]) * 1e3
                xy = np.vstack([xy, xy[:1]])
                ax.plot(xy[:, 0], xy[:, 1], color=COLOR[pol], lw=1.4, label=LABEL[pol], zorder=2)
                errs.append("%s %.2f" % (pol, s[o]["rmse_mm"]))
            ax.set_aspect("equal")
            ax.set_xlim(-35, 35); ax.set_ylim(-35, 35)
            ax.set_xticks([-30, 0, 30]); ax.set_yticks([-30, 0, 30]); ax.tick_params(labelsize=7)
            ax.grid(color="0.9", lw=0.5)
            title = ("%s\n" % PAIR_TITLE[pair] if i == 0 else "") + "%s  RMSE mm: %s" % (o, ", ".join(errs))
            ax.set_title(title, fontsize=7.5)
    handles = [plt.Line2D([], [], color=COLOR[k], lw=1.8, ls=(0, (4, 2)) if k == "true" else "-", label=LABEL[k])
               for k in ("true",) + RECON]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Held-out objects: true cross-section and reconstruction (mm, object frame)", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


# --------------------------------------------------------------------------- 3D (Isaac Gym)
def write_outline_urdf(radii_m, height_m, folder, name):
    """Extrude a 64-radius outline (object frame, metres) to a column centred on the origin, as the task's
    objects are. Caps are fans from the centre, valid for any outline given as radii (star-shaped)."""
    pts, n, h = ring(radii_m), len(radii_m), height_m / 2
    v = ["v %.6f %.6f %.6f" % (x, y, z) for z in (-h, h) for x, y in pts] + ["v 0 0 %.6f" % -h, "v 0 0 %.6f" % h]
    b, t, cb, ct = (lambda i: i % n + 1), (lambda i: i % n + 1 + n), 2 * n + 1, 2 * n + 2
    f = []
    for i in range(n):
        f += ["f %d %d %d" % (b(i), b(i + 1), t(i + 1)), "f %d %d %d" % (b(i), t(i + 1), t(i)),
              "f %d %d %d" % (ct, t(i), t(i + 1)), "f %d %d %d" % (cb, b(i + 1), b(i))]
    with open(os.path.join(folder, name + ".obj"), "w") as fh:
        fh.write("\n".join(v + f) + "\n")
    with open(os.path.join(folder, name + ".urdf"), "w") as fh:
        fh.write('<?xml version="1.0"?>\n<robot name="object"><link name="object"><visual><origin xyz="0 0 0"/>'
                 '<geometry><mesh filename="%s.obj" scale="1 1 1"/></geometry></visual>'
                 '<inertial><mass value="0.3"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/>'
                 '</inertial></link></robot>\n' % name)
    return name + ".urdf"


def column_faces(radii_m, height_m):
    """Faces (sides, caps) of a column extruded from a 64-radius outline, centred on the origin [m]."""
    pts, n, h = ring(radii_m), len(radii_m), height_m / 2
    bot = np.column_stack([pts, np.full(n, -h)])
    top = np.column_stack([pts, np.full(n, h)])
    sides = np.stack([np.stack([bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]]) for i in range(n)])
    return sides, np.stack([top, bot[::-1]])


def render_mpl(panels, height_m, args):
    """Fallback renderer: shaded grey mesh on white with matplotlib (no Isaac Gym needed)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    light = np.array([-0.4, -0.7, 0.6]); light /= np.linalg.norm(light)
    imgs = []
    for p in panels:
        if p is None:
            imgs.append(None)
            continue
        fig = plt.figure(figsize=(3, 3), dpi=args.px // 3)
        ax = fig.add_subplot(projection="3d")
        az, el = np.radians(args.cam_az), np.radians(args.cam_el)
        view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])   # towards the camera
        sides, caps = column_faces(p, height_m)
        polys, cols = [], []
        for f in list(sides) + [caps[0], caps[1]]:               # own Lambert shading + back-face culling:
            nrm = np.cross(f, np.roll(f, -1, axis=0)).sum(0)       # Newell normal (robust for collinear points);
                                                                    # culling: matplotlib can't depth-sort a closed mesh
            nrm /= np.linalg.norm(nrm)
            if nrm @ view <= 0:
                continue
            polys.append(f)
            cols.append([0.30 + 0.55 * max(nrm @ light, 0.0)] * 3)
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.3))
        lim = 0.04
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=args.cam_el, azim=args.cam_az)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return imgs


def render_isaac(panels, height_m, args):
    """panels: list of (asset_root, urdf_file) or None. Isaac Gym camera-sensor renders of each mesh, grey,
    with everything that is not the object (sky, other envs) set to white via the segmentation image."""
    from isaacgym import gymapi
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    opts = gymapi.AssetOptions()
    opts.fix_base_link = True
    opts.disable_gravity = True
    cp = gymapi.CameraProperties()
    cp.width, cp.height, cp.horizontal_fov = args.px, args.px, args.fov
    az, el = np.radians(args.cam_az), np.radians(args.cam_el)
    target = np.zeros(3)
    eye = target + args.cam_dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    spacing = 2.0                                                 # metres between envs: neighbours stay out of view
    lower, upper = gymapi.Vec3(-spacing, -spacing, -spacing), gymapi.Vec3(spacing, spacing, spacing)
    cams, cache = [], {}
    for k, p in enumerate(panels):
        env = gym.create_env(sim, lower, upper, int(np.ceil(np.sqrt(len(panels)))))
        if p is None:
            cams.append(None)
            continue
        if p not in cache:
            cache[p] = gym.load_asset(sim, p[0], p[1], opts)
        actor = gym.create_actor(env, cache[p], gymapi.Transform(), "obj", k, 0, 1)   # segmentation id 1
        gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.62, 0.62, 0.62))
        cam = gym.create_camera_sensor(env, cp)
        gym.set_camera_location(cam, env, gymapi.Vec3(*eye.tolist()), gymapi.Vec3(*target.tolist()))
        cams.append((env, cam))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    imgs = []
    for c in cams:
        if c is None:
            imgs.append(None)
            continue
        rgb = np.asarray(gym.get_camera_image(sim, c[0], c[1], gymapi.IMAGE_COLOR), np.uint8)
        rgb = rgb.reshape(args.px, args.px, 4)[..., :3].copy()
        seg = np.asarray(gym.get_camera_image(sim, c[0], c[1], gymapi.IMAGE_SEGMENTATION)).reshape(args.px, args.px)
        rgb[seg == 0] = 255                                       # white background, as in the paper's Fig. 9
        imgs.append(rgb)
    gym.destroy_sim(sim)
    return imgs


def crop_common(imgs, margin=8):
    """Crop every image to the same box (union of all non-white pixels), so sizes stay comparable."""
    masks = [(im < 250).any(-1) for im in imgs if im is not None]
    if not masks:
        return imgs
    m = np.any(masks, axis=0)
    ys, xs = np.where(m)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, m.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, m.shape[1])
    return [None if im is None else im[y0:y1, x0:x1] for im in imgs]


ROW_LABEL = {"true": "GT", "none": "Prediction\nw/o touch", "baseline": "Prediction\nw/ binary touch",
             "park": "Prediction\nw/ Park skin"}


def fig_3d(picks, polys, sets, height_m, args):
    """Paper Fig. 9 layout: rows GT | w/o touch | w/ binary | w/ Park, columns = held-out objects,
    grey meshes on white, tilted so the 3D shape is visible. One figure per checkpoint pair."""
    tmp = tempfile.mkdtemp(prefix="recon_urdf_")
    rows = ["true"] + list(RECON)
    for pair in ("latest", "best"):
        panels, notes = [], []
        for r in rows:
            for o in picks:
                if r == "true":
                    outline = None
                    notes.append("")
                else:
                    s = sets[(pair, r)]
                    outline = s[o]["outline"] if s is not None and o in s else None
                    notes.append("not available yet" if outline is None else "%.2f mm" % s[o]["rmse_mm"])
                if args.renderer == "isaac":
                    if r == "true":
                        panels.append((OBJ_DIR, o + ".urdf"))
                    elif outline is None:
                        panels.append(None)
                    else:
                        panels.append((tmp, write_outline_urdf(outline, height_m, tmp, "%s_%s_%s" % (pair, r, o))))
                else:
                    if r == "true":                               # true outline as 64 radii of the real polygon
                        panels.append(np.asarray(json.load(open(COLUMNS))["outline_radii_m"])[
                            json.load(open(COLUMNS))["names"].index(o)])
                    else:
                        panels.append(outline)
        imgs = crop_common(render_isaac(panels, height_m, args) if args.renderer == "isaac"
                           else render_mpl(panels, height_m, args))
        fig, axs = plt.subplots(len(rows), len(picks), figsize=(1.0 + 2.0 * len(picks), 2.0 * len(rows)),
                                squeeze=False)
        fig.patch.set_facecolor("white")
        for i, r in enumerate(rows):
            for j, o in enumerate(picks):
                ax = axs[i, j]
                ax.axis("off")
                im, note = imgs[i * len(picks) + j], notes[i * len(picks) + j]
                if im is None:
                    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=8, color="0.55", transform=ax.transAxes)
                else:
                    ax.imshow(im)
                    if note:
                        ax.text(0.5, -0.02, "RMSE " + note, ha="center", va="top", fontsize=7, color="0.3",
                                transform=ax.transAxes)
                if i == 0:
                    ax.set_title(o, fontsize=8, color="0.3")
            axs[i, 0].text(-0.12, 0.5, ROW_LABEL[r], ha="right", va="center", fontsize=10,
                           transform=axs[i, 0].transAxes)
        fig.tight_layout(rect=(0.1, 0, 1, 0.96))
        for i in range(1, len(rows)):                             # thin rules between rows, as in the paper
            y = 0.5 * (axs[i - 1, 0].get_position().y0 + axs[i, 0].get_position().y1)
            fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], color="0.2", lw=0.6))
        fig.suptitle(PAIR_TITLE[pair], fontsize=9)
        out = os.path.join(args.out, "recon_3d_%s.png" % pair)
        fig.savefig(out, dpi=200, facecolor="white")
        plt.close(fig)
        print("saved", out)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for pair, pol in SETS:
        ap.add_argument("--%s-%s" % (pol, pair), default="models/shape_recon/%s_%s/seed1" % (pol, pair),
                        help="shape_recon.py output folder for the %s %s set" % (pair, pol))
    ap.add_argument("--none-from", choices=["baseline", "park"], default="baseline",
                    help="which set's joints-only CNN is shown as 'No touch' (default: the baseline set)")
    ap.add_argument("--objects", nargs="+", default=["col_006", "col_054", "col_110"],
                    help="held-out objects to show (default: the presentation's three; 'auto' = best, median, worst)")
    ap.add_argument("--n", type=int, default=3, help="number of objects when --objects is not given (paper: 3)")
    ap.add_argument("--out", default="deliverables/3_shape_reconstruction")
    ap.add_argument("--no-3d", action="store_true", help="skip the 3D figures")
    ap.add_argument("--renderer", choices=["isaac", "mpl"], default="isaac",
                    help="isaac: Isaac Gym camera renders (default); mpl: matplotlib fallback, no Isaac Gym")
    ap.add_argument("--px", type=int, default=480, help="render size per panel [px]")
    ap.add_argument("--fov", type=float, default=35.0, help="camera horizontal field of view [deg]")
    ap.add_argument("--cam-dist", type=float, default=0.20, help="camera distance from the object centre [m]")
    ap.add_argument("--cam-az", type=float, default=-60.0)
    ap.add_argument("--cam-el", type=float, default=30.0)
    args = ap.parse_args()

    reg = json.load(open(COLUMNS))
    labels = {n: np.asarray(r) for n, r in zip(reg["names"], reg["outline_radii_m"])}
    polys = {n: np.asarray(p) for n, p in zip(reg["names"], reg["polygon_xy_m"])}
    sets = {}
    for pair, pol in SETS:
        d = getattr(args, "%s_%s" % (pol, pair))
        sets[(pair, pol)] = load_set(d, SETS[(pair, pol)], labels)
        print("%-6s %-8s %s: %s" % (pair, pol, d, "found" if sets[(pair, pol)] is not None else "not available yet"))
    for pair in ("latest", "best"):
        d = getattr(args, "%s_%s" % (args.none_from, pair))
        sets[(pair, "none")] = load_set(d, "none", labels)
        print("%-6s %-8s %s: %s" % (pair, "no touch", d, "found" if sets[(pair, "none")] is not None else "not available yet"))
    if all(s is None for s in sets.values()):
        raise SystemExit("no results found")
    picks = pick_objects(sets, args.n) if args.objects == ["auto"] else args.objects
    test = set(reg["test"])
    assert all(o in test for o in picks), "only held-out (test) objects have reconstructions"
    print("objects:", " ".join(picks))
    os.makedirs(args.out, exist_ok=True)
    fig_outlines(picks, polys, sets, os.path.join(args.out, "recon_outlines.png"))
    if not args.no_3d:
        fig_3d(picks, polys, sets, float(reg["height_m"]), args)


if __name__ == "__main__":
    main()
