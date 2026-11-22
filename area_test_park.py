"""Same force, different contact area: Park readings with the load on the whole pad vs on one cell.

For every upward-facing sensor, a plate of weight F rests on
  "all" : the whole pad (plate covers 90% of the pad, load spread over the 4 cells), or
  "c0".."c3": only one cell (plate covers 90% of that cell, about a quarter of the pad).
Cell order (pad frame, y and z in-plane): c0 (-y,-z), c1 (+y,-z), c2 (-y,+z), c3 (+y,+z).
Same F in all five cases. The weights are 1 ... 20 N for Park's EcoFlex skin and scale with E,
up to MAX_PLATE_KG (heavier plates slide on the finger pads).

Park model (cells in series, pad readout capped at g_max = 9), g(p) = 1 / (1 - p / p*) - 1:
  whole pad:   min(g(F / A_pad), 9)
  one cell:    min(g(F / A_cell) / 4, 9)
g is convex with g(0) = 0, so the same force on a smaller area never reads lower, and a closed
cell saturates the pad.

Each case runs in its own simulation of 16 hands (flat, as in weight_test_park.py): with more hands
in one simulation the reported finger-pad forces became unreliable (up to 20 x the plate weight).

Outputs (area_test/): park_area_pads.png (4 x 4), park_area_all.png, area_results.npz.
Usage (from the repo root):
  python area_test_park.py                       # Dragon Skin 20 (the skin used for training)
  python area_test_park.py --E 69e3              # Park's EcoFlex 00-30
  python area_test_park.py --view --sensor 3 --case c1
"""
import argparse
import os
import subprocess
import sys

import numpy as np

CASES = ["all", "c0", "c1", "c2", "c3"]
F_BASE = np.array([1, 2, 3, 4, 6, 8, 10, 12, 15, 20], dtype=float)   # N, for E = 69 kPa
MAX_PLATE_KG = 8.0
G = 9.81
GAP = 0.0002                                                          # cell gap, as in the split URDF
COLORS = {"all": "black", "c0": "tab:red", "c1": "tab:orange", "c2": "tab:green", "c3": "tab:blue"}
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "area_test")


def force_levels(E):
    f = F_BASE * E / 69e3
    return f[f / G <= MAX_PLATE_KG]


def run_case(case, E, view=False, sensor=3):
    """Simulate one case on 16 hands; returns readings (L, 16), cell normal forces (L, 16, 4), tested mask."""
    from isaacgym import gymapi, gymtorch           # before torch
    from isaacgym.torch_utils import quat_apply
    import torch
    from weight_test_park import (ASSETS, SPLIT, PADS, PLATE_T, SETTLE, AVG,
                                  load_park, pad_geometry, make_sim, load_robot)

    levels = force_levels(E)
    gym = gymapi.acquire_gym()
    sim = make_sim(gym, gravity=True)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties()) if view else None

    def step():
        gym.simulate(sim); gym.fetch_results(sim, True)
        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                raise SystemExit("viewer closed")
            gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

    pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, pp)
    robot, dof, q0 = load_robot(gym, sim)
    n_rb = gym.get_asset_rigid_body_count(robot)
    geom = pad_geometry()
    po = gymapi.AssetOptions(); po.density = 1000.0

    sizes, offsets = [], []
    for p in PADS:
        (sx, sy, sz), org = geom[p]
        lift = sx / 2 + PLATE_T / 2 + 0.0005                        # plate starts 0.5 mm above the face
        if case == "all":
            sizes.append((0.9 * sy, 0.9 * sz)); offsets.append(org + np.array([lift, 0, 0]))
        else:
            k = int(case[1]); dy = -1 if k in (0, 2) else 1; dz = -1 if k in (0, 1) else 1
            sizes.append((0.9 * (sy / 2 - GAP), 0.9 * (sz / 2 - GAP)))
            offsets.append(org + np.array([lift, dy * sy / 4, dz * sz / 4]))

    envs, robot_h, plate_h = [], [], []
    for i in range(16):
        env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0), gymapi.Vec3(1.0, 1.0, 1), 4)
        r = gym.create_actor(env, robot, gymapi.Transform(), "robot", i, 1)   # filter 1: no self-collision
        gym.set_actor_dof_properties(env, r, dof)
        b = gym.create_actor(env, gym.create_box(sim, PLATE_T, *sizes[i], po),
                             gymapi.Transform(p=gymapi.Vec3(0, 0, 3.0)), "plate", i, 0)
        envs.append(env); robot_h.append(r); plate_h.append(b)
    pad_rb = [gym.find_actor_rigid_body_handle(envs[0], robot_h[0], p) for p in PADS]
    cell_rb = [[gym.find_actor_rigid_body_handle(envs[0], robot_h[0], p if k == 0 else f"{p}_c{k}") for k in range(4)]
               for p in PADS]

    gym.prepare_sim(sim)
    dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(16, -1, 2)
    root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(16, 2, 13)
    rb = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(16, n_rb + 1, 13)
    cf = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(16, n_rb + 1, 3)
    dev = dof_state.device

    dof_state[..., 0] = torch.as_tensor(q0, device=dev); dof_state[..., 1] = 0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
    for _ in range(30):
        step()
    gym.refresh_rigid_body_state_tensor(sim)
    if not torch.isfinite(rb).all():
        raise SystemExit("NaN after settling the hands")

    q = torch.stack([rb[i, pad_rb[i], 3:7] for i in range(16)])
    pos = torch.stack([rb[i, pad_rb[i], 0:3] for i in range(16)])
    normal = quat_apply(q, torch.tensor([[1.0, 0, 0]], device=dev).repeat(16, 1))
    tested = (normal[:, 2] > 0.9).cpu().numpy()                     # pads facing up can hold a plate
    local = torch.as_tensor(np.array(offsets), dtype=torch.float32, device=dev)
    plate_pos, plate_rot = pos + quat_apply(q, local), q.clone()
    plate_ids = torch.arange(1, 32, 2, dtype=torch.int32, device=dev)

    def place_plates():
        gym.refresh_actor_root_state_tensor(sim)
        root[:, 1, 0:3] = plate_pos; root[:, 1, 3:7] = plate_rot; root[:, 1, 7:13] = 0
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root.view(-1, 13)),
                                                gymtorch.unwrap_tensor(plate_ids), len(plate_ids))

    place_plates()
    for _ in range(20):
        step()
    if viewer is not None:
        org = gym.get_env_origin(envs[sensor])
        tgt = plate_pos[sensor].cpu().numpy() - np.array([org.x, org.y, org.z])
        gym.viewer_camera_look_at(viewer, envs[sensor], gymapi.Vec3(*(tgt + [0.10, -0.10, 0.08]).tolist()),
                                  gymapi.Vec3(*tgt.tolist()))
        print(f"[view] watching sensor {sensor} ({PADS[sensor]}), case {case}")

    park = load_park()
    cell_names = [p if k == 0 else f"{p}_c{k}" for p in PADS for k in range(4)]
    skin = park.ParkSkin(park.cell_areas_from_urdf(os.path.join(ASSETS, SPLIT), cell_names), n_cells=4, E=E, device=dev)
    all_cells = torch.as_tensor(sum(cell_rb, []), device=dev)
    ex = torch.tensor([[1.0, 0, 0]], device=dev)
    own = torch.stack([torch.arange(4 * i, 4 * i + 4, device=dev) for i in range(16)])

    R, CF = [], []
    for F in levels:
        for i in range(16):
            props = gym.get_actor_rigid_body_properties(envs[i], plate_h[i])
            props[0].mass = float(F / G)
            gym.set_actor_rigid_body_properties(envs[i], plate_h[i], props, True)
        step()            # the new mass applies in this step, which also sends the plate to its spawn pose
        place_plates()    # so re-place it after that step
        acc = torch.zeros(16, 64, device=dev)
        for s in range(SETTLE):
            step()
            if s >= SETTLE - AVG:
                gym.refresh_net_contact_force_tensor(sim); gym.refresh_rigid_body_state_tensor(sim)
                cells = cf[:, all_cells, :]
                n_c = quat_apply(rb[:, all_cells, 3:7].reshape(-1, 4), ex.repeat(16 * 64, 1)).view(16, 64, 3)
                acc += torch.clamp(-(cells * n_c).sum(-1), min=0.0)
        fn = acc / AVG
        if not torch.isfinite(fn).all():
            raise SystemExit(f"NaN at {F:.1f} N")
        R.append((skin.pad_readings(fn) * skin.g_max).diagonal().cpu().numpy())        # pad i in env i
        CF.append(torch.gather(fn, 1, own).cpu().numpy())
        ratio = CF[-1].sum(-1)[tested] / F
        print(f"[{case}] {F:6.1f} N: contact/weight median {np.median(ratio):.2f} (min {ratio.min():.2f}, max {ratio.max():.2f})",
              flush=True)
    gym.destroy_sim(sim)
    return np.array(R), np.array(CF), tested


def report(E, levels, R, CF, tested):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from weight_test_park import ASSETS, SPLIT, PADS, load_park       # imports isaacgym first (before torch)

    park = load_park()
    cell_names = [p if k == 0 else f"{p}_c{k}" for p in PADS for k in range(4)]
    skin = park.ParkSkin(park.cell_areas_from_urdf(os.path.join(ASSETS, SPLIT), cell_names), n_cells=4, E=E)
    areas = skin.area.view(16, 4).numpy()

    def g_whole(F, A):                              # uniform load over the pad
        return 1 / (1 - np.minimum(F / A.sum() / skin.p_star, skin.alpha)) - 1

    def g_one(F, A):                                # load on one (average) cell
        return np.minimum((1 / (1 - np.minimum(F / A.mean() / skin.p_star, skin.alpha_cell)) - 1) / 4, skin.g_max)

    print(f"\nskin E = p* = {skin.p_star / 1e3:.0f} kPa; weights {levels[0]:.1f}-{levels[-1]:.1f} N")
    ratio = CF.sum(-1) / levels[:, None, None]                                       # (L, case, sensor)
    share = np.stack([CF[:, 1 + c, :, c] / np.maximum(CF[:, 1 + c].sum(-1), 1e-9) for c in range(4)], 1)  # (L, 4, 16)
    t = tested
    print(f"contact/weight over all levels, sensors and cases: median {np.median(ratio[:, :, t]):.2f}, "
          f"5-95% {np.percentile(ratio[:, :, t], 5):.2f}-{np.percentile(ratio[:, :, t], 95):.2f}")
    print(f"one-cell plates, share of the force on the target cell: median {np.median(share[:, :, t]):.2f}, "
          f"min {share[:, :, t].min():.2f}")
    # a comparison counts only if both plates actually delivered their weight to the intended area
    ok_whole = (ratio[:, 0] >= 0.8) & (ratio[:, 0] <= 1.2)                          # (L, sensor)
    ok_one = (ratio[:, 1:] >= 0.8) & (ratio[:, 1:] <= 1.2) & (share >= 0.8)         # (L, 4, sensor)
    valid = ok_one & ok_whole[:, None] & t[None, None]
    one_ge = R[:, 1:] >= R[:, :1] - 1e-6                                            # (L, 4, sensor)
    n_all = int(t.sum()) * 4 * len(levels)
    print(f"plates that delivered 0.8-1.2 x their weight (one-cell: at least 80% of it on the target cell): "
          f"whole pad {ok_whole[:, t].mean():.0%}, one cell {ok_one[:, :, t].mean():.0%}")
    print(f"same force, one cell vs whole pad: one cell reads at least as high in {one_ge[valid].mean():.0%} of the "
          f"{int(valid.sum())} valid comparisons (all {n_all}: {one_ge[:, :, t].mean():.0%}; model: 100%)")
    bad = np.argwhere(valid & ~one_ge)
    for k, c, i in bad[:10]:
        print(f"  lower: {PADS[i]:18s} c{c} at {levels[k]:5.1f} N: one cell {R[k, 1 + c, i]:.2f} vs whole {R[k, 0, i]:.2f}")

    for k in (len(levels) // 3, 2 * len(levels) // 3, len(levels) - 1):
        F0 = levels[k]
        print(f"\nat {F0:.1f} N: reading whole pad | one cell c0 c1 c2 c3 | model whole / one cell | "
              f"share on target cell c0..c3")
        for i, p in enumerate(PADS):
            if t[i]:
                print(f"  {p:18s} {R[k, 0, i]:5.2f} | " + " ".join(f"{R[k, 1 + c, i]:5.2f}" for c in range(4)) +
                      f" | {g_whole(F0, areas[i]):5.2f} / {g_one(F0, areas[i]):5.2f} | {np.round(share[k, :, i], 2)}")

    f_fine = np.linspace(0, levels.max(), 400)

    def panel(ax, i, small):
        A = areas[i]
        ax.plot(f_fine, g_whole(f_fine, A), color="0.6", lw=3 if small else 4, zorder=1,
                label=None if small else "model, whole pad")
        ax.plot(f_fine, g_one(f_fine, A), color="0.6", lw=3 if small else 4, ls=":", zorder=1,
                label=None if small else "model, one cell")
        for c, case in enumerate(CASES):
            ax.scatter(levels, R[:, c, i], s=12 if small else 16, color=COLORS[case], zorder=3,
                       marker="s" if case == "all" else "o",
                       label=None if small else ("whole pad" if case == "all" else f"only {case}"))
        ax.set_box_aspect(1); ax.set_xlim(0, levels.max() * 1.03); ax.set_ylim(0, skin.g_max * 1.08)

    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axs.ravel()):
        if t[i]:
            panel(ax, i, small=True)
        else:
            ax.set_box_aspect(1); ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "not tested\n(faces sideways)", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(f"{i}: {PADS[i]} (pad {areas[i].sum() * 1e6:.0f} mm2)", fontsize=8); ax.tick_params(labelsize=7)
        if i % 4 == 0: ax.set_ylabel("dR/R0", fontsize=8)
        if i >= 12: ax.set_xlabel("plate weight (N)", fontsize=8)
    fig.suptitle(f"Same weight, different area (E = {E / 1e3:.0f} kPa): whole pad (black squares) vs one cell "
                 "(red c0, orange c1, green c2, blue c3)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "park_area_pads.png"), dpi=110)

    big = int(np.argmax([areas[i].sum() if t[i] else 0 for i in range(16)]))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    panel(ax, big, small=False)
    ax.set_xlabel("plate weight (N)"); ax.set_ylabel("dR/R0"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Same weight, different area: {PADS[big]} (E = {E / 1e3:.0f} kPa)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "park_area_all.png"), dpi=130)
    np.savez(os.path.join(OUT, "area_results.npz"), E=E, force=levels, cases=np.array(CASES), pads=np.array(PADS),
             reading=R, cell_fn=CF, cell_area=areas, tested=tested)
    print(f"\nsaved park_area_pads.png, park_area_all.png, area_results.npz in {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=float, default=338e3,
                    help="elastomer modulus, Pa (default Dragon Skin 20, 338e3; Park's EcoFlex 00-30: 69e3)")
    ap.add_argument("--view", action="store_true", help="watch one case of one sensor (no report)")
    ap.add_argument("--sensor", type=int, default=3, help="sensor index (0-15) to watch with --view")
    ap.add_argument("--case", default="c0", choices=CASES, help="case to watch with --view")
    ap.add_argument("--report-only", action="store_true", help="redo the printout and charts from area_results.npz")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)          # internal: run one case, save it
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.report_only:
        d = np.load(os.path.join(OUT, "area_results.npz"))
        report(float(d["E"]), d["force"], d["reading"], d["cell_fn"], d["tested"])
        return
    if args.view:
        run_case(args.case, args.E, view=True, sensor=args.sensor)
        return
    if args._child:
        R, CF, tested = run_case(args._child, args.E)
        np.savez(os.path.join(OUT, f"part_{args._child}.npz"), R=R, CF=CF, tested=tested)
        return

    for case in CASES:                               # one fresh simulation (process) per case
        subprocess.run([sys.executable, os.path.abspath(__file__), "--E", str(args.E), "--_child", case], check=True)
    parts = [np.load(os.path.join(OUT, f"part_{c}.npz")) for c in CASES]
    R = np.stack([d["R"] for d in parts], 1)          # (L, case, sensor)
    CF = np.stack([d["CF"] for d in parts], 1)        # (L, case, sensor, 4)
    for c in CASES:
        os.remove(os.path.join(OUT, f"part_{c}.npz"))
    report(args.E, force_levels(args.E), R, CF, parts[0]["tested"])


if __name__ == "__main__":
    main()
