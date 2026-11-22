"""Dead-weight test of the Park skin on every sensor of the split robot, in Isaac Gym.

Like calibrating a sensor with weights: a flat plate of mass m rests on the whole pad, pressing with
its weight F = m g, i.e. a uniform pressure p = m g / A_pad. The plate's mass is increased in steps
and each sensor's simulated reading dR/R0 is compared with Park's Eq. 4, g(p) = 1 / (1 - p / p*) - 1.

Setup: 16 hands in parallel, one per sensor, all in the same flat pose (unrotated base as in the
task), joints locked straight by their limits, hand gravity off and self-collision off (as in the
task). In hand i, a plate covering 90% of pad i (2 mm thick) starts 0.5 mm above its face, gravity
on. The 13 pads that face up are tested; the 3 thumb pads face sideways and cannot hold a plate.
For each weight: settle SETTLE steps, average the cells' compressive normal forces over the last
AVG steps, convert them to Park readings with park_skin.py (nominal p*, no calibration noise).
Pressures are 5 ... 80 kPa for Park's EcoFlex skin and scale with E, up to MAX_PLATE_KG (heavier
plates slide on the finger pads).

Outputs (weight_test/): park_weight_pads.png (4 x 4), park_weight_all.png (one chart),
weight_results.npz; prints pad force vs plate weight and how far each plate moved.

Usage (from the repo root):
  python weight_test_park.py                  # Dragon Skin 20 (the skin used for training)
  python weight_test_park.py --E 69e3         # Park's EcoFlex 00-30
  python weight_test_park.py --view --env 3   # watch sensor 3 being loaded
"""
import argparse
import importlib.util
import os
import xml.etree.ElementTree as ET

from isaacgym import gymapi, gymtorch           # before torch
from isaacgym.torch_utils import quat_apply
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")
SPLIT = "urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_split4.urdf"
ORIG = os.path.join(ASSETS, "urdf/xarm6/xarm6_allegro_right_fsr_2023_thin.urdf")   # unsplit pad geometry
PADS = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr", "link_5.0_fsr",
        "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr", "link_10.0_fsr",
        "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
        "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]
P_LEVELS_KPA = np.arange(5, 80.1, 5.0)
SETTLE, AVG = 90, 30
PLATE_T = 0.002
MAX_PLATE_KG = 8.0
G = 9.81


def load_park():
    spec = importlib.util.spec_from_file_location("park_skin", os.path.join(ROOT, "isaacgymenvs/utils/park_skin.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def pad_geometry():
    links = {l.get("name"): l for l in ET.parse(ORIG).getroot().findall("link")}
    out = {}
    for p in PADS:
        col = links[p].find("collision")
        out[p] = (np.array([float(v) for v in col.find("geometry").find("box").get("size").split()]),
                  np.array([float(v) for v in col.find("origin").get("xyz").split()]))
    return out


def make_sim(gym, gravity=True):
    sp = gymapi.SimParams()
    sp.dt = 1 / 60; sp.substeps = 2; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -G if gravity else 0.0)
    sp.use_gpu_pipeline = True
    sp.physx.use_gpu = True; sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 8; sp.physx.num_velocity_iterations = 0
    sp.physx.contact_offset = 0.002; sp.physx.rest_offset = 0.0
    sp.physx.contact_collection = gymapi.ContactCollection(2)
    sp.physx.max_gpu_contact_pairs = 8388608; sp.physx.default_buffer_size_multiplier = 5.0
    return gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)


def load_robot(gym, sim):
    o = gymapi.AssetOptions()                     # same options as the task
    o.fix_base_link = True; o.collapse_fixed_joints = False; o.disable_gravity = True
    o.thickness = 0.001; o.angular_damping = 0.01; o.use_physx_armature = True
    o.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    robot = gym.load_asset(sim, ASSETS, SPLIT, o)
    dof = gym.get_asset_dof_properties(robot)     # joint properties as in the task, locked by limits
    q0 = np.clip(np.zeros(len(dof)), dof["lower"], dof["upper"]).astype(np.float32)
    dof["driveMode"].fill(int(gymapi.DOF_MODE_EFFORT))
    dof["velocity"][:6] = 1.0; dof["velocity"][6:] = 3.14
    dof["effort"].fill(20.0); dof["friction"].fill(0.1); dof["stiffness"].fill(0.0); dof["armature"].fill(0.1)
    dof["damping"][:6] = 100.0; dof["damping"][6:] = 0.0
    dof["lower"] = q0; dof["upper"] = q0 + 1e-5
    return robot, dof, q0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--env", type=int, default=0, help="sensor index (0-15) to watch with --view")
    ap.add_argument("--E", type=float, default=338e3,
                    help="elastomer modulus, Pa (default Dragon Skin 20, 338e3; Park's EcoFlex 00-30: 69e3)")
    args = ap.parse_args()
    global P_LEVELS_KPA
    P_LEVELS_KPA = P_LEVELS_KPA * args.E / 69e3
    PMAX = 80 * args.E / 69e3
    park_lines = abs(args.E - 69e3) < 1                           # Park's measured range only applies to EcoFlex 00-30

    gym = gymapi.acquire_gym()

    sim = make_sim(gym, gravity=True)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties()) if args.view else None

    def step():
        gym.simulate(sim); gym.fetch_results(sim, True)
        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                raise SystemExit("viewer closed")
            gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)

    pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, pp)                                        # floor at z = 0
    robot, dof, q0 = load_robot(gym, sim)
    n_rb = gym.get_asset_rigid_body_count(robot)
    geom = pad_geometry()
    po = gymapi.AssetOptions(); po.density = 1000.0               # mass is set per weight level below
    plates = [gym.create_box(sim, PLATE_T, 0.9 * geom[p][0][1], 0.9 * geom[p][0][2], po) for p in PADS]

    envs, robot_h, plate_h = [], [], []
    for i in range(16):
        env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0), gymapi.Vec3(1.0, 1.0, 1), 4)
        # every hand in the same flat pose (unrotated base, as in the task); env i tests pad i
        r = gym.create_actor(env, robot, gymapi.Transform(), "robot", i, 1)   # filter 1: no hand self-collision
        gym.set_actor_dof_properties(env, r, dof)
        b = gym.create_actor(env, plates[i], gymapi.Transform(p=gymapi.Vec3(0, 0, 3.0)), "plate", i, 0)
        envs.append(env); robot_h.append(r); plate_h.append(b)
    pad_rb = [gym.find_actor_rigid_body_handle(envs[0], robot_h[0], p) for p in PADS]
    cell_rb = [[gym.find_actor_rigid_body_handle(envs[0], robot_h[0], p if k == 0 else f"{p}_c{k}") for k in range(4)]
               for p in PADS]
    body_names = list(gym.get_actor_rigid_body_names(envs[0], robot_h[0])) + ["plate"]

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

    # place each plate 0.5 mm above its pad's outer face, aligned with the pad (which now faces up)
    q = torch.stack([rb[i, pad_rb[i], 3:7] for i in range(16)])
    pos = torch.stack([rb[i, pad_rb[i], 0:3] for i in range(16)])
    normal = quat_apply(q, torch.tensor([[1.0, 0, 0]], device=dev).repeat(16, 1))
    tested = (normal[:, 2] > 0.9).cpu().numpy()                    # pads facing up can hold a weight
    print("[setup] pads facing up (tested): " + ", ".join(p for p, t in zip(PADS, tested) if t))
    print("[setup] pads facing sideways (not tested): " + (", ".join(p for p, t in zip(PADS, tested) if not t) or "none"))
    local = torch.as_tensor(np.array([geom[p][1] + np.array([geom[p][0][0] / 2 + PLATE_T / 2 + 0.0005, 0, 0])
                                      for p in PADS]), dtype=torch.float32, device=dev)
    face_center = pos + quat_apply(q, torch.as_tensor(np.array([geom[p][1] + np.array([geom[p][0][0] / 2, 0, 0])
                                                                for p in PADS]), dtype=torch.float32, device=dev))
    plate_ids = torch.arange(1, 32, 2, dtype=torch.int32, device=dev)
    plate_pos, plate_rot = pos + quat_apply(q, local), q.clone()

    def place_plates():
        """Put every plate 0.5 mm above its pad, at rest. Also needed after a mass change: on the GPU
        pipeline, the step after set_actor_rigid_body_properties sends the actor back to its creation pose."""
        gym.refresh_actor_root_state_tensor(sim)
        root[:, 1, 0:3] = plate_pos
        root[:, 1, 3:7] = plate_rot
        root[:, 1, 7:13] = 0
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root.view(-1, 13)),
                                                gymtorch.unwrap_tensor(plate_ids), len(plate_ids))

    place_plates()
    for _ in range(20):
        step()

    if viewer is not None:
        e = args.env
        org = gym.get_env_origin(envs[e])
        tgt = face_center[e].cpu().numpy() - np.array([org.x, org.y, org.z])
        cam = tgt + np.array([0.12, -0.12, 0.10])
        gym.viewer_camera_look_at(viewer, envs[e], gymapi.Vec3(*cam.tolist()), gymapi.Vec3(*tgt.tolist()))
        print(f"[view] watching sensor {e}: {PADS[e]}")

    park = load_park()
    cell_names = [p if k == 0 else f"{p}_c{k}" for p in PADS for k in range(4)]
    skin = park.ParkSkin(park.cell_areas_from_urdf(os.path.join(ASSETS, SPLIT), cell_names), n_cells=4, E=args.E, device=dev)
    print(f"[setup] skin E = p* = {skin.p_star / 1e3:.0f} kPa; plate pressures {P_LEVELS_KPA[0]:.0f}-{P_LEVELS_KPA[-1]:.0f} kPa")
    pad_area = np.array([skin.area[4 * i:4 * i + 4].sum().item() for i in range(16)])
    keep = P_LEVELS_KPA * 1e3 * pad_area[tested].max() / G <= MAX_PLATE_KG
    if not keep.all():
        print(f"[setup] plates limited to {MAX_PLATE_KG:.0f} kg: pressures up to {P_LEVELS_KPA[keep][-1]:.0f} kPa "
              f"({P_LEVELS_KPA[keep][-1] * 1e3 / skin.p_star:.2f} p*)")
    P_LEVELS_KPA = P_LEVELS_KPA[keep]
    all_cells = torch.as_tensor(sum(cell_rb, []), device=dev)
    ex = torch.tensor([[1.0, 0, 0]], device=dev)

    rec_mass, rec_weight, rec_contact, rec_reading, rec_slide, rec_cells = [], [], [], [], [], []
    for p_kpa in P_LEVELS_KPA:
        masses = p_kpa * 1e3 * pad_area / G                           # kg, per sensor
        for i in range(16):
            props = gym.get_actor_rigid_body_properties(envs[i], plate_h[i])
            props[0].mass = float(masses[i])
            gym.set_actor_rigid_body_properties(envs[i], plate_h[i], props, True)
        step()            # the new mass is applied in this step, which also resets the plate to its spawn pose
        place_plates()    # so put the plates back on their pads only after it
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
            raise SystemExit(f"NaN at {p_kpa:.0f} kPa")
        reading = (skin.pad_readings(fn) * skin.g_max).cpu().numpy()            # (16, 16) dR/R0
        cell_r = (skin.cell_response(fn) / skin.g_max).cpu().numpy()
        contact = fn.view(16, 16, 4).sum(-1).cpu().numpy()
        gym.refresh_rigid_body_state_tensor(sim)
        slide = torch.norm(rb[:, n_rb, 0:3] - (face_center + normal * PLATE_T / 2), dim=-1).cpu().numpy() * 1e3
        rec_mass.append(masses); rec_weight.append(masses * G)
        rec_contact.append(np.array([contact[i, i] for i in range(16)]))
        rec_reading.append(np.array([reading[i, i] for i in range(16)]))
        rec_slide.append(slide)
        fn_np = fn.cpu().numpy()
        rec_cells.append(np.stack([fn_np[i, 4 * i:4 * i + 4] for i in range(16)]))   # (16, 4) N per cell
        if viewer is not None:
            for i in range(16):
                for k in range(4):
                    r = float(cell_r[i, 4 * i + k])
                    gym.set_rigid_body_color(envs[i], robot_h[i], cell_rb[i][k], gymapi.MESH_VISUAL, gymapi.Vec3(1 - r, r, 0))
        ratio = (rec_contact[-1] / rec_weight[-1])[tested]
        print(f"{p_kpa:4.0f} kPa: weights {masses.min():.2f}-{masses.max():.2f} kg; contact/weight "
              f"min {ratio.min():.2f} median {np.median(ratio):.2f} max {ratio.max():.2f}; plate offset max {slide[tested].max():.1f} mm")

    M, W, C, R, S = map(np.array, (rec_mass, rec_weight, rec_contact, rec_reading, rec_slide))   # (L, 16)
    out = os.path.join(ROOT, "weight_test"); os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "weight_results.npz"), p_kpa=P_LEVELS_KPA, mass=M, weight=W, contact=C,
             reading=R, slide_mm=S, pads=np.array(PADS), pad_area=pad_area, cell_fn=np.array(rec_cells))

    k50 = min(9, len(P_LEVELS_KPA) - 1)                           # 50 kPa for EcoFlex, scaled with E
    print(f"\nsensor              weight at {P_LEVELS_KPA[k50]:.0f} kPa   contact/weight there   "
          f"plate offset at {P_LEVELS_KPA[-1]:.0f} kPa")
    for i, p in enumerate(PADS):
        if not tested[i]:
            print(f"  {p:18s} not tested (faces sideways)"); continue
        print(f"  {p:18s} {M[k50, i]:6.2f} kg          {C[k50, i] / W[k50, i]:6.2f}                  {S[-1, i]:5.1f} mm")

    # where each sensor's load goes among its 4 cells, and what the reading would be if that same
    # measured force were spread evenly (separates "force lost" from "force unevenly split")
    CF = np.array(rec_cells)                                                      # (L, 16, 4)
    areas = np.array([skin.area[4 * i:4 * i + 4].cpu().numpy() for i in range(16)])
    def g(p_pa):
        return 1 / (1 - np.minimum(p_pa / skin.p_star, skin.alpha)) - 1
    for k in sorted({min(9, len(P_LEVELS_KPA) - 1), len(P_LEVELS_KPA) - 1}):   # 50 kPa (EcoFlex) and the top level
        kp = P_LEVELS_KPA[k]
        print(f"\ncell loads at {kp:.0f} kPa: share of pad force per cell (cell 0..3), cell pressure (kPa), "
              f"reading if the measured force were even, actual reading, Eq. 4 at nominal")
        for i, p in enumerate(PADS):
            if not tested[i]:
                continue
            f = CF[k, i]; share = f / max(f.sum(), 1e-9); pc = f / areas[i] / 1e3
            even = g(f.sum() / areas[i].sum())
            print(f"  {p:18s} share {np.round(share, 2)}  p {np.round(pc, 0)}  even {even:4.2f}  "
                  f"actual {R[k, i]:4.2f}  nominal {g(kp * 1e3):4.2f}")

    p_fine = np.linspace(0, PMAX, 400)
    curve = 1 / (1 - np.minimum(p_fine * 1e3 / skin.p_star, skin.alpha)) - 1

    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axs.ravel()):
        ax.plot(p_fine, curve, color="0.75", lw=4, zorder=1)
        if tested[i]:
            ax.scatter(P_LEVELS_KPA, R[:, i], s=16, color="tab:blue", zorder=3)
        else:
            ax.text(PMAX / 2, skin.g_max * 0.5, "not tested\n(faces sideways)", ha="center", va="center", fontsize=8)
        for kpa, c in (((15, "g"), (50, "orange")) if park_lines else ()):
            ax.axvline(kpa, color=c, lw=0.8, ls="--")
        ax.set_box_aspect(1); ax.set_xlim(0, PMAX); ax.set_ylim(0, skin.g_max * 1.08)
        ax.set_title(f"{i}: {PADS[i]} (max {M[-1, i]:.1f} kg)", fontsize=8); ax.tick_params(labelsize=7)
        if i % 4 == 0: ax.set_ylabel("dR/R0", fontsize=8)
        if i >= 12: ax.set_xlabel("weight / pad area (kPa)", fontsize=8)
    fig.suptitle("Dead-weight test: simulated sensor reading (points) vs Park Eq. 4 (grey)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "park_weight_pads.png"), dpi=110)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(p_fine, curve, color="0.75", lw=5, zorder=1, label="Park Eq. 4")
    for j, i in enumerate(np.where(tested)[0]):
        ax.scatter(P_LEVELS_KPA, R[:, i], s=10, color="tab:blue", zorder=3,
                   label=f"{int(tested.sum())} upward-facing sensors" if j == 0 else None)
    for kpa, c, lab in (((15, "g", "15 kPa (min. detectable)"), (50, "orange", "50 kPa (max. tested)")) if park_lines else ()):
        ax.axvline(kpa, color=c, lw=0.8, ls="--", label=lab)
    ax.set_box_aspect(1); ax.set_xlim(0, PMAX); ax.set_ylim(0, skin.g_max * 1.08)
    ax.set_xlabel("weight / pad area (kPa)"); ax.set_ylabel("dR/R0"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Dead-weight test, upward-facing sensors", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "park_weight_all.png"), dpi=130)
    print(f"\nsaved park_weight_pads.png, park_weight_all.png, weight_results.npz in {out}")


if __name__ == "__main__":
    main()
