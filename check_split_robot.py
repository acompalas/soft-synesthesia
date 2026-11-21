"""Stage-1 check for the split-pad robot: does it load and simulate in Isaac Gym, and is the
pad mass unchanged? Loads the original and the split URDF with the task's asset options,
creates one env each on the GPU pipeline, steps it, and prints body counts and per-pad masses.

Usage (from the repo root):
  python check_split_robot.py --split assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_split4.urdf
"""
import argparse
import os

from isaacgym import gymapi, gymtorch  # must be imported before torch
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ORIG = "urdf/xarm6/xarm6_allegro_right_fsr_2023_thin.urdf"
PADS = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr", "link_5.0_fsr",
        "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr", "link_10.0_fsr",
        "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
        "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]


def make_sim(gym):
    sp = gymapi.SimParams()
    sp.dt = 1 / 60
    sp.substeps = 2
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81)
    sp.use_gpu_pipeline = True
    sp.physx.use_gpu = True
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 8
    sp.physx.contact_collection = gymapi.ContactCollection(2)
    return gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)


def load(gym, sim, rel):
    o = gymapi.AssetOptions()  # same options as the task (allegro_arm_morb_axis.py ~539-548)
    o.flip_visual_attachments = False
    o.fix_base_link = True
    o.collapse_fixed_joints = False
    o.disable_gravity = True
    o.thickness = 0.001
    o.angular_damping = 0.01
    o.use_physx_armature = True
    o.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    return gym.load_asset(sim, ROOT, rel, o)


def pad_masses(names, masses, n_cells):
    by_name = dict(zip(names, masses))
    out = []
    for p in PADS:
        parts = [p] + [f"{p}_c{k}" for k in range(1, n_cells)]
        out.append(sum(by_name[q] for q in parts if q in by_name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="path of the split URDF")
    args = ap.parse_args()
    split_rel = os.path.relpath(os.path.abspath(args.split), ROOT)
    n_cells = 4 if "split4" in split_rel else 2

    gym = gymapi.acquire_gym()
    sim = make_sim(gym)
    gym.add_ground(sim, gymapi.PlaneParams())
    results = {}
    for label, rel in (("orig", ORIG), ("split", split_rel)):
        asset = load(gym, sim, rel)
        n_bodies = gym.get_asset_rigid_body_count(asset)
        n_shapes = gym.get_asset_rigid_shape_count(asset)
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 2)
        actor = gym.create_actor(env, asset, gymapi.Transform(), label, 0 if label == "orig" else 1, 0)
        names = gym.get_actor_rigid_body_names(env, actor)
        masses = [p.mass for p in gym.get_actor_rigid_body_properties(env, actor)]
        results[label] = (n_bodies, n_shapes, pad_masses(names, masses, n_cells if label == "split" else 1),
                          sum(masses), list(gym.get_asset_dof_names(asset)), list(names))
        print(f"[{label}] {rel}: {n_bodies} rigid bodies, {n_shapes} shapes, total mass {sum(masses):.4f} kg")

    gym.prepare_sim(sim)
    for _ in range(10):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    gym.refresh_net_contact_force_tensor(sim)
    cf = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
    print(f"simulated 10 steps on the GPU pipeline; net contact force tensor shape {tuple(cf.shape)}")

    print("\nper-pad mass (original pad vs sum of its cells), grams:")
    for p, mo, ms in zip(PADS, results["orig"][2], results["split"][2]):
        flag = "" if abs(mo - ms) < 0.06 * mo else "   <-- MISMATCH"  # cells are shrunk by the gap (~4% area)
        print(f"  {p:20s} {1e3 * mo:8.3f} {1e3 * ms:8.3f}{flag}")
    print(f"total robot mass: orig {results['orig'][3]:.4f} kg, split {results['split'][3]:.4f} kg")

    dof_o, dof_s = results["orig"][4], results["split"][4]
    print(f"\nDOF order identical: {dof_o == dof_s}  ({len(dof_o)} vs {len(dof_s)} DOFs)")
    if dof_o != dof_s:
        for i, (a, b) in enumerate(zip(dof_o, dof_s)):
            print(f"  {i:2d} {a:20s} {b:20s} {'' if a == b else '<--'}")
    body_o = results["orig"][5]
    body_s = [n for n in results["split"][5] if n in set(body_o)]
    print(f"original bodies keep their relative order in the split robot: {body_o == body_s}")
    print("split robot body order (first 60):", results["split"][5][:60])


if __name__ == "__main__":
    main()
