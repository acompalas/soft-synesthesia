"""Split each of the 16 FSR pads of the robot URDF into cells, each its own rigid body.

Every pad is a thin box (thin along its link's x axis) on a fixed joint. For each pad:
  * the original pad link keeps its name and becomes cell 0 (so all existing code that
    looks up pad links by name, e.g. fingertip positions, still finds the same link frame);
  * cells 1..n-1 are new links "<pad>_c1", "<pad>_c2", ... attached to the pad's PARENT (finger)
    link by fixed joints identical to the pad's joint, so every cell frame has the pad's position
    and orientation (the pad normal is each cell's local x axis). PhysX never collides links that
    share a joint, so, like the original pad, no cell collides with its own finger link;
  * each cell is a box of the pad's thickness covering 1/n of the pad face, placed by its
    visual/collision origin, shrunk by --gap (default 0.2 mm) in each in-plane direction so that
    neighbouring cells never touch each other (they are rigidly fixed, so the gap never closes).

--cells 4 splits both in-plane axes in half (2x2). --cells 2 splits only the longer in-plane
axis (ties: z). Cell order for 4 cells: c0 (-y,-z), c1 (+y,-z), c2 (-y,+z), c3 (+y,+z).

Usage (from the repo root):
  python make_split_robot_urdf.py               # writes ..._thin_split4.urdf next to the source
  python make_split_robot_urdf.py --cells 2
"""
import argparse
import copy
import os
import xml.etree.ElementTree as ET

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "assets", "urdf", "xarm6", "xarm6_allegro_right_fsr_2023_thin.urdf")
PADS = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr", "link_5.0_fsr",
        "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr", "link_10.0_fsr",
        "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
        "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]  # task's contact_sensor_names order


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def cell_layout(size, n_cells, gap):
    """Return [(cell_size, offset_in_pad_frame), ...] for a box pad thin along x.
    Cells are centred on their quarter (or half) of the pad and shrunk by `gap` along split axes."""
    sx, sy, sz = size
    if n_cells == 4:
        return [((sx, sy / 2 - gap, sz / 2 - gap), (0.0, dy * sy / 4, dz * sz / 4))
                for dz in (-1, 1) for dy in (-1, 1)]
    if n_cells == 2:
        if sy > sz:
            return [((sx, sy / 2 - gap, sz), (0.0, d * sy / 4, 0.0)) for d in (-1, 1)]
        return [((sx, sy, sz / 2 - gap), (0.0, 0.0, d * sz / 4)) for d in (-1, 1)]
    raise ValueError("cells must be 2 or 4")


def set_box(elem, size, xyz):
    elem.find("geometry").find("box").set("size", fmt(size))
    elem.find("origin").set("xyz", fmt(xyz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, choices=[2, 4], default=4)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--gap", type=float, default=0.0002, help="gap between neighbouring cells, m")
    args = ap.parse_args()
    dst = args.dst or args.src.replace(".urdf", f"_split{args.cells}.urdf")

    tree = ET.parse(args.src)
    robot = tree.getroot()
    links = {l.get("name"): l for l in robot.findall("link")}
    joint_of = {j.find("child").get("link"): j for j in robot.findall("joint")}
    n_before = len(links)

    for pad in PADS:
        link = links[pad]
        vis, col = link.find("visual"), link.find("collision")
        size = [float(v) for v in col.find("geometry").find("box").get("size").split()]
        origin = col.find("origin")
        xyz = [float(v) for v in origin.get("xyz").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        assert min(range(3), key=lambda i: size[i]) == 0, f"{pad}: pad is not thin along x"
        assert all(abs(a) < 1e-12 for a in rpy), f"{pad}: rotated pad geometry not supported"
        assert link.find("inertial") is None, f"{pad}: has an inertial; mass split not implemented"

        cells = cell_layout(size, args.cells, args.gap)
        pad_joint = joint_of[pad]
        parent = pad_joint.find("parent").get("link")
        jorigin = pad_joint.find("origin")
        for k, (csize, off) in enumerate(cells):
            cxyz = [xyz[i] + off[i] for i in range(3)]
            if k == 0:  # the pad link itself becomes cell 0
                set_box(vis, csize, cxyz)
                set_box(col, csize, cxyz)
                continue
            name = f"{pad}_c{k}"
            new = ET.SubElement(robot, "link", {"name": name})
            for src_elem in (vis, col):
                e = copy.deepcopy(src_elem)
                set_box(e, csize, cxyz)
                new.append(e)
            joint = ET.SubElement(robot, "joint", {"name": f"joint_{name}", "type": "fixed"})
            ET.SubElement(joint, "parent", {"link": parent})   # same parent and origin as the pad's joint
            ET.SubElement(joint, "child", {"link": name})
            ET.SubElement(joint, "origin", {"rpy": jorigin.get("rpy", "0 0 0"), "xyz": jorigin.get("xyz", "0 0 0")})

    tree.write(dst, xml_declaration=True, encoding="utf-8")
    n_after = len(robot.findall("link"))
    print(f"wrote {dst}")
    print(f"links: {n_before} -> {n_after}  (pads: 16 -> {16 * args.cells} cells)")


if __name__ == "__main__":
    main()
