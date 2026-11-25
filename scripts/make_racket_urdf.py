"""Write data/g1_racket.urdf: SONIC's G1 URDF plus a racket on the right wrist.

  --mount bracket (default): the direct-wrist-bracket model of
      vision_robotic_table_tennis hitting_plane/mount.py (bracket-v4):
        * both Dex3 hands removed (palm + finger links, ~0.70 kg per side), so the
          wrist links keep only their bare inertia (identical to g1_29dof.urdf),
        * bracket box   0.050 x 0.036 x 0.036 m at x = 0.07,   0.035 kg
        * handle        r 0.012, x 0.09 -> 0.17,              0.025 kg
        * blade         disk r 0.075, 12.4 mm, centre x = 0.2365, normal +Y, 0.14 kg
      total 0.20 kg. Masses/geometry are their provisional values, not measured.
  --mount dex3: Dex3 hands kept, single 0.16 kg disk (the previous model).

The racket links hang off right_wrist_yaw_link with fixed joints. Isaac Lab merges
fixed joints, so everything ends up on the wrist link and the env computes the
racket pose from that link (constants.RACKET_*). The blade collider is a box: the
asset converts cylinders to capsules, which would turn a thin disk into a sphere.

  python3 scripts/make_racket_urdf.py [--mount bracket|dex3] [--mass-scale 1.0]
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/g1_sonic_tt"))
from g1_sonic_tt import paths  # noqa: E402
from g1_sonic_tt.sonic.constants import RACKET_FACE_RADIUS, RACKET_OFFSET, RACKET_PARENT_LINK  # noqa: E402

PI2 = 1.5707963


def _link(root, name, mass, com, inertia, visuals=(), collisions=()):
    link = ET.SubElement(root, "link", name=name)
    i = ET.SubElement(link, "inertial")
    ET.SubElement(i, "origin", xyz=" ".join(map(str, com)), rpy="0 0 0")
    ET.SubElement(i, "mass", value=f"{mass:.5f}")
    ixx, iyy, izz = inertia
    ET.SubElement(i, "inertia", ixx=f"{ixx:.4e}", ixy="0", ixz="0", iyy=f"{iyy:.4e}", iyz="0", izz=f"{izz:.4e}")
    for tag, items in (("visual", visuals), ("collision", collisions)):
        for xyz, rpy, geom, attrs, color in items:
            v = ET.SubElement(link, tag)
            ET.SubElement(v, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
            ET.SubElement(ET.SubElement(v, "geometry"), geom, **attrs)
            if tag == "visual" and color:
                m = ET.SubElement(v, "material", name=f"{name}_mat")
                ET.SubElement(m, "color", rgba=color)
    return link


def _fixed(root, name, parent, child, xyz):
    j = ET.SubElement(root, "joint", name=name, type="fixed")
    ET.SubElement(j, "origin", xyz=" ".join(map(str, xyz)), rpy="0 0 0")
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)


def _box_inertia(m, x, y, z):
    return (m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12)


def _disk_inertia_y(m, r, t):  # disk with its axis along y
    i = m * r * r / 4 + m * t * t / 12
    return (i, m * r * r / 2, i)


def _rod_inertia_x(m, r, length):
    i = m * (3 * r * r + length * length) / 12
    return (m * r * r / 2, i, i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount", choices=["bracket", "dex3"], default="bracket")
    ap.add_argument("--mass-scale", type=float, default=1.0)
    args = ap.parse_args()
    s = args.mass_scale

    text = paths.SONIC_URDF.read_text()
    text = re.sub(r"package://robot_description/meshes/", str(paths.SONIC_MESH_DIR) + "/", text)
    assert "package://" not in text, "unresolved package:// paths remain"
    root = ET.fromstring(text)
    assert root.find(f"link[@name='{RACKET_PARENT_LINK}']") is not None

    r, t = RACKET_FACE_RADIUS, 0.0124
    red, wood, grey = "0.75 0.06 0.06 1", "0.6 0.45 0.25 1", "0.25 0.25 0.28 1"
    blade_vis = ((0, 0, 0), (PI2, 0, 0), "cylinder", {"radius": str(r), "length": str(t)}, red)
    blade_col = ((0, 0, 0), (0, 0, 0), "box", {"size": f"{2 * r} {t} {2 * r}"}, None)
    if args.mount == "bracket":
        removed = [j for j in root.findall("joint") if "_hand_" in j.find("child").get("link")]
        hand_links = [lk for lk in root.findall("link") if "_hand_" in lk.get("name")]
        hand_mass = sum(float(lk.find("inertial/mass").get("value")) for lk in hand_links)
        for e in removed + hand_links:
            root.remove(e)
        mb, mh, mf = 0.035 * s, 0.025 * s, 0.14 * s
        _link(root, "right_racket_bracket", mb, (0, 0, 0), _box_inertia(mb, 0.05, 0.036, 0.036),
              visuals=[((0, 0, 0), (0, 0, 0), "box", {"size": "0.05 0.036 0.036"}, grey)],
              collisions=[((0, 0, 0), (0, 0, 0), "box", {"size": "0.05 0.036 0.036"}, None)])
        _fixed(root, "right_racket_bracket_joint", RACKET_PARENT_LINK, "right_racket_bracket", (0.07, 0, 0))
        _link(root, "right_racket_handle", mh, (0, 0, 0), _rod_inertia_x(mh, 0.012, 0.08),
              visuals=[((0, 0, 0), (0, PI2, 0), "cylinder", {"radius": "0.012", "length": "0.08"}, wood)],
              collisions=[((0, 0, 0), (0, PI2, 0), "cylinder", {"radius": "0.012", "length": "0.08"}, None)])
        _fixed(root, "right_racket_handle_joint", RACKET_PARENT_LINK, "right_racket_handle", (0.13, 0, 0))
        _link(root, "right_racket", mf, (0, 0, 0), _disk_inertia_y(mf, r, t), visuals=[blade_vis],
              collisions=[blade_col])
        total = mb + mh + mf
        note = f"bracket mount, Dex3 hands removed ({hand_mass:.2f} kg), racket {total:.3f} kg"
    else:
        m = 0.16 * s
        _link(root, "right_racket", m, (-0.03, 0, 0), _disk_inertia_y(m, r, t), visuals=[blade_vis],
              collisions=[blade_col])
        note = f"Dex3 hands kept, racket {m:.3f} kg"
    _fixed(root, "right_racket_joint", RACKET_PARENT_LINK, "right_racket", RACKET_OFFSET)

    ET.indent(root)
    out = paths.RACKET_URDF
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ET.tostring(root, encoding="unicode"))
    print(f"saved {out}: {note}")


if __name__ == "__main__":
    main()
