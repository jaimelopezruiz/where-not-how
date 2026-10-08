"""Push scene: table, box from props, capsule pusher on the SO-101 EE (C3.2).

Two scene variants:
  load_friction_test() -- table + cube + mocap capsule; use for friction tuning.
  load_push_scene()    -- table + cube + SO-101 arm with capsule on the gripper.

For the friction test the pusher moves in +x via d.mocap_pos.  The contact height
is set low on the 32.5 mm cube face (z ~15 mm) and clear of the table (z > 3 mm).

Friction values (per the tipping analysis in tests/test_scene.py):
  table-cube  mu=0.3 -- cube slides under a realistic push force
  pusher-cube mu=0.8 -- pusher grips the cube face without slipping off

For load_push_scene the SO-101 MJCF is loaded and augmented in-memory:
  - meshdir is set to the absolute path so from_xml_string finds the STL files
  - robot base body is moved to robot_base_pos
  - a table plane geom and a free cube body are added to worldbody
  - a capsule collision geom and a tip site are appended to the gripper body
    (the existing gripper mesh geoms are all class="visual", contype=0, so only
    the new capsule makes contact)
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from capture.common import DEFAULT_PROPS, load_props

REPO_ROOT = Path(__file__).resolve().parent.parent
SO101_MJCF = REPO_ROOT / "sim" / "assets" / "so101" / "so101_new_calib.xml"
SO101_ASSETS_DIR = REPO_ROOT / "sim" / "assets" / "so101" / "assets"

EE_SITE = "gripperframe"   # existing site on the SO-101 gripper body
PUSHER_SITE = "pusher_tip" # site added by build_push_scene_xml for contact tracking

# Arm joints in PoE column order (matches fk_check.arm_joint_names / findMnS)
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _box_inertia(mass, side, height):
    """Principal inertia of a solid box with square base (side × side × height)."""
    ixx = mass * (side ** 2 + height ** 2) / 12  # about x (or y)
    izz = mass * 2 * side ** 2 / 12              # about z (vertical)
    return ixx, ixx, izz


def build_friction_test_xml(props):
    """Return MJCF XML string: table + cube + mocap capsule pusher.

    The pusher capsule:
      radius 6 mm, half-length 8 mm, axis along z (vertical)
      body centre at z = 15 mm -- low on the 32.5 mm cube face, 3 mm clear of table
    Push direction is +x; the test advances d.mocap_pos[0, 0] each step.
    """
    side   = props["cube_side"]
    height = props["cube_height"]
    mass   = props["cube_mass"]
    hs, hh = side / 2, height / 2
    ixx, iyy, izz = _box_inertia(mass, side, height)

    cap_r  = 0.006   # capsule radius (m)
    cap_hl = 0.008   # capsule half-length (m)
    z_c    = 0.015   # capsule-body z: 15 mm from table, low on the 32.5 mm face
    # bottom of capsule hemisphere: z_c - cap_hl - cap_r = 0.015 - 0.008 - 0.006 = 0.001 m
    # top  of capsule hemisphere: 0.015 + 0.008 + 0.006 = 0.029 m  (89 % of face height)

    # pusher x: capsule surface just touches cube face at x = -hs when x advances by 0.002
    px = -(hs + cap_r + 0.002)

    return (
        '<mujoco model="friction_test">\n'
        '  <option timestep="0.002" gravity="0 0 -9.81"/>\n'
        '  <worldbody>\n'
        '    <geom name="table" type="plane" size="1 1 0.1"'
        ' friction="0.3 0.005 0.0001" rgba="0.9 0.85 0.7 1"/>\n'
        f'    <body name="cube" pos="0 0 {hh:.6f}">\n'
        '      <freejoint/>\n'
        f'      <inertial pos="0 0 0" mass="{mass:.6f}"'
        f' diaginertia="{ixx:.8f} {iyy:.8f} {izz:.8f}"/>\n'
        f'      <geom type="box" size="{hs:.6f} {hs:.6f} {hh:.6f}"'
        ' friction="0.3 0.005 0.0001" rgba="0.8 0.6 0.2 1"/>\n'
        '    </body>\n'
        f'    <body name="pusher" mocap="true" pos="{px:.6f} 0 {z_c:.6f}">\n'
        f'      <geom type="capsule" size="{cap_r:.4f} {cap_hl:.4f}"'
        ' friction="0.8 0.005 0.0001" rgba="0.2 0.4 0.8 1"/>\n'
        '    </body>\n'
        '  </worldbody>\n'
        '</mujoco>'
    )


def build_push_scene_xml(props, robot_base_pos=(0.0, 0.0, 0.0)):
    """Return MJCF XML string: table + cube + SO-101 arm + pusher capsule.

    The robot base is placed at robot_base_pos.  The cube's default start position
    is (0.15, 0.15, cube_height/2); the env overrides this at reset.
    A camera looking down at the workspace is included for render().
    """
    side   = props["cube_side"]
    height = props["cube_height"]
    mass   = props["cube_mass"]
    hs, hh = side / 2, height / 2
    ixx, iyy, izz = _box_inertia(mass, side, height)

    tree = ET.parse(str(SO101_MJCF))
    root = tree.getroot()

    # meshdir → absolute path so from_xml_string resolves the STL assets
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(SO101_ASSETS_DIR))

    worldbody = root.find("worldbody")

    # Move robot base body to robot_base_pos
    for body in list(worldbody):
        if body.tag == "body" and body.get("name") == "base":
            x, y, z = robot_base_pos
            body.set("pos", f"{x:.6f} {y:.6f} {z:.6f}")
            break

    # Table plane (inserted at index 0 so it renders under the robot)
    tbl = ET.Element("geom")
    for k, v in [("name", "table"), ("type", "plane"), ("size", "1 1 0.1"),
                 ("friction", "0.3 0.005 0.0001"), ("rgba", "0.9 0.85 0.7 1")]:
        tbl.set(k, v)
    worldbody.insert(0, tbl)

    # Overhead camera for rendering
    cam = ET.Element("camera")
    cam.set("name", "overhead")
    cam.set("pos", "0.2 0.2 0.8")
    cam.set("quat", "0.924 -0.383 0 0")   # ~45 deg tilt, similar to recording angle
    worldbody.insert(1, cam)

    # Cube free body (default position; env overrides at reset)
    cube = ET.SubElement(worldbody, "body")
    cube.set("name", "cube"); cube.set("pos", f"0.15 0.15 {hh:.6f}")
    ET.SubElement(cube, "freejoint")
    ini = ET.SubElement(cube, "inertial")
    ini.set("pos", "0 0 0"); ini.set("mass", f"{mass:.6f}")
    ini.set("diaginertia", f"{ixx:.8f} {iyy:.8f} {izz:.8f}")
    cg = ET.SubElement(cube, "geom")
    cg.set("type", "box"); cg.set("size", f"{hs:.6f} {hs:.6f} {hh:.6f}")
    cg.set("friction", "0.3 0.005 0.0001"); cg.set("rgba", "0.8 0.6 0.2 1")

    # Pusher capsule + tip site on the gripper body
    # gripperframe site in the gripper body: pos="-0.0079 -0.000218 -0.0981274"
    # The capsule is placed at that site; contype/conaffinity explicitly set to 1
    # because the so101_new_calib default class sets contype=0 for all geoms.
    gripper = next((b for b in root.iter("body") if b.get("name") == "gripper"), None)
    if gripper is not None:
        pcap = ET.SubElement(gripper, "geom")
        for k, v in [("name", "pusher_cap"), ("type", "capsule"), ("size", "0.006 0.008"),
                     ("pos", "-0.0079 -0.000218 -0.0981274"),
                     ("contype", "1"), ("conaffinity", "1"),
                     ("friction", "0.8 0.005 0.0001"), ("rgba", "0.2 0.4 0.8 1")]:
            pcap.set(k, v)
        ptip = ET.SubElement(gripper, "site")
        for k, v in [("name", PUSHER_SITE), ("pos", "-0.0079 -0.000218 -0.1081274"),
                     ("size", "0.003"), ("group", "3")]:
            ptip.set(k, v)

    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode")


def load_friction_test(props_path=DEFAULT_PROPS):
    """Friction-test scene (table + cube + mocap pusher). Returns (model, data)."""
    props = load_props(props_path, require=("cube_side", "cube_height", "cube_mass"))
    m = mujoco.MjModel.from_xml_string(build_friction_test_xml(props))
    return m, mujoco.MjData(m)


def load_push_scene(props_path=DEFAULT_PROPS, robot_base_pos=(0.0, 0.0, 0.0)):
    """Full push scene (table + cube + SO-101 + pusher). Returns (model, data)."""
    props = load_props(props_path, require=("cube_side", "cube_height", "cube_mass"))
    m = mujoco.MjModel.from_xml_string(build_push_scene_xml(props, robot_base_pos))
    return m, mujoco.MjData(m)


# ---------------------------------------------------------------------------
# Simulation helpers shared by scene tests and the env
# ---------------------------------------------------------------------------

def set_joints(m, d, names, q):
    """Set arm joint positions by name and refresh forward kinematics."""
    for name, v in zip(names, q):
        d.qpos[m.joint(name).qposadr[0]] = v
    mujoco.mj_kinematics(m, d)


def get_site_pos(m, d, site_name):
    """World position of a named site (3-vector, copy)."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return d.site_xpos[sid].copy()
