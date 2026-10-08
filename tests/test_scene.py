"""Tests for C3.2: push scene and friction tuning.

Friction analysis (box 81 x 81 x 32.5 mm, 80 g):
  Tipping condition:  F_push * z_contact > W * (side/2)
                      F_push * 0.015 > 0.08 * 9.81 * 0.0405 = 0.0318 N·m
                      F_push > 2.12 N
  Max friction force (table mu=0.3):  0.3 * 0.08 * 9.81 = 0.235 N << 2.12 N
  => sliding without tipping is physically guaranteed for any realistic push force.

Run:  python -m pytest tests/test_scene.py -v
"""
import mujoco
import numpy as np
import pytest

from sim.scene import ARM_JOINTS, load_friction_test, load_push_scene


def test_friction_test_loads():
    """Friction-test scene (table + cube + mocap pusher) loads without error."""
    m, d = load_friction_test()
    assert m.nbody >= 3   # worldbody + cube + pusher (at least)
    cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    push_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pusher")
    assert cube_id >= 0 and push_id >= 0
    # pusher must be a mocap body
    assert m.body_mocapid[push_id] >= 0


def test_push_scene_loads():
    """Full push scene (table + cube + SO-101 + pusher capsule) loads without error."""
    m, d = load_push_scene()
    assert m.nbody > 5   # robot bodies + table + cube
    cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    assert cube_id >= 0
    # Pusher capsule geom must exist
    pcap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pusher_cap")
    assert pcap_id >= 0
    # Arm joints must be present
    for name in ARM_JOINTS:
        assert m.joint(name).id >= 0, f"joint {name!r} not found"


def test_cube_slides_not_tips():
    """A constant push makes the cube translate without tipping.

    Uses the mocap pusher so the test does not depend on the IK controller.
    The pusher advances at 0.05 m/s in +x for 1000 steps (2 s).

    Assertions:
      - cube moves > 20 mm in +x  (it slides)
      - cube z changes < 5 mm     (it stays on the table)
      - cube roll/pitch < 5 deg   (it does not tip)
    """
    m, d = load_friction_test()

    cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    push_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pusher")
    mocap_id = m.body_mocapid[push_id]

    mujoco.mj_forward(m, d)
    x0 = float(d.xpos[cube_id, 0])
    z0 = float(d.xpos[cube_id, 2])

    push_speed = 0.05   # m/s
    dt = float(m.opt.timestep)
    for _ in range(1000):
        d.mocap_pos[mocap_id, 0] += push_speed * dt
        mujoco.mj_step(m, d)

    xf = float(d.xpos[cube_id, 0])
    zf = float(d.xpos[cube_id, 2])

    # Sliding: cube must have moved
    assert xf - x0 > 0.02, (
        f"cube moved only {(xf - x0)*1000:.1f} mm in +x; expected > 20 mm sliding"
    )

    # No tipping: z stays near initial (table contact maintained)
    assert abs(zf - z0) < 0.005, (
        f"cube z changed {abs(zf - z0)*1000:.1f} mm; may be tipping or leaving table"
    )

    # No tipping: roll/pitch from quaternion qw component
    # Cube freejoint: qpos[adr+3:adr+7] = (qw, qx, qy, qz)
    jnt = m.body_jntadr[cube_id]
    qadr = int(m.jnt_qposadr[jnt])
    qw = float(d.qpos[qadr + 3])
    # Total rotation angle from identity = 2*arccos(|qw|); < 5 deg means |qw| > 0.9990
    tilt_rad = 2.0 * np.arccos(np.clip(abs(qw), 0.0, 1.0))
    assert tilt_rad < np.radians(5.0), (
        f"cube tilted {np.degrees(tilt_rad):.1f} deg — tipping detected"
    )
