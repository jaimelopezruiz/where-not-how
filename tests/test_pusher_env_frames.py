"""Frame and yaw convention between PushTrackEnv observations and ScriptedPusher (C5.2).

The controller works from obs = [ee_x, ee_y, cube_x, cube_y, sin_yaw, cos_yaw, ...] and models the cube as a
square rotated by yaw about its centre. Here the observation is compared with the MuJoCo state, and the
controller's contact point with the actual box geom.

Run: python -m pytest tests/test_pusher_env_frames.py
"""
import mujoco
import numpy as np
import pytest

from control.scripted_pusher import ScriptedPusher, contact_point
from sim.push_env import PushTrackEnv
from sim.scene import EE_SITE


@pytest.fixture(scope="module")
def env():
    e = PushTrackEnv(split="train")
    yield e
    e.close()


def _geom_pose(env):
    m, d = env._model, env._data
    gid = int(np.flatnonzero(m.geom_bodyid == env._cube_id)[0])
    return d.geom_xpos[gid][:2].copy(), d.geom_xmat[gid].reshape(3, 3)[:2, :2].copy(), m.geom_size[gid][:2].copy()


def test_obs_cube_pose_matches_mujoco(env):
    # the episode whose recorded start yaw is furthest from a multiple of 90 deg (the box is square)
    i = int(np.argmax(np.abs(np.sin(2 * np.array(env._yaw0)))))
    assert abs(np.sin(2 * env._yaw0[i])) > 0.3
    obs, _ = env.reset(options={"episode": i})
    centre, rot, _ = _geom_pose(env)
    assert obs[2:4] == pytest.approx(centre, abs=1e-6)
    # geom x axis in the world is (cos yaw, sin yaw): yaw from obs is the box rotation, CCW, same frame as xy
    assert rot[:, 0] == pytest.approx([obs[5], obs[4]], abs=1e-6)
    assert np.arctan2(obs[4], obs[5]) == pytest.approx(env._yaw0[i], abs=1e-6)
    assert obs[2:4] == pytest.approx(env.reference_path[0], abs=1e-6)    # cube starts on the reference


def test_obs_ee_matches_mujoco_site_at_rest_and_moving(env):
    env.reset(options={"episode": 0})
    sid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    for k in range(8):
        obs, *_ = env.step(np.array([-0.01, 0.008]))
        assert obs[:2] == pytest.approx(env._data.site_xpos[sid][:2], abs=1e-3), f"step {k}"


def test_controller_contact_point_lies_on_the_mujoco_box(env):
    i = int(np.argmax(np.abs(np.sin(2 * np.array(env._yaw0)))))
    obs, _ = env.reset(options={"episode": i})
    pusher = ScriptedPusher()
    assert pusher.half_side == pytest.approx(_geom_pose(env)[2][0])      # same measured side as the scene
    centre, rot, half = _geom_pose(env)
    yaw = np.arctan2(obs[4], obs[5])
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        d = np.array([np.cos(ang), np.sin(ang)])
        p = contact_point(obs[2:4], yaw, d, pusher.half_side)
        local = rot.T @ (p - centre)
        assert np.max(np.abs(local) / half) == pytest.approx(1.0, abs=1e-6)     # on the boundary of the geom
