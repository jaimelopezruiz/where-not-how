"""C6.1 / C6.2: phase tracker, reward terms and reference-point observation of PushTrack-v0.

Synthetic straight and L-shaped paths, with the cube placed by hand (no pushing needed).

Run: python -m pytest tests/test_push_env_reward.py
"""
import mujoco
import numpy as np
import pytest

from sim.push_env import PushTrackEnv


def _line(length=0.30, n=301):
    return np.column_stack([np.linspace(0, length, n), np.zeros(n)])


@pytest.fixture
def env():
    e = PushTrackEnv(paths=[_line()], max_episode_steps=50)
    e.reset(seed=0)
    yield e
    e.close()


def _put_cube(env, xy, yaw=0.0):
    q = env._cube_qadr
    env._data.qpos[q:q + 2] = xy
    env._data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(env._model, env._data)


def _at(env, arc, lateral=0.0):
    """Point on the (workspace-placed) path at an arc coordinate, offset sideways."""
    p = env._point_at(np.array([arc]))[0]
    return p + np.array([0.0, lateral])


def _expected_reach_penalty(env):
    """beta * |EE - pre-contact point|, the point `standoff` behind the cube against the push direction."""
    cube = env._cube_xy_yaw()[0]
    ahead = env._point_at(np.array([env._phase + env.ref_spacing]))[0] - cube
    goal = cube - env._standoff * ahead / np.linalg.norm(ahead)
    return env.reach_weight * np.linalg.norm(env._ee_xy() - goal)


def test_progress_gain_is_arc_fraction(env):
    total = env._total_arc_len
    _put_cube(env, _at(env, 0.02))
    reward, info = env._compute_reward()
    assert info["arc_progress"] == pytest.approx(0.02 / total, rel=1e-6)
    assert info["lateral_dev_m"] == pytest.approx(0.0, abs=1e-9)
    assert reward == pytest.approx(0.02 / total - _expected_reach_penalty(env), rel=1e-6)


def test_lateral_and_reach_terms(env):
    _put_cube(env, _at(env, 0.0, lateral=0.01))
    reward, info = env._compute_reward()
    assert info["lateral_dev_m"] == pytest.approx(0.01, abs=1e-9)
    assert reward == pytest.approx(0.0 - 0.1 * 0.01 - _expected_reach_penalty(env), abs=1e-9)


def test_phase_is_monotone_and_backwards_gives_no_reward(env):
    _put_cube(env, _at(env, 0.05))
    env._compute_reward()
    phase = env._phase
    _put_cube(env, _at(env, 0.02))                       # pushed back
    reward, info = env._compute_reward()
    assert env._phase == phase
    # no progress term; the lateral term is the distance back to the phase point, 3 cm behind the cube
    assert reward == pytest.approx(-0.1 * 0.03 - _expected_reach_penalty(env), abs=1e-9)


def test_no_progress_credit_beyond_r_adv(env):
    _put_cube(env, _at(env, 0.05, lateral=env.r_adv + 0.005))
    _, info = env._compute_reward()
    assert info["arc_progress"] == 0.0 and info["lateral_dev_m"] > env.r_adv
    _put_cube(env, _at(env, 0.05, lateral=env.r_adv - 0.005))
    _, info = env._compute_reward()
    assert info["arc_progress"] > 0.0


def test_phase_cannot_skip_ahead_of_the_window(env):
    """A path that comes back near itself must not let the phase jump along it."""
    far = env.progress_window + 0.05
    _put_cube(env, _at(env, far))                        # on the path, but beyond the search window
    _, info = env._compute_reward()
    assert info["arc_progress"] == 0.0


def test_phase_does_not_jump_to_a_later_leg_of_a_looping_path():
    # out along +x, up 3 cm, back along -x: the return leg passes 3 cm from the outbound one
    path = np.array([[0, 0], [0.3, 0], [0.3, 0.03], [0.0, 0.03]], float)
    e = PushTrackEnv(paths=[path], max_episode_steps=50)
    e.reset(seed=0)
    pts = e._path
    _put_cube(e, pts[0] + [0.2, 0.03])                   # on the return leg; the nearest vertex of the whole path
    _, info = e._compute_reward()
    assert e._phase == 0.0 and info["arc_progress"] == 0.0
    e.close()


def test_success_bonus_needs_progress_and_final_error(env):
    total = env._total_arc_len
    env._phase = 0.95 * total
    env._max_arc_progress = 0.95
    _put_cube(env, env._path[-1] + [0.01, 0.0])
    reward, info = env._compute_reward()
    assert info["success"] and reward > 9.0
    env.reset(seed=0)
    env._phase, env._max_arc_progress = 0.5 * total, 0.5     # cube at the end but progress was not earned
    _put_cube(env, env._path[-1])
    _, info = env._compute_reward()
    assert not info["success"] or info["arc_progress"] >= 0.9


def test_success_just_either_side_of_two_centimetres(env):
    total = env._total_arc_len
    for off, ok in ((0.019, True), (0.021, False)):
        env.reset(seed=0)
        env._phase, env._max_arc_progress = 0.95 * total, 0.95
        _put_cube(env, env._path[-1] + [off, 0.0])
        assert env._compute_reward()[1]["success"] == ok


def test_reference_points_are_arc_length_spaced_in_the_cube_frame():
    # a path with pauses: half of its samples repeat, so vertex-indexed lookahead would see nothing
    xs = np.repeat(np.linspace(0, 0.3, 61), 2)
    e = PushTrackEnv(paths=[np.column_stack([xs, np.zeros_like(xs)])], k_ref=5, ref_spacing=0.025)
    obs, _ = e.reset(seed=0)
    ref = obs[6:].reshape(5, 2)
    assert ref[:, 0] == pytest.approx(0.025 * np.arange(5), abs=1e-6)   # cube at the start, yaw 0
    assert ref[:, 1] == pytest.approx(0.0, abs=1e-6)
    e.close()


def test_reference_points_follow_the_phase_and_clamp_at_the_end(env):
    total = env._total_arc_len
    env._phase = total - 0.04
    obs = env._get_obs().reshape(-1)
    cube = env._cube_xy_yaw()[0]
    pts = obs[6:].reshape(5, 2)
    arcs = np.minimum(env._phase + 0.025 * np.arange(5), total)
    assert pts == pytest.approx(env._point_at(arcs) - cube, abs=1e-5)       # yaw 0: box frame = world offsets
    assert pts[2:] == pytest.approx(np.tile(pts[2], (3, 1)), abs=1e-6)        # all clamped to the path end


def test_reference_points_rotate_with_the_box(env):
    yaw = 0.7
    _put_cube(env, env._path[0], yaw=yaw)
    obs = env._get_obs()
    pts = obs[6:].reshape(5, 2)
    world = env._point_at(0.025 * np.arange(5)) - env._path[0]
    c, s = np.cos(yaw), np.sin(yaw)
    expected = world @ np.array([[c, -s], [s, c]])            # R(yaw)^T applied to row vectors
    assert pts == pytest.approx(expected, abs=1e-5)
    assert obs[4:6] == pytest.approx([s, c], abs=1e-6)


def test_episodes_option_restricts_split_env():
    e = PushTrackEnv(split="train", episodes=["ep_010"])
    assert e.episode_ids == ["ep_010"]
    _, info = e.reset(seed=5)
    assert info["episode_id"] == "ep_010"
    e.close()
    with pytest.raises(ValueError):
        PushTrackEnv(split="train", episodes=["ep_005"])        # a test episode


def test_precontact_reach_target_is_behind_the_cube_and_literal_option_is_the_cube(env):
    cube = env._cube_xy_yaw()[0]
    goal = env._reach_target(cube)
    assert goal - cube == pytest.approx([-env._standoff, 0.0], abs=1e-9)            # path runs along +x
    assert env._standoff == pytest.approx(0.081 / 2 + 0.006 + 0.005)
    literal = PushTrackEnv(paths=[_line()], precontact_reach=False, reach_weight=0.01)
    literal.reset(seed=0)
    assert literal._reach_target(cube) == pytest.approx(cube)
    literal.close()


def test_reach_term_vanishes_at_the_precontact_point(env, monkeypatch):
    """Beside the cube the penalty is large; at the pre-contact point it is zero."""
    cube = env._cube_xy_yaw()[0]
    _, info = env._compute_reward()
    assert info["reach_dist_m"] > 0.1
    monkeypatch.setattr(env, "_ee_xy", lambda: env._reach_target(cube))
    reward, info = env._compute_reward()
    assert info["reach_dist_m"] == pytest.approx(0.0, abs=1e-9)


# --- C6.7: displacement-weighted deviation penalty -------------------------------------------------

@pytest.fixture
def dev_env():
    e = PushTrackEnv(paths=[_line()], max_episode_steps=50, deviation_weight=5.0)
    e.reset(seed=0)
    yield e
    e.close()


def test_deviation_weight_none_keeps_the_original_reward(env):
    """Default env: every reward is progress - 0.1 * lateral - reach_weight * reach + bonus, from the step info."""
    assert env.deviation_weight is None
    rng = np.random.default_rng(0)
    prev = 0.0
    for _ in range(6):
        _, reward, _, _, info = env.step(rng.uniform(-0.02, 0.02, 2))
        expected = ((info["arc_progress"] - prev) - 0.1 * info["lateral_dev_m"]
                    - env.reach_weight * info["reach_dist_m"] + (10.0 if info["success"] else 0.0))
        assert reward == pytest.approx(expected, abs=1e-9)
        assert info["dev_penalty"] == pytest.approx(0.1 * info["lateral_dev_m"], abs=1e-12)
        prev = info["arc_progress"]


def test_deviation_penalty_sums_to_weight_times_mean_deviation_cm(dev_env):
    """A box pushed along the whole reference at a constant 1 cm offset: total penalty ~ weight x 1.0."""
    env = dev_env
    total = env._total_arc_len
    _put_cube(env, _at(env, 0.0, lateral=0.01))
    env._compute_reward()                                   # sync the previous position; not counted
    penalty = 0.0
    for arc in np.arange(0.01, total + 1e-9, 0.01):
        _put_cube(env, _at(env, arc, lateral=0.01))
        _, info = env._compute_reward()
        assert info["lateral_dev_m"] == pytest.approx(0.01, abs=1e-6)
        penalty += info["dev_penalty"]
    assert penalty == pytest.approx(env.deviation_weight * 1.0, rel=0.10)


def test_stationary_offset_box_gets_no_deviation_penalty(dev_env):
    env = dev_env
    _put_cube(env, _at(env, 0.05, lateral=0.02))
    env._compute_reward()                                   # the jump from the start is movement
    for _ in range(3):
        reward, info = env._compute_reward()
        assert info["lateral_dev_m"] == pytest.approx(0.02, abs=1e-9)
        assert info["dev_penalty"] == 0.0
        assert reward == pytest.approx(-_expected_reach_penalty(env), abs=1e-9)
