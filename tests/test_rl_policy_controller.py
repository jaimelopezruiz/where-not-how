"""PolicyController stopping rule, action scaling and the C6.4 evaluation subset (no training)."""
import numpy as np
import pytest

from rl.policy_controller import PolicyController


class _Model:
    def __init__(self, action):
        self.action, self.seen = np.asarray(action, np.float32), None

    def predict(self, obs, deterministic=True):
        self.seen = obs
        return self.action[None], None


def _obs(ref):
    return np.concatenate([[0.3, 0.0, 0.2, 0.0, 0.0, 1.0], np.asarray(ref, np.float32).ravel()]).astype(np.float32)


MID_PATH = [[0.0, 0.0], [0.025, 0.0], [0.05, 0.0], [0.075, 0.0], [0.1, 0.0]]
AT_END = [[0.01, 0.0]] * 5                                   # all points clamped to the path end, 1 cm away


def test_policy_actions_are_scaled_to_metres_and_clipped():
    c = PolicyController(_Model([0.5, -3.0]), max_delta=0.02)
    assert c(_obs(MID_PATH)) == pytest.approx([0.01, -0.02])


def test_observation_goes_through_vecnormalize_when_given():
    class _Norm:
        def normalize_obs(self, o):
            return o - 1.0
    m = _Model([0.0, 0.0])
    PolicyController(m, _Norm())(_obs(MID_PATH))
    assert m.seen[0, 0] == pytest.approx(0.3 - 1.0)


def test_arrival_holds_still_then_reports_done():
    m = _Model([1.0, 1.0])
    c = PolicyController(m, hold_steps=3)
    assert not c.done and np.any(c(_obs(MID_PATH)) != 0)
    for _ in range(3):
        assert np.all(c(_obs(AT_END)) == 0) and not c.done
    assert np.all(c(_obs(AT_END)) == 0) and c.done
    c.reset(None)
    assert not c.done


def test_not_arrived_when_end_is_far_or_points_not_clamped():
    c = PolicyController(_Model([1.0, 0.0]))
    c(_obs([[0.05, 0.0]] * 5))                                  # clamped but 5 cm from the end
    assert c._hold is None
    c(_obs([[0.01, 0.0]] * 4 + [[0.02, 0.0]]))                   # near, but the points have not all reached the end
    assert c._hold is None


def test_eval_subset_is_train_only_two_per_category():
    from rl.train_full import eval_subset
    import json
    from capture.common import REPO_ROOT
    splits = json.loads((REPO_ROOT / "data" / "splits.json").read_text())
    sub = eval_subset("train")
    assert set(sub) <= set(splits["train"]) and not set(sub) & set(splits["test"])
    assert len(sub) == 8
