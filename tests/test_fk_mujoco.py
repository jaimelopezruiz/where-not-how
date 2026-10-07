"""PoE forward kinematics against the MuJoCo SO-101 model (G1).

Run from the repo root:  python -m tests.test_fk_mujoco
"""
import numpy as np

from control.kinematics.core import FKinSpace
from control.kinematics.parser import findMnS
from sim import fk_check

EXPECTED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def test_fk_matches_mujoco():
    res = fk_check.run(n=300, seed=1)
    pe = res["pos_err"] * 1000
    print(f"\nFK position error over {res['n']} configs: max {pe.max():.5f} mm, mean {pe.mean():.5f} mm")
    assert res["names"] == EXPECTED_JOINTS
    assert res["limit_diff"] < 1e-4            # same joint ranges in both models
    assert res["screw_diff"] < 1e-4            # joint zeros and signs agree
    assert pe.max() < 0.05                     # mm; the two models are rounded differently (rpy 1.5708 vs quats)
    assert res["offset_spread_deg"].max() < 0.01   # the EE frames differ by a constant rotation only


def test_check_detects_a_flipped_joint_sign():
    """Negative control: if one joint's sign were wrong, this comparison must show it."""
    M, Slist, limits = findMnS()
    m, d = fk_check.load()
    rng = np.random.default_rng(2)
    errs = []
    for _ in range(50):
        q = rng.uniform(limits[:, 0], limits[:, 1])
        q_wrong = q.copy()
        q_wrong[2] = -q_wrong[2]                         # elbow_flex sign flipped on the MuJoCo side only
        fk_check.set_arm(m, d, fk_check.arm_joint_names(), q_wrong)
        errs.append(np.linalg.norm(fk_check.ee_pose(m, d)[:3, 3] - FKinSpace(M, Slist, q)[:3, 3]))
    assert np.mean(errs) > 0.01, np.mean(errs)           # metres: centimetres of error, against 0.004 mm when correct


def test_ik_misses_come_from_the_joint_limit_clamp():
    ik = fk_check.ik_in_mujoco(n=150, seed=3)
    print(f"\nIK: {ik['success_rate'] * 100:.0f}% converged, {ik['miss']} misses, {ik['miss_at_limit']} at a limit")
    assert ik["success_rate"] > 0.95
    assert ik["miss"] == ik["miss_at_limit"]             # no solution away from a limit misses its tolerance
    assert np.sum(ik["errs"] <= ik["ev"]) >= 0.9 * ik["n"]   # the large majority land within the tolerance


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
