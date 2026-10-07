"""Pytest wrappers for the checks vendored with control/kinematics (originals: the SO-101 teleop repo,
see control/kinematics/ORIGIN.md).

The originals are scripts that print PASS/FAIL, so pytest would skip them or they could never fail. These
wrappers call the same code with the same configurations and thresholds and turn a miss into a test failure:

- test_jacobian.py: analytic space Jacobian against numerical differentiation of FK (five configurations).
- test_ik.py: DLS IK round-trip success rate against its per-noise-level floors, and behaviour at an elbow singularity.

Run from the repo root:  python -m pytest tests/test_kinematics_vendored.py
"""
import sys

import numpy as np
import pytest

from tests import test_ik, test_jacobian

JACOBIAN_TOL = 1e-4     # the same threshold test_jacobian.verify_jac prints against
JACOBIAN_CONFIGS = [
    [0, 0, 0, 0, 0],
    [-np.pi / 8, 0, 0, 0, 0],
    [0, np.pi / 4, 0, 0, 0],
    [0, 0, -np.pi / 4, 0, 0],
    [np.pi / 8, -np.pi / 4, np.pi / 6, -np.pi / 8, np.pi / 3],
]
FAST_NOISE_LEVELS = [0.0, 0.01, 0.1, 0.5, 0.75, 1.0]    # the cheap end of the sweep, ~3.5 s in total


@pytest.mark.parametrize("theta", JACOBIAN_CONFIGS, ids=lambda t: "theta_" + "_".join(f"{v:.2f}" for v in t))
def test_jacobian_matches_numerical_derivative(theta):
    """A wrong screw axis or Jacobian column shows as a mismatch with the finite-difference Jacobian of FK."""
    assert test_jacobian.verify_jac(theta) < JACOBIAN_TOL


def test_ik_round_trip_success_at_low_noise():
    """An IK regression (damping, step, in-loop clamping) drops round-trip success below its floor for near starts."""
    for i, noise in enumerate(test_ik.NOISE_LEVELS):
        if noise not in FAST_NOISE_LEVELS:
            continue
        # same per-level seed offset as noise_sweep, so these are the same trials as the full sweep
        rate, _ = test_ik.round_trip(test_ik.limits, test_ik.ITERS, noise, seed=test_ik.SEED + i)
        assert rate >= test_ik.MIN_SUCCESS[noise], f"noise {noise}: {rate:.1f}% < {test_ik.MIN_SUCCESS[noise]}%"


def test_ik_dls_survives_an_elbow_singularity():
    """A NaN or runaway step in the damped solve near the singular elbow pose shows as non-finite or non-converging history."""
    cond, hist_dls, _ = test_ik.verify_singular()
    assert cond > 10                      # the pose really is ill-conditioned (about 30 here)
    assert np.all(np.isfinite(hist_dls))
    assert len(hist_dls) - 1 < 50         # converged before verify_singular's iteration cap


@pytest.mark.slow
def test_ik_full_noise_sweep_meets_all_thresholds():
    """The original test_ik run end to end: every one of ten noise levels at 200 trials meets its floor."""
    argv, sys.argv = sys.argv, ["test_ik"]            # main() parses sys.argv; give it none of pytest's
    try:
        test_ik.main()
    except SystemExit as e:                           # main() signals a threshold failure with SystemExit(message)
        pytest.fail(str(e))
    finally:
        sys.argv = argv


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and name != "test_jacobian_matches_numerical_derivative":
            fn()
            print(f"PASS {name}")
    for theta in JACOBIAN_CONFIGS:
        test_jacobian_matches_numerical_derivative(theta)
    print("PASS test_jacobian_matches_numerical_derivative")
