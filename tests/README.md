# Tests

| Test file | Module under test | What a failure means | Runtime |
|-----------|-------------------|----------------------|---------|
| `test_calibrate.py` | `capture/calibrate.py` | Intrinsics no longer match a known synthetic camera, junk frames are accepted, or the live loop saves a frame with no board in view | fast |
| `test_live_check.py` | `capture/live_check.py`, `capture/common.py` | Cube pose (x, y, yaw) is wrong against a rendered scene, jitter or yaw wrap-around broke, or null `data/props.yaml` values are being accepted | fast |
| `test_fk_mujoco.py` | `sim/fk_check.py`, `control/kinematics` | The PoE model disagrees with the MuJoCo SO-101 (joint zero, sign, order, axis), or IK misses have a cause other than the joint-limit clamp | fast |
| `test_kinematics_vendored.py` | `control/kinematics` (checks vendored from the teleop repo) | Jacobian differs from finite differences, or IK round-trip success fell below its floor | fast, plus one slow test (full IK sweep, ~25 s) |

Helpers, not tests: `synth.py` renders synthetic camera images with known geometry; `robot.py` builds the SO-101 model for the vendored checks. `test_ik.py` and `test_jacobian.py` are the vendored scripts that `test_kinematics_vendored.py` calls.

Every test file also runs as a script: `python -m tests.<name>`.

## Running

Before every commit (about 25 s):

    python -m pytest -m "not slow"

Everything, including the slow tests (about 45 s):

    python -m pytest

Install pytest with `pip install -e ".[dev]"`.
