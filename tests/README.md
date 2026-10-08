# Tests

| Test file | Module under test | What a failure means | Runtime |
|-----------|-------------------|----------------------|---------|
| `test_calibrate.py` | `capture/calibrate.py` | Intrinsics no longer match a known synthetic camera, junk frames are accepted, or the live loop saves a frame with no board in view | fast |
| `test_live_check.py` | `capture/live_check.py`, `capture/common.py` | Cube pose (x, y, yaw) is wrong against a rendered scene, jitter or yaw wrap-around broke, or null `data/props.yaml` values are being accepted | fast |
| `test_fk_mujoco.py` | `sim/fk_check.py`, `control/kinematics` | The PoE model disagrees with the MuJoCo SO-101 (joint zero, sign, order, axis), or IK misses have a cause other than the joint-limit clamp | fast |
| `test_kinematics_vendored.py` | `control/kinematics` (checks vendored from the teleop repo) | Jacobian differs from finite differences, or IK round-trip success fell below its floor | fast, plus one slow test (full IK sweep, ~25 s) |
| `test_record.py` | `capture/record.py` | Video, timestamp CSV and camera frames disagree, the written clip biases the cube pose, a wrong frame size is accepted, or an existing episode could be overwritten | fast |
| `test_extract.py` | `extract/` (table frame, cube pose, cleaning, fingertip geometry, crop and gap bridging, coverage gaps, schema, split) | Table-frame pose is off against an analytic tilted-table scene or a rendered one (frame or yaw sign), a 2-marker board pose leaks in, short gaps are not bridged or long ones are, yaw is not unwrapped, the filter ignores timestamps, a processed `.npz` fails the schema, or `data/splits.json` no longer matches the manifest | fast (~9 s) |
| `test_smoke.py` | `eval/smoke.py` (rendered clip → extract → `PushTrackEnv(episode_files=...)` → `ScriptedPusher` → `eval/metrics.py`) | A link in the chain broke: extraction no longer recovers the rendered path (within 6 mm), the env cannot load an extracted `.npz`, or the scripted pusher fails the episode | fast (~7 s) |

Helpers, not tests: `synth.py` renders synthetic camera images with known geometry; `robot.py` builds the SO-101 model for the vendored checks. `test_ik.py` and `test_jacobian.py` are the vendored scripts that `test_kinematics_vendored.py` calls.

Every test file also runs as a script: `python -m tests.<name>`.

## Running

Before every commit (about 25 s):

    python -m pytest -m "not slow"

Everything, including the slow tests (about 45 s):

    python -m pytest

Install pytest with `pip install -e ".[dev]"`.
