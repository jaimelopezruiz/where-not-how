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
| `test_residual.py` | `rl/residual.py` (`ResidualActions`, `ResidualController`), `ScriptedPusher.executed` | A zero residual no longer reproduces the scripted pusher's actions or metrics, the appended base action is wrong, the residual exceeds `residual_scale` · `max_delta`, or the stuck detection is not told the executed command | fast |
| `test_ee_controller.py` | `control/ee_controller.py` (DLS IK wrapper) | A limit-clamped IK miss is accepted as a success, the target orientation drifts from the current FK pose, `step(vx, vy, dt)` moves the end effector by more than 1 mm off `v·dt`, chunked solving loses a success the single 200-iteration call had, or an unreachable target burns the full iteration budget | fast (~5 s; also prints solves/s) |
| `test_workspace.py` | `control/workspace.py` (human table frame → robot frame map) | The fitted scale leaves (0, 1], the affine map is not affine, the base footprint or a far point counts as reachable, or (slow test) a train trajectory is infeasible or `data/workspace_map.json` differs from a fresh fit on the train split | fast (~6 s), plus one slow test (train-split fit, ~55 s) |
| `test_scene.py` | `sim/scene.py` | The friction-test or push scene fails to load, a steady push tips the cube instead of sliding it, or an arm link other than the pusher capsule can touch the cube | fast |
| `test_push_env.py` | `sim/push_env.py` (`PushTrack-v0`) | The env fails `gymnasium.check_env`, a seeded reset is not reproducible, observation or step types are wrong, resampling mishandles the ±π yaw wrap, `paths=` and `split=` can be mixed, or (needs `data/`) the cube does not start at the mapped first sample, a split loads another split's episodes, or `episode_files=` differs from the same episode loaded by split | fast (~6 s) |
| `test_push_env_reward.py` | `sim/push_env.py` (phase tracker, reward, reference-point observation) | The phase goes backwards or skips ahead (also on a looping path), progress is paid beyond the window or for backward motion, the success bonus ignores progress or the 2 cm final error, reference points are not arc-length spaced, rotated into the cube frame or clamped at the end, the reach target is not behind the cube, or the deviation penalty (C6.7) changes the default reward or fails to sum to weight · mean deviation | fast (~12 s) |
| `test_pusher_env_frames.py` | `sim/push_env.py` observation vs MuJoCo, `control/scripted_pusher.py` contact point | The observed cube pose or end-effector position disagrees with the MuJoCo state (frame or yaw convention), or the controller's contact point is not on the real box geom | fast |
| `test_scripted_pusher.py` | `control/scripted_pusher.py` (synthetic paths and a kinematic box, no simulator) | The contact point is off the box boundary or the wrong face, the approach cuts through the footprint, actions exceed `max_delta`, push/approach switching chatters, progress jumps back or across a self-crossing path, the lookahead is not a fixed arc length, a finished path still moves, the cube side is not read from `props.yaml`, closed-loop pushing misses a straight or curved path, or the step size does not shrink and recover when the arm stalls | fast (209 cases, ~1 s) |
| `test_metrics.py` | `eval/metrics.py` | Deviation, final error or progress is wrong on a known line, arc or S-curve, success flips at the 2 cm or 90 % thresholds, completion time is wrong or not NaN for a stalled run, progress exceeds 1, a stationary prefix changes the mean deviation, or the per-category summary miscounts | fast |
| `test_report.py` | `eval/report.py` | The results CSV has wrong columns or values, the category table or overlay plot is not written, a GIF is the wrong size or accepts mismatched frames, or `evaluate` mis-scores a stub env, does not reset the controller, or stops at the env's first success when the controller has not said done | fast (~2 s) |
| `test_rl_policy_controller.py` | `rl/policy_controller.py`, `rl.train_full.eval_subset` | Policy output is not scaled to metres and clipped to `max_delta`, VecNormalize is skipped, the controller does not hold still for `hold_steps` after arriving and then report done, it declares arrival when the end is far or the points are unclamped, or the C6.4 monitoring subset leaks a test episode or is not two per category | fast (~3 s, imports the trainer) |
| `test_results_figures.py` | `scripts/results_figures.py` | The results table differs from the known numbers in the committed scripted test CSV, curves from several progress files are not concatenated, sorted and de-duplicated, the training-curves figure is not written, or a run with no curves does not fail naming the missing pattern | fast |
| `test_showcase.py` | `scripts/showcase.py` | The push window or GIF timeline is wrong, a crop leaves the frame or misses the track, the raw and sim views disagree after `orient` (+x up, +y left), the feasibility check passes a path off the reachable set or inside the base radius, real-video panels pick the wrong frame, the GIF writer ignores the size limit, or the pipeline fails end to end on one episode with a synthetic video | fast (~4 s) |

Helpers, not tests: `synth.py` renders synthetic camera images with known geometry; `robot.py` builds the SO-101 model for the vendored checks. `test_ik.py` and `test_jacobian.py` are the vendored scripts that `test_kinematics_vendored.py` calls.

Every test file also runs as a script: `python -m tests.<name>`.

## Running

Before every commit (about 1 min):

    python -m pytest -m "not slow"

Everything, including the slow tests (about 2.5 min):

    python -m pytest

Install pytest with `pip install -e ".[dev]"`.
