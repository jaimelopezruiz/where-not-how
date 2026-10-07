# PLAN: Humanoid intern challenge

**Deadline:** Fri 9 Oct 2026, 23:59 BST. **Target submit:** Fri 21:00 BST.
**Brief constraint:** data I personally collected must drive a robotic manipulator in simulation.
**Scored on:** creativity, policy performance in sim, implementation simplicity, clear results without slop. *See CHALLENGE.md file for details.*

## Thesis

Human demos tell the robot *what* should happen to the object; the robot learns *how*.
Record myself pushing a cube with a webcam, extract only the cube's trajectory, and use it
as the command for an RL pushing policy in MuJoCo. Compare against copying the hand
(open-loop fingertip replay) and a hand-coded pusher. At test time the policy takes a
newly recorded trajectory as input, so my data is what drives the robot.

## Priorities

- **P0 (submission floor):** data, extraction, sim, eval harness, scripted pusher, README, submit.
  With P0 done the repo meets the brief.
- **P1 (what makes it stand out):** RL policy, hand-replay baseline, showcase GIF.
- **P2 (stretch):** data-scaling curve, yaw tracking.

Rule: no P1 work until the P0 floor is green, except launching RL runs to train overnight.

## Status

Statuses: `todo` / `in progress` / `done` / `blocked` / `cut`.

| ID  | Task                               | Pri | Est.   | Day | Status |
|-----|------------------------------------|-----|--------|-----|--------|
| T0  | Repo + environment                 | P0  | 1 h    | Wed | done   |
| T1  | Data capture                       | P0  | 2–3 h  | Wed | in progress |
| T2  | Extraction pipeline                | P0  | 3–4 h  | Wed | todo   |
| T3  | Simulation env                     | P0  | 3–4 h  | Wed/Thu | in progress |
| T4  | Evaluation harness                 | P0  | 2 h    | Thu | todo   |
| T5  | Scripted pusher                    | P0  | 3 h    | Thu | todo   |
| T6  | RL policy                          | P1  | 6–8 h  | Thu/Fri | todo |
| T7  | Hand-replay baseline               | P1  | 2 h    | Thu | todo   |
| T8  | Showcase trajectory                | P1  | 1 h    | Fri | todo   |
| T9  | Data-scaling curve                 | P2  | compute| Thu night | todo |
| T10 | Yaw tracking                       | P2  | 2–3 h  | Fri | todo   |
| T11 | README / presentation              | P0  | 4–5 h  | Fri | todo   |
| T12 | Submission                         | P0  | 0.5 h  | Fri | todo   |

## Gates (decide at the time, log the decision)

- **G1, Wed after T3.1 (1 h timebox):** SO-101 MJCF loads and IK matches → use SO-101. Otherwise Panda from MuJoCo Menagerie.
- **G2, Thu 12:00:** scripted pusher completes held-out trajectories in sim → submittable floor reached. If not, all time goes to T5 until it does.
- **G3, Thu 23:00:** RL learning curve rising on the full train split → keep training overnight. If flat, switch T6 to residual RL on top of the scripted pusher (C6.5).
- **G4, Fri 18:00:** code freeze. README, GIFs and submission only after this.

## Workflow rules

1. Each commit leaves the repo runnable. Update the status table and decision log as you go.
2. Timeboxes are hard. When one expires, take the stated fallback and log why.
3. The train/test split is frozen (C2.6) before any method is evaluated. No tuning on test.
4. Record what didn't work as it happens. The README requires it and it's easy to forget by Friday.
5. Write the README yourself.
6. T3, T5 and C6.1 do not wait for real data: develop them on synthetic reference paths
   (line, arc, S-curve) generated in code. This is not tuning on test; the split doesn't exist yet.
7. Each module lands with its test in the same commit.
   `python -m pytest -m "not slow"` passes before every commit.

## Schedule

- **Wed:** before campus: T0, C1.1 (monitor checkerboard), G1, measure props, fix camera and lighting.
  Print markers on campus (C1.2), record in the afternoon while there's daylight (C1.3).
  Evening: T2 on the recordings; C3.2, C3.5 and T5 on synthetic paths (rule 6).
- **Thu:** finish T3, T4, T5 → G2 at 12:00. Then T6 and T7. Launch overnight runs (T6 full, T9).
- **Fri:** evaluate, T8, optional T10 until 18:00 (G4). T11, clean-clone test, T12 by 21:00.

----

## T0: Repo + environment (P0)

- [x] **C0.1 Skeleton.** New public repo. `pyproject.toml` with pinned deps: `mujoco`, `gymnasium`,
  `stable-baselines3`, `opencv-contrib-python`, `mediapipe`, `numpy`, `scipy`, `matplotlib`, `imageio`.
  Layout: `capture/`, `extract/`, `sim/`, `control/`, `rl/`, `eval/`, `scripts/`, `data/processed/`.
  `.gitignore` raw videos. README stub stating the data constraint. MIT `LICENSE` for my code;
  any copied third-party assets (robot MJCF, meshes) keep their own LICENSE files beside them.
  Add `capture/make_markers.py` (generates the print sheet).
- [x] **C0.2 Kinematics.** Bring in the PoE/DLS library: git dependency if the teleop repo is public,
  otherwise copy the needed modules with a note on origin. Watch the known
  `opencv-contrib-python` vs `opencv-python-headless` clash; force-reinstall contrib if anything strips it.

## T1: Data capture (P0)

- [ ] **C1.1 Camera calibration.** Webcam (C270, fixed focus): set the recording resolution and lock exposure
  first. Show a checkerboard full-screen on the monitor (measure one square with a ruler), or use page 3 of the
  print sheet on something rigid. 20–30 sharp frames covering centre, corners and tilts, at the recording
  resolution. `cv2.calibrateCamera` → `intrinsics.npz`. Accept if reprojection error < 1 px.
  Intrinsics don't depend on camera placement, so this can be done before the camera's final position is set.
  Tooling written and tested on synthetic views: `python -m capture.calibrate --square-mm <measured> --width W --height H`
  (writes `data/intrinsics.npz`; `--from-frames` recomputes from saved frames). **Calibration run still pending the camera.**
  If using printed page 3: it came out at ~95% scale, so the nominal 22 mm square is ~20.9 mm. Measure across
  several squares and pass that. Square size only sets the board's metric scale, not the intrinsics.
- [ ] **C1.2 Props.** Print `make_markers.py`'s sheet at 100% / Actual size; check the 100 mm bar.
  **Printed 2026-10-07: the bar measures 95 mm (printer fit-to-page). Not reprinted:** uniform scaling is harmless
  because every size comes from `props.yaml`. Check it is uniform: board span ID 0→1 (horizontal) and ID 0→2
  (vertical) should both be ~152 mm and agree within ~0.5 mm, else reprint (anisotropic scale skews pose).
  Measure long spans, not single markers: board marker side = (span − gap) / 2 from the ~152 mm span and one
  ~38 mm gap; cube marker edge to edge (~38 mm for the 40 mm print). No 0.95 factor in code.
  Object: rigid, flat square-ish top ≥ 40 mm, height under about half its width (slides, doesn't tip), smooth base,
  ~50–300 g, matte, not round. A square sticky-note pad (76 mm, ~100 g) fits the 50 mm marker. Sim box uses
  measured dimensions, so a non-cube box is fine.
  Table board: GridBoard 2×2, DICT_4X4_50, IDs 0–3, 60 mm markers, 40 mm gap, uncut, taped flat on all
  edges *beside* the workspace (not under it: paper changes friction and can shift).
  Cube: ID 10, largest size whose dashed line fits the top face, glued flat, edges aligned with the cube's.
  No clear tape over markers (glare). Measure marker side, gap, cube side and mass with a ruler and scale;
  write the measured values, not nominal, in `data/props.yaml`.
  Live check before recording: cube still → pose jitter ≲ 1 mm; during a push, no marker dropouts
  (push low on the side face so the finger doesn't cover the top marker).
  Tooling: `python -m capture.live_check --width W --height H` (needs `data/intrinsics.npz` and measured
  values in `data/props.yaml`, which currently holds `null` placeholders; the scripts refuse to run on them).
- [ ] **C1.3 Recording protocol.** Webcam fixed (same placement for every clip), angled down 45–60°, whole board and workspace in frame.
  30–40 episodes, slow quasi-static pushes, index finger for most:
  straight ×10, curve ×10, turn ×8, natural multi-finger ×5 (labelled), plus 3–5 "letter" traces kept
  aside for the showcase. Hand out of frame for 1 s at start and end of each clip.
  Log `data/manifest.csv` (id, category, notes). Upload raw videos to a Hugging Face dataset (CC BY 4.0,
  plain files, not LeRobot format); keep only processed data in git. Not critical path: a sample upload is
  enough until Friday.

## T2: Extraction pipeline (P0)

- [ ] **C2.1 Table frame.** Detect board each frame, `solvePnP` → camera pose in table frame.
  Per-frame pose absorbs small tripod shake.
- [ ] **C2.2 Cube pose.** `solvePnP` on the cube marker, transform to table frame, drop to (x, y, yaw).
  Do not push the cube marker through a table-plane homography: it sits at cube height and parallax biases position.
  **Test (C2.1 and C2.2):** synthetic render with a known cube path (reuse `tests/synth.py`); the extracted
  (x, y, yaw) in the z-up table frame matches. Catches OpenCV's y-down/z-in frame leaking through, and a flipped yaw sign.
- [ ] **C2.3 Fingertip.** MediaPipe Hands landmark 8, back-project the ray, intersect with the plane
  z = fingertip radius (~8 mm). Needed only for T7.
- [ ] **C2.4 Cleaning.** Drop low-confidence frames, interpolate gaps ≤ 5 frames, Savitzky–Golay filter,
  resample to the sim control rate. Contact window from cube speed threshold.
  **Test:** gaps of ≤ 5 frames are interpolated, longer gaps are left as gaps; yaw is unwrapped before filtering.
- [ ] **C2.5 Output + checks.** `data/processed/ep_XXX.npz` with `t, cube_xy_yaw, finger_xy, contact`.
  Plot all trajectories on one figure. Sanity: recovered cube size matches the measured one.
  **Test:** a schema validator (keys, shapes, metres, increasing time, recovered cube size vs `props.yaml`),
  run on every real episode as well as in the tests.
- [ ] **C2.6 Freeze split.** Seeded 80/20 train/test by episode, stratified by category; showcase set separate.
  Write `data/splits.json` and commit before any method runs.
  **Test:** same split for the same seed, no train/test overlap, showcase episodes excluded.

## T3: Simulation env (P0)

- [x] **C3.1 Robot model (G1, 1 h).** Look for an SO-101 MJCF (TheRobotStudio SO-ARM repo, LeRobot assets).
  Fallback: `franka_emika_panda` from MuJoCo Menagerie.
  **G1 passed: SO-101.** MJCF in `sim/assets/so101/`; `python -m sim.fk_check` compares PoE FK with MuJoCo
  (1000 random configs: max 0.0038 mm, mean 0.0018 mm). Tables in `results/`.
- [ ] **C3.2 Scene.** Table, cube with measured size and mass, small capsule pusher at the end effector
  for clean single-point contact. Start friction values, then tune: cube must slide, not tip or stick.
  **Test:** a constant push makes the cube slide without tipping.
- [ ] **C3.3 EE controller.** Action = planar EE velocity at fixed height → target → DLS IK → position actuators.
  SO-101: your library, position-only IK. Panda: MuJoCo site Jacobian with the same damped pseudo-inverse.
  Verify your FK against the MuJoCo EE site on random configs; save the error table for the README.
  FK check done under G1 (`sim/fk_check.py`). Two IK findings the wrapper must handle (see decision log):
  take the target orientation from the current configuration, and re-check FK on the returned angles.
  **Test:** the IK wrapper rejects a "converged" solution that lands on a joint limit and misses by > 1 mm.
  This is the regression test for the clamp finding in the decision log.
- [ ] **C3.4 Workspace map.** One affine map (translation + uniform scale ≤ 1) from human table frame to robot
  workspace, shared by all episodes. Report how many trajectories fall outside reach.
- [ ] **C3.5 Gymnasium env `PushTrack-v0`.** Reset samples a trajectory from a given split, places the cube at
  its start pose, pusher at standby. `render()` returns RGB for GIFs. Observation, action and reward as in T6.
  **Test:** gymnasium's `check_env` passes; a seeded reset gives the same result twice.

## T4: Evaluation harness (P0)

Built before any method so every row of the results table is measured the same way.

- [ ] **C4.1 Metrics.** Mean deviation from the reference path (cm), final position error (cm),
  success (final error < 2 cm and progress ≥ 90%), completion time. Breakdown by category.
  **Test:** hand-computed cases: perfect tracking scores 0, a constant offset gives the known deviation, and
  results just either side of the success threshold.
- [ ] **C4.2 Runner.** `evaluate(controller, split) → results.csv`, trajectory overlay plots,
  GIF writer with real video and sim render side by side.
- [ ] **C4.3 Smoke test.** One synthetic episode through extract → env → scripted pusher → metrics in under 30 s,
  not marked slow. This is also what C11.2's clean-clone test runs.

## T5: Scripted pusher (P0, critical path to G2)

- [ ] **C5.1 Controller.** Lookahead point on the reference path ahead of the cube's progress, push direction d.
  Pusher goal = cube − (half side + margin)·d. Approach by arcing around the cube, not through it.
  Push along d with lateral correction.
- [ ] **C5.2 Evaluate** on the test split. First row of the results table.

## T6: RL policy (P1)

- [ ] **C6.1 Phase tracker + reward, with unit tests on synthetic paths.**
  Progress-indexed, not time-indexed: phase p advances while the cube is within r_adv (~1.5 cm) of ref[p].
  Reward = progress gain − α·lateral deviation − β·‖pusher − cube‖ (reach shaping) + success bonus.
- [ ] **C6.2 Observation/action.** Obs: EE xy, cube (x, y, sin θ, cos θ), next k reference points in the cube frame.
  Action: clipped EE Δx, Δy.
- [ ] **C6.3 Overfit one trajectory.** PPO (SB3), 8 subprocess envs, `VecNormalize`. Must succeed within
  ~10–20 min. If it can't learn one path, fix the env or reward before scaling.
- [ ] **C6.4 Full train** on the train split, fixed seeds, TensorBoard logs, checkpoints. Evaluate at G3.
- [ ] **C6.5 Fallback: residual RL.** Action = scripted pusher + learned residual. Matches the brief's
  "bootstrap a policy and use RL" suggestion; use only if G3 fails, and say so in the README.

## T7: Hand-replay baseline (P1)

- [ ] **C7.1** Fingertip path through the same workspace map, followed open-loop at the recorded timing.
  Evaluate with T4. Expected to fail; the failure modes are evidence for the thesis, so capture GIFs of them.

## T8: Showcase (P1)

- [ ] **C8.1** Run the best controller on the held-aside letter traces. Side-by-side GIF for the top of the README.

## T9: Data-scaling curve (P2)

- [ ] **C9.1** Train on 5 / 10 / 20 demos, 2 seeds each, same held-out test set. Overnight Thursday. One plot.

## T10: Yaw tracking (P2)

- [ ] **C10.1** Add orientation error to reward and metrics. Only if position tracking is solid by Friday midday.

## T11: README / presentation (P0)

- [ ] **C11.1 Structure.** Showcase GIF; one-paragraph thesis; data collection (what, how, how much, link to raw);
  method; results table (hand replay / scripted / RL); design choices with rationale
  (object vs hand, progress indexing, RL in EE space with analytic IK); what didn't work; how to run; limitations.
- [ ] **C11.2 Clean-clone test.** Fresh venv, follow the README literally, confirm the eval script runs.

## T12: Submission (P0)

- [x] **C12.0** Plan lives in `docs/PLAN.md`, so the README is what reviewers see first.
- [ ] **C12.1** Repo public, README renders, links work. Apply with name, CV (robotics variant) and repo URL by 21:00 Fri.

----

## Decision log

(Append as decisions are made: date, decision, reason.)

- 2026-10-07: Pushing over grasping. Removes fragile sim grasping and keeps RL tractable on a laptop.
- 2026-10-07: Object trajectory as the policy's command, hand only as a baseline. Avoids the embodiment gap by construction.
- 2026-10-07: Per-demo leave-one-out attribution rejected (RL retrain cost, seed variance). Scaling curve instead (T9).
- 2026-10-07: Webcam (C270) instead of phone. Already mounted; fixed focus removes one calibration variable.
- 2026-10-07: ArUco over colour segmentation + A4-corner homography. ArUco gives metric pose at an oblique
  angle, per-frame table pose, unambiguous yaw, and fails visibly on occlusion instead of biasing the centroid.
  Colour + homography would be the documented fallback (needs near-overhead camera, blue/green object, yaw mod 90°).
  Downstream code only consumes `cube_xy_yaw`, so the tracker is swappable.
- 2026-10-07: MIT for code, CC BY 4.0 for the raw-video dataset.
- 2026-10-07: C0.2: vendored the PoE/DLS library into `control/kinematics/` instead of a git dependency. The
  teleop repo is public, but its distribution requires `lerobot[feetech]==0.5.1` (the source of the
  `opencv-python-headless` clash) and installs generic top-level packages `kinematics`, `urdf`, `tests`.
  Only `ik.py`'s import line changed; the vendored tests give output byte-identical to the original's seeded
  run. Origin, commit and licenses (Modern Robotics MIT, SO-ARM100 Apache-2.0) in `control/kinematics/ORIGIN.md`.
- 2026-10-07: Environment: conda env `where-not-how`, Python 3.11 (`conda create -n where-not-how python=3.11`,
  then `pip install -e .`). Dependencies in `pyproject.toml` pinned to what resolved, including
  `opencv-contrib-python==5.0.0.93` (OpenCV 5: no `estimatePoseSingleMarkers` or `drawAxis`; use `solvePnP` and
  `cv2.drawFrameAxes`), `torch==2.14.1` (SB3 backend, pinned for reproducibility) and `pyyaml` (for
  `data/props.yaml`, not in the original C0.1 list). `cv2.aruco` and `cv2.imshow` verified after install.
- 2026-10-07: Added a top-level `tests/` directory (not in the C0.1 layout). C6.1 needs unit tests and the
  vendored kinematics tests need a home. Run as `python -m tests.<name>`.
- 2026-10-07: OpenCV's `GridBoard` frame is x right, y down, z into the table (origin at marker 0's corner), so
  a cube on the table has negative z and yaw is clockwise-positive seen from above. C2.1 must define the
  table frame explicitly (z-up) rather than inherit this.
- 2026-10-07: C1.1 tooling tested on synthetic OpenCV renders with known geometry (`tests/synth.py`). Calibration
  recovers fx, fy within 0.2% and the principal point within ~2 px (25 views, RMS 0.35 px). `--fix-k3` added because
  with few frames k2/k3 absorb noise (k3 came out near -1 against a true 0) at identical reprojection error.
  Pose maths in `capture/live_check.py` is exact on exact corners (error ~1e-4 mm), so frame conventions, yaw sign
  and z are verified. On rendered images the cost is detector corner accuracy: ~0.25 px rms corner scatter gives up to
  ~3.6 mm in y and ~3.2 mm in z, 1.6 mm in x and 0.06 deg in yaw (oblique ~45 deg view, 40 mm cube marker,
  0.6 m; line-of-sight position is the weak direction), but a still cube's jitter was 0.05 / 0.18 mm. Repeatability is
  fine; absolute accuracy of a few mm is the limit, so a larger cube marker helps. These are synthetic
  numbers; the real figures come from running `live_check` on the camera.
- 2026-10-07: G1 passed, use SO-101 (took minutes of the 1 h box). MJCF and PoE URDF are the same CAD export
  (URDFs byte-identical). Joint order, zeros and signs resolved from the files: MuJoCo's world-frame joint axes and
  anchors at q=0 give screws matching the PoE `Slist` to 1e-5, all joint axes are +z in their body frames, ranges agree
  to 4e-6 rad. FK position error vs the MuJoCo `gripperframe` site over 1000 random configs (seed 0): max 0.0038 mm,
  mean 0.0018 mm, p99 0.0036 mm (the residue is `rpy 1.5708` against full-precision quaternions). The site is
  rotated 90 deg from the URDF `gripper_frame_link` by a constant offset (spread 0.001 deg), irrelevant to
  position-only IK. A flipped elbow sign is detected as centimetres of error (negative control in the tests).
- 2026-10-07: IK findings from G1 (position-only DLS evaluated in MuJoCo, 200 targets, start 0.3 rad away, ev 1 mm).
  (a) A fixed target orientation (home pose) converged only 61% of the time, because the orientation is
  unreachable at most targets; taking the target orientation from the start configuration converged 100%.
  (b) `IKinBodyDLS` clamps to joint limits *after* evaluating the error, so a "converged" solution that hit a limit is
  returned clamped but verified unclamped: 8 of 200 converged solutions missed by more than 1 mm (max 4.9 mm), and
  all 8 have a joint exactly on its limit; no solution away from a limit missed by more than 0.99 mm. The library is
  not modified (wrap, don't rewrite): the C3.3 wrapper re-checks FK on the returned angles and treats a miss as a
  failure or re-solves. Worth fixing upstream in the teleop repo too.
- 2026-10-07: Marker sheet printed at 95% (campus printer fit-to-page). Kept rather than reprinted: pose code takes
  every size from measured `props.yaml` values, and uniform scale doesn't change the marker bit patterns. Only an
  anisotropic print would force a reprint, hence the two-axis span check in C1.2.
- 2026-10-07: OpenCV 5 differences hit so far: `findChessboardCornersSB` returns corners as (N, 2), not (N, 1, 2);
  ArUco corner refinement defaults to none, so `live_check` sets `CORNER_REFINE_SUBPIX` (improves synthetic
  error from ~5-8 mm to ~3 mm).
- 2026-10-07: pytest is the single test command (`pytest==9.1.1`, `dev` extra; `slow` marker for tests over ~5 s;
  `python -m tests.<name>` still works). Two gaps found in the vendored checks: `test_ik.py` and `test_jacobian.py`
  had no `test_` functions, so pytest skipped them, and `test_jacobian.py` printed FAIL but always exited 0, so it
  could never fail. `tests/test_kinematics_vendored.py` wraps both with real assertions (one-line change to
  `verify_jac` to return its error; `test_ik` is called unmodified). Slow: the full IK sweep (~24 s); a fast subset
  of it (noise <= 1.0, same seeds, ~4 s) is in the commit gate.

## What didn't work

(To be filled  in as it happens.)

- G1, first hypothesis wrong: the position-only stop criterion (linear part of the body twist is not the position
  error when rotation error is large) was the first suspect for IK misses. The maths says that part can only
  over-estimate the error, and the data pointed instead at the joint-limit clamp running after the error check.
- Fixed target orientation for position-only IK: 61% convergence against 100% with the start configuration's own
  orientation.
- Cube-marker pose accuracy on synthetic renders is limited by the ArUco corner detector to a few mm in y and z at a
  45 deg view (jitter is far smaller). Own `cornerSubPix` pass cut it by about a third; not adopted without real images.
