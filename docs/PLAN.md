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
| T1  | Data capture                       | P0  | 2–3 h  | Wed | done   |
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

- [x] **C1.1 Camera calibration.** Webcam (C270, fixed focus): set the recording resolution and lock exposure
  first. Show a checkerboard full-screen on the monitor (measure one square with a ruler), or use page 3 of the
  print sheet on something rigid. 20–30 sharp frames covering centre, corners and tilts, at the recording
  resolution. `cv2.calibrateCamera` → `intrinsics.npz`. Accept if reprojection error < 1 px.
  Intrinsics don't depend on camera placement, so this can be done before the camera's final position is set.
  Tooling written and tested on synthetic views: `python -m capture.calibrate --square-mm <measured> --width W --height H`
  (writes `data/intrinsics.npz`; `--from-frames` recomputes from saved frames).
  **Done 2026-10-07.** Page 3 printed at 94.2%: 186.5 mm over 9 squares gives `--square-mm 20.72`. The C270 is
  DirectShow index 2 (`--source 2`). 25 frames, recomputed with `--fix-k3`: RMS 0.925 px, fx 1462.0, fy 1461.2,
  cx 701.0, cy 333.7, k1 0.144, k2 0.186. About 0.59 px of the 0.92 is the same at each board corner in every
  frame (board not perfectly flat). Coverage is centre-heavy with tilts up to 30 deg. Accepted: the live check
  (C1.2) gives sub-mm jitter and a z check within 2 mm, and positions are measured relative to the board.
- [x] **C1.2 Props.** Print `make_markers.py`'s sheet at 100% / Actual size; check the 100 mm bar.
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
  values in `data/props.yaml`; the scripts refuse to run on nulls).
  **Done 2026-10-07.** Print scale 0.9406 on both paper axes (sheet 1: 150.5 mm over 160; sheet 3: 186.5 over 198),
  so board marker 56.4 mm, gap 37.6 mm. Object: AirPods box 81 x 81 x 32.5 mm, 80 g, 50 mm print marker
  (47.0 mm measured). Board taped at the far end of the table, pushing area in front of it.
  Capture settings for everything from here on: `--source 2 --width 1280 --height 720 --exposure -7 --fourcc MJPG`
  (-7 with the room light on for recording; -6 was right under the dimmer calibration lighting)
  (30 fps; without MJPG the C270 sends YUY2 at 7.5 fps).
  Live check: box still, std x 0.05-0.11 mm, y 0.06-0.20 mm, yaw 0.02-0.05 deg. Three test pushes (straight x,
  straight y, curve): 0 dropouts. A 300 mm move along a ruler read 304.5 mm (1.5%, likely hand placement).
  Box-top positions at four places fit one plane to < 1 mm, tilted 3.5 deg from the board frame (see C2.1).
- [x] **C1.3 Recording protocol.** Webcam fixed (same placement for every clip), angled down 45–60°, whole board and workspace in frame.
  30–40 episodes, slow quasi-static pushes, index finger for most:
  straight ×10, curve ×10, turn ×8, natural multi-finger ×5 (labelled), plus 3–5 "letter" traces kept
  aside for the showcase. Hand out of frame for 1 s at start and end of each clip.
  Log `data/manifest.csv` (id, category, notes). Upload raw videos to a Hugging Face dataset (CC BY 4.0,
  plain files, not LeRobot format); keep only processed data in git. Not critical path: a sample upload is
  enough until Friday.
  Tooling to write: `capture/record.py`, same camera options and exposure lock as live_check (MJPG default),
  keypress start/stop per episode, `data/raw/ep_XXX.avi` (MJPG, top quality) plus `ep_XXX_t.csv` with per-frame
  `perf_counter()` timestamps, category prompt appending to `data/manifest.csv`, live preview with REC and fps,
  refuses to start if the frame size doesn't match `data/intrinsics.npz`. Verify on a 5 s clip run offline.
  **Tooling written** (`python -m capture.record`; defaults are the C1.2 capture settings). Keys in the preview
  window: `s c t m l` sticky category (straight, curve, turn, multi, showcase), `d` discard (still logged), `n` notes,
  SPACE start/stop, `q` quit. Also writes `ep_XXX_meta.json` (source, size, camera format, exposure read-back, measured
  fps, sha256 of `intrinsics.npz`, props). REC is refused below `--min-fps 20` (the YUY2 fallback). `--test` writes
  `test_XXX.*` with no manifest row. Tested offline; **episodes still to record.**
  **Test:** `tests/test_record.py`: video frames == CSV rows == frames fed, pose round-trip through the written
  video, frame-size refusal before any file exists, an existing episode is never overwritten.
  **Done 2026-10-07.** 72 episodes recorded (ep_000-ep_071), 38 kept: straight 10, curve 11, turn 8, multi 5,
  showcase 4 (L, U, S and a C traced NL -> MR -> FL, which reads as a C from the board side). 34 marked `discard`
  in the manifest (bad takes, plus 6 removed by the QA pass below). An earlier batch of 6 that broke the one-finger
  rule is in `data/raw/_practice/` with its own manifest and is not used. Recorded at `--exposure -7`.
  QA pass over every kept episode (detection on the saved video): box visible in 100% of frames in 37 of 38, no
  gap over 5 frames mid-push, max frame-to-frame step <= 5 mm, max tilt 11 deg. Discarded by QA: ep_019 and
  ep_022 (first 2.5-3.1 s of the push hidden), ep_023 (0.9 s hidden mid-push, box moved 57 mm), ep_052 (6.5 s
  hidden), ep_032 and ep_039 (camera moved while the board was hidden; see decision log). ep_030, 034, 049, 050
  kept with the note `board occluded ~3-4 s` (camera static, so harmless). In 11 episodes the box was already
  moving in the first frame, so extraction must not assume a still start.

## T2: Extraction pipeline (P0)

- [ ] **C2.1 Table frame.** Detect board each frame, `solvePnP` → camera pose in table frame.
  Per-frame pose absorbs small tripod shake.
  The board's orientation estimate is the weak part (3.5 deg tilt found in the C1.2 live check), so: fit a plane
  to all box-top positions (the box always lies flat), take the table normal from that fit, and use the board only
  for the origin and the x direction projected onto the plane. Report the plane-fit residual (< 1 mm at C1.2).
  The camera is not fixed between episodes and occasionally moved within one (decision log, C1.3 QA), so the board
  pose must be per frame, never one pose per session. Use only frames with all 4 board markers: with 2 markers
  visible the board depth wanders 10-25 mm with no real motion. Hold or interpolate the 4-marker pose across
  occlusions (the arm often hides the board for 3-7 s). Per episode, report the board's pixel drift from first to
  last frame as a QA figure (0.0-0.3 px for a static camera).
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
  The object is a box, not a cube (81 x 81 x 32.5 mm): add `height_mm` to `load_props` and use it for the geom.
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
- 2026-10-07: Object is an AirPods box, 81 x 81 x 32.5 mm, 80 g: square top fits the 50 mm marker, height/width
  0.40 so it slides without tipping. `props.yaml` gains `height_mm` (not yet read by `load_props`).
- 2026-10-07: Calibration recomputed with `--fix-k3`: identical RMS (0.925 vs 0.924 px) and k2 drops from 0.88 to
  0.19, so the free k3 was only cancelling k2. Accepted at 0.925 px rather than recalibrating with a stiffer board.
- 2026-10-07: `open_capture` now sets the format after the frame size. On DirectShow, setting the size after the
  FOURCC reset it to YUY2, which the C270 delivers at 7.5 fps at 1280x720; MJPG gives 29.9 fps. Static jitter
  with MJPG stays sub-mm.
- 2026-10-07: Tracking accuracy (C1.2 live check). Static std <= 0.2 mm and 0.05 deg. Box-top z ranged -17 to -26 mm
  across the table against -32.5 expected; the four positions fit z = -34.2 - 0.057x - 0.022y with < 1 mm
  residual, i.e. the box moves on a flat plane tilted 3.5 deg from the board frame, and the plane sits 1.7 mm from
  the expected height at the board origin. Cause: the board's orientation estimate (15 cm board, oblique view).
  Effect on (x, y): 1 - cos 3.5 deg = 0.2% plus ~2 mm offset. To be handled in extraction by fitting the plane (C2.1).
- 2026-10-07: Video writer for `record.py` (C1.3). Plain `cv2.VideoWriter(path, MJPG, ...)` uses the FFMPEG backend at
  a low fixed quality (29 KB/frame at 1280x720): detection on the read-back clip shifted the pose by up to 6.4 mm,
  5.3 mm and 1.7 deg against the source frames, far outside the 0.05-0.2 mm static jitter. FFMPEG's quality cannot
  be set (`writer.set` returns False; `OPENCV_FFMPEG_WRITER_OPTIONS` with qmin/qmax, global_quality, q, qscale all
  gave byte-identical output). OpenCV's built-in encoder (`CAP_OPENCV_MJPEG`) takes a 0-100 quality through
  `writer.set(VIDEOWRITER_PROP_QUALITY, q)`: q=100 read back within 0.012 / 0.111 mm and 0.010 deg (258 KB/frame,
  7.7 MB/s at 30 fps on noisy synthetic frames); q=50 0.03 / 0.20 mm; q=10 0.10 / 0.58 mm. `record.py` selects that
  backend explicitly at quality 100 and refuses to run if the backend or quality read-back is anything else. The
  rate depends on image content, so startup prints the rate measured by encoding the first frames.
- 2026-10-07: Camera movement (C1.3 QA). Tracking the board's marker centres in raw pixels: static to 0.0-0.3 px
  within most episodes, but the camera shifted between episodes (several jumps of 20-50 px, mostly during
  discarded retakes) and within ep_018 (21 px), ep_039 (41 px, board 4.7% larger, ~35 mm closer) and a slow 3-7 px
  creep over ep_030-034. Positions are measured against the board each frame, so movement only corrupts data
  while the board is hidden: ep_018, 032, 039 discarded for that. Likely cause: the clip-on mount on a free-standing
  mirror, or the cable, being nudged while resetting the box. The 9 re-recorded episodes all show <= 0.2 px drift.
- 2026-10-07: Discard criteria for recorded episodes: box hidden for more than 5 frames while moving, start of the
  push hidden, or board hidden while the camera moved. Board occlusion with a static camera is kept with a note.

## What didn't work

(To be filled  in as it happens.)

- G1, first hypothesis wrong: the position-only stop criterion (linear part of the body twist is not the position
  error when rotation error is large) was the first suspect for IK misses. The maths says that part can only
  over-estimate the error, and the data pointed instead at the joint-limit clamp running after the error check.
- Fixed target orientation for position-only IK: 61% convergence against 100% with the start configuration's own
  orientation.
- Cube-marker pose accuracy on synthetic renders is limited by the ArUco corner detector to a few mm in y and z at a
  45 deg view (jitter is far smaller). Own `cornerSubPix` pass cut it by about a third; not adopted without real images.
- Camera index 0 (the scripts' default) is another device on this PC: `camera returned no frame`. Found the C270 by
  probing indices 0-3 on DirectShow and MSMF with it plugged and unplugged.
- Manual exposure: DirectShow reports auto-exposure mode -1 and the lock warning stays, though exposure reads back
  -6 as requested. The setting does take effect despite the warning: -6 to -7 visibly darkened the image.
- YUY2 at 7.5 fps: `--fourcc MJPG` had no effect until the format was set after the frame size.
- Wrong first diagnosis of the z offset: blamed the cube marker size (one 1.5% error would explain both the z offset
  and the ruler test). The marker measured 47.0 mm, and z varying with position pointed to the board's orientation.
- Default video writer: `cv2.VideoWriter(path, MJPG, ...)` silently degraded the clips (up to 6.4 mm pose shift on
  read-back). The first probe looked fine (0.07 mm) because its synthetic scene had only 4 of 30 frames detected;
  the scene was fixed before the result was believed. Quality set via the constructor parameter or via FFMPEG
  environment options did nothing either.
- `tests/test_live_check.py::test_refuses_null_props` depended on the real `data/props.yaml` holding nulls and failed
  once the measured values went in. It now builds its own null file.
- Recording: 34 of 72 takes discarded. Causes: first batch ignored the one-finger rule (moved to `_practice`),
  reaching over the board hid it, the hand covered the box marker at the start of a push, and pushing started
  straight after SPACE instead of after a 1 s still.
- QA first pass required board and box in the same frame and so reported board occlusions as box dropouts;
  the box was visible throughout in those episodes. Separating the two changed the verdict on 5 episodes.
