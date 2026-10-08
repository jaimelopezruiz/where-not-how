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

The principle itself (object motion as the embodiment-agnostic signal) is established: Im2Flow2Act
(CoRL 2024), HuDOR (arXiv 2410.23289), Human2Sim2Robot (CoRL 2025). The claims here are narrower:

1. One goal-conditioned policy over many recorded trajectories, scored on held-out recordings
   (Human2Sim2Robot trains one policy per task; Im2Flow2Act conditions on object flow from a generative model,
   not on recorded trajectories).
2. Non-prehensile pushing, where copying the hand is expected to fail hardest (T7 was to test this; it stopped at its coverage gate, see C7.1): the pusher-slider system is hybrid and
   underactuated (Hogan & Rodriguez, IJRR 2020).
3. A low-cost 5-DOF arm with closed-form PoE kinematics (DLS IK) validated against MuJoCo.
4. A progress-indexed (path-following) reward rather than time-indexed tracking (Aguiar, Kokotovic &
   Hespanha, IEEE TAC 2005).

## Priorities

- **P0 (submission floor):** data, extraction, sim, eval harness, scripted pusher, README, submit.
  With P0 done the repo meets the brief.
- **P1 (what makes it stand out):** RL policy, hand-replay baseline, showcase GIF, data-scaling curve.
- **P2 (stretch):** time-indexed reward ablation (C6.6), yaw tracking.

Rule: no P1 work until the P0 floor is green, except launching RL runs to train overnight.

## Status

Statuses: `todo` / `in progress` / `done` / `blocked` / `cut`.

| ID  | Task                               | Pri | Est.   | Day | Status |
|-----|------------------------------------|-----|--------|-----|--------|
| T0  | Repo + environment                 | P0  | 1 h    | Wed | done   |
| T1  | Data capture                       | P0  | 2–3 h  | Wed | done   |
| T2  | Extraction pipeline                | P0  | 3–4 h  | Wed | done   |
| T3  | Simulation env                     | P0  | 3–4 h  | Wed/Thu | done |
| T4  | Evaluation harness                 | P0  | 2 h    | Thu | done   |
| T5  | Scripted pusher                    | P0  | 3 h    | Thu | done   |
| T6  | RL policy                          | P1  | 6–8 h  | Thu/Fri | in progress (C6.5 training) |
| T7  | Hand-replay baseline               | P1  | 2 h    | Thu | cut (C7.1 coverage gate) |
| T8  | Showcase trajectory                | P1  | 1 h    | Fri | in progress (pipeline done) |
| T9  | Data-scaling curve                 | P1  | compute| Thu night | cut  |
| T10 | Yaw tracking                       | P2  | 2–3 h  | Fri | cut    |
| T11 | README / presentation              | P0  | 4–5 h  | Fri | todo   |
| T12 | Submission                         | P0  | 0.5 h  | Fri | todo   |

## Gates (decide at the time, log the decision)

- **G1, Wed after T3.1 (1 h timebox):** SO-101 MJCF loads and IK matches → use SO-101. Otherwise Panda from MuJoCo Menagerie.
- **G2, Thu 15:00:** scripted pusher completes held-out trajectories in sim → submittable floor reached. If not, all time goes to T5 until it does.
  **Passed Thu 12:08:** scripted pusher 6/7 on the test split (C5.2).
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
- **Thu:** finish T3, T4, T5 → G2 at 15:00. Then T6 and T7. Launch overnight runs (T6 full).
- **Fri:** evaluate, T8 until 18:00 (G4). T11, clean-clone test, T12 by 21:00.

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
  **Uploaded 2026-10-08:** the 38 kept episodes (`.avi`, `_t.csv`, `_meta.json`), `calib/` and `manifest.csv` to
  https://huggingface.co/datasets/JaimeLR/where-not-how-pushes (CC BY 4.0). Discarded takes, `test_*` clips and
  `_practice/` are not uploaded; the manifest lists the discards.
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

- [x] **C2.1 Table frame.** Detect board each frame, `solvePnP` → camera pose in table frame.
  Per-frame pose absorbs small tripod shake.
  The board's orientation estimate is the weak part (3.5 deg tilt found in the C1.2 live check), so: fit a plane
  to all box-top positions (the box always lies flat), take the table normal from that fit, and use the board only
  for the origin and the x direction projected onto the plane. Report the plane-fit residual (< 1 mm at C1.2).
  The camera is not fixed between episodes and occasionally moved within one (decision log, C1.3 QA), so the board
  pose must be per frame, never one pose per session. Use only frames with all 4 board markers: with 2 markers
  visible the board depth wanders 10-25 mm with no real motion. Hold or interpolate the 4-marker pose across
  occlusions (the arm often hides the board for 3-7 s). Per episode, report the board's pixel drift from first to
  last frame as a QA figure (0.0-0.3 px for a static camera).
  **Done 2026-10-07** (`extract/detect.py`, `extract/table.py`). Board pose per frame from frames with 4 markers and
  board reprojection RMS <= 1.2 px (clean frames sit at 0.72-0.78 px; partly covered markers give 1.4-5.6 px), then
  slerp + linear interpolation between them, nearest held at the ends. Table frame: origin = the board origin's foot
  on the fitted plane, x = board x projected onto it, z up, y = z cross x (the board's y flipped), yaw
  counter-clockwise. A straight push cannot fix the plane's tilt across the line (minor-axis spread 1-8 mm), so the
  fit is ridge-pulled to a prior slope, the median of the 23 episodes whose points spread >= 25 mm in every
  direction (b = -0.082, c = 0.000, tilt 4.7 deg). Weight on the prior: 0.84-1.0 in 14 episodes (the 10 straight
  pushes, and ep_036, 057, 059, 066 whose paths are near-collinear too), 0.02-0.28 in the other 24. Plane residual
  (rms): median 1.9 mm, max 4.9 mm, not the < 1 mm of the C1.2 live check (that was four stationary positions; here
  every frame counts, and the marker's line-of-sight position scatters). Plane tilt: 2.4-6.2 deg in the 16 episodes
  spread >= 35 mm; 1.3-11.7 over all. ep_014 (11.7 deg) is the outlier: its data dominated the fit at prior weight
  0.25, so its xy may carry up to ~2% scale error (1 - cos 11.7 deg against 0.3% at the typical tilt). Board-pixel
  drift first to last clean frame: 0.01-0.51 px in 32 of 38 episodes; ep_021 5.7, ep_034 4.6, ep_030 3.0, ep_055 2.3,
  ep_027 1.2, ep_049 0.9 (camera moved, absorbed by the per-frame pose). Board pose measured in 100% of frames in 30
  episodes, 35-81% in the other 8 (longest hold 9.9 s, ep_057). While held, partly visible board markers sit 2.5-3.9
  px (p95) from the held track in those 8 (sustained 0.5-1.3 px with brief spikes at the edges of an occlusion): up
  to ~1.5 mm of table position if the camera really moved, not separable from corner bias on half-covered markers.
- [x] **C2.2 Cube pose.** `solvePnP` on the cube marker, transform to table frame, drop to (x, y, yaw).
  Do not push the cube marker through a table-plane homography: it sits at cube height and parallax biases position.
  **Test (C2.1 and C2.2):** synthetic render with a known cube path (reuse `tests/synth.py`); the extracted
  (x, y, yaw) in the z-up table frame matches. Catches OpenCV's y-down/z-in frame leaking through, and a flipped yaw sign.
  **Done 2026-10-07.** Marker pose from `solvePnP` (IPPE_SQUARE + LM, as in `live_check`) in the per-frame board frame,
  then into the table frame. Cube detected in 100% of frames of all 38 episodes, no frame dropped by any gate.
  C1.3's QA said 37 of 38; the one miss is unidentified and not reproducible: marker 10 is found in every frame of all
  38 under OpenCV 4.13 and 5.0, with and without sub-pixel refinement (the QA script is not in the repo).
  **Tests** (`tests/test_extract.py`): analytic scene with a table tilted 4.5, -7 and 5 deg (about board y and x)
  and a 0 deg control, camera drifting and shaking, 2-marker poses 20 mm off during a 50-frame occlusion: x, y, yaw
  exact to 1e-6 including the held frames. Rendered board + cube through the detector: x, y within 5 mm, yaw within
  1 deg, with table y = -board y and table yaw = -board yaw. Recovered marker size (corner rays against the plane,
  so independent of the pose solver) matches the measured 47.0 mm: -1.8 to +3.5% over episodes, median +0.4%.
- [ ] **C2.3 Fingertip.** MediaPipe Hands landmark 8, back-project the ray, intersect with the plane
  z = fingertip radius (~8 mm). Needed only for T7, so it moves there (P1); T2 is done without it.
  **Code and tests done, not run on the data: blocked on a measurement.** `extract/fingertip.py` cuts the camera ray
  with z = tip radius - box height (in the marker-plane frame); `data/props.yaml` has no fingertip radius and the plan's
  "~8 mm" is a guess, so the code requires `finger: tip_radius_mm: <measured>` and has no default. Until it exists
  `finger_xy` is NaN in every `.npz`; then `python -m extract.fingertip --fetch-model` (once) and
  `python -m extract.run --finger`. Checked once on 5 episodes with a scratch 8 mm: during contact the fingertip sits
  a median 47-50 mm from the box centre (half side 40.5 + radius 8 = 48.5), so the geometry is right. Coverage is
  the problem, see "What didn't work".
  **Cut with T7 (2026-10-08):** the fingertip radius was never measured and `finger_xy` stays NaN.
- [x] **C2.4 Cleaning.** Drop low-confidence frames, interpolate gaps ≤ 5 frames, Savitzky–Golay filter,
  resample to the sim control rate. Contact window from cube speed threshold.
  **Test:** gaps of ≤ 5 frames are interpolated, longer gaps are left as gaps; yaw is unwrapped before filtering.
  **Done 2026-10-07** (`extract/clean.py`). Gates: cube reprojection RMS <= 1.5 px (clean frames 0.1-0.8), marker tilt
  <= 20 deg, a Hampel test (7 frames, 10 mm / 15 deg from the local median). None fired on the real data (0 frames
  dropped, 0 interpolated, no gap in 38 episodes): the gates are there for the next recording. Smoothing: Savitzky-Golay
  window 11, order 2, fitted on the CSV timestamps (the camera clock jitters up to 55 ms between frames; scipy's filter
  assumes uniform spacing and put up to 1 mm of error on linear motion in a test, so `savgol_times` fits the local
  quadratic at the true times; identical to scipy on uniform times, tested). Raw-to-smoothed residual 0.19-0.51 mm,
  so the filter removes little noise and mostly keeps the motion as measured. Output rate `CONTROL_HZ = 20`: the sim
  env may resample further (`t` is stored); change it in one place if T3 settles on another rate. Contact = smoothed
  cube speed > 5 mm/s, pauses < 0.5 s bridged, runs < 0.2 s dropped: 34-73% of each clip, first contact at 1.5-4.5 s,
  speed while pushing about 30-90 mm/s. The box is not assumed still at the start: the speed at the first sample is
  0.1-8 mm/s (ep_012 and ep_071 above the 5 mm/s contact threshold, and contact is read from the speed, not from t = 0).
- [x] **C2.5 Output + checks.** `data/processed/ep_XXX.npz` with `t, cube_xy_yaw, finger_xy, contact`.
  Plot all trajectories on one figure. Sanity: recovered cube size matches the measured one.
  **Test:** a schema validator (keys, shapes, metres, increasing time, recovered cube size vs `props.yaml`),
  run on every real episode as well as in the tests.
  **Done 2026-10-07.** `python -m extract.run` (about 2 min, detection runs in a process pool; `--cache-dir` reuses
  detections while developing) writes the 38 `data/processed/ep_XXX.npz`, `results/qa_extraction.csv` and
  `results/trajectories_overlay.png`. A full re-run from raw video reproduces the files exactly. The `.npz` holds the
  four planned keys plus `cube_marker_side` (the recovered marker edge in m, for the schema's size check; downstream
  ignores it). `finger_xy` is NaN until C2.3 runs. `python -m extract.schema` validates every file (38/38; also a
  test). Sanity on size: the plan asks for the recovered cube size; only the marker is observable (one marker on the top
  face), so that is what is checked, to 5%, and it is within 3.5% everywhere. Workspace covered: x -0.37 to -0.08 m,
  y -0.24 to 0.07 m in the table frame (board origin): the pushing area lies to the left of the board's origin.
  `qa_extraction.csv` has the asked columns (visibility, gaps, plane residual, board pixel drift) plus hold length,
  hold deviation, plane spread and prior weight, marker size, contact fraction, path length and a `notes` column of
  warnings.
- [x] **C2.6 Freeze split.** Seeded 80/20 train/test by episode, stratified by category; showcase set separate.
  Write `data/splits.json` and commit before any method runs.
  **Test:** same split for the same seed, no train/test overlap, showcase episodes excluded.
  **Done 2026-10-07** (`extract/split.py`, `data/splits.json`, seed 0). From the manifest alone, not from extraction
  results, before any method exists. Each non-showcase category gives max(1, round(0.2 n)) test episodes: 27 train,
  7 test (ep_005, ep_009 straight; ep_016, ep_057 curve; ep_027, ep_031 turn; ep_036 multi), 4 showcase held out
  entirely (ep_049, 050, 055, 071). Re-running refuses to overwrite a split that differs from what the manifest and
  seed give, and a test checks the committed file against a regeneration and against the processed episodes. The
  test set holds ep_057, one of the episodes with the board hidden for 9.9 s (board measured in 36% of frames).

## T3: Simulation env (P0)

- [x] **C3.1 Robot model (G1, 1 h).** Look for an SO-101 MJCF (TheRobotStudio SO-ARM repo, LeRobot assets).
  Fallback: `franka_emika_panda` from MuJoCo Menagerie.
  **G1 passed: SO-101.** MJCF in `sim/assets/so101/`; `python -m sim.fk_check` compares PoE FK with MuJoCo
  (1000 random configs: max 0.0038 mm, mean 0.0018 mm). Tables in `results/`.
- [x] **C3.2 Scene.** Table, cube with measured size and mass, small capsule pusher at the end effector
  for clean single-point contact. Start friction values, then tune: cube must slide, not tip or stick.
  The object is a box, not a cube (81 x 81 x 32.5 mm): add `height_mm` to `load_props` and use it for the geom.
  **Test:** a constant push makes the cube slide without tipping.
  **Done 2026-10-08** (`sim/scene.py`). Box 81 x 81 x 32.5 mm, 80 g from `props.yaml` (`cube_height` added to
  `load_props`). Table friction 0.3, pusher-box 0.8; 6 mm capsule on the gripper, centred 15 mm above the table.
  The test pushes at constant speed: the box slides > 20 mm with < 5 deg tilt. Gripper and jaw collision is off, so
  the capsule is the only contact (decision log).
- [x] **C3.3 EE controller.** Action = planar EE velocity at fixed height → target → DLS IK → position actuators.
  SO-101: your library, position-only IK. Panda: MuJoCo site Jacobian with the same damped pseudo-inverse.
  Verify your FK against the MuJoCo EE site on random configs; save the error table for the README.
  FK check done under G1 (`sim/fk_check.py`). Two IK findings the wrapper must handle (see decision log):
  take the target orientation from the current configuration, and re-check FK on the returned angles.
  **Test:** the IK wrapper rejects a "converged" solution that lands on a joint limit and misses by > 1 mm.
  This is the regression test for the clamp finding in the decision log.
  **Done 2026-10-08** (`control/ee_controller.py`). Planar EE displacement, position-only DLS IK, joint position
  targets. Target orientation from the current configuration; FK re-check rejects clamped misses (regression test).
- [x] **C3.4 Workspace map.** One affine map (translation + uniform scale ≤ 1) from human table frame to robot
  workspace, shared by all episodes. Report how many trajectories fall outside reach.
  **Done 2026-10-08** (`control/workspace.py`, `data/workspace_map.json`). Fitted on train only: scale 0.80, offset
  (0.447, 0.063) m, 27/27 train trajectories feasible at every point. Feasible = box centre >= 147 mm from the base
  axis (90 mm base footprint + 57 mm half-diagonal) and the pusher position behind the box within 12 mm of a
  reachable FK sample at push height (200 000 random configurations, 17 280 at push height, median spacing 1.9 mm).
- [x] **C3.5 Gymnasium env `PushTrack-v0`.** Reset samples a trajectory from a given split, places the cube at
  its start pose, pusher at standby. `render()` returns RGB for GIFs. Observation, action and reward as in T6.
  **Test:** gymnasium's `check_env` passes; a seeded reset gives the same result twice.
  **Done 2026-10-08** (`sim/push_env.py`). 20 Hz (25 substeps of 2 ms), 600-step limit (longest train episode 20.2 s
  + 50%). `PushTrackEnv(split=...)` maps that split's episodes with the stored workspace map and starts the box at the
  recorded start pose and yaw; `reset(options={"episode_id": ...})` selects one. Progress is arc length, as in
  `eval/metrics.py`; results are scored by `eval/metrics.py`, not by the env's success flag. `check_env` passes, seeded
  resets reproduce.

## T4: Evaluation harness (P0)

Built before any method so every row of the results table is measured the same way.

- [x] **C4.1 Metrics.** Mean deviation from the reference path (cm), final position error (cm),
  success (final error < 2 cm and progress ≥ 90%), completion time. Breakdown by category.
  **Test:** hand-computed cases: perfect tracking scores 0, a constant offset gives the known deviation, and
  results just either side of the success threshold.
  **Done 2026-10-08** (`eval/metrics.py`). Progress = furthest projection on the reference over its length. Mean
  deviation = distance to the reference weighted by box displacement, so time spent stationary does not lower it.
  Final error to the reference end point. Success = final error < 2 cm and progress >= 0.9.
- [x] **C4.2 Runner.** `evaluate(controller, split) → results.csv`, trajectory overlay plots,
  GIF writer with real video and sim render side by side.
  **Done 2026-10-08** (`eval/report.py`). `evaluate(controller, split)` runs one rollout per episode (parallel
  workers) and writes the results CSV, per-category summary and overlay plot. Side-by-side GIF writer exists; no
  real-vs-sim GIF made yet (T8).
- [x] **C4.3 Smoke test.** One synthetic episode through extract → env → scripted pusher → metrics in under 30 s,
  not marked slow. This is also what C11.2's clean-clone test runs.
  **Done 2026-10-08** (`eval/smoke.py`, `tests/test_smoke.py`). A rendered 14 cm curved push goes through detection
  and extraction, `PushTrackEnv(episode_files=...)` (new: explicit `.npz` files loaded like split episodes),
  `ScriptedPusher` and `evaluate()` (until-done); no `data/raw/` needed. Extracted path within 2.4 mm of the rendered
  truth (test asserts 6 mm); scripted pusher succeeds, final error 0.44 cm, progress 0.99. About 7 s.
  `python -m eval.smoke` prints the metrics and `SMOKE OK`.

## T5: Scripted pusher (P0, critical path to G2)

A closed-loop tracking controller on the object trajectory, no learning. Not equivalent to
object-aware replay (that is T7).

- [x] **C5.1 Controller.** Lookahead point on the reference path ahead of the cube's progress, push direction d.
  Pusher goal = cube − (half side + margin)·d. Approach by arcing around the cube, not through it.
  Push along d with lateral correction.
  **Done 2026-10-08** (`control/scripted_pusher.py`). Goal on the line through the box centre (where the ray along -d
  leaves the footprint, using the box yaw), approach around a circle clear of the half-diagonal, push with lateral
  correction and hysteresis. The step halves while the arm is not following (near the reach limit).
- [x] **C5.2 Evaluate** on the test split. First row of the results table.
  **Done 2026-10-08.** Tuned on train only (`results/scripted_tuning_log.csv`): lookahead 2.5 cm, push speed
  3.3 mm/step (6.6 cm/s, inside the recorded 3-9 cm/s). Train 27/27, mean deviation 0.53 cm. Test, run once at the
  final parameters: 6/7, mean deviation 0.59 cm. Failure: ep_027 (turn), the box overshot the corner and the pusher
  needed a position beyond reach; it stalled until the step limit (final error 20 cm). Evaluation runs each episode
  until the pusher reports done (box within 5 mm of the end) or the step limit, not to the env's first success:
  final error 0.38 cm on train, 0.34-0.38 cm by category on the six test successes (3.24 cm overall with ep_027).
  Re-run once more after the C6.3 scene change (arm links no longer touch the box), same parameters: train 27/27,
  mean deviation 0.53 cm, final error 0.37 cm; test 7/7, 0.61 cm, 0.39 cm. ep_027 now succeeds, which suggests its
  failure was the forearm hitting the box rather than the reach limit. These are the frozen scripted numbers.

## T6: RL policy (P1)

- [x] **C6.1 Phase tracker + reward, with unit tests on synthetic paths.**
  Progress-indexed, not time-indexed: phase p advances while the cube is within r_adv (~1.5 cm) of ref[p].
  Reward = progress gain − α·lateral deviation − β·‖pusher − cube‖ (reach shaping) + success bonus.
  **Done 2026-10-08** (`sim/push_env.py`). Phase tracker searches 10 cm ahead and advances while the box is within
  r_adv = 3 cm of the path (not 1.5 cm). Reach shaping targets the pre-contact point 5.15 cm behind the box along
  the push direction, weight 0.1 (distance to the box centre pulled the pusher into the box and off the path).
- [x] **C6.2 Observation/action.** Obs: EE xy, cube (x, y, sin θ, cos θ), next k reference points in the cube frame.
  Action: clipped EE Δx, Δy.
  **Done 2026-10-08.** As planned, k = 5 reference points spaced 2.5 cm in arc length from the phase (they were
  spaced by time samples, spanning ~0.5 cm, and identical in 40-70% of steps).
- [x] **C6.3 Overfit one trajectory.** PPO (SB3), 8 subprocess envs, `VecNormalize`. Must succeed within
  ~10–20 min. If it can't learn one path, fix the env or reward before scaling.
  **Done 2026-10-08** (`rl/train_ppo.py`, `runs/overfit_ep010_a3`). ep_010 (12 cm straight), seed 0: the deterministic
  policy succeeds from 200k steps (~10 min); final model progress 1.00, final error 0.38 cm, mean deviation 1.67 cm
  (scripted ~0.5 cm). Training success 0.27 at 86k, 0.90 at 410k. Two attempts failed first (decision log).
- [ ] **C6.4 Full train** on the train split, fixed seeds, TensorBoard logs, checkpoints. Evaluate at G3.
  Launched Thu 14:43 (`python -m rl.train_full --seed 0 --name full_s0`, in the t4 worktree): 8 envs, 10M steps,
  ~500 steps/s, eval every 100k on 8 train episodes (2 per category), checkpoints every 500k. At 1M steps: training
  success ~1%, return flat near -9.5 since 250k, deterministic eval does not move the box yet.
  At 1.1M (15:20): rollout success 2-4% (a 5-7% spell at 790-890k fell back to 0 by 950k), return about -8;
  deterministic eval 0/8 at every checkpoint, and at 0, 200k, 300k, 600k and 900k the pusher never touches the box.
  At 2M (G3 rule check, ~15:50): rollout success 2%, deterministic eval progress <= 0.03 (one eval shoved the box
  away), so T6 switched to C6.5. Stopped at 3M (16:15) on this PC and resumed from the 3M checkpoint on a second,
  identical PC (fresh clone, Python 3.11 venv + `pip install -e .`, no conda) to finish 10M as the record of plain PPO:
  `--resume --total-steps 7000000` (SB3 adds it to the current count). `progress.csv` for 0-3M saved as
  `progress_0-3M.csv` (SB3 overwrites it on resume).
- [ ] **C6.5 Fallback: residual RL.** Action = scripted pusher + learned residual. Matches the brief's
  "bootstrap a policy and use RL" suggestion; use only if G3 fails, and say so in the README.
  Residual RL on a base controller is used in HuDOR and ManipTrans.
  **Code done 2026-10-08** (`rl/residual.py`, `rl/train_residual.py`, `tests/test_residual.py`; `residual_scale`
  option in `rl/train_ppo.py` and `rl/eval_policy.py`, None = plain PPO). Executed step = scripted pusher step
  (frozen C5.2 parameters) + 0.25 x max_delta x a (at most 5 mm), clipped per component as the env does; the policy
  sees the env observation plus the base action / max_delta (18 values); log_std_init -2.0, other settings as C6.4.
  The pusher's stuck detection is told the executed command (`ScriptedPusher.executed`). Zero residual on train
  reproduces the frozen scripted result exactly (27/27, 0.534 cm, 0.367 cm). Sanity run (200k steps, 4 envs, 8 train
  eval episodes): 8/8 at every eval, deviation 0.55-0.65 cm (untrained 0.58; scripted 0.66 on these 8), final error
  0.35 -> 0.24 cm, episodes 147 -> 122 steps. Full run `residual_s0`: 8 envs, 3M steps, launched 16:25.
  `residual_s0` (stopped at 1M, 17:05, to free the CPU for C6.7): eval 8/8 at every point, rollout success
  0.98-1.0; train-eval mean deviation 0.52 cm untrained, 0.56-0.71 cm from 200k to 900k (one eval at 500k read 1.00,
  an outlier on 8 episodes); episodes 146 -> 103 steps. It learned to finish about 30% sooner, not to track better.
- [x] **C6.7 Deviation-weighted reward for the residual run.** `deviation_weight` on PushTrackEnv (None = C6.1
  reward): the per-step `0.1 x lateral` term becomes `w x (lateral / 1 cm) x (box displacement / path length)`,
  which sums to about w x mean deviation (cm), the metric's weighting, and does not pay for speed. w = 5, fixed in
  advance: 0.5 cm of extra mean deviation costs 2.5, about twice the ~1.2 that finishing 146 -> 108 steps earns
  through the discounted +10 bonus; the bonus equals 2 cm of mean deviation, the success threshold. Run
  `residual_dev5`, 3M steps. **Code done 2026-10-08** (`sim/push_env.py`, `rl/train_ppo.py`, `rl/train_residual.py`,
  `tests/test_push_env_reward.py`): None reproduces the C6.1 reward step for step; a box pushed at a constant 1 cm
  offset accumulates 5 x 1.0 within 10%; a stationary offset box gets 0. Sanity (100k, 4 envs): eval 8/8, 0.52-0.58 cm,
  143-147 steps. `residual_dev5` launched ~17:15 from t3. w is not retuned after seeing its curve.
- [ ] **C6.6 Time-indexed reward ablation (P2).** The same PPO setup with a time-indexed tracking reward
  (reference point at time t, not at progress p); compare on the test split. Turns claim (4) from a cited design
  choice into a result.

## T7: Hand-replay baseline (P1)

This is the "replay" baseline of Human2Sim2Robot and HuDOR's base policy (retargeted hand, open-loop). Since the
sim cube starts at the recorded start pose, their "object-aware" variant is identical here.

- [ ] **C7.1** Fingertip path through the same workspace map, followed open-loop at the recorded timing.
  Evaluate with T4. Expected to fail; the failure modes are evidence for the thesis, so capture GIFs of them.
  **Stopped at the coverage gate 2026-10-08** (`extract/finger_coverage.py`, `results/finger_coverage.csv`,
  `results/finger_coverage_summary.txt`). Stop rule set before the run: fewer than 4 of 7 usable test episodes.
  Usable = no fingertip gap over 0.5 s left in the contact window after bridging gaps up to 0.5 s (gaps at the window
  edges tolerated up to 0.5 s); nothing is inferred from the box. Detector settings chosen on 9 train episodes only:
  720 px crop around the box, confidence 0.1, video mode (mean in-window coverage 0.45 -> 0.72). Usable: train 5/27,
  test 3/7 (ep_009, ep_036, ep_057), showcase 0/4. No hand replay was run, so the results table has no hand-replay row.

## T8: Showcase (P1)

- [ ] **C8.1** Run the best controller on the held-aside letter traces. Side-by-side GIF for the top of the README.
  **Pipeline done 2026-10-08** (`scripts/showcase.py`, `tests/test_showcase.py`), run with the scripted pusher; the
  final controller is chosen Fri from the test results (`--run runs/<name>`, not yet run on a trained policy).
  Feasibility under the frozen map (C3.4 test): 4/4 showcase traces, 100% of points. Scripted, until-done: 4/4,
  mean deviation 0.48 / 0.58 / 1.00 / 0.55 cm (L, U, S, C), progress 0.99-1.00. GIFs: real video and sim on one
  clock at real speed, letters upright (C shown from the board side, rotation only), reference path dashed;
  `eval.report.write_gif` gave 19.5 MB per episode, so the script has its own palette-limited writer (< 8 MB).
  The sim pushes about twice as fast as the human (frozen 6.6 cm/s; ep_071 10 s against 18 s).

## T9: Data-scaling curve (P1)

Extra evidence for claim (1); the main evidence is the T6 policy scored on the test split. Cut for now: its six runs
would compete with the T6 full run for CPU overnight. Revisit at G3.

- [ ] **C9.1** Train on 5 / 10 / 20 demos, 2 seeds each, same held-out test set. Overnight Thursday. One plot.

## T10: Yaw tracking (P2)

Rotating the cube needs pusher contact-mode switching, so yaw stays P2.

- [ ] **C10.1** Add orientation error to reward and metrics. Only if position tracking is solid by Friday midday.

## T11: README / presentation (P0)

- [ ] **C11.1 Structure.** Showcase GIF; one-paragraph thesis; data collection (what, how, how much, link to raw);
  method; results table (hand replay / scripted / RL); design choices with rationale
  (object vs hand, progress indexing, RL in EE space with analytic IK); what didn't work; how to run; limitations.
- [ ] **C11.2 Clean-clone test.** Fresh venv, follow the README literally, confirm the eval script runs.
- [x] **C11.3 Results table and training curves** (`scripts/results_figures.py`, `tests/test_results_figures.py`):
  `results/results_table.md` from `results/<method>_test.csv` files, `results/training_curves.png` from run
  directories (concatenates `progress*.csv` of a resumed run). Mean deviation is displacement-weighted, so a box
  that never moves scores near 0: report progress and success alongside it (plain PPO's curve shows this).

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
- 2026-10-07: Replan at 21:00, about half a day behind (T2 not started Wed evening). G2 moved from Thu 12:00
  to Thu 15:00. T9 (data-scaling curve) and T10 (yaw tracking) cut; the time goes to the P0 floor and T6.
- 2026-10-07: Related-work repositioning. Object motion as the embodiment-agnostic signal is established (Im2Flow2Act,
  HuDOR, Human2Sim2Robot), so it is no longer claimed as the contribution. Claims narrowed to: (1) one
  goal-conditioned policy over many recorded trajectories, scored on held-out recordings; (2) non-prehensile
  pushing, where copying the hand is expected to fail hardest; (3) a low-cost 5-DOF arm with closed-form PoE kinematics validated
  against MuJoCo; (4) a progress-indexed reward rather than time-indexed tracking. T5, T7, C6.5 and T10 annotated
  with how they relate to that prior work. README is unchanged.
- 2026-10-07: T9 moved to P1 but stays cut: its six runs would compete with the T6 full run for CPU overnight, and
  the T6 test-split evaluation already supports claim (1). Revisit at G3. C6.6 (time-indexed reward ablation, P2)
  added: it turns claim (4) from a cited design choice into a result.
- 2026-10-07: "Hand as hint" considered, not planned: use the fingertip's approach side to choose where the pusher
  starts (after Human2Sim2Robot's pre-contact hand-pose initialisation).
- 2026-10-07: T2 table frame. Origin is the board origin's foot on the fitted marker-height plane, so (x, y) do not
  depend on the box height; z up, y the board's y flipped (the board frame is OpenCV's, y down, z into the table),
  yaw counter-clockwise. The plane is fitted per episode, with a ridge prior on its slope from the episodes that span
  the plane, because a straight push is a line and cannot fix the tilt across it. The camera moved between and
  within episodes, so a per-session plane would be wrong; the prior is only a fallback for the degenerate directions.
- 2026-10-07: Board pose gates and holds. A board pose counts only with all 4 markers and reprojection RMS <= 1.2 px
  (partly covered markers give 1.4-5.6 px with the depth jumping up to 5 mm). Holds are interpolated (slerp) in the
  camera frame. The hold is not verified directly while no marker is seen; QA reports where partly visible markers
  disagree with it (p95, a percentile because a half-covered marker's corners spike for a few frames at the edges
  of an occlusion).
- 2026-10-07: Smoothing on true timestamps (`savgol_times`) rather than scipy's index-based filter: the camera clock
  jitters up to 55 ms around 33 ms, and the index-based filter gave up to 1 mm of error on linear motion in a test.
- 2026-10-07: Marker pose from `solvePnP` as the plan says, not a ray cut with the fitted plane. Cutting the marker
  centre's ray with the plane would remove the line-of-sight scatter (plane residual median 1.9 mm), at the cost of
  using the same data twice (plane fit and position). Not done; the smoothed residual to the raw data is 0.2-0.5 mm
  and 2 mm is a tenth of the 2 cm success threshold. Candidate if tracking error turns out to matter.
- 2026-10-07: Control rate set to 20 Hz in `extract/clean.py` (`CONTROL_HZ`) since T3 has not fixed one; `.npz` files
  carry `t`, so the env can resample. To be reconciled when C3.5 lands.
- 2026-10-07: C2.3 does not guess the fingertip radius. `data/props.yaml` needs `finger.tip_radius_mm`; the code
  refuses without it (the "~8 mm" in the plan is a guess, and a scratch run with it gave a fingertip-to-box-centre
  distance of 47-50 mm, consistent). The hand model (`hand_landmarker.task`, 7.8 MB, sha256 checked) is downloaded to
  gitignored `data/models/` by `python -m extract.fingertip --fetch-model`.
- 2026-10-08: G2 passed at 12:08, ahead of the 15:00 gate: scripted pusher 6/7 on the test split.
- 2026-10-08: Workspace reachability. The first check used the convex hull of FK samples, which fills the dead zone
  around the base: it reported 27/27 at scale 1.0 with paths running over the base. Replaced by nearest-sample
  reachability plus base clearance and pusher reach behind the box. At 20 000 random configurations only ~1 700 land
  at push height and the sample has holes, so the fit uses 200 000. The 12 mm tolerance (~6x the sample spacing) is
  the remaining slack at the reach boundary.
- 2026-10-08: Gripper and jaw collision disabled in the sim; the 6 mm capsule is the only contact. With the meshes on,
  the gripper hit the box about 4 cm before the capsule did. A modelling simplification, stated in the README; the
  alternative is a longer capsule below the gripper.
- 2026-10-08: The observation's EE position is FK of the measured joint angles, not the last IK target, which led the
  arm by up to 3 cm while moving.
- 2026-10-08: Env speed. Normal rollouts ran at ~220 steps/s; the slowdown was failed IK solves, where the library runs
  all 200 iterations (~3 ms each) before giving up: 8.5 steps/s when pushing into the base. The wrapper now runs
  `IKinBodyDLS` in chunks of 2 iterations from the measured joints and stops when the error stops falling (cap 12);
  same solutions as the 200-iteration call at the env's 2 cm step. 154-256 steps/s in every case tested.
- 2026-10-08: Evaluation protocol: `evaluate(..., until_done=True)` runs past the env's first success until the
  controller reports done or the step limit, so final error is where the box ends. The first C5.2 run stopped at the
  first success (final error capped near 2 cm); re-run once with the same parameters, success and deviation unchanged.
- 2026-10-08: Arm links no longer collide with the box (they still collide with the table and each other). In the
  first C6.3 attempt the policy reached 38% training success with its EE 17 cm from the box: it swept the box with
  the forearm. With the gripper change, only the 6 mm capsule touches the box; stated as a simplification in the README.
- 2026-10-08: Test-split runs of the scripted pusher, all with the same parameters: (1) 11:39, stop at first success,
  6/7; (2) until-done protocol, 6/7; (3) after the arm-collision change, 7/7. No parameter was changed after any of them.
- 2026-10-08: TensorBoard (`tensorboard==2.21.0`) added to the env and `pyproject.toml`; `runs/` is gitignored. OneDrive
  sync paused during training: the repo lives in OneDrive and its sync took most of the CPU.
- 2026-10-08: T7 stopped at its coverage gate (3/7 usable test episodes against the rule's 4). ep_005 fails only on a
  0.58 s gap at the start of its contact window (edge tolerance 0.5 s); tolerating edge gaps of any length would give
  4/7 (train 12/27), but that reading came up after the test count was known, so it was not adopted. C2.3 is cut with
  T7. Claim (2) (copying the hand fails hardest at pushing) is therefore not tested here; the README says so.
- 2026-10-08: Switched T6 to C6.5 at the 2M check instead of waiting for G3 (rule agreed in advance): plain PPO at
  2M had 2-4% rollout success and its deterministic eval never moved the box along the path. Residual
  parameters were fixed before the sanity run and not swept. Results from it measure what RL adds to the scripted
  pusher, not RL learning to push; the README says so.
- 2026-10-08: Tracking accuracy is the objective; speed is not (the progress-indexed reward exists to drop timing).
  The C6.1 reward did not encode that: with gamma 0.99 the discounted success bonus pays about +1.2 for finishing
  146 -> 108 steps sooner, while 0.1 x lateral (m) charges about 0.05 for 0.5 cm more deviation, so residual_s0
  optimised speed. Fixed for the residual run by C6.7 (weight derived above, not swept). Plain PPO keeps the
  C6.1 reward; it never reached the regime where this matters. Missed when C6.1 and C6.5 were reviewed.
- 2026-10-08: Showcase letter traces run once with the frozen workspace map and scripted pusher (C8.1), no tuning;
  the final showcase controller is picked Fri from test results only.
- 2026-10-08: Results-table rule, fixed before any RL test evaluation: each run is evaluated once on test with its
  final model: residual_dev5 (3M; the residual row), residual_s0 at 1M (the C6.1-reward comparison), full_s0 at 10M
  (plain PPO). No checkpoint selection; whatever comes out is reported.

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
- Plane fit per straight episode: the points lie on a line (minor-axis spread 1-8 mm), the tilt across it is
  undetermined, and a plain fit gave tilts of 6-68 deg (against 2-6 deg where the points spread). The ridge prior
  fixed that; the plan's "fit a plane to the box-top positions" alone does not work for the 14 near-collinear paths.
- Plane residual was expected below 1 mm (C1.2) and is 1-4.9 mm per episode (median 1.9). The C1.2 figure came
  from four stationary positions.
- Using max pixel deviation of partly visible board markers as the "camera moved while the board was hidden" check:
  5-8 px, which looked like a moved camera. The cause is a few frames of biased corners at the edges of occlusions;
  the p95 is 2.5-3.9 px and the sustained offset about 1 px.
- `scipy.signal.savgol_filter` on the jittery camera timestamps (see decision log).
- MediaPipe hand landmarks on these clips: the camera is nearly overhead and the hand mostly leaves the frame, so
  only the fingertip is in view. The hand was found in a median 42% of frames (range 0-70%, confidence 0.3; 0 of 38
  episodes above 90%, 31 below 50%; ep_000 and ep_058 0%). That includes the hand-out-of-frame second at each end
  of every clip, but inside the contact window it is also patchy (0-100% in the 5 episodes checked, confidence 0.5). Lowering the
  detection confidence from 0.5 to 0.1 raised ep_000 from 0 to 34% and left ep_025 at 19%. T7 cannot rely on
  MediaPipe alone; options: infer the fingertip from the box's motion and contact point (the pusher's position follows
  from the contact), or colour/skin segmentation of the fingertip, or cut T7's fidelity. Decide when T7 starts.
- Convex hull of FK samples as the reachable set (see decision log): counted the base itself as reachable.
- Env first stepped at 100 Hz with a 500-step (5 s) cap, against 10-20 s recordings, and counted progress in
  time-sampled reference points rather than arc length. Both fixed in C3.5.
- Push speed 1 cm/step (0.2 m/s): 23/27 on train, the box slid away from the pusher. 3.3 mm/step gave 27/27.
- Gripper meshes touching the box before the pusher capsule, and the commanded EE position in the observation
  (see decision log).
- C6.3 attempt 1: the policy pushed the box with its forearm (see decision log).
- C6.3 attempt 2: reach shaping to the box centre; 0% success after 200k steps, the pusher drove into the box and
  shoved it off the path.
- Reference points in the observation spaced by time samples: ~0.5 cm span, often identical, so no lookahead.
- MediaPipe fingertip coverage, second attempt (T7): a 720 px crop around the box at confidence 0.1 raised mean
  in-window coverage from 0.45 to 0.72 on train, still only 5/27 train and 3/7 test episodes without a gap over 0.5 s.
  In the missed frames checked (ep_026) the hand is out of frame, or the finger is hidden while the box sits at the
  image edge, so there is nothing to detect. A 400 px crop found no hand at all in ep_010 (0% against 84% full
  frame): the palm detector needs the whole hand in view.
- The C6.1 reward on the residual run: it paid for finishing sooner (discounted success bonus) far more than for
  tracking, so residual_s0 got about 30% faster with mean deviation 0.6-0.7 cm against 0.52 untrained (C6.5, C6.7).
