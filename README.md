# Where, not how
### Human demos tell robot what should happen to an object, robot learns how.

## Thesis
When learning from a human demonstration, copy the effect on the object, not the motion of the hand.

A human hand and a robot arm, in this case the SO-101, are different in basically all the ways that matter for copying motion. Retargetting the hand's path to robot means inheriting the mismatch.

By supplying the *what* (where the object should go), robot is left to figure our *how* for it's own body.

## How my data drives the robot
Data enters in 3 places.
1. **As the command at test time.** Record a push, robot is asked to reproduce the trajectory in sim. Recording is task.
2. **As the training distribution.** The RL policy is trained on reference paths sampled from train split. 
3. **As the baseline.** FIngertip trajectory, from the same videos, is what the hand-replay comparison runs on.
## Pipeline
<!-- WIP, but planned table (have to update) -->
| Stage | Folder | What it does | Status |
|---|---|---|---|
| Capture | `capture/` | Marker sheet, camera calibration, live pose check, recording | Tools done, recording in progress |
| Extraction | `extract/` | Board pose gives the table frame, cube marker gives (x, y, yaw) in metres, fingertip from MediaPipe for the baseline. Cleaned, resampled and saved as `ep_XXX.npz`. Train/test split fixed here | Not started |
| Kinematics | `control/` | PoE forward kinematics, Jacobian and damped least-squares IK, brought over from my SO-101 teleop project | Done, FK within 0.004 mm of MuJoCo over 1000 configs |
| Simulation | `sim/` | SO-101 in MuJoCo with a table, a box matching the real cube's size and mass, and a pusher tip. Each episode starts from one of my recorded trajectories | Robot model done, scene in progress |
| Controllers | `control/`, `rl/` | Scripted pusher, and a PPO policy that sees the pusher, the cube and the next few path points, and outputs a planar pusher velocity. IK turns that into joint targets | Not started |
| Evaluation | `eval/` | Path deviation, final error and success rate, measured the same way for every method. Overlay plots and side-by-side GIFs | Not started |

## Data collection
<!-- Camera, markers, episodes by category, dataset link + license. -->

## Results


## Design choices
**RL acts in end-effector space with analytic IK underneath.**
Policy learns the pushing strategy, not kinematics.

**Progress is indexed instead of timed.** Reward based on advancing cube along path, not matching hand's speed.

**Pushing instead of grasping.** To remove fragile part of sim manipulation while keeping comparison about where the command comes from.

## What didn't work
<!-- Start now: fixed-orientation IK (61% vs 100%), joint-limit clamp misses. -->

## Setup and running
<!-- Conda env, pip install -e ".[dev]", commands, tests (link tests/README.md). -->
