# Related work: where-not-how

Verified 2026-10-07 against publisher or arXiv pages. Venue noted for each; HuDOR has no peer-reviewed venue found.

## Closest: object motion as the embodiment-agnostic signal

- **Human2Sim2Robot.** Lum, Lee, Liu, Bohg. "Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration." CoRL 2025, PMLR v305. https://proceedings.mlr.press/v305/lum25a.html
  Object 6D pose trajectory from one RGB-D human video = object-centric reward for PPO in sim; one retargeted pre-manipulation hand pose = initialisation. Kuka + Allegro, 7 tasks; beats object-aware replay by 55% and BC by 68%. One policy per task/demo.
- **Im2Flow2Act.** Xu et al. "Flow as the Cross-Domain Manipulation Interface." CoRL 2024, PMLR v270. https://proceedings.mlr.press/v270/xu25a.html
  Object flow generated from human videos; flow-conditioned policy trained on simulated robot play. Core argument: object flow describes the change in the object, not the action, so it is independent of embodiment.
- **HuDOR.** Guzey, Dai, Savva, Bhirangi, Pinto. "Bridging the Human to Robot Dexterity Gap through Object-Oriented Rewards." arXiv 2410.23289 (no venue found). https://arxiv.org/abs/2410.23289
  Reward = negative RMSE between robot and human object trajectories (point tracks). Base policy replays retargeted hand; online residual RL on the real robot. ArUco markers for frames.

## Object-centric transfer and data generation

- **MimicGen.** Mandlekar et al. CoRL 2023, PMLR v229. https://proceedings.mlr.press/v229/mandlekar23a.html  ~200 human demos to 50K+ via object-relative segment transformation.
- **DITTO.** Heppert, Argus, Welschehold, Brox, Valada. IROS 2024. https://arxiv.org/abs/2403.15203  Object trajectory from one RGB-D demo, warped to the new scene.
- **Track2Act.** Bharadhwaj, Mottaghi, Gupta, Tulsiani. ECCV 2024. https://arxiv.org/abs/2405.01527  Point tracks from web video, rigid object transforms to EE poses, residual policy for closed loop.
- **ORION.** Zhu, Lim, Stone, Zhu. Autonomous Robots 2026. https://arxiv.org/abs/2405.20321  Object-centric plan from a single human video.
- **MT-π (Motion Tracks).** Ren, Sundaresan, Sadigh, Choudhury, Bohg. ICRA 2025. https://arxiv.org/abs/2501.06994  Contrast: unifies hand and EE motion in image space (copies the actor, not the object).
- **ManipTrans.** Li et al. CVPR 2025. https://arxiv.org/abs/2503.21860  Residual RL to track human hand + object trajectories in sim.

## Pushing, tracking and reward design

- **Hogan & Rodriguez.** "Reactive planar nonprehensile manipulation with hybrid model predictive control." IJRR 39(7), 2020. doi:10.1177/0278364920913938  Pusher-slider is hybrid and underactuated; sticking/sliding modes.
- **Bauza, Hogan, Rodriguez.** "A Data-Efficient Approach to Precise and Controlled Pushing." CoRL 2018. https://arxiv.org/abs/1807.09904
- **Diffusion Policy.** Chi et al. RSS 2023. https://roboticsproceedings.org/rss19/p026.html  Push-T benchmark (goal pose, not trajectory).
- **DeepMimic.** Peng, Abbeel, Levine, van de Panne. ACM TOG (SIGGRAPH) 2018. https://arxiv.org/abs/1804.02717  Reference tracking with a time-based phase variable.
- **Aguiar, Kokotović, Hespanha.** "Path-following for nonminimum phase systems removes performance limitations." IEEE TAC 50, 234–239, 2005.  Path following (progress-indexed) vs trajectory tracking (time-indexed).

## Positioning

Not novel: object trajectory as an embodiment-agnostic command/reward (Im2Flow2Act, HuDOR, Human2Sim2Robot).
Our distinct parts: one goal-conditioned policy over a distribution of the author's recorded trajectories, evaluated on unseen ones; non-prehensile pushing, where the robot's strategy (reposition, push from behind) differs most from the hand's; low-cost 5-DOF arm with validated analytic kinematics; progress-indexed (path-following) reward; quantified hand-replay gap.