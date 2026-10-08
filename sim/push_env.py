"""PushTrack-v0: push a cube along a reference trajectory (C3.5).

Observation (float32, shape (6 + 2*k_ref,)):
    ee_x, ee_y                          -- EE position in robot base frame (m)
    cube_x, cube_y, sin_yaw, cos_yaw    -- cube pose in world frame
    ref_0_x, ref_0_y, ..., ref_{k-1}   -- k reference points in the cube frame (m), spaced ref_spacing
                                           apart in arc length from the current phase (point 0 is the
                                           path point at the phase, the rest lie ahead; clamped at the end)

Action (float32, shape (2,)):
    dx, dy  -- EE displacement per env step, clipped to ±max_delta (m)

Reward:
    progress_delta - 0.1 * lateral_dev - reach_weight * reach_dist + 10 * success

    reach_dist is the distance from the EE to the pre-contact point: `standoff` behind the cube centre
    along the push direction d = unit(path point ref_spacing ahead of the phase - cube).  With the plan's
    literal distance to the cube centre (precontact_reach=False, reach_weight 0.01) the pusher is pulled
    straight at the cube and pushes it off the path whenever the robot starts beside or ahead of it (C6.3).

    progress_delta is the arc-length fraction gained this step.  The phase (arc
    coordinate on the reference) is monotone: each step the cube is projected on the
    stretch of path from the phase to `progress_window` ahead, and the phase advances
    to that projection only while the cube is within `r_adv` of it (C6.1; this is
    eval/metrics.py's furthest projection, restricted to a local window so a path that
    comes back near itself cannot be skipped to).  lateral_dev is the distance to
    that stretch.

Episode ends (terminated) on success: final_error < 2 cm and arc_progress >= 90 %.
Episode is truncated at max_episode_steps.

Control rate: timestep 0.002 s × n_steps_per_action 25 = 0.05 s/step = 20 Hz,
matching extract/clean.py CONTROL_HZ.  max_episode_steps default 600 = 30 s ×
20 Hz (max observed train duration 20.2 s + 50 % slack).

Reference paths come from one of three sources, in this order:
  paths=...      pre-mapped (N, 2) arrays in robot base-frame coords (tests, ad hoc);
  split=...      "train" | "test" | "showcase": that split's data/processed/*.npz,
                 cube_xy mapped through the stored WorkspaceMap
                 (control.workspace.load_workspace_map; fitted on train only),
                 resampled to the control rate from `t`, cube placed at the
                 recorded start pose including yaw;
  episode_files=...  the same for explicit .npz files (the smoke test's synthetic episode); the
                 episode ids are the file stems;
  neither        synthetic paths from control.workspace.generate_synthetic_paths(),
                 translated to the centre of the robot's reachable workspace.
"""
import json
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from capture.common import DEFAULT_PROPS, REPO_ROOT, load_props
from control.ee_controller import EEController
from control.workspace import WORKSPACE_MAP_PATH, generate_synthetic_paths, load_workspace_map
from sim.scene import ARM_JOINTS, PUSHER_CAPSULE_RADIUS, build_push_scene_xml

SPLITS_PATH = REPO_ROOT / "data" / "splits.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def resample_episode(t, cube_xy_yaw, dt):
    """Resample (t, x, y, yaw) onto a uniform grid of step dt starting at t[0].

    Returns an (M, 3) array.  Yaw is unwrapped before interpolation and
    re-wrapped to (-pi, pi]; samples already on the grid are returned unchanged.
    """
    t = np.asarray(t, dtype=float)
    pose = np.asarray(cube_xy_yaw, dtype=float).copy()
    pose[:, 2] = np.unwrap(pose[:, 2])
    grid = t[0] + dt * np.arange(int(np.floor((t[-1] - t[0]) / dt + 1e-9)) + 1)
    out = np.column_stack([np.interp(grid, t, pose[:, i]) for i in range(3)])
    out[:, 2] = np.arctan2(np.sin(out[:, 2]), np.cos(out[:, 2]))
    return out


class PushTrackEnv(gym.Env):
    """PushTrack-v0 gymnasium environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        k_ref: int = 5,
        ref_spacing: float = 0.025,
        r_adv: float = 0.03,
        progress_window: float = 0.10,
        precontact_reach: bool = True,
        reach_weight: float = 0.1,
        episodes=None,
        max_delta: float = 0.02,
        n_steps_per_action: int = 25,   # 0.002 s × 25 = 0.05 s/step = 20 Hz
        max_episode_steps: int = 600,   # max train 20.2 s × 20 Hz + 50% slack
        props_path=DEFAULT_PROPS,
        paths=None,
        split=None,
        workspace_map_path=WORKSPACE_MAP_PATH,
        episode_files=None,
    ):
        """
        :param k_ref:               Reference points per observation.
        :param ref_spacing:         Arc length between observed reference points (m).
        :param r_adv:               The phase advances only while the cube is within this of the path (m).
        :param progress_window:     How far ahead of the phase the cube is searched on the path (m).
        :param precontact_reach:    Reach shaping targets the point behind the cube, not its centre.
        :param reach_weight:        Weight of the reach term (m^-1 per step).
        :param episodes:            Split envs: restrict to these episode ids (e.g. to overfit one).
        :param max_delta:           Max EE displacement per action (m).
        :param n_steps_per_action:  MuJoCo steps per env step.
        :param max_episode_steps:   Truncation horizon.
        :param paths:               Iterable of (N_i, 2) reference paths in robot
                                    base-frame coords (m).  None → synthetic paths.
        :param split:               "train", "test" or "showcase": use that split's
                                    recorded episodes (mutually exclusive with paths).
        :param workspace_map_path:  Stored WorkspaceMap used for split episodes.
        :param episode_files:       Processed .npz files used like split episodes (mapped, resampled, start
                                    yaw); mutually exclusive with `paths` and `split`.
        """
        if sum(v is not None for v in (paths, split, episode_files)) > 1:
            raise ValueError("pass only one of `paths`, `split` and `episode_files`")
        super().__init__()

        self.render_mode = render_mode
        self.k_ref = k_ref
        self.ref_spacing = float(ref_spacing)
        self.r_adv = float(r_adv)
        self.progress_window = float(progress_window)
        self.precontact_reach = bool(precontact_reach)
        self.reach_weight = float(reach_weight)
        self.max_delta = float(max_delta)
        self.n_steps_per_action = int(n_steps_per_action)
        self.max_episode_steps = int(max_episode_steps)

        # Scene
        props = load_props(props_path, require=("cube_side", "cube_height", "cube_mass"))
        self._cube_height = props["cube_height"]
        # EE standoff behind the cube centre: half side + capsule radius + 5 mm
        self._standoff = props["cube_side"] / 2 + PUSHER_CAPSULE_RADIUS + 0.005
        self._model = mujoco.MjModel.from_xml_string(build_push_scene_xml(props))
        self._data  = mujoco.MjData(self._model)

        # Body / joint / site IDs (looked up once)
        self._cube_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        _jnt = self._model.body_jntadr[self._cube_id]
        self._cube_qadr = int(self._model.jnt_qposadr[_jnt])

        self._arm_jnt_qadr = [
            int(self._model.jnt_qposadr[
                self._model.joint(name).id
            ])
            for name in ARM_JOINTS
        ]
        self._arm_act_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ARM_JOINTS
        ]

        # EE controller
        self._ctrl = EEController()

        # Reference paths (robot frame) and the recorded start yaw of each
        self.split = split
        self._episode_ids = None
        if split is not None:
            self._paths, self._yaw0, self._episode_ids = self._load_split(
                split, load_workspace_map(workspace_map_path))
            if episodes is not None:
                keep = [self._episode_ids.index(e) for e in episodes]   # ValueError if not in the split
                self._paths = [self._paths[i] for i in keep]
                self._yaw0 = [self._yaw0[i] for i in keep]
                self._episode_ids = [self._episode_ids[i] for i in keep]
        elif episode_files is not None:
            self._paths, self._yaw0, self._episode_ids = self._load_files(
                episode_files, load_workspace_map(workspace_map_path))
        elif paths is None:
            line, arc, s_curve = generate_synthetic_paths()
            self._paths = [line, arc, s_curve]
            self._yaw0 = [0.0] * len(self._paths)
        else:
            self._paths = list(paths)
            self._yaw0 = [0.0] * len(self._paths)

        # Centre for placing synthetic / unmapped paths in the robot workspace
        # (paths from a split are already in the robot frame and are not shifted)
        self._path_origin_xy = (None if split is not None or episode_files is not None
                                else self._find_workspace_centre())

        # Gymnasium spaces
        n_obs = 2 + 4 + 2 * k_ref
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.max_delta, high=self.max_delta, shape=(2,), dtype=np.float32
        )

        # Episode state (initialised in reset)
        self._path: np.ndarray = self._paths[0]
        self._arc_lengths: np.ndarray = np.array([0.0])
        self._total_arc_len: float = 0.0
        self._max_arc_progress: float = 0.0   # fraction in [0, 1], monotone
        self._phase: float = 0.0              # arc coordinate (m), monotone
        self._lateral: float = 0.0
        self._step_count: int = 0
        self._renderer = None

    # ------------------------------------------------------------------
    # Workspace centre
    # ------------------------------------------------------------------

    @property
    def episode_ids(self):
        """Episode ids of the split (None for synthetic or explicit paths)."""
        return None if self._episode_ids is None else list(self._episode_ids)

    @property
    def reference_path(self):
        """Reference path of the current episode, robot frame (copy)."""
        return self._path.copy()

    @property
    def dt(self):
        """Seconds per env step."""
        return float(self._model.opt.timestep * self.n_steps_per_action)

    @property
    def cube_xy_yaw(self):
        """Cube pose from the simulator state: (xy copy, yaw)."""
        return self._cube_xy_yaw()

    def _find_workspace_centre(self):
        """Estimate the centroid of FK positions reachable at push height."""
        from control.kinematics.core import Adjoint, FKinBody, TransInv
        from control.kinematics.parser import DEFAULT_URDF, findMnS

        M, Slist, limits = findMnS(DEFAULT_URDF)
        n = Slist.shape[1]
        Blist = np.array([Adjoint(TransInv(M)) @ Slist[:, i] for i in range(n)]).T
        z_push = self._cube_height / 2
        rng = np.random.default_rng(0)
        positions = []
        for _ in range(2000):
            q = rng.uniform(limits[:, 0], limits[:, 1])
            pos = FKinBody(M, Blist, q)[:3, 3]
            if abs(pos[2] - z_push) < 0.03:
                positions.append(pos[:2].copy())
        if positions:
            return np.mean(positions, axis=0)
        # Fallback: FK at q=0 XY
        return FKinBody(M, Blist, np.zeros(n))[:2, 3].copy()

    def _place_path(self, path_local):
        """Translate a path (starting at (0,0)) to the robot workspace."""
        if self._path_origin_xy is None:      # split episode, already in robot frame
            return path_local
        return path_local - path_local[0] + self._path_origin_xy

    def _load_split(self, split, ws_map):
        """Load a split's episodes as (paths, start yaws, episode ids).

        Paths are the cube (x, y) mapped to the robot frame and resampled to the
        control rate.  The map is a pure scale + translation, so yaw is unchanged.
        """
        with open(SPLITS_PATH) as f:
            splits = json.load(f)
        if split not in ("train", "test", "showcase"):
            raise ValueError(f"split must be train, test or showcase, got {split!r}")
        return self._load_files([PROCESSED_DIR / f"{ep}.npz" for ep in splits[split]], ws_map)

    def _load_files(self, files, ws_map):
        """Processed .npz files as (paths, start yaws, ids); ids are the file stems."""
        dt = self._model.opt.timestep * self.n_steps_per_action
        paths, yaw0 = [], []
        for f in files:
            d = np.load(f)
            pose = resample_episode(d["t"], d["cube_xy_yaw"], dt)
            paths.append(ws_map.transform(pose[:, :2]))
            yaw0.append(float(pose[0, 2]))
        return paths, yaw0, [Path(f).stem for f in files]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # sets self.np_random

        # Sample a path (reproducible with seed via self.np_random);
        # options={"episode": i} picks path i instead (deterministic evaluation)
        idx = int(self.np_random.integers(len(self._paths)))
        if options and "episode" in options:
            idx = int(options["episode"])
        if options and "episode_id" in options:
            idx = self._episode_ids.index(options["episode_id"])   # e.g. "ep_005"; split envs only
        self._path = self._place_path(self._paths[idx])
        yaw0 = self._yaw0[idx]
        diffs = np.diff(self._path, axis=0)
        self._arc_lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
        self._total_arc_len = float(self._arc_lengths[-1])
        self._max_arc_progress = 0.0
        self._phase = 0.0
        self._lateral = 0.0
        self._step_count = 0

        # Reset MuJoCo state
        mujoco.mj_resetData(self._model, self._data)

        # Place cube at the path start with the recorded yaw (0 for synthetic paths)
        x0, y0 = self._path[0]
        self._data.qpos[self._cube_qadr:self._cube_qadr + 3] = [x0, y0, self._cube_height / 2]
        self._data.qpos[self._cube_qadr + 3:self._cube_qadr + 7] = [
            np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]   # qw, qx, qy, qz

        # Home the robot (q=0, ctrl=0)
        for qadr, aid in zip(self._arm_jnt_qadr, self._arm_act_ids):
            self._data.qpos[qadr] = 0.0
            if aid >= 0:
                self._data.ctrl[aid] = 0.0

        mujoco.mj_forward(self._model, self._data)
        self._ctrl.reset(q0=np.zeros(len(ARM_JOINTS)),
                         z_push=self._cube_height / 2)

        info = {"episode_id": self._episode_ids[idx] if self._episode_ids else None}
        return self._get_obs(), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy = float(action[0]), float(action[1])

        # Sync controller from actual sim state before solving
        self._sync_ctrl()

        # Compute new EE target
        T_cur = self._ctrl.ee_pose()
        new_xy = T_cur[:2, 3] + np.array([dx, dy])
        q_new, success = self._ctrl.solve(new_xy)

        if success:
            for i, aid in enumerate(self._arm_act_ids):
                if aid >= 0:
                    self._data.ctrl[aid] = float(q_new[i])

        for _ in range(self.n_steps_per_action):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        obs = self._get_obs()
        reward, info = self._compute_reward()
        terminated = bool(info["success"])
        truncated  = self._step_count >= self.max_episode_steps

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        self._renderer.update_scene(self._data, camera=cam_id if cam_id >= 0 else -1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_ctrl(self):
        """Update controller's internal q from the actual simulation joint positions."""
        q = np.array([self._data.qpos[qadr] for qadr in self._arm_jnt_qadr])
        self._ctrl._q = q  # direct write; EEController._q is the warm-start

    def _ee_xy(self):
        """EE position from the measured joint angles (not the last IK command, which leads it)."""
        q = np.array([self._data.qpos[qadr] for qadr in self._arm_jnt_qadr])
        return self._ctrl.ee_pose(q=q)[:2, 3]

    def _cube_xy_yaw(self):
        pos  = self._data.xpos[self._cube_id]
        quat = self._data.xquat[self._cube_id]   # qw, qx, qy, qz
        qw, qx, qy, qz = quat
        yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy),
                                1.0 - 2.0 * (qy ** 2 + qz ** 2)))
        return pos[:2].copy(), yaw

    def _point_at(self, arc):
        """Reference point(s) at arc coordinate(s) (m), clamped to the path."""
        arc = np.clip(arc, 0.0, self._total_arc_len)
        return np.column_stack([np.interp(arc, self._arc_lengths, self._path[:, 0]),
                                np.interp(arc, self._arc_lengths, self._path[:, 1])])

    def _advance_phase(self, cube_xy):
        """Phase tracker (C6.1). Returns the arc-length fraction gained; sets self._lateral.

        Projects the cube on the path from the phase to `progress_window` ahead. The phase moves to the
        projection if the cube is within r_adv of it, never backwards.
        """
        from eval.metrics import _polyline_project
        s0 = self._phase
        end = min(s0 + self.progress_window, self._total_arc_len)
        if end - s0 < 1e-9:
            self._lateral = float(np.linalg.norm(cube_xy - self._path[-1]))
            return 0.0
        inner = self._arc_lengths[(self._arc_lengths > s0) & (self._arc_lengths < end)]
        arcs = np.concatenate([[s0], inner, [end]])
        arc, dist, _ = _polyline_project(self._point_at(arcs), np.asarray(cube_xy, float)[None])
        self._lateral = float(dist[0])
        if self._lateral <= self.r_adv:
            self._phase = min(s0 + float(arc[0]), self._total_arc_len)
        gain = (self._phase - s0) / self._total_arc_len if self._total_arc_len > 0 else 0.0
        self._max_arc_progress = min(self._max_arc_progress + gain, 1.0)
        return gain

    def _get_obs(self):
        ee_xy = self._ee_xy()
        cube_xy, yaw = self._cube_xy_yaw()
        sin_y, cos_y = float(np.sin(yaw)), float(np.cos(yaw))

        # k reference points at fixed arc-length spacing from the phase (clamped at the path end), in the cube frame
        pts = self._point_at(self._phase + self.ref_spacing * np.arange(self.k_ref))
        R = np.array([[cos_y, sin_y], [-sin_y, cos_y]])
        pts_cube = (pts - cube_xy) @ R.T

        obs = np.concatenate([ee_xy, cube_xy, [sin_y, cos_y], pts_cube.ravel()])
        return obs.astype(np.float32)

    def _reach_target(self, cube_xy):
        """Where the EE should be heading: behind the cube along the push direction (or the cube itself)."""
        if not self.precontact_reach:
            return cube_xy
        ahead = self._point_at(np.array([self._phase + self.ref_spacing]))[0] - cube_xy
        n = np.linalg.norm(ahead)
        return cube_xy if n < 1e-6 else cube_xy - self._standoff * ahead / n

    def _compute_reward(self):
        cube_xy, _ = self._cube_xy_yaw()
        progress_delta = self._advance_phase(cube_xy)
        new_prog = self._max_arc_progress
        lateral   = self._lateral
        ee_xy     = self._ee_xy()
        reach     = float(np.linalg.norm(ee_xy - self._reach_target(cube_xy)))
        final_err = float(np.linalg.norm(cube_xy - self._path[-1]))
        # Arc-length success: same threshold as T4's eval/metrics.py
        success   = (final_err < 0.02) and (new_prog >= 0.90)

        reward = progress_delta - 0.1 * lateral - self.reach_weight * reach + (10.0 if success else 0.0)
        info = {"arc_progress": new_prog, "lateral_dev_m": lateral, "reach_dist_m": reach,
                "final_err_m": final_err, "success": success}
        return float(reward), info


# Register with gymnasium when this module is imported
gym.register(
    id="PushTrack-v0",
    entry_point="sim.push_env:PushTrackEnv",
    max_episode_steps=500,
)
