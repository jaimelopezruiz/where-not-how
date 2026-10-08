"""Workspace map: human table frame → robot EE frame at push height (C3.4).

One affine map per robot placement, shared by all episodes:
    robot_xy = scale * human_xy + offset_xy

scale ≤ 1 (uniform, no shear), preserving relative distances up to a constant
factor.  The map never expands; it only shrinks and shifts.

Human workspace (from the T1 recordings, table frame):
    x ∈ [−0.37, −0.08] m,  y ∈ [−0.24, 0.07] m  (≈ 0.29 × 0.31 m)

The class has three constructors:
  fit(trajectories_xy, reachable_xy)       -- centroid-based fit; fast, no
                                              feasibility check.
  fit_feasible(trajectories_xy, ...)       -- sweep scale 1→0.5, maximise
                                              n_feasible train trajectories.
  from_robot_reach(...)                    -- sample robot workspace by FK to get
                                              reachable_xy, then call fit().

Reachability check: a KD-tree on FK-sampled EE positions replaces the former
convex-hull test.  The hull incorrectly treated the inner unreachable zone
(near the robot base, inside the arm's arc) as reachable.

Feasibility of a path point requires both:
  (a) norm(robot_xy) > r_min_cube = BASE_FOOTPRINT_RADIUS + cube_half_diagonal
      [cube footprint clears the robot base column]
  (b) nearest FK sample to the pusher position is within REACH_TOLERANCE
      [EE can physically reach the pusher position]

Synthetic reference paths for C3.2–C3.5 (workflow rule 6; no real data needed):
  line:    0.25 m straight push in +x
  arc:     quarter-circle, radius 0.12 m, starting at the origin
  s_curve: sinusoidal path, 0.20 m long, amplitude 0.06 m
All coordinates are in the path frame with the cube start at (0, 0).
"""
import json
from pathlib import Path

import numpy as np

# Measured from T1 recordings (decision log 2026-10-08)
HUMAN_X_RANGE = (-0.37, -0.08)
HUMAN_Y_RANGE = (-0.24,  0.07)

# SO-101 physical constants used for feasibility checks
# BASE_FOOTPRINT_RADIUS: max XY radius of SO-101 base body geoms; derived from
#   MjModel.geom_pos / geom_size analysis of so101_new_calib.xml (2026-10-08).
BASE_FOOTPRINT_RADIUS = 0.0895  # m
# REACH_TOLERANCE: KD-tree query radius for EE reachability, 2x the median
#   nearest-neighbour spacing of the 20 000-sample FK set at push height
#   (1741 hits, median NN 5.97 mm, seed 42; measured 2026-10-08).
REACH_TOLERANCE = 0.012  # m


def generate_synthetic_paths(n: int = 60):
    """Return (line, arc, s_curve) each as an (n, 2) float64 array.

    Coordinates are in the path frame (cube start at origin), in metres.
    These paths are used directly in the robot EE frame; no workspace map is needed
    until real trajectory data is available (workflow rule 6).
    """
    t = np.linspace(0.0, 1.0, n)

    # Straight push 0.25 m in +x
    line = np.column_stack([0.25 * t, np.zeros(n)])

    # Quarter-circle, radius 0.12 m, centred at (0, 0.12): starts (0,0), ends (0.12, 0.12)
    theta = t * (np.pi / 2)
    arc = np.column_stack([0.12 * np.sin(theta), 0.12 * (1.0 - np.cos(theta))])

    # Sinusoidal S-curve: x from 0 to 0.20, y = 0.06 * sin(2π t)
    s_curve = np.column_stack([0.20 * t, 0.06 * np.sin(2.0 * np.pi * t)])

    return line, arc, s_curve


class WorkspaceMap:
    """Translation + uniform scale ≤ 1 from human table frame to robot EE frame.

    ``robot_xy = scale * human_xy + offset_xy``
    """

    def __init__(self, scale: float, offset_xy):
        scale = float(scale)
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"scale must be in (0, 1], got {scale:.4f}")
        self.scale = scale
        self.offset_xy = np.asarray(offset_xy, dtype=float).copy()

    def transform(self, xy_human):
        """Map (N, 2) or (2,) human table coords → robot base-frame EE (x, y)."""
        return self.scale * np.asarray(xy_human, dtype=float) + self.offset_xy

    def reachable_fraction(self, path_xy_human, reachable_xy_robot,
                           tolerance: float = REACH_TOLERANCE):
        """Fraction of mapped path points within tolerance of any FK sample.

        Uses a KD-tree on reachable_xy_robot (EE positions sampled at push height).
        The former convex-hull test over-counted positions inside the unreachable
        inner zone near the robot base (the reachable set is annular, so the hull
        filled the gap).

        :param path_xy_human:      (N, 2) path in human table coords.
        :param reachable_xy_robot: (M, 2) sampled EE positions at push height.
        :param tolerance:          KD-tree query radius (m).
        :returns: float in [0, 1].
        """
        from scipy.spatial import cKDTree
        mapped = self.transform(np.asarray(path_xy_human, dtype=float))
        tree = cKDTree(np.asarray(reachable_xy_robot, dtype=float))
        dists, _ = tree.query(mapped)
        return float((dists <= tolerance).mean())

    def path_feasibility_fraction(self, path_xy_human, reachable_xy_robot,
                                   r_min_cube: float, pusher_offset: float,
                                   reach_tolerance: float = REACH_TOLERANCE):
        """Fraction of path points feasible: cube clears base AND pusher reachable.

        For each mapped path point p:
          (a) norm(p) > r_min_cube       -- cube footprint clears the robot body
          (b) nearest FK sample to the pusher position is within reach_tolerance

        The local push direction d at each point is the unit tangent along the
        mapped path; pusher_xy = p − pusher_offset · d.

        :param r_min_cube:    Minimum robot-frame radius for the cube centre (m).
                              Typically BASE_FOOTPRINT_RADIUS + cube_half_diagonal.
        :param pusher_offset: Distance from cube centre to pusher EE centre (m).
                              Typically cube_half_diagonal + capsule_r + margin.
        :returns: float in [0, 1].
        """
        from scipy.spatial import cKDTree
        mapped = self.transform(np.asarray(path_xy_human, dtype=float))
        tree = cKDTree(np.asarray(reachable_xy_robot, dtype=float))

        cube_ok = np.linalg.norm(mapped, axis=1) > r_min_cube

        diffs = np.diff(mapped, axis=0)
        dnorm = np.linalg.norm(diffs, axis=1, keepdims=True)
        dnorm = np.where(dnorm < 1e-9, 1.0, dnorm)
        dirs = np.concatenate([diffs / dnorm, diffs[-1:] / dnorm[-1:]], axis=0)

        pd, _ = tree.query(mapped - pusher_offset * dirs)
        pusher_ok = pd <= reach_tolerance

        return float((cube_ok & pusher_ok).mean())

    @classmethod
    def fit(cls, trajectories_xy, reachable_xy_robot):
        """Fit the map from a list of human-frame trajectories.

        Intended to be called once the train split is extracted (not on test data).
        The scale is set so the human workspace fits within the robot's reachable hull.

        :param trajectories_xy:    list of (N_i, 2) arrays in human table coords.
        :param reachable_xy_robot: (M, 2) sampled reachable EE positions at push height.
        :returns: WorkspaceMap
        """
        all_pts = np.concatenate([np.asarray(p, dtype=float) for p in trajectories_xy], axis=0)
        human_center  = all_pts.mean(axis=0)
        human_half    = np.linalg.norm(all_pts - human_center, axis=1).max()

        reach         = np.asarray(reachable_xy_robot, dtype=float)
        robot_center  = reach.mean(axis=0)
        robot_half    = np.linalg.norm(reach - robot_center, axis=1).max()

        scale  = min(1.0, robot_half / human_half) if human_half > 0 else 1.0
        offset = robot_center - scale * human_center
        return cls(scale, offset)

    @classmethod
    def fit_feasible(cls, trajectories_xy, reachable_xy,
                     r_min_cube: float, pusher_offset: float,
                     reach_tolerance: float = REACH_TOLERANCE):
        """Fit (scale ≤ 1, offset) maximising feasible trajectories.

        Sweeps scale 1.0 → 0.50 (step 0.05).  At each scale the offset_y aligns
        the trajectory centroid with the valid-reach centroid (FK positions at
        r > r_min_cube).  offset_x is searched from the minimum that keeps the
        workspace inner edge clear of the base exclusion zone upward (+15 cm
        range, 1 cm steps).  Returns the LARGEST scale that achieves the best
        feasibility count.

        Feasibility criterion per trajectory: 100 % of path points satisfy
        both cube clearance (a) and pusher reachability (b).

        :param trajectories_xy:  list of (N_i, 2) arrays in human table coords.
        :param reachable_xy:     (M, 2) FK-sampled EE positions at push height.
        :param r_min_cube:       Minimum cube-centre radius from robot base (m).
        :param pusher_offset:    Cube-centre to pusher distance in push dir (m).
        :param reach_tolerance:  KD-tree query radius (m).
        :returns: (WorkspaceMap, n_feasible)
        """
        from scipy.spatial import cKDTree

        trajs = [np.asarray(t, dtype=float) for t in trajectories_xy]
        all_pts = np.concatenate(trajs, axis=0)
        traj_cy = all_pts[:, 1].mean()
        x_min_human = all_pts[:, 0].min()

        reach = np.asarray(reachable_xy, dtype=float)
        valid_mask = np.linalg.norm(reach, axis=1) > r_min_cube
        valid_cy = reach[valid_mask, 1].mean() if valid_mask.any() else 0.0

        tree = cKDTree(reach)

        def _n_ok(scale, offset):
            n = 0
            for traj in trajs:
                mapped = scale * traj + offset
                cube_ok = np.linalg.norm(mapped, axis=1) > r_min_cube
                diffs = np.diff(mapped, axis=0)
                dnorm = np.linalg.norm(diffs, axis=1, keepdims=True)
                dnorm = np.where(dnorm < 1e-9, 1.0, dnorm)
                dirs = np.concatenate([diffs / dnorm, diffs[-1:] / dnorm[-1:]], axis=0)
                pd, _ = tree.query(mapped - pusher_offset * dirs)
                if (cube_ok & (pd <= reach_tolerance)).all():
                    n += 1
            return n

        best_n = -1
        best_scale = 0.5
        best_offset = np.zeros(2)

        for scale in np.arange(1.0, 0.45, -0.05):
            oy = valid_cy - scale * traj_cy
            # ox_lo: smallest offset_x that keeps the inner workspace edge at
            # x_robot >= r_min_cube (tight clearance at y ≈ 0)
            ox_lo = r_min_cube + 0.005 - scale * x_min_human
            best_n_s, best_ox = -1, ox_lo
            for ox in np.arange(ox_lo - 0.01, ox_lo + 0.16, 0.01):
                n = _n_ok(scale, np.array([ox, oy]))
                if n > best_n_s:
                    best_n_s, best_ox = n, ox

            if best_n_s > best_n:     # strictly better → prefer larger scale
                best_n = best_n_s
                best_scale = float(scale)
                best_offset = np.array([best_ox, oy])

            if best_n == len(trajs):
                break                 # all feasible; largest scale wins

        return cls(best_scale, best_offset), best_n

    @classmethod
    def from_robot_reach(cls, urdf_path=None, z_push: float = 0.01625,
                         z_tol: float = 0.025, n_samples: int = 4000, seed: int = 42):
        """Fit by sampling the robot's FK workspace at the push height.

        Samples random joint configurations, runs FK, keeps positions within
        z_tol of z_push, then calls fit() with the human-workspace extent.

        :param z_push:    Target push height in robot base frame (m).
                          Default cube_height/2 = 32.5/2 mm = 0.01625 m.
        :param z_tol:     Tolerance around z_push (m).
        :param n_samples: Number of random FK evaluations.
        :param seed:      RNG seed for reproducibility.
        :returns: (WorkspaceMap, reachable_xy) where reachable_xy is the (M, 2)
                  EE position sample used for KD-tree reachability queries.
        """
        from control.kinematics.core import Adjoint, FKinBody, TransInv
        from control.kinematics.parser import DEFAULT_URDF, findMnS

        if urdf_path is None:
            urdf_path = DEFAULT_URDF

        M, Slist, limits = findMnS(urdf_path)
        n = Slist.shape[1]
        Blist = np.array([Adjoint(TransInv(M)) @ Slist[:, i] for i in range(n)]).T
        rng = np.random.default_rng(seed)

        reachable = []
        for _ in range(n_samples):
            q = rng.uniform(limits[:, 0], limits[:, 1])
            pos = FKinBody(M, Blist, q)[:3, 3]
            if abs(pos[2] - z_push) < z_tol:
                reachable.append(pos[:2].copy())

        if len(reachable) < 4:
            # Arm cannot reach z_push within z_tol; fall back to FK at q=0
            pos0 = FKinBody(M, Blist, np.zeros(n))[:3, 3]
            reachable = [pos0[:2] + d for d in [np.array([0, 0]), np.array([0.05, 0]),
                                                 np.array([0, 0.05]), np.array([0.05, 0.05])]]

        reachable_xy = np.array(reachable)

        # Build the human-workspace path set to use as the fit reference
        line, arc, s_curve = generate_synthetic_paths()
        # Shift paths to the human workspace centre
        hx = (HUMAN_X_RANGE[0] + HUMAN_X_RANGE[1]) / 2
        hy = (HUMAN_Y_RANGE[0] + HUMAN_Y_RANGE[1]) / 2
        human_center = np.array([hx, hy])
        trajectories = [p + human_center for p in (line, arc, s_curve)]

        ws_map = cls.fit(trajectories, reachable_xy)
        return ws_map, reachable_xy


def report_reachable(ws_map, reachable_xy_robot):
    """Print reachable fraction for the three synthetic paths and return results dict."""
    line, arc, s_curve = generate_synthetic_paths()
    hx = (HUMAN_X_RANGE[0] + HUMAN_X_RANGE[1]) / 2
    hy = (HUMAN_Y_RANGE[0] + HUMAN_Y_RANGE[1]) / 2
    human_center = np.array([hx, hy])
    paths = {"line": line + human_center, "arc": arc + human_center,
             "s_curve": s_curve + human_center}

    print(f"WorkspaceMap  scale={ws_map.scale:.3f}  "
          f"offset=({ws_map.offset_xy[0]:.3f}, {ws_map.offset_xy[1]:.3f}) m")
    print(f"Reachable sample at push height: {len(reachable_xy_robot)} configs")
    results = {}
    for name, path in paths.items():
        frac = ws_map.reachable_fraction(path, reachable_xy_robot)
        results[name] = frac
        print(f"  {name:10s}: {frac * 100:.1f}% reachable")
    return results


# ---------------------------------------------------------------------------
# Train-only fit and the stored map (single source of scale/offset for the env)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_MAP_PATH = REPO_ROOT / "data" / "workspace_map.json"
PUSHER_MARGIN = 0.010  # m, clearance between cube corner and pusher capsule


def sample_reachable_xy(n_samples: int = 20_000, seed: int = 42,
                        z_push: float = 0.01625, z_tol: float = 0.025):
    """(M, 2) FK-sampled EE positions within z_tol of z_push (robot base frame)."""
    from control.kinematics.core import Adjoint, FKinBody, TransInv
    from control.kinematics.parser import DEFAULT_URDF, findMnS

    M, Slist, limits = findMnS(DEFAULT_URDF)
    n = Slist.shape[1]
    Blist = np.array([Adjoint(TransInv(M)) @ Slist[:, i] for i in range(n)]).T
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(n_samples):
        q = rng.uniform(limits[:, 0], limits[:, 1])
        pos = FKinBody(M, Blist, q)[:3, 3]
        if abs(pos[2] - z_push) < z_tol:
            hits.append(pos[:2].copy())
    return np.array(hits)


def median_nn_spacing(points_xy) -> float:
    """Median distance from each point to its nearest other point (m)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(points_xy, dtype=float)).query(points_xy, k=2)
    return float(np.median(d[:, 1]))


def fit_train_map(splits_path=None, processed_dir=None, props_path=None,
                  n_samples: int = 200_000, seed: int = 42):
    """Fit the WorkspaceMap on the TRAIN split only.

    200 000 FK samples (~3 min, 17 280 hits, median NN 1.9 mm) rather than 20 000:
    at 20 000 the sample has holes wider than REACH_TOLERANCE and no trajectory
    passes the 100 % test, although the reachable region itself is filled.

    :returns: (WorkspaceMap, n_feasible, n_train, info) where info holds the
              fit inputs (r_min_cube, pusher_offset, reach_tolerance, n_fk_samples).
    """
    from capture.common import DEFAULT_PROPS, load_props
    from sim.scene import PUSHER_CAPSULE_RADIUS

    splits_path = Path(splits_path) if splits_path else REPO_ROOT / "data" / "splits.json"
    processed_dir = Path(processed_dir) if processed_dir else REPO_ROOT / "data" / "processed"
    with open(splits_path) as f:
        train_eps = json.load(f)["train"]
    trajs = [np.load(processed_dir / f"{ep}.npz")["cube_xy_yaw"][:, :2] for ep in train_eps]

    props = load_props(props_path or DEFAULT_PROPS, require=("cube_side",))
    r_half_diag = np.sqrt(2.0) * props["cube_side"] / 2.0
    r_min_cube = BASE_FOOTPRINT_RADIUS + r_half_diag
    pusher_offset = r_half_diag + PUSHER_CAPSULE_RADIUS + PUSHER_MARGIN

    reach = sample_reachable_xy(n_samples=n_samples, seed=seed)
    ws_map, n_ok = WorkspaceMap.fit_feasible(trajs, reach, r_min_cube, pusher_offset)
    info = {"r_min_cube": float(r_min_cube), "pusher_offset": float(pusher_offset),
            "reach_tolerance": REACH_TOLERANCE, "n_fk_samples": int(len(reach)),
            "median_nn_spacing_m": median_nn_spacing(reach)}
    return ws_map, n_ok, len(trajs), info


def save_workspace_map(ws_map, n_feasible, n_train, info, path=WORKSPACE_MAP_PATH):
    out = {"scale": ws_map.scale, "offset_xy": ws_map.offset_xy.tolist(),
           "fitted_on": "train", "n_feasible": int(n_feasible), "n_train": int(n_train),
           **info}
    with open(path, "w", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


def load_workspace_map(path=WORKSPACE_MAP_PATH):
    """Load the stored map (run ``python -m control.workspace`` to create it)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python -m control.workspace")
    with open(path) as f:
        d = json.load(f)
    return WorkspaceMap(d["scale"], d["offset_xy"])


if __name__ == "__main__":
    ws, n_ok, n_train, info = fit_train_map()
    save_workspace_map(ws, n_ok, n_train, info)
    print(f"median NN spacing {info['median_nn_spacing_m'] * 1000:.2f} mm, "
          f"tolerance {REACH_TOLERANCE * 1000:.1f} mm, {info['n_fk_samples']} FK samples")
    print(f"scale={ws.scale:.4f}  offset=({ws.offset_xy[0]:.4f}, {ws.offset_xy[1]:.4f})  "
          f"feasible {n_ok}/{n_train}  -> {WORKSPACE_MAP_PATH}")
