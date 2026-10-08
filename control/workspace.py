"""Workspace map: human table frame → robot EE frame at push height (C3.4).

One affine map per robot placement, shared by all episodes:
    robot_xy = scale * human_xy + offset_xy

scale ≤ 1 (uniform, no shear), preserving relative distances up to a constant
factor.  The map never expands; it only shrinks and shifts.

Human workspace (from the T1 recordings, table frame):
    x ∈ [−0.37, −0.08] m,  y ∈ [−0.24, 0.07] m  (≈ 0.29 × 0.31 m)

The class has two constructors:
  fit(trajectories_xy, reachable_xy)  -- fit from a list of human-frame paths and
                                         a sample of reachable robot EE positions.
                                         Call this once the train split is extracted.
  from_robot_reach(...)               -- sample robot workspace by FK to get
                                         reachable_xy, then fit; for use before
                                         real trajectories are available.

Synthetic reference paths for C3.2–C3.5 (workflow rule 6; no real data needed):
  line:    0.25 m straight push in +x
  arc:     quarter-circle, radius 0.12 m, starting at the origin
  s_curve: sinusoidal path, 0.20 m long, amplitude 0.06 m
All coordinates are in the path frame with the cube start at (0, 0).
"""
import numpy as np

# Measured from T1 recordings (decision log 2026-10-08)
HUMAN_X_RANGE = (-0.37, -0.08)
HUMAN_Y_RANGE = (-0.24,  0.07)


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

    def reachable_fraction(self, path_xy_human, reachable_xy_robot, margin: float = 0.005):
        """Fraction of mapped path points inside the robot's reachable hull.

        Uses the convex hull of reachable_xy_robot (at the push height), expanded
        by margin metres.  Falls back to a circle test if the hull cannot be built.

        :param path_xy_human:      (N, 2) path in human table coords.
        :param reachable_xy_robot: (M, 2) sampled EE positions at push height.
        :param margin:             Tolerance in metres (default 5 mm).
        :returns: float in [0, 1].
        """
        from scipy.spatial import ConvexHull, QhullError

        mapped = self.transform(np.asarray(path_xy_human, dtype=float))
        reach  = np.asarray(reachable_xy_robot, dtype=float)

        try:
            hull = ConvexHull(reach)
            A, b = hull.equations[:, :2], hull.equations[:, 2]
            inside = np.all(mapped @ A.T + b[None, :] <= margin, axis=1)
        except (QhullError, ValueError):
            c = reach.mean(axis=0)
            r = np.linalg.norm(reach - c, axis=1).max()
            inside = np.linalg.norm(mapped - c, axis=1) <= r + margin

        return float(inside.mean())

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
                  EE position sample used to build the hull.
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
