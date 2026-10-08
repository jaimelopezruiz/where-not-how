"""EE velocity controller wrapping the DLS IK library (C3.3).

Action: (vx, vy) planar EE velocity at a fixed height, or a direct (x, y) target,
        → DLS IK → joint angles for the position actuators.

Two G1 findings are applied (decision log 2026-10-07):
  1. Target orientation is taken from the current FK, not a fixed home pose:
     the home-pose orientation is often unreachable at the target position and
     dropped IK convergence from 100 % to 61 % in the G1 sweep.
  2. FK re-check on the returned angles: IKinBodyDLS clamps joint angles inside
     the loop after evaluating the convergence test, so a "converged" solution
     that hit a joint limit is returned clamped but verified unclamped.  In the
     200-trial G1 sweep, 8 solutions missed by more than 1 mm; all 8 had a joint
     exactly on a limit.  The wrapper rejects these by re-running FK and comparing
     the actual position error against ev.
"""
import numpy as np

from control.kinematics.core import Adjoint, FKinBody, TransInv
from control.kinematics.ik import IKinBodyDLS
from control.kinematics.parser import DEFAULT_URDF, findMnS


class EEController:
    """Planar EE velocity → DLS IK → joint angles.

    Works in the robot base frame.  When the robot is placed at the world origin
    (as in the push scene) the base frame equals the world frame.

    :param urdf_path: URDF for the PoE model (defaults to the vendored SO-101).
    :param ev:        Linear position error tolerance in metres (default 1 mm).
    """

    def __init__(self, urdf_path=DEFAULT_URDF, ev: float = 1e-3, chunk_iters: int = 2,
                 max_chunks: int = 6, min_contraction: float = 0.7):
        M, Slist, limits = findMnS(urdf_path)
        n = Slist.shape[1]
        Blist = np.array([Adjoint(TransInv(M)) @ Slist[:, i] for i in range(n)]).T
        self._M = M
        self._Blist = Blist
        self._limits = limits
        self._ev = float(ev)
        self._chunk_iters, self._max_chunks, self._min_contraction = chunk_iters, max_chunks, min_contraction
        self._q = np.zeros(n)
        self._z_push: float | None = None  # locked on the first solve call

    def reset(self, q0=None, z_push=None):
        """Set the starting configuration and optionally lock the push height.

        :param q0:     Initial joint angles (n,).  Defaults to all-zeros.
        :param z_push: Fixed EE height for subsequent calls to solve() / step().
                       If None, the height is locked from FK at the first solve().
        """
        self._q = (np.zeros(self._limits.shape[0])
                   if q0 is None else np.asarray(q0, dtype=float).copy())
        self._z_push = None if z_push is None else float(z_push)

    @property
    def q(self):
        """Current internal joint angles (copy)."""
        return self._q.copy()

    def ee_pose(self, q=None):
        """FK: 4×4 EE pose in robot base frame at the given (or current) angles."""
        return FKinBody(self._M, self._Blist, self._q if q is None else q)

    def _pos_err(self, q, T_tgt):
        return float(np.linalg.norm(FKinBody(self._M, self._Blist, q)[:3, 3] - T_tgt[:3, 3]))

    def solve(self, xy_target, z_fixed=None):
        """Compute joint angles to reach (x, y) at fixed height.

        On the first call, if neither z_fixed nor self._z_push is set, the height
        is locked to the current FK z-coordinate (the height the arm is already at).

        :param xy_target: Desired (x, y) in robot base frame (metres).
        :param z_fixed:   Override the locked height for this call only.
        :returns: (q, success)
            q:       Solution angles (copy of previous q on failure).
            success: False if IK did not converge OR the FK re-check finds the
                     returned solution misses the target by more than ev.
        """
        T_cur = self.ee_pose()

        if z_fixed is None:
            if self._z_push is None:
                self._z_push = float(T_cur[2, 3])
            z = self._z_push
        else:
            z = float(z_fixed)

        # G1 finding 1: use current FK orientation so the target is always reachable
        T_tgt = T_cur.copy()
        T_tgt[0, 3] = float(xy_target[0])
        T_tgt[1, 3] = float(xy_target[1])
        T_tgt[2, 3] = z

        # Warm start from the current angles and run the library in short chunks. A reachable target is a
        # Newton problem that contracts the error by an order of magnitude per iteration (1-6 iterations in
        # practice); an unreachable one (joint limit, out of reach) stalls. Stop as soon as a chunk fails to
        # shrink the error, instead of burning the library's 200 iterations on every step. The iterates are
        # exactly those of one long call: the only state carried between chunks is the joint vector.
        q_sol, converged = self._q, False
        err = self._pos_err(q_sol, T_tgt)
        for _ in range(self._max_chunks):
            q_sol, converged = IKinBodyDLS(
                self._Blist, self._M, T_tgt, q_sol, self._limits,
                ev=self._ev, position_only=True, maxiters=self._chunk_iters,
            )
            if converged:
                break
            new_err = self._pos_err(q_sol, T_tgt)
            if new_err > self._min_contraction * err:
                break
            err = new_err

        if not converged:
            return self._q.copy(), False

        # G1 finding 2: FK re-check catches joint-limit clamp errors
        actual_err = np.linalg.norm(FKinBody(self._M, self._Blist, q_sol)[:3, 3] - T_tgt[:3, 3])
        if actual_err > self._ev:
            return self._q.copy(), False

        self._q = q_sol
        return q_sol.copy(), True

    def step(self, vx: float, vy: float, dt: float, z_fixed=None):
        """Integrate planar EE velocity over dt and solve IK.

        :param vx, vy: EE velocity in robot base frame (m/s).
        :param dt:     Integration time step (s).
        :returns: (q, success)
        """
        T = self.ee_pose()
        return self.solve((T[0, 3] + vx * dt, T[1, 3] + vy * dt), z_fixed=z_fixed)
