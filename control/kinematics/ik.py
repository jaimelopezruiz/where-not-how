import numpy as np

from .core import FKinBody, JacobianBody, TransInv, MatrixLog6, se3ToVec

# A body twist is ordered [omega; v], so a position-only task steps on the linear
# rows. Defined once and used by BOTH dls_operator (slicing the Jacobian) and its
# callers (slicing the matching task vector). If those two ever disagree the step
# is silently meaningless rather than an error, so there is one definition.
LINEAR_ROWS = slice(3, 6)


def dls_operator(Blist, thetalist, *, position_only: bool = False, lam: float = 0.01):
    """Damped least-squares pseudo-inverse and null-space projector at ``thetalist``.

    Returns ``(J_pinv, N)`` with ``J_pinv = Jt^T (Jt Jt^T + lam^2 I)^-1`` and
    ``N = I - J_pinv Jt``. Both are policy-free: no gains, no step size, no
    integration. That is the point of the split. The iterative solver applies
    dimensionless per-iteration gains, while a velocity controller applies rates
    in 1/s and integrates, and the two share this linear algebra without sharing
    a control law.

    ``position_only`` slices the Jacobian here rather than in the caller, so the
    consumers cannot drift apart on the convention; the caller slices its own
    task vector with ``LINEAR_ROWS`` to match.

    The arithmetic is deliberately ``Jt.T @ inv(A)`` and must stay that way. The
    algebraically identical ``solve(A, Jt).T`` is not numerically identical: it
    moves round-trip solutions by up to 5.6 rad, because a 1e-15 difference here
    amplifies through Newton iteration into a different IK basin (see the V0
    update in DEVLOG Entry 6). ``solve(A, eye)`` is bit-identical and safe, since
    numpy implements ``inv`` that way.

    :param Blist: Screw axes in the end-effector (body) frame at home, columns
    :param thetalist: Joint angles to evaluate at (n,)
    :param position_only: Slice to the linear rows, leaving orientation free
    :param lam: DLS damping factor: higher = more stable but slower
    :return: (J_pinv, N): damped pseudo-inverse (n, m) and null-space projector (n, n)
    """
    J = JacobianBody(Blist, thetalist)
    Jt = J[LINEAR_ROWS, :] if position_only else J
    m = Jt.shape[0]
    J_pinv = Jt.T @ np.linalg.inv(Jt @ Jt.T + lam**2 * np.eye(m))  # damped pseudo-inverse
    N = np.eye(Jt.shape[1]) - J_pinv @ Jt
    return J_pinv, N


def IKinBodyDLS(Blist, M, T, thetalist0, joints_limits, eomg=1e-2, ev=5e-3, lam=0.01, maxiters=200,
               position_only: bool = False, theta_pref=None, k0: float = 0.0):
    """Computes inverse kinematics in the body frame for an open chain robot.

    Uses damped least-squares (DLS) Newton-Raphson iteration. Joint limits are
    re-clamped every iteration so each iterate stays inside the reachable joint
    space. Empirically this widens the convergence basin substantially versus
    clamping only the final result: at a noisy initial guess (~2 rad off) the
    round-trip success rate is 83% with in-loop clamping vs 35% without (see
    DEVLOG Entry 6 for the full sweep).

    When the arm is redundant for the task (e.g. a 5-DOF arm on a position-only
    target), the leftover freedom is otherwise resolved arbitrarily, which can
    leave the elbow/wrist in awkward poses. Passing ``theta_pref`` with ``k0>0``
    adds a null-space secondary task that biases the redundant joints toward the
    preferred posture without disturbing the primary (position) tracking, since
    the bias is projected through ``(I - J^+ J)``.

    :param Blist: Screw axes in the end-effector (body) frame at home, columns
    :param M: Home configuration of the end-effector (4x4)
    :param T: Desired end-effector configuration Tsd (4x4)
    :param thetalist0: Initial joint angle guess (n,)
    :param joints_limits: Joint limits array (n, 2), columns [lower, upper]
    :param eomg: Angular error tolerance (rad)
    :param ev: Linear error tolerance (m)
    :param lam: DLS damping factor: higher = more stable but slower
    :param maxiters: Maximum Newton-Raphson iterations
    :param position_only: Determines whether rotation is kept floating
    :param theta_pref: Preferred posture (n,) for null-space biasing, or None
    :param k0: Null-space gain; 0 disables the secondary task (default)
    :return: (thetalist, success): clamped joint angles and convergence flag
    """
    theta_pref = None if theta_pref is None else np.asarray(theta_pref, dtype=float)
    
    thetalist = np.array(thetalist0).copy()
    i = 0

    Tsb = FKinBody(M, Blist, thetalist)
    Vb = se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T)))  # Body twist to desired pose

    omega_b_mag = np.linalg.norm(Vb[0:3])  # Angular error magnitude
    v_b_mag     = np.linalg.norm(Vb[3:6])  # Linear error magnitude
    err = omega_b_mag > eomg or v_b_mag > ev

    while err and i < maxiters:
        thetalist_previous = thetalist.copy()

        J_pinv, N = dls_operator(Blist, thetalist, position_only=position_only, lam=lam)
        
        vb = Vb[LINEAR_ROWS] if position_only else Vb

        delta_theta = J_pinv @ vb
        if theta_pref is not None and k0 != 0.0:
            # Secondary task: pull toward theta_pref in the null space only.
            delta_theta = delta_theta + N @ (k0 * (theta_pref - thetalist))


        if not np.all(np.isfinite(delta_theta)):  # NaN/Inf guard: bail with last good theta
            return (thetalist_previous, False)

        thetalist = thetalist + delta_theta
        i += 1

        Tsb = FKinBody(M, Blist, thetalist)
        Vb = se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T)))
        omega_b_mag = np.linalg.norm(Vb[0:3])
        v_b_mag     = np.linalg.norm(Vb[3:6])
        err = (v_b_mag > ev) if position_only else (omega_b_mag > eomg or v_b_mag > ev)

        # Clamp to joint limits WITHIN the while loop
        thetalist = np.clip(thetalist, joints_limits[:, 0], joints_limits[:, 1])

    return (thetalist, not err)
