"""Space-Jacobian verification by numerical differentiation of FK.

Run from the repo root:  python -m tests.test_jacobian
"""
import numpy as np

from control.kinematics.core import FKinSpace, JacobianSpace
from tests.robot import M, Slist


def verify_jac(thetalist):
    thetalist = np.array(thetalist, dtype=float)
    step = 1e-4  # must be > NearZero threshold (1e-6) in core.MatrixExp6
    J_num = np.zeros((6, len(thetalist)))

    for i in range(len(thetalist)):
        theta_plus = thetalist.copy()
        theta_plus[i] += step

        T_plus = FKinSpace(M, Slist, theta_plus)
        T_base = FKinSpace(M, Slist, thetalist)

        dX = (T_plus @ np.linalg.inv(T_base)) / step

        omega = [dX[2, 1], dX[0, 2], dX[1, 0]]
        v = dX[:3, 3]

        J_num[:, i] = np.concatenate([omega, v])

    J_analytical = JacobianSpace(Slist, thetalist)

    err = np.max(np.abs(J_analytical - J_num))
    print(f"{'PASS' if err < 1e-4 else 'FAIL'}  max_err={err:.2e}  theta={np.round(thetalist, 3)}")
    return err


if __name__ == "__main__":
    verify_jac([0, 0, 0, 0, 0])
    verify_jac([-np.pi / 8, 0, 0, 0, 0])
    verify_jac([0, np.pi / 4, 0, 0, 0])
    verify_jac([0, 0, -np.pi / 4, 0, 0])
    verify_jac([np.pi / 8, -np.pi / 4, np.pi / 6, -np.pi / 8, np.pi / 3])
