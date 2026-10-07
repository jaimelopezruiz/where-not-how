"""Shared SO-101 model fixtures for the verification scripts."""
import numpy as np

from control.kinematics.core import Adjoint, TransInv, FKinBody
from control.kinematics.parser import findMnS

M, Slist, limits = findMnS()
# Body-frame screw axes: Blist = [Ad_{M^-1}] Slist, column by column.
Blist = np.array([Adjoint(TransInv(M)) @ Slist[:, i] for i in range(Slist.shape[1])]).T

# Order matches how findMnS builds Slist (reversed URDF order, arm joints only)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

thetalist = np.array([0, 0, -1.69, 0, 0])
T = FKinBody(M, Blist, thetalist)
reach = M[:3, 3] - T[:3, 3]

THETALIST_REST = np.array([0, -1.3, 0, 0, 0])
REACH_XYZ = FKinBody(M, Blist, THETALIST_REST)[:3, 3] - reach