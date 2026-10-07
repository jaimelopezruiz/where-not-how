import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Bundled SO-101 description, resolved relative to this file so the parser
# works regardless of the current working directory.
DEFAULT_URDF = Path(__file__).resolve().parent / "so101_new_calib.urdf"


def rpyToRot(rpy):
    """Converts fixed-axis roll-pitch-yaw angles to a rotation matrix (Rz @ Ry @ Rx)."""
    r, p, y = rpy
    Rx = np.array([[1, 0,         0        ],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [ 0,         1, 0        ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]])
    return Rz @ Ry @ Rx


def findMnS(urdf_path=DEFAULT_URDF):
    """Parses a URDF and extracts the kinematic model in PoE form.

    Walks the joints in reverse URDF order (arm joints only, gripper skipped),
    accumulating the home transform and building the space-frame screw axes.

    :param urdf_path: Path to the URDF file
    :return M: Home configuration of the end-effector (4x4)
    :return Slist: Space-frame screw axes at home, as columns (6 x n)
    :return limits: Joint limits, rows [lower, upper] (n x 2)
    """
    root = ET.parse(urdf_path).getroot()
    robot = SimpleNamespace()

    T = np.eye(4)
    Slist = np.empty((6, 0))
    limits = []
    for joint in reversed(root.findall('joint')):
        origin = joint.find('origin')
        axis = joint.find('axis')
        name = joint.get('name')

        T_local = np.zeros((4, 4))
        R_cumulative = np.zeros((3, 3))

        if origin is not None and name != 'gripper':
            setattr(robot, name, SimpleNamespace(
                xyz=np.array([float(v) for v in origin.get('xyz').split()]),
                rpy=np.array([float(v) for v in origin.get('rpy').split()]),
                axis=np.array([float(v) for v in axis.get('xyz').split()])
            ))

            ## FOR M:
            T_local[:3, :3] = rpyToRot(getattr(robot, name).rpy)
            T_local[:3, 3] = getattr(robot, name).xyz
            T_local[3, 3] = 1

            T = T @ T_local
            R_cumulative = T[:3, :3]

            ## FOR Slist and limits:
            if axis is not None and joint.get('type') != 'fixed' and name != 'gripper':
                omega = R_cumulative @ getattr(robot, name).axis
                q = T[:3, 3]
                v = -np.cross(omega, q)
                S_local = np.concatenate([omega, v]).reshape(6, 1)
                Slist = np.hstack([Slist, S_local])

                lim = joint.find('limit')
                limits.append([float(lim.get('lower')), float(lim.get('upper'))])

    return T, Slist, np.array(limits)
