# Origin of this directory

Product-of-exponentials kinematics for the SO-101, copied from my teleoperation repo
(https://github.com/jaimelopezruiz/so101-markerless-teleop, commit `ad74b78`). Vendored rather than
installed as a dependency: that repo's distribution requires `lerobot[feetech]==0.5.1`
(which pulls in `opencv-python-headless` and breaks `cv2.imshow`) and installs generic top-level
packages named `kinematics`, `urdf` and `tests`.

| File | Source | Change |
|------|--------|--------|
| `core.py` | `kinematics/core.py` | none |
| `ik.py` | `kinematics/ik.py` | `from kinematics.core` became `from .core` |
| `parser.py` | `urdf/parser.py` | none |
| `so101_new_calib.urdf` | `urdf/so101_new_calib.urdf` | none |

Tests copied to `tests/` (`robot.py`, `test_ik.py`, `test_jacobian.py`) have only their imports
rewritten, plus one line in `test_jacobian.py` (below). Their docstrings still cite DEVLOG entries that live
in the teleop repo.

| Test file | Change |
|-----------|--------|
| `tests/robot.py` | imports: `kinematics.core` and `urdf.parser` became `control.kinematics.core` and `control.kinematics.parser` |
| `tests/test_ik.py` | imports: `kinematics.core` and `kinematics.ik` became `control.kinematics.core` and `control.kinematics.ik` |
| `tests/test_jacobian.py` | imports as above; `verify_jac` now ends with `return err` so a wrapper can assert on it (it printed FAIL but could not fail) |

`tests/test_kinematics_vendored.py` runs these under pytest without modifying `test_ik.py` further.
`test_fk.py` (FK against yourdfpy) was not copied; FK is checked against MuJoCo instead.

## Licenses

- `core.py` is a trimmed copy of the Modern Robotics code library (MIT, NxRLab): `LICENSE-ModernRobotics`.
- `so101_new_calib.urdf` is the SO-101 description from TheRobotStudio/SO-ARM100 (Apache-2.0): `LICENSE-SO-ARM100`.
  The URDF references meshes under `assets/` that are not copied; the PoE parser reads only joint
  origins, axes and limits.
- `ik.py` and `parser.py` are my own code, covered by the repository's MIT `LICENSE`.
