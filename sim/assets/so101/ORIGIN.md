# Origin of this directory

The SO-101 MuJoCo model from TheRobotStudio/SO-ARM100 (https://github.com/TheRobotStudio/SO-ARM100,
commit `a758567`, `Simulation/SO101/`), copied unchanged:

- `so101_new_calib.xml`: the MJCF (new calibration; joint ranges match the URDF in `control/kinematics`).
- `assets/`: only the 13 STL meshes the MJCF references (the repository's other files are not copied).
- `LICENSE`: Apache-2.0, from that repository, covering these files.

`Simulation/SO101/so101_new_calib.urdf` in that repository is byte-identical (modulo line endings) to
`control/kinematics/so101_new_calib.urdf`, so the PoE model and this MJCF come from the same CAD export.
`sim/fk_check.py` checks them against each other.
