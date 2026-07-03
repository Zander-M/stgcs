# Official OMPL K-CBS Native Backend

This optional extension builds `baselines._ompl_kcbs_native`, used by
`baselines.ompl_kcbs.OfficialOMPLKCBS`.

Unlike ST-RRT*, K-CBS is not available in standard OMPL. Build this module
against the official K-CBS Multi-Robot-OMPL fork:

```bash
git clone https://github.com/aria-systems-group/Multi-Robot-OMPL.git /tmp/Multi-Robot-OMPL-KCBS
cmake -S /tmp/Multi-Robot-OMPL-KCBS -B /tmp/Multi-Robot-OMPL-KCBS-build -G Ninja \
  -DCMAKE_INSTALL_PREFIX=/tmp/Multi-Robot-OMPL-KCBS-install
cmake --build /tmp/Multi-Robot-OMPL-KCBS-build
cmake --install /tmp/Multi-Robot-OMPL-KCBS-build
```

Then build the adapter from this repository root:

```bash
cmake \
  -S baselines/ompl_kcbs_native \
  -B /private/tmp/stgcs_ompl_kcbs_build \
  -G Ninja \
  -DPython_EXECUTABLE=/Users/jingtao/opt/miniconda3/envs/agmt/bin/python \
  -DCMAKE_PREFIX_PATH=/tmp/Multi-Robot-OMPL-KCBS-install
cmake --build /private/tmp/stgcs_ompl_kcbs_build
```

The adapter instantiates the official `ompl::multirobot::control::KCBS`
planner. Each MRMP robot is modeled as a 2D disk with single-integrator
controls `x_dot = u`, `u_i in [-vlimit, vlimit]`. Static workspace validity is
checked inside this native extension from the `Env` polygon geometry: polygonal
obstacles are tested against the robot footprint, and grid-style environments
without explicit static obstacles test membership in the C-space polygon union.
Robot-robot collision is the disk-distance check used by the K-CBS demo.
Constraint-tree behavior remains inside the official OMPL fork.
