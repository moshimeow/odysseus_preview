Hi Claude! Thanks for being here!

Please run the demos in release mode

Remember to use your subagents!

Structure of the latest SLAM system:
odysseus-slam/examples/incremental_slam_demo.rs, vio_demo.rs are the main files that calls everything else. incremental_slam_demo doesn't use the IMU and vio_demo does.
odysseus-slam/src/optimization contains the code that actually calls the solver to do the hard work. This is where things get parameterized. mod.rs has utilities, slam.rs and vio.rs do most of the work for their respective demos, and marginalization.rs handles marginalization.
odysseus-solver/src/sparse_solver.rs is the solver
odysseus-slam/src contains all of the suporting stuff for the demo and optimization, including some of the 3D math
odysseus-solver/src contains definitions for Jets and Reals to do autodiff with, the solver, and some 3D math

When running odysseus-slam/examples/incremental_slam_demo.rs , it is necessary to first cd into odysseus-slam, it needs to access data files found in that folder.

See odysseus-solver/SPRS_NOTES.md for documentation on using the sprs sparse matrix library (trait bounds, gotchas, etc).

Remember that you need to initialize Rerun recording streams