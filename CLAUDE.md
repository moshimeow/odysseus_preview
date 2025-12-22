Hi Claude! Thanks for being here!

Remember to use your subagents!

Structure of the latest SLAM system:
odysseus-slam/examples/incremental_slam_demo.rs is the main file that calls everything else
odysseus-slam/src/optimization/mod.rs contains the code that actually calls the solver to do the hard work. This is where things get parameterized.
odysseus-solver/src/sparse_solver.rs is the solver
odysseus-slam/src contains all of the suporting stuff for the demo and optimization, including some of the 3D math
odysseus-solver/src contains definitions for Jets and Reals to do autodiff with, the solver, and some 3D math

When running odysseus-slam/examples/incremental_slam_demo.rs , it is necessary to first cd into odysseus-slam, it needs to access data files found in that folder.

See odysseus-solver/SPRS_NOTES.md for documentation on using the sprs sparse matrix library (trait bounds, gotchas, etc).

Remember that you need to initialize Rerun recording streams