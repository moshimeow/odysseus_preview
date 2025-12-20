//! odysseus-solver: Compile-time sized automatic differentiation
//!
//! This crate provides forward-mode automatic differentiation using const generics.
//! All sizes are known at compile time, with workspace pre-allocated on the heap
//! for optimal performance in long-running VR tracking sessions.

mod jet;
pub mod math3d;
pub mod solver;
pub mod sparse_solver;

pub use jet::{Jet, Real};
pub use solver::LevenbergMarquardt;
pub use sparse_solver::{SparseLevenbergMarquardt, build_slam_entries};

// Re-export nalgebra for convenience
pub use nalgebra;
