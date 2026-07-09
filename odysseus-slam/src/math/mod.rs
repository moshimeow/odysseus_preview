//! Lie group math for rotations and poses.
//!
//! The implementations moved into `odysseus_solver::math3d` (2026-07) so the
//! solver layer owns all geometry; these re-exports keep existing
//! `odysseus_slam::{SE3, SO3}` / `odysseus_slam::math::stereographic` paths
//! working.

pub use odysseus_solver::math3d::stereographic;
pub use odysseus_solver::math3d::{SE3, SE3Tangent, SO3};
