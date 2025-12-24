#![recursion_limit = "1024"]
//! Odysseus SLAM: A compile-time SLAM backend using automatic differentiation
//!
//! This library implements bundle adjustment for visual SLAM using const generics
//! and forward-mode automatic differentiation from odysseus-solver.

pub mod camera;
pub mod frame_graph;
pub mod geometry;
pub mod imu;
pub mod math;
pub mod optimization;
pub mod simulation;
pub mod slam_system;
pub mod spline;
pub mod trajectory;
pub mod world_state;
// Re-export key types
pub use camera::{PinholeCamera, StereoCamera};
pub use frame_graph::{FrameGraph, FrameRole, FrameState, OptimizationState};
pub use geometry::{Point3D, StereoObservation};
pub use math::{SE3, SO3};
pub use odysseus_solver::{Jet, Real};
pub use optimization::{run_bundle_adjustment, BundleAdjustmentConfig, BundleAdjustmentResult};
pub use slam_system::{GbaToLbaMsg, LbaToGbaMsg, SlamSystem};
pub use world_state::{PointInfo, WorldState};
