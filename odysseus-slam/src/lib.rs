//! Odysseus SLAM: A compile-time SLAM backend using automatic differentiation
//!
//! This library implements bundle adjustment for visual SLAM using const generics
//! and forward-mode automatic differentiation from odysseus-solver.

pub mod math;
pub mod camera;
pub mod geometry;
pub mod world_state;
pub mod optimization;
pub mod simulation;
pub mod trajectory;
pub mod frame_graph;
pub mod slam_system;
pub mod imu;
// Re-export key types
pub use math::{SO3, SE3};
pub use camera::{PinholeCamera, StereoCamera};
pub use geometry::{Point3D, StereoObservation};
pub use world_state::{WorldState, PointInfo};
pub use frame_graph::{FrameGraph, FrameState, FrameRole, OptimizationState};
pub use slam_system::{SlamSystem, LbaToGbaMsg, GbaToLbaMsg};
pub use optimization::{run_bundle_adjustment, BundleAdjustmentConfig, BundleAdjustmentResult};
pub use odysseus_solver::{Jet, Real};