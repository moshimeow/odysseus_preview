//! IMU integration module for Visual-Inertial Odometry
//!
//! This module provides IMU preintegration and residual computation for
//! tightly-coupled visual-inertial SLAM.
//!
//! Key components:
//! - `ImuMeasurement`: Raw gyroscope and accelerometer readings
//! - `ImuFrameState`: Velocity and bias state for a frame (separate from Pose)
//! - `PreintegratedImu`: Preintegrated IMU measurements between keyframes
//! - `ImuSimulator`: Synthetic IMU generation from trajectories

pub mod types;
pub mod preintegration;
pub mod simulator;
pub mod residuals;
pub mod optimization;

pub use types::{ImuMeasurement, ImuFrameState};
pub use preintegration::PreintegratedImu;
pub use simulator::ImuSimulator;
pub use residuals::{imu_preintegration_residual, bias_residual};
pub use optimization::{run_imu_optimization, ImuOptimizationResult};
