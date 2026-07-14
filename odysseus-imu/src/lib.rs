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
pub mod spline;
pub mod residuals;
pub mod optimization;

pub use types::{ImuMeasurement, ImuFrameState};
pub use preintegration::PreintegratedImu;
pub use simulator::ImuSimulator;
pub use residuals::{imu_preintegration_residual, bias_residual};
pub use optimization::{run_imu_optimization, ImuOptimizationResult};

// ── Continuous trajectory abstraction (moved from odysseus-slam 2026-07) ──
// The simulator consumes this; implementors (splines etc.) live with the
// consumers that own the trajectory types.

use nalgebra::Vector3;
use odysseus_solver::math3d::SE3;

/// Trait for continuous trajectories with analytical derivatives
///
/// Provides exact position, orientation, and velocity at any time t.
/// Time is normalized: t ∈ [0, 1] maps to the full trajectory duration.
pub trait ContinuousTrajectory {
    /// Get pose at normalized time t ∈ [0, 1]
    fn pose(&self, t: f64) -> SE3<f64>;

    /// Get linear velocity in world frame at normalized time t
    fn linear_velocity(&self, t: f64) -> Vector3<f64>;

    /// Get angular velocity in body frame at normalized time t
    ///
    /// For trajectories where angular velocity isn't analytically available,
    /// this can return None and the simulator will use finite differences.
    fn angular_velocity(&self, t: f64) -> Option<Vector3<f64>> {
        let _ = t;
        None
    }

    /// Sample n discrete poses from the trajectory
    fn sample_poses(&self, n: usize) -> Vec<SE3<f64>> {
        (0..n)
            .map(|i| {
                let t = if n > 1 {
                    i as f64 / (n - 1) as f64
                } else {
                    0.0
                };
                self.pose(t)
            })
            .collect()
    }
}

// ── Synthetic circular trajectory (moved with the simulator, 2026-07) ──

use odysseus_solver::math3d::{Vec3, SO3};
use odysseus_solver::{real_fn, Jet, Real};

/// Continuous circular trajectory with analytical derivatives
///
/// A proper horizontal circle in the XY plane with optional orientation oscillation.
/// Unlike `CircularTrajectory`, this correctly separates position and rotation.
///
/// Position: p(t) = R * [cos(2πt), sin(2πt), 0]
/// Velocity: v(t) = (2πR/T) * [-sin(2πt), cos(2πt), 0]  (in m/s)
///
/// Where t is normalized time [0, 1] and T is the total duration in seconds.
pub struct ContinuousCircularTrajectory {
    /// Circle radius in meters
    pub radius: f64,
    /// Total trajectory duration in seconds
    pub duration: f64,
    /// Amplitude of orientation oscillation in radians (0 for fixed orientation)
    pub orientation_amplitude: f64,
    /// Whether the camera looks tangent to the path (vs fixed orientation)
    pub look_tangent: bool,
    /// Amplitude of vertical (Z-axis) sinusoidal oscillation in meters (0 for flat circle)
    pub vertical_oscillation_amplitude: f64,
}

impl ContinuousCircularTrajectory {
    /// Create a simple horizontal circle
    ///
    /// Camera looks tangent to the path (in the direction of motion).
    pub fn new(radius: f64, duration: f64) -> Self {
        Self {
            radius,
            duration,
            orientation_amplitude: 0.0,
            look_tangent: true,
            vertical_oscillation_amplitude: 0.0,
        }
    }

    /// Create a circle with oscillating orientation
    ///
    /// Orientation oscillates around the tangent direction.
    pub fn with_oscillation(radius: f64, duration: f64, amplitude: f64) -> Self {
        Self {
            radius,
            duration,
            orientation_amplitude: amplitude,
            look_tangent: true,
            vertical_oscillation_amplitude: 0.0,
        }
    }

    /// Create a circle with fixed world-frame orientation
    pub fn fixed_orientation(radius: f64, duration: f64) -> Self {
        Self {
            radius,
            duration,
            orientation_amplitude: 0.0,
            look_tangent: false,
            vertical_oscillation_amplitude: 0.0,
        }
    }

    /// Add vertical (Z-axis) sinusoidal oscillation: z(t) = A * sin(2πt)
    pub fn with_vertical_oscillation(mut self, amplitude: f64) -> Self {
        self.vertical_oscillation_amplitude = amplitude;
        self
    }

}

impl ContinuousCircularTrajectory {
    #[real_fn]
    fn pose_generic<T: Real<Scalar = f64>>(&self, t: T) -> SE3<T> {
        let pi = T::from_f64(std::f64::consts::PI);
        let radius = T::from_f64(self.radius);
        let amplitude = T::from_f64(self.orientation_amplitude);
        let theta = t * 2.0R * pi;

        let vert_amp = T::from_f64(self.vertical_oscillation_amplitude);
        let translation = Vec3::new(
            theta.cos() * radius,
            theta.sin() * radius,
            (theta * 4.0R).sin() * vert_amp,
        );

        let rotation = if self.look_tangent {
            let yaw = theta + pi / 2.0R;
            let osc_x = (theta * 2.0R).sin() * amplitude;
            let osc_y = (theta * 2.0R + 2.0R * pi / 3.0R).sin() * amplitude;
            let rot_yaw = SO3::exp(Vec3::new(T::zero(), T::zero(), yaw));
            let rot_osc = SO3::exp(Vec3::new(osc_x, osc_y, T::zero()));
            (rot_yaw * rot_osc).normalize()
        } else if self.orientation_amplitude > 0.0 {
            let osc_x = (theta * 2.0R).sin() * amplitude;
            let osc_y = (theta * 2.0R + 2.0R * pi / 3.0R).sin() * amplitude;
            let osc_z = (theta * 2.0R + 4.0R * pi / 3.0R).sin() * amplitude;
            SO3::exp(Vec3::new(osc_x, osc_y, osc_z))
        } else {
            SO3::identity()
        };

        SE3::from_rotation_translation(rotation, translation)
    }
}

impl ContinuousTrajectory for ContinuousCircularTrajectory {
    fn pose(&self, t: f64) -> SE3<f64> {
        self.pose_generic(t)
    }

    fn linear_velocity(&self, t: f64) -> Vector3<f64> {
        // Differentiate translation w.r.t. normalized t, then scale to real time
        let t_jet = Jet::<f64, 1>::variable(t, 0);
        let pose = self.pose_generic(t_jet);
        Vector3::new(
            pose.translation.x.derivs[0] / self.duration,
            pose.translation.y.derivs[0] / self.duration,
            pose.translation.z.derivs[0] / self.duration,
        )
    }

    fn angular_velocity(&self, t: f64) -> Option<Vector3<f64>> {
        // Body-frame angular velocity: ω = 2 * Im(q̄ * dq/dt_real)
        let t_jet = Jet::<f64, 1>::variable(t, 0);
        let pose = self.pose_generic(t_jet);
        let q = pose.rotation.quat;

        // Quaternion values and their derivatives w.r.t. normalized t
        let (qw, qx, qy, qz) = (q.w.value, q.x.value, q.y.value, q.z.value);
        // Scale by 1/duration to convert from d/dt_normalized to d/dt_real
        let (dqw, dqx, dqy, dqz) = (
            q.w.derivs[0] / self.duration,
            q.x.derivs[0] / self.duration,
            q.y.derivs[0] / self.duration,
            q.z.derivs[0] / self.duration,
        );

        // Im(q̄ * dq) where q̄ = (qw, -qx, -qy, -qz)
        let wx = 2.0 * (qw * dqx - dqw * qx - qy * dqz + qz * dqy);
        let wy = 2.0 * (qw * dqy - dqw * qy - qz * dqx + qx * dqz);
        let wz = 2.0 * (qw * dqz - dqw * qz - qx * dqy + qy * dqx);

        Some(Vector3::new(wx, wy, wz))
    }
}
