//! IMU Simulator
//!
//! Generates synthetic IMU measurements from ground truth trajectories.
//! Useful for testing VIO algorithms before real sensor integration.

use nalgebra::Vector3;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use super::types::ImuMeasurement;
use crate::math::SE3;
use crate::trajectory::ContinuousTrajectory;

/// IMU noise parameters
///
/// Typical values for a consumer-grade IMU (e.g., BMI055):
/// - gyro_noise_density: 0.004 rad/s/√Hz
/// - accel_noise_density: 0.08 m/s²/√Hz
/// - gyro_bias_random_walk: 0.0002 rad/s²/√Hz
/// - accel_bias_random_walk: 0.003 m/s³/√Hz
#[derive(Debug, Clone)]
pub struct ImuNoiseParams {
    /// Gyroscope white noise density (rad/s/√Hz)
    pub gyro_noise_density: f64,
    /// Accelerometer white noise density (m/s²/√Hz)
    pub accel_noise_density: f64,
    /// Gyroscope bias random walk (rad/s²/√Hz)
    pub gyro_bias_random_walk: f64,
    /// Accelerometer bias random walk (m/s³/√Hz)
    pub accel_bias_random_walk: f64,
}

impl ImuNoiseParams {
    /// Noise-free IMU (for testing)
    pub fn zero() -> Self {
        Self {
            gyro_noise_density: 0.0,
            accel_noise_density: 0.0,
            gyro_bias_random_walk: 0.0,
            accel_bias_random_walk: 0.0,
        }
    }

    /// Valve Index IMU noise
    /// from https://invensense.tdk.com/wp-content/uploads/2016/10/DS-000176-ICM-20602-v1.0.pdf
    /// Gyro noise density: 7e-5
    /// Accel noise density: 1e-3
    /// estimated safe bias random walk:
    /// gyro: 4e-6
    /// accel: 4e-4
    pub fn consumer_grade() -> Self {
        Self {
            gyro_noise_density: 7e-5,
            accel_noise_density: 1e-3,
            gyro_bias_random_walk: 4e-6,
            accel_bias_random_walk: 4e-4,
        }
    }
}

impl Default for ImuNoiseParams {
    fn default() -> Self {
        Self::consumer_grade()
    }
}

/// IMU Simulator
///
/// Generates synthetic IMU measurements from a sequence of SE3 poses.
#[derive(Debug, Clone)]
pub struct ImuSimulator {
    /// Noise parameters
    pub noise_params: ImuNoiseParams,
    /// IMU sampling rate (Hz)
    pub imu_rate: f64,
    /// Gravity vector in world frame (m/s²)
    /// Default: [0, 0, -9.81] (gravity points down in world frame)
    pub gravity: Vector3<f64>,
}

impl ImuSimulator {
    /// Create a new IMU simulator
    pub fn new(noise_params: ImuNoiseParams, imu_rate: f64) -> Self {
        Self {
            noise_params,
            imu_rate,
            gravity: Vector3::new(0.0, 0.0, -9.81),
        }
    }

    /// Create a noise-free simulator for testing
    pub fn ideal(imu_rate: f64) -> Self {
        Self::new(ImuNoiseParams::zero(), imu_rate)
    }

    /// Set custom gravity vector
    pub fn with_gravity(mut self, gravity: Vector3<f64>) -> Self {
        self.gravity = gravity;
        self
    }

    /// Generate IMU measurements from ground truth poses
    ///
    /// # Arguments
    /// * `poses` - Sequence of (timestamp, pose) pairs representing the trajectory
    /// * `seed` - Random seed for noise generation
    ///
    /// # Returns
    /// Vector of IMU measurements at the configured rate
    pub fn generate_from_trajectory(
        &self,
        poses: &[(f64, SE3<f64>)],
        seed: u64,
    ) -> Vec<ImuMeasurement> {
        if poses.len() < 3 {
            return Vec::new();
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut measurements = Vec::new();

        // Current bias state (random walk)
        let mut gyro_bias = Vector3::zeros();
        let mut accel_bias = Vector3::zeros();

        let start_time = poses[0].0;
        let end_time = poses[poses.len() - 1].0;
        let dt = 1.0 / self.imu_rate;

        let mut current_time = start_time + dt; // Start after first pose to have history

        while current_time < end_time - dt {
            // Find the pose index where poses[idx].0 <= current_time < poses[idx+1].0
            let pose_idx = self.find_pose_index(poses, current_time);

            if pose_idx == 0 || pose_idx + 1 >= poses.len() {
                current_time += dt;
                continue;
            }

            // Use three poses for acceleration computation
            let (t_prev, pose_prev) = &poses[pose_idx - 1];
            let (t_curr, pose_curr) = &poses[pose_idx];
            let (t_next, pose_next) = &poses[pose_idx + 1];

            // Get angular velocity and linear acceleration at this time
            let (gyro, accel) = self.compute_imu_at_time_3point(
                pose_prev,
                *t_prev,
                pose_curr,
                *t_curr,
                pose_next,
                *t_next,
                current_time,
            );

            // Add measurement noise (Gaussian)
            let gyro_noise = self.sample_gyro_noise(&mut rng, dt);
            let accel_noise = self.sample_accel_noise(&mut rng, dt);

            // Update bias random walk
            self.update_bias_random_walk(&mut rng, dt, &mut gyro_bias, &mut accel_bias);

            // Create measurement with noise and bias
            let measurement = ImuMeasurement::new(
                current_time,
                gyro + gyro_noise + gyro_bias,
                accel + accel_noise + accel_bias,
            );

            measurements.push(measurement);
            current_time += dt;
        }

        measurements
    }

    /// Generate IMU measurements from a continuous trajectory with analytical derivatives
    ///
    /// This method uses the trajectory's analytical velocity to compute acceleration
    /// as Δv/Δt, which models how real IMUs work (averaging over the sample period).
    /// This avoids the systematic errors from finite-differencing discrete poses.
    ///
    /// # Arguments
    /// * `trajectory` - Continuous trajectory with analytical derivatives
    /// * `seed` - Random seed for noise generation
    ///
    /// # Returns
    /// Vector of IMU measurements at the configured rate
    pub fn generate_from_continuous_trajectory<T: ContinuousTrajectory>(
        &self,
        trajectory: &T,
        duration: f64,
        seed: u64,
    ) -> Vec<ImuMeasurement> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut measurements = Vec::new();

        // Current bias state (random walk)
        let mut gyro_bias = Vector3::zeros();
        let mut accel_bias = Vector3::zeros();

        let dt = 1.0 / self.imu_rate;

        // Start at dt to have a previous sample for Δv computation
        let mut current_time = dt;

        while current_time < duration - dt {
            // Normalized times
            let t_curr = current_time / duration;
            let t_prev = (current_time - dt) / duration;

            // Get poses at current and previous time
            let pose_curr = trajectory.pose(t_curr);
            let pose_prev = trajectory.pose(t_prev);

            // === Gyroscope: relative rotation over dt ===
            let gyro = if let Some(ang_vel) = trajectory.angular_velocity(t_curr) {
                // Use analytical angular velocity if available
                ang_vel
            } else {
                // Compute from relative rotation: ω = log(R_prev^T * R_curr) / dt
                let r_rel = pose_prev.rotation.inverse() * pose_curr.rotation;
                let omega_vec = r_rel.log();
                Vector3::new(omega_vec.x, omega_vec.y, omega_vec.z) / dt
            };

            // === Accelerometer: Δv/Δt using analytical velocity ===
            let v_curr = trajectory.linear_velocity(t_curr);
            let v_prev = trajectory.linear_velocity(t_prev);
            let accel_world = (v_curr - v_prev) / dt;

            // Transform to body frame at midpoint of interval for better accuracy
            let t_mid = (t_curr + t_prev) * 0.5;
            let pose_mid = trajectory.pose(t_mid);
            let r_body_from_world = pose_mid.rotation.inverse();

            let accel_world_vec =
                odysseus_solver::math3d::Vec3::new(accel_world.x, accel_world.y, accel_world.z);
            let accel_body_vec = r_body_from_world.rotate(accel_world_vec);
            let accel_motion_body =
                Vector3::new(accel_body_vec.x, accel_body_vec.y, accel_body_vec.z);

            // Transform gravity to body frame
            let gravity_body_vec = r_body_from_world.rotate(odysseus_solver::math3d::Vec3::new(
                self.gravity.x,
                self.gravity.y,
                self.gravity.z,
            ));
            let gravity_body =
                Vector3::new(gravity_body_vec.x, gravity_body_vec.y, gravity_body_vec.z);

            // Accelerometer measures specific force: a_imu = a_motion - g_body
            let accel_body = accel_motion_body - gravity_body;

            // Add measurement noise
            let gyro_noise = self.sample_gyro_noise(&mut rng, dt);
            let accel_noise = self.sample_accel_noise(&mut rng, dt);

            // Update bias random walk
            self.update_bias_random_walk(&mut rng, dt, &mut gyro_bias, &mut accel_bias);

            // Create measurement with noise and bias
            let measurement = ImuMeasurement::new(
                current_time,
                gyro + gyro_noise + gyro_bias,
                accel_body + accel_noise + accel_bias,
            );

            measurements.push(measurement);
            current_time += dt;
        }

        measurements
    }

    /// Find the pose index where poses[idx].0 <= time < poses[idx+1].0
    fn find_pose_index(&self, poses: &[(f64, SE3<f64>)], time: f64) -> usize {
        for i in 0..poses.len() - 1 {
            if poses[i].0 <= time && time < poses[i + 1].0 {
                return i;
            }
        }
        poses.len() - 2
    }

    /// Compute ideal (noise-free) IMU readings using 3-point stencil for acceleration
    fn compute_imu_at_time_3point(
        &self,
        pose_prev: &SE3<f64>,
        t_prev: f64,
        pose_curr: &SE3<f64>,
        t_curr: f64,
        pose_next: &SE3<f64>,
        t_next: f64,
        current_time: f64,
    ) -> (Vector3<f64>, Vector3<f64>) {
        // Compute velocities using finite differences
        let dt_prev = t_curr - t_prev;
        let dt_next = t_next - t_curr;

        // Velocity at t_curr (backward difference)
        let v_prev = Vector3::new(
            (pose_curr.translation.x - pose_prev.translation.x) / dt_prev,
            (pose_curr.translation.y - pose_prev.translation.y) / dt_prev,
            (pose_curr.translation.z - pose_prev.translation.z) / dt_prev,
        );

        // Velocity at t_next (forward difference from curr)
        let v_next = Vector3::new(
            (pose_next.translation.x - pose_curr.translation.x) / dt_next,
            (pose_next.translation.y - pose_curr.translation.y) / dt_next,
            (pose_next.translation.z - pose_curr.translation.z) / dt_next,
        );

        // Acceleration at t_curr (central difference of velocity)
        let dt_mid = (dt_prev + dt_next) / 2.0;
        let accel_world = (v_next - v_prev) / dt_mid;

        // Angular velocity: from relative rotation between adjacent poses
        let r_curr = pose_curr.rotation;
        let r_next = pose_next.rotation;
        let r_rel = r_curr.inverse() * r_next;
        let omega_vec = r_rel.log();
        let omega = Vector3::new(omega_vec.x, omega_vec.y, omega_vec.z) / dt_next;

        // Interpolate rotation for body frame at current_time
        let alpha = (current_time - t_curr) / dt_next;
        let tangent_curr = pose_curr.log();
        let tangent_next = pose_next.log();
        let tangent_interp = [
            tangent_curr[0] * (1.0 - alpha) + tangent_next[0] * alpha,
            tangent_curr[1] * (1.0 - alpha) + tangent_next[1] * alpha,
            tangent_curr[2] * (1.0 - alpha) + tangent_next[2] * alpha,
            tangent_curr[3] * (1.0 - alpha) + tangent_next[3] * alpha,
            tangent_curr[4] * (1.0 - alpha) + tangent_next[4] * alpha,
            tangent_curr[5] * (1.0 - alpha) + tangent_next[5] * alpha,
        ];
        let pose_interp = SE3::exp(tangent_interp);

        // Transform to body frame
        let r_body_from_world = pose_interp.rotation.inverse();

        // Transform gravity to body frame
        let gravity_body_vec = r_body_from_world.rotate(odysseus_solver::math3d::Vec3::new(
            self.gravity.x,
            self.gravity.y,
            self.gravity.z,
        ));
        let gravity_body = Vector3::new(gravity_body_vec.x, gravity_body_vec.y, gravity_body_vec.z);

        // Transform motion acceleration to body frame
        let accel_world_vec =
            odysseus_solver::math3d::Vec3::new(accel_world.x, accel_world.y, accel_world.z);
        let accel_body_vec = r_body_from_world.rotate(accel_world_vec);
        let accel_motion_body = Vector3::new(accel_body_vec.x, accel_body_vec.y, accel_body_vec.z);

        // Accelerometer measures specific force: a_imu = a_motion - g_body
        // (IMU measures the force that would be needed to produce the motion, minus gravity)
        let accel_body = accel_motion_body - gravity_body;

        (omega, accel_body)
    }

    /// Sample gyroscope measurement noise (Gaussian)
    fn sample_gyro_noise<R: Rng>(&self, rng: &mut R, dt: f64) -> Vector3<f64> {
        let sigma = self.noise_params.gyro_noise_density / dt.sqrt();
        if sigma < 1e-15 {
            return Vector3::zeros();
        }
        let normal = Normal::new(0.0, sigma).unwrap();
        Vector3::new(normal.sample(rng), normal.sample(rng), normal.sample(rng))
    }

    /// Sample accelerometer measurement noise (Gaussian)
    fn sample_accel_noise<R: Rng>(&self, rng: &mut R, dt: f64) -> Vector3<f64> {
        let sigma = self.noise_params.accel_noise_density / dt.sqrt();
        if sigma < 1e-15 {
            return Vector3::zeros();
        }
        let normal = Normal::new(0.0, sigma).unwrap();
        Vector3::new(normal.sample(rng), normal.sample(rng), normal.sample(rng))
    }

    /// Update bias random walk (Gaussian)
    fn update_bias_random_walk<R: Rng>(
        &self,
        rng: &mut R,
        dt: f64,
        gyro_bias: &mut Vector3<f64>,
        accel_bias: &mut Vector3<f64>,
    ) {
        let gyro_sigma = self.noise_params.gyro_bias_random_walk * dt.sqrt();
        let accel_sigma = self.noise_params.accel_bias_random_walk * dt.sqrt();

        if gyro_sigma > 1e-15 {
            let gyro_normal = Normal::new(0.0, gyro_sigma).unwrap();
            *gyro_bias += Vector3::new(
                gyro_normal.sample(rng),
                gyro_normal.sample(rng),
                gyro_normal.sample(rng),
            );
        }

        if accel_sigma > 1e-15 {
            let accel_normal = Normal::new(0.0, accel_sigma).unwrap();
            *accel_bias += Vector3::new(
                accel_normal.sample(rng),
                accel_normal.sample(rng),
                accel_normal.sample(rng),
            );
        }
    }
}

/// Generate timestamped poses from a trajectory generator
///
/// # Arguments
/// * `poses` - Poses from trajectory generator (no timestamps)
/// * `total_duration` - Total duration in seconds
///
/// # Returns
/// Vector of (timestamp, pose) pairs
pub fn add_timestamps_to_poses(poses: Vec<SE3<f64>>, total_duration: f64) -> Vec<(f64, SE3<f64>)> {
    if poses.is_empty() {
        return Vec::new();
    }

    let n = poses.len();
    poses
        .into_iter()
        .enumerate()
        .map(|(i, pose)| {
            let t = total_duration * (i as f64) / ((n - 1).max(1) as f64);
            (t, pose)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{
        CircularTrajectory, ContinuousCircularTrajectory, ContinuousTrajectory, TrajectoryGenerator,
    };

    #[test]
    fn test_imu_simulator_creation() {
        let sim = ImuSimulator::new(ImuNoiseParams::consumer_grade(), 200.0);
        assert_eq!(sim.imu_rate, 200.0);
        assert_eq!(sim.gravity, Vector3::new(0.0, 0.0, -9.81));
    }

    #[test]
    fn test_ideal_simulator() {
        let sim = ImuSimulator::ideal(100.0);
        assert_eq!(sim.noise_params.gyro_noise_density, 0.0);
        assert_eq!(sim.noise_params.accel_noise_density, 0.0);
    }

    #[test]
    fn test_generate_from_stationary() {
        // Three identical poses = stationary (need 3 for 3-point stencil)
        let pose = SE3::<f64>::identity();
        let poses = vec![(0.0, pose), (0.5, pose), (1.0, pose)];

        let sim = ImuSimulator::ideal(100.0);
        let measurements = sim.generate_from_trajectory(&poses, 42);

        // Should have measurements for 1 second at 100 Hz
        // The 3-point stencil skips boundary regions, so expect fewer measurements
        assert!(
            measurements.len() >= 40 && measurements.len() <= 110,
            "Expected 40-110 measurements, got {}",
            measurements.len()
        );

        // Stationary: gyro should be ~0, accel should be ~-gravity in body frame
        for m in &measurements {
            assert!(m.gyro.norm() < 1e-6, "Gyro should be zero for stationary");
            // Accelerometer measures -gravity when stationary
            // With gravity = [0, 0, -9.81], and identity rotation,
            // accel = 0 - (-9.81 in z) = [0, 0, 9.81]
            assert!((m.accel.z - 9.81).abs() < 0.1, "Accel Z should be ~9.81");
        }
    }

    #[test]
    fn test_generate_from_trajectory() {
        let traj = CircularTrajectory::new(1.0);
        let poses = traj.generate(20, 42);
        let timestamped = add_timestamps_to_poses(poses, 2.0);

        let sim = ImuSimulator::new(ImuNoiseParams::consumer_grade(), 200.0);
        let measurements = sim.generate_from_trajectory(&timestamped, 123);

        // Should have measurements spanning the trajectory
        assert!(!measurements.is_empty());
        assert!(measurements.len() > 100); // At least some measurements

        // Check timestamps are monotonic
        for i in 1..measurements.len() {
            assert!(measurements[i].timestamp > measurements[i - 1].timestamp);
        }
    }

    #[test]
    fn test_add_timestamps() {
        let poses = vec![
            SE3::<f64>::identity(),
            SE3::<f64>::identity(),
            SE3::<f64>::identity(),
        ];
        let timestamped = add_timestamps_to_poses(poses, 2.0);

        assert_eq!(timestamped.len(), 3);
        assert_eq!(timestamped[0].0, 0.0);
        assert_eq!(timestamped[1].0, 1.0);
        assert_eq!(timestamped[2].0, 2.0);
    }

    #[test]
    fn test_noise_params_presets() {
        let zero = ImuNoiseParams::zero();
        assert_eq!(zero.gyro_noise_density, 0.0);

        let consumer = ImuNoiseParams::consumer_grade();
        assert!(consumer.gyro_noise_density > 0.0);
    }

    #[test]
    fn test_continuous_trajectory_stationary() {
        // Stationary trajectory (radius 0)
        struct StationaryTrajectory;

        impl ContinuousTrajectory for StationaryTrajectory {
            fn pose(&self, _t: f64) -> SE3<f64> {
                SE3::identity()
            }
            fn linear_velocity(&self, _t: f64) -> Vector3<f64> {
                Vector3::zeros()
            }
            fn angular_velocity(&self, _t: f64) -> Option<Vector3<f64>> {
                Some(Vector3::zeros())
            }
        }

        let sim = ImuSimulator::ideal(100.0);
        let measurements = sim.generate_from_continuous_trajectory(&StationaryTrajectory, 1.0, 42);

        assert!(!measurements.is_empty());

        // Stationary: gyro should be 0, accel should be -gravity in body frame = [0, 0, 9.81]
        for m in &measurements {
            assert!(
                m.gyro.norm() < 1e-10,
                "Gyro should be zero, got {:?}",
                m.gyro
            );
            assert!(
                (m.accel.z - 9.81).abs() < 0.01,
                "Accel Z should be ~9.81, got {}",
                m.accel.z
            );
            assert!(m.accel.x.abs() < 1e-10, "Accel X should be ~0");
            assert!(m.accel.y.abs() < 1e-10, "Accel Y should be ~0");
        }
    }

    #[test]
    fn test_continuous_trajectory_circular() {
        let radius = 2.0;
        let duration = 10.0;
        let traj = ContinuousCircularTrajectory::fixed_orientation(radius, duration);

        let sim = ImuSimulator::ideal(200.0);
        let measurements = sim.generate_from_continuous_trajectory(&traj, duration, 42);

        assert!(!measurements.is_empty());
        assert!(measurements.len() > 1000); // 10s at 200Hz

        // For a horizontal circle with fixed orientation:
        // - Angular velocity should be 0 (no rotation)
        // - Centripetal acceleration = ω²R = (2π/T)² * R, pointing toward center
        let omega = 2.0 * std::f64::consts::PI / duration;
        let expected_centripetal = omega * omega * radius;

        // Check a sample in the middle
        let mid_idx = measurements.len() / 2;
        let m = &measurements[mid_idx];

        // Gyro should be ~0 (fixed orientation)
        assert!(m.gyro.norm() < 0.01, "Gyro should be ~0, got {:?}", m.gyro);

        // Acceleration should be centripetal (in XY plane) + gravity contribution
        let accel_xy = (m.accel.x.powi(2) + m.accel.y.powi(2)).sqrt();
        assert!(
            (accel_xy - expected_centripetal).abs() < 0.5,
            "Centripetal accel should be ~{:.3}, got {:.3}",
            expected_centripetal,
            accel_xy
        );
    }

    #[test]
    fn test_continuous_trajectory_constant_velocity() {
        // Constant velocity trajectory (no acceleration except gravity)
        struct ConstantVelocityTrajectory {
            velocity: Vector3<f64>,
            duration: f64,
        }

        impl ContinuousTrajectory for ConstantVelocityTrajectory {
            fn pose(&self, t: f64) -> SE3<f64> {
                let pos = self.velocity * t * self.duration;
                SE3::from_rotation_translation(
                    crate::math::SO3::identity(),
                    odysseus_solver::math3d::Vec3::new(pos.x, pos.y, pos.z),
                )
            }
            fn linear_velocity(&self, _t: f64) -> Vector3<f64> {
                self.velocity
            }
            fn angular_velocity(&self, _t: f64) -> Option<Vector3<f64>> {
                Some(Vector3::zeros())
            }
        }

        let traj = ConstantVelocityTrajectory {
            velocity: Vector3::new(1.0, 0.0, 0.0),
            duration: 5.0,
        };

        let sim = ImuSimulator::ideal(100.0);
        let measurements = sim.generate_from_continuous_trajectory(&traj, 5.0, 42);

        // With constant velocity: dv/dt = 0
        // So accel should only be gravity: [0, 0, 9.81] (since orientation is identity)
        for m in &measurements {
            assert!(m.gyro.norm() < 1e-10, "Gyro should be 0");
            assert!(
                m.accel.x.abs() < 0.01,
                "Accel X should be ~0, got {}",
                m.accel.x
            );
            assert!(m.accel.y.abs() < 0.01, "Accel Y should be ~0");
            assert!(
                (m.accel.z - 9.81).abs() < 0.01,
                "Accel Z should be ~9.81, got {}",
                m.accel.z
            );
        }
    }

    #[test]
    fn test_continuous_trajectory_timestamps_monotonic() {
        let traj = ContinuousCircularTrajectory::new(1.0, 5.0);
        let sim = ImuSimulator::ideal(200.0);
        let measurements = sim.generate_from_continuous_trajectory(&traj, 5.0, 42);

        for i in 1..measurements.len() {
            assert!(
                measurements[i].timestamp > measurements[i - 1].timestamp,
                "Timestamps should be monotonically increasing"
            );
        }
    }
}
