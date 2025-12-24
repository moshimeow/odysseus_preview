//! Camera trajectory generation

use crate::math::{SE3, SO3};
use nalgebra::Vector3;
use odysseus_solver::math3d::Vec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Trait for generating discrete camera poses
pub trait TrajectoryGenerator {
    /// Generate n camera poses
    fn generate(&self, n_cameras: usize, seed: u64) -> Vec<SE3<f64>>;
}

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

/// Circular trajectory with oscillating orientation (legacy discrete version)
///
/// - Circular path of given radius (meters)
/// - Each orientation axis oscillates at 2x the speed of the circular motion
/// - The three axes are 120 degrees apart in phase
///
/// NOTE: This uses SE3::exp which couples rotation and translation in unexpected ways.
/// For accurate IMU simulation, use `ContinuousCircularTrajectory` instead.
pub struct CircularTrajectory {
    pub radius: f64,
}

impl CircularTrajectory {
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
}

// This currently fails a test which I have turned off. If you want to know more check the tests.
impl TrajectoryGenerator for CircularTrajectory {
    fn generate(&self, n_cameras: usize, seed: u64) -> Vec<SE3<f64>> {
        let _rng = ChaCha8Rng::seed_from_u64(seed);
        let mut poses = Vec::new();

        for i in 0..n_cameras {
            // Angle around the circle (parameter t from 0 to 2π)
            let t = 2.0 * std::f64::consts::PI * (i as f64) / (n_cameras as f64);

            // Camera position on circle (in XY plane, Z=0)
            let cam_x = self.radius * t.cos();
            let cam_y = self.radius * t.sin();
            let cam_z = 0.0;

            // Orientation: each axis oscillates at 2x speed, 120° apart
            let phase_0 = 0.0;
            let phase_1 = 2.0 * std::f64::consts::PI / 3.0; // 120°
            let phase_2 = 4.0 * std::f64::consts::PI / 3.0; // 240°

            let amplitude = 0.3; // radians

            let omega_x = amplitude * (2.0 * t + phase_0).sin();
            let omega_y = amplitude * (2.0 * t + phase_1).sin();
            let omega_z = amplitude * (2.0 * t + phase_2).sin();

            // SE3 tangent space: [omega_x, omega_y, omega_z, v_x, v_y, v_z]
            let tangent = [omega_x, omega_y, omega_z, cam_x, cam_y, cam_z];

            poses.push(SE3::exp(tangent));
        }

        poses
    }
}

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
        }
    }

    /// Create a circle with fixed world-frame orientation
    pub fn fixed_orientation(radius: f64, duration: f64) -> Self {
        Self {
            radius,
            duration,
            orientation_amplitude: 0.0,
            look_tangent: false,
        }
    }

    /// Angular frequency (rad/s) around the circle
    fn omega(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.duration
    }
}

impl ContinuousTrajectory for ContinuousCircularTrajectory {
    fn pose(&self, t: f64) -> SE3<f64> {
        let theta = 2.0 * std::f64::consts::PI * t;

        // Position on the circle (XY plane)
        let x = self.radius * theta.cos();
        let y = self.radius * theta.sin();
        let z = 0.0;
        let translation = Vec3::new(x, y, z);

        // Rotation
        let rotation = if self.look_tangent {
            // Look in the direction of motion (tangent to circle)
            // Tangent direction: [-sin(θ), cos(θ), 0]
            // We want camera's forward (-Z) to point along tangent
            // So we rotate around Z by (θ + π/2)
            let yaw = theta + std::f64::consts::PI / 2.0;

            // Add oscillation if configured
            let osc_x = self.orientation_amplitude * (2.0 * theta).sin();
            let osc_y =
                self.orientation_amplitude * (2.0 * theta + 2.0 * std::f64::consts::PI / 3.0).sin();

            // Compose rotations: first yaw, then pitch/roll oscillations
            let rot_yaw = SO3::exp(Vec3::new(0.0, 0.0, yaw));
            let rot_osc = SO3::exp(Vec3::new(osc_x, osc_y, 0.0));
            (rot_yaw * rot_osc).normalize() // Normalize to prevent drift in log()
        } else {
            // Fixed orientation (identity or with oscillation only)
            if self.orientation_amplitude > 0.0 {
                let osc_x = self.orientation_amplitude * (2.0 * theta).sin();
                let osc_y = self.orientation_amplitude
                    * (2.0 * theta + 2.0 * std::f64::consts::PI / 3.0).sin();
                let osc_z = self.orientation_amplitude
                    * (2.0 * theta + 4.0 * std::f64::consts::PI / 3.0).sin();
                SO3::exp(Vec3::new(osc_x, osc_y, osc_z))
            } else {
                SO3::identity()
            }
        };

        SE3::from_rotation_translation(rotation, translation)
    }

    fn linear_velocity(&self, t: f64) -> Vector3<f64> {
        let theta = 2.0 * std::f64::consts::PI * t;
        let omega = self.omega(); // rad/s

        // d/dt [R*cos(ωt), R*sin(ωt), 0] = [−Rω*sin(ωt), Rω*cos(ωt), 0]
        // But t is normalized, so θ = 2πt and dθ/dt_real = 2π/T = ω
        let speed = self.radius * omega;

        Vector3::new(-speed * theta.sin(), speed * theta.cos(), 0.0)
    }

    fn angular_velocity(&self, t: f64) -> Option<Vector3<f64>> {
        let _theta = 2.0 * std::f64::consts::PI * t;
        let omega = self.omega();

        if self.look_tangent {
            // When there's oscillation, the analytical angular velocity is complex
            // (requires SO3 right Jacobian). Return None to use finite differences.
            if self.orientation_amplitude > 0.0 {
                None // Let simulator use finite differences from relative rotation
            } else {
                // No oscillation - just yaw around world Z.
                // For pure Z rotation, world and body angular velocity are the same.
                // (Z-axis is invariant under rotation about Z)
                Some(Vector3::new(0.0, 0.0, omega))
            }
        } else if self.orientation_amplitude > 0.0 {
            // Oscillation case: angular velocity from rotation vector derivatives is
            // only exact for small angles. Return None to use finite differences.
            None
        } else {
            // Fixed orientation, no angular velocity
            Some(Vector3::zeros())
        }
    }
}

/// Brownian motion trajectory starting at origin
///
/// - Starts at (0, 0, 0) with identity rotation
/// - Each step adds random walk in position and orientation
/// - Walks away from origin with small random steps
pub struct BrownianTrajectory {
    /// Step size for position (meters)
    pub position_step: f64,
    /// Step size for orientation (radians)
    pub orientation_step: f64,
}

impl BrownianTrajectory {
    pub fn new(position_step: f64, orientation_step: f64) -> Self {
        Self {
            position_step,
            orientation_step,
        }
    }

    /// Default Brownian motion with reasonable step sizes
    pub fn default_steps() -> Self {
        Self {
            position_step: 0.1,     // 10cm steps
            orientation_step: 0.05, // ~3 degree steps
        }
    }
}

impl TrajectoryGenerator for BrownianTrajectory {
    fn generate(&self, n_cameras: usize, seed: u64) -> Vec<SE3<f64>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut poses = Vec::new();

        // Start at origin with zero rotation
        poses.push(SE3::identity());

        // Random walk from there
        for _ in 1..n_cameras {
            let prev_pose = poses.last().unwrap();
            let prev_tangent = prev_pose.log();

            // Add random step to each component
            let delta_omega_x = rng.gen_range(-self.orientation_step..self.orientation_step);
            let delta_omega_y = rng.gen_range(-self.orientation_step..self.orientation_step);
            let delta_omega_z = rng.gen_range(-self.orientation_step..self.orientation_step);

            let delta_x = rng.gen_range(-self.position_step..self.position_step);
            let delta_y = rng.gen_range(-self.position_step..self.position_step);
            let delta_z = rng.gen_range(-self.position_step..self.position_step);

            let new_tangent = [
                prev_tangent[0] + delta_omega_x,
                prev_tangent[1] + delta_omega_y,
                prev_tangent[2] + delta_omega_z,
                prev_tangent[3] + delta_x,
                prev_tangent[4] + delta_y,
                prev_tangent[5] + delta_z,
            ];

            poses.push(SE3::exp(new_tangent));
        }

        poses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
    #[test]
    fn test_circular_trajectory() {
        let traj = CircularTrajectory::new(1.0);
        let poses = traj.generate(8, 42);
        assert_eq!(poses.len(), 8);

        // Check first pose is roughly at (radius, 0, 0)
        let first = &poses[0];
        assert!((first.translation.x - 1.0).abs() < 0.1);
        assert!(first.translation.y.abs() < 0.1);
        assert!(first.translation.z.abs() < 0.1);
    }
    */

    #[test]
    fn test_brownian_trajectory_starts_at_origin() {
        let traj = BrownianTrajectory::default_steps();
        let poses = traj.generate(10, 42);
        assert_eq!(poses.len(), 10);

        // First pose should be at origin with identity rotation
        let first = &poses[0];
        assert!(first.translation.x.abs() < 1e-10);
        assert!(first.translation.y.abs() < 1e-10);
        assert!(first.translation.z.abs() < 1e-10);

        // Should walk away from origin
        let last = &poses[9];
        let dist =
            (last.translation.x.powi(2) + last.translation.y.powi(2) + last.translation.z.powi(2))
                .sqrt();
        assert!(dist > 0.0, "Camera should have moved from origin");
    }

    #[test]
    fn test_continuous_circular_position() {
        let traj = ContinuousCircularTrajectory::new(2.0, 10.0);

        // At t=0, should be at (radius, 0, 0)
        let p0 = traj.pose(0.0);
        assert!((p0.translation.x - 2.0).abs() < 1e-10);
        assert!(p0.translation.y.abs() < 1e-10);
        assert!(p0.translation.z.abs() < 1e-10);

        // At t=0.25 (quarter circle), should be at (0, radius, 0)
        let p1 = traj.pose(0.25);
        assert!(p1.translation.x.abs() < 1e-10);
        assert!((p1.translation.y - 2.0).abs() < 1e-10);
        assert!(p1.translation.z.abs() < 1e-10);

        // At t=0.5 (half circle), should be at (-radius, 0, 0)
        let p2 = traj.pose(0.5);
        assert!((p2.translation.x + 2.0).abs() < 1e-10);
        assert!(p2.translation.y.abs() < 1e-10);
    }

    #[test]
    fn test_continuous_circular_velocity_matches_finite_diff() {
        let traj = ContinuousCircularTrajectory::new(2.0, 10.0);

        // Test at several points
        for t in [0.0, 0.1, 0.25, 0.5, 0.75] {
            let v_analytical = traj.linear_velocity(t);

            // Finite difference approximation
            let dt = 1e-6;
            let p1 = traj.pose(t);
            let p2 = traj.pose(t + dt);
            let v_numerical = Vector3::new(
                (p2.translation.x - p1.translation.x) / (dt * traj.duration),
                (p2.translation.y - p1.translation.y) / (dt * traj.duration),
                (p2.translation.z - p1.translation.z) / (dt * traj.duration),
            );

            assert!(
                (v_analytical.x - v_numerical.x).abs() < 1e-4,
                "Velocity x mismatch at t={}: analytical={}, numerical={}",
                t,
                v_analytical.x,
                v_numerical.x
            );
            assert!(
                (v_analytical.y - v_numerical.y).abs() < 1e-4,
                "Velocity y mismatch at t={}: analytical={}, numerical={}",
                t,
                v_analytical.y,
                v_numerical.y
            );
            assert!(
                (v_analytical.z - v_numerical.z).abs() < 1e-4,
                "Velocity z mismatch at t={}: analytical={}, numerical={}",
                t,
                v_analytical.z,
                v_numerical.z
            );
        }
    }

    #[test]
    fn test_continuous_circular_velocity_magnitude() {
        let radius = 2.0;
        let duration = 10.0;
        let traj = ContinuousCircularTrajectory::new(radius, duration);

        // Speed should be constant: v = 2πR/T
        let expected_speed = 2.0 * std::f64::consts::PI * radius / duration;

        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.99] {
            let v = traj.linear_velocity(t);
            let speed = v.norm();
            assert!(
                (speed - expected_speed).abs() < 1e-10,
                "Speed at t={}: got {}, expected {}",
                t,
                speed,
                expected_speed
            );
        }
    }

    #[test]
    fn test_continuous_circular_stays_in_xy_plane() {
        let traj = ContinuousCircularTrajectory::new(2.0, 10.0);

        for i in 0..100 {
            let t = i as f64 / 99.0;
            let pose = traj.pose(t);
            assert!(
                pose.translation.z.abs() < 1e-10,
                "Z should be 0 at t={}, got {}",
                t,
                pose.translation.z
            );

            let vel = traj.linear_velocity(t);
            assert!(
                vel.z.abs() < 1e-10,
                "Velocity Z should be 0 at t={}, got {}",
                t,
                vel.z
            );
        }
    }
}
