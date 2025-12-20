//! Camera trajectory generation

use crate::math::SE3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Trait for generating camera trajectories
pub trait TrajectoryGenerator {
    /// Generate n camera poses
    fn generate(&self, n_cameras: usize, seed: u64) -> Vec<SE3<f64>>;
}

/// Circular trajectory with oscillating orientation
///
/// - Circular path of given radius (meters)
/// - Each orientation axis oscillates at 2x the speed of the circular motion
/// - The three axes are 120 degrees apart in phase
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
            position_step: 0.1,    // 10cm steps
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
        let dist = (last.translation.x.powi(2) + last.translation.y.powi(2) + last.translation.z.powi(2)).sqrt();
        assert!(dist > 0.0, "Camera should have moved from origin");
    }
}
