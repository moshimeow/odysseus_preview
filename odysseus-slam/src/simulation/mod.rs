//! Simulation and synthetic data generation

use crate::camera::StereoCamera;
use crate::geometry::{StereoObservation, Point3D};
use crate::math::SE3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use odysseus_solver::math3d::Vec3;

// Re-export trajectory generators
pub use crate::trajectory::{TrajectoryGenerator, CircularTrajectory, BrownianTrajectory};

/// Generate random 3D points within a bounding box
///
/// # Arguments
/// * `n_points` - Number of points to generate
/// * `bounds` - Bounding box as [min_x, max_x, min_y, max_y, min_z, max_z]
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// * Vector of 3D points
pub fn generate_random_points(
    n_points: usize,
    bounds: [f64; 6],
    seed: u64,
) -> Vec<Point3D<f64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let [min_x, max_x, min_y, max_y, min_z, max_z] = bounds;

    (0..n_points)
        .map(|_| {
            Vec3::new(
                rng.gen_range(min_x..max_x),
                rng.gen_range(min_y..max_y),
                rng.gen_range(min_z..max_z),
            )
        })
        .collect()
}

// Trajectory generation moved to trajectory module - see trajectory::CircularTrajectory and trajectory::BrownianTrajectory

/// Generate stereo observations of 3D points from camera poses
///
/// Only creates observations for points that are:
/// 1. In front of both left and right cameras (z > 0)
/// 2. Project into image bounds for both cameras
///
/// # Arguments
/// * `points` - 3D points in world coordinates (meters)
/// * `poses` - Camera poses (world_T_camera_left)
/// * `stereo_camera` - Stereo camera with baseline
/// * `image_width` - Image width for bounds checking (pixels)
/// * `image_height` - Image height for bounds checking (pixels)
///
/// # Returns
/// * Vector of stereo observations
pub fn generate_stereo_observations(
    points: &[Point3D<f64>],
    poses: &[SE3<f64>],
    stereo_camera: &StereoCamera<f64>,
    image_width: f64,
    image_height: f64,
) -> Vec<StereoObservation> {
    let mut observations = Vec::new();

    for (camera_id, pose) in poses.iter().enumerate() {
        // Get camera_T_world (inverse of world_T_camera)
        let left_t_world = pose.inverse();

        for (point_id, point_world) in points.iter().enumerate() {
            // Transform point to left camera frame
            let point_left = left_t_world.transform_point(*point_world);

            // Check if point is in front of left camera (visibility check)
            if point_left.z <= 0.1 {
                continue;
            }


            // Project to stereo pair
            let (left_u, left_v, right_u, right_v) = stereo_camera.project_stereo(point_left);

            // Check if BOTH projections are in image bounds
            let left_in_bounds = left_u >= 0.0 && left_u < image_width
                              && left_v >= 0.0 && left_v < image_height;
            let right_in_bounds = right_u >= 0.0 && right_u < image_width
                               && right_v >= 0.0 && right_v < image_height;

            // Also check that point is in front of right camera
            // let point_right_z = point_left.z; // Z doesn't change for horizontal baseline
            // let right_visible = point_right_z > 0.0;

            if left_in_bounds && right_in_bounds { //  && right_visible {
                observations.push(StereoObservation::new(
                    point_id,
                    camera_id,
                    left_u,
                    left_v,
                    right_u,
                    right_v,
                ));
            }
        }
    }

    observations
}

/// Add Gaussian noise to stereo observations
///
/// # Arguments
/// * `observations` - Original stereo observations
/// * `sigma` - Standard deviation of noise (pixels)
/// * `seed` - Random seed
///
/// # Returns
/// * Noisy stereo observations
pub fn add_noise_to_stereo_observations(
    observations: &[StereoObservation],
    sigma: f64,
    seed: u64,
) -> Vec<StereoObservation> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    observations
        .iter()
        .map(|obs| {
            let noise_left_u = rng.gen_range(-sigma..sigma);
            let noise_left_v = rng.gen_range(-sigma..sigma);
            let noise_right_u = rng.gen_range(-sigma..sigma);
            let noise_right_v = rng.gen_range(-sigma..sigma);

            StereoObservation::new(
                obs.point_id,
                obs.camera_id,
                obs.left_u + noise_left_u,
                obs.left_v + noise_left_v,
                obs.right_u + noise_right_u,
                obs.right_v + noise_right_v,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_points() {
        let points = generate_random_points(10, [-5.0, 5.0, -5.0, 5.0, 0.0, 10.0], 42);

        assert_eq!(points.len(), 10);

        // Check all points are within bounds
        for p in &points {
            assert!(p.x >= -5.0 && p.x <= 5.0);
            assert!(p.y >= -5.0 && p.y <= 5.0);
            assert!(p.z >= 0.0 && p.z <= 10.0);
        }
    }

    #[test]
    fn test_generate_circular_trajectory() {
        let traj = CircularTrajectory::new(10.0);
        let poses = traj.generate(8, 42);

        assert_eq!(poses.len(), 8);

        // Check that cameras are roughly at the right distance
        for pose in &poses {
            let dist = (pose.translation.x * pose.translation.x
                      + pose.translation.y * pose.translation.y).sqrt();
            // Should be roughly at radius 10
            assert!(dist > 8.0 && dist < 12.0);
        }
    }

}
