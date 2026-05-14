//! Stereo camera pair

use crate::camera::PinholeCamera;
use odysseus_solver::math3d::Vec3;
use odysseus_solver::Real;

/// Stereo camera pair with a horizontal baseline
///
/// Standard stereo setup:
/// - Left camera at origin
/// - Right camera translated along +X axis by baseline distance
/// - Both cameras looking in +Z direction
#[derive(Debug, Clone, Copy)]
pub struct StereoCamera<T> {
    pub left: PinholeCamera<T>,
    pub right: PinholeCamera<T>,
    /// Baseline distance (meters) - right camera is at [baseline, 0, 0] relative to left
    pub baseline: T,
}

impl<T: Real> StereoCamera<T> {
    pub fn new(camera: PinholeCamera<T>, baseline: T) -> Self {
        Self {
            left: camera,
            right: camera,
            baseline,
        }
    }

    /// Project a 3D point (in left camera frame) to both left and right images
    pub fn project_stereo(&self, point_left: Vec3<T>) -> (T, T, T, T) {
        let (u_left, v_left) = self.left.project(point_left);

        let point_right = Vec3::new(
            point_left.x - self.baseline,
            point_left.y,
            point_left.z,
        );

        let (u_right, v_right) = self.right.project(point_right);

        (u_left, v_left, u_right, v_right)
    }

    pub fn simple(focal_length: T, image_width: T, image_height: T, baseline: T) -> Self {
        let camera = PinholeCamera::simple(focal_length, image_width, image_height);
        Self::new(camera, baseline)
    }
}
