//! Geometry primitives

use crate::camera::StereoCamera;
use crate::math::SE3;
use odysseus_solver::math3d::Vec3;

/// 3D point (just a type alias for Vec3)
pub type Point3D<T> = Vec3<T>;

/// A stereo observation of a 3D point
///
/// Links a 3D point to its 2D projections in both left and right stereo cameras
#[derive(Debug, Clone, Copy)]
pub struct StereoObservation {
    /// Index of the 3D point being observed
    pub point_id: usize,
    /// Index of the camera/pose from which the observation was made
    pub camera_id: usize,
    /// Left camera pixel coordinates [u, v]
    pub left_u: f64,
    pub left_v: f64,
    /// Right camera pixel coordinates [u, v]
    pub right_u: f64,
    pub right_v: f64,
}

impl StereoObservation {
    /// Create a new stereo observation
    pub fn new(
        point_id: usize,
        camera_id: usize,
        left_u: f64,
        left_v: f64,
        right_u: f64,
        right_v: f64,
    ) -> Self {
        Self {
            point_id,
            camera_id,
            left_u,
            left_v,
            right_u,
            right_v,
        }
    }

    /// Triangulate this observation into a 3D point
    ///
    /// Computes the 3D point in world frame from the stereo pixel coordinates.
    ///
    /// # Arguments
    /// * `camera` - Stereo camera with intrinsics and baseline
    /// * `pose` - Camera pose (transform from world to camera frame)
    ///
    /// # Returns
    /// * 3D point in world frame, or None if triangulation fails (e.g., negative disparity)
    ///
    /// # Algorithm
    /// Uses disparity to compute depth: depth = baseline * fx / disparity,
    /// then unprojects to 3D and transforms to world frame.
    pub fn triangulate(
        &self,
        camera: &StereoCamera<f64>,
        pose: &SE3<f64>,
    ) -> Option<Point3D<f64>> {
        // Compute disparity (horizontal pixel difference)
        let disparity = self.left_u - self.right_u;

        // Disparity must be positive (right image should be shifted left)
        if disparity <= 0.0 {
            return None;
        }

        // Depth from disparity: Z = baseline * fx / disparity
        let depth = camera.baseline * camera.left.fx / disparity;

        // Check for reasonable depth
        if depth <= 0.0 || !depth.is_finite() {
            return None;
        }

        // Unproject to 3D in left camera frame
        let point_in_camera = camera.left.unproject(self.left_u, self.left_v, depth);

        // Transform from camera frame to world frame
        // pose transforms world → camera, so pose.inverse() transforms camera → world
        let world_point = pose.transform_point(point_in_camera);

        Some(world_point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_stereo_observation_creation() {
        let obs = StereoObservation::new(0, 1, 320.5, 240.7, 280.3, 240.7);
        assert_eq!(obs.point_id, 0);
        assert_eq!(obs.camera_id, 1);
        assert_abs_diff_eq!(obs.left_u, 320.5);
        assert_abs_diff_eq!(obs.left_v, 240.7);
        assert_abs_diff_eq!(obs.right_u, 280.3);
        assert_abs_diff_eq!(obs.right_v, 240.7);
    }

    #[test]
    fn test_point3d_alias() {
        let p: Point3D<f64> = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn test_triangulate() {
        // Create a simple stereo camera
        let camera = StereoCamera::simple(500.0, 640.0, 480.0, 0.1);

        // Camera at origin looking down +Z
        let pose = SE3::<f64>::identity();

        // Create a point at (0, 0, 2.0) in world frame
        // Should project to center of left image: (320, 240)
        // In right image, shifted left by: disparity = baseline * fx / depth = 0.1 * 500 / 2.0 = 25 pixels
        // So right image: (320 - 25, 240) = (295, 240)

        let obs = StereoObservation::new(0, 0, 320.0, 240.0, 295.0, 240.0);
        let result = obs.triangulate(&camera, &pose);

        assert!(result.is_some());
        let point = result.unwrap();

        // Should recover the original point (0, 0, 2.0)
        assert_abs_diff_eq!(point.x, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(point.y, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(point.z, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_triangulate_fails_on_negative_disparity() {
        let camera = StereoCamera::simple(500.0, 640.0, 480.0, 0.1);
        let pose = SE3::<f64>::identity();

        // Invalid: right_u > left_u (negative disparity)
        let obs = StereoObservation::new(0, 0, 295.0, 240.0, 320.0, 240.0);
        assert!(obs.triangulate(&camera, &pose).is_none());
    }

    #[test]
    fn test_triangulate_fails_on_zero_disparity() {
        let camera = StereoCamera::simple(500.0, 640.0, 480.0, 0.1);
        let pose = SE3::<f64>::identity();

        // Invalid: same pixel in both images (infinite depth)
        let obs = StereoObservation::new(0, 0, 320.0, 240.0, 320.0, 240.0);
        assert!(obs.triangulate(&camera, &pose).is_none());
    }
}
