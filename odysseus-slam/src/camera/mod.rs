//! Camera models

use odysseus_solver::math3d::Vec3;
use odysseus_solver::Real;

/// Pinhole camera model with intrinsic parameters
///
/// Represents a simple pinhole camera with focal lengths and principal point.
/// Uses the standard pinhole projection model:
///   u = fx * X/Z + cx
///   v = fy * Y/Z + cy
#[derive(Debug, Clone, Copy)]
pub struct PinholeCamera<T> {
    /// Focal length in x direction (pixels)
    pub fx: T,
    /// Focal length in y direction (pixels)
    pub fy: T,
    /// Principal point x coordinate (pixels)
    pub cx: T,
    /// Principal point y coordinate (pixels)
    pub cy: T,
}

impl<T: Real> PinholeCamera<T> {
    /// Create a new pinhole camera
    pub fn new(fx: T, fy: T, cx: T, cy: T) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Project a 3D point in camera coordinates to 2D image coordinates
    ///
    /// # Arguments
    /// * `point_cam` - 3D point in camera frame [X, Y, Z]
    ///
    /// # Returns
    /// * 2D pixel coordinates [u, v]
    ///
    /// # Note
    /// The point must be in front of the camera (Z > 0) for a valid projection.
    pub fn project(&self, point_cam: Vec3<T>) -> (T, T) {
        // Perspective division
        let inv_z = T::one() / point_cam.z;
        let x_normalized = point_cam.x * inv_z;
        let y_normalized = point_cam.y * inv_z;

        // Apply intrinsics
        let u = self.fx * x_normalized + self.cx;
        let v = self.fy * y_normalized + self.cy;

        (u, v)
    }

    /// Unproject a 2D pixel to a 3D ray in camera coordinates
    ///
    /// # Arguments
    /// * `u` - Pixel x coordinate
    /// * `v` - Pixel y coordinate
    /// * `depth` - Depth value (Z coordinate)
    ///
    /// # Returns
    /// * 3D point at the given depth
    pub fn unproject(&self, u: T, v: T, depth: T) -> Vec3<T> {
        let x = (u - self.cx) * depth / self.fx;
        let y = (v - self.cy) * depth / self.fy;
        Vec3::new(x, y, depth)
    }

    /// Create a simple camera with square pixels
    pub fn simple(focal_length: T, image_width: T, image_height: T) -> Self {
        let cx = image_width * T::from_literal(0.5);
        let cy = image_height * T::from_literal(0.5);
        Self::new(focal_length, focal_length, cx, cy)
    }
}

/// Stereo camera pair with a horizontal baseline
///
/// Standard stereo setup:
/// - Left camera at origin
/// - Right camera translated along +X axis by baseline distance
/// - Both cameras looking in +Z direction
#[derive(Debug, Clone, Copy)]
pub struct StereoCamera<T> {
    /// Left camera intrinsics
    pub left: PinholeCamera<T>,
    /// Right camera intrinsics (usually same as left)
    pub right: PinholeCamera<T>,
    /// Baseline distance (meters) - right camera is at [baseline, 0, 0] relative to left
    pub baseline: T,
}

impl<T: Real> StereoCamera<T> {
    /// Create a stereo camera with identical left/right intrinsics
    pub fn new(camera: PinholeCamera<T>, baseline: T) -> Self {
        Self {
            left: camera,
            right: camera,
            baseline,
        }
    }

    /// Project a 3D point (in left camera frame) to both left and right images
    ///
    /// # Arguments
    /// * `point_left` - 3D point in left camera coordinate frame
    ///
    /// # Returns
    /// * `(u_left, v_left, u_right, v_right)` - Pixel coordinates in both images
    pub fn project_stereo(&self, point_left: Vec3<T>) -> (T, T, T, T) {
        // Project to left camera (trivial)
        let (u_left, v_left) = self.left.project(point_left);

        // Transform point to right camera frame
        // Right camera is at [baseline, 0, 0] in left camera frame
        // So point in right camera frame is point_left - [baseline, 0, 0]
        let point_right = Vec3::new(
            point_left.x - self.baseline,
            point_left.y,
            point_left.z,
        );

        // Project to right camera
        let (u_right, v_right) = self.right.project(point_right);

        (u_left, v_left, u_right, v_right)
    }

    /// Create a simple stereo camera with square pixels and identical intrinsics
    pub fn simple(focal_length: T, image_width: T, image_height: T, baseline: T) -> Self {
        let camera = PinholeCamera::simple(focal_length, image_width, image_height);
        Self::new(camera, baseline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_project_center() {
        // Point on optical axis should project to principal point
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let point = Vec3::new(0.0, 0.0, 1.0);
        let (u, v) = camera.project(point);

        assert_abs_diff_eq!(u, 320.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v, 240.0, epsilon = 1e-10);
    }

    #[test]
    fn test_project_offset() {
        // Point offset from optical axis
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let point = Vec3::new(1.0, 0.5, 2.0); // X=1, Y=0.5, Z=2

        let (u, v) = camera.project(point);

        // u = 500 * (1/2) + 320 = 250 + 320 = 570
        // v = 500 * (0.5/2) + 240 = 125 + 240 = 365
        assert_abs_diff_eq!(u, 570.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v, 365.0, epsilon = 1e-10);
    }

    #[test]
    fn test_unproject_project_roundtrip() {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);

        // Project then unproject
        let original = Vec3::new(1.0, 2.0, 5.0);
        let (u, v) = camera.project(original);
        let reconstructed = camera.unproject(u, v, 5.0);

        assert_abs_diff_eq!(reconstructed.x, original.x, epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed.y, original.y, epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed.z, original.z, epsilon = 1e-10);
    }

    #[test]
    fn test_simple_camera() {
        let camera = PinholeCamera::simple(500.0, 640.0, 480.0);

        assert_abs_diff_eq!(camera.fx, 500.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera.fy, 500.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera.cx, 320.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera.cy, 240.0, epsilon = 1e-10);
    }

    #[test]
    fn test_with_autodiff() {
        use odysseus_solver::Jet;

        type Jet3 = Jet<f64, 3>;

        let camera = PinholeCamera::new(
            Jet3::constant(500.0),
            Jet3::constant(500.0),
            Jet3::constant(320.0),
            Jet3::constant(240.0),
        );

        let point = Vec3::new(
            Jet3::variable(1.0, 0),
            Jet3::variable(2.0, 1),
            Jet3::variable(5.0, 2),
        );

        let (u, v) = camera.project(point);

        // Check that we have derivatives
        assert!(u.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(v.derivs.iter().any(|&d| d.abs() > 1e-10));

        // Check derivative magnitudes are reasonable
        // du/dX should be fx/Z = 500/5 = 100
        assert_abs_diff_eq!(u.derivs[0], 100.0, epsilon = 1e-6);
        // dv/dY should be fy/Z = 500/5 = 100
        assert_abs_diff_eq!(v.derivs[1], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_different_focal_lengths() {
        // Test camera with different fx and fy
        let camera = PinholeCamera::new(600.0, 400.0, 320.0, 240.0);
        let point = Vec3::new(2.0, 3.0, 4.0);

        let (u, v) = camera.project(point);

        // u = 600 * (2/4) + 320 = 300 + 320 = 620
        // v = 400 * (3/4) + 240 = 300 + 240 = 540
        assert_abs_diff_eq!(u, 620.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v, 540.0, epsilon = 1e-10);
    }
}
