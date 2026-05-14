//! Pinhole camera model

use crate::camera::CameraModel;
use odysseus_solver::math3d::Vec3;
use odysseus_solver::Real;
use std::ops::{Add, Mul};

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
    pub fn project<S: Real, R>(&self, point_cam: Vec3<S>) -> (R, R)
    where
        T: Mul<S, Output = R>,
        R: Add<T, Output = R>,
    {
        let inv_z = S::one() / point_cam.z;
        let x_normalized = point_cam.x * inv_z;
        let y_normalized = point_cam.y * inv_z;

        let u = self.fx * x_normalized + self.cx;
        let v = self.fy * y_normalized + self.cy;

        (u, v)
    }

    /// Unproject a 2D pixel to a 3D point at the given depth
    pub fn unproject(&self, u: T, v: T, depth: T) -> Vec3<T> {
        let x = (u - self.cx) * depth / self.fx;
        let y = (v - self.cy) * depth / self.fy;
        Vec3::new(x, y, depth)
    }

    /// Create a simple camera with square pixels
    pub fn simple(focal_length: T, image_width: T, image_height: T) -> Self {
        let cx = image_width * T::from_f64(0.5);
        let cy = image_height * T::from_f64(0.5);
        Self::new(focal_length, focal_length, cx, cy)
    }
}

impl PinholeCamera<f64> {
    /// Convert to a camera with Jet values for automatic differentiation.
    /// All parameters become constant Jets (derivatives are zero).
    pub fn to_constant<const N: usize>(&self) -> PinholeCamera<odysseus_solver::Jet<f64, N>> {
        use odysseus_solver::Jet;
        PinholeCamera::new(
            Jet::constant(self.fx),
            Jet::constant(self.fy),
            Jet::constant(self.cx),
            Jet::constant(self.cy),
        )
    }
}

impl PinholeCamera<f32> {
    /// Convert to a camera with Jet values for automatic differentiation.
    pub fn to_jet<const N: usize>(&self) -> PinholeCamera<odysseus_solver::Jet<f32, N>> {
        use odysseus_solver::Jet;
        PinholeCamera::new(
            Jet::constant(self.fx),
            Jet::constant(self.fy),
            Jet::constant(self.cx),
            Jet::constant(self.cy),
        )
    }
}

impl<T: Real> CameraModel<T> for PinholeCamera<T> {
    const NUM_INTRINSICS: usize = 4;

    fn project(&self, point_cam: Vec3<T>) -> (T, T) {
        PinholeCamera::project(self, point_cam)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_project_center() {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let point = Vec3::new(0.0, 0.0, 1.0);
        let (u, v) = camera.project(point);

        assert_abs_diff_eq!(u, 320.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v, 240.0, epsilon = 1e-10);
    }

    #[test]
    fn test_project_offset() {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let point = Vec3::new(1.0, 0.5, 2.0);

        let (u, v) = camera.project(point);

        let expected_u = 500.0 * (1.0 / 2.0) + 320.0;
        let expected_v = 500.0 * (0.5 / 2.0) + 240.0;
        assert_abs_diff_eq!(u, expected_u, epsilon = 1e-10);
        assert_abs_diff_eq!(v, expected_v, epsilon = 1e-10);
    }

    #[test]
    fn test_unproject_project_roundtrip() {
        let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
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

        assert!(u.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(v.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert_abs_diff_eq!(u.derivs[0], 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(v.derivs[1], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_to_jet() {
        use odysseus_solver::Jet;

        let camera_f64: PinholeCamera<f64> = PinholeCamera::new(500.0, 600.0, 320.0, 240.0);
        let camera_jet: PinholeCamera<Jet<f64, 4>> = camera_f64.to_constant();

        assert_abs_diff_eq!(camera_jet.fx.value, 500.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera_jet.fy.value, 600.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera_jet.cx.value, 320.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera_jet.cy.value, 240.0, epsilon = 1e-10);

        assert!(camera_jet.fx.derivs.iter().all(|&d| d == 0.0));
        assert!(camera_jet.fy.derivs.iter().all(|&d| d == 0.0));
        assert!(camera_jet.cx.derivs.iter().all(|&d| d == 0.0));
        assert!(camera_jet.cy.derivs.iter().all(|&d| d == 0.0));
    }

    #[test]
    fn test_different_focal_lengths() {
        let camera = PinholeCamera::new(600.0, 400.0, 320.0, 240.0);
        let point = Vec3::new(2.0, 3.0, 4.0);
        let (u, v) = camera.project(point);

        assert_abs_diff_eq!(u, 620.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v, 540.0, epsilon = 1e-10);
    }
}
