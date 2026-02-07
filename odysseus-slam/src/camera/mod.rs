//! Camera models

use odysseus_solver::math3d::Vec3;
use odysseus_solver::Real;

/// Kannala-Brandt fisheye camera model (KB4)
///
/// A fisheye projection model that handles wide-angle lenses.
/// Uses the equidistant projection with polynomial distortion:
///   θ_d = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
/// where θ = atan2(r, z) is the angle from the optical axis.
///
/// This model is used in ORB-SLAM3 and handles up to 180° FOV.
#[derive(Debug, Clone, Copy)]
pub struct KannalaBrandtCamera<T> {
    /// Focal length in x direction (pixels)
    pub fx: T,
    /// Focal length in y direction (pixels)
    pub fy: T,
    /// Principal point x coordinate (pixels)
    pub cx: T,
    /// Principal point y coordinate (pixels)
    pub cy: T,
    /// Distortion coefficient k1
    pub k1: T,
    /// Distortion coefficient k2
    pub k2: T,
    /// Distortion coefficient k3
    pub k3: T,
    /// Distortion coefficient k4
    pub k4: T,
}

impl<T: Real> KannalaBrandtCamera<T>
where
    T::Scalar: PartialOrd<f64>,
{
    /// Create a new Kannala-Brandt camera
    pub fn new(fx: T, fy: T, cx: T, cy: T, k1: T, k2: T, k3: T, k4: T) -> Self {
        Self { fx, fy, cx, cy, k1, k2, k3, k4 }
    }

    /// Create a camera with no distortion (equivalent to equidistant projection)
    pub fn no_distortion(fx: T, fy: T, cx: T, cy: T) -> Self {
        Self::new(fx, fy, cx, cy, T::zero(), T::zero(), T::zero(), T::zero())
    }

    /// Apply the distortion polynomial: θ_d = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
    fn distort_theta(&self, theta: T) -> T {
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        theta + self.k1 * theta3 + self.k2 * theta5 + self.k3 * theta7 + self.k4 * theta9
    }

    /// Derivative of distortion polynomial: d(θ_d)/d(θ) = 1 + 3*k1*θ² + 5*k2*θ⁴ + 7*k3*θ⁶ + 9*k4*θ⁸
    fn distort_theta_derivative(&self, theta: T) -> T {
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta6 * theta2;

        T::one()
            + T::from_literal(3.0) * self.k1 * theta2
            + T::from_literal(5.0) * self.k2 * theta4
            + T::from_literal(7.0) * self.k3 * theta6
            + T::from_literal(9.0) * self.k4 * theta8
    }

    /// Project a 3D point in camera coordinates to 2D image coordinates
    ///
    /// # Arguments
    /// * `point_cam` - 3D point in camera frame [X, Y, Z]
    ///
    /// # Returns
    /// * 2D pixel coordinates [u, v]
    pub fn project(&self, point_cam: Vec3<T>) -> (T, T) {
        let x = point_cam.x;
        let y = point_cam.y;
        let z = point_cam.z;

        // Radius in XY plane
        let r_sq = x * x + y * y;
        let r = r_sq.sqrt();

        // Handle point on or very close to optical axis
        // Use scalar() for branching only
        if r.scalar() < 1e-10 {
            // Point on optical axis projects to principal point
            return (self.cx, self.cy);
        }

        // Angle from optical axis: theta = atan2(r, z)
        // Use acos(z/norm) which gives the same result for r >= 0
        let norm = (r_sq + z * z).sqrt();
        let cos_theta = z / norm;
        let theta = cos_theta.acos();

        // Apply distortion to get θ_d
        let theta_d = self.distort_theta(theta);

        // Compute distorted normalized coordinates
        // Direction is preserved, magnitude is θ_d
        let scale = theta_d / r;
        let x_d = x * scale;
        let y_d = y * scale;

        // Apply intrinsics
        let u = self.fx * x_d + self.cx;
        let v = self.fy * y_d + self.cy;

        (u, v)
    }

    /// Unproject a 2D pixel to a 3D ray direction in camera coordinates
    ///
    /// Uses Newton-Raphson iteration to invert the distortion polynomial.
    ///
    /// # Arguments
    /// * `u` - Pixel x coordinate
    /// * `v` - Pixel y coordinate
    ///
    /// # Returns
    /// * Unit vector representing the ray direction
    pub fn unproject(&self, u: T, v: T) -> Vec3<T> {
        // Remove intrinsics to get distorted normalized coordinates
        let x_d = (u - self.cx) / self.fx;
        let y_d = (v - self.cy) / self.fy;

        // θ_d is the radius in distorted space
        let theta_d = (x_d * x_d + y_d * y_d).sqrt();

        // Handle point at principal point
        if theta_d.scalar() < 1e-10 {
            return Vec3::new(T::zero(), T::zero(), T::one());
        }

        // Solve for θ using Newton-Raphson
        // We need to find θ such that distort_theta(θ) = θ_d
        let mut theta = theta_d; // Initial guess

        for _ in 0..10 {
            let f = self.distort_theta(theta) - theta_d;
            let df = self.distort_theta_derivative(theta);
            let delta = f / df;
            theta = theta - delta;

            if delta.abs().scalar() < 1e-12 {
                break;
            }
        }

        // Compute the 3D ray direction
        // The ray lies in the plane containing the optical axis and (x_d, y_d)
        // with angle θ from the optical axis
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        // Direction in distorted space (normalized)
        let inv_theta_d = T::one() / theta_d;
        let dir_x = x_d * inv_theta_d;
        let dir_y = y_d * inv_theta_d;

        // Scale by sin(θ) for XY, use cos(θ) for Z
        Vec3::new(dir_x * sin_theta, dir_y * sin_theta, cos_theta)
    }

    /// Unproject and scale by depth
    ///
    /// # Arguments
    /// * `u` - Pixel x coordinate
    /// * `v` - Pixel y coordinate
    /// * `depth` - Depth value (distance along Z axis, not along ray)
    ///
    /// # Returns
    /// * 3D point at the given Z depth
    pub fn unproject_with_depth(&self, u: T, v: T, depth: T) -> Vec3<T> {
        let ray = self.unproject(u, v);
        // Scale ray so that z = depth
        let scale = depth / ray.z;
        Vec3::new(ray.x * scale, ray.y * scale, depth)
    }
}

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

impl PinholeCamera<f64> {
    /// Convert to a camera with Jet values for automatic differentiation.
    ///
    /// All parameters become constant Jets (derivatives are zero).
    /// This is useful when the camera intrinsics are fixed during optimization.
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
    ///
    /// All parameters become constant Jets (derivatives are zero).
    /// This is useful when the camera intrinsics are fixed during optimization.
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
        let expected_u = 500.0 * (1.0/2.0) + 320.0;
        let expected_v = 500.0 * (0.5/2.0) + 240.0;
        assert_abs_diff_eq!(u, expected_u, epsilon = 1e-10);
        assert_abs_diff_eq!(v, expected_v, epsilon = 1e-10);
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
    fn test_to_jet() {
        use odysseus_solver::Jet;

        // Create a regular f64 camera
        let camera_f64: PinholeCamera<f64> = PinholeCamera::new(500.0, 600.0, 320.0, 240.0);

        // Convert to Jet camera
        let camera_jet: PinholeCamera<Jet<f64, 4>> = camera_f64.to_jet();

        // Values should be preserved
        assert_abs_diff_eq!(camera_jet.fx.value, 500.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera_jet.fy.value, 600.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera_jet.cx.value, 320.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera_jet.cy.value, 240.0, epsilon = 1e-10);

        // Derivatives should all be zero (constant)
        assert!(camera_jet.fx.derivs.iter().all(|&d| d == 0.0));
        assert!(camera_jet.fy.derivs.iter().all(|&d| d == 0.0));
        assert!(camera_jet.cx.derivs.iter().all(|&d| d == 0.0));
        assert!(camera_jet.cy.derivs.iter().all(|&d| d == 0.0));
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

    // Kannala-Brandt camera tests

    #[test]
    fn test_kb_project_center() {
        // Point on optical axis should project to principal point
        let camera = KannalaBrandtCamera::new(500.0, 500.0, 320.0, 240.0, 0.1, -0.05, 0.01, -0.005);
        let point = Vec3::new(0.0, 0.0, 1.0);
        let (u, v) = camera.project(point);

        assert_abs_diff_eq!(u, 320.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v, 240.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kb_no_distortion_matches_equidistant() {
        // With no distortion, KB model reduces to equidistant projection
        let camera = KannalaBrandtCamera::no_distortion(500.0, 500.0, 320.0, 240.0);
        let point = Vec3::new(1.0, 0.0, 1.0);

        let (u, _v) = camera.project(point);

        // For equidistant: θ = atan2(1, 1) = π/4
        // x_d = θ * (x/r) = π/4 * 1 = π/4
        // u = 500 * π/4 + 320
        let expected_u = 500.0 * std::f64::consts::FRAC_PI_4 + 320.0;
        assert_abs_diff_eq!(u, expected_u, epsilon = 1e-10);
    }

    #[test]
    fn test_kb_unproject_project_roundtrip() {
        // Test with various distortion coefficients
        let camera = KannalaBrandtCamera::new(500.0, 500.0, 320.0, 240.0, 0.1, -0.05, 0.01, -0.005);

        // Test multiple points at different angles
        let test_points = [
            Vec3::new(0.5, 0.3, 2.0),
            Vec3::new(-1.0, 0.5, 1.5),
            Vec3::new(0.2, -0.8, 3.0),
            Vec3::new(1.0, 1.0, 1.0), // 45 degree angle
        ];

        for original in test_points {
            let (u, v) = camera.project(original);
            let reconstructed = camera.unproject_with_depth(u, v, original.z);

            assert_abs_diff_eq!(reconstructed.x, original.x, epsilon = 1e-8);
            assert_abs_diff_eq!(reconstructed.y, original.y, epsilon = 1e-8);
            assert_abs_diff_eq!(reconstructed.z, original.z, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_kb_unproject_returns_unit_vector() {
        let camera = KannalaBrandtCamera::new(500.0, 500.0, 320.0, 240.0, 0.1, -0.05, 0.01, -0.005);

        let ray = camera.unproject(400.0, 300.0);
        let norm = (ray.x * ray.x + ray.y * ray.y + ray.z * ray.z).sqrt();

        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kb_with_autodiff() {
        use odysseus_solver::Jet;

        type Jet3 = Jet<f64, 3>;

        let camera = KannalaBrandtCamera::new(
            Jet3::constant(500.0),
            Jet3::constant(500.0),
            Jet3::constant(320.0),
            Jet3::constant(240.0),
            Jet3::constant(0.1),
            Jet3::constant(-0.05),
            Jet3::constant(0.01),
            Jet3::constant(-0.005),
        );

        let point = Vec3::new(
            Jet3::variable(1.0, 0),
            Jet3::variable(0.5, 1),
            Jet3::variable(2.0, 2),
        );

        let (u, v) = camera.project(point);

        // Check that we have non-zero derivatives
        assert!(u.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(v.derivs.iter().any(|&d| d.abs() > 1e-10));

        // u should depend on x and z, but not y
        assert!(u.derivs[0].abs() > 1e-10); // du/dx
        assert!(u.derivs[2].abs() > 1e-10); // du/dz

        // v should depend on y and z, but not x
        assert!(v.derivs[1].abs() > 1e-10); // dv/dy
        assert!(v.derivs[2].abs() > 1e-10); // dv/dz
    }

    #[test]
    fn test_kb_wide_angle() {
        // Test at wide angles (fisheye behavior)
        let camera = KannalaBrandtCamera::no_distortion(500.0, 500.0, 320.0, 240.0);

        // Point at 60 degrees from optical axis
        let theta = std::f64::consts::FRAC_PI_3; // 60 degrees
        let point = Vec3::new(theta.sin(), 0.0, theta.cos());

        let (u, v) = camera.project(point);
        let ray = camera.unproject(u, v);

        // Verify the angle is preserved
        let reconstructed_theta = ray.z.acos();
        assert_abs_diff_eq!(reconstructed_theta, theta, epsilon = 1e-10);
    }
}
