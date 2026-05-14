//! Kannala-Brandt fisheye camera model (KB4)

use crate::camera::CameraModel;
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
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub k1: T,
    pub k2: T,
    pub k3: T,
    pub k4: T,
}

impl<T: Real> KannalaBrandtCamera<T>
where
    T::Scalar: PartialOrd<f64>,
{
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

    /// Derivative of distortion polynomial
    fn distort_theta_derivative(&self, theta: T) -> T {
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta6 * theta2;

        T::one()
            + T::from_f64(3.0) * self.k1 * theta2
            + T::from_f64(5.0) * self.k2 * theta4
            + T::from_f64(7.0) * self.k3 * theta6
            + T::from_f64(9.0) * self.k4 * theta8
    }

    pub fn project(&self, point_cam: Vec3<T>) -> (T, T) {
        let x = point_cam.x;
        let y = point_cam.y;
        let z = point_cam.z;

        let r_sq = x * x + y * y;
        let r = r_sq.sqrt();

        if r.scalar() < 1e-10 {
            return (self.cx, self.cy);
        }

        let norm = (r_sq + z * z).sqrt();
        let cos_theta = z / norm;
        let theta = cos_theta.acos();

        let theta_d = self.distort_theta(theta);

        let scale = theta_d / r;
        let x_d = x * scale;
        let y_d = y * scale;

        let u = self.fx * x_d + self.cx;
        let v = self.fy * y_d + self.cy;

        (u, v)
    }

    /// Unproject a 2D pixel to a 3D ray direction (unit vector).
    /// Uses Newton-Raphson iteration to invert the distortion polynomial.
    pub fn unproject(&self, u: T, v: T) -> Vec3<T> {
        let x_d = (u - self.cx) / self.fx;
        let y_d = (v - self.cy) / self.fy;

        let theta_d = (x_d * x_d + y_d * y_d).sqrt();

        if theta_d.scalar() < 1e-10 {
            return Vec3::new(T::zero(), T::zero(), T::one());
        }

        let mut theta = theta_d;
        for _ in 0..10 {
            let f = self.distort_theta(theta) - theta_d;
            let df = self.distort_theta_derivative(theta);
            let delta = f / df;
            theta = theta - delta;

            if delta.abs().scalar() < 1e-12 {
                break;
            }
        }

        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let inv_theta_d = T::one() / theta_d;
        let dir_x = x_d * inv_theta_d;
        let dir_y = y_d * inv_theta_d;

        Vec3::new(dir_x * sin_theta, dir_y * sin_theta, cos_theta)
    }

    pub fn unproject_with_depth(&self, u: T, v: T, depth: T) -> Vec3<T> {
        let ray = self.unproject(u, v);
        let scale = depth / ray.z;
        Vec3::new(ray.x * scale, ray.y * scale, depth)
    }
}

impl<T: Real> CameraModel<T> for KannalaBrandtCamera<T>
where
    T::Scalar: PartialOrd<f64>,
{
    const NUM_INTRINSICS: usize = 8;

    fn project(&self, point_cam: Vec3<T>) -> (T, T) {
        KannalaBrandtCamera::project(self, point_cam)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_kb_project_center() {
        let camera = KannalaBrandtCamera::new(500.0, 500.0, 320.0, 240.0, 0.1, -0.05, 0.01, -0.005);
        let point = Vec3::new(0.0, 0.0, 1.0);
        let (u, v) = camera.project(point);

        assert_abs_diff_eq!(u, 320.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v, 240.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kb_no_distortion_matches_equidistant() {
        let camera = KannalaBrandtCamera::no_distortion(500.0, 500.0, 320.0, 240.0);
        let point = Vec3::new(1.0, 0.0, 1.0);
        let (u, _v) = camera.project(point);

        let expected_u = 500.0 * std::f64::consts::FRAC_PI_4 + 320.0;
        assert_abs_diff_eq!(u, expected_u, epsilon = 1e-10);
    }

    #[test]
    fn test_kb_unproject_project_roundtrip() {
        let camera = KannalaBrandtCamera::new(500.0, 500.0, 320.0, 240.0, 0.1, -0.05, 0.01, -0.005);

        let test_points = [
            Vec3::new(0.5, 0.3, 2.0),
            Vec3::new(-1.0, 0.5, 1.5),
            Vec3::new(0.2, -0.8, 3.0),
            Vec3::new(1.0, 1.0, 1.0),
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

        assert!(u.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(v.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(u.derivs[0].abs() > 1e-10);
        assert!(u.derivs[2].abs() > 1e-10);
        assert!(v.derivs[1].abs() > 1e-10);
        assert!(v.derivs[2].abs() > 1e-10);
    }

    #[test]
    fn test_kb_wide_angle() {
        let camera = KannalaBrandtCamera::no_distortion(500.0, 500.0, 320.0, 240.0);

        let theta = std::f64::consts::FRAC_PI_3;
        let point = Vec3::new(theta.sin(), 0.0, theta.cos());

        let (u, v) = camera.project(point);
        let ray = camera.unproject(u, v);

        let reconstructed_theta = ray.z.acos();
        assert_abs_diff_eq!(reconstructed_theta, theta, epsilon = 1e-10);
    }
}
