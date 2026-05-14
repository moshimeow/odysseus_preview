//! Blender Cycles "Fisheye Polynomial" camera model.
//!
//! Blender Cycles offers a fisheye sensor whose radial mapping is a 4th-degree
//! polynomial in the angle from the optical axis:
//!
//!   r_d(θ) = k0 + k1·θ + k2·θ² + k3·θ³ + k4·θ⁴
//!
//! where θ = angle from the optical axis (radians) and `r_d` plays the same
//! role as the equidistant fisheye radius in KB4 — i.e. the normalized image
//! plane radius before applying focal length and principal point.
//!
//! NOTE: the exact normalization Blender uses internally depends on
//! `fisheye_fov` and the sensor size. Verify against Blender source
//! (`intern/cycles/kernel/camera/projection.h`) if/when you wire up
//! cross-validation. Coefficients can be absorbed into `fx, fy, k0..k4`
//! to match whatever convention is in use.

use crate::camera::CameraModel;
use odysseus_solver::math3d::Vec3;
use odysseus_solver::Real;

/// Polynomial fisheye (Blender Cycles convention).
#[derive(Debug, Clone, Copy)]
pub struct BlenderPolyFisheyeCamera<T> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub k0: T,
    pub k1: T,
    pub k2: T,
    pub k3: T,
    pub k4: T,
}

impl<T: Real> BlenderPolyFisheyeCamera<T>
where
    T::Scalar: PartialOrd<f64>,
{
    pub fn new(fx: T, fy: T, cx: T, cy: T, k0: T, k1: T, k2: T, k3: T, k4: T) -> Self {
        Self { fx, fy, cx, cy, k0, k1, k2, k3, k4 }
    }

    /// Equidistant fisheye (k1 = 1, others zero), useful as a sanity reference.
    pub fn equidistant(fx: T, fy: T, cx: T, cy: T) -> Self {
        Self::new(
            fx, fy, cx, cy,
            T::zero(), T::one(), T::zero(), T::zero(), T::zero(),
        )
    }

    /// Apply the polynomial r_d(θ).
    fn poly(&self, theta: T) -> T {
        let t2 = theta * theta;
        let t3 = t2 * theta;
        let t4 = t2 * t2;
        self.k0 + self.k1 * theta + self.k2 * t2 + self.k3 * t3 + self.k4 * t4
    }

    /// Derivative of r_d(θ) w.r.t. θ.
    fn poly_derivative(&self, theta: T) -> T {
        let t2 = theta * theta;
        let t3 = t2 * theta;
        self.k1
            + T::from_literal(2.0) * self.k2 * theta
            + T::from_literal(3.0) * self.k3 * t2
            + T::from_literal(4.0) * self.k4 * t3
    }

    /// Project a 3D point in camera coordinates to 2D image coordinates.
    pub fn project(&self, point_cam: Vec3<T>) -> (T, T) {
        let x = point_cam.x;
        let y = point_cam.y;
        let z = point_cam.z;

        let r_sq = x * x + y * y;
        let r = r_sq.sqrt();

        if r.scalar() < 1e-10 {
            // On-axis point: r_d = poly(0) = k0. With the standard k0 = 0 this
            // collapses to the principal point.
            let r_d = self.k0;
            return (self.fx * r_d + self.cx, self.fy * r_d + self.cy);
        }

        let norm = (r_sq + z * z).sqrt();
        let theta = (z / norm).acos();
        let r_d = self.poly(theta);

        let scale = r_d / r;
        let x_d = x * scale;
        let y_d = y * scale;

        (self.fx * x_d + self.cx, self.fy * y_d + self.cy)
    }

    /// Unproject a 2D pixel to a 3D ray (unit vector).
    /// Inverts the polynomial via Newton-Raphson.
    pub fn unproject(&self, u: T, v: T) -> Vec3<T> {
        let x_d = (u - self.cx) / self.fx;
        let y_d = (v - self.cy) / self.fy;
        let r_d = (x_d * x_d + y_d * y_d).sqrt();

        if r_d.scalar() < 1e-10 {
            return Vec3::new(T::zero(), T::zero(), T::one());
        }

        // Initial guess: linear inversion using k1, falling back to r_d itself.
        let mut theta = if self.k1.scalar() > 1e-6_f64 {
            (r_d - self.k0) / self.k1
        } else {
            r_d
        };

        for _ in 0..15 {
            let f = self.poly(theta) - r_d;
            let df = self.poly_derivative(theta);
            let delta = f / df;
            theta = theta - delta;

            if delta.abs().scalar() < 1e-12 {
                break;
            }
        }

        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let inv_r_d = T::one() / r_d;
        let dir_x = x_d * inv_r_d;
        let dir_y = y_d * inv_r_d;

        Vec3::new(dir_x * sin_theta, dir_y * sin_theta, cos_theta)
    }

    pub fn unproject_with_depth(&self, u: T, v: T, depth: T) -> Vec3<T> {
        let ray = self.unproject(u, v);
        let scale = depth / ray.z;
        Vec3::new(ray.x * scale, ray.y * scale, depth)
    }
}

impl BlenderPolyFisheyeCamera<f64> {
    pub fn to_constant<const N: usize>(
        &self,
    ) -> BlenderPolyFisheyeCamera<odysseus_solver::Jet<f64, N>> {
        use odysseus_solver::Jet;
        BlenderPolyFisheyeCamera::new(
            Jet::constant(self.fx),
            Jet::constant(self.fy),
            Jet::constant(self.cx),
            Jet::constant(self.cy),
            Jet::constant(self.k0),
            Jet::constant(self.k1),
            Jet::constant(self.k2),
            Jet::constant(self.k3),
            Jet::constant(self.k4),
        )
    }
}

impl<T: Real> CameraModel<T> for BlenderPolyFisheyeCamera<T>
where
    T::Scalar: PartialOrd<f64>,
{
    const NUM_INTRINSICS: usize = 9;

    fn project(&self, point_cam: Vec3<T>) -> (T, T) {
        BlenderPolyFisheyeCamera::project(self, point_cam)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_blender_project_center() {
        let camera = BlenderPolyFisheyeCamera::new(
            500.0, 500.0, 320.0, 240.0, 0.0, 1.0, 0.05, -0.01, 0.001,
        );
        let point = Vec3::new(0.0, 0.0, 1.0);
        let (u, v) = camera.project(point);

        assert_abs_diff_eq!(u, 320.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v, 240.0, epsilon = 1e-10);
    }

    #[test]
    fn test_blender_equidistant_matches_kb4() {
        // With k0=0, k1=1, k2..4=0 the model is equidistant projection,
        // which equals KB4 with all distortion coefficients zero.
        let bp = BlenderPolyFisheyeCamera::equidistant(500.0, 500.0, 320.0, 240.0);
        let kb = crate::camera::KannalaBrandtCamera::no_distortion(500.0, 500.0, 320.0, 240.0);

        let point = Vec3::new(1.0, 0.5, 1.5);
        let (u_bp, v_bp) = bp.project(point);
        let (u_kb, v_kb) = kb.project(point);

        assert_abs_diff_eq!(u_bp, u_kb, epsilon = 1e-10);
        assert_abs_diff_eq!(v_bp, v_kb, epsilon = 1e-10);
    }

    #[test]
    fn test_blender_unproject_returns_unit_vector() {
        let camera = BlenderPolyFisheyeCamera::new(
            500.0, 500.0, 320.0, 240.0, 0.0, 1.0, 0.05, -0.01, 0.001,
        );
        for &(u, v) in &[(400.0, 300.0), (200.0, 150.0), (450.0, 280.0)] {
            let ray = camera.unproject(u, v);
            let norm = (ray.x * ray.x + ray.y * ray.y + ray.z * ray.z).sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_blender_project_unproject_roundtrip() {
        let camera = BlenderPolyFisheyeCamera::new(
            500.0, 500.0, 320.0, 240.0, 0.0, 1.0, 0.05, -0.01, 0.001,
        );

        let test_points = [
            Vec3::new(0.5, 0.3, 2.0),
            Vec3::new(-1.0, 0.5, 1.5),
            Vec3::new(0.2, -0.8, 3.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];

        for original in test_points {
            let (u, v) = camera.project(original);
            let ray = camera.unproject(u, v);

            let orig_norm = (original.x * original.x
                + original.y * original.y
                + original.z * original.z)
                .sqrt();
            assert_abs_diff_eq!(ray.x, original.x / orig_norm, epsilon = 1e-9);
            assert_abs_diff_eq!(ray.y, original.y / orig_norm, epsilon = 1e-9);
            assert_abs_diff_eq!(ray.z, original.z / orig_norm, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_blender_unproject_project_roundtrip() {
        let camera = BlenderPolyFisheyeCamera::new(
            500.0, 500.0, 320.0, 240.0, 0.0, 1.0, 0.05, -0.01, 0.001,
        );
        for &(u, v) in &[(400.0, 300.0), (200.0, 150.0), (450.0, 280.0), (320.0, 240.0)] {
            let ray = camera.unproject(u, v);
            let (u2, v2) = camera.project(ray);
            assert_abs_diff_eq!(u2, u, epsilon = 1e-9);
            assert_abs_diff_eq!(v2, v, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_blender_with_autodiff() {
        use odysseus_solver::Jet;
        type Jet3 = Jet<f64, 3>;

        let camera = BlenderPolyFisheyeCamera::new(
            Jet3::constant(500.0),
            Jet3::constant(500.0),
            Jet3::constant(320.0),
            Jet3::constant(240.0),
            Jet3::constant(0.0),
            Jet3::constant(1.0),
            Jet3::constant(0.05),
            Jet3::constant(-0.01),
            Jet3::constant(0.001),
        );
        let point = Vec3::new(
            Jet3::variable(1.0, 0),
            Jet3::variable(0.5, 1),
            Jet3::variable(2.0, 2),
        );
        let (u, v) = camera.project(point);

        assert!(u.derivs[0].abs() > 1e-6);
        assert!(u.derivs[2].abs() > 1e-6);
        assert!(v.derivs[1].abs() > 1e-6);
        assert!(v.derivs[2].abs() > 1e-6);
    }

    #[test]
    fn test_blender_intrinsic_derivatives() {
        use odysseus_solver::Jet;
        type J = Jet<f64, 9>;

        let camera = BlenderPolyFisheyeCamera::new(
            J::variable(500.0, 0),
            J::variable(500.0, 1),
            J::variable(320.0, 2),
            J::variable(240.0, 3),
            J::variable(0.0, 4),
            J::variable(1.0, 5),
            J::variable(0.05, 6),
            J::variable(-0.01, 7),
            J::variable(0.001, 8),
        );
        let point = Vec3::new(J::constant(1.0), J::constant(0.5), J::constant(2.0));
        let (u, _v) = camera.project(point);

        // u must depend on every distortion coefficient (else autodiff is broken).
        for idx in 4..=8 {
            assert!(
                u.derivs[idx].abs() > 1e-9,
                "u has zero derivative w.r.t. coefficient idx {idx}"
            );
        }
    }
}
