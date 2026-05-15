//! Double Sphere camera model
//!
//! Reference: Usenko, Demmel, Cremers,
//! "The Double Sphere Camera Model", 3DV 2018.
//! <https://arxiv.org/abs/1807.08957>

use crate::camera::CameraModel;
use odysseus_solver::math3d::Vec3;
use odysseus_solver::Real;

/// Double Sphere camera model.
///
/// Projects a 3D point through two spheres (separated by `xi`) and a final
/// perspective division parameterized by `alpha`. Captures fisheye lenses with
/// FoV up to and beyond 180° while keeping both projection and unprojection in
/// closed form.
///
/// Convention: `xi ∈ [-1, 1]`, `alpha ∈ [0, 1]`.
/// - `xi = 0, alpha = 0` reduces to a pinhole camera.
/// - `alpha = 0` reduces to the unified camera model (UCM) with parameter xi.
#[derive(Debug, Clone, Copy)]
pub struct DoubleSphereCamera<T> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub xi: T,
    pub alpha: T,
}

impl<T: Real> DoubleSphereCamera<T>
where
    T::Scalar: PartialOrd<f64>,
{
    pub fn new(fx: T, fy: T, cx: T, cy: T, xi: T, alpha: T) -> Self {
        Self { fx, fy, cx, cy, xi, alpha }
    }

    /// A pinhole-equivalent DS camera (`xi = 0`, `alpha = 0`).
    pub fn pinhole_equivalent(fx: T, fy: T, cx: T, cy: T) -> Self {
        Self::new(fx, fy, cx, cy, T::zero(), T::zero())
    }

    /// Project a 3D point in camera coordinates to 2D image coordinates.
    pub fn project(&self, point_cam: Vec3<T>) -> (T, T) {
        let x = point_cam.x;
        let y = point_cam.y;
        let z = point_cam.z;

        let d1 = (x * x + y * y + z * z).sqrt();
        let xi_d1_plus_z = self.xi * d1 + z;
        let d2 = (x * x + y * y + xi_d1_plus_z * xi_d1_plus_z).sqrt();
        let denom = self.alpha * d2 + (T::one() - self.alpha) * xi_d1_plus_z;

        let u = self.fx * x / denom + self.cx;
        let v = self.fy * y / denom + self.cy;

        (u, v)
    }

    /// Unproject a 2D pixel to a 3D ray (unit vector) in camera coordinates.
    ///
    /// Closed form — no iteration. Returns a unit vector when the pixel lies
    /// within the model's valid region.
    pub fn unproject(&self, u: T, v: T) -> Vec3<T> {
        let mx = (u - self.cx) / self.fx;
        let my = (v - self.cy) / self.fy;
        let r2 = mx * mx + my * my;

        // mz = (1 - α² r²) / (α sqrt(1 - (2α - 1) r²) + 1 - α)
        let two = T::from_f64(2.0);
        let alpha_sq = self.alpha * self.alpha;
        let inner = T::one() - (two * self.alpha - T::one()) * r2;
        let mz_num = T::one() - alpha_sq * r2;
        let mz_den = self.alpha * inner.sqrt() + (T::one() - self.alpha);
        let mz = mz_num / mz_den;

        // factor = (mz·ξ + sqrt(mz² + (1 - ξ²) r²)) / (mz² + r²)
        let mz_sq = mz * mz;
        let factor_num = mz * self.xi + (mz_sq + (T::one() - self.xi * self.xi) * r2).sqrt();
        let factor_den = mz_sq + r2;
        let factor = factor_num / factor_den;

        Vec3::new(factor * mx, factor * my, factor * mz - self.xi)
    }

    /// Unproject and scale so the resulting point has the given Z depth.
    pub fn unproject_with_depth(&self, u: T, v: T, depth: T) -> Vec3<T> {
        let ray = self.unproject(u, v);
        let scale = depth / ray.z;
        Vec3::new(ray.x * scale, ray.y * scale, depth)
    }
}

impl DoubleSphereCamera<f64> {
    pub fn to_constant<const N: usize>(&self) -> DoubleSphereCamera<odysseus_solver::Jet<f64, N>> {
        use odysseus_solver::Jet;
        DoubleSphereCamera::new(
            Jet::constant(self.fx),
            Jet::constant(self.fy),
            Jet::constant(self.cx),
            Jet::constant(self.cy),
            Jet::constant(self.xi),
            Jet::constant(self.alpha),
        )
    }
}

impl<T: Real> CameraModel<T> for DoubleSphereCamera<T>
where
    T::Scalar: PartialOrd<f64>,
{
    const NUM_INTRINSICS: usize = 6;

    fn project(&self, point_cam: Vec3<T>) -> (T, T) {
        DoubleSphereCamera::project(self, point_cam)
    }

    fn unproject(&self, u: T, v: T) -> Vec3<T> {
        DoubleSphereCamera::unproject(self, u, v)
    }
}

impl crate::camera::CameraConstantJet for DoubleSphereCamera<f64> {
    type Jet<const N: usize> = DoubleSphereCamera<odysseus_solver::Jet<f64, N>>;

    fn constant_jet<const N: usize>(&self) -> Self::Jet<N> {
        self.to_constant::<N>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ds_project_center() {
        let camera = DoubleSphereCamera::new(500.0, 500.0, 320.0, 240.0, -0.2, 0.5);
        let point = Vec3::new(0.0, 0.0, 1.0);
        let (u, v) = camera.project(point);

        assert_abs_diff_eq!(u, 320.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v, 240.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ds_pinhole_equivalent_matches_pinhole() {
        // xi = 0, alpha = 0  =>  pinhole projection
        let ds = DoubleSphereCamera::pinhole_equivalent(500.0, 500.0, 320.0, 240.0);
        let point = Vec3::new(1.0, 0.5, 2.0);
        let (u, v) = ds.project(point);

        let expected_u = 500.0 * (1.0 / 2.0) + 320.0;
        let expected_v = 500.0 * (0.5 / 2.0) + 240.0;
        assert_abs_diff_eq!(u, expected_u, epsilon = 1e-10);
        assert_abs_diff_eq!(v, expected_v, epsilon = 1e-10);
    }

    #[test]
    fn test_ds_unproject_returns_unit_vector() {
        let camera = DoubleSphereCamera::new(500.0, 500.0, 320.0, 240.0, -0.2, 0.5);

        for &(u, v) in &[(400.0, 300.0), (320.0, 240.0), (200.0, 150.0)] {
            let ray = camera.unproject(u, v);
            let norm = (ray.x * ray.x + ray.y * ray.y + ray.z * ray.z).sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ds_project_unproject_roundtrip() {
        let camera = DoubleSphereCamera::new(500.0, 500.0, 320.0, 240.0, -0.2, 0.5);

        let test_points = [
            Vec3::new(0.5, 0.3, 2.0),
            Vec3::new(-1.0, 0.5, 1.5),
            Vec3::new(0.2, -0.8, 3.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];

        for original in test_points {
            let (u, v) = camera.project(original);
            let ray = camera.unproject(u, v);

            // ray is a unit vector in the same direction as original
            let orig_norm = (original.x * original.x
                + original.y * original.y
                + original.z * original.z)
                .sqrt();
            let expected = Vec3::new(
                original.x / orig_norm,
                original.y / orig_norm,
                original.z / orig_norm,
            );

            assert_abs_diff_eq!(ray.x, expected.x, epsilon = 1e-9);
            assert_abs_diff_eq!(ray.y, expected.y, epsilon = 1e-9);
            assert_abs_diff_eq!(ray.z, expected.z, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_ds_unproject_project_roundtrip() {
        let camera = DoubleSphereCamera::new(500.0, 500.0, 320.0, 240.0, -0.2, 0.5);

        for &(u, v) in &[(400.0, 300.0), (200.0, 150.0), (450.0, 280.0), (320.0, 240.0)] {
            let ray = camera.unproject(u, v);
            let (u2, v2) = camera.project(ray);
            assert_abs_diff_eq!(u2, u, epsilon = 1e-9);
            assert_abs_diff_eq!(v2, v, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_ds_with_autodiff() {
        use odysseus_solver::Jet;
        type Jet3 = Jet<f64, 3>;

        let camera = DoubleSphereCamera::new(
            Jet3::constant(500.0),
            Jet3::constant(500.0),
            Jet3::constant(320.0),
            Jet3::constant(240.0),
            Jet3::constant(-0.2),
            Jet3::constant(0.5),
        );
        let point = Vec3::new(
            Jet3::variable(1.0, 0),
            Jet3::variable(0.5, 1),
            Jet3::variable(2.0, 2),
        );
        let (u, v) = camera.project(point);

        assert!(u.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(v.derivs.iter().any(|&d| d.abs() > 1e-10));

        // u depends on x and z but not y (small numeric leak through r tolerable).
        assert!(u.derivs[0].abs() > 1e-6);
        assert!(u.derivs[2].abs() > 1e-6);
        // v depends on y and z but not x.
        assert!(v.derivs[1].abs() > 1e-6);
        assert!(v.derivs[2].abs() > 1e-6);
    }

    #[test]
    fn test_ds_intrinsic_derivatives() {
        // With the camera intrinsics as variables, project should have non-zero
        // derivatives in fx, fy, cx, cy, xi, alpha.
        use odysseus_solver::Jet;
        type J = Jet<f64, 6>;

        let camera = DoubleSphereCamera::new(
            J::variable(500.0, 0),
            J::variable(500.0, 1),
            J::variable(320.0, 2),
            J::variable(240.0, 3),
            J::variable(-0.2, 4),
            J::variable(0.5, 5),
        );
        let point = Vec3::new(J::constant(1.0), J::constant(0.5), J::constant(2.0));
        let (u, v) = camera.project(point);

        // u depends on fx, cx, xi, alpha (and indirectly through denom on others).
        assert!(u.derivs[0].abs() > 1e-6); // fx
        assert!(u.derivs[2].abs() > 1e-6); // cx
        assert!(u.derivs[4].abs() > 1e-6); // xi
        assert!(u.derivs[5].abs() > 1e-6); // alpha
        // v depends on fy, cy, xi, alpha.
        assert!(v.derivs[1].abs() > 1e-6); // fy
        assert!(v.derivs[3].abs() > 1e-6); // cy
        assert!(v.derivs[4].abs() > 1e-6); // xi
        assert!(v.derivs[5].abs() > 1e-6); // alpha
    }

    #[test]
    fn test_ds_to_constant() {
        use odysseus_solver::Jet;

        let camera_f64 = DoubleSphereCamera::new(500.0, 500.0, 320.0, 240.0, -0.2, 0.5);
        let camera_jet: DoubleSphereCamera<Jet<f64, 6>> = camera_f64.to_constant();

        assert_abs_diff_eq!(camera_jet.fx.value, 500.0, epsilon = 1e-10);
        assert_abs_diff_eq!(camera_jet.xi.value, -0.2, epsilon = 1e-10);
        assert_abs_diff_eq!(camera_jet.alpha.value, 0.5, epsilon = 1e-10);
        assert!(camera_jet.fx.derivs.iter().all(|&d| d == 0.0));
    }
}
