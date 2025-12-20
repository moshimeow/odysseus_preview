//! SE(3) - Special Euclidean Group (3D Rigid Transformations)
//!
//! This module provides SE(3) which represents rigid body transformations
//! (rotation + translation) in 3D space.

use super::SO3;
use odysseus_solver::math3d::Vec3;
use odysseus_solver::Real;
use std::ops::Mul;

/// SE(3) rigid transformation (rotation + translation)
///
/// Represents a 3D pose: a rotation followed by a translation.
/// Stored on the manifold (not in tangent space).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SE3<T> {
    /// Rotation component
    pub rotation: SO3<T>,
    /// Translation component
    pub translation: Vec3<T>,
}

impl<T: Real> SE3<T> {
    /// Create identity transformation
    pub fn identity() -> Self {
        Self {
            rotation: SO3::identity(),
            translation: Vec3::new(T::zero(), T::zero(), T::zero()),
        }
    }

    /// Create SE3 from rotation and translation
    pub fn from_rotation_translation(rotation: SO3<T>, translation: Vec3<T>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Exponential map: se(3) -> SE(3)
    ///
    /// Converts a 6D tangent vector [omega, v] to a rigid transformation.
    /// - omega (first 3): rotation part (axis-angle)
    /// - v (last 3): translation part
    ///
    /// The translation is not directly v, but V(omega) * v, where V is the
    /// left Jacobian of SO(3).
    ///
    /// # Arguments
    /// * `tangent` - 6D tangent vector [omega_x, omega_y, omega_z, v_x, v_y, v_z]
    ///
    /// # Returns
    /// * SE(3) transformation
    pub fn exp(tangent: [T; 6]) -> Self {
        let omega = Vec3::new(tangent[0], tangent[1], tangent[2]);
        let v = Vec3::new(tangent[3], tangent[4], tangent[5]);

        // Rotation part
        let rotation = SO3::exp(omega);

        // Translation part: t = V(omega) * v
        // V(omega) = I + ((1-cos(theta))/theta^2)*K + ((theta-sin(theta))/theta^3)*K^2
        // where K = [omega]_x (skew-symmetric matrix)

        let theta_sq = omega.x * omega.x + omega.y * omega.y + omega.z * omega.z;
        let theta = theta_sq.sqrt();
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        // Taylor series for small angles
        let taylor_b = T::from_literal(0.5) - theta_sq * T::from_literal(1.0 / 24.0);
        let taylor_c = T::from_literal(1.0 / 6.0) - theta_sq * T::from_literal(1.0 / 120.0);

        // Exact formulas with safe division
        let eps_sq = T::from_literal(1e-20);
        let theta_sq_safe = theta_sq + eps_sq;
        let theta_cubed_safe = theta_sq_safe * (theta_sq + eps_sq).sqrt();

        let exact_b = (T::one() - cos_theta) / theta_sq_safe;
        let exact_c = (theta - sin_theta) / theta_cubed_safe;

        // Blend between Taylor and exact
        let blend = theta_sq / (theta_sq + T::from_literal(0.001));
        let b = taylor_b * (T::one() - blend) + exact_b * blend;
        let c = taylor_c * (T::one() - blend) + exact_c * blend;

        // Compute V * v
        // V * v = v + b * (omega x v) + c * (omega x (omega x v))
        let omega_cross_v = Vec3::new(
            omega.y * v.z - omega.z * v.y,
            omega.z * v.x - omega.x * v.z,
            omega.x * v.y - omega.y * v.x,
        );

        let omega_cross_omega_cross_v = Vec3::new(
            omega.y * omega_cross_v.z - omega.z * omega_cross_v.y,
            omega.z * omega_cross_v.x - omega.x * omega_cross_v.z,
            omega.x * omega_cross_v.y - omega.y * omega_cross_v.x,
        );

        let translation = Vec3::new(
            v.x + b * omega_cross_v.x + c * omega_cross_omega_cross_v.x,
            v.y + b * omega_cross_v.y + c * omega_cross_omega_cross_v.y,
            v.z + b * omega_cross_v.z + c * omega_cross_omega_cross_v.z,
        );

        Self {
            rotation,
            translation,
        }
    }

    /// Logarithm map: SE(3) -> se(3)
    ///
    /// Converts a rigid transformation to a 6D tangent vector.
    ///
    /// # Returns
    /// * 6D tangent vector [omega_x, omega_y, omega_z, v_x, v_y, v_z]
    pub fn log(&self) -> [T; 6] {
        // Rotation part
        let omega = self.rotation.log();

        // Translation part: v = V^(-1) * t
        // V^(-1) = I - 0.5*K + (1/theta^2 - (1+cos(theta))/(2*theta*sin(theta)))*K^2

        let theta_sq = omega.x * omega.x + omega.y * omega.y + omega.z * omega.z;
        let theta = theta_sq.sqrt();
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        // Taylor series for small angles
        let taylor_b = T::from_literal(-0.5) + theta_sq * T::from_literal(1.0 / 24.0);
        let taylor_c = T::from_literal(1.0 / 12.0) - theta_sq * T::from_literal(1.0 / 720.0);

        // Exact formulas
        let eps = T::from_literal(1e-10);
        let eps_sq = T::from_literal(1e-20);
        let theta_safe = theta + eps;
        let theta_sq_safe = theta_sq + eps_sq;
        let sin_theta_safe = sin_theta + eps;

        let exact_b = T::from_literal(-0.5);
        let exact_c = T::one() / theta_sq_safe
            - (T::one() + cos_theta) / (T::from_literal(2.0) * theta_safe * sin_theta_safe);

        // Blend
        let blend = theta_sq / (theta_sq + T::from_literal(0.001));
        let b = taylor_b * (T::one() - blend) + exact_b * blend;
        let c = taylor_c * (T::one() - blend) + exact_c * blend;

        // Compute V^(-1) * t
        let t = self.translation;
        let omega_cross_t = Vec3::new(
            omega.y * t.z - omega.z * t.y,
            omega.z * t.x - omega.x * t.z,
            omega.x * t.y - omega.y * t.x,
        );

        let omega_cross_omega_cross_t = Vec3::new(
            omega.y * omega_cross_t.z - omega.z * omega_cross_t.y,
            omega.z * omega_cross_t.x - omega.x * omega_cross_t.z,
            omega.x * omega_cross_t.y - omega.y * omega_cross_t.x,
        );

        let v = Vec3::new(
            t.x + b * omega_cross_t.x + c * omega_cross_omega_cross_t.x,
            t.y + b * omega_cross_t.y + c * omega_cross_omega_cross_t.y,
            t.z + b * omega_cross_t.z + c * omega_cross_omega_cross_t.z,
        );

        [omega.x, omega.y, omega.z, v.x, v.y, v.z]
    }

    /// Transform a 3D point
    ///
    /// Applies the transformation: point_out = R * point_in + t
    ///
    /// # Arguments
    /// * `point` - Point to transform
    ///
    /// # Returns
    /// * Transformed point
    pub fn transform_point(&self, point: Vec3<T>) -> Vec3<T> {
        let rotated = self.rotation.rotate(point);
        Vec3::new(
            rotated.x + self.translation.x,
            rotated.y + self.translation.y,
            rotated.z + self.translation.z,
        )
    }

    /// Get the inverse transformation
    ///
    /// For SE(3), the inverse is: [R, t]^(-1) = [R^T, -R^T * t]
    pub fn inverse(&self) -> Self {
        let r_inv = self.rotation.inverse();
        let t_inv = r_inv.rotate(Vec3::new(
            T::zero() - self.translation.x,
            T::zero() - self.translation.y,
            T::zero() - self.translation.z,
        ));

        Self {
            rotation: r_inv,
            translation: t_inv,
        }
    }
}

/// Composition: SE3 * SE3
///
/// Combines two transformations: (R1, t1) * (R2, t2) = (R1*R2, R1*t2 + t1)
impl<T: Real> Mul for SE3<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            rotation: self.rotation * rhs.rotation,
            translation: Vec3::new(
                self.rotation.rotate(rhs.translation).x + self.translation.x,
                self.rotation.rotate(rhs.translation).y + self.translation.y,
                self.rotation.rotate(rhs.translation).z + self.translation.z,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_identity() {
        let pose = SE3::<f64>::identity();
        let p = Vec3::new(1.0, 2.0, 3.0);
        let transformed = pose.transform_point(p);

        assert_abs_diff_eq!(transformed.x, p.x, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed.y, p.y, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed.z, p.z, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_identity() {
        // Zero tangent vector should give identity
        let tangent = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let pose = SE3::exp(tangent);

        let p = Vec3::new(1.0, 2.0, 3.0);
        let transformed = pose.transform_point(p);

        assert_abs_diff_eq!(transformed.x, p.x, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed.y, p.y, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed.z, p.z, epsilon = 1e-10);
    }

    #[test]
    fn test_pure_translation() {
        // Only translation, no rotation
        let tangent = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let pose = SE3::exp(tangent);

        let p = Vec3::new(4.0, 5.0, 6.0);
        let transformed = pose.transform_point(p);

        assert_abs_diff_eq!(transformed.x, 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed.y, 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed.z, 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pure_rotation() {
        // 90-degree rotation around Z, no translation
        let tangent = [0.0, 0.0, std::f64::consts::PI / 2.0, 0.0, 0.0, 0.0];
        let pose = SE3::exp(tangent);

        let p = Vec3::new(1.0, 0.0, 0.0);
        let transformed = pose.transform_point(p);

        assert_abs_diff_eq!(transformed.x, 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(transformed.y, 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(transformed.z, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let test_cases = vec![
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.1, 0.1, 0.1],
        ];

        for tangent in test_cases {
            let pose = SE3::exp(tangent);
            let tangent_recovered = pose.log();
            let pose_recovered = SE3::exp(tangent_recovered);

            // Test by transforming a point
            let p = Vec3::new(1.0, 2.0, 3.0);
            let t1 = pose.transform_point(p);
            let t2 = pose_recovered.transform_point(p);

            assert_abs_diff_eq!(t1.x, t2.x, epsilon = 1e-4);
            assert_abs_diff_eq!(t1.y, t2.y, epsilon = 1e-4);
            assert_abs_diff_eq!(t1.z, t2.z, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_log_exp_roundtrip() {
        // For small tangent vectors, log(exp(xi)) should equal xi
        let test_cases = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        ];

        for tangent in test_cases {
            let pose = SE3::exp(tangent);
            let tangent_recovered = pose.log();

            for i in 0..6 {
                assert_abs_diff_eq!(tangent_recovered[i], tangent[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_composition() {
        // Two transformations: first translate, then rotate
        let tangent1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // Translate by (1,0,0)
        let tangent2 = [0.0, 0.0, std::f64::consts::PI / 2.0, 0.0, 0.0, 0.0]; // Rotate 90deg around Z

        let pose1 = SE3::exp(tangent1);
        let pose2 = SE3::exp(tangent2);
        let combined = pose2 * pose1;

        // Apply to origin
        let p = Vec3::new(0.0, 0.0, 0.0);
        let transformed = combined.transform_point(p);

        // Should translate by (1,0,0) then rotate to (0,1,0)
        assert_abs_diff_eq!(transformed.x, 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(transformed.y, 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(transformed.z, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_inverse() {
        let tangent = [0.3, 0.4, 0.5, 1.0, 2.0, 3.0];
        let pose = SE3::exp(tangent);
        let pose_inv = pose.inverse();
        let identity = pose * pose_inv;

        // Should be identity
        let p = Vec3::new(1.0, 2.0, 3.0);
        let transformed = identity.transform_point(p);

        assert_abs_diff_eq!(transformed.x, p.x, epsilon = 1e-4);
        assert_abs_diff_eq!(transformed.y, p.y, epsilon = 1e-4);
        assert_abs_diff_eq!(transformed.z, p.z, epsilon = 1e-4);
    }

    #[test]
    fn test_with_autodiff() {
        use odysseus_solver::Jet;

        type Jet6 = Jet<f64, 6>;

        // Create SE3 with Jets
        let tangent: [Jet6; 6] = [
            Jet6::variable(0.2, 0),
            Jet6::variable(0.3, 1),
            Jet6::variable(0.1, 2),
            Jet6::variable(1.0, 3),
            Jet6::variable(2.0, 4),
            Jet6::variable(3.0, 5),
        ];

        let pose = SE3::exp(tangent);

        // Transform a point
        let p = Vec3::new(
            Jet6::constant(1.0),
            Jet6::constant(2.0),
            Jet6::constant(3.0),
        );
        let transformed = pose.transform_point(p);

        // Check that we have derivatives
        assert!(transformed.x.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(transformed.y.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(transformed.z.derivs.iter().any(|&d| d.abs() > 1e-10));
    }

    #[test]
    fn test_transform_consistency() {
        // Test that transforming a point and then transforming back gives the original
        let tangent = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0];
        let pose = SE3::exp(tangent);
        let pose_inv = pose.inverse();

        let p_original = Vec3::new(5.0, 6.0, 7.0);
        let p_transformed = pose.transform_point(p_original);
        let p_back = pose_inv.transform_point(p_transformed);

        assert_abs_diff_eq!(p_back.x, p_original.x, epsilon = 1e-4);
        assert_abs_diff_eq!(p_back.y, p_original.y, epsilon = 1e-4);
        assert_abs_diff_eq!(p_back.z, p_original.z, epsilon = 1e-4);
    }
}
