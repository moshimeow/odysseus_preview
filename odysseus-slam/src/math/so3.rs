//! SO(3) - Special Orthogonal Group (3D Rotations)
//!
//! This module provides a type-safe wrapper around rotation matrices with
//! Lie algebra operations (exp, log, composition).

use odysseus_solver::math3d::{Mat3, Vec3, rodrigues_to_matrix};
use odysseus_solver::Real;
use std::ops::Mul;

/// SO(3) rotation representation
///
/// Internally stored as a 3x3 rotation matrix.
/// Use `exp()` to convert from axis-angle (tangent space) to SO3.
/// Use `log()` to convert from SO3 to axis-angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SO3<T> {
    /// Rotation matrix (column-major)
    pub matrix: Mat3<T>,
}

impl<T: Real> SO3<T> {
    /// Create identity rotation
    pub fn identity() -> Self {
        Self {
            matrix: Mat3::identity(),
        }
    }

    /// Exponential map: axis-angle -> SO3
    ///
    /// Converts a rotation vector (axis-angle representation) to a rotation matrix.
    /// The direction of the vector is the rotation axis, the magnitude is the angle.
    ///
    /// # Arguments
    /// * `rvec` - Rotation vector in tangent space R^3
    ///
    /// # Returns
    /// * Rotation in SO(3)
    pub fn exp(rvec: Vec3<T>) -> Self {
        Self {
            matrix: rodrigues_to_matrix(rvec),
        }
    }

    /// Logarithm map: SO3 -> axis-angle
    ///
    /// Converts a rotation matrix to axis-angle representation.
    /// This is the inverse of `exp()`.
    ///
    /// # Returns
    /// * Rotation vector in tangent space R^3
    pub fn log(&self) -> Vec3<T> {
        // Extract rotation angle from trace
        // trace(R) = 1 + 2*cos(theta)
        // theta = arccos((trace - 1) / 2)
        let trace = self.matrix.x_axis.x + self.matrix.y_axis.y + self.matrix.z_axis.z;
        let cos_theta = (trace - T::one()) * T::from_literal(0.5);
        let theta = cos_theta.acos();
        let theta_sq = theta * theta;

        // For small angles, use Taylor series to avoid division by sin(theta)
        // For large angles, use exact formula
        let eps_sq = T::from_literal(1e-20);
        let theta_safe = (theta_sq + eps_sq).sqrt();
        let sin_theta = theta.sin();

        // Taylor series: k ~= 0.5 + theta^2/12
        let taylor_k = T::from_literal(0.5) + theta_sq * T::from_literal(1.0 / 12.0);

        // Exact: k = theta / (2 * sin(theta))
        let exact_k = theta_safe / (T::from_literal(2.0) * sin_theta + eps_sq);

        // Blend between Taylor and exact
        let blend = theta_sq / (theta_sq + T::from_literal(0.001));
        let k = taylor_k * (T::one() - blend) + exact_k * blend;

        // Extract axis from skew-symmetric part: omega = k * (R - R^T)
        Vec3::new(
            k * (self.matrix.y_axis.z - self.matrix.z_axis.y),
            k * (self.matrix.z_axis.x - self.matrix.x_axis.z),
            k * (self.matrix.x_axis.y - self.matrix.y_axis.x),
        )
    }

    /// Rotate a 3D vector
    ///
    /// # Arguments
    /// * `v` - Vector to rotate
    ///
    /// # Returns
    /// * Rotated vector
    pub fn rotate(&self, v: Vec3<T>) -> Vec3<T> {
        self.matrix.mul_vec(v)
    }

    /// Get the inverse rotation (transpose for rotation matrices)
    pub fn inverse(&self) -> Self {
        // For rotation matrices, inverse = transpose
        Self {
            matrix: Mat3::from_cols(
                Vec3::new(
                    self.matrix.x_axis.x,
                    self.matrix.y_axis.x,
                    self.matrix.z_axis.x,
                ),
                Vec3::new(
                    self.matrix.x_axis.y,
                    self.matrix.y_axis.y,
                    self.matrix.z_axis.y,
                ),
                Vec3::new(
                    self.matrix.x_axis.z,
                    self.matrix.y_axis.z,
                    self.matrix.z_axis.z,
                ),
            ),
        }
    }
}

/// Composition: SO3 * SO3
impl<T: Real> Mul for SO3<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // Matrix multiplication
        let m1 = self.matrix;
        let m2 = rhs.matrix;

        Self {
            matrix: Mat3::from_cols(
                Vec3::new(
                    m1.x_axis.x * m2.x_axis.x
                        + m1.y_axis.x * m2.x_axis.y
                        + m1.z_axis.x * m2.x_axis.z,
                    m1.x_axis.y * m2.x_axis.x
                        + m1.y_axis.y * m2.x_axis.y
                        + m1.z_axis.y * m2.x_axis.z,
                    m1.x_axis.z * m2.x_axis.x
                        + m1.y_axis.z * m2.x_axis.y
                        + m1.z_axis.z * m2.x_axis.z,
                ),
                Vec3::new(
                    m1.x_axis.x * m2.y_axis.x
                        + m1.y_axis.x * m2.y_axis.y
                        + m1.z_axis.x * m2.y_axis.z,
                    m1.x_axis.y * m2.y_axis.x
                        + m1.y_axis.y * m2.y_axis.y
                        + m1.z_axis.y * m2.y_axis.z,
                    m1.x_axis.z * m2.y_axis.x
                        + m1.y_axis.z * m2.y_axis.y
                        + m1.z_axis.z * m2.y_axis.z,
                ),
                Vec3::new(
                    m1.x_axis.x * m2.z_axis.x
                        + m1.y_axis.x * m2.z_axis.y
                        + m1.z_axis.x * m2.z_axis.z,
                    m1.x_axis.y * m2.z_axis.x
                        + m1.y_axis.y * m2.z_axis.y
                        + m1.z_axis.y * m2.z_axis.z,
                    m1.x_axis.z * m2.z_axis.x
                        + m1.y_axis.z * m2.z_axis.y
                        + m1.z_axis.z * m2.z_axis.z,
                ),
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
        let rot = SO3::<f64>::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = rot.rotate(v);

        assert_abs_diff_eq!(rotated.x, v.x, epsilon = 1e-10);
        assert_abs_diff_eq!(rotated.y, v.y, epsilon = 1e-10);
        assert_abs_diff_eq!(rotated.z, v.z, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_identity() {
        // Zero rotation vector should give identity
        let rvec = Vec3::new(0.0, 0.0, 0.0);
        let rot = SO3::exp(rvec);

        let id = SO3::<f64>::identity();
        assert_abs_diff_eq!(rot.matrix.x_axis.x, id.matrix.x_axis.x, epsilon = 1e-10);
        assert_abs_diff_eq!(rot.matrix.y_axis.y, id.matrix.y_axis.y, epsilon = 1e-10);
        assert_abs_diff_eq!(rot.matrix.z_axis.z, id.matrix.z_axis.z, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        // Test that exp(log(R)) = R for various rotations
        let test_cases = vec![
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(3.0, 0.0, 0.0), // Large rotation
        ];

        for rvec in test_cases {
            let rot = SO3::exp(rvec);
            let rvec_recovered = rot.log();
            let rot_recovered = SO3::exp(rvec_recovered);

            // Check that matrices match
            // 1e-2 seems pretty high? Consider consulting
            assert_abs_diff_eq!(
                rot_recovered.matrix.x_axis.x,
                rot.matrix.x_axis.x,
                epsilon = 1e-2
            );
            assert_abs_diff_eq!(
                rot_recovered.matrix.y_axis.y,
                rot.matrix.y_axis.y,
                epsilon = 1e-2
            );
            assert_abs_diff_eq!(
                rot_recovered.matrix.z_axis.z,
                rot.matrix.z_axis.z,
                epsilon = 1e-2
            );
        }
    }

    #[test]
    fn test_log_exp_roundtrip() {
        // Test that log(exp(omega)) = omega for small angles
        let test_cases = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.01, 0.02, 0.03),
        ];

        for rvec in test_cases {
            let rot = SO3::exp(rvec);
            let rvec_recovered = rot.log();

            assert_abs_diff_eq!(rvec_recovered.x, rvec.x, epsilon = 1e-4);
            assert_abs_diff_eq!(rvec_recovered.y, rvec.y, epsilon = 1e-4);
            assert_abs_diff_eq!(rvec_recovered.z, rvec.z, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_rotation_z_axis() {
        // 90-degree rotation around Z axis
        let rvec = Vec3::new(0.0, 0.0, std::f64::consts::PI / 2.0);
        let rot = SO3::exp(rvec);

        // Rotate X axis, should get Y axis
        let x_axis = Vec3::new(1.0, 0.0, 0.0);
        let rotated = rot.rotate(x_axis);

        assert_abs_diff_eq!(rotated.x, 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(rotated.y, 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(rotated.z, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_composition() {
        // Two 90-degree rotations around Z should give 180 degrees
        let rvec = Vec3::new(0.0, 0.0, std::f64::consts::PI / 2.0);
        let rot1 = SO3::exp(rvec);
        let rot2 = SO3::exp(rvec);
        let combined = rot1 * rot2;

        // Apply to X axis, should get -X
        let x_axis = Vec3::new(1.0, 0.0, 0.0);
        let rotated = combined.rotate(x_axis);

        assert_abs_diff_eq!(rotated.x, -1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(rotated.y, 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(rotated.z, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_inverse() {
        let rvec = Vec3::new(0.3, 0.4, 0.5);
        let rot = SO3::exp(rvec);
        let rot_inv = rot.inverse();
        let identity = rot * rot_inv;

        // Should be identity
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = identity.rotate(v);

        assert_abs_diff_eq!(result.x, v.x, epsilon = 1e-4);
        assert_abs_diff_eq!(result.y, v.y, epsilon = 1e-4);
        assert_abs_diff_eq!(result.z, v.z, epsilon = 1e-4);
    }

    #[test]
    fn test_with_autodiff() {
        use odysseus_solver::Jet;

        type Jet3 = Jet<f64, 3>;

        // Create rotation with Jets
        let rx = Jet3::variable(0.2, 0);
        let ry = Jet3::variable(0.3, 1);
        let rz = Jet3::variable(0.1, 2);

        let rvec = Vec3::new(rx, ry, rz);
        let rot = SO3::exp(rvec);

        // Rotate a point
        let p = Vec3::new(
            Jet3::constant(1.0),
            Jet3::constant(2.0),
            Jet3::constant(3.0),
        );
        let rotated = rot.rotate(p);

        // Check that we have non-zero derivatives
        assert!(rotated.x.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(rotated.y.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(rotated.z.derivs.iter().any(|&d| d.abs() > 1e-10));
    }
}
