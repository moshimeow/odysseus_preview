//! SO(3) - Special Orthogonal Group (3D Rotations)
//!
//! This module provides a type-safe wrapper around unit quaternions with
//! Lie algebra operations (exp, log, composition).

use odysseus_solver::math3d::{Mat3, Quat, Vec3};
use odysseus_solver::Real;
use std::ops::Mul;

/// SO(3) rotation representation
///
/// Internally stored as a unit quaternion.
/// Use `exp()` to convert from axis-angle (tangent space) to SO3.
/// Use `log()` to convert from SO3 to axis-angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SO3<T> {
    /// Unit quaternion representing the rotation
    pub quat: Quat<T>,
}

impl<T: Real> SO3<T> {
    /// Create identity rotation
    pub fn identity() -> Self {
        Self {
            quat: Quat::identity(),
        }
    }

    /// Exponential map: axis-angle -> SO3
    ///
    /// Converts a rotation vector (axis-angle representation) to a rotation.
    /// The direction of the vector is the rotation axis, the magnitude is the angle.
    ///
    /// # Arguments
    /// * `rvec` - Rotation vector in tangent space R^3
    ///
    /// # Returns
    /// * Rotation in SO(3)
    pub fn exp(rvec: Vec3<T>) -> Self {
        Self {
            quat: Quat::from_axis_angle(rvec),
        }
    }

    /// Logarithm map: SO3 -> axis-angle
    ///
    /// Converts a rotation to axis-angle representation.
    /// This is the inverse of `exp()`.
    ///
    /// # Returns
    /// * Rotation vector in tangent space R^3
    pub fn log(&self) -> Vec3<T> {
        self.quat.to_axis_angle()
    }

    /// Rotate a 3D vector
    ///
    /// # Arguments
    /// * `v` - Vector to rotate
    ///
    /// # Returns
    /// * Rotated vector
    pub fn rotate(&self, v: Vec3<T>) -> Vec3<T> {
        self.quat.rotate_vec(v)
    }

    /// Get the inverse rotation (conjugate for unit quaternions)
    pub fn inverse(&self) -> Self {
        Self {
            quat: self.quat.conjugate(),
        }
    }

    /// Normalize the quaternion to unit length
    ///
    /// This corrects for numerical drift after many quaternion multiplications.
    /// Should be called periodically in integration loops to prevent NaN.
    pub fn normalize(&self) -> Self {
        Self {
            quat: self.quat.normalize(),
        }
    }

    /// Convert to rotation matrix
    ///
    /// Returns the equivalent 3x3 rotation matrix.
    pub fn to_matrix(&self) -> Mat3<T> {
        self.quat.to_matrix()
    }
}

impl SO3<f64> {
    /// Create SO3 from a rotation matrix
    ///
    /// Converts a 3x3 rotation matrix to SO3 (quaternion representation).
    /// Only implemented for f64 since it requires runtime branching.
    pub fn from_matrix(matrix: Mat3<f64>) -> Self {
        Self {
            quat: Quat::from_matrix(matrix),
        }
    }
}

/// Composition: SO3 * SO3
impl<T: Real> Mul for SO3<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            quat: self.quat * rhs.quat,
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
        assert_abs_diff_eq!(rot.quat.w, id.quat.w, epsilon = 1e-10);
        assert_abs_diff_eq!(rot.quat.x, id.quat.x, epsilon = 1e-10);
        assert_abs_diff_eq!(rot.quat.y, id.quat.y, epsilon = 1e-10);
        assert_abs_diff_eq!(rot.quat.z, id.quat.z, epsilon = 1e-10);
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

            // Test by rotating a point - the rotations should match
            let p = Vec3::new(1.0, 2.0, 3.0);
            let r1 = rot.rotate(p);
            let r2 = rot_recovered.rotate(p);

            // Use relative epsilon for large values
            assert_abs_diff_eq!(r1.x, r2.x, epsilon = 1e-4);
            assert_abs_diff_eq!(r1.y, r2.y, epsilon = 1e-4);
            assert_abs_diff_eq!(r1.z, r2.z, epsilon = 1e-4);
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
