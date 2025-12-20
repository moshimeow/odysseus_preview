//! 3D math primitives with automatic differentiation support
//!
//! Provides Vec3 and Mat3 types that work generically with any Real type,
//! enabling the same code to work with or without autodiff.

use crate::Real;

// ============================================================================
// Vec3 - 3D Vector
// ============================================================================

/// 3D vector generic over any Real type
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy> Vec3<T> {
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T: Real> Vec3<T> {
    /// Create a zero vector
    pub fn zero() -> Self {
        Self {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    /// Dot product
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Length squared
    pub fn length_squared(self) -> T {
        self.dot(self)
    }

    /// Length (magnitude)
    pub fn length(self) -> T {
        self.length_squared().sqrt()
    }

    /// Norm (alias for length)
    pub fn norm(self) -> T {
        self.length()
    }

    /// Cross product
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

// Arithmetic operations for Vec3
impl<T: Real> std::ops::Sub for Vec3<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: Real> std::ops::Add for Vec3<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

// ============================================================================
// Mat3 - 3x3 Matrix (column-major)
// ============================================================================

/// 3x3 matrix stored in column-major order
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3<T> {
    pub x_axis: Vec3<T>,
    pub y_axis: Vec3<T>,
    pub z_axis: Vec3<T>,
}

impl<T: Copy> Mat3<T> {
    pub const fn from_cols(x_axis: Vec3<T>, y_axis: Vec3<T>, z_axis: Vec3<T>) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    /// Element accessors (column-major indexing)
    pub fn m00(&self) -> T { self.x_axis.x }
    pub fn m10(&self) -> T { self.x_axis.y }
    pub fn m20(&self) -> T { self.x_axis.z }
    pub fn m01(&self) -> T { self.y_axis.x }
    pub fn m11(&self) -> T { self.y_axis.y }
    pub fn m21(&self) -> T { self.y_axis.z }
    pub fn m02(&self) -> T { self.z_axis.x }
    pub fn m12(&self) -> T { self.z_axis.y }
    pub fn m22(&self) -> T { self.z_axis.z }
}

impl<T: Real> Mat3<T> {
    /// Identity matrix
    pub fn identity() -> Self {
        Self {
            x_axis: Vec3::new(T::one(), T::zero(), T::zero()),
            y_axis: Vec3::new(T::zero(), T::one(), T::zero()),
            z_axis: Vec3::new(T::zero(), T::zero(), T::one()),
        }
    }

    /// Transpose matrix
    pub fn transpose(self) -> Self {
        Self {
            x_axis: Vec3::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            y_axis: Vec3::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            z_axis: Vec3::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
        }
    }

    /// Determinant of the matrix
    pub fn determinant(self) -> T {
        // det(M) = dot(col0, cross(col1, col2))
        let col0 = self.x_axis;
        let col1 = self.y_axis;
        let col2 = self.z_axis;

        col0.x * (col1.y * col2.z - col1.z * col2.y)
            - col0.y * (col1.x * col2.z - col1.z * col2.x)
            + col0.z * (col1.x * col2.y - col1.y * col2.x)
    }

    /// Inverse of the matrix using adjugate method
    /// For rotation matrices, inverse = transpose, but this works for any invertible matrix
    pub fn inverse(self) -> Self {
        let det = self.determinant();

        // Compute adjugate matrix (transpose of cofactor matrix)
        // Cofactor matrix elements:
        let c00 = self.y_axis.y * self.z_axis.z - self.y_axis.z * self.z_axis.y;
        let c01 = self.y_axis.z * self.z_axis.x - self.y_axis.x * self.z_axis.z;
        let c02 = self.y_axis.x * self.z_axis.y - self.y_axis.y * self.z_axis.x;

        let c10 = self.x_axis.z * self.z_axis.y - self.x_axis.y * self.z_axis.z;
        let c11 = self.x_axis.x * self.z_axis.z - self.x_axis.z * self.z_axis.x;
        let c12 = self.x_axis.y * self.z_axis.x - self.x_axis.x * self.z_axis.y;

        let c20 = self.x_axis.y * self.y_axis.z - self.x_axis.z * self.y_axis.y;
        let c21 = self.x_axis.z * self.y_axis.x - self.x_axis.x * self.y_axis.z;
        let c22 = self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x;

        let inv_det = T::one() / det;

        Self {
            x_axis: Vec3::new(c00 * inv_det, c10 * inv_det, c20 * inv_det),
            y_axis: Vec3::new(c01 * inv_det, c11 * inv_det, c21 * inv_det),
            z_axis: Vec3::new(c02 * inv_det, c12 * inv_det, c22 * inv_det),
        }
    }
}

// ============================================================================
// Matrix-vector multiplication
// ============================================================================

impl<T: Real> Mat3<T> {
    /// Multiply matrix by vector
    pub fn mul_vec(self, v: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.x_axis.x * v.x + self.y_axis.x * v.y + self.z_axis.x * v.z,
            y: self.x_axis.y * v.x + self.y_axis.y * v.y + self.z_axis.y * v.z,
            z: self.x_axis.z * v.x + self.y_axis.z * v.y + self.z_axis.z * v.z,
        }
    }
}

// Matrix-matrix multiplication
impl<T: Real> std::ops::Mul for Mat3<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // Multiply each column of other by self
        Self {
            x_axis: self.mul_vec(other.x_axis),
            y_axis: self.mul_vec(other.y_axis),
            z_axis: self.mul_vec(other.z_axis),
        }
    }
}

// ============================================================================
// Rodrigues rotation formula
// ============================================================================

/// Convert a rotation vector (axis-angle) to a rotation matrix
///
/// The rotation vector encodes both axis (direction) and angle (magnitude).
/// Uses Rodrigues' formula with hybrid Taylor/exact approach for numerical stability.
pub fn rodrigues_to_matrix<T: Real>(rvec: Vec3<T>) -> Mat3<T> {
    // Compute angle (magnitude of rotation vector)
    let theta_sq = rvec.x * rvec.x + rvec.y * rvec.y + rvec.z * rvec.z;

    // Hybrid approach: Taylor series for small angles, exact for large angles
    // This ensures correct derivatives at theta=0 and accuracy for large rotations

    let theta = theta_sq.sqrt();
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();

    // Taylor series approximations (exact at theta=0)
    let taylor_sinc = T::one() - theta_sq * T::from_literal(1.0 / 6.0);
    let taylor_versin = T::from_literal(0.5) - theta_sq * T::from_literal(1.0 / 24.0);

    // Exact formulas with safe division
    let eps_sq = T::from_literal(1e-20);
    let theta_safe = (theta_sq + eps_sq).sqrt();
    let theta_sq_safe = theta_sq + eps_sq;

    let exact_sinc = sin_theta / theta_safe;
    let exact_versin = (T::one() - cos_theta) / theta_sq_safe;

    // Smooth blending between Taylor and exact
    // For θ² < 0.001 (θ < 0.03 rad ≈ 2°): mostly Taylor
    // For θ² > 0.001: mostly exact
    let blend_factor = theta_sq / (theta_sq + T::from_literal(0.001));

    let sin_theta_over_theta = taylor_sinc * (T::one() - blend_factor) + exact_sinc * blend_factor;
    let one_minus_cos_over_theta_sq =
        taylor_versin * (T::one() - blend_factor) + exact_versin * blend_factor;

    // K (skew-symmetric matrix) components
    let kx = rvec.x;
    let ky = rvec.y;
    let kz = rvec.z;

    // K² terms
    let kx2 = kx * kx;
    let ky2 = ky * ky;
    let kz2 = kz * kz;
    let kxky = kx * ky;
    let kxkz = kx * kz;
    let kykz = ky * kz;

    // R = I + (sin(θ)/θ) * K + ((1-cos(θ))/θ²) * K²

    // First column
    let r00 = T::one() - one_minus_cos_over_theta_sq * (ky2 + kz2);
    let r10 = sin_theta_over_theta * kz + one_minus_cos_over_theta_sq * kxky;
    let r20 = T::zero() - sin_theta_over_theta * ky + one_minus_cos_over_theta_sq * kxkz;

    // Second column
    let r01 = T::zero() - sin_theta_over_theta * kz + one_minus_cos_over_theta_sq * kxky;
    let r11 = T::one() - one_minus_cos_over_theta_sq * (kx2 + kz2);
    let r21 = sin_theta_over_theta * kx + one_minus_cos_over_theta_sq * kykz;

    // Third column
    let r02 = sin_theta_over_theta * ky + one_minus_cos_over_theta_sq * kxkz;
    let r12 = T::zero() - sin_theta_over_theta * kx + one_minus_cos_over_theta_sq * kykz;
    let r22 = T::one() - one_minus_cos_over_theta_sq * (kx2 + ky2);

    Mat3::from_cols(
        Vec3::new(r00, r10, r20),
        Vec3::new(r01, r11, r21),
        Vec3::new(r02, r12, r22),
    )
}

/// Apply a rigid transformation: point_transformed = R * point + translation
pub fn transform_point<T: Real>(rotation: Mat3<T>, translation: Vec3<T>, point: Vec3<T>) -> Vec3<T> {
    let rotated = rotation.mul_vec(point);
    Vec3 {
        x: rotated.x + translation.x,
        y: rotated.y + translation.y,
        z: rotated.z + translation.z,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Jet;

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_rodrigues_identity() {
        // Zero rotation should give identity matrix
        let rvec = Vec3::new(0.0, 0.0, 0.0);
        let mat = rodrigues_to_matrix(rvec);

        let id = Mat3::<f64>::identity();
        assert!(f64::abs(mat.x_axis.x - id.x_axis.x) < 1e-10);
        assert!(f64::abs(mat.y_axis.y - id.y_axis.y) < 1e-10);
        assert!(f64::abs(mat.z_axis.z - id.z_axis.z) < 1e-10);
    }

    #[test]
    fn test_rodrigues_with_jets() {
        // Test that rodrigues works with autodiff
        type Jet6 = Jet<f64, 6>;

        let rx = Jet6::variable(0.2, 3);
        let ry = Jet6::variable(0.3, 4);
        let rz = Jet6::variable(0.1, 5);

        let rvec = Vec3::new(rx, ry, rz);
        let mat = rodrigues_to_matrix(rvec);

        // Check that we got a matrix with derivatives
        assert!(mat.x_axis.x.derivs.iter().any(|&d| d != 0.0));
    }

    #[test]
    fn test_transform_point() {
        // Identity transform
        let rotation = Mat3::identity();
        let translation = Vec3::new(1.0, 2.0, 3.0);
        let point = Vec3::new(4.0, 5.0, 6.0);

        let result = transform_point(rotation, translation, point);
        assert_eq!(result.x, 5.0);
        assert_eq!(result.y, 7.0);
        assert_eq!(result.z, 9.0);
    }

    #[test]
    fn test_mat3_mul_vec_identity() {
        // Identity matrix should not change vector
        let identity = Mat3::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = identity.mul_vec(v);

        assert_eq!(result.x, 1.0);
        assert_eq!(result.y, 2.0);
        assert_eq!(result.z, 3.0);
    }

    #[test]
    fn test_mat3_mul_vec_rotation() {
        // 90 degree rotation around Z axis
        // In column-major storage: [cos, sin, 0], [-sin, cos, 0], [0, 0, 1]
        // For 90 degrees: cos=0, sin=1
        // Columns: [0, 1, 0], [-1, 0, 0], [0, 0, 1]
        let rot_z_90 = Mat3 {
            x_axis: Vec3::new(0.0, 1.0, 0.0),   // First column
            y_axis: Vec3::new(-1.0, 0.0, 0.0),  // Second column
            z_axis: Vec3::new(0.0, 0.0, 1.0),   // Third column
        };

        // Rotating [1, 0, 0] by 90° around Z should give [0, 1, 0]
        let x_unit = Vec3::new(1.0, 0.0, 0.0);
        let result = rot_z_90.mul_vec(x_unit);

        assert!((result.x - 0.0).abs() < 1e-10, "Expected x=0, got {}", result.x);
        assert!((result.y - 1.0).abs() < 1e-10, "Expected y=1, got {}", result.y);
        assert!((result.z - 0.0).abs() < 1e-10, "Expected z=0, got {}", result.z);
    }

    #[test]
    fn test_mat3_inverse() {
        // Test inverse on identity
        let identity = Mat3::<f64>::identity();
        let inv = identity.inverse();

        assert!((inv.x_axis.x - 1.0).abs() < 1e-10);
        assert!((inv.y_axis.y - 1.0).abs() < 1e-10);
        assert!((inv.z_axis.z - 1.0).abs() < 1e-10);

        // Test inverse on rotation matrix (inverse should equal transpose)
        let rot_z_90 = Mat3 {
            x_axis: Vec3::new(0.0, 1.0, 0.0),
            y_axis: Vec3::new(-1.0, 0.0, 0.0),
            z_axis: Vec3::new(0.0, 0.0, 1.0),
        };

        let inv = rot_z_90.inverse();
        let transp = rot_z_90.transpose();

        assert!((inv.x_axis.x - transp.x_axis.x).abs() < 1e-10);
        assert!((inv.x_axis.y - transp.x_axis.y).abs() < 1e-10);
        assert!((inv.y_axis.x - transp.y_axis.x).abs() < 1e-10);
        assert!((inv.y_axis.y - transp.y_axis.y).abs() < 1e-10);

        // Test that M * M^-1 = I
        let product = rot_z_90 * inv;
        assert!((product.x_axis.x - 1.0).abs() < 1e-10);
        assert!((product.y_axis.y - 1.0).abs() < 1e-10);
        assert!((product.z_axis.z - 1.0).abs() < 1e-10);
        assert!((product.x_axis.y).abs() < 1e-10);
        assert!((product.y_axis.x).abs() < 1e-10);
    }
}
