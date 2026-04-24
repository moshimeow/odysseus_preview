//! 3D math primitives with automatic differentiation support
//!
//! Provides Vec3 and Mat3 types that work generically with any Real type,
//! enabling the same code to work with or without autodiff.

use crate::Real;
use std::ops::{Add, Mul, Sub};

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

    /// Convert to a Vec3 of a different Real type, treating values as constants
    ///
    /// This is useful for converting e.g. Vec3<f64> to Vec3<Jet<f64, N>>
    /// where the f64 values become constant Jets (zero derivatives).
    pub fn to_constant<U: Real<Scalar = T>>(self) -> Vec3<U> {
        Vec3 {
            x: U::constant(self.x),
            y: U::constant(self.y),
            z: U::constant(self.z),
        }
    }

    /// Convert to a Vec3 of Jets, treating values as variables with sequential derivative indices
    ///
    /// - x gets derivative index `deriv_offset`
    /// - y gets derivative index `deriv_offset + 1`
    /// - z gets derivative index `deriv_offset + 2`
    ///
    /// Example:
    /// ```
    /// use odysseus_solver::math3d::Vec3;
    /// use odysseus_solver::Jet;
    ///
    /// let v = Vec3::new(1.0, 2.0, 3.0);
    /// let v_jet: Vec3<Jet<f64, 6>> = v.to_variable(3);
    /// // v_jet.x has derivative 1.0 at index 3
    /// // v_jet.y has derivative 1.0 at index 4
    /// // v_jet.z has derivative 1.0 at index 5
    /// ```
    pub fn to_variable<const N: usize>(self, deriv_offset: usize) -> Vec3<crate::Jet<T, N>>
    where
        T: Default + num_traits::One,
    {
        Vec3 {
            x: crate::Jet::variable(self.x, deriv_offset),
            y: crate::Jet::variable(self.y, deriv_offset + 1),
            z: crate::Jet::variable(self.z, deriv_offset + 2),
        }
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

impl<R, T> std::ops::Add<Vec3<T>> for Vec3<R>
where
    R: Add<T, Output = R>,
{
    type Output = Vec3<R>;

    fn add(self, other: Vec3<T>) -> Vec3<R> {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: Real> std::ops::Mul<T> for Vec3<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl<T: Real> std::ops::Div<T> for Vec3<T> {
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl<T: Real> std::ops::Neg for Vec3<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: T::zero() - self.x,
            y: T::zero() - self.y,
            z: T::zero() - self.z,
        }
    }
}

// Note: RHS multiplication (scalar * Vec3) is tricky with generics and orphan rules.

// ============================================================================
// Conversions from nalgebra
// ============================================================================

impl<T: Copy> From<nalgebra::Vector3<T>> for Vec3<T> {
    fn from(v: nalgebra::Vector3<T>) -> Self {
        Self {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }
}

impl<T: Copy> From<&nalgebra::Vector3<T>> for Vec3<T> {
    fn from(v: &nalgebra::Vector3<T>) -> Self {
        Self {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }
}

impl<T> From<[T; 3]> for Vec3<T> {
    fn from(arr: [T; 3]) -> Self {
        let [x, y, z] = arr;
        Self { x, y, z }
    }
}               
                                                                                                                   
impl<T: Copy> From<&[T; 3]> for Vec3<T> {
    fn from(arr: &[T; 3]) -> Self {
        Self { x: arr[0], y: arr[1], z: arr[2] }
    }
}

impl<T: Copy + nalgebra::Scalar> From<Vec3<T>> for nalgebra::Vector3<T> {
    fn from(v: Vec3<T>) -> Self {
        nalgebra::Vector3::new(v.x, v.y, v.z)
    }
}
// For now, we only provide Vec3 * scalar and Vec3 / scalar.

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

    /// Convert to a Mat3 of a different Real type, treating values as constants
    pub fn to_constant<U: Real<Scalar = T>>(self) -> Mat3<U> {
        Mat3 {
            x_axis: self.x_axis.to_constant(),
            y_axis: self.y_axis.to_constant(),
            z_axis: self.z_axis.to_constant(),
        }
    }

    /// Convert to a Mat3 of Jets, treating values as variables with sequential derivative indices
    ///
    /// Uses column-major ordering: x_axis (0-2), y_axis (3-5), z_axis (6-8)
    pub fn to_variable<const N: usize>(self, deriv_offset: usize) -> Mat3<crate::Jet<T, N>>
    where
        T: Default + num_traits::One,
    {
        Mat3 {
            x_axis: self.x_axis.to_variable(deriv_offset),
            y_axis: self.y_axis.to_variable(deriv_offset + 3),
            z_axis: self.z_axis.to_variable(deriv_offset + 6),
        }
    }

    /// Element accessors (column-major indexing)
    pub fn m00(&self) -> T {
        self.x_axis.x
    }
    pub fn m10(&self) -> T {
        self.x_axis.y
    }
    pub fn m20(&self) -> T {
        self.x_axis.z
    }
    pub fn m01(&self) -> T {
        self.y_axis.x
    }
    pub fn m11(&self) -> T {
        self.y_axis.y
    }
    pub fn m21(&self) -> T {
        self.y_axis.z
    }
    pub fn m02(&self) -> T {
        self.z_axis.x
    }
    pub fn m12(&self) -> T {
        self.z_axis.y
    }
    pub fn m22(&self) -> T {
        self.z_axis.z
    }
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

        col0.x * (col1.y * col2.z - col1.z * col2.y) - col0.y * (col1.x * col2.z - col1.z * col2.x)
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
// Quat - Unit Quaternion (for 3D rotations)
// ============================================================================

/// Unit quaternion for 3D rotations, generic over any Real type
///
/// Uses scalar-first convention: q = w + xi + yj + zk
/// For rotations, quaternions should be normalized (|q| = 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy> Quat<T> {
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }

    /// Convert to a Quat of a different Real type, treating values as constants
    pub fn to_constant<U: Real<Scalar = T>>(self) -> Quat<U> {
        Quat {
            w: U::constant(self.w),
            x: U::constant(self.x),
            y: U::constant(self.y),
            z: U::constant(self.z),
        }
    }

    /// Convert to a Quat of Jets, treating values as variables with sequential derivative indices
    ///
    /// Order: w (offset), x (offset+1), y (offset+2), z (offset+3)
    ///
    /// Note: For rotation optimization, you typically parameterize with 3 axis-angle
    /// parameters rather than 4 quaternion components. This method is provided for
    /// cases where you need all 4 components as variables.
    pub fn to_variable<const N: usize>(self, deriv_offset: usize) -> Quat<crate::Jet<T, N>>
    where
        T: Default + num_traits::One,
    {
        Quat {
            w: crate::Jet::variable(self.w, deriv_offset),
            x: crate::Jet::variable(self.x, deriv_offset + 1),
            y: crate::Jet::variable(self.y, deriv_offset + 2),
            z: crate::Jet::variable(self.z, deriv_offset + 3),
        }
    }
}

impl<T: Real> Quat<T> {
    /// Create identity quaternion (no rotation)
    pub fn identity() -> Self {
        Self {
            w: T::one(),
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    pub fn dot(self, other: Self) -> T {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Quaternion conjugate (inverse for unit quaternions)
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: T::zero() - self.x,
            y: T::zero() - self.y,
            z: T::zero() - self.z,
        }
    }

    /// Quaternion inverse: q^(-1) = conjugate(q) / |q|²
    ///
    /// For unit quaternions (rotation quaternions), this is equivalent to conjugate.
    /// For non-unit quaternions, this computes the true multiplicative inverse.
    pub fn inverse(self) -> Self {
        let norm_sq = self.norm_squared();
        let conj = self.conjugate();
        Self {
            w: conj.w / norm_sq,
            x: conj.x / norm_sq,
            y: conj.y / norm_sq,
            z: conj.z / norm_sq,
        }
    }

    /// Squared norm of the quaternion
    pub fn norm_squared(self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Norm (magnitude) of the quaternion
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// Normalize the quaternion to unit length
    pub fn normalize(self) -> Self {
        let n = self.norm();
        Self {
            w: self.w / n,
            x: self.x / n,
            y: self.y / n,
            z: self.z / n,
        }
    }

    /// Hamilton product (quaternion multiplication)
    ///
    /// q1 * q2 represents applying rotation q1 after q2
    pub fn mul<S: Copy, R>(self, other: Quat<S>) -> Quat<R>
    where
        T: Mul<S, Output = R>,
        R: Sub<Output = R> + Add<Output = R> + Copy,
    {
        Quat {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Rotate a 3D vector by this quaternion
    ///
    /// Uses the optimized formula: v' = v + 2w(q_xyz × v) + 2(q_xyz × (q_xyz × v))
    /// which avoids full quaternion multiplication.
    pub fn rotate_vec<S: Copy, R>(self, v: Vec3<S>) -> Vec3<R>
    where
        T: Mul<S, Output = R> + Mul<R, Output = R>,
        S: Add<R, Output = R>,
        R: Sub<Output = R> + Add<Output = R> + Copy,
    {
        // t = 2 * (q_xyz × v)
        let tx = T::from_literal(2.0) * (self.y * v.z - self.z * v.y);
        let ty = T::from_literal(2.0) * (self.z * v.x - self.x * v.z);
        let tz = T::from_literal(2.0) * (self.x * v.y - self.y * v.x);

        // v' = v + w*t + (q_xyz × t)
        Vec3 {
            x: v.x + self.w * tx + (self.y * tz - self.z * ty),
            y: v.y + self.w * ty + (self.z * tx - self.x * tz),
            z: v.z + self.w * tz + (self.x * ty - self.y * tx),
        }
    }

    /// Rotate a 3x3 matrix by this quaternion
    ///
    /// Computes R * M where R is the rotation matrix equivalent to this quaternion.
    /// This is done by rotating each column of M individually.
    pub fn rotate_matrix(self, m: Mat3<T>) -> Mat3<T> {
        Mat3::from_cols(
            self.rotate_vec(m.x_axis),
            self.rotate_vec(m.y_axis),
            self.rotate_vec(m.z_axis),
        )
    }

    /// Exponential map: axis-angle vector to unit quaternion
    ///
    /// Given rotation vector ω (axis × angle), computes the quaternion:
    /// q = (cos(θ/2), sin(θ/2) * axis) where θ = ||ω||
    pub fn from_axis_angle(rvec: Vec3<T>) -> Self {
        let theta_sq = rvec.x * rvec.x + rvec.y * rvec.y + rvec.z * rvec.z;
        let theta = theta_sq.sqrt();
        let half_theta = theta * T::from_literal(0.5);

        let sin_half = half_theta.sin();
        let cos_half = half_theta.cos();

        // Taylor series for small angles: sin(θ/2)/(θ) ≈ 0.5 - θ²/48
        let taylor_sinc_half = T::from_literal(0.5) - theta_sq * T::from_literal(1.0 / 48.0);

        // Exact formula with safe division
        let eps_sq = T::from_literal(1e-20);
        let theta_safe = (theta_sq + eps_sq).sqrt();
        let exact_sinc_half = sin_half / theta_safe;

        // Blend between Taylor and exact
        let blend = theta_sq / (theta_sq + T::from_literal(0.001));
        let sinc_half = taylor_sinc_half * (T::one() - blend) + exact_sinc_half * blend;

        Self {
            w: cos_half,
            x: sinc_half * rvec.x,
            y: sinc_half * rvec.y,
            z: sinc_half * rvec.z,
        }
    }

    /// Logarithm map: unit quaternion to axis-angle vector
    ///
    /// Returns the rotation vector ω = θ * axis where θ = 2 * acos(w)
    pub fn to_axis_angle(self) -> Vec3<T> {
        // Canonicalize to w >= 0 to stay on the principal branch (angle in [0, π]).
        // Negating a quaternion gives the same rotation but flips the sign of w,
        // so we always work with the representative that has w >= 0.
        let sign = if self.w.scalar() < T::Scalar::default() { -T::one() } else { T::one() };
        let (x, y, z, w) = (sign * self.x, sign * self.y, sign * self.z, sign * self.w);

        // θ = 2 * acos(w), but we need to handle the sign of w
        // For unit quaternion, |xyz| = sin(θ/2)
        let xyz_norm_sq = x * x + y * y + z * z;
        let xyz_norm = xyz_norm_sq.sqrt();

        // half_theta = asin(|xyz|) or acos(w).
        // Clamp w above 1 to defend against FP drift on products of near-unit quats
        // (w.acos() is NaN for w > 1). Canonicalization above already guarantees w ≥ 0.
        let w_safe = if w.scalar() > T::one().scalar() { T::one() } else { w };
        let half_theta = w_safe.acos();
        let theta = half_theta * T::from_literal(2.0);
        let theta_sq = theta * theta;

        // Taylor series for small angles: θ / sin(θ/2) ≈ 2 + θ²/12
        let taylor_k = T::from_literal(2.0) + theta_sq * T::from_literal(1.0 / 12.0);

        // Exact formula: θ / sin(θ/2) = θ / |xyz|
        let eps = T::from_literal(1e-10);
        let exact_k = theta / (xyz_norm + eps);

        // Blend between Taylor and exact
        let blend = xyz_norm_sq / (xyz_norm_sq + T::from_literal(0.0001));
        let k = taylor_k * (T::one() - blend) + exact_k * blend;

        Vec3 {
            x: k * x,
            y: k * y,
            z: k * z,
        }
    }

    /// Convert quaternion to rotation matrix
    ///
    /// Returns the equivalent 3x3 rotation matrix.
    pub fn to_matrix(self) -> Mat3<T> {
        let w = self.w;
        let x = self.x;
        let y = self.y;
        let z = self.z;

        let two = T::from_literal(2.0);

        // Compute rotation matrix elements
        // R = I + 2w*K + 2*K^2 where K is skew-symmetric from (x,y,z)
        // Or equivalently:
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        Mat3::from_cols(
            Vec3::new(T::one() - two * (yy + zz), two * (xy + wz), two * (xz - wy)),
            Vec3::new(two * (xy - wz), T::one() - two * (xx + zz), two * (yz + wx)),
            Vec3::new(two * (xz + wy), two * (yz - wx), T::one() - two * (xx + yy)),
        )
    }
}

/// Convert rotation matrix to quaternion (for f64)
///
/// Uses Shepperd's method for numerical stability.
/// Only implemented for f64 since it requires runtime branching.
impl Quat<f64> {
    pub fn from_matrix(m: Mat3<f64>) -> Self {
        // Shepperd's method: find the largest diagonal element to avoid division by small numbers
        let trace = m.m00() + m.m11() + m.m22();

        let (w, x, y, z) = if trace > 0.0 {
            // w is largest
            let s = (1.0 + trace).sqrt() * 2.0; // s = 4*w
            let w = s * 0.25;
            let x = (m.m21() - m.m12()) / s;
            let y = (m.m02() - m.m20()) / s;
            let z = (m.m10() - m.m01()) / s;
            (w, x, y, z)
        } else if m.m00() > m.m11() && m.m00() > m.m22() {
            // x is largest
            let s = (1.0 + m.m00() - m.m11() - m.m22()).sqrt() * 2.0; // s = 4*x
            let w = (m.m21() - m.m12()) / s;
            let x = s * 0.25;
            let y = (m.m01() + m.m10()) / s;
            let z = (m.m02() + m.m20()) / s;
            (w, x, y, z)
        } else if m.m11() > m.m22() {
            // y is largest
            let s = (1.0 + m.m11() - m.m00() - m.m22()).sqrt() * 2.0; // s = 4*y
            let w = (m.m02() - m.m20()) / s;
            let x = (m.m01() + m.m10()) / s;
            let y = s * 0.25;
            let z = (m.m12() + m.m21()) / s;
            (w, x, y, z)
        } else {
            // z is largest
            let s = (1.0 + m.m22() - m.m00() - m.m11()).sqrt() * 2.0; // s = 4*z
            let w = (m.m10() - m.m01()) / s;
            let x = (m.m02() + m.m20()) / s;
            let y = (m.m12() + m.m21()) / s;
            let z = s * 0.25;
            (w, x, y, z)
        };

        Self { w, x, y, z }
    }
}

impl<T: Real> std::ops::Neg for Quat<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            w: T::zero() - self.w,
            x: T::zero() - self.x,
            y: T::zero() - self.y,
            z: T::zero() - self.z,
        }
    }
}

// Quaternion multiplication via Mul trait
impl<T: Real, S: Copy, R> std::ops::Mul<Quat<S>> for Quat<T>
where
    T: Mul<S, Output = R>,
    R: Sub<Output = R> + Add<Output = R> + Copy,
{
    type Output = Quat<R>;

    fn mul(self, other: Quat<S>) -> Quat<R> {
        self.mul(other)
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
pub fn transform_point<T: Real>(
    rotation: Mat3<T>,
    translation: Vec3<T>,
    point: Vec3<T>,
) -> Vec3<T> {
    let rotated = rotation.mul_vec(point);
    Vec3 {
        x: rotated.x + translation.x,
        y: rotated.y + translation.y,
        z: rotated.z + translation.z,
    }
}

/// Apply a rigid transformation: point_transformed = R * point + translation
pub fn transform_point_quat<T: Real>(
    rotation: Quat<T>,
    translation: Vec3<T>,
    point: Vec3<T>,
) -> Vec3<T> {
    let rotated = rotation.rotate_vec(point);
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
    fn test_to_constant() {
        type Jet6 = Jet<f64, 6>;

        // Vec3<f64> -> Vec3<Jet<f64, 6>>
        let v_f64 = Vec3::new(1.0, 2.0, 3.0);
        let v_jet: Vec3<Jet6> = v_f64.to_constant();

        assert_eq!(v_jet.x.value, 1.0);
        assert_eq!(v_jet.y.value, 2.0);
        assert_eq!(v_jet.z.value, 3.0);
        // All derivatives should be zero (constant)
        assert!(v_jet.x.derivs.iter().all(|&d| d == 0.0));
        assert!(v_jet.y.derivs.iter().all(|&d| d == 0.0));
        assert!(v_jet.z.derivs.iter().all(|&d| d == 0.0));

        // Can now do operations with Jet variables
        let rx = Jet6::variable(0.1, 0);
        let ry = Jet6::variable(0.2, 1);
        let rz = Jet6::variable(0.3, 2);
        let rotation = Vec3::new(rx, ry, rz);
        let q = Quat::from_axis_angle(rotation);

        // Rotate the constant point - result has derivatives wrt rotation
        let rotated = q.rotate_vec(v_jet);
        assert!(rotated.x.derivs.iter().any(|&d| d != 0.0));
    }

    #[test]
    fn test_to_variable() {
        type Jet6 = Jet<f64, 6>;

        // Vec3<f64> -> Vec3<Jet<f64, 6>> as variables starting at offset 2
        let v_f64 = Vec3::new(1.0, 2.0, 3.0);
        let v_jet: Vec3<Jet6> = v_f64.to_variable(2);

        // Values should be preserved
        assert_eq!(v_jet.x.value, 1.0);
        assert_eq!(v_jet.y.value, 2.0);
        assert_eq!(v_jet.z.value, 3.0);

        // Derivatives should be 1.0 at the correct indices
        assert_eq!(v_jet.x.derivs[2], 1.0); // x at offset 2
        assert_eq!(v_jet.y.derivs[3], 1.0); // y at offset 3
        assert_eq!(v_jet.z.derivs[4], 1.0); // z at offset 4

        // Other derivatives should be 0
        assert_eq!(v_jet.x.derivs[0], 0.0);
        assert_eq!(v_jet.x.derivs[1], 0.0);
        assert_eq!(v_jet.x.derivs[3], 0.0);
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
            x_axis: Vec3::new(0.0, 1.0, 0.0),  // First column
            y_axis: Vec3::new(-1.0, 0.0, 0.0), // Second column
            z_axis: Vec3::new(0.0, 0.0, 1.0),  // Third column
        };

        // Rotating [1, 0, 0] by 90° around Z should give [0, 1, 0]
        let x_unit = Vec3::new(1.0, 0.0, 0.0);
        let result = rot_z_90.mul_vec(x_unit);

        assert!(
            (result.x - 0.0).abs() < 1e-10,
            "Expected x=0, got {}",
            result.x
        );
        assert!(
            (result.y - 1.0).abs() < 1e-10,
            "Expected y=1, got {}",
            result.y
        );
        assert!(
            (result.z - 0.0).abs() < 1e-10,
            "Expected z=0, got {}",
            result.z
        );
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

    // ========================================================================
    // Quaternion tests
    // ========================================================================

    #[test]
    fn test_quat_identity() {
        let q = Quat::<f64>::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn test_quat_identity_rotation() {
        // Identity quaternion should not change vector
        let q = Quat::<f64>::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = q.rotate_vec(v);

        assert!((result.x - 1.0).abs() < 1e-10);
        assert!((result.y - 2.0).abs() < 1e-10);
        assert!((result.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_quat_90deg_z_rotation() {
        // 90 degree rotation around Z axis
        let rvec = Vec3::new(0.0, 0.0, std::f64::consts::PI / 2.0);
        let q = Quat::from_axis_angle(rvec);

        // Rotate X axis, should get Y axis
        let x_axis = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vec(x_axis);

        assert!(
            (rotated.x - 0.0).abs() < 1e-5,
            "Expected x=0, got {}",
            rotated.x
        );
        assert!(
            (rotated.y - 1.0).abs() < 1e-5,
            "Expected y=1, got {}",
            rotated.y
        );
        assert!(
            (rotated.z - 0.0).abs() < 1e-5,
            "Expected z=0, got {}",
            rotated.z
        );
    }

    #[test]
    fn test_quat_composition() {
        // Two 90-degree rotations around Z should give 180 degrees
        let rvec = Vec3::new(0.0, 0.0, std::f64::consts::PI / 2.0);
        let q1 = Quat::from_axis_angle(rvec);
        let q2 = Quat::from_axis_angle(rvec);
        let combined = q1 * q2;

        // Apply to X axis, should get -X
        let x_axis = Vec3::new(1.0, 0.0, 0.0);
        let rotated = combined.rotate_vec(x_axis);

        assert!(
            (rotated.x - (-1.0)).abs() < 1e-5,
            "Expected x=-1, got {}",
            rotated.x
        );
        assert!(
            (rotated.y - 0.0).abs() < 1e-5,
            "Expected y=0, got {}",
            rotated.y
        );
        assert!(
            (rotated.z - 0.0).abs() < 1e-5,
            "Expected z=0, got {}",
            rotated.z
        );
    }

    #[test]
    fn test_quat_inverse() {
        let rvec = Vec3::new(0.3, 0.4, 0.5);
        let q = Quat::from_axis_angle(rvec);
        let q_inv = q.conjugate();
        let identity: Quat<f64> = q * q_inv;

        // Should be close to identity
        assert!((identity.w - 1.0).abs() < 1e-5);
        assert!(identity.x.abs() < 1e-5);
        assert!(identity.y.abs() < 1e-5);
        assert!(identity.z.abs() < 1e-5);
    }

    #[test]
    fn test_quat_exp_log_roundtrip() {
        let test_cases = vec![
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
        ];

        for rvec in test_cases {
            let q = Quat::from_axis_angle(rvec);
            let rvec_recovered = q.to_axis_angle();
            let q_recovered = Quat::from_axis_angle(rvec_recovered);

            // Test by rotating a point
            let p = Vec3::new(1.0, 2.0, 3.0);
            let r1: Vec3<f64> = q.rotate_vec(p);
            let r2: Vec3<f64> = q_recovered.rotate_vec(p);

            assert!(
                (r1.x - r2.x).abs() < 1e-4,
                "x mismatch: {} vs {}",
                r1.x,
                r2.x
            );
            assert!(
                (r1.y - r2.y).abs() < 1e-4,
                "y mismatch: {} vs {}",
                r1.y,
                r2.y
            );
            assert!(
                (r1.z - r2.z).abs() < 1e-4,
                "z mismatch: {} vs {}",
                r1.z,
                r2.z
            );
        }
    }

    #[test]
    fn test_quat_log_exp_roundtrip() {
        // For small angles, log(exp(ω)) should equal ω
        let test_cases = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.01, 0.02, 0.03),
        ];

        for rvec in test_cases {
            let q = Quat::from_axis_angle(rvec);
            let rvec_recovered = q.to_axis_angle();

            assert!((rvec_recovered.x - rvec.x).abs() < 1e-6);
            assert!((rvec_recovered.y - rvec.y).abs() < 1e-6);
            assert!((rvec_recovered.z - rvec.z).abs() < 1e-6);
        }
    }

    #[test]
    fn test_quat_with_autodiff() {
        type Jet3 = Jet<f64, 3>;

        let rx = Jet3::variable(0.2, 0);
        let ry = Jet3::variable(0.3, 1);
        let rz = Jet3::variable(0.1, 2);

        let rvec = Vec3::new(rx, ry, rz);
        let q = Quat::from_axis_angle(rvec);

        // Rotate a point
        let p = Vec3::new(
            Jet3::constant(1.0),
            Jet3::constant(2.0),
            Jet3::constant(3.0),
        );
        let rotated = q.rotate_vec(p);

        // Check that we have non-zero derivatives
        assert!(rotated.x.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(rotated.y.derivs.iter().any(|&d| d.abs() > 1e-10));
        assert!(rotated.z.derivs.iter().any(|&d| d.abs() > 1e-10));
    }

    #[test]
    fn test_quat_matches_rodrigues() {
        // Verify quaternion rotation gives same result as rotation matrix
        let test_cases = vec![
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.0, 0.0, std::f64::consts::PI / 2.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
        ];

        for rvec in test_cases {
            let q = Quat::from_axis_angle(rvec);
            let mat = rodrigues_to_matrix(rvec);

            let p = Vec3::new(1.0, 2.0, 3.0);
            let r_quat = q.rotate_vec(p);
            let r_mat = mat.mul_vec(p);

            assert!(
                (r_quat.x - r_mat.x).abs() < 1e-4,
                "x mismatch for rvec {:?}: {} vs {}",
                rvec,
                r_quat.x,
                r_mat.x
            );
            assert!(
                (r_quat.y - r_mat.y).abs() < 1e-4,
                "y mismatch for rvec {:?}: {} vs {}",
                rvec,
                r_quat.y,
                r_mat.y
            );
            assert!(
                (r_quat.z - r_mat.z).abs() < 1e-4,
                "z mismatch for rvec {:?}: {} vs {}",
                rvec,
                r_quat.z,
                r_mat.z
            );
        }
    }
}
