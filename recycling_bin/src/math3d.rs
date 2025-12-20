/// 3D math library for transformations and geometry
///
/// Provides glam-style ergonomics for 3D vectors, matrices, and transformations.
/// All operations work generically over MathContext for automatic differentiation.

use crate::math_context::MathContext;

// ============================================================================
// Vec3 - 3D Vector
// ============================================================================

/// 3D vector that works with any MathContext (f64, JetHandle, etc.)
#[derive(Debug, Clone, Copy)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl Vec3<f64> {
    /// Create a zero vector
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Create a vector with all components set to the same value
    pub fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }
}

// ============================================================================
// Mat3 - 3x3 Matrix (column-major)
// ============================================================================

/// 3x3 matrix stored in column-major order
/// Columns: x_axis, y_axis, z_axis
#[derive(Debug, Clone, Copy)]
pub struct Mat3<T> {
    pub x_axis: Vec3<T>,
    pub y_axis: Vec3<T>,
    pub z_axis: Vec3<T>,
}

impl<T: Copy> Mat3<T> {
    pub fn from_cols(x_axis: Vec3<T>, y_axis: Vec3<T>, z_axis: Vec3<T>) -> Self {
        Self { x_axis, y_axis, z_axis }
    }
}

impl Mat3<f64> {
    /// Create an identity matrix
    pub fn identity() -> Self {
        Self {
            x_axis: Vec3::new(1.0, 0.0, 0.0),
            y_axis: Vec3::new(0.0, 1.0, 0.0),
            z_axis: Vec3::new(0.0, 0.0, 1.0),
        }
    }
}

// ============================================================================
// Rodrigues rotation formula
// ============================================================================

/// Convert a rotation vector (axis-angle, aka rodrigues vector) to a rotation matrix
///
/// The rotation vector encodes both the axis (direction) and angle (magnitude):
/// - Direction: axis of rotation (normalized internally)
/// - Magnitude: angle in radians
///
/// Uses Rodrigues' rotation formula:
/// R = I + sin(θ) * K + (1 - cos(θ)) * K²
/// where K is the skew-symmetric matrix of the axis
///
/// For small angles (|theta| < 1e-6), uses first-order Taylor approximation to avoid division by zero.
pub fn rodrigues_to_matrix<Ctx: MathContext>(
    ctx: &mut Ctx,
    rvec: Vec3<Ctx::Value>,
) -> Mat3<Ctx::Value> {
    use crate::expr;

    // Compute angle (magnitude of rotation vector)
    let theta_sq = expr!(ctx, rvec.x * rvec.x + rvec.y * rvec.y + rvec.z * rvec.z);

    // Hybrid approach for Rodrigues formula that works for all angles:
    // - Small angles (θ < threshold): Use Taylor series to avoid division by zero
    // - Large angles (θ >= threshold): Use exact trigonometric formulas
    //
    // We compute both and blend smoothly to maintain differentiability.
    //
    // Key limits as θ → 0:
    // - sin(θ)/θ → 1
    // - (1-cos(θ))/θ² → 1/2

    let theta = expr!(ctx, sqrt(theta_sq));
    let sin_theta = expr!(ctx, sin(theta));
    let cos_theta = expr!(ctx, cos(theta));

    // For safe division, we add a small epsilon that won't affect large angles
    // but prevents division by zero. The key is to use formulations that
    // approach the correct limits.
    //
    // For sin(θ)/θ, we can safely compute sin(θ)/(θ + ε) since:
    // - When θ is large, ε is negligible
    // - When θ → 0, we get sin(0)/(0 + ε) = 0/ε = 0
    //   But we WANT the limit to be 1, not 0!
    //
    // Better approach: Use the identity sin(θ)/θ = sinc(θ), and for small θ:
    // sinc(θ) ≈ 1 - θ²/6
    //
    // We use: sin(θ) / max(θ, ε) which gives:
    // - Large θ: sin(θ)/θ (correct)
    // - Small θ: sin(θ)/ε ≈ θ/ε (wrong!)
    //
    // Instead, add ε² not ε, and use sqrt(θ² + ε²):

    let eps_sq = ctx.constant(1e-20);
    let theta_safe = expr!(ctx, sqrt(theta_sq + eps_sq));

    // sin(θ)/θ with safe division
    // As θ→0: sin(θ)→0 and θ_safe→√ε, giving 0/√ε = 0 (wrong, should be 1)
    // We need to use 2nd order Taylor for small angles

    // Better: blend between Taylor (small θ) and exact (large θ)
    // Taylor: sin(θ)/θ ≈ 1 - θ²/6
    let taylor_sinc = expr!(ctx, 1.0 - theta_sq / 6.0);

    // Exact: sin(θ)/θ
    let exact_sinc = expr!(ctx, sin_theta / theta_safe);

    // Blend using theta_sq as the parameter (smooth transition around θ ≈ 0.03 rad)
    // For θ² < 0.001 (θ < 0.03 rad ≈ 2°): mostly Taylor
    // For θ² > 0.001: mostly exact
    // This gives blend_factor ≈ 0.999 for θ = 0.1 rad (already 99.9% exact)
    let blend_factor = expr!(ctx, theta_sq / (theta_sq + 0.001));
    let sin_theta_over_theta = expr!(ctx, taylor_sinc * (1.0 - blend_factor) + exact_sinc * blend_factor);

    // Similarly for (1-cos(θ))/θ²:
    // Taylor: ≈ 1/2 - θ²/24
    let taylor_versin = expr!(ctx, 0.5 - theta_sq / 24.0);

    // Exact: (1-cos(θ))/θ²
    let theta_sq_safe = expr!(ctx, theta_sq + eps_sq);
    let exact_versin = expr!(ctx, (1.0 - cos_theta) / theta_sq_safe);

    let one_minus_cos_over_theta_sq = expr!(ctx, taylor_versin * (1.0 - blend_factor) + exact_versin * blend_factor);

    // K (skew-symmetric) uses rvec components directly (unnormalized)
    let kx = rvec.x;
    let ky = rvec.y;
    let kz = rvec.z;

    // Skew-symmetric matrix K (using unnormalized rvec):
    // K = [  0  -kz   ky ]
    //     [ kz    0  -kx ]
    //     [-ky   kx    0 ]

    // K² terms (precompute for efficiency)
    let kx2 = expr!(ctx, kx * kx);
    let ky2 = expr!(ctx, ky * ky);
    let kz2 = expr!(ctx, kz * kz);
    let kxky = expr!(ctx, kx * ky);
    let kxkz = expr!(ctx, kx * kz);
    let kykz = expr!(ctx, ky * kz);

    // R = I + (sin(θ)/θ) * K + ((1-cos(θ))/θ²) * K²
    // where K uses unnormalized rvec components

    // First column: [1, 0, 0] + (sin/θ) * [0, kz, -ky] + ((1-cos)/θ²) * [-(ky²+kz²), kxky, kxkz]
    let r00 = expr!(ctx, 1.0 - one_minus_cos_over_theta_sq * (ky2 + kz2));
    let r10 = expr!(ctx, sin_theta_over_theta * kz + one_minus_cos_over_theta_sq * kxky);
    let r20 = expr!(ctx, 0.0 - sin_theta_over_theta * ky + one_minus_cos_over_theta_sq * kxkz);

    // Second column: [0, 1, 0] + (sin/θ) * [-kz, 0, kx] + ((1-cos)/θ²) * [kxky, -(kx²+kz²), kykz]
    let r01 = expr!(ctx, 0.0 - sin_theta_over_theta * kz + one_minus_cos_over_theta_sq * kxky);
    let r11 = expr!(ctx, 1.0 - one_minus_cos_over_theta_sq * (kx2 + kz2));
    let r21 = expr!(ctx, sin_theta_over_theta * kx + one_minus_cos_over_theta_sq * kykz);

    // Third column: [0, 0, 1] + (sin/θ) * [ky, -kx, 0] + ((1-cos)/θ²) * [kxkz, kykz, -(kx²+ky²)]
    let r02 = expr!(ctx, sin_theta_over_theta * ky + one_minus_cos_over_theta_sq * kxkz);
    let r12 = expr!(ctx, 0.0 - sin_theta_over_theta * kx + one_minus_cos_over_theta_sq * kykz);
    let r22 = expr!(ctx, 1.0 - one_minus_cos_over_theta_sq * (kx2 + ky2));

    Mat3::from_cols(
        Vec3::new(r00, r10, r20),
        Vec3::new(r01, r11, r21),
        Vec3::new(r02, r12, r22),
    )
}

// ============================================================================
// Matrix-vector multiplication
// ============================================================================

/// Multiply a 3x3 matrix by a 3D vector
pub fn mat3_mul_vec3<Ctx: MathContext>(
    ctx: &mut Ctx,
    mat: Mat3<Ctx::Value>,
    vec: Vec3<Ctx::Value>,
) -> Vec3<Ctx::Value> {
    use crate::expr;

    let x = expr!(ctx, mat.x_axis.x * vec.x + mat.y_axis.x * vec.y + mat.z_axis.x * vec.z);
    let y = expr!(ctx, mat.x_axis.y * vec.x + mat.y_axis.y * vec.y + mat.z_axis.y * vec.z);
    let z = expr!(ctx, mat.x_axis.z * vec.x + mat.y_axis.z * vec.y + mat.z_axis.z * vec.z);

    Vec3::new(x, y, z)
}

// ============================================================================
// 3D Rigid Transform
// ============================================================================

/// Apply a rigid transformation to a point: point_transformed = R * point + translation
pub fn transform_point<Ctx: MathContext>(
    ctx: &mut Ctx,
    rotation: Mat3<Ctx::Value>,
    translation: Vec3<Ctx::Value>,
    point: Vec3<Ctx::Value>,
) -> Vec3<Ctx::Value> {
    use crate::expr;

    let rotated = mat3_mul_vec3(ctx, rotation, point);

    Vec3::new(
        expr!(ctx, rotated.x + translation.x),
        expr!(ctx, rotated.y + translation.y),
        expr!(ctx, rotated.z + translation.z),
    )
}
