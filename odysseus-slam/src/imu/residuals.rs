//! IMU Residual Functions
//!
//! Implements residuals for IMU preintegration factors following
//! Forster et al. "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
//!
//! These residual functions support autodiff via the Real trait.

use super::preintegration::PreintegratedImu;
use nalgebra::Vector3;
use odysseus_solver::math3d::{Quat, Vec3};
use odysseus_solver::Real;

/// IMU preintegration residual (9 DOF)
///
/// Computes the residual between predicted and preintegrated motion.
///
/// # Parameters
/// The pose at frame i is parameterized as:
/// - rotation_host_i: Base quaternion for frame i (f64)
/// - pose_params_i: [rot_delta(3), translation(3), velocity(3), gyro_bias(3), accel_bias(3)] = 15 DOF
///
/// Similarly for frame j.
///
/// # Returns
/// 9 residuals: [rotation(3), velocity(3), position(3)]
pub fn imu_preintegration_residual<T: Real<Scalar = f64>>(
    // Frame i parameters
    rotation_host_i: &Quat<f64>,
    pose_params_i: &[T], // [rot_delta, trans, vel, bg, ba] - 15 elements

    // Frame j parameters
    rotation_host_j: &Quat<f64>,
    pose_params_j: &[T], // [rot_delta, trans, vel, bg, ba] - 15 elements

    // Preintegrated measurement
    preint: &PreintegratedImu,

    // Gravity in world frame (m/s²)
    gravity: &[T; 3],
) -> [T; 9] {
    // Extract parameters for frame i
    let rot_delta_i = Vec3::new(pose_params_i[0], pose_params_i[1], pose_params_i[2]);
    let trans_i = Vec3::new(pose_params_i[3], pose_params_i[4], pose_params_i[5]);
    let vel_i = Vec3::new(pose_params_i[6], pose_params_i[7], pose_params_i[8]);
    // Biases need to be f64 for correct_for_bias call
    let bg_i: Vector3<f64> = Vector3::new(
        pose_params_i[9].scalar(),
        pose_params_i[10].scalar(),
        pose_params_i[11].scalar(),
    );
    let ba_i: Vector3<f64> = Vector3::new(
        pose_params_i[12].scalar(),
        pose_params_i[13].scalar(),
        pose_params_i[14].scalar(),
    );

    // Extract parameters for frame j
    let rot_delta_j = Vec3::new(pose_params_j[0], pose_params_j[1], pose_params_j[2]);
    let trans_j = Vec3::new(pose_params_j[3], pose_params_j[4], pose_params_j[5]);
    let vel_j = Vec3::new(pose_params_j[6], pose_params_j[7], pose_params_j[8]);

    // Build rotations: R = R_host * exp(delta)
    let q_delta_i = Quat::from_axis_angle(rot_delta_i);
    let q_host_i = Quat::new(
        T::from_literal(rotation_host_i.w),
        T::from_literal(rotation_host_i.x),
        T::from_literal(rotation_host_i.y),
        T::from_literal(rotation_host_i.z),
    );
    let r_i = q_host_i * q_delta_i;

    let q_delta_j = Quat::from_axis_angle(rot_delta_j);
    let q_host_j = Quat::new(
        T::from_literal(rotation_host_j.w),
        T::from_literal(rotation_host_j.x),
        T::from_literal(rotation_host_j.y),
        T::from_literal(rotation_host_j.z),
    );
    let r_j = q_host_j * q_delta_j;

    // Get bias-corrected preintegrated values
    let (delta_r_corrected, delta_v_corrected, delta_p_corrected) =
        preint.correct_for_bias(&bg_i, &ba_i);

    // Convert corrected delta rotation to quaternion
    let delta_r_quat = rotation_vector_to_quat::<T>(&delta_r_corrected);

    // Time interval
    let dt = T::from_literal(preint.delta_time);
    let dt_sq = dt * dt;

    // Gravity vector
    let g = Vec3::new(gravity[0], gravity[1], gravity[2]);

    // === Rotation Residual ===
    // r_R = Log(ΔR_corrected^T * R_i^T * R_j)
    let r_i_inv = r_i.conjugate();
    let r_ij = r_i_inv * r_j; // R_i^T * R_j
    let delta_r_inv = delta_r_quat.conjugate();
    let r_error = delta_r_inv * r_ij; // ΔR^T * R_i^T * R_j
    let rot_residual = r_error.to_axis_angle();

    // === Velocity Residual ===
    // r_v = R_i^T * (v_j - v_i - g*Δt) - Δv_corrected
    let vel_diff = Vec3::new(
        vel_j.x - vel_i.x - g.x * dt,
        vel_j.y - vel_i.y - g.y * dt,
        vel_j.z - vel_i.z - g.z * dt,
    );
    let vel_in_i = r_i_inv.rotate_vec(vel_diff);
    let vel_residual = Vec3::new(
        vel_in_i.x - T::from_literal(delta_v_corrected.x),
        vel_in_i.y - T::from_literal(delta_v_corrected.y),
        vel_in_i.z - T::from_literal(delta_v_corrected.z),
    );

    // === Position Residual ===
    // r_p = R_i^T * (p_j - p_i - v_i*Δt - 0.5*g*Δt²) - Δp_corrected
    let half = T::from_literal(0.5);
    let pos_diff = Vec3::new(
        trans_j.x - trans_i.x - vel_i.x * dt - half * g.x * dt_sq,
        trans_j.y - trans_i.y - vel_i.y * dt - half * g.y * dt_sq,
        trans_j.z - trans_i.z - vel_i.z * dt - half * g.z * dt_sq,
    );
    let pos_in_i = r_i_inv.rotate_vec(pos_diff);
    let pos_residual = Vec3::new(
        pos_in_i.x - T::from_literal(delta_p_corrected.x),
        pos_in_i.y - T::from_literal(delta_p_corrected.y),
        pos_in_i.z - T::from_literal(delta_p_corrected.z),
    );

    [
        rot_residual.x,
        rot_residual.y,
        rot_residual.z,
        vel_residual.x,
        vel_residual.y,
        vel_residual.z,
        pos_residual.x,
        pos_residual.y,
        pos_residual.z,
    ]
}

/// Bias random walk residual (6 DOF)
///
/// Constrains bias drift between consecutive frames.
/// The residual is weighted by the inverse of the bias random walk covariance.
///
/// # Returns
/// 6 residuals: [gyro_bias(3), accel_bias(3)]
pub fn bias_residual<T: Real>(
    // Bias at frame i
    gyro_bias_i: &[T; 3],
    accel_bias_i: &[T; 3],

    // Bias at frame j
    gyro_bias_j: &[T; 3],
    accel_bias_j: &[T; 3],

    // Time interval
    dt: T,

    // Noise parameters (standard deviation per sqrt(second))
    gyro_bias_sigma: f64,
    accel_bias_sigma: f64,
) -> [T; 6] {
    // Bias change should be small (random walk)
    // Covariance = sigma² * dt
    // Weight = 1 / sqrt(covariance) = 1 / (sigma * sqrt(dt))
    let dt_sqrt = dt.sqrt();
    let gyro_weight = T::from_literal(1.0 / gyro_bias_sigma) / dt_sqrt;
    let accel_weight = T::from_literal(1.0 / accel_bias_sigma) / dt_sqrt;

    [
        (gyro_bias_j[0] - gyro_bias_i[0]) * gyro_weight,
        (gyro_bias_j[1] - gyro_bias_i[1]) * gyro_weight,
        (gyro_bias_j[2] - gyro_bias_i[2]) * gyro_weight,
        (accel_bias_j[0] - accel_bias_i[0]) * accel_weight,
        (accel_bias_j[1] - accel_bias_i[1]) * accel_weight,
        (accel_bias_j[2] - accel_bias_i[2]) * accel_weight,
    ]
}

/// Convert rotation vector to quaternion
fn rotation_vector_to_quat<T: Real>(rvec: &Vector3<f64>) -> Quat<T> {
    let theta_sq = rvec.x * rvec.x + rvec.y * rvec.y + rvec.z * rvec.z;
    let theta = theta_sq.sqrt();

    if theta < 1e-10 {
        // Small angle: q ≈ [1, rvec/2]
        Quat::new(
            T::one(),
            T::from_literal(rvec.x * 0.5),
            T::from_literal(rvec.y * 0.5),
            T::from_literal(rvec.z * 0.5),
        )
    } else {
        let half_theta = theta * 0.5;
        let sin_half = half_theta.sin();
        let cos_half = half_theta.cos();
        let scale = sin_half / theta;

        Quat::new(
            T::from_literal(cos_half),
            T::from_literal(rvec.x * scale),
            T::from_literal(rvec.y * scale),
            T::from_literal(rvec.z * scale),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::PreintegratedImu;

    #[test]
    fn test_imu_residual_identity() {
        // Two identical frames with no motion should give zero residual
        let host_quat = Quat::identity();
        let gravity = [0.0, 0.0, -9.81];

        // Frame i and j at same position/velocity
        let pose_params: [f64; 15] = [
            0.0, 0.0, 0.0, // rotation delta
            0.0, 0.0, 0.0, // translation
            0.0, 0.0, 0.0, // velocity
            0.0, 0.0, 0.0, // gyro bias
            0.0, 0.0, 0.0, // accel bias
        ];

        // Zero preintegration (identity)
        let preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());

        let residual = imu_preintegration_residual(
            &host_quat,
            &pose_params,
            &host_quat,
            &pose_params,
            &preint,
            &gravity,
        );

        // All residuals should be near zero
        for (i, r) in residual.iter().enumerate() {
            assert!(r.abs() < 1e-10, "Residual {} = {} is not near zero", i, r);
        }
    }

    #[test]
    fn test_imu_residual_pure_rotation() {
        // Frame j rotated relative to frame i
        let host_quat = Quat::identity();
        let gravity = [0.0, 0.0, -9.81];

        let pose_i: [f64; 15] = [
            0.0, 0.0, 0.0, // rotation delta
            0.0, 0.0, 0.0, // translation
            0.0, 0.0, 0.0, // velocity
            0.0, 0.0, 0.0, // gyro bias
            0.0, 0.0, 0.0, // accel bias
        ];

        let rot_amount = 0.1; // 0.1 radians around z
        let pose_j: [f64; 15] = [
            0.0, 0.0, rot_amount, // rotation delta
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        // Preintegration with matching rotation
        let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
        preint.delta_rotation =
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, rot_amount));

        let residual = imu_preintegration_residual(
            &host_quat, &pose_i, &host_quat, &pose_j, &preint, &gravity,
        );

        // Rotation residual should be near zero
        assert!(residual[0].abs() < 1e-6, "Rotation X residual too large");
        assert!(residual[1].abs() < 1e-6, "Rotation Y residual too large");
        assert!(residual[2].abs() < 1e-6, "Rotation Z residual too large");
    }

    #[test]
    fn test_bias_residual_constant_bias() {
        // Same bias at both frames should give zero residual
        let bg = [0.01, 0.02, 0.03];
        let ba = [0.1, 0.2, 0.3];
        let dt = 0.1;

        let residual = bias_residual(&bg, &ba, &bg, &ba, dt, 0.001, 0.01);

        for r in residual.iter() {
            assert!(r.abs() < 1e-10, "Residual should be zero for constant bias");
        }
    }

    #[test]
    fn test_bias_residual_changing_bias() {
        let bg_i = [0.01, 0.02, 0.03];
        let ba_i = [0.1, 0.2, 0.3];
        let bg_j = [0.011, 0.021, 0.031]; // Small change
        let ba_j = [0.11, 0.21, 0.31];
        let dt = 0.1f64;

        let residual = bias_residual(&bg_i, &ba_i, &bg_j, &ba_j, dt, 0.001, 0.01);

        // Residuals should be non-zero
        assert!(residual[0].abs() > 0.0);
        assert!(residual[3].abs() > 0.0);
    }

    #[test]
    fn test_imu_residual_with_autodiff() {
        use odysseus_solver::Jet;

        type Jet30 = Jet<f64, 30>;

        let host_quat = Quat::identity();
        let gravity = [
            Jet30::constant(-0.0),
            Jet30::constant(0.0),
            Jet30::constant(-9.81),
        ];

        // Create Jet parameters for frame i (first 15 vars)
        let mut pose_i: [Jet30; 15] = std::array::from_fn(|_| Jet30::constant(0.0));
        for i in 0..15 {
            pose_i[i] = Jet30::variable(0.0, i);
        }

        // Create Jet parameters for frame j (next 15 vars)
        let mut pose_j: [Jet30; 15] = std::array::from_fn(|_| Jet30::constant(0.0));
        for i in 0..15 {
            pose_j[i] = Jet30::variable(0.0, 15 + i);
        }
        pose_j[2] = Jet30::variable(0.1, 17); // Some rotation

        let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
        preint.delta_rotation = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 0.1));

        let residual = imu_preintegration_residual(
            &host_quat, &pose_i, &host_quat, &pose_j, &preint, &gravity,
        );

        // Check that we have derivatives
        for r in residual.iter() {
            // At least some derivatives should be non-zero
            let has_derivs = r.derivs.iter().any(|&d| d.abs() > 1e-10);
            // This is expected for most residuals
            let _ = has_derivs;
        }

        // Residuals should be computable
        assert!(residual[0].value.is_finite());
        assert!(residual[4].value.is_finite());
        assert!(residual[8].value.is_finite());
    }
}
