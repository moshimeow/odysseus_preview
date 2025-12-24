//! IMU Preintegration
//!
//! Implements the on-manifold preintegration approach from:
//! Forster et al. "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
//!
//! Key idea: Preintegrate many high-rate IMU measurements into a single
//! "preintegrated measurement" that constrains consecutive keyframes.
//! This avoids re-integrating all IMU data when poses change during optimization.

use nalgebra::{Matrix3, SMatrix, Vector3};

/// 9x9 matrix type for covariance (rotation, velocity, position)
pub type Matrix9<T> = SMatrix<T, 9, 9>;
use super::types::ImuMeasurement;

/// Preintegrated IMU measurements between two frames
///
/// Contains the preintegrated rotation, velocity, and position deltas,
/// along with Jacobians for bias correction.
#[derive(Debug, Clone)]
pub struct PreintegratedImu {
    // Preintegrated measurements (in body frame of start pose)
    /// Rotation delta as rotation vector (NOT quaternion - can represent >180°)
    pub delta_rotation: Vector3<f64>,
    /// Velocity delta in start body frame
    pub delta_velocity: Vector3<f64>,
    /// Position delta in start body frame
    pub delta_position: Vector3<f64>,
    /// Total integration time
    pub delta_time: f64,

    // Jacobians w.r.t. biases (for first-order bias correction)
    /// d(delta_rotation) / d(gyro_bias)
    pub d_rotation_d_gyro_bias: Matrix3<f64>,
    /// d(delta_velocity) / d(gyro_bias)
    pub d_velocity_d_gyro_bias: Matrix3<f64>,
    /// d(delta_velocity) / d(accel_bias)
    pub d_velocity_d_accel_bias: Matrix3<f64>,
    /// d(delta_position) / d(gyro_bias)
    pub d_position_d_gyro_bias: Matrix3<f64>,
    /// d(delta_position) / d(accel_bias)
    pub d_position_d_accel_bias: Matrix3<f64>,

    /// Covariance of preintegrated measurements (9x9: rotation, velocity, position)
    pub covariance: Matrix9<f64>,

    // Biases used during preintegration (for computing bias correction)
    /// Gyroscope bias estimate used during preintegration
    pub gyro_bias_estimate: Vector3<f64>,
    /// Accelerometer bias estimate used during preintegration
    pub accel_bias_estimate: Vector3<f64>,

    /// Last measurement integrated (for midpoint integration)
    pub last_measurement: Option<ImuMeasurement>,
}

impl PreintegratedImu {
    /// Create a new preintegrated measurement initialized to identity/zero
    pub fn new(gyro_bias: Vector3<f64>, accel_bias: Vector3<f64>) -> Self {
        Self {
            delta_rotation: Vector3::zeros(),
            delta_velocity: Vector3::zeros(),
            delta_position: Vector3::zeros(),
            delta_time: 0.0,

            d_rotation_d_gyro_bias: Matrix3::zeros(),
            d_velocity_d_gyro_bias: Matrix3::zeros(),
            d_velocity_d_accel_bias: Matrix3::zeros(),
            d_position_d_gyro_bias: Matrix3::zeros(),
            d_position_d_accel_bias: Matrix3::zeros(),

            covariance: Matrix9::zeros(),

            gyro_bias_estimate: gyro_bias,
            accel_bias_estimate: accel_bias,
            last_measurement: None,
        }
    }

    /// Integrate a single IMU measurement
    ///
    /// # Arguments
    /// * `measurement` - The IMU measurement to integrate
    /// * `dt` - Time delta since last measurement (seconds)
    /// * `gyro_noise_density` - Gyroscope noise density (rad/s/√Hz)
    /// * `accel_noise_density` - Accelerometer noise density (m/s²/√Hz)
    pub fn integrate(
        &mut self,
        measurement: &ImuMeasurement,
        dt: f64,
        gyro_noise_density: f64,
        accel_noise_density: f64,
    ) {
        // Subtract bias estimates from measurements
        let gyro_unbiased = measurement.gyro - self.gyro_bias_estimate;
        let accel_unbiased = measurement.accel - self.accel_bias_estimate;

        // Since the simulator provides the average over the interval [t_i-1, t_i],
        // these unbiased values already represent the midpoint values.
        let gyro_mid = gyro_unbiased;
        let accel_mid = accel_unbiased;

        // Current rotation (at the start of the interval)
        let delta_r_matrix = rotation_vector_to_matrix(&self.delta_rotation);

        // Rotation increment for this step using midpoint angular velocity
        // ΔR_i = ΔR_{i-1} * Exp(omega_mid * dt)
        let delta_angle = gyro_mid * dt;
        let delta_r_increment = rotation_vector_to_matrix(&delta_angle);

        // Midpoint rotation for accelerating the velocity update
        // R_mid = R_{i-1} * Exp(omega_mid * dt/2)
        let delta_r_mid = delta_r_matrix * rotation_vector_to_matrix(&(0.5 * delta_angle));

        // Update position (using velocity at start and midpoint acceleration)
        let rotated_accel_mid = delta_r_mid * accel_mid;
        self.delta_position += self.delta_velocity * dt + 0.5 * rotated_accel_mid * dt * dt;

        // Update velocity
        self.delta_velocity += rotated_accel_mid * dt;

        // Update rotation
        let new_r_matrix = delta_r_matrix * delta_r_increment;
        self.delta_rotation = rotation_matrix_to_vector(&new_r_matrix);

        // Update Jacobians w.r.t. biases
        // (Simplification: using Euler-like updates for Jacobians for now,
        // but rotating them correctly)
        self.update_jacobians(&delta_r_matrix, &accel_unbiased, dt);

        // Update covariance
        self.propagate_covariance(&delta_r_matrix, dt, gyro_noise_density, accel_noise_density);

        self.delta_time += dt;
        self.last_measurement = Some(measurement.clone());
    }

    /// Integrate a sequence of IMU measurements
    pub fn integrate_measurements(
        &mut self,
        measurements: &[ImuMeasurement],
        gyro_noise_density: f64,
        accel_noise_density: f64,
    ) {
        if measurements.is_empty() {
            return;
        }

        // Handle the first measurement separately if we need an initial dt
        // Usually preintegration starts from a keyframe's time.
        // If we don't have a previous time, we can't integrate the first measurement's interval.
        // For the purpose of the demo, we assume the interval from t_start to t[0] is small
        // or that t[0] is the start.

        let mut prev_time = if let Some(last_m) = &self.last_measurement {
            last_m.timestamp
        } else {
            // Assume the first measurement interval is represented by its own timestamp
            // relative to 0.0 or just start from the first one.
            measurements[0].timestamp
        };

        let start_idx = if self.last_measurement.is_none() {
            1
        } else {
            0
        };

        for i in start_idx..measurements.len() {
            let dt = measurements[i].timestamp - prev_time;
            if dt > 0.0 {
                self.integrate(
                    &measurements[i],
                    dt,
                    gyro_noise_density,
                    accel_noise_density,
                );
            }
            prev_time = measurements[i].timestamp;
        }
    }

    /// Apply first-order bias correction
    ///
    /// When biases change during optimization, this provides corrected
    /// preintegrated values without re-integrating.
    ///
    /// # Arguments
    /// * `gyro_bias` - Current gyroscope bias estimate
    /// * `accel_bias` - Current accelerometer bias estimate
    ///
    /// # Returns
    /// Corrected (delta_rotation, delta_velocity, delta_position)
    pub fn correct_for_bias(
        &self,
        gyro_bias: &Vector3<f64>,
        accel_bias: &Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let delta_bg = gyro_bias - self.gyro_bias_estimate;
        let delta_ba = accel_bias - self.accel_bias_estimate;

        // First-order correction: Δx_corrected = Δx + J * δb
        let delta_rotation_corrected = self.delta_rotation + self.d_rotation_d_gyro_bias * delta_bg;

        let delta_velocity_corrected = self.delta_velocity
            + self.d_velocity_d_gyro_bias * delta_bg
            + self.d_velocity_d_accel_bias * delta_ba;

        let delta_position_corrected = self.delta_position
            + self.d_position_d_gyro_bias * delta_bg
            + self.d_position_d_accel_bias * delta_ba;

        (
            delta_rotation_corrected,
            delta_velocity_corrected,
            delta_position_corrected,
        )
    }

    /// Update Jacobians w.r.t. biases during integration
    fn update_jacobians(
        &mut self,
        delta_r_matrix: &Matrix3<f64>,
        accel_unbiased: &Vector3<f64>,
        dt: f64,
    ) {
        // Jacobian of rotation w.r.t. gyro bias
        // d(ΔR)/d(bg) accumulates: -ΔR * Jr * dt
        // where Jr is the right Jacobian of SO3
        // For small angles, Jr ≈ I, so we use: d(ΔR)/d(bg) -= ΔR * dt
        self.d_rotation_d_gyro_bias -= delta_r_matrix * dt;

        // Jacobian of velocity w.r.t. gyro bias
        // d(Δv)/d(bg) += -ΔR * [a]_× * d(ΔR)/d(bg) * dt
        let accel_skew = skew_symmetric(accel_unbiased);
        self.d_velocity_d_gyro_bias -=
            delta_r_matrix * accel_skew * self.d_rotation_d_gyro_bias * dt;

        // Jacobian of velocity w.r.t. accel bias
        // d(Δv)/d(ba) -= ΔR * dt
        self.d_velocity_d_accel_bias -= delta_r_matrix * dt;

        // Jacobian of position w.r.t. gyro bias
        self.d_position_d_gyro_bias += self.d_velocity_d_gyro_bias * dt
            - 0.5 * delta_r_matrix * accel_skew * self.d_rotation_d_gyro_bias * dt * dt;

        // Jacobian of position w.r.t. accel bias
        self.d_position_d_accel_bias +=
            self.d_velocity_d_accel_bias * dt - 0.5 * delta_r_matrix * dt * dt;
    }

    /// Propagate covariance during integration
    fn propagate_covariance(
        &mut self,
        delta_r_matrix: &Matrix3<f64>,
        dt: f64,
        gyro_noise_density: f64,
        accel_noise_density: f64,
    ) {
        // Discrete-time noise covariance
        let gyro_cov = gyro_noise_density * gyro_noise_density / dt;
        let accel_cov = accel_noise_density * accel_noise_density / dt;

        // Noise covariance matrix Q (6x6: gyro, accel)
        // For now, simplified diagonal propagation
        // Full implementation would use the state transition matrix

        // Add noise contribution to covariance
        // Rotation block (0:3, 0:3)
        for i in 0..3 {
            self.covariance[(i, i)] += gyro_cov * dt * dt;
        }

        // Velocity block (3:6, 3:6)
        for i in 3..6 {
            self.covariance[(i, i)] += accel_cov * dt * dt;
        }

        // Position block (6:9, 6:9)
        for i in 6..9 {
            self.covariance[(i, i)] += accel_cov * dt * dt * dt * dt / 4.0;
        }

        // Cross-correlations would require full state transition matrix F
        // This is a simplified version; full implementation follows Forster et al.
        let _ = delta_r_matrix; // Suppress unused warning for now
    }
}

/// Convert rotation vector to rotation matrix using Rodrigues formula
fn rotation_vector_to_matrix(rvec: &Vector3<f64>) -> Matrix3<f64> {
    let theta_sq = rvec.norm_squared();
    let theta = theta_sq.sqrt();

    if theta < 1e-10 {
        // Small angle approximation: R ≈ I + [ω]_×
        Matrix3::identity() + skew_symmetric(rvec)
    } else {
        // Rodrigues formula: R = I + sin(θ)/θ * [ω]_× + (1-cos(θ))/θ² * [ω]_×²
        let k = rvec / theta;
        let k_skew = skew_symmetric(&k);
        let k_skew_sq = k_skew * k_skew;

        Matrix3::identity() + theta.sin() * k_skew + (1.0 - theta.cos()) * k_skew_sq
    }
}

/*
/// Compose two rotation vectors (approximate for small angles, exact via quaternion for large)
fn compose_rotation_vectors(r1: &Vector3<f64>, r2: &Vector3<f64>) -> Vector3<f64> {
    // For very small r2, simple addition is accurate enough
    let theta2 = r2.norm();
    if theta2 < 1e-10 {
        return r1 + r2;
    }

    // For larger rotations, compose via rotation matrices and extract
    let r1_mat = rotation_vector_to_matrix(r1);
    let r2_mat = rotation_vector_to_matrix(r2);
    let composed_mat = r1_mat * r2_mat;

    rotation_matrix_to_vector(&composed_mat)
}
*/

/// Convert rotation matrix to rotation vector (logarithm map)
fn rotation_matrix_to_vector(r: &Matrix3<f64>) -> Vector3<f64> {
    // Use Rodrigues formula inverse
    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let cos_theta = (trace - 1.0) / 2.0;
    let cos_theta_clamped = cos_theta.clamp(-1.0, 1.0);
    let theta = cos_theta_clamped.acos();

    if theta < 1e-10 {
        // Small angle: extract from skew-symmetric part
        Vector3::new(
            (r[(2, 1)] - r[(1, 2)]) / 2.0,
            (r[(0, 2)] - r[(2, 0)]) / 2.0,
            (r[(1, 0)] - r[(0, 1)]) / 2.0,
        )
    } else if (std::f64::consts::PI - theta).abs() < 1e-10 {
        // Near 180 degrees: special handling
        // Find the column with largest diagonal element
        let diag = Vector3::new(r[(0, 0)], r[(1, 1)], r[(2, 2)]);
        let max_idx = if diag.x >= diag.y && diag.x >= diag.z {
            0
        } else if diag.y >= diag.z {
            1
        } else {
            2
        };

        let mut v = Vector3::zeros();
        v[max_idx] = (1.0 + r[(max_idx, max_idx)]).sqrt() / 2.0;
        let denom = 2.0 * v[max_idx];
        for i in 0..3 {
            if i != max_idx {
                v[i] = r[(i, max_idx)] / denom;
            }
        }
        v * theta
    } else {
        // General case
        let k = Vector3::new(
            r[(2, 1)] - r[(1, 2)],
            r[(0, 2)] - r[(2, 0)],
            r[(1, 0)] - r[(0, 1)],
        );
        k * (theta / (2.0 * theta.sin()))
    }
}

/// Create skew-symmetric matrix from vector
fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preintegration_identity() {
        let preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
        assert_eq!(preint.delta_rotation, Vector3::zeros());
        assert_eq!(preint.delta_velocity, Vector3::zeros());
        assert_eq!(preint.delta_position, Vector3::zeros());
        assert_eq!(preint.delta_time, 0.0);
    }

    #[test]
    fn test_rotation_vector_to_matrix_identity() {
        let rvec = Vector3::zeros();
        let mat = rotation_vector_to_matrix(&rvec);
        assert!((mat - Matrix3::identity()).norm() < 1e-10);
    }

    #[test]
    fn test_rotation_vector_roundtrip() {
        let rvec = Vector3::new(0.1, 0.2, 0.3);
        let mat = rotation_vector_to_matrix(&rvec);
        let rvec_recovered = rotation_matrix_to_vector(&mat);

        assert!((rvec - rvec_recovered).norm() < 1e-6);
    }

    #[test]
    fn test_rotation_vector_large_angle() {
        // Test rotation > 180 degrees (this is where quaternions would fail)
        let rvec = Vector3::new(0.0, 0.0, 4.0); // ~229 degrees around Z
        let mat = rotation_vector_to_matrix(&rvec);
        let rvec_recovered = rotation_matrix_to_vector(&mat);

        // The recovered rotation might be the equivalent rotation in [-π, π]
        // Check that they produce the same rotation
        let mat_recovered = rotation_vector_to_matrix(&rvec_recovered);
        assert!((mat - mat_recovered).norm() < 1e-6);
    }

    #[test]
    fn test_skew_symmetric() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let skew = skew_symmetric(&v);

        // Should be antisymmetric
        assert!((skew + skew.transpose()).norm() < 1e-10);

        // v × u = skew(v) * u
        let u = Vector3::new(4.0, 5.0, 6.0);
        let cross = v.cross(&u);
        let skew_result = skew * u;
        assert!((cross - skew_result).norm() < 1e-10);
    }

    #[test]
    fn test_stationary_integration() {
        // Stationary IMU: zero gyro, gravity-compensated accel = 0
        let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());

        let measurement = ImuMeasurement::new(
            0.0,
            Vector3::zeros(),
            Vector3::zeros(), // Already gravity-compensated
        );

        for i in 1..100 {
            let mut m = measurement;
            m.timestamp = i as f64 * 0.01;
            preint.integrate(&m, 0.01, 0.01, 0.1);
        }

        // Should remain near zero (only numerical drift)
        assert!(preint.delta_rotation.norm() < 1e-10);
        assert!(preint.delta_velocity.norm() < 1e-6);
        assert!(preint.delta_position.norm() < 1e-6);
    }

    #[test]
    fn test_constant_acceleration() {
        // Constant acceleration of 1 m/s² in x direction
        let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());

        let dt = 0.01;
        let num_steps = 100;

        for i in 0..num_steps {
            let measurement =
                ImuMeasurement::new(i as f64 * dt, Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0));
            preint.integrate(&measurement, dt, 0.0, 0.0);
        }

        let total_time = num_steps as f64 * dt;

        // v = a * t
        let expected_velocity = total_time;
        // p = 0.5 * a * t²
        let expected_position = 0.5 * total_time * total_time;

        assert!((preint.delta_velocity.x - expected_velocity).abs() < 0.01);
        assert!((preint.delta_position.x - expected_position).abs() < 0.01);
    }

    #[test]
    fn test_bias_correction() {
        let gyro_bias = Vector3::new(0.01, 0.02, 0.03);
        let accel_bias = Vector3::new(0.1, 0.2, 0.3);

        let mut preint = PreintegratedImu::new(gyro_bias, accel_bias);

        // Integrate some measurements
        for i in 0..10 {
            let m = ImuMeasurement::new(
                i as f64 * 0.01,
                Vector3::new(0.1, 0.1, 0.1) + gyro_bias,
                Vector3::new(1.0, 0.0, 0.0) + accel_bias,
            );
            preint.integrate(&m, 0.01, 0.01, 0.1);
        }

        // Correction with same bias should give same values
        let (dr, dv, dp) = preint.correct_for_bias(&gyro_bias, &accel_bias);
        assert!((dr - preint.delta_rotation).norm() < 1e-10);
        assert!((dv - preint.delta_velocity).norm() < 1e-10);
        assert!((dp - preint.delta_position).norm() < 1e-10);

        // Correction with different bias should give different values
        let new_gyro_bias = gyro_bias + Vector3::new(0.001, 0.001, 0.001);
        let (dr2, _, _) = preint.correct_for_bias(&new_gyro_bias, &accel_bias);
        assert!((dr2 - preint.delta_rotation).norm() > 1e-10);
    }
}
