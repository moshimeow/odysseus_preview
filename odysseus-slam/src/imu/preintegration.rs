//! IMU Preintegration
//!
//! Implements the on-manifold preintegration approach from:
//! Forster et al. "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
//! https://arxiv.org/abs/1512.02363
//!
//! Key idea: Preintegrate many high-rate IMU measurements into a single
//! "preintegrated measurement" that constrains consecutive keyframes.
//! This avoids re-integrating all IMU data when poses change during optimization.

use nalgebra::{Matrix3, SMatrix, Vector3};
use odysseus_solver::math3d::{Quat, Vec3 as SolverVec3};

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
    /// Rotation delta as unit quaternion
    pub delta_rotation: Quat<f64>,
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
            delta_rotation: Quat::identity(),
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

        // Save current rotation quaternion (needed for Jacobians and covariance)
        let delta_q_prev = self.delta_rotation;

        // Rotation increment for this step using midpoint angular velocity
        // ΔR_i = ΔR_{i-1} * Exp(omega_mid * dt)
        let delta_angle = gyro_mid * dt;
        let delta_angle_solver = SolverVec3::new(delta_angle.x, delta_angle.y, delta_angle.z);
        let delta_q_increment = Quat::from_axis_angle(delta_angle_solver);

        // Midpoint rotation for accelerating the velocity update
        // R_mid = R_{i-1} * Exp(omega_mid * dt/2)
        let half_angle_solver = SolverVec3::new(
            0.5 * delta_angle.x,
            0.5 * delta_angle.y,
            0.5 * delta_angle.z,
        );
        let delta_q_mid = delta_q_prev * Quat::from_axis_angle(half_angle_solver);

        // Rotate acceleration using midpoint quaternion
        let accel_mid_solver = SolverVec3::new(accel_mid.x, accel_mid.y, accel_mid.z);
        let rotated_accel_solver = delta_q_mid.rotate_vec(accel_mid_solver);
        let rotated_accel_mid = Vector3::new(
            rotated_accel_solver.x,
            rotated_accel_solver.y,
            rotated_accel_solver.z,
        );

        // Update position (using velocity at start and midpoint acceleration)
        self.delta_position += self.delta_velocity * dt + 0.5 * rotated_accel_mid * dt * dt;

        // Update velocity
        self.delta_velocity += rotated_accel_mid * dt;

        // Update rotation using quaternion composition and normalize
        self.delta_rotation = (delta_q_prev * delta_q_increment).normalize();

        // Update Jacobians w.r.t. biases using quaternions directly
        // These track how the preintegrated values change with bias using the
        // recursive formulation from Forster et al.
        self.update_jacobians(
            &delta_q_prev,
            &delta_q_increment,
            &gyro_unbiased,
            &accel_unbiased,
            dt,
        );

        // Update covariance using full state transition matrix
        self.propagate_covariance(
            &delta_q_prev,
            &delta_q_increment,
            &gyro_unbiased,
            &accel_unbiased,
            dt,
            gyro_noise_density,
            accel_noise_density,
        );

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
    /// The rotation is returned as a rotation vector (axis-angle) for tangent space operations.
    pub fn correct_for_bias(
        &self,
        gyro_bias: &Vector3<f64>,
        accel_bias: &Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let delta_bg = gyro_bias - self.gyro_bias_estimate;
        let delta_ba = accel_bias - self.accel_bias_estimate;

        // Convert quaternion to rotation vector for tangent space correction
        let delta_rot_vec = self.delta_rotation.to_axis_angle();
        let delta_rotation_rvec = Vector3::new(delta_rot_vec.x, delta_rot_vec.y, delta_rot_vec.z);

        // First-order correction: Δx_corrected = Δx + J * δb
        let delta_rotation_corrected = delta_rotation_rvec + self.d_rotation_d_gyro_bias * delta_bg;

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
        delta_q: &Quat<f64>,          // ΔR_k as quaternion (rotation from i to k)
        delta_q_step: &Quat<f64>,     // Step increment quaternion
        gyro_unbiased: &Vector3<f64>, // ω_k (needed for right Jacobian)
        accel_unbiased: &Vector3<f64>, // a_k
        dt: f64,
    ) {
        let gyro_angle = gyro_unbiased * dt;
        let jr = right_jacobian(&gyro_angle);

        // Save previous Jacobians to use in velocity/position updates
        // This ensures recursive consistency as defined in Section V-B of the TRO paper
        let prev_dr_dbg = self.d_rotation_d_gyro_bias;
        let prev_dv_dbg = self.d_velocity_d_gyro_bias;
        let prev_dv_dba = self.d_velocity_d_accel_bias;

        // Jacobian of rotation w.r.t. gyro bias
        // J_R(k+1) = ΔR(step)^T * J_R(k) - J_r(step) * dt
        // Use quaternion conjugate (inverse) to rotate
        let dr_step_inv_times_jr = delta_q_step
            .conjugate()
            .rotate_nalgebra_matrix(&prev_dr_dbg);
        self.d_rotation_d_gyro_bias = dr_step_inv_times_jr - jr * dt;

        let accel_skew = skew_symmetric(accel_unbiased);
        // Rotate the skew-symmetric matrix by delta_q
        let rot_accel_skew = delta_q.rotate_nalgebra_matrix(&accel_skew);

        // Jacobian of velocity w.r.t. gyro bias
        // J_v(k+1) = J_v(k) - ΔR_k * [a_k]_x * J_R(k) * dt
        self.d_velocity_d_gyro_bias -= rot_accel_skew * prev_dr_dbg * dt;

        // Jacobian of velocity w.r.t. accel bias
        // J_v(k+1) = J_v(k) - ΔR_k * dt
        // For this term, we need the rotation matrix as a Jacobian contribution
        let delta_r_matrix = delta_q.rotate_nalgebra_matrix(&Matrix3::identity());
        self.d_velocity_d_accel_bias -= delta_r_matrix * dt;

        // Jacobian of position w.r.t. gyro bias
        // J_p(k+1) = J_p(k) + J_v(k)*dt - 1/2 * ΔR_k * [a_k]_x * J_R(k) * dt^2
        self.d_position_d_gyro_bias +=
            prev_dv_dbg * dt - 0.5 * rot_accel_skew * prev_dr_dbg * dt * dt;

        // Jacobian of position w.r.t. accel bias
        // J_p(k+1) = J_p(k) + J_v(k)*dt - 1/2 * ΔR_k * dt^2
        self.d_position_d_accel_bias += prev_dv_dba * dt - 0.5 * delta_r_matrix * dt * dt;
    }

    /// Propagate covariance during integration using the full state transition matrix
    fn propagate_covariance(
        &mut self,
        delta_q: &Quat<f64>,
        delta_q_step: &Quat<f64>,
        gyro_unbiased: &Vector3<f64>,
        accel_unbiased: &Vector3<f64>,
        dt: f64,
        gyro_noise_density: f64,
        accel_noise_density: f64,
    ) {
        if dt <= 0.0 {
            return;
        }

        let mut f = Matrix9::zeros();

        // State transition matrix F (9x9) from Forster et al. , Eq. (39-40)
        // [ DeltaR_step^T,             0,    0 ]
        // [ -R_k * [a]_x * dt,         I,    0 ]
        // [ -1/2 * R_k * [a]_x * dt^2, I*dt, I ]

        // Rotation block (0:3, 0:3)
        let dr_step_inv = delta_q_step
            .conjugate()
            .rotate_nalgebra_matrix(&Matrix3::identity());
        f.fixed_view_mut::<3, 3>(0, 0).copy_from(&dr_step_inv);

        // Velocity blocks
        f.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&Matrix3::identity());

        let accel_skew = skew_symmetric(accel_unbiased);
        let rot_accel_skew = delta_q.rotate_nalgebra_matrix(&accel_skew);
        f.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(-rot_accel_skew * dt));

        // Position blocks
        f.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&Matrix3::identity());
        f.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(Matrix3::identity() * dt));
        f.fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-0.5 * rot_accel_skew * dt * dt));

        // Noise input matrix G (9x6)
        // [ Jr * dt, 0 ]
        // [ 0,       R_k * dt ]
        // [ 0,       1/2 * R_k * dt^2 ]
        let mut g = SMatrix::<f64, 9, 6>::zeros();
        let jr_dt = right_jacobian(&(gyro_unbiased * dt)) * dt;
        let r_mat = delta_q.rotate_nalgebra_matrix(&Matrix3::identity());
        let r_dt = r_mat * dt;
        let r_dt2 = r_mat * (0.5 * dt * dt);

        g.fixed_view_mut::<3, 3>(0, 0).copy_from(&jr_dt);
        g.fixed_view_mut::<3, 3>(3, 3).copy_from(&r_dt);
        g.fixed_view_mut::<3, 3>(6, 3).copy_from(&r_dt2);

        // Discrete noise covariance Q_d (6x6)
        // Assuming white noise density (unit/√Hz)
        let mut q_d = SMatrix::<f64, 6, 6>::zeros();
        let g2 = gyro_noise_density * gyro_noise_density / dt;
        let a2 = accel_noise_density * accel_noise_density / dt;

        for i in 0..3 {
            q_d[(i, i)] = g2;
            q_d[(i + 3, i + 3)] = a2;
        }

        // Propagate: Sigma = F * Sigma * F^T + G * Q_d * G^T
        self.covariance = f * self.covariance * f.transpose() + g * q_d * g.transpose();
    }
}

/// Extension trait to add nalgebra Matrix3 rotation to Quat
trait QuatNalgebraExt {
    /// Rotate a nalgebra Matrix3 by this quaternion (computes R * M)
    fn rotate_nalgebra_matrix(&self, m: &Matrix3<f64>) -> Matrix3<f64>;
}

impl QuatNalgebraExt for Quat<f64> {
    fn rotate_nalgebra_matrix(&self, m: &Matrix3<f64>) -> Matrix3<f64> {
        // Rotate each column of the matrix
        let col0 = SolverVec3::new(m[(0, 0)], m[(1, 0)], m[(2, 0)]);
        let col1 = SolverVec3::new(m[(0, 1)], m[(1, 1)], m[(2, 1)]);
        let col2 = SolverVec3::new(m[(0, 2)], m[(1, 2)], m[(2, 2)]);

        let r0 = self.rotate_vec(col0);
        let r1 = self.rotate_vec(col1);
        let r2 = self.rotate_vec(col2);

        Matrix3::new(r0.x, r1.x, r2.x, r0.y, r1.y, r2.y, r0.z, r1.z, r2.z)
    }
}

/// Create skew-symmetric matrix from vector
fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Compute the right Jacobian of SO(3)
///
/// Jr(phi) relates the change in rotation vector to the change in tangent space
/// such that Exp(phi + dphi) \approx Exp(phi) * Exp(Jr(phi) * dphi)
fn right_jacobian(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta2 = phi.norm_squared();
    let theta = theta2.sqrt();
    let k = skew_symmetric(phi);
    let k2 = k * k;

    let (a, b) = if theta < 1e-4 {
        // Taylor expansion for small angles to avoid subtractive cancellation:
        // a = (1 - cos(theta)) / theta^2 = 1/2 - theta^2/24 + theta^4/720
        // b = (theta - sin(theta)) / theta^3 = 1/6 - theta^2/120 + theta^4/5040
        let theta4 = theta2 * theta2;
        let a = 0.5 - theta2 / 24.0 + theta4 / 720.0;
        let b = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0;
        (a, b)
    } else {
        // Use stable formula for a to avoid subtractive cancellation
        // 1 - cos(theta) = 2 * sin^2(theta/2)
        let half_theta = 0.5 * theta;
        let sin_half = half_theta.sin();
        let a = (2.0 * sin_half * sin_half) / theta2;
        let b = (theta - theta.sin()) / (theta2 * theta);
        (a, b)
    };

    Matrix3::identity() - a * k + b * k2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preintegration_identity() {
        let preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
        // delta_rotation should be identity quaternion
        let id = Quat::<f64>::identity();
        assert!((preint.delta_rotation.w - id.w).abs() < 1e-10);
        assert!((preint.delta_rotation.x - id.x).abs() < 1e-10);
        assert!((preint.delta_rotation.y - id.y).abs() < 1e-10);
        assert!((preint.delta_rotation.z - id.z).abs() < 1e-10);
        assert_eq!(preint.delta_velocity, Vector3::zeros());
        assert_eq!(preint.delta_position, Vector3::zeros());
        assert_eq!(preint.delta_time, 0.0);
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

        // Should remain near identity/zero (only numerical drift)
        // Check quaternion is close to identity: (1, 0, 0, 0)
        let rot_vec = preint.delta_rotation.to_axis_angle();
        assert!(rot_vec.length() < 1e-10, "Rotation not near identity");
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
        // dr is now returned as rotation vector, compare with quaternion's rotation vector
        let delta_rot_vec = preint.delta_rotation.to_axis_angle();
        let delta_rot_nalgebra = Vector3::new(delta_rot_vec.x, delta_rot_vec.y, delta_rot_vec.z);
        assert!((dr - delta_rot_nalgebra).norm() < 1e-10);
        assert!((dv - preint.delta_velocity).norm() < 1e-10);
        assert!((dp - preint.delta_position).norm() < 1e-10);

        // Correction with different bias should give different values
        let new_gyro_bias = gyro_bias + Vector3::new(0.001, 0.001, 0.001);
        let (dr2, _, _) = preint.correct_for_bias(&new_gyro_bias, &accel_bias);
        assert!((dr2 - delta_rot_nalgebra).norm() > 1e-10);
    }

    #[test]
    fn test_right_jacobian_stability() {
        // Case 1: Zero rotation
        let phi_zero = Vector3::zeros();
        let jr_zero = right_jacobian(&phi_zero);
        assert!((jr_zero - Matrix3::identity()).norm() < 1e-15);

        // Case 2: Very small rotation (well below threshold)
        let phi_tiny = Vector3::new(1e-10, 0.0, 0.0);
        let jr_tiny = right_jacobian(&phi_tiny);
        // Should be approx I - 0.5*[phi]_x
        let expected_tiny = Matrix3::identity() - 0.5 * skew_symmetric(&phi_tiny);
        assert!((jr_tiny - expected_tiny).norm() < 1e-15);

        // Case 3: Just below threshold (1e-4 - eps)
        let eps = 1e-12;
        let phi_below = Vector3::new(1e-4 - eps, 0.0, 0.0);
        let jr_below = right_jacobian(&phi_below);

        // Case 4: Just above threshold (1e-4 + eps)
        let phi_above = Vector3::new(1e-4 + eps, 0.0, 0.0);
        let jr_above = right_jacobian(&phi_above);

        // Continuity check: the difference between just below and just above should be very small.
        // Since the inputs differ by 2*eps, we expect a difference of ~eps * ||dJr/dphi||.
        // ||dJr/dphi|| is roughly 0.5, so difference is ~eps.
        assert!(
            (jr_below - jr_above).norm() < 2e-12,
            "Discontinuity at threshold: norm diff = {}",
            (jr_below - jr_above).norm()
        );

        // Property check: Jr(phi) * phi should be equal to phi
        let phi_large = Vector3::new(0.5, 0.2, -0.3);
        let jr_large = right_jacobian(&phi_large);
        let res = jr_large * phi_large;
        assert!((res - phi_large).norm() < 1e-12);

        // Check a known value: for phi = [pi/2, 0, 0]
        // Jr = [ 1,            0,           0 ]
        //      [ 0,  sin(p)/p,  (1-cos(p))/p ]
        //      [ 0, -(1-cos(p))/p,  sin(p)/p ]
        // where p = pi/2
        let p = std::f64::consts::FRAC_PI_2;
        let phi_pi2 = Vector3::new(p, 0.0, 0.0);
        let jr_pi2 = right_jacobian(&phi_pi2);
        let s = p.sin() / p;
        let c = (1.0 - p.cos()) / p;
        let expected_pi2 = Matrix3::new(1.0, 0.0, 0.0, 0.0, s, c, 0.0, -c, s);
        assert!((jr_pi2 - expected_pi2).norm() < 1e-12);
    }
}
