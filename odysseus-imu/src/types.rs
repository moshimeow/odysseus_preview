//! IMU data types
//!
//! Uses nalgebra types since these structures don't need autodiff support.

use nalgebra::Vector3;

/// A single IMU measurement (gyroscope + accelerometer)
#[derive(Debug, Clone, Copy)]
pub struct ImuMeasurement {
    /// Timestamp in seconds
    pub timestamp: f64,
    /// Angular velocity from gyroscope (rad/s) in body frame
    pub gyro: Vector3<f64>,
    /// Linear acceleration from accelerometer (m/s²) in body frame
    /// Note: Includes gravity! A stationary IMU measures +g upward.
    pub accel: Vector3<f64>,
}

impl ImuMeasurement {
    /// Create a new IMU measurement
    pub fn new(timestamp: f64, gyro: Vector3<f64>, accel: Vector3<f64>) -> Self {
        Self {
            timestamp,
            gyro,
            accel,
        }
    }
}

/// IMU-specific state for a frame
///
/// This is stored separately from Pose to keep visual SLAM components
/// (including GBA) unchanged. Only used by VIO optimization.
#[derive(Debug, Clone)]
pub struct ImuFrameState {
    /// Velocity in world frame (m/s)
    pub velocity: Vector3<f64>,
    /// Gyroscope bias (rad/s)
    pub gyro_bias: Vector3<f64>,
    /// Accelerometer bias (m/s²)
    pub accel_bias: Vector3<f64>,
}

impl ImuFrameState {
    /// Create state with zero velocity and biases
    pub fn zero() -> Self {
        Self {
            velocity: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            accel_bias: Vector3::zeros(),
        }
    }

    /// Create state with given velocity and zero biases
    pub fn with_velocity(velocity: Vector3<f64>) -> Self {
        Self {
            velocity,
            gyro_bias: Vector3::zeros(),
            accel_bias: Vector3::zeros(),
        }
    }

    /// Create state with all values specified
    pub fn new(
        velocity: Vector3<f64>,
        gyro_bias: Vector3<f64>,
        accel_bias: Vector3<f64>,
    ) -> Self {
        Self {
            velocity,
            gyro_bias,
            accel_bias,
        }
    }
}

impl Default for ImuFrameState {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imu_measurement_creation() {
        let meas = ImuMeasurement::new(
            0.5,
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.0, 0.0, 9.81),
        );
        assert_eq!(meas.timestamp, 0.5);
        assert_eq!(meas.gyro.x, 0.1);
        assert_eq!(meas.accel.z, 9.81);
    }

    #[test]
    fn test_imu_frame_state_zero() {
        let state = ImuFrameState::zero();
        assert_eq!(state.velocity, Vector3::zeros());
        assert_eq!(state.gyro_bias, Vector3::zeros());
        assert_eq!(state.accel_bias, Vector3::zeros());
    }

    #[test]
    fn test_imu_frame_state_with_velocity() {
        let vel = Vector3::new(1.0, 2.0, 3.0);
        let state = ImuFrameState::with_velocity(vel);
        assert_eq!(state.velocity, vel);
        assert_eq!(state.gyro_bias, Vector3::zeros());
    }
}
