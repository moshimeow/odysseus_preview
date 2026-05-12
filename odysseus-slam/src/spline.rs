use crate::math::{SE3, SO3};
use nalgebra as na;
use odysseus_solver::math3d::{Quat, Vec3};
pub type Vector3<T> = Vec3<T>;
pub use crate::trajectory::ContinuousTrajectory;
use std::fs::File;
use std::io::{Read, Result};

#[derive(Debug, Clone)]
pub struct BezierKeyframe {
    pub time: f64,
    pub value: f64,
    pub handle_left: (f64, f64),
    pub handle_right: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct BezierCurve {
    pub keyframes: Vec<BezierKeyframe>,
}

#[derive(Debug, Clone)]
pub enum RotationMode {
    EulerXYZ = 0,
    Quaternion = 1,
    EulerXZY = 2,
    EulerYXZ = 3,
    EulerYZX = 4,
    EulerZXY = 5,
    EulerZYX = 6,
}

pub struct BezierSplineTrajectory {
    pub position_curves: [BezierCurve; 3], // X, Y, Z in OpenCV space
    pub rotation_curves: Vec<BezierCurve>, // Original Blender curves
    pub rotation_mode: RotationMode,
    pub fps: f64,
    /// Absolute time (seconds) of the first keyframe.
    pub start_time: f64,
    /// Absolute time (seconds) of the last keyframe.
    pub duration: f64,
}

impl BezierCurve {
    /// Evaluate the cubic Bezier curve at a given time.
    pub fn evaluate(&self, t: f64) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if t <= self.keyframes[0].time {
            return self.keyframes[0].value;
        }
        if t >= self.keyframes.last().unwrap().time {
            return self.keyframes.last().unwrap().value;
        }

        let idx = self
            .keyframes
            .binary_search_by(|k| k.time.partial_cmp(&t).unwrap())
            .unwrap_or_else(|e| e - 1);
        let k0 = &self.keyframes[idx];
        let k1 = &self.keyframes[idx + 1];

        let dt = k1.time - k0.time;
        if dt < 1e-9 {
            return k0.value;
        }

        let u = (t - k0.time) / dt;
        let p0 = k0.value;
        let p1 = k0.handle_right.1;
        let p2 = k1.handle_left.1;
        let p3 = k1.value;

        // Cubic Bezier: B(u) = (1-u)³P₀ + 3(1-u)²uP₁ + 3(1-u)u²P₂ + u³P₃
        let v = (1.0 - u).powi(3) * p0
            + 3.0 * (1.0 - u).powi(2) * u * p1
            + 3.0 * (1.0 - u) * u.powi(2) * p2
            + u.powi(3) * p3;
        v
    }

    /// Analytical derivative B'(t) of the Bezier curve.
    pub fn derivative(&self, t: f64) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if t <= self.keyframes[0].time || t >= self.keyframes.last().unwrap().time {
            return 0.0;
        }

        let idx = self
            .keyframes
            .binary_search_by(|k| k.time.partial_cmp(&t).unwrap())
            .unwrap_or_else(|e| e - 1);
        let k0 = &self.keyframes[idx];
        let k1 = &self.keyframes[idx + 1];

        let dt = k1.time - k0.time;
        if dt < 1e-9 {
            return 0.0;
        }

        let u = (t - k0.time) / dt;
        let p0 = k0.value;
        let p1 = k0.handle_right.1;
        let p2 = k1.handle_left.1;
        let p3 = k1.value;

        // B'(u) = 3(1-u)²(P₁-P₀) + 6(1-u)u(P₂-P₁) + 3u²(P₃-P₂)
        let du = 3.0 * (1.0 - u).powi(2) * (p1 - p0)
            + 6.0 * (1.0 - u) * u * (p2 - p1)
            + 3.0 * u.powi(2) * (p3 - p2);

        // Chain rule: df/dt = df/du * du/dt
        du / dt
    }
}

impl BezierSplineTrajectory {
    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;

        // Helper to read u32 and f64 directly to avoid borrow issues
        fn read_u32(file: &mut File) -> u32 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf).unwrap();
            u32::from_le_bytes(buf)
        }
        fn read_f64(file: &mut File) -> f64 {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf).unwrap();
            f64::from_le_bytes(buf)
        }

        let num_channels = read_u32(&mut file);
        let rot_mode_val = read_u32(&mut file);
        let fps = read_f64(&mut file);

        let rot_mode = match rot_mode_val {
            0 => RotationMode::EulerXYZ,
            1 => RotationMode::Quaternion,
            2 => RotationMode::EulerXZY,
            3 => RotationMode::EulerYXZ,
            4 => RotationMode::EulerYZX,
            5 => RotationMode::EulerZXY,
            6 => RotationMode::EulerZYX,
            _ => RotationMode::EulerXYZ,
        };

        let mut channels = Vec::new();
        for _ in 0..num_channels {
            let num_keys = read_u32(&mut file);
            let mut keyframes = Vec::new();
            for _ in 0..num_keys {
                let kf = BezierKeyframe {
                    time: read_f64(&mut file),
                    value: read_f64(&mut file),
                    handle_left: (read_f64(&mut file), read_f64(&mut file)),
                    handle_right: (read_f64(&mut file), read_f64(&mut file)),
                };
                // Blender keyframes can modify only a subset of channels (e.g. a
                // rotation-only keyframe at frame 197). The exporter emits one
                // keyframe per channel for the union of all keyframe times, and
                // for channels without a real keyframe at that time it writes a
                // synthetic keyframe with both handles collapsed to (time, value).
                // That collapse makes p1==p0 and p2==p3 in the cubic Bezier,
                // pinning the derivative to zero and producing a brief "pause"
                // in the channel — visible as a bump in dead-reckoned IMU and as
                // a delta-function spike in derived acceleration. Skip these.
                let is_synthetic = kf.handle_left.0 == kf.time
                    && kf.handle_right.0 == kf.time
                    && kf.handle_left.1 == kf.value
                    && kf.handle_right.1 == kf.value;
                if !is_synthetic {
                    keyframes.push(kf);
                }
            }
            channels.push(BezierCurve { keyframes });
        }

        let pos_channels = [
            channels[0].clone(),
            channels[1].clone(),
            channels[2].clone(),
        ];
        let rot_channels = channels[3..].to_vec();

        // Start/end of the trajectory is the union across all channels, not the
        // X-position channel alone, since dropping synthetics may have removed a
        // channel's boundary keyframe.
        let start_time = channels
            .iter()
            .filter_map(|c| c.keyframes.first().map(|k| k.time))
            .fold(f64::INFINITY, f64::min);
        let duration = channels
            .iter()
            .filter_map(|c| c.keyframes.last().map(|k| k.time))
            .fold(f64::NEG_INFINITY, f64::max);

        Ok(Self {
            position_curves: pos_channels,
            rotation_curves: rot_channels,
            rotation_mode: rot_mode,
            fps,
            start_time,
            duration,
        })
    }

    pub fn sample_poses(&self, num_frames: usize) -> Vec<SE3<f64>> {
        let mut poses = Vec::with_capacity(num_frames);
        if num_frames == 0 {
            return poses;
        }

        for i in 0..num_frames {
            let t = if num_frames > 1 {
                i as f64 / (num_frames - 1) as f64
            } else {
                0.0
            };
            poses.push(self.pose(t));
        }
        poses
    }
}

impl ContinuousTrajectory for BezierSplineTrajectory {
    fn pose(&self, t: f64) -> SE3<f64> {
        let t_abs = t * self.duration;
        let p_cv = Vec3::new(
            self.position_curves[0].evaluate(t_abs),
            self.position_curves[1].evaluate(t_abs),
            self.position_curves[2].evaluate(t_abs),
        );

        // W2O: Blender world (+Y up) → OpenCV world (+Y down): X_cv=X_b, Y_cv=-Z_b, Z_cv=Y_b
        let r_pre = na::Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);

        let r_b = match self.rotation_mode {
            RotationMode::EulerXYZ => {
                let rx = na::UnitQuaternion::from_axis_angle(
                    &na::Vector3::x_axis(),
                    self.rotation_curves[0].evaluate(t_abs),
                );
                let ry = na::UnitQuaternion::from_axis_angle(
                    &na::Vector3::y_axis(),
                    self.rotation_curves[1].evaluate(t_abs),
                );
                let rz = na::UnitQuaternion::from_axis_angle(
                    &na::Vector3::z_axis(),
                    self.rotation_curves[2].evaluate(t_abs),
                );
                (rz * ry * rx).to_rotation_matrix().matrix().clone()
            }
            RotationMode::Quaternion => {
                // Blender rotation_quaternion channels: [W, X, Y, Z] (W first)
                // na::Quaternion::new(w, i, j, k)
                let q = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
                    self.rotation_curves[0].evaluate(t_abs), // w
                    self.rotation_curves[1].evaluate(t_abs), // x
                    self.rotation_curves[2].evaluate(t_abs), // y
                    self.rotation_curves[3].evaluate(t_abs), // z
                ));
                q.to_rotation_matrix().matrix().clone()
            }
            _ => na::Matrix3::identity(),
        };

        // Apply world-frame transform only (no camera body-frame flip — this is for
        // generic objects, not cameras).
        let r_cv_na = r_pre * r_b;
        let q_cv_na = na::UnitQuaternion::from_matrix(&r_cv_na);
        let q_cv = Quat::new(
            q_cv_na.coords[3],
            q_cv_na.coords[0],
            q_cv_na.coords[1],
            q_cv_na.coords[2],
        );

        SE3::from_rotation_translation(SO3 { quat: q_cv }, p_cv)
    }

    fn linear_velocity(&self, t: f64) -> na::Vector3<f64> {
        let t_abs = t * self.duration;
        // BezierCurve::derivative(t_abs) gives d(value)/d(seconds) = m/s directly
        na::Vector3::new(
            self.position_curves[0].derivative(t_abs),
            self.position_curves[1].derivative(t_abs),
            self.position_curves[2].derivative(t_abs),
        )
    }

    fn angular_velocity(&self, t: f64) -> Option<na::Vector3<f64>> {
        let t_abs = t * self.duration;

        match self.rotation_mode {
            RotationMode::EulerXYZ => {
                // R_b = Rz(psi) * Ry(theta) * Rx(phi)
                // Body-frame: ω_body = x·dphi + Rx^T·y·dtheta + Rx^T·Ry^T·z·dpsi
                // (r_pre is constant so it cancels in R^T·dR/dt)
                let dphi = self.rotation_curves[0].derivative(t_abs);
                let dtheta = self.rotation_curves[1].derivative(t_abs);
                let dpsi = self.rotation_curves[2].derivative(t_abs);

                let rx = na::Rotation3::from_axis_angle(
                    &na::Vector3::x_axis(),
                    self.rotation_curves[0].evaluate(t_abs),
                );
                let ry = na::Rotation3::from_axis_angle(
                    &na::Vector3::y_axis(),
                    self.rotation_curves[1].evaluate(t_abs),
                );

                let k = na::Vector3::z();
                let j = na::Vector3::y();
                let i = na::Vector3::x();

                Some(i * dphi + rx.transpose() * j * dtheta + rx.transpose() * ry.transpose() * k * dpsi)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bezier_linear() {
        // Precise linear curve
        let curve = BezierCurve {
            keyframes: vec![
                BezierKeyframe {
                    time: 0.0,
                    value: 0.0,
                    handle_left: (-0.1, 0.0),
                    handle_right: (1.0 / 3.0, 1.0 / 3.0),
                },
                BezierKeyframe {
                    time: 1.0,
                    value: 1.0,
                    handle_left: (2.0 / 3.0, 2.0 / 3.0),
                    handle_right: (1.1, 1.0),
                },
            ],
        };

        // At t=0.5, value should be 0.5
        assert_abs_diff_eq!(curve.evaluate(0.5), 0.5, epsilon = 1e-10);
        // Derivative should be 1.0
        assert_abs_diff_eq!(curve.derivative(0.5), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bezier_smooth_step() {
        // Curve that starts and ends with zero velocity
        let curve = BezierCurve {
            keyframes: vec![
                BezierKeyframe {
                    time: 0.0,
                    value: 0.0,
                    handle_left: (-0.1, 0.0),
                    handle_right: (0.5, 0.0), // horizontal handle
                },
                BezierKeyframe {
                    time: 1.0,
                    value: 1.0,
                    handle_left: (0.5, 1.0), // horizontal handle
                    handle_right: (1.1, 1.0),
                },
            ],
        };

        assert_abs_diff_eq!(curve.evaluate(0.0), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(curve.evaluate(1.0), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(curve.evaluate(0.5), 0.5, epsilon = 1e-6);

        // Velocities at ends should be 0
        assert_abs_diff_eq!(curve.derivative(0.01), 0.0, epsilon = 0.1);
        // Peak velocity at center should be 1.5 (for this specific setup)
        // B'(0.5) = 3(0.25)(0) + 6(0.25)(1) + 3(0.25)(0) = 1.5
        assert_abs_diff_eq!(curve.derivative(0.5), 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_spline_derivatives() {
        // Create a moving and rotating spline
        let mut curves = Vec::new();
        for i in 0..6 {
            curves.push(BezierCurve {
                keyframes: vec![
                    BezierKeyframe {
                        time: 0.0,
                        value: 0.0,
                        handle_left: (-0.1, 0.0),
                        handle_right: (0.33, if i < 3 { 1.0 } else { 0.1 }),
                    },
                    BezierKeyframe {
                        time: 1.0,
                        value: if i < 3 { 1.0 } else { 0.5 },
                        handle_left: (0.66, if i < 3 { 0.0 } else { 0.4 }),
                        handle_right: (1.1, 1.0),
                    },
                ],
            });
        }

        let trajectory = BezierSplineTrajectory {
            start_time: 0.0,
            position_curves: [curves[0].clone(), curves[1].clone(), curves[2].clone()],
            rotation_curves: vec![curves[3].clone(), curves[4].clone(), curves[5].clone()],
            rotation_mode: RotationMode::EulerXYZ,
            fps: 30.0,
            duration: 1.0,
        };

        let t = 0.5;
        let v_ana = trajectory.linear_velocity(t);

        // Numerical derivative of position w.r.t. real time (seconds)
        // pose takes normalized t, so dt_norm corresponds to dt_real = dt_norm * duration
        let dt_norm = 1e-6;
        let dt_real = dt_norm * trajectory.duration;
        let p_plus = trajectory.pose(t + dt_norm).translation;
        let p_minus = trajectory.pose(t - dt_norm).translation;
        let diff = p_plus - p_minus;
        let v_num = na::Vector3::new(
            diff.x / (2.0 * dt_real),
            diff.y / (2.0 * dt_real),
            diff.z / (2.0 * dt_real),
        );

        assert_abs_diff_eq!(v_ana.x, v_num.x, epsilon = 1e-6);
        assert_abs_diff_eq!(v_ana.y, v_num.y, epsilon = 1e-6);
        assert_abs_diff_eq!(v_ana.z, v_num.z, epsilon = 1e-6);

        // Angular velocity check (Euler XYZ) — body frame: ω = 2·Im(conj(q)·q_dot)
        if let Some(omega_ana) = trajectory.angular_velocity(t) {
            let q_plus = trajectory.pose(t + dt_norm).rotation.quat;
            let q_minus = trajectory.pose(t - dt_norm).rotation.quat;
            let q_t = trajectory.pose(t).rotation.quat;

            let qw_dot = (q_plus.w - q_minus.w) / (2.0 * dt_real);
            let qx_dot = (q_plus.x - q_minus.x) / (2.0 * dt_real);
            let qy_dot = (q_plus.y - q_minus.y) / (2.0 * dt_real);
            let qz_dot = (q_plus.z - q_minus.z) / (2.0 * dt_real);

            // conj(q_t) * q_dot  (body-frame formula)
            let res_x = q_t.w * qx_dot - q_t.x * qw_dot - q_t.y * qz_dot + q_t.z * qy_dot;
            let res_y = q_t.w * qy_dot + q_t.x * qz_dot - q_t.y * qw_dot - q_t.z * qx_dot;
            let res_z = q_t.w * qz_dot - q_t.x * qy_dot + q_t.y * qx_dot - q_t.z * qw_dot;

            let omega_num = na::Vector3::new(2.0 * res_x, 2.0 * res_y, 2.0 * res_z);

            assert_abs_diff_eq!(omega_ana.x, omega_num.x, epsilon = 1e-5);
            assert_abs_diff_eq!(omega_ana.y, omega_num.y, epsilon = 1e-5);
            assert_abs_diff_eq!(omega_ana.z, omega_num.z, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_spline_vs_discrete_poses() {
        let spline_path = "blender_stuff/greeble_room/camera_spline.bin";
        let poses_path = "blender_stuff/greeble_room/camera_poses.bin";

        // Skip test if files don't exist (e.g. in CI without submodules)
        if !std::path::Path::new(spline_path).exists() || !std::path::Path::new(poses_path).exists()
        {
            println!("skipping test_spline_vs_discrete_poses: data files not found");
            return;
        }

        let trajectory = BezierSplineTrajectory::load(spline_path).unwrap();

        // Load discrete poses (manual implementation of the loader)
        let file = File::open(poses_path).unwrap();
        let mut reader = std::io::BufReader::new(file);

        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).unwrap();
        let num_frames = u32::from_le_bytes(buf) as usize;

        let mut discrete_poses = Vec::with_capacity(num_frames);
        for _ in 0..num_frames {
            let mut matrix = [[0.0f32; 4]; 4];
            for row in 0..4 {
                for col in 0..4 {
                    let mut val_buf = [0u8; 4];
                    reader.read_exact(&mut val_buf).unwrap();
                    matrix[row][col] = f32::from_le_bytes(val_buf);
                }
            }

            let x_axis = Vec3::new(
                matrix[0][0] as f64,
                matrix[1][0] as f64,
                matrix[2][0] as f64,
            );
            let y_axis = Vec3::new(
                matrix[0][1] as f64,
                matrix[1][1] as f64,
                matrix[2][1] as f64,
            );
            let z_axis = Vec3::new(
                matrix[0][2] as f64,
                matrix[1][2] as f64,
                matrix[2][2] as f64,
            );
            let rotation_matrix = odysseus_solver::math3d::Mat3::from_cols(x_axis, y_axis, z_axis);
            let rotation = SO3::from_matrix(rotation_matrix);
            let translation = Vec3::new(
                matrix[0][3] as f64,
                matrix[1][3] as f64,
                matrix[2][3] as f64,
            );
            discrete_poses.push(SE3 {
                rotation,
                translation,
            });
        }

        // Compare spline samples to discrete poses
        // The discrete poses were exported at 30 FPS starting at frame 1 (t = 1/30)
        for i in 0..num_frames {
            let t_abs = (i + 1) as f64 / trajectory.fps;
            let t = t_abs / trajectory.duration;
            let spline_pose = trajectory.pose(t);
            let discrete_pose = discrete_poses[i];

            // Translation should be quite close (within 1mm)
            assert_abs_diff_eq!(
                spline_pose.translation.x,
                discrete_pose.translation.x,
                epsilon = 1e-3
            );
            assert_abs_diff_eq!(
                spline_pose.translation.y,
                discrete_pose.translation.y,
                epsilon = 1e-3
            );
            assert_abs_diff_eq!(
                spline_pose.translation.z,
                discrete_pose.translation.z,
                epsilon = 1e-3
            );

            // Rotation should also be close
            // We compare the quaternions (standardizing the sign)
            let q1 = spline_pose.rotation.quat;
            let q2 = discrete_pose.rotation.quat;
            let dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
            assert!(dot.abs() > 0.999);
        }
    }
}
