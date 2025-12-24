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
                keyframes.push(BezierKeyframe {
                    time: read_f64(&mut file),
                    value: read_f64(&mut file),
                    handle_left: (read_f64(&mut file), read_f64(&mut file)),
                    handle_right: (read_f64(&mut file), read_f64(&mut file)),
                });
            }
            channels.push(BezierCurve { keyframes });
        }

        let pos_channels = [
            channels[0].clone(),
            channels[1].clone(),
            channels[2].clone(),
        ];
        let rot_channels = channels[3..].to_vec();

        let duration = pos_channels[0]
            .keyframes
            .last()
            .map(|k| k.time)
            .unwrap_or(0.0);

        Ok(Self {
            position_curves: pos_channels,
            rotation_curves: rot_channels,
            rotation_mode: rot_mode,
            fps,
            duration,
        })
    }

    pub fn sample_poses(&self, num_frames: usize) -> Vec<SE3<f64>> {
        let mut poses = Vec::with_capacity(num_frames);
        if num_frames == 0 {
            return poses;
        }

        let dt = if num_frames > 1 {
            self.duration / (num_frames - 1) as f64
        } else {
            0.0
        };
        for i in 0..num_frames {
            poses.push(self.pose(i as f64 * dt));
        }
        poses
    }
}

impl ContinuousTrajectory for BezierSplineTrajectory {
    fn pose(&self, t: f64) -> SE3<f64> {
        let p_cv = Vec3::new(
            self.position_curves[0].evaluate(t),
            self.position_curves[1].evaluate(t),
            self.position_curves[2].evaluate(t),
        );

        let r_pre = na::Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);

        let r_post = na::Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);

        let r_b = match self.rotation_mode {
            RotationMode::EulerXYZ => {
                let rx = na::UnitQuaternion::from_axis_angle(
                    &na::Vector3::x_axis(),
                    self.rotation_curves[0].evaluate(t),
                );
                let ry = na::UnitQuaternion::from_axis_angle(
                    &na::Vector3::y_axis(),
                    self.rotation_curves[1].evaluate(t),
                );
                let rz = na::UnitQuaternion::from_axis_angle(
                    &na::Vector3::z_axis(),
                    self.rotation_curves[2].evaluate(t),
                );
                (rz * ry * rx).to_rotation_matrix().matrix().clone()
            }
            RotationMode::Quaternion => {
                let q = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
                    self.rotation_curves[1].evaluate(t), // x
                    self.rotation_curves[2].evaluate(t), // y
                    self.rotation_curves[3].evaluate(t), // z
                    self.rotation_curves[0].evaluate(t), // w
                ));
                q.to_rotation_matrix().matrix().clone()
            }
            _ => na::Matrix3::identity(),
        };

        let r_cv_na = r_pre * r_b * r_post;
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
        na::Vector3::new(
            self.position_curves[0].derivative(t),
            self.position_curves[1].derivative(t),
            self.position_curves[2].derivative(t),
        )
    }

    fn angular_velocity(&self, t: f64) -> Option<na::Vector3<f64>> {
        let r_pre = na::Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);

        let omega_b = match self.rotation_mode {
            RotationMode::EulerXYZ => {
                let dphi = self.rotation_curves[0].derivative(t);
                let dtheta = self.rotation_curves[1].derivative(t);
                let dpsi = self.rotation_curves[2].derivative(t);

                let rz = na::Rotation3::from_axis_angle(
                    &na::Vector3::z_axis(),
                    self.rotation_curves[2].evaluate(t),
                );
                let ry = na::Rotation3::from_axis_angle(
                    &na::Vector3::y_axis(),
                    self.rotation_curves[1].evaluate(t),
                );

                let k = na::Vector3::z();
                let j = na::Vector3::y();
                let i = na::Vector3::x();

                let omega = k * dpsi + rz * j * dtheta + (rz * ry) * i * dphi;
                omega
            }
            _ => return None,
        };

        let omega_cv_na = r_pre * omega_b;
        Some(omega_cv_na)
    }
}
