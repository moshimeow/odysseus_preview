//! Interpolated trajectory built from per-frame metadata poses, plus a
//! direct-from-pose IMU synthesizer.
//!
//! Avoids the Blender spline convention headaches — all pose samples come from
//! the same `t_world_controller` that bootstrap & the filter consume.
//!
//! - Position: cubic Hermite (Catmull-Rom-style tangents from central differences).
//! - Rotation: slerp, so body-frame angular velocity is constant within an interval.
//! - IMU: specific force = R_wb * (a_world − g_world), gyro = body angular velocity,
//!   plus Gaussian white noise and a random-walk bias per `ImuNoiseParams`.

use nalgebra::Vector3;
use crate::simulator::{ImuNoiseParams, ImuSimulator};
use crate::ImuMeasurement;
use crate::ContinuousTrajectory;
use odysseus_solver::math3d::{SE3, SO3};
use odysseus_solver::math3d::{Quat, Vec3};

pub struct CatmullRomSlerpTrajectory {
    times: Vec<f64>,
    translations: Vec<Vec3<f64>>,
    rotations: Vec<Quat<f64>>,
    /// Catmull-Rom tangents for translation (world units per second).
    trans_tangents: Vec<Vec3<f64>>,
}

impl CatmullRomSlerpTrajectory {
    /// Build from a sorted-by-time list of `(time_s, t_world_controller)` samples.
    /// Requires at least 2 samples.
    pub fn from_samples(samples: &[(f64, SE3<f64>)]) -> Self {
        assert!(
            samples.len() >= 2,
            "CatmullRomSlerpTrajectory needs ≥ 2 samples"
        );
        let n = samples.len();
        let times: Vec<f64> = samples.iter().map(|(t, _)| *t).collect();
        let translations: Vec<Vec3<f64>> = samples.iter().map(|(_, p)| p.translation).collect();
        let rotations: Vec<Quat<f64>> = samples.iter().map(|(_, p)| p.rotation.quat).collect();

        let mut trans_tangents = Vec::with_capacity(n);
        for i in 0..n {
            let tangent = if i == 0 {
                scale_vec(
                    translations[1] - translations[0],
                    1.0 / (times[1] - times[0]),
                )
            } else if i == n - 1 {
                scale_vec(
                    translations[n - 1] - translations[n - 2],
                    1.0 / (times[n - 1] - times[n - 2]),
                )
            } else {
                scale_vec(
                    translations[i + 1] - translations[i - 1],
                    1.0 / (times[i + 1] - times[i - 1]),
                )
            };
            trans_tangents.push(tangent);
        }

        Self {
            times,
            translations,
            rotations,
            trans_tangents,
        }
    }

    pub fn start_time(&self) -> f64 {
        self.times[0]
    }

    pub fn end_time(&self) -> f64 {
        *self.times.last().unwrap()
    }

    fn interval(&self, t: f64) -> (usize, f64, f64) {
        // Returns (j, u, dt) where t ∈ [times[j], times[j+1]], u = (t-times[j])/dt.
        // Clamped to the available range.
        if t <= self.times[0] {
            return (0, 0.0, self.times[1] - self.times[0]);
        }
        let last = self.times.len() - 1;
        if t >= self.times[last] {
            return (last - 1, 1.0, self.times[last] - self.times[last - 1]);
        }
        let j = match self.times.binary_search_by(|x| x.partial_cmp(&t).unwrap()) {
            Ok(i) if i == last => i - 1,
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let dt = self.times[j + 1] - self.times[j];
        let u = (t - self.times[j]) / dt;
        (j, u, dt)
    }

    /// The knots the trajectory interpolates, `(time, pose)` in order — the
    /// samples `from_samples` was given, bit for bit (no renormalisation).
    pub fn knots(&self) -> impl Iterator<Item = (f64, SE3<f64>)> + '_ {
        self.times
            .iter()
            .zip(&self.translations)
            .zip(&self.rotations)
            .map(|((&t, &p), &q)| (t, SE3::from_rotation_translation(SO3 { quat: q }, p)))
    }

    pub fn pose(&self, t: f64) -> SE3<f64> {
        let (j, u, dt) = self.interval(t);
        // Hermite position.
        let (h00, h10, h01, h11) = hermite_basis(u);
        let p = sum4(
            scale_vec(self.translations[j], h00),
            scale_vec(self.trans_tangents[j], dt * h10),
            scale_vec(self.translations[j + 1], h01),
            scale_vec(self.trans_tangents[j + 1], dt * h11),
        );
        let q = slerp(self.rotations[j], self.rotations[j + 1], u);
        SE3::from_rotation_translation(SO3 { quat: q }, p)
    }

    pub fn linear_velocity(&self, t: f64) -> Vec3<f64> {
        let (j, u, dt) = self.interval(t);
        let (dh00, dh10, dh01, dh11) = hermite_basis_d1(u);
        let dpdu = sum4(
            scale_vec(self.translations[j], dh00),
            scale_vec(self.trans_tangents[j], dt * dh10),
            scale_vec(self.translations[j + 1], dh01),
            scale_vec(self.trans_tangents[j + 1], dt * dh11),
        );
        scale_vec(dpdu, 1.0 / dt)
    }

    /// World-frame linear acceleration averaged over `[t_prev, t]`, computed
    /// exactly as `(v(t) − v(t_prev)) / (t − t_prev)`. The velocity field is
    /// C¹ across slerp anchors (Catmull-Rom Hermite), so this is exact and
    /// free of slerp-boundary artifacts. Use this when you need the value an
    /// IMU would report — i.e. the average of the instantaneous derivative
    /// over the sampling interval — instead of the instantaneous derivative
    /// itself (which is what `linear_acceleration` returns).
    pub fn average_linear_acceleration(&self, t_prev: f64, t: f64) -> Vec3<f64> {
        let dt = t - t_prev;
        if dt <= 0.0 {
            return Vec3::new(0.0, 0.0, 0.0);
        }
        let v_prev = self.linear_velocity(t_prev);
        let v_curr = self.linear_velocity(t);
        scale_vec(v_curr - v_prev, 1.0 / dt)
    }

    pub fn linear_acceleration(&self, t: f64) -> Vec3<f64> {
        let (j, u, dt) = self.interval(t);
        let (dd00, dd10, dd01, dd11) = hermite_basis_d2(u);
        let d2 = sum4(
            scale_vec(self.translations[j], dd00),
            scale_vec(self.trans_tangents[j], dt * dd10),
            scale_vec(self.translations[j + 1], dd01),
            scale_vec(self.trans_tangents[j + 1], dt * dd11),
        );
        scale_vec(d2, 1.0 / (dt * dt))
    }

    pub fn rotation_delta(&self, t: f64) -> Vec3<f64> {
        let (j, _u, _dt) = self.interval(t);
        let mut q1 = self.rotations[j + 1];
        // Canonicalize for short-arc slerp.
        let q0 = self.rotations[j];
        if q0.dot(q1) < 0.0 {
            q1 = -q1;
        }
        let rel = q0.conjugate() * q1;
        rel.to_axis_angle()
    }

    /// Body-frame angular velocity at `t` (slerp → piecewise constant within
    /// each anchor interval). Convenient for evaluating ω at a single time
    /// but **not** what you want when sampling for IMU integration: the value
    /// jumps discontinuously across slerp anchors, so per-sample omega
    /// readings drift when a sample's preint step crosses an anchor. Use
    /// `average_angular_velocity_body(t_prev, t)` for that case.
    pub fn angular_velocity_body(&self, t: f64) -> Vec3<f64> {
        let (_j, _u, dt) = self.interval(t);
        scale_vec(self.rotation_delta(t), 1.0 / dt)
    }

    /// Body-frame angular velocity averaged over `[t_prev, t]`, computed
    /// exactly as the body-frame relative rotation between the slerp poses
    /// at the two endpoints divided by the elapsed time. The relative
    /// rotation between any two slerp samples is well-defined regardless of
    /// whether anchors fall between them, so this absorbs slerp-boundary
    /// crossings cleanly. Use this when synthesizing IMU samples or
    /// otherwise integrating ω over a step.
    pub fn average_angular_velocity_body(&self, t_prev: f64, t: f64) -> Vec3<f64> {
        let dt = t - t_prev;
        if dt <= 0.0 {
            return Vec3::new(0.0, 0.0, 0.0);
        }
        let q_prev = self.pose(t_prev).rotation.quat;
        let q_curr = self.pose(t).rotation.quat;
        let mut q_rel = q_prev.conjugate() * q_curr;
        if q_rel.w < 0.0 {
            q_rel = Quat::new(-q_rel.w, -q_rel.x, -q_rel.y, -q_rel.z);
        }
        scale_vec(q_rel.to_axis_angle(), 1.0 / dt)
    }
}

impl ContinuousTrajectory for CatmullRomSlerpTrajectory {
    /// `t` is normalized ∈ [0, 1], mapped to [start_time, end_time].
    fn pose(&self, t: f64) -> SE3<f64> {
        let t_abs = self.start_time() + t * (self.end_time() - self.start_time());
        CatmullRomSlerpTrajectory::pose(self, t_abs)
    }

    /// Returns linear velocity in m/s (absolute-time units) at normalized `t`.
    /// The simulator computes Δv / dt_abs which gives correct m/s² acceleration.
    fn linear_velocity(&self, t: f64) -> Vector3<f64> {
        let t_abs = self.start_time() + t * (self.end_time() - self.start_time());
        let v = CatmullRomSlerpTrajectory::linear_velocity(self, t_abs);
        Vector3::new(v.x, v.y, v.z)
    }

    /// Returns None so ImuSimulator uses finite-difference rotation (equivalent
    /// to average_angular_velocity_body, handling slerp-anchor crossings).
    fn angular_velocity(&self, _t: f64) -> Option<Vector3<f64>> {
        None
    }
}

/// Synthesize IMU measurements at `imu_rate_hz` from a pose-based trajectory.
/// Gravity is in world frame. Noise + bias-walk follows the standard model.
pub fn synthesize_imu(
    traj: &CatmullRomSlerpTrajectory,
    imu_rate_hz: f64,
    gravity_world: [f64; 3],
    noise: &ImuNoiseParams,
    seed: u64,
) -> Vec<ImuMeasurement> {
    let duration = traj.end_time() - traj.start_time();
    ImuSimulator::new(noise.clone(), imu_rate_hz)
        .with_gravity(Vector3::new(
            gravity_world[0],
            gravity_world[1],
            gravity_world[2],
        ))
        .generate_from_continuous_trajectory(traj, duration, seed)
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn scale_vec(v: Vec3<f64>, s: f64) -> Vec3<f64> {
    Vec3::new(v.x * s, v.y * s, v.z * s)
}

fn sum4(a: Vec3<f64>, b: Vec3<f64>, c: Vec3<f64>, d: Vec3<f64>) -> Vec3<f64> {
    Vec3::new(
        a.x + b.x + c.x + d.x,
        a.y + b.y + c.y + d.y,
        a.z + b.z + c.z + d.z,
    )
}

fn hermite_basis(u: f64) -> (f64, f64, f64, f64) {
    let u2 = u * u;
    let u3 = u2 * u;
    (
        2.0 * u3 - 3.0 * u2 + 1.0,
        u3 - 2.0 * u2 + u,
        -2.0 * u3 + 3.0 * u2,
        u3 - u2,
    )
}

fn hermite_basis_d1(u: f64) -> (f64, f64, f64, f64) {
    let u2 = u * u;
    (
        6.0 * u2 - 6.0 * u,
        3.0 * u2 - 4.0 * u + 1.0,
        -6.0 * u2 + 6.0 * u,
        3.0 * u2 - 2.0 * u,
    )
}

fn hermite_basis_d2(u: f64) -> (f64, f64, f64, f64) {
    (
        12.0 * u - 6.0,
        6.0 * u - 4.0,
        -12.0 * u + 6.0,
        6.0 * u - 2.0,
    )
}

fn slerp(q0: Quat<f64>, q1: Quat<f64>, u: f64) -> Quat<f64> {
    let mut q1 = q1;
    if q0.dot(q1) < 0.0 {
        q1 = Quat::new(-q1.w, -q1.x, -q1.y, -q1.z);
    }
    let rel = q0.conjugate() * q1;
    let axis_angle = rel.to_axis_angle();
    let scaled = Vec3::new(u * axis_angle.x, u * axis_angle.y, u * axis_angle.z);
    q0 * Quat::from_axis_angle(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use odysseus_solver::math3d::catmull_rom::{catmull_rom_end_knot, catmull_rom_pose, RotationBlend};

    /// The trajectory here and the generic evaluator in `odysseus_solver::math3d::catmull_rom`
    /// (the one a solver differentiates) are the same curve.
    #[test]
    fn agrees_with_the_generic_evaluator_in_slerp_mode() {
        let samples: Vec<(f64, SE3<f64>)> = (0..6)
            .map(|i| {
                let t = 0.033 * i as f64 + 0.002 * (i as f64).sin();
                let rotation = SO3::exp(Vec3::new(0.02 * i as f64, -0.004 * (i * i) as f64, 0.02));
                let translation = Vec3::new(0.1 * i as f64, (i as f64).cos(), -0.2 * i as f64);
                (t, SE3::from_rotation_translation(rotation, translation))
            })
            .collect();
        let trajectory = CatmullRomSlerpTrajectory::from_samples(&samples);
        for j in 0..samples.len() - 1 {
            let before = if j == 0 { catmull_rom_end_knot(samples[0], samples[1]) } else { samples[j - 1] };
            let after = if j + 2 == samples.len() {
                catmull_rom_end_knot(samples[j + 1], samples[j])
            } else {
                samples[j + 2]
            };
            let four = [before, samples[j], samples[j + 1], after];
            for fraction in [0.0, 0.25, 0.6, 1.0] {
                let t = samples[j].0 + fraction * (samples[j + 1].0 - samples[j].0);
                let expected = trajectory.pose(t);
                let got = catmull_rom_pose(four.map(|k| k.1), four.map(|k| k.0), t, RotationBlend::Slerp);
                assert!((got.translation - expected.translation).norm() < 1e-12);
                let apart = (got.rotation.inverse() * expected.rotation).log().norm();
                // `SO3::exp(log(..))` round-trips to about 1e-9 rad at these step sizes; at an interval's
                // end the trajectory returns the knot itself and the evaluator goes through exp/log.
                assert!(apart < 1e-8, "interval {j} at {fraction}: {apart:e} rad");
            }
        }
    }
}
