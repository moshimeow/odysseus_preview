//! Ping-pong replay of a [`CatmullRomSlerpTrajectory`]: forward, backward,
//! forward, … for a fixed number of passes, with each reversal smeared over a
//! blend window so the replayed motion has no velocity jumps.
//!
//! The wrapper is a pure time-warp `τ(t)` composed with the base trajectory:
//! `pose(t) = base.pose(τ(t))`, `velocity(t) = base.velocity(τ(t)) · τ'(t)`.
//! During a reversal the playback speed `τ'` follows a half-cosine from +1 to
//! −1 (or back), so `τ'` is continuous everywhere and `τ''` is continuous and
//! zero at the blend boundaries. Velocity is therefore continuous through
//! every turnaround, and acceleration is as continuous as the base
//! trajectory's own (C¹ Hermite) motion allows.
//!
//! Geometry of one reversal: integrating the cosine speed profile over the
//! blend window of length `L` advances the base time by `Δ = L/π` before
//! returning, so a cruise segment hands over to the blend `Δ` before the
//! base-trajectory endpoint and the apex of the blend lands exactly on the
//! endpoint.

use crate::catmull_rom_slerp_trajectory::CatmullRomSlerpTrajectory;
use nalgebra::Vector3;
use crate::ContinuousTrajectory;
use odysseus_solver::math3d::{Vec3, SE3};

/// Direction of travel through the base trajectory, used as the cruise
/// speed `τ'` and as the sign of the blend's sine excursion.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Segment {
    /// Constant-speed playback: `τ' = direction`.
    Cruise { direction: f64 },
    /// Half-cosine reversal around a base-time apex:
    /// `τ' = direction_in · cos(π·u / len)` for local time `u ∈ [0, len]`.
    Turn { direction_in: f64 },
}

/// One piece of the global-time → base-time warp. `t_start`/`tau_start` are
/// the segment's entry points on the two time axes; `len` is its global-time
/// duration.
#[derive(Clone, Copy, Debug)]
struct WarpSegment {
    t_start: f64,
    tau_start: f64,
    len: f64,
    kind: Segment,
}

pub struct PingPongTrajectory {
    base: CatmullRomSlerpTrajectory,
    segments: Vec<WarpSegment>,
    duration: f64,
}

impl PingPongTrajectory {
    /// Replay `base` for `cycles` passes (1 = plain forward playback,
    /// identical to the base trajectory), reversing direction between
    /// passes. Each reversal is blended over `blend_seconds` of global time.
    ///
    /// Fails when `cycles == 0`, or when `cycles > 1` and the blend is
    /// non-positive or eats more than the whole trajectory (the two blend
    /// penetrations `2·blend_seconds/π` must fit inside the base duration).
    pub fn new(
        base: CatmullRomSlerpTrajectory,
        cycles: usize,
        blend_seconds: f64,
    ) -> Result<Self, String> {
        if cycles == 0 {
            return Err("ping-pong cycle count must be at least 1".into());
        }
        let base_duration = base.end_time() - base.start_time();
        let mut segments = Vec::new();
        let mut t = 0.0;
        if cycles == 1 {
            segments.push(WarpSegment {
                t_start: t,
                tau_start: 0.0,
                len: base_duration,
                kind: Segment::Cruise { direction: 1.0 },
            });
            t += base_duration;
        } else {
            if !(blend_seconds.is_finite() && blend_seconds > 0.0) {
                return Err("ping-pong blend must be positive and finite".into());
            }
            // Base time consumed by the accelerating half of one blend; the
            // apex of each reversal reaches exactly the trajectory endpoint.
            let penetration = blend_seconds / std::f64::consts::PI;
            if 2.0 * penetration >= base_duration {
                return Err(format!(
                    "ping-pong blend of {blend_seconds} s penetrates {penetration:.3} s past each \
                     cruise handover, which does not fit twice into the {base_duration:.3} s \
                     trajectory; shorten --ping-pong-blend-seconds"
                ));
            }
            let mut direction = 1.0;
            let mut tau = 0.0;
            for pass in 0..cycles {
                let first = pass == 0;
                let last = pass == cycles - 1;
                // A pass cruises between the blend handover points, except
                // that the very first start and very last stop are hard
                // trajectory endpoints with no blend on that side.
                let cruise_len =
                    base_duration - penetration * (2 - first as i32 - last as i32) as f64;
                segments.push(WarpSegment {
                    t_start: t,
                    tau_start: tau,
                    len: cruise_len,
                    kind: Segment::Cruise { direction },
                });
                t += cruise_len;
                tau += direction * cruise_len;
                if !last {
                    segments.push(WarpSegment {
                        t_start: t,
                        tau_start: tau,
                        len: blend_seconds,
                        kind: Segment::Turn {
                            direction_in: direction,
                        },
                    });
                    t += blend_seconds;
                    // The blend is symmetric: it returns to the base time it
                    // entered at, now moving the other way.
                    direction = -direction;
                }
            }
        }
        Ok(Self {
            base,
            segments,
            duration: t,
        })
    }

    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// The base trajectory when this is plain forward playback (one cycle,
    /// `τ(t) = t`): its knots are then the ground truth at their own
    /// instants, bit for bit. `None` for a ping-ponged replay.
    pub fn plain_playback(&self) -> Option<&CatmullRomSlerpTrajectory> {
        match self.segments.as_slice() {
            [WarpSegment { kind: Segment::Cruise { direction }, .. }] if *direction == 1.0 => Some(&self.base),
            _ => None,
        }
    }

    /// Warped base time and playback speed `(τ(t), τ'(t))`, with `t` and `τ`
    /// both measured from the base trajectory's start. Clamped outside
    /// `[0, duration]`.
    fn warp(&self, t: f64) -> (f64, f64) {
        let t = t.clamp(0.0, self.duration);
        let segment = self
            .segments
            .iter()
            .rev()
            .find(|s| t >= s.t_start)
            .expect("warp segments start at t = 0");
        let u = t - segment.t_start;
        match segment.kind {
            Segment::Cruise { direction } => (segment.tau_start + direction * u, direction),
            Segment::Turn { direction_in } => {
                let phase = std::f64::consts::PI * u / segment.len;
                let excursion = segment.len / std::f64::consts::PI;
                (
                    segment.tau_start + direction_in * excursion * phase.sin(),
                    direction_in * phase.cos(),
                )
            }
        }
    }

    /// The base-trajectory time (on the source recording's clock) that
    /// global time `t` plays back, so per-frame data of the source can be
    /// sampled through the same warp as the poses.
    pub fn source_time(&self, t: f64) -> f64 {
        let (tau, _) = self.warp(t);
        self.base.start_time() + tau
    }

    pub fn pose(&self, t: f64) -> SE3<f64> {
        self.base.pose(self.source_time(t))
    }

    pub fn linear_velocity(&self, t: f64) -> Vec3<f64> {
        let (tau, speed) = self.warp(t);
        let v = self.base.linear_velocity(self.base.start_time() + tau);
        Vec3::new(v.x * speed, v.y * speed, v.z * speed)
    }
}

impl ContinuousTrajectory for PingPongTrajectory {
    /// `t` is normalized ∈ [0, 1], mapped to the full ping-ponged span.
    fn pose(&self, t: f64) -> SE3<f64> {
        PingPongTrajectory::pose(self, t * self.duration)
    }

    /// Velocity in m/s (absolute-time units) at normalized `t`, matching the
    /// base trajectory's convention.
    fn linear_velocity(&self, t: f64) -> Vector3<f64> {
        let v = PingPongTrajectory::linear_velocity(self, t * self.duration);
        Vector3::new(v.x, v.y, v.z)
    }

    /// None: the simulator's finite-difference rotation handles slerp-anchor
    /// and reversal crossings exactly.
    fn angular_velocity(&self, _t: f64) -> Option<Vector3<f64>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use odysseus_solver::math3d::{Quat, SO3};

    fn base() -> CatmullRomSlerpTrajectory {
        let samples: Vec<(f64, SE3<f64>)> = (0..=10)
            .map(|i| {
                let t = i as f64 * 0.1;
                (
                    t,
                    SE3::from_rotation_translation(
                        SO3 {
                            quat: Quat::from_axis_angle(Vec3::new(0.0, 0.4 * t, 0.0)),
                        },
                        Vec3::new(t * t, (3.0 * t).sin(), -t),
                    ),
                )
            })
            .collect();
        CatmullRomSlerpTrajectory::from_samples(&samples)
    }

    #[test]
    fn single_cycle_matches_base() {
        let b = base();
        let wrapped = PingPongTrajectory::new(base(), 1, 0.2).unwrap();
        assert_eq!(wrapped.duration(), 1.0);
        for i in 0..=20 {
            let t = i as f64 * 0.05;
            let dp = wrapped.pose(t).translation - b.pose(t).translation;
            assert!(dp.norm() < 1e-12);
            let dv = wrapped.linear_velocity(t) - b.linear_velocity(t);
            assert!(dv.norm() < 1e-12);
        }
    }

    #[test]
    fn reversal_apex_touches_endpoint_and_velocity_is_continuous() {
        let blend = 0.2;
        let wrapped = PingPongTrajectory::new(base(), 3, blend).unwrap();
        let penetration = blend / std::f64::consts::PI;
        // 3 passes: two boundary cruises of D−Δ, one middle cruise of D−2Δ,
        // two blends.
        let expected = 2.0 * (1.0 - penetration) + (1.0 - 2.0 * penetration) + 2.0 * blend;
        assert!((wrapped.duration() - expected).abs() < 1e-12);
        // Apex of the first reversal lands exactly on the base end pose.
        let apex_t = (1.0 - penetration) + 0.5 * blend;
        let b = base();
        let dp = wrapped.pose(apex_t).translation - b.pose(1.0).translation;
        assert!(dp.norm() < 1e-9);
        // Velocity continuity across every segment boundary, sampled densely.
        let dt = 1e-6;
        let mut t = dt;
        while t < wrapped.duration() - dt {
            let dv = wrapped.linear_velocity(t + dt) - wrapped.linear_velocity(t - dt);
            assert!(
                dv.norm() < 1e-3,
                "velocity jump {} at t = {t}",
                dv.norm()
            );
            t += 7.7e-4;
        }
    }

    #[test]
    fn zero_cycles_and_oversized_blend_are_rejected() {
        assert!(PingPongTrajectory::new(base(), 0, 0.2).is_err());
        assert!(PingPongTrajectory::new(base(), 2, 0.0).is_err());
        // Δ = blend/π must satisfy 2Δ < 1.0 s.
        assert!(PingPongTrajectory::new(base(), 2, 2.0).is_err());
        assert!(PingPongTrajectory::new(base(), 2, 1.5).is_ok());
    }
}
