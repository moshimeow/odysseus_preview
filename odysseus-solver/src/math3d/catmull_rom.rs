//! Catmull-Rom interpolation of poses through timed knots, generic over [`Real`].
//!
//! A Catmull-Rom curve passes through its knots: the pose at a knot's time is
//! that knot. Between two knots the curve is fixed by those two and their
//! outer neighbours, so one evaluation touches exactly four knots — which is
//! what makes it usable as a trajectory inside a least-squares solve: the
//! knots are the unknowns, each one readable as a pose, and a residual at
//! time `t` depends on the four knots around `t` and on `t` itself.
//!
//! Knot times need not be evenly spaced. Everything here is written against
//! `T: Real`, so the same code evaluates with `f64` or differentiates with
//! `Jet` — with respect to the knots, the evaluation time, or both.
//!
//! Position is the cubic Hermite curve whose tangent at a knot is the slope
//! between that knot's neighbours. It is C¹.
//!
//! Rotation has two forms ([`RotationBlend`]):
//!
//! - `Slerp` interpolates the two middle knots along the geodesic. Angular
//!   velocity is constant between knots and jumps at each one.
//! - `Smooth` applies the same four Catmull-Rom weights on the rotation
//!   group, in cumulative form (Kim, Kim & Shin, "A General Construction
//!   Scheme for Unit Quaternion Curves with Simple High Order Derivatives",
//!   1995 — the construction Basalt's `So3Spline` uses with B-spline weights):
//!
//!   `R(t) = R₀ · exp(b₁ d₁) · exp(b₂ d₂) · exp(b₃ d₃)`,
//!   `dᵢ = log(Rᵢ₋₁⁻¹ Rᵢ)`, `bᵢ = wᵢ + … + w₃`.
//!
//!   It still passes through the knots, and its body-frame angular velocity
//!   is continuous: at a knot it is `(d_before + d_after) / (t_after − t_before)`,
//!   the rotation analogue of the position tangent.
//!
//! The first and last interval of a knot sequence have no outer neighbour on
//! one side; [`catmull_rom_end_knot`] supplies one by reflection, which makes
//! the end tangent the slope of the end interval.

use crate::math3d::{Vec3, SE3, SO3};
use crate::Real;

/// How rotation is carried between the two middle knots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationBlend {
    /// Geodesic between the two middle knots; angular velocity jumps at knots.
    Slerp,
    /// Cumulative Catmull-Rom on SO(3); angular velocity is continuous.
    Smooth,
}

/// Weights of four consecutive knots for the curve at time `t`, where
/// `times[1] <= t <= times[2]`. They sum to one; the position is the weighted
/// sum of the four knot positions.
pub fn catmull_rom_weights<T: Real>(times: [f64; 4], t: T) -> [T; 4] {
    let dt = times[2] - times[1];
    let u = (t - T::from_f64(times[1])) / T::from_f64(dt);
    let u2 = u * u;
    let u3 = u2 * u;
    // Cubic Hermite basis on the middle interval.
    let h00 = u3 * T::from_f64(2.0) - u2 * T::from_f64(3.0) + T::one();
    let h10 = u3 - u2 * T::from_f64(2.0) + u;
    let h01 = u2 * T::from_f64(3.0) - u3 * T::from_f64(2.0);
    let h11 = u3 - u2;
    // Tangents (p₂ − p₀)/(t₂ − t₀) and (p₃ − p₁)/(t₃ − t₁), times dt.
    let a = T::from_f64(dt / (times[2] - times[0]));
    let b = T::from_f64(dt / (times[3] - times[1]));
    [-(h10 * a), h00 - h11 * b, h01 + h10 * a, h11 * b]
}

/// Position at `t` on the curve through four consecutive knots.
pub fn catmull_rom_position<T: Real>(knots: [Vec3<T>; 4], times: [f64; 4], t: T) -> Vec3<T> {
    let w = catmull_rom_weights(times, t);
    knots[0] * w[0] + knots[1] * w[1] + knots[2] * w[2] + knots[3] * w[3]
}

/// Rotation at `t` on the curve through four consecutive knots.
pub fn catmull_rom_rotation<T: Real>(
    knots: [SO3<T>; 4],
    times: [f64; 4],
    t: T,
    blend: RotationBlend,
) -> SO3<T> {
    let step = |from: SO3<T>, to: SO3<T>| (from.inverse() * to).log();
    match blend {
        RotationBlend::Slerp => {
            let u = (t - T::from_f64(times[1])) / T::from_f64(times[2] - times[1]);
            knots[1] * SO3::exp(step(knots[1], knots[2]) * u)
        }
        RotationBlend::Smooth => {
            let w = catmull_rom_weights(times, t);
            knots[0]
                * SO3::exp(step(knots[0], knots[1]) * (w[1] + w[2] + w[3]))
                * SO3::exp(step(knots[1], knots[2]) * (w[2] + w[3]))
                * SO3::exp(step(knots[2], knots[3]) * w[3])
        }
    }
}

/// Pose at `t` on the curve through four consecutive knots: rotation and
/// translation interpolated separately (not as a screw motion).
pub fn catmull_rom_pose<T: Real>(
    knots: [SE3<T>; 4],
    times: [f64; 4],
    t: T,
    blend: RotationBlend,
) -> SE3<T> {
    SE3::from_rotation_translation(
        catmull_rom_rotation(knots.map(|k| k.rotation), times, t, blend),
        catmull_rom_position(knots.map(|k| k.translation), times, t),
    )
}

/// The missing outer neighbour of an end interval: `end` reflected through
/// itself away from `inner`, as `(time, pose)`. With it the curve's tangent at
/// `end` is the slope of the end interval, in position and in rotation.
pub fn catmull_rom_end_knot<T: Real>(end: (f64, SE3<T>), inner: (f64, SE3<T>)) -> (f64, SE3<T>) {
    let (t_end, end) = end;
    let (t_inner, inner) = inner;
    let rotation = end.rotation * inner.rotation.inverse() * end.rotation;
    let translation = end.translation * T::from_f64(2.0) - inner.translation;
    (2.0 * t_end - t_inner, SE3::from_rotation_translation(rotation, translation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Jet;
    use approx::assert_abs_diff_eq;

    const TIMES: [f64; 4] = [0.0, 0.031, 0.070, 0.101];

    fn rotations() -> [SO3<f64>; 4] {
        [
            SO3::exp(Vec3::new(0.10, -0.20, 0.05)),
            SO3::exp(Vec3::new(0.14, -0.17, 0.02)),
            SO3::exp(Vec3::new(0.11, -0.10, 0.06)),
            SO3::exp(Vec3::new(0.05, -0.12, 0.13)),
        ]
    }

    fn positions() -> [Vec3<f64>; 4] {
        [
            Vec3::new(0.0, 1.0, -2.0),
            Vec3::new(0.3, 1.1, -1.8),
            Vec3::new(0.5, 0.9, -1.9),
            Vec3::new(0.4, 0.6, -2.3),
        ]
    }

    /// `exp` and `log` here agree to about 1e-9 rad at these angles, so that is the floor.
    fn assert_same_rotation(a: SO3<f64>, b: SO3<f64>, epsilon: f64) {
        let apart = (a.inverse() * b).log();
        assert!(apart.norm() < epsilon, "rotations {} rad apart", apart.norm());
    }

    #[test]
    fn weights_sum_to_one_and_pick_the_middle_knots_at_their_times() {
        for t in [0.031, 0.04, 0.055, 0.070] {
            let w = catmull_rom_weights(TIMES, t);
            assert_abs_diff_eq!(w[0] + w[1] + w[2] + w[3], 1.0, epsilon = 1e-12);
        }
        let at_start = catmull_rom_weights(TIMES, TIMES[1]);
        let at_end = catmull_rom_weights(TIMES, TIMES[2]);
        for i in 0..4 {
            assert_abs_diff_eq!(at_start[i], if i == 1 { 1.0 } else { 0.0 }, epsilon = 1e-12);
            assert_abs_diff_eq!(at_end[i], if i == 2 { 1.0 } else { 0.0 }, epsilon = 1e-12);
        }
    }

    #[test]
    fn evenly_spaced_weights_are_the_textbook_catmull_rom_basis() {
        let u = 0.3;
        let w = catmull_rom_weights([-1.0, 0.0, 1.0, 2.0], u);
        let (u2, u3) = (u * u, u * u * u);
        assert_abs_diff_eq!(w[0], 0.5 * (-u3 + 2.0 * u2 - u), epsilon = 1e-12);
        assert_abs_diff_eq!(w[1], 0.5 * (3.0 * u3 - 5.0 * u2 + 2.0), epsilon = 1e-12);
        assert_abs_diff_eq!(w[2], 0.5 * (-3.0 * u3 + 4.0 * u2 + u), epsilon = 1e-12);
        assert_abs_diff_eq!(w[3], 0.5 * (u3 - u2), epsilon = 1e-12);
    }

    #[test]
    fn both_rotation_blends_pass_through_the_middle_knots() {
        for blend in [RotationBlend::Slerp, RotationBlend::Smooth] {
            assert_same_rotation(catmull_rom_rotation(rotations(), TIMES, TIMES[1], blend), rotations()[1], 1e-7);
            assert_same_rotation(catmull_rom_rotation(rotations(), TIMES, TIMES[2], blend), rotations()[2], 1e-7);
        }
    }

    #[test]
    fn smooth_rotation_about_one_axis_is_catmull_rom_of_the_angle() {
        // Rotations about a common axis commute, so the curve must reduce to the scalar one.
        let axis = Vec3::new(0.6, 0.0, 0.8);
        let angles = [0.10, 0.25, 0.20, 0.45];
        let knots = angles.map(|a| SO3::exp(axis * a));
        let t = 0.052;
        let w = catmull_rom_weights(TIMES, t);
        let angle = (0..4).map(|i| w[i] * angles[i]).sum::<f64>();
        assert_same_rotation(
            catmull_rom_rotation(knots, TIMES, t, RotationBlend::Smooth),
            SO3::exp(axis * angle),
            1e-7,
        );
    }

    /// Body-frame angular velocity by central difference, from the four knots starting at `first`.
    fn angular_velocity(knots: &[SO3<f64>], times: &[f64], first: usize, t: f64, blend: RotationBlend) -> Vec3<f64> {
        let k: [SO3<f64>; 4] = std::array::from_fn(|i| knots[first + i]);
        let ts: [f64; 4] = std::array::from_fn(|i| times[first + i]);
        let h = 5e-6;
        let before = catmull_rom_rotation(k, ts, t - h, blend);
        let after = catmull_rom_rotation(k, ts, t + h, blend);
        (before.inverse() * after).log() / (2.0 * h)
    }

    #[test]
    fn smooth_rotation_has_continuous_angular_velocity_across_a_knot() {
        let knots = [
            SO3::exp(Vec3::new(0.10, -0.20, 0.05)),
            SO3::exp(Vec3::new(0.14, -0.17, 0.02)),
            SO3::exp(Vec3::new(0.11, -0.10, 0.06)),
            SO3::exp(Vec3::new(0.05, -0.12, 0.13)),
            SO3::exp(Vec3::new(0.09, -0.02, 0.10)),
        ];
        let times = [0.0, 0.031, 0.070, 0.101, 0.140];
        // Just before knot 2 (segment 0..3) and just after it (segment 1..4).
        // 10 µs either side: the curve's own angular acceleration (hundreds of rad/s² with these
        // knots) accounts for a few mrad/s over that gap.
        let h = 1e-5;
        let left = angular_velocity(&knots, &times, 0, times[2] - h, RotationBlend::Smooth);
        let right = angular_velocity(&knots, &times, 1, times[2] + h, RotationBlend::Smooth);
        assert!((left - right).norm() < 2e-2, "jump of {} rad/s", (left - right).norm());
        // And it is the neighbour-to-neighbour rate, as for position.
        let expected = ((knots[1].inverse() * knots[2]).log() + (knots[2].inverse() * knots[3]).log())
            / (times[3] - times[1]);
        assert!((left - expected).norm() < 2e-2);
        // Slerp, for contrast, jumps there.
        let left = angular_velocity(&knots, &times, 0, times[2] - h, RotationBlend::Slerp);
        let right = angular_velocity(&knots, &times, 1, times[2] + h, RotationBlend::Slerp);
        assert!((left - right).norm() > 0.5);
    }

    #[test]
    fn position_is_the_hermite_curve_with_neighbour_slopes() {
        let p = positions();
        let (j, u) = (1, 0.4);
        let dt = TIMES[2] - TIMES[1];
        let t = TIMES[1] + u * dt;
        let m1 = (p[2] - p[0]) / (TIMES[2] - TIMES[0]);
        let m2 = (p[3] - p[1]) / (TIMES[3] - TIMES[1]);
        let (u2, u3) = (u * u, u * u * u);
        let expected = p[j] * (2.0 * u3 - 3.0 * u2 + 1.0)
            + m1 * (dt * (u3 - 2.0 * u2 + u))
            + p[j + 1] * (-2.0 * u3 + 3.0 * u2)
            + m2 * (dt * (u3 - u2));
        let got = catmull_rom_position(p, TIMES, t);
        assert!((got - expected).norm() < 1e-12);
    }

    #[test]
    fn end_knot_makes_the_end_tangent_the_slope_of_the_end_interval() {
        let pose = |i: usize| SE3::from_rotation_translation(rotations()[i], positions()[i]);
        let (t_outer, outer) = catmull_rom_end_knot((TIMES[0], pose(0)), (TIMES[1], pose(1)));
        assert_abs_diff_eq!(t_outer, -TIMES[1], epsilon = 1e-12);
        let knots = [outer, pose(0), pose(1), pose(2)];
        let times = [t_outer, TIMES[0], TIMES[1], TIMES[2]];
        let h = 1e-6;
        let a = catmull_rom_pose(knots, times, TIMES[0], RotationBlend::Smooth);
        let b = catmull_rom_pose(knots, times, TIMES[0] + h, RotationBlend::Smooth);
        let velocity = (b.translation - a.translation) / h;
        let slope = (positions()[1] - positions()[0]) / (TIMES[1] - TIMES[0]);
        assert!((velocity - slope).norm() < 1e-3);
        let rate = (a.rotation.inverse() * b.rotation).log() / h;
        let turn = (rotations()[0].inverse() * rotations()[1]).log() / (TIMES[1] - TIMES[0]);
        assert!((rate - turn).norm() < 1e-3);
    }

    #[test]
    fn jet_derivatives_match_finite_differences_in_knots_and_time() {
        // Unknowns: a tangent-space nudge of knot 2's rotation (3), of its position (3), and time (1).
        type J = Jet<f64, 7>;
        let observed = Vec3::new(0.2, -0.4, 1.5);
        let eval = |x: [f64; 7]| -> [f64; 3] {
            let mut r = rotations();
            let mut p = positions();
            r[2] = r[2] * SO3::exp(Vec3::new(x[0], x[1], x[2]));
            p[2] = p[2] + Vec3::new(x[3], x[4], x[5]);
            let pose = catmull_rom_pose(
                std::array::from_fn(|i| SE3::from_rotation_translation(r[i], p[i])),
                TIMES,
                0.05 + x[6],
                RotationBlend::Smooth,
            );
            pose.transform_point(observed).into()
        };
        let x: [J; 7] = std::array::from_fn(|i| J::variable(0.0, i));
        let mut r = rotations().map(|k| k.to_constant::<J>());
        let mut p = positions().map(|k| k.to_constant::<J>());
        r[2] = r[2] * SO3::exp(Vec3::new(x[0], x[1], x[2]));
        p[2] = p[2] + Vec3::new(x[3], x[4], x[5]);
        let pose = catmull_rom_pose(
            std::array::from_fn(|i| SE3::from_rotation_translation(r[i], p[i])),
            TIMES,
            x[6] + 0.05,
            RotationBlend::Smooth,
        );
        let out: [J; 3] = pose.transform_point(observed.to_constant::<J>()).into();
        let h = 1e-6;
        for k in 0..7 {
            let (mut plus, mut minus) = ([0.0; 7], [0.0; 7]);
            plus[k] = h;
            minus[k] = -h;
            let (fp, fm) = (eval(plus), eval(minus));
            for row in 0..3 {
                let numeric = (fp[row] - fm[row]) / (2.0 * h);
                assert_abs_diff_eq!(out[row].derivs[k], numeric, epsilon = 1e-6);
            }
        }
    }
}
