//! 2D Controller Tracking - WRITTEN-OUT FORMULAS
//!
//! This version uses explicit analytical formulas for marginalization.
//! Each state dimension is treated independently with explicit variance propagation.

use super::marginalization_common::*;
use nalgebra::{SMatrix, SVector};
use odysseus_solver::{Jet, LevenbergMarquardt};

#[cfg(feature = "visualization")]
use rerun as rr;

const N_RESIDUALS: usize = 12; // 8 obs + 4 motion

type Residuals = SVector<f64, N_RESIDUALS>;
type Jacobian = SMatrix<f64, N_RESIDUALS, N_PARAMS>;

/// Marginalized prior (independent variances per dimension)
#[derive(Debug, Clone, Copy)]
struct MarginalizedPrior {
    px_mean: f64,
    py_mean: f64,
    vx_mean: f64,
    vy_mean: f64,
    px_var: f64,
    py_var: f64,
    vx_var: f64,
    vy_var: f64,
}

fn marginalize_oldest_state(obs_old: &Observation) -> MarginalizedPrior {
    // Propagate position: px₁ = px₀ + vx₀·dt
    let px_mean = obs_old.px + obs_old.vx * DT;
    let py_mean = obs_old.py + obs_old.vy * DT;
    let p_var = POS_NOISE_STD.powi(2) + (DT * VEL_NOISE_STD).powi(2) + MOTION_NOISE_STD.powi(2);

    // Propagate velocity: vx₁ = vx₀
    let vx_mean = obs_old.vx;
    let vy_mean = obs_old.vy;
    let v_var = VEL_NOISE_STD.powi(2) + MOTION_NOISE_STD.powi(2);

    MarginalizedPrior {
        px_mean,
        py_mean,
        vx_mean,
        vy_mean,
        px_var: p_var,
        py_var: p_var,
        vx_var: v_var,
        vy_var: v_var,
    }
}

fn compute_residuals_and_jacobian(
    params: &Params,
    observations: &[Observation; 2],
    prior: Option<&MarginalizedPrior>,
    residuals: &mut Residuals,
    jacobian: &mut Jacobian,
) {
    type Jet4 = Jet<f64, STATE_DIM>;

    let s0 = [
        Jet4::variable(params[0], 0),
        Jet4::variable(params[1], 1),
        Jet4::variable(params[2], 2),
        Jet4::variable(params[3], 3),
    ];
    let s1 = [
        Jet4::variable(params[4], 0),
        Jet4::variable(params[5], 1),
        Jet4::variable(params[6], 2),
        Jet4::variable(params[7], 3),
    ];

    let obs_weight_p = 1.0 / POS_NOISE_STD;
    let obs_weight_v = 1.0 / VEL_NOISE_STD;

    // Observation residuals for state 0
    let r0 = if let Some(p) = prior {
        // Replace with prior
        let prior_weight_p = 1.0 / p.px_var.sqrt();
        let prior_weight_v = 1.0 / p.vx_var.sqrt();
        [
            (s0[0] - Jet4::constant(p.px_mean)) * Jet4::constant(prior_weight_p),
            (s0[1] - Jet4::constant(p.py_mean)) * Jet4::constant(prior_weight_p),
            (s0[2] - Jet4::constant(p.vx_mean)) * Jet4::constant(prior_weight_v),
            (s0[3] - Jet4::constant(p.vy_mean)) * Jet4::constant(prior_weight_v),
        ]
    } else {
        [
            (s0[0] - Jet4::constant(observations[0].px)) * Jet4::constant(obs_weight_p),
            (s0[1] - Jet4::constant(observations[0].py)) * Jet4::constant(obs_weight_p),
            (s0[2] - Jet4::constant(observations[0].vx)) * Jet4::constant(obs_weight_v),
            (s0[3] - Jet4::constant(observations[0].vy)) * Jet4::constant(obs_weight_v),
        ]
    };

    // Observation residuals for state 1
    let r1 = [
        (s1[0] - Jet4::constant(observations[1].px)) * Jet4::constant(obs_weight_p),
        (s1[1] - Jet4::constant(observations[1].py)) * Jet4::constant(obs_weight_p),
        (s1[2] - Jet4::constant(observations[1].vx)) * Jet4::constant(obs_weight_v),
        (s1[3] - Jet4::constant(observations[1].vy)) * Jet4::constant(obs_weight_v),
    ];

    for i in 0..4 {
        residuals[i] = r0[i].value;
        residuals[4 + i] = r1[i].value;
        for j in 0..STATE_DIM {
            jacobian[(i, j)] = r0[i].derivs[j];
            jacobian[(i, STATE_DIM + j)] = 0.0;
            jacobian[(4 + i, j)] = 0.0;
            jacobian[(4 + i, STATE_DIM + j)] = r1[i].derivs[j];
        }
    }

    // Motion model residuals
    let motion_weight = 1.0 / MOTION_NOISE_STD;
    let r_motion = [
        (s1[0] - (s0[0] + s0[2] * Jet4::constant(DT))) * Jet4::constant(motion_weight),
        (s1[1] - (s0[1] + s0[3] * Jet4::constant(DT))) * Jet4::constant(motion_weight),
        (s1[2] - s0[2]) * Jet4::constant(motion_weight),
        (s1[3] - s0[3]) * Jet4::constant(motion_weight),
    ];

    for i in 0..4 {
        residuals[8 + i] = r_motion[i].value;
        for j in 0..STATE_DIM {
            jacobian[(8 + i, j)] = r_motion[i].derivs[j];
            jacobian[(8 + i, STATE_DIM + j)] = 0.0;
        }
    }
}

fn optimize_window(
    initial_params: &Params,
    observations: &[Observation; 2],
    prior: Option<MarginalizedPrior>,
) -> Params {
    let mut solver = LevenbergMarquardt::<f64, N_PARAMS, N_RESIDUALS>::new().with_verbose(false);

    solver.solve(
        *initial_params,
        |params, residuals, jacobian| {
            compute_residuals_and_jacobian(params, observations, prior.as_ref(), residuals, jacobian);
        },
        |_, _, _| {},
    )
}

/// Run written-out marginalization and return estimated trajectory
pub fn run(observations: &[Observation]) -> Vec<State> {
    let mut solver = LevenbergMarquardt::<f64, N_PARAMS, N_RESIDUALS>::new().with_verbose(false);
    let mut estimated_trajectory = Vec::new();
    let mut prior: Option<MarginalizedPrior> = None;

    for step in 0..(N_STEPS - 1) {
        let window_obs = [observations[step], observations[step + 1]];

        let initial_params = Params::from_column_slice(&[
            window_obs[0].px, window_obs[0].py, window_obs[0].vx, window_obs[0].vy,
            window_obs[1].px, window_obs[1].py, window_obs[1].vx, window_obs[1].vy,
        ]);

        let result = solver.solve(
            initial_params,
            |params, residuals, jacobian| {
                compute_residuals_and_jacobian(params, &window_obs, prior.as_ref(), residuals, jacobian);
            },
            |_, _, _| {},
        );

        let window_states = params_to_states(&result);
        estimated_trajectory.push(window_states[0]);
        if step == N_STEPS - 2 {
            estimated_trajectory.push(window_states[1]);
        }

        prior = Some(marginalize_oldest_state(&window_obs[0]));
    }

    estimated_trajectory
}
