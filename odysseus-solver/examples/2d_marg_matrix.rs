//! 2D Controller Tracking - MATRIX (SCHUR COMPLEMENT)
//!
//! This version computes the Schur complement to marginalize and feeds
//! the resulting information matrix into the solver via solve_with_prior().

use super::marginalization_common::*;
use nalgebra::{SMatrix, SVector};
use odysseus_solver::{Jet, LevenbergMarquardt};

#[cfg(feature = "visualization")]
use rerun as rr;

const N_RESIDUALS: usize = 12; // 8 obs + 4 motion

type Residuals = SVector<f64, N_RESIDUALS>;
type Jacobian = SMatrix<f64, N_RESIDUALS, N_PARAMS>;
type JetN = Jet<f64, N_PARAMS>;

/// Marginalized prior as information matrix
struct MarginalizedPrior {
    hessian: SMatrix<f64, STATE_DIM, STATE_DIM>,
    rhs: SVector<f64, STATE_DIM>,
    linearization_point: SVector<f64, STATE_DIM>,
}

fn compute_residuals_and_jacobian_full(
    params: &Params,
    observations: &[Observation; 2],
) -> (Residuals, Jacobian) {
    let mut residuals = Residuals::zeros();
    let mut jacobian = Jacobian::zeros();

    let s = [
        JetN::variable(params[0], 0),
        JetN::variable(params[1], 1),
        JetN::variable(params[2], 2),
        JetN::variable(params[3], 3),
        JetN::variable(params[4], 4),
        JetN::variable(params[5], 5),
        JetN::variable(params[6], 6),
        JetN::variable(params[7], 7),
    ];

    let obs_weight_p = 1.0 / POS_NOISE_STD;
    let obs_weight_v = 1.0 / VEL_NOISE_STD;

    let r0 = [
        (s[0] - JetN::constant(observations[0].px)) * JetN::constant(obs_weight_p),
        (s[1] - JetN::constant(observations[0].py)) * JetN::constant(obs_weight_p),
        (s[2] - JetN::constant(observations[0].vx)) * JetN::constant(obs_weight_v),
        (s[3] - JetN::constant(observations[0].vy)) * JetN::constant(obs_weight_v),
    ];

    let r1 = [
        (s[4] - JetN::constant(observations[1].px)) * JetN::constant(obs_weight_p),
        (s[5] - JetN::constant(observations[1].py)) * JetN::constant(obs_weight_p),
        (s[6] - JetN::constant(observations[1].vx)) * JetN::constant(obs_weight_v),
        (s[7] - JetN::constant(observations[1].vy)) * JetN::constant(obs_weight_v),
    ];

    for i in 0..4 {
        residuals[i] = r0[i].value;
        residuals[4 + i] = r1[i].value;
        for j in 0..N_PARAMS {
            jacobian[(i, j)] = r0[i].derivs[j];
            jacobian[(4 + i, j)] = r1[i].derivs[j];
        }
    }

    let motion_weight = 1.0 / MOTION_NOISE_STD;
    let r_motion = [
        (s[4] - (s[0] + s[2] * JetN::constant(DT))) * JetN::constant(motion_weight),
        (s[5] - (s[1] + s[3] * JetN::constant(DT))) * JetN::constant(motion_weight),
        (s[6] - s[2]) * JetN::constant(motion_weight),
        (s[7] - s[3]) * JetN::constant(motion_weight),
    ];

    for i in 0..4 {
        residuals[8 + i] = r_motion[i].value;
        for j in 0..N_PARAMS {
            jacobian[(8 + i, j)] = r_motion[i].derivs[j];
        }
    }

    (residuals, jacobian)
}

fn marginalize_oldest_state(params: &Params, observations: &[Observation; 2]) -> MarginalizedPrior {
    let (residuals, jacobian) = compute_residuals_and_jacobian_full(params, observations);

    let hessian = jacobian.transpose() * jacobian;
    let rhs = -jacobian.transpose() * residuals;

    // Partition into blocks: [old, new]
    let h_oo = hessian.fixed_view::<STATE_DIM, STATE_DIM>(0, 0).into_owned();
    let h_on = hessian.fixed_view::<STATE_DIM, STATE_DIM>(0, STATE_DIM).into_owned();
    let h_no = hessian.fixed_view::<STATE_DIM, STATE_DIM>(STATE_DIM, 0).into_owned();
    let h_nn = hessian.fixed_view::<STATE_DIM, STATE_DIM>(STATE_DIM, STATE_DIM).into_owned();

    let b_o = rhs.fixed_rows::<STATE_DIM>(0).into_owned();
    let b_n = rhs.fixed_rows::<STATE_DIM>(STATE_DIM).into_owned();

    // Schur complement
    let h_oo_inv = h_oo.try_inverse().expect("H_oo must be invertible");
    let h_marg = h_nn - h_no * h_oo_inv * h_on;
    let b_marg = b_n - h_no * h_oo_inv * b_o;

    let lin_point = params.fixed_rows::<STATE_DIM>(STATE_DIM).into_owned();

    MarginalizedPrior {
        hessian: h_marg,
        rhs: b_marg,
        linearization_point: lin_point,
    }
}

fn optimize_window(
    initial_params: &Params,
    observations: &[Observation; 2],
    prior: Option<&MarginalizedPrior>,
) -> Params {
    let mut solver = LevenbergMarquardt::<f64, N_PARAMS, N_RESIDUALS>::new().with_verbose(false);

    if let Some(p) = prior {
        let mut full_hessian = SMatrix::<f64, N_PARAMS, N_PARAMS>::zeros();
        let mut full_rhs = SVector::<f64, N_PARAMS>::zeros();

        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                full_hessian[(i, j)] = p.hessian[(i, j)];
            }
            full_rhs[i] = p.rhs[i];
        }

        solver.solve_with_prior(
            *initial_params,
            |params, residuals, jacobian| {
                let (r, j) = compute_residuals_and_jacobian_full(params, observations);
                *residuals = r;
                *jacobian = j;
            },
            |_, _, _| {},
            Some(&full_hessian),
            Some(&full_rhs),
        )
    } else {
        solver.solve(
            *initial_params,
            |params, residuals, jacobian| {
                let (r, j) = compute_residuals_and_jacobian_full(params, observations);
                *residuals = r;
                *jacobian = j;
            },
            |_, _, _| {},
        )
    }
}

/// Run matrix marginalization and return estimated trajectory
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

        let result = if let Some(p) = &prior {
            let mut full_hessian = SMatrix::<f64, N_PARAMS, N_PARAMS>::zeros();
            let mut full_rhs = SVector::<f64, N_PARAMS>::zeros();

            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    full_hessian[(i, j)] = p.hessian[(i, j)];
                }
                full_rhs[i] = p.rhs[i];
            }

            solver.solve_with_prior(
                initial_params,
                |params, residuals, jacobian| {
                    let (r, j) = compute_residuals_and_jacobian_full(params, &window_obs);
                    *residuals = r;
                    *jacobian = j;
                },
                |_, _, _| {},
                Some(&full_hessian),
                Some(&full_rhs),
            )
        } else {
            solver.solve(
                initial_params,
                |params, residuals, jacobian| {
                    let (r, j) = compute_residuals_and_jacobian_full(params, &window_obs);
                    *residuals = r;
                    *jacobian = j;
                },
                |_, _, _| {},
            )
        };

        let window_states = params_to_states(&result);
        estimated_trajectory.push(window_states[0]);
        if step == N_STEPS - 2 {
            estimated_trajectory.push(window_states[1]);
        }

        prior = Some(marginalize_oldest_state(&result, &window_obs));
    }

    estimated_trajectory
}
