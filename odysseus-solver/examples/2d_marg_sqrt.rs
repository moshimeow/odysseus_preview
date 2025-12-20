//! 2D Controller Tracking - SQUARE ROOT (CHOLESKY)
//!
//! This version stores the Cholesky factor L (where H = L^T L) and adds
//! pseudo-residuals to the cost function. No solver API changes needed!

use super::marginalization_common::*;
use nalgebra::{SMatrix, SVector};
use odysseus_solver::{Jet, LevenbergMarquardt};

#[cfg(feature = "visualization")]
use rerun as rr;

const N_OBS_RESIDUALS: usize = 12; // 8 obs + 4 motion
const N_RESIDUALS: usize = N_OBS_RESIDUALS + STATE_DIM; // 16 total

type Residuals = SVector<f64, N_RESIDUALS>;
type Jacobian = SMatrix<f64, N_RESIDUALS, N_PARAMS>;
type JetN = Jet<f64, N_PARAMS>;

// For marginalization computation (no prior)
type ObsResiduals = SVector<f64, N_OBS_RESIDUALS>;
type ObsJacobian = SMatrix<f64, N_OBS_RESIDUALS, N_PARAMS>;

/// Marginalized prior as square root information matrix
struct MarginalizedPrior {
    sqrt_information: SMatrix<f64, STATE_DIM, STATE_DIM>,
    linearization_point: SVector<f64, STATE_DIM>,
}

fn compute_obs_residuals(
    params: &Params,
    observations: &[Observation; 2],
) -> (ObsResiduals, ObsJacobian) {
    let mut residuals = ObsResiduals::zeros();
    let mut jacobian = ObsJacobian::zeros();

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

fn compute_residuals_and_jacobian(
    params: &Params,
    observations: &[Observation; 2],
    prior: Option<&MarginalizedPrior>,
    residuals: &mut Residuals,
    jacobian: &mut Jacobian,
) {
    // First 12 residuals: observations + motion model
    let (obs_residuals, obs_jacobian) = compute_obs_residuals(params, observations);
    residuals.fixed_rows_mut::<N_OBS_RESIDUALS>(0).copy_from(&obs_residuals);
    jacobian.fixed_rows_mut::<N_OBS_RESIDUALS>(0).copy_from(&obs_jacobian);

    // Last 4 residuals: prior (if present)
    if let Some(p) = prior {
        let x = params.fixed_rows::<STATE_DIM>(0);
        let delta = x - p.linearization_point;
        let r_prior = p.sqrt_information * delta;

        residuals
            .fixed_rows_mut::<STATE_DIM>(N_OBS_RESIDUALS)
            .copy_from(&r_prior);

        jacobian
            .fixed_view_mut::<STATE_DIM, STATE_DIM>(N_OBS_RESIDUALS, 0)
            .copy_from(&p.sqrt_information);

        jacobian
            .fixed_view_mut::<STATE_DIM, STATE_DIM>(N_OBS_RESIDUALS, STATE_DIM)
            .fill(0.0);
    } else {
        residuals.fixed_rows_mut::<STATE_DIM>(N_OBS_RESIDUALS).fill(0.0);
        jacobian.fixed_rows_mut::<STATE_DIM>(N_OBS_RESIDUALS).fill(0.0);
    }
}

fn marginalize_oldest_state(params: &Params, observations: &[Observation; 2]) -> MarginalizedPrior {
    let (_residuals, jacobian) = compute_obs_residuals(params, observations);

    let hessian = jacobian.transpose() * jacobian;

    // Partition into blocks: [old, new]
    let h_oo = hessian.fixed_view::<STATE_DIM, STATE_DIM>(0, 0).into_owned();
    let h_on = hessian.fixed_view::<STATE_DIM, STATE_DIM>(0, STATE_DIM).into_owned();
    let h_no = hessian.fixed_view::<STATE_DIM, STATE_DIM>(STATE_DIM, 0).into_owned();
    let h_nn = hessian.fixed_view::<STATE_DIM, STATE_DIM>(STATE_DIM, STATE_DIM).into_owned();

    // Schur complement
    let h_oo_inv = h_oo.try_inverse().expect("H_oo must be invertible");
    let h_marg = h_nn - h_no * h_oo_inv * h_on;

    // Cholesky factorization: H_marg = L * L^T
    let cholesky = h_marg.cholesky().expect("H_marg must be positive definite");
    let sqrt_info = cholesky.l();

    let lin_point = params.fixed_rows::<STATE_DIM>(STATE_DIM).into_owned();

    MarginalizedPrior {
        sqrt_information: sqrt_info,
        linearization_point: lin_point,
    }
}

/// Run square root marginalization and return estimated trajectory
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

        prior = Some(marginalize_oldest_state(&result, &window_obs));
    }

    estimated_trajectory
}
