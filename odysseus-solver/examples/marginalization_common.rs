//! Shared code for 2D controller marginalization examples
//!
//! This module contains common simulation, observation generation, and types
//! used across all three 2D marginalization implementations.

use nalgebra::{SMatrix, SVector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "visualization")]
use rerun as rr;

// Shared constants
pub const STATE_DIM: usize = 4; // [px, py, vx, vy]
pub const WINDOW_SIZE: usize = 2;
pub const N_PARAMS: usize = STATE_DIM * WINDOW_SIZE; // 8 parameters
pub const N_STEPS: usize = 200;
pub const DT: f64 = 0.1;

// Noise parameters
pub const POS_NOISE_STD: f64 = 0.1;
pub const VEL_NOISE_STD: f64 = 0.2;
pub const MOTION_NOISE_STD: f64 = 0.1;
pub const VELOCITY_DIFFUSION: f64 = 0.5;

pub type Params = SVector<f64, N_PARAMS>;

/// State at a single time step (2D position and velocity)
#[derive(Debug, Clone, Copy)]
pub struct State {
    pub px: f64,
    pub py: f64,
    pub vx: f64,
    pub vy: f64,
}

impl State {
    pub fn new(px: f64, py: f64, vx: f64, vy: f64) -> Self {
        Self { px, py, vx, vy }
    }

    pub fn to_vector(&self) -> SVector<f64, STATE_DIM> {
        SVector::<f64, STATE_DIM>::from_row_slice(&[self.px, self.py, self.vx, self.vy])
    }

    pub fn from_vector(v: &SVector<f64, STATE_DIM>) -> Self {
        Self::new(v[0], v[1], v[2], v[3])
    }

    pub fn position_error(&self, other: &State) -> f64 {
        let dx = self.px - other.px;
        let dy = self.py - other.py;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn velocity_error(&self, other: &State) -> f64 {
        let dvx = self.vx - other.vx;
        let dvy = self.vy - other.vy;
        (dvx * dvx + dvy * dvy).sqrt()
    }
}

/// Observation at a single time step
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub px: f64,
    pub py: f64,
    pub vx: f64,
    pub vy: f64,
}

/// Convert parameters vector to array of states
pub fn params_to_states(params: &Params) -> [State; WINDOW_SIZE] {
    [
        State::new(params[0], params[1], params[2], params[3]),
        State::new(params[4], params[5], params[6], params[7]),
    ]
}

/// Simulate Brownian motion trajectory
pub fn simulate_brownian_motion(seed: u64) -> Vec<State> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut states = Vec::new();

    let mut state = State::new(0.0, 0.0, 0.1, 0.05);
    states.push(state);

    for _ in 1..N_STEPS {
        state.vx += rng.gen_range(-VELOCITY_DIFFUSION..VELOCITY_DIFFUSION);
        state.vy += rng.gen_range(-VELOCITY_DIFFUSION..VELOCITY_DIFFUSION);

        state = State::new(
            state.px + state.vx * DT,
            state.py + state.vy * DT,
            state.vx,
            state.vy,
        );
        states.push(state);
    }

    states
}

/// Generate noisy observations from true states
pub fn generate_observations(true_states: &[State], seed: u64) -> Vec<Observation> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    true_states
        .iter()
        .map(|state| Observation {
            px: state.px + rng.gen_range(-POS_NOISE_STD..POS_NOISE_STD),
            py: state.py + rng.gen_range(-POS_NOISE_STD..POS_NOISE_STD),
            vx: state.vx + rng.gen_range(-VEL_NOISE_STD..VEL_NOISE_STD),
            vy: state.vy + rng.gen_range(-VEL_NOISE_STD..VEL_NOISE_STD),
        })
        .collect()
}

/// Compute RMSE between estimated and true states
pub fn compute_rmse(estimated: &[State], truth: &[State]) -> (f64, f64) {
    assert_eq!(estimated.len(), truth.len());

    let n = estimated.len() as f64;
    let mut sum_pos_sq = 0.0;
    let mut sum_vel_sq = 0.0;

    for (est, tru) in estimated.iter().zip(truth.iter()) {
        let pos_err = est.position_error(tru);
        let vel_err = est.velocity_error(tru);
        sum_pos_sq += pos_err * pos_err;
        sum_vel_sq += vel_err * vel_err;
    }

    let rmse_pos = (sum_pos_sq / n).sqrt();
    let rmse_vel = (sum_vel_sq / n).sqrt();

    (rmse_pos, rmse_vel)
}

/// Print window state comparison
pub fn print_window_error(estimated: &[State], true_states: &[State]) {
    for (i, (est, truth)) in estimated.iter().zip(true_states.iter()).enumerate() {
        println!(
            "    [{}] est: p=({:6.3}, {:6.3}), v=({:6.3}, {:6.3})  |  err: Δp={:.4}, Δv={:.4}",
            i,
            est.px,
            est.py,
            est.vx,
            est.vy,
            est.position_error(truth),
            est.velocity_error(truth)
        );
    }
}

/// Log trajectories to Rerun
#[cfg(feature = "visualization")]
pub fn log_trajectory(
    rec: &rr::RecordingStream,
    iteration: usize,
    true_states: &[State],
    estimated_trajectory: &[State],
    marginalized_trajectory: &[State],
) {
    rec.set_time_sequence("step", iteration as i64);

    // Log full ground truth trajectory (3D: [x, y, time])
    let true_positions: Vec<[f32; 3]> = true_states
        .iter()
        .enumerate()
        .map(|(i, s)| [s.px as f32, s.py as f32, i as f32 * DT as f32])
        .collect();

    rec.log(
        "trajectory/ground_truth",
        &rr::LineStrips3D::new([true_positions.clone()])
            .with_colors([[150, 150, 150, 255]]),
    )
    .unwrap();

    rec.log(
        "trajectory/ground_truth_points",
        &rr::Points3D::new(true_positions)
            .with_colors([[150, 150, 150, 255]])
            .with_radii([0.02]),
    )
    .unwrap();

    // Log estimated trajectory
    let est_positions: Vec<[f32; 3]> = estimated_trajectory
        .iter()
        .enumerate()
        .map(|(i, s)| [s.px as f32, s.py as f32, i as f32 * DT as f32])
        .collect();

    rec.log(
        "trajectory/estimated",
        &rr::LineStrips3D::new([est_positions.clone()])
            .with_colors([[100, 100, 255, 255]]),
    )
    .unwrap();

    rec.log(
        "trajectory/estimated_points",
        &rr::Points3D::new(est_positions)
            .with_colors([[100, 100, 255, 255]])
            .with_radii([0.025]),
    )
    .unwrap();

    // Log marginalized prior trajectory
    if !marginalized_trajectory.is_empty() {
        let marg_positions: Vec<[f32; 3]> = marginalized_trajectory
            .iter()
            .enumerate()
            .map(|(i, s)| [s.px as f32, s.py as f32, (i + 1) as f32 * DT as f32])
            .collect();

        rec.log(
            "trajectory/marginalized",
            &rr::LineStrips3D::new([marg_positions.clone()])
                .with_colors([[255, 150, 100, 255]]),
        )
        .unwrap();

        rec.log(
            "trajectory/marginalized_points",
            &rr::Points3D::new(marg_positions)
                .with_colors([[255, 150, 100, 255]])
                .with_radii([0.025]),
        )
        .unwrap();
    }

    // Also log 2D top-down view (X-Y plane)
    let true_positions_2d: Vec<[f32; 2]> = true_states
        .iter()
        .map(|s| [s.px as f32, s.py as f32])
        .collect();

    rec.log(
        "trajectory_2d/ground_truth",
        &rr::LineStrips2D::new([true_positions_2d.clone()])
            .with_colors([[150, 150, 150, 255]]),
    )
    .unwrap();

    let est_positions_2d: Vec<[f32; 2]> = estimated_trajectory
        .iter()
        .map(|s| [s.px as f32, s.py as f32])
        .collect();

    rec.log(
        "trajectory_2d/estimated",
        &rr::LineStrips2D::new([est_positions_2d])
            .with_colors([[100, 100, 255, 255]]),
    )
    .unwrap();

    if !marginalized_trajectory.is_empty() {
        let marg_positions_2d: Vec<[f32; 2]> = marginalized_trajectory
            .iter()
            .map(|s| [s.px as f32, s.py as f32])
            .collect();

        rec.log(
            "trajectory_2d/marginalized",
            &rr::LineStrips2D::new([marg_positions_2d])
                .with_colors([[255, 150, 100, 255]]),
        )
        .unwrap();
    }
}
