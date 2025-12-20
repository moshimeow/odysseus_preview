//! 1D Controller Tracking with Sliding Window Marginalization
//!
//! This example demonstrates sliding window optimization with marginalization
//! using EXPLICIT EQUATIONS instead of matrix operations.
//!
//! We track position and velocity of a 1D controller, maintaining a window of
//! just 2 time steps at a time. When we add a new observation, we marginalize
//! out the oldest time step to get priors on the remaining states.
//!
//! ## The Idea
//!
//! For 2 time steps (t=0, t=1), we optimize:
//!   cost = (x₀ - x_obs₀)²/σ²_x               [position observation at t=0]
//!        + (v₀ - v_obs₀)²/σ²_v               [velocity observation at t=0]
//!        + (x₁ - x₀ - v₀·dt)²/σ²_motion      [motion model: position prediction]
//!        + (v₁ - v₀)²/σ²_motion              [motion model: velocity prediction]
//!        + (x₁ - x_obs₁)²/σ²_x               [position observation at t=1]
//!        + (v₁ - v_obs₁)²/σ²_v               [velocity observation at t=1]
//!
//! After optimizing, we marginalize out (x₀, v₀) to get priors on (x₁, v₁):
//!   x₁_prior_mean = x_obs₀ + v_obs₀·dt
//!   x₁_prior_var = σ²_x + (dt)²·σ²_v + σ²_motion
//!
//!   v₁_prior_mean = v_obs₀
//!   v₁_prior_var = σ²_v + σ²_motion
//!
//! These priors replace the observations and motion model from t=0.
//!
//! ## Running with Visualization
//!
//! ```bash
//! # First install Rerun viewer (one-time setup)
//! cargo install rerun-cli
//!
//! # Run with visualization
//! cargo run -p odysseus-solver --example 1d_controller_marginalization --features visualization
//! ```

use nalgebra::{SMatrix, SVector};
use odysseus_solver::{Jet, LevenbergMarquardt};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "visualization")]
use rerun as rr;

// Problem dimensions - WINDOW OF 2 TIME STEPS
const STATE_DIM: usize = 2; // [position, velocity]
const WINDOW_SIZE: usize = 2; // Just 2 time steps in window
const N_PARAMS: usize = STATE_DIM * WINDOW_SIZE; // 4 parameters total
const N_STEPS: usize = 200; // Total simulation steps

// Residual dimensions
// For 2 time steps, we have:
//   - 2 observation residuals per time step (pos, vel) = 4 total
//   - 2 motion model residuals connecting t=0 to t=1 = 2 total
// Total = 6 residuals
const N_RESIDUALS: usize = 6;

// Simulation parameters
const DT: f64 = 0.1; // Time step (seconds)
const VELOCITY_DIFFUSION: f64 = 0.5; // Random walk on velocity

// Noise parameters
const POS_NOISE_STD: f64 = 0.1; // σ_x (position measurement noise)
const VEL_NOISE_STD: f64 = 0.2; // σ_v (velocity measurement noise)
const MOTION_NOISE_STD: f64 = 0.1; // σ_motion (how much motion model can be wrong)

type Params = SVector<f64, N_PARAMS>;
type Residuals = SVector<f64, N_RESIDUALS>;
type Jacobian = SMatrix<f64, N_RESIDUALS, N_PARAMS>;
type JetN = Jet<f64, N_PARAMS>;

/// State at a single time step
#[derive(Debug, Clone, Copy)]
struct State {
    position: f64,
    velocity: f64,
}

impl State {
    fn new(position: f64, velocity: f64) -> Self {
        Self { position, velocity }
    }
}

/// Observation at a single time step
#[derive(Debug, Clone, Copy)]
struct Observation {
    position: f64,
    velocity: f64,
}

/// Marginalized prior on a single state (position and velocity)
#[derive(Debug, Clone, Copy)]
struct MarginalizedPrior {
    /// Prior mean for position
    x_mean: f64,
    /// Prior mean for velocity
    v_mean: f64,
    /// Prior variance for position
    x_var: f64,
    /// Prior variance for velocity
    v_var: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "visualization")]
    let rec = rr::RecordingStreamBuilder::new("1d_controller_tracking").spawn()?;

    println!("🎮 1D Controller Tracking with Marginalization");
    println!("================================================\n");

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Simulate ground truth trajectory with Brownian motion
    let true_states = simulate_brownian_motion(&mut rng, N_STEPS);

    // Generate noisy observations
    let observations = generate_observations(&true_states, &mut rng);

    println!("📊 Simulation:");
    println!("  Total time steps: {}", N_STEPS);
    println!("  Window size: {} (just 2 time steps!)", WINDOW_SIZE);
    println!("  σ_x (pos obs noise): {:.3}", POS_NOISE_STD);
    println!("  σ_v (vel obs noise): {:.3}", VEL_NOISE_STD);
    println!("  σ_motion: {:.3}\n", MOTION_NOISE_STD);

    // Track estimated states and marginalized priors over time
    let mut estimated_trajectory = Vec::new();
    let mut marginalized_trajectory = Vec::new();

    // Initialize with first 2 observations
    println!("🔧 Step 0-1: Initial optimization (no prior)");
    let mut params = Params::from_row_slice(&[
        observations[0].position,
        observations[0].velocity,
        observations[1].position,
        observations[1].velocity,
    ]);

    params = optimize_window(&params, &[observations[0], observations[1]], None);

    let mut window_states = params_to_states(&params);
    print_window_error(&window_states, &[true_states[0], true_states[1]]);

    // Store initial estimates
    estimated_trajectory.push(window_states[0]);
    estimated_trajectory.push(window_states[1]);

    #[cfg(feature = "visualization")]
    log_trajectory(&rec, 0, &true_states, &estimated_trajectory, &marginalized_trajectory);

    // Slide window forward, marginalizing oldest state each time
    for step in 2..N_STEPS {
        println!("\n📍 Step {}: Marginalize t={}, optimize t={}-{}",
                 step, step - 2, step - 1, step);

        // 1. Marginalize out the oldest state (t = step - 2)
        // This gives us priors on the state at t = step - 1
        let prior = marginalize_oldest_state(&params, &observations[step - 2], &observations[step - 1]);

        println!("  Marginalized prior for t={}:", step - 1);
        println!("    x_mean={:.4}, x_var={:.6}", prior.x_mean, prior.x_var);
        println!("    v_mean={:.4}, v_var={:.6}", prior.v_mean, prior.v_var);

        // Store marginalized state
        let marginalized_state = State::new(prior.x_mean, prior.v_mean);
        marginalized_trajectory.push(marginalized_state);

        // 2. Set up new window: [t = step-1, t = step]
        // Initialize with observations
        params = Params::from_row_slice(&[
            observations[step - 1].position,
            observations[step - 1].velocity,
            observations[step].position,
            observations[step].velocity,
        ]);

        // 3. Optimize with marginalized prior
        params = optimize_window(&params, &[observations[step - 1], observations[step]], Some(prior));

        window_states = params_to_states(&params);
        print_window_error(&window_states, &[true_states[step - 1], true_states[step]]);

        // Update trajectory (replace last estimate with new optimized one, add new state)
        estimated_trajectory[step - 1] = window_states[0];
        estimated_trajectory.push(window_states[1]);

        #[cfg(feature = "visualization")]
        log_trajectory(&rec, step - 1, &true_states, &estimated_trajectory, &marginalized_trajectory);
    }

    println!("\n✅ Sliding window optimization complete!");

    #[cfg(feature = "visualization")]
    println!("\n📺 Open Rerun to see the visualization!");

    Ok(())
}

/// Simulate 1D Brownian motion
fn simulate_brownian_motion(rng: &mut ChaCha8Rng, n_steps: usize) -> Vec<State> {
    let mut states = Vec::with_capacity(n_steps);
    let mut state = State::new(0.0, 0.0);

    states.push(state);

    for _ in 1..n_steps {
        // Velocity random walk
        let velocity_change = rng.gen_range(-VELOCITY_DIFFUSION..VELOCITY_DIFFUSION);
        state.velocity += velocity_change;

        // Position integration
        state.position += state.velocity * DT;

        states.push(state);
    }

    states
}

/// Generate noisy observations from true states
fn generate_observations(true_states: &[State], rng: &mut ChaCha8Rng) -> Vec<Observation> {
    true_states
        .iter()
        .map(|state| {
            let pos_noise = rng.gen_range(-POS_NOISE_STD..POS_NOISE_STD);
            let vel_noise = rng.gen_range(-VEL_NOISE_STD..VEL_NOISE_STD);

            Observation {
                position: state.position + pos_noise,
                velocity: state.velocity + vel_noise,
            }
        })
        .collect()
}

/// Convert parameter vector to state vector
fn params_to_states(params: &Params) -> Vec<State> {
    (0..WINDOW_SIZE)
        .map(|i| State::new(params[i * STATE_DIM], params[i * STATE_DIM + 1]))
        .collect()
}

/// Optimize window with optional marginalized prior on FIRST state
fn optimize_window(
    initial_params: &Params,
    observations: &[Observation; 2],
    prior: Option<MarginalizedPrior>,
) -> Params {
    let cost_fn = |params: &Params,
                   residuals: &mut Residuals,
                   jacobian: &mut Jacobian| {
        compute_residuals_and_jacobian(params, observations, prior, residuals, jacobian);
    };

    let mut solver = LevenbergMarquardt::<f64, N_PARAMS, N_RESIDUALS>::new()
        .with_tolerance(1e-10)
        .with_max_iterations(20)
        .with_verbose(false);

    solver.solve(*initial_params, cost_fn, |_iter, _result, _params| {})
}

/// Compute residuals and Jacobian using explicit equations
///
/// Window contains 2 time steps: t=0 and t=1 (local indices)
///
/// Parameters: [x₀, v₀, x₁, v₁]
///
/// Residuals (6 total):
///   r[0] = (x₀ - prior.x_mean) / √x_var     [prior on x₀] OR (x₀ - x_obs₀) / σ_x
///   r[1] = (v₀ - prior.v_mean) / √v_var     [prior on v₀] OR (v₀ - v_obs₀) / σ_v
///   r[2] = (x₁ - x₀ - v₀·dt) / σ_motion     [motion model: position]
///   r[3] = (v₁ - v₀) / σ_motion             [motion model: velocity]
///   r[4] = (x₁ - x_obs₁) / σ_x              [observation at t=1]
///   r[5] = (v₁ - v_obs₁) / σ_v              [observation at t=1]
fn compute_residuals_and_jacobian(
    params: &Params,
    observations: &[Observation; 2],
    prior: Option<MarginalizedPrior>,
    residuals: &mut Residuals,
    jacobian: &mut Jacobian,
) {
    // Create Jets for automatic differentiation
    let x0 = JetN::variable(params[0], 0); // position at t=0
    let v0 = JetN::variable(params[1], 1); // velocity at t=0
    let x1 = JetN::variable(params[2], 2); // position at t=1
    let v1 = JetN::variable(params[3], 3); // velocity at t=1

    let dt = JetN::constant(DT);

    // Residual 0 & 1: Prior or observation at t=0
    let (r0, r1) = if let Some(p) = prior {
        // Use marginalized prior
        let w_x = 1.0 / p.x_var.sqrt();
        let w_v = 1.0 / p.v_var.sqrt();

        let r_x = (x0 - JetN::constant(p.x_mean)) * JetN::constant(w_x);
        let r_v = (v0 - JetN::constant(p.v_mean)) * JetN::constant(w_v);

        (r_x, r_v)
    } else {
        // Use observations at t=0
        let w_x = 1.0 / POS_NOISE_STD;
        let w_v = 1.0 / VEL_NOISE_STD;

        let r_x = (x0 - JetN::constant(observations[0].position)) * JetN::constant(w_x);
        let r_v = (v0 - JetN::constant(observations[0].velocity)) * JetN::constant(w_v);

        (r_x, r_v)
    };

    // Residual 2 & 3: Motion model connecting t=0 to t=1
    let w_motion = 1.0 / MOTION_NOISE_STD;

    // Position prediction: x₁ should equal x₀ + v₀·dt
    let r_motion_x = (x1 - x0 - v0 * dt) * JetN::constant(w_motion);

    // Velocity prediction: v₁ should equal v₀ (constant velocity)
    let r_motion_v = (v1 - v0) * JetN::constant(w_motion);

    // Residual 4 & 5: Observations at t=1
    let w_x = 1.0 / POS_NOISE_STD;
    let w_v = 1.0 / VEL_NOISE_STD;

    let r_obs_x1 = (x1 - JetN::constant(observations[1].position)) * JetN::constant(w_x);
    let r_obs_v1 = (v1 - JetN::constant(observations[1].velocity)) * JetN::constant(w_v);

    // Extract values and derivatives
    residuals[0] = r0.value;
    residuals[1] = r1.value;
    residuals[2] = r_motion_x.value;
    residuals[3] = r_motion_v.value;
    residuals[4] = r_obs_x1.value;
    residuals[5] = r_obs_v1.value;

    // Jacobian (6 residuals × 4 parameters)
    for j in 0..N_PARAMS {
        jacobian[(0, j)] = r0.derivs[j];
        jacobian[(1, j)] = r1.derivs[j];
        jacobian[(2, j)] = r_motion_x.derivs[j];
        jacobian[(3, j)] = r_motion_v.derivs[j];
        jacobian[(4, j)] = r_obs_x1.derivs[j];
        jacobian[(5, j)] = r_obs_v1.derivs[j];
    }
}

/// Marginalize out the oldest state using EXPLICIT FORMULAS
///
/// Given optimized states at t=0 and t=1, we want to eliminate t=0
/// and get priors on t=1.
///
/// The formulas are derived from the cost function. After optimizing,
/// the state at t=0 is influenced by:
///   - Its observation: x_obs₀, v_obs₀
///   - The motion model connecting it to t=1
///
/// The marginalized prior "propagates" information from t=0 to t=1:
///
///   x₁_prior_mean = x_obs₀ + v_obs₀·dt
///   x₁_prior_var = σ²_x + (dt)²·σ²_v + σ²_motion
///
///   v₁_prior_mean = v_obs₀
///   v₁_prior_var = σ²_v + σ²_motion
///
/// This is saying: "based on the observation at t=0 and the motion model,
/// here's what we expect at t=1"
fn marginalize_oldest_state(
    _params: &Params,
    obs_old: &Observation,
    _obs_new: &Observation,
) -> MarginalizedPrior {
    // Position prior: propagate position observation through motion model
    // x₁ ≈ x₀ + v₀·dt
    let x_mean = obs_old.position + obs_old.velocity * DT;

    // Uncertainty propagates:
    //   - σ²_x from position measurement
    //   - (dt)²·σ²_v from velocity uncertainty affecting position prediction
    //   - σ²_motion from motion model uncertainty
    let x_var = POS_NOISE_STD.powi(2)
              + (DT * VEL_NOISE_STD).powi(2)
              + MOTION_NOISE_STD.powi(2);

    // Velocity prior: propagate velocity observation
    // v₁ ≈ v₀ (constant velocity model)
    let v_mean = obs_old.velocity;

    // Uncertainty propagates:
    //   - σ²_v from velocity measurement
    //   - σ²_motion from motion model uncertainty
    let v_var = VEL_NOISE_STD.powi(2)
              + MOTION_NOISE_STD.powi(2);

    MarginalizedPrior {
        x_mean,
        v_mean,
        x_var,
        v_var,
    }
}

/// Print estimated and true states side-by-side
fn print_window_error(estimated: &[State], true_states: &[State]) {
    for (i, (est, truth)) in estimated.iter().zip(true_states.iter()).enumerate() {
        println!("    [{}] est: x={:7.4}, v={:7.4}  |  true: x={:7.4}, v={:7.4}  |  err: Δx={:+.4}, Δv={:+.4}",
                 i,
                 est.position, est.velocity,
                 truth.position, truth.velocity,
                 est.position - truth.position,
                 est.velocity - truth.velocity);
    }
}

#[cfg(feature = "visualization")]
fn log_trajectory(
    rec: &rr::RecordingStream,
    iteration: usize,
    true_states: &[State],
    estimated_trajectory: &[State],
    marginalized_trajectory: &[State],
) {
    rec.set_time_sequence("step", iteration as i64);

    // Log full ground truth trajectory
    let true_positions: Vec<[f32; 2]> = true_states
        .iter()
        .enumerate()
        .map(|(i, s)| [i as f32 * DT as f32, s.position as f32])
        .collect();

    rec.log(
        "trajectory/ground_truth",
        &rr::LineStrips2D::new([true_positions])
            .with_colors([[150, 150, 150, 255]]),
    )
    .unwrap();

    // Log estimated trajectory (continuously updated)
    let est_positions: Vec<[f32; 2]> = estimated_trajectory
        .iter()
        .enumerate()
        .map(|(i, s)| [i as f32 * DT as f32, s.position as f32])
        .collect();

    rec.log(
        "trajectory/estimated",
        &rr::LineStrips2D::new([est_positions.clone()])
            .with_colors([[100, 100, 255, 255]]),
    )
    .unwrap();

    rec.log(
        "trajectory/estimated_points",
        &rr::Points2D::new(est_positions)
            .with_colors([[100, 100, 255, 255]])
            .with_radii([0.015]),
    )
    .unwrap();

    // Log marginalized prior trajectory (what we expect from propagating observations)
    if !marginalized_trajectory.is_empty() {
        let marg_positions: Vec<[f32; 2]> = marginalized_trajectory
            .iter()
            .enumerate()
            .map(|(i, s)| [(i + 1) as f32 * DT as f32, s.position as f32])  // +1 because first marg is for t=1
            .collect();

        rec.log(
            "trajectory/marginalized",
            &rr::LineStrips2D::new([marg_positions.clone()])
                .with_colors([[255, 150, 100, 255]]),
        )
        .unwrap();

        rec.log(
            "trajectory/marginalized_points",
            &rr::Points2D::new(marg_positions)
                .with_colors([[255, 150, 100, 255]])
                .with_radii([0.015]),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_brownian_motion_generates_trajectory() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let states = simulate_brownian_motion(&mut rng, 10);

        assert_eq!(states.len(), 10);
        assert_eq!(states[0].position, 0.0);
        assert_eq!(states[0].velocity, 0.0);

        // Position should drift due to velocity integration
        assert_ne!(states[9].position, 0.0);
    }

    #[test]
    fn test_observations_are_noisy() {
        let mut rng = ChaCha8Rng::seed_from_u64(456);
        let true_states = vec![
            State::new(1.0, 0.5),
            State::new(2.0, 0.3),
        ];

        let observations = generate_observations(&true_states, &mut rng);

        assert_eq!(observations.len(), 2);

        // Observations should be different from true values (noisy)
        for (obs, truth) in observations.iter().zip(true_states.iter()) {
            let pos_diff = (obs.position - truth.position).abs();
            let vel_diff = (obs.velocity - truth.velocity).abs();

            // Should be within reasonable noise bounds
            assert!(pos_diff < POS_NOISE_STD * 3.0); // 3-sigma
            assert!(vel_diff < VEL_NOISE_STD * 3.0);
        }
    }

    #[test]
    fn test_marginalization_formulas() {
        // Test the explicit marginalization formulas
        let obs_old = Observation {
            position: 0.0,
            velocity: 5.0,
        };
        let obs_new = Observation {
            position: 0.6,
            velocity: 4.8,
        };

        let params = Params::zeros(); // Not used in explicit formulas
        let prior = marginalize_oldest_state(&params, &obs_old, &obs_new);

        // Check position prior
        let expected_x_mean = 0.0 + 5.0 * DT; // x₀ + v₀·dt
        assert_relative_eq!(prior.x_mean, expected_x_mean, epsilon = 1e-10);

        let expected_x_var = POS_NOISE_STD.powi(2)
                           + (DT * VEL_NOISE_STD).powi(2)
                           + MOTION_NOISE_STD.powi(2);
        assert_relative_eq!(prior.x_var, expected_x_var, epsilon = 1e-10);

        // Check velocity prior
        let expected_v_mean = 5.0; // v₀
        assert_relative_eq!(prior.v_mean, expected_v_mean, epsilon = 1e-10);

        let expected_v_var = VEL_NOISE_STD.powi(2) + MOTION_NOISE_STD.powi(2);
        assert_relative_eq!(prior.v_var, expected_v_var, epsilon = 1e-10);
    }

    #[test]
    fn test_optimization_reduces_error() {
        let mut rng = ChaCha8Rng::seed_from_u64(161718);

        // Generate ground truth
        let true_states = simulate_brownian_motion(&mut rng, 2);
        let observations = generate_observations(&true_states, &mut rng);

        // Start with noisy initial guess
        let initial_params = Params::from_row_slice(&[
            observations[0].position + 0.5,
            observations[0].velocity + 0.5,
            observations[1].position + 0.5,
            observations[1].velocity + 0.5,
        ]);

        // Optimize
        let optimized_params = optimize_window(
            &initial_params,
            &[observations[0], observations[1]],
            None
        );

        let initial_states = params_to_states(&initial_params);
        let optimized_states = params_to_states(&optimized_params);

        // Compute errors
        let initial_error: f64 = initial_states
            .iter()
            .zip(true_states.iter())
            .map(|(est, truth)| {
                (est.position - truth.position).powi(2) + (est.velocity - truth.velocity).powi(2)
            })
            .sum();

        let optimized_error: f64 = optimized_states
            .iter()
            .zip(true_states.iter())
            .map(|(est, truth)| {
                (est.position - truth.position).powi(2) + (est.velocity - truth.velocity).powi(2)
            })
            .sum();

        // Optimization should reduce error
        assert!(
            optimized_error < initial_error,
            "Optimization should reduce error: {} -> {}",
            initial_error,
            optimized_error
        );
    }

    #[test]
    fn test_residual_computation_without_prior() {
        let params = Params::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
        let observations = [
            Observation { position: 1.0, velocity: 2.0 },
            Observation { position: 3.0, velocity: 4.0 },
        ];

        let mut residuals = Residuals::zeros();
        let mut jacobian = Jacobian::zeros();

        compute_residuals_and_jacobian(&params, &observations, None, &mut residuals, &mut jacobian);

        // With perfect match to observations, residuals should be small
        // (only motion model residuals might be non-zero)
        assert!(residuals[0].abs() < 1e-10); // position obs at t=0
        assert!(residuals[1].abs() < 1e-10); // velocity obs at t=0
        assert!(residuals[4].abs() < 1e-10); // position obs at t=1
        assert!(residuals[5].abs() < 1e-10); // velocity obs at t=1
    }

    #[test]
    fn test_residual_computation_with_prior() {
        let params = Params::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
        let observations = [
            Observation { position: 1.0, velocity: 2.0 },
            Observation { position: 3.0, velocity: 4.0 },
        ];

        let prior = MarginalizedPrior {
            x_mean: 1.0,
            v_mean: 2.0,
            x_var: 0.1,
            v_var: 0.2,
        };

        let mut residuals = Residuals::zeros();
        let mut jacobian = Jacobian::zeros();

        compute_residuals_and_jacobian(&params, &observations, Some(prior), &mut residuals, &mut jacobian);

        // With perfect match to prior, first 2 residuals should be small
        assert!(residuals[0].abs() < 1e-10); // prior on x₀
        assert!(residuals[1].abs() < 1e-10); // prior on v₀
    }

    #[test]
    fn test_sliding_window_with_marginalization() {
        let mut rng = ChaCha8Rng::seed_from_u64(999);

        // Generate a few steps
        let true_states = simulate_brownian_motion(&mut rng, 5);
        let observations = generate_observations(&true_states, &mut rng);

        // Initial window: t=0, t=1
        let mut params = Params::from_row_slice(&[
            observations[0].position,
            observations[0].velocity,
            observations[1].position,
            observations[1].velocity,
        ]);

        params = optimize_window(&params, &[observations[0], observations[1]], None);

        // Slide to t=1, t=2 with marginalization
        let prior = marginalize_oldest_state(&params, &observations[0], &observations[1]);

        params = Params::from_row_slice(&[
            observations[1].position,
            observations[1].velocity,
            observations[2].position,
            observations[2].velocity,
        ]);

        params = optimize_window(&params, &[observations[1], observations[2]], Some(prior));

        // Should complete without panicking
        let states = params_to_states(&params);
        assert_eq!(states.len(), 2);
    }
}
