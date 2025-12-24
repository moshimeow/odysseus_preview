//! IMU-only optimization
//!
//! Provides a simplified optimizer for testing IMU factors in isolation.
//! This is useful for validating the IMU residuals before integrating with visual SLAM.

use nalgebra::DVector;
#[cfg(test)]
use nalgebra::Vector3;
use odysseus_solver::math3d::Quat;
use odysseus_solver::sparse_solver::SparseLevenbergMarquardt;
use odysseus_solver::Jet;

use super::preintegration::PreintegratedImu;
use super::residuals::{bias_residual, imu_preintegration_residual};
use super::simulator::ImuNoiseParams;

/// Number of parameters per frame in VIO optimization
/// [rot_delta(3), translation(3), velocity(3), gyro_bias(3), accel_bias(3)]
pub const PARAMS_PER_FRAME: usize = 15;

/// Number of residuals per IMU factor (rotation + velocity + position)
pub const IMU_RESIDUALS: usize = 9;

/// Number of residuals for bias random walk (gyro + accel)
pub const BIAS_RESIDUALS: usize = 6;

/// Result of IMU-only optimization
#[derive(Debug, Clone)]
pub struct ImuOptimizationResult {
    /// Final error (sum of squared residuals)
    pub final_error: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Optimized frame states [frame0, frame1, ...]
    /// Each frame: [rot_delta(3), trans(3), vel(3), bg(3), ba(3)]
    pub states: Vec<[f64; PARAMS_PER_FRAME]>,
}

/// Run IMU-only optimization on a sequence of frames
///
/// # Arguments
/// * `host_quats` - Host quaternions for each frame (for rotation parameterization)
/// * `initial_states` - Initial guess for each frame's state
/// * `preintegrations` - Preintegrated IMU measurements between consecutive frames
/// * `gravity` - Gravity vector in world frame
/// * `noise_params` - IMU noise parameters (for bias residual weighting)
/// * `fix_first_pose` - Whether to fix the first frame's pose (recommended)
///
/// # Returns
/// Optimization result with final states
pub fn run_imu_optimization(
    host_quats: &[Quat<f64>],
    initial_states: &[[f64; PARAMS_PER_FRAME]],
    preintegrations: &[PreintegratedImu],
    gravity: [f64; 3],
    noise_params: &ImuNoiseParams,
    fix_first_pose: bool,
) -> ImuOptimizationResult {
    let n_frames = initial_states.len();
    assert_eq!(host_quats.len(), n_frames);
    assert_eq!(preintegrations.len(), n_frames - 1);
    assert!(n_frames >= 2, "Need at least 2 frames for IMU optimization");

    // Calculate dimensions
    let first_optimized = if fix_first_pose { 1 } else { 0 };
    let n_optimized_frames = n_frames - first_optimized;
    let n_params = n_optimized_frames * PARAMS_PER_FRAME;

    // Each consecutive pair has 1 IMU factor (9 residuals) + 1 bias factor (6 residuals)
    let n_imu_factors = n_frames - 1;
    let n_residuals = n_imu_factors * (IMU_RESIDUALS + BIAS_RESIDUALS);

    // Build sparsity pattern
    // IMU factors connect consecutive frames
    let mut entries = Vec::new();

    for factor_idx in 0..n_imu_factors {
        let frame_i = factor_idx;
        let frame_j = factor_idx + 1;

        // Residual rows for this factor
        let imu_res_start = factor_idx * (IMU_RESIDUALS + BIAS_RESIDUALS);
        let bias_res_start = imu_res_start + IMU_RESIDUALS;

        // IMU residual (9) depends on both frames (15 params each)
        for res in 0..IMU_RESIDUALS {
            let row = imu_res_start + res;

            // Frame i params (if optimized)
            if frame_i >= first_optimized {
                let param_start = (frame_i - first_optimized) * PARAMS_PER_FRAME;
                for p in 0..PARAMS_PER_FRAME {
                    entries.push((row, param_start + p));
                }
            }

            // Frame j params (if optimized)
            if frame_j >= first_optimized {
                let param_start = (frame_j - first_optimized) * PARAMS_PER_FRAME;
                for p in 0..PARAMS_PER_FRAME {
                    entries.push((row, param_start + p));
                }
            }
        }

        // Bias residual (6) depends on biases of both frames (6 params each)
        for res in 0..BIAS_RESIDUALS {
            let row = bias_res_start + res;

            // Frame i bias params (if optimized)
            if frame_i >= first_optimized {
                let param_start = (frame_i - first_optimized) * PARAMS_PER_FRAME;
                // Biases are at indices 9-14
                for p in 9..PARAMS_PER_FRAME {
                    entries.push((row, param_start + p));
                }
            }

            // Frame j bias params (if optimized)
            if frame_j >= first_optimized {
                let param_start = (frame_j - first_optimized) * PARAMS_PER_FRAME;
                for p in 9..PARAMS_PER_FRAME {
                    entries.push((row, param_start + p));
                }
            }
        }
    }

    // Sort entries by (row, col) for CSR format
    entries.sort();
    entries.dedup();

    // Initialize parameters
    let mut params = DVector::zeros(n_params);
    for (i, state) in initial_states.iter().enumerate().skip(first_optimized) {
        let param_start = (i - first_optimized) * PARAMS_PER_FRAME;
        for (j, &val) in state.iter().enumerate() {
            params[param_start + j] = val;
        }
    }

    // Create solver
    let mut solver = SparseLevenbergMarquardt::new(n_residuals, n_params, &entries)
        .with_tolerance(1e-8)
        .with_max_iterations(50)
        .with_verbose(false);

    // Store fixed frame state if needed
    let fixed_state = if fix_first_pose {
        Some(initial_states[0])
    } else {
        None
    };

    // Clone data for closure
    let host_quats = host_quats.to_vec();
    let preintegrations = preintegrations.to_vec();
    let noise_params = noise_params.clone();

    let mut iteration_count = 0;
    let mut final_error = 0.0;
    let mut did_converge = false;

    // Run optimization
    let result = solver.solve(
        params,
        |params, residuals, jacobian_data| {
            compute_imu_cost(
                params,
                residuals,
                jacobian_data,
                &host_quats,
                fixed_state.as_ref(),
                &preintegrations,
                &gravity,
                &noise_params,
                first_optimized,
            );
        },
        |iter, result, _params| {
            iteration_count = iter + 1;
            final_error = result.error;
            did_converge = result.converged;
        },
    );

    // Extract final states
    let mut states = Vec::with_capacity(n_frames);

    // Add fixed state if applicable
    if fix_first_pose {
        states.push(initial_states[0]);
    }

    // Add optimized states
    for i in 0..n_optimized_frames {
        let param_start = i * PARAMS_PER_FRAME;
        let mut state = [0.0; PARAMS_PER_FRAME];
        for j in 0..PARAMS_PER_FRAME {
            state[j] = result[param_start + j];
        }
        states.push(state);
    }

    ImuOptimizationResult {
        final_error,
        iterations: iteration_count,
        converged: did_converge,
        states,
    }
}

/// Compute IMU cost function with Jacobians
fn compute_imu_cost(
    params: &DVector<f64>,
    residuals: &mut [f64],
    jacobian_data: &mut [f64],
    host_quats: &[Quat<f64>],
    fixed_state: Option<&[f64; PARAMS_PER_FRAME]>,
    preintegrations: &[PreintegratedImu],
    gravity: &[f64; 3],
    noise_params: &ImuNoiseParams,
    first_optimized: usize,
) {
    let n_frames = host_quats.len();
    let n_imu_factors = n_frames - 1;

    let mut jac_idx = 0;

    for factor_idx in 0..n_imu_factors {
        let frame_i = factor_idx;
        let frame_j = factor_idx + 1;

        // Get states for frames i and j
        let state_i = get_frame_state(params, frame_i, first_optimized, fixed_state);
        let state_j = get_frame_state(params, frame_j, first_optimized, fixed_state);

        let preint = &preintegrations[factor_idx];
        let dt = preint.delta_time;

        // Residual indices
        let imu_res_start = factor_idx * (IMU_RESIDUALS + BIAS_RESIDUALS);
        let bias_res_start = imu_res_start + IMU_RESIDUALS;

        // Compute IMU residual with autodiff
        compute_imu_residual_with_jacobian(
            &host_quats[frame_i],
            &state_i,
            frame_i >= first_optimized,
            &host_quats[frame_j],
            &state_j,
            frame_j >= first_optimized,
            preint,
            gravity,
            &mut residuals[imu_res_start..imu_res_start + IMU_RESIDUALS],
            jacobian_data,
            &mut jac_idx,
        );

        // Compute bias residual
        compute_bias_residual_with_jacobian(
            &state_i,
            frame_i >= first_optimized,
            &state_j,
            frame_j >= first_optimized,
            dt,
            noise_params,
            &mut residuals[bias_res_start..bias_res_start + BIAS_RESIDUALS],
            jacobian_data,
            &mut jac_idx,
        );
    }
}

/// Get frame state from params or fixed state
fn get_frame_state(
    params: &DVector<f64>,
    frame_idx: usize,
    first_optimized: usize,
    fixed_state: Option<&[f64; PARAMS_PER_FRAME]>,
) -> [f64; PARAMS_PER_FRAME] {
    if frame_idx < first_optimized {
        *fixed_state.expect("Fixed state required for frame 0")
    } else {
        let param_start = (frame_idx - first_optimized) * PARAMS_PER_FRAME;
        let mut state = [0.0; PARAMS_PER_FRAME];
        for i in 0..PARAMS_PER_FRAME {
            state[i] = params[param_start + i];
        }
        state
    }
}

/// Compute IMU residual with Jacobian using autodiff
fn compute_imu_residual_with_jacobian(
    host_i: &Quat<f64>,
    state_i: &[f64; PARAMS_PER_FRAME],
    optimize_i: bool,
    host_j: &Quat<f64>,
    state_j: &[f64; PARAMS_PER_FRAME],
    optimize_j: bool,
    preint: &PreintegratedImu,
    gravity: &[f64; 3],
    residuals: &mut [f64],
    jacobian_data: &mut [f64],
    jac_idx: &mut usize,
) {
    // Use Jets for autodiff
    // Total derivatives: up to 30 (15 for frame i + 15 for frame j)
    type Jet30 = Jet<f64, 30>;

    let _n_active_params = (if optimize_i { PARAMS_PER_FRAME } else { 0 })
        + (if optimize_j { PARAMS_PER_FRAME } else { 0 });

    // Build Jet parameters
    let mut deriv_idx = 0;

    let params_i: [Jet30; PARAMS_PER_FRAME] = std::array::from_fn(|k| {
        if optimize_i {
            let j = Jet30::variable(state_i[k], deriv_idx);
            deriv_idx += 1;
            j
        } else {
            Jet30::constant(state_i[k])
        }
    });

    let params_j: [Jet30; PARAMS_PER_FRAME] = std::array::from_fn(|k| {
        if optimize_j {
            let j = Jet30::variable(state_j[k], deriv_idx);
            deriv_idx += 1;
            j
        } else {
            Jet30::constant(state_j[k])
        }
    });

    let gravity_jet = [
        Jet30::constant(gravity[0]),
        Jet30::constant(gravity[1]),
        Jet30::constant(gravity[2]),
    ];

    // Compute residual
    let res =
        imu_preintegration_residual(host_i, &params_i, host_j, &params_j, preint, &gravity_jet);

    // Extract residuals and Jacobians
    for r in 0..IMU_RESIDUALS {
        residuals[r] = res[r].value;

        // Write Jacobian entries (frame i params, then frame j params)
        if optimize_i {
            for p in 0..PARAMS_PER_FRAME {
                jacobian_data[*jac_idx] = res[r].derivs[p];
                *jac_idx += 1;
            }
        }
        if optimize_j {
            let offset = if optimize_i { PARAMS_PER_FRAME } else { 0 };
            for p in 0..PARAMS_PER_FRAME {
                jacobian_data[*jac_idx] = res[r].derivs[offset + p];
                *jac_idx += 1;
            }
        }
    }
}

/// Compute bias residual with Jacobian
fn compute_bias_residual_with_jacobian(
    state_i: &[f64; PARAMS_PER_FRAME],
    optimize_i: bool,
    state_j: &[f64; PARAMS_PER_FRAME],
    optimize_j: bool,
    dt: f64,
    noise_params: &ImuNoiseParams,
    residuals: &mut [f64],
    jacobian_data: &mut [f64],
    jac_idx: &mut usize,
) {
    // Extract biases
    let bg_i = [state_i[9], state_i[10], state_i[11]];
    let ba_i = [state_i[12], state_i[13], state_i[14]];
    let bg_j = [state_j[9], state_j[10], state_j[11]];
    let ba_j = [state_j[12], state_j[13], state_j[14]];

    // Use safe minimum for noise params to avoid division by zero
    let gyro_bias_rw = noise_params.gyro_bias_random_walk.max(1e-6);
    let accel_bias_rw = noise_params.accel_bias_random_walk.max(1e-6);

    // Compute residual
    let res = bias_residual(&bg_i, &ba_i, &bg_j, &ba_j, dt, gyro_bias_rw, accel_bias_rw);

    // Weights for Jacobian
    let dt_sqrt = dt.sqrt().max(1e-6);
    let gyro_weight = 1.0 / (gyro_bias_rw * dt_sqrt);
    let accel_weight = 1.0 / (accel_bias_rw * dt_sqrt);

    // Extract residuals and compute Jacobians
    for r in 0..BIAS_RESIDUALS {
        residuals[r] = res[r];

        // Jacobian is simple: d(bj - bi)/d(bi) = -I, d(bj - bi)/d(bj) = I
        // Weighted by the appropriate weight

        let weight = if r < 3 { gyro_weight } else { accel_weight };

        // Frame i bias derivatives (indices 9-14 in full state)
        if optimize_i {
            for p in 9..PARAMS_PER_FRAME {
                let bias_idx = p - 9; // 0-5
                if bias_idx == r {
                    jacobian_data[*jac_idx] = -weight;
                } else {
                    jacobian_data[*jac_idx] = 0.0;
                }
                *jac_idx += 1;
            }
        }

        // Frame j bias derivatives
        if optimize_j {
            for p in 9..PARAMS_PER_FRAME {
                let bias_idx = p - 9;
                if bias_idx == r {
                    jacobian_data[*jac_idx] = weight;
                } else {
                    jacobian_data[*jac_idx] = 0.0;
                }
                *jac_idx += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::simulator::{add_timestamps_to_poses, ImuNoiseParams};
    use crate::imu::{ImuSimulator, PreintegratedImu};
    use crate::trajectory::{ContinuousCircularTrajectory, ContinuousTrajectory};

    #[test]
    fn test_imu_only_optimization_identity() {
        // Two frames at the same position - should converge to zero residual
        let host_quat = Quat::identity();
        let host_quats = vec![host_quat, host_quat];

        // Both frames at origin with zero velocity/bias
        let initial_states = vec![[0.0; PARAMS_PER_FRAME], [0.0; PARAMS_PER_FRAME]];

        // Preintegration with small dt but zero motion
        let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
        preint.delta_time = 0.1; // 100ms between frames

        let gravity = [0.0, 0.0, -9.81];
        let noise_params = ImuNoiseParams::consumer_grade();

        let result = run_imu_optimization(
            &host_quats,
            &initial_states,
            &[preint],
            gravity,
            &noise_params,
            true, // Fix first pose
        );

        println!(
            "Identity test - Error: {}, Iterations: {}",
            result.final_error, result.iterations
        );

        // Should have very small error
        assert!(
            result.final_error < 1.0,
            "Error too large: {}",
            result.final_error
        );
    }

    #[test]
    fn test_imu_only_optimization_translation() {
        // Two frames with translation - test that optimizer finds correct motion
        let host_quat = Quat::identity();
        let host_quats = vec![host_quat, host_quat];

        let gravity = [0.0, 0.0, -9.81];
        let noise_params = ImuNoiseParams::zero(); // No noise for clean test

        // Frame 0 at origin, frame 1 translated by [1, 0, 0] with velocity
        // For constant velocity motion: p1 = p0 + v0*dt
        // v0 = [1, 0, 0], dt = 1.0 => p1 = [1, 0, 0]
        let dt = 1.0;
        let velocity = 1.0;

        let mut state0 = [0.0; PARAMS_PER_FRAME];
        state0[6] = velocity; // velocity x

        let mut state1 = [0.0; PARAMS_PER_FRAME];
        state1[3] = velocity * dt; // translation x = v * t
        state1[6] = velocity; // same velocity

        let initial_states = vec![state0, state1];

        // Create preintegration for constant velocity motion
        // With no acceleration (after gravity cancellation), delta_v = 0, delta_p = 0
        // The residual equation handles the v*dt term
        let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
        preint.delta_time = dt;
        // For no acceleration: delta_velocity = 0 (in body frame, relative)
        // delta_position = 0 (the v*dt + 0.5*g*t^2 is in the residual formula)

        let result = run_imu_optimization(
            &host_quats,
            &initial_states,
            &[preint],
            gravity,
            &noise_params,
            true,
        );

        println!(
            "Translation test - Error: {}, Iterations: {}",
            result.final_error, result.iterations
        );

        // Should converge
        assert!(
            result.final_error < 1.0,
            "Error too large: {}",
            result.final_error
        );
    }

    #[test]
    fn test_imu_optimization_convergence() {
        // Test with simulated IMU data
        // Use 10 poses to ensure enough measurements between each pair
        let duration = 4.0;
        let trajectory = ContinuousCircularTrajectory::new(1.0, duration);
        let poses = trajectory.sample_poses(10);
        let timestamped = add_timestamps_to_poses(poses.clone(), duration);

        // Generate IMU measurements
        let noise_params = ImuNoiseParams::zero();
        let simulator = ImuSimulator::new(noise_params.clone(), 200.0);
        let measurements =
            simulator.generate_from_continuous_trajectory(&trajectory, duration, 123);

        // Build host quaternions and initial states from ground truth
        let mut host_quats = Vec::new();
        let mut gt_states = Vec::new();

        for (i, (t, pose)) in timestamped.iter().enumerate() {
            // Host quaternion from pose rotation
            let rot_log = pose.rotation.log();
            let host_quat = Quat::from_axis_angle(odysseus_solver::math3d::Vec3::new(
                rot_log.x, rot_log.y, rot_log.z,
            ));
            host_quats.push(host_quat);

            // Ground truth state: zero rotation delta (matches host), actual translation
            let mut state = [0.0; PARAMS_PER_FRAME];
            // rotation delta = 0 (encoded in host)
            state[3] = pose.translation.x;
            state[4] = pose.translation.y;
            state[5] = pose.translation.z;
            // velocity - approximate from position difference
            if i > 0 {
                let dt = t - timestamped[i - 1].0;
                if dt > 0.0 {
                    let prev_pose = &timestamped[i - 1].1;
                    state[6] = (pose.translation.x - prev_pose.translation.x) / dt;
                    state[7] = (pose.translation.y - prev_pose.translation.y) / dt;
                    state[8] = (pose.translation.z - prev_pose.translation.z) / dt;
                }
            }
            // biases = 0
            gt_states.push(state);
        }

        // Preintegrate IMU between consecutive frames
        // Skip first and last segments which may have no measurements due to 3-point stencil
        let mut preintegrations = Vec::new();
        let mut valid_frame_indices = vec![0usize]; // Start with frame 0

        for i in 0..timestamped.len() - 1 {
            let t_start = timestamped[i].0;
            let t_end = timestamped[i + 1].0;

            let relevant: Vec<_> = measurements
                .iter()
                .filter(|m| m.timestamp >= t_start && m.timestamp <= t_end)
                .cloned()
                .collect();

            // Skip segments with no measurements
            if relevant.is_empty() {
                continue;
            }

            let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
            preint.integrate_measurements(&relevant, 0.0, 0.0);
            preintegrations.push(preint);
            valid_frame_indices.push(i + 1);
        }

        // Filter states and host_quats to match valid frames
        let host_quats: Vec<_> = valid_frame_indices.iter().map(|&i| host_quats[i]).collect();
        let gt_states: Vec<_> = valid_frame_indices.iter().map(|&i| gt_states[i]).collect();

        println!(
            "Generated {} measurements, {} valid preintegrations, {} frames",
            measurements.len(),
            preintegrations.len(),
            valid_frame_indices.len()
        );
        assert!(
            valid_frame_indices.len() >= 3,
            "Need at least 3 valid frames"
        );
        assert!(
            !preintegrations.is_empty(),
            "Need at least one preintegration"
        );

        // Perturb initial guess
        let mut perturbed_states = gt_states.clone();
        for state in perturbed_states.iter_mut().skip(1) {
            state[3] += 0.1; // Add 10cm error to translation
            state[4] += 0.05;
        }

        let gravity = [0.0, 0.0, -9.81];

        let result = run_imu_optimization(
            &host_quats,
            &perturbed_states,
            &preintegrations,
            gravity,
            &noise_params,
            true,
        );

        println!(
            "Convergence test - Final error: {}, Iterations: {}, Converged: {}",
            result.final_error, result.iterations, result.converged
        );

        // Should reduce error significantly
        assert!(result.iterations > 0, "Should run at least one iteration");
    }
}
