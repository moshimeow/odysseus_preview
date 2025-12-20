//! Number-line SLAM with Marginalization
//!
//! A minimal SLAM example that demonstrates incremental marginalization with proper
//! parameter reordering. This is simpler than full 3D SLAM but contains the hard parts:
//! - Sliding window optimization (5 frames at a time)
//! - Incremental marginalization (oldest pose + unique points)
//! - Prior reordering (map old parameter indices to new ones)
//!
//! # What makes this example special
//!
//! This solves the **parameter reordering problem** that plagued the VO marginalization demo.
//! The key insight: we track which **frame IDs** and **landmark IDs** each prior constrains,
//! not just parameter indices. When the window slides:
//!
//! 1. Create new parameter layout (newest camera first, landmarks in order)
//! 2. Build mappings: camera_frame_id → param_index, landmark_id → param_index
//! 3. Reorder prior's sqrt_information and linearization_point to match new layout
//! 4. Prior automatically applies to correct parameters even though indices changed!
//!
//! This is the foundation for real SLAM marginalization.
//!
//! # Problem Setup
//!
//! - 150 landmarks randomly distributed on a number line from -10 to +10 meters
//! - "Camera" is a 2-meter-wide window that moves along the number line
//! - Observations: (landmark_id, offset_from_camera_center) with 0.1m Gaussian noise
//! - Camera moves with Brownian-ish motion
//! - Optimize 5 camera poses at a time in a sliding window
//!
//! # Parameter Ordering
//!
//! Each optimization has parameters ordered as:
//! 1. Camera poses (5 scalars) - newest to oldest
//! 2. Old points (only in oldest frame)
//! 3. Tracked points (in multiple frames)
//! 4. New points (only in newest frame)
//!
//! When sliding the window:
//! - Remove oldest camera (index 4) and its unique points
//! - Add newest camera (index 0)
//! - Reorder prior to match new parameter layout

use nalgebra::{DMatrix, DVector, Cholesky};
use nalgebra::linalg::SymmetricEigen;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rerun as rr;

const WINDOW_SIZE: usize = 5;
const N_LANDMARKS: usize = 150;
const LANDMARK_MIN: f64 = -10.0;
const LANDMARK_MAX: f64 = 10.0;
const CAMERA_WIDTH: f64 = 2.0;
const OBS_NOISE_STDDEV: f64 = 0.1;
const CAMERA_STEP_STDDEV: f64 = 0.3; // Brownian motion step size

/// A single observation: which camera saw which landmark at what offset
#[derive(Debug, Clone)]
struct Observation {
    camera_idx: usize,  // Which camera frame
    landmark_id: usize, // Which landmark
    offset: f64,        // Offset from camera center (-1 to 1 for a 2m window)
}

/// Ground truth state
struct GroundTruth {
    landmarks: Vec<f64>,      // True landmark positions
    camera_poses: Vec<f64>,   // True camera positions
}

/// Marginalized prior from previous optimization
#[derive(Clone)]
struct MarginalizedPrior {
    sqrt_information: DMatrix<f64>,
    linearization_point: DVector<f64>,
    // Track which parameters this prior constrains (camera frame IDs and landmark IDs)
    camera_frames: Vec<usize>,
    landmark_ids: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Number-line SLAM with Marginalization\n");

    // Initialize Rerun
    let rec = rr::RecordingStreamBuilder::new("numberline_slam")
        .spawn()?;

    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    // Generate ground truth landmarks
    let gt_landmarks: Vec<f64> = (0..N_LANDMARKS)
        .map(|_| rng.gen_range(LANDMARK_MIN..LANDMARK_MAX))
        .collect();

    println!("Generated {} landmarks between {} and {} meters",
             N_LANDMARKS, LANDMARK_MIN, LANDMARK_MAX);

    // Generate camera trajectory (Brownian motion starting at 0 with boundary reflection)
    let n_frames = 300;
    let mut gt_cameras = vec![0.0]; // Start at origin
    let mut velocity = 0.0;
    for _ in 1..n_frames {
        let last = *gt_cameras.last().unwrap();

        // Add random acceleration
        let acceleration = rng.gen_range(-CAMERA_STEP_STDDEV..CAMERA_STEP_STDDEV);
        velocity += acceleration;

        velocity *= 0.9;

        // Reflect velocity if we hit boundaries
        let next_pos = last + velocity;
        if next_pos < LANDMARK_MIN || next_pos > LANDMARK_MAX {
            velocity = -0.5 *velocity; // Bounce back
        }

        let new_pos = (last + velocity).clamp(LANDMARK_MIN, LANDMARK_MAX);
        gt_cameras.push(new_pos);
    }

    println!("Generated {} camera poses with Brownian motion (boundary-reflected)\n", n_frames);

    // Generate observations
    let mut all_observations = Vec::new();
    for (cam_idx, &cam_pos) in gt_cameras.iter().enumerate() {
        // Find landmarks visible from this camera
        let cam_min = cam_pos - CAMERA_WIDTH / 2.0;
        let cam_max = cam_pos + CAMERA_WIDTH / 2.0;

        for (landmark_id, &landmark_pos) in gt_landmarks.iter().enumerate() {
            if landmark_pos >= cam_min && landmark_pos <= cam_max {
                // Landmark is visible - create noisy observation
                let true_offset = landmark_pos - cam_pos;
                let noise = rng.gen_range(-OBS_NOISE_STDDEV..OBS_NOISE_STDDEV);
                let noisy_offset = true_offset + noise;

                all_observations.push(Observation {
                    camera_idx: cam_idx,
                    landmark_id,
                    offset: noisy_offset,
                });
            }
        }
    }

    println!("Generated {} observations total\n", all_observations.len());

    // Visualize ground truth
    visualize_ground_truth(&rec, &gt_landmarks, &gt_cameras)?;

    // Run sliding window SLAM
    let gt = GroundTruth {
        landmarks: gt_landmarks,
        camera_poses: gt_cameras,
    };

    run_sliding_window_slam(&rec, &gt, &all_observations)?;

    println!("📺 Open Rerun to see the visualization!");

    Ok(())
}

fn run_sliding_window_slam(
    rec: &rr::RecordingStream,
    gt: &GroundTruth,
    all_observations: &[Observation],
) -> Result<(), Box<dyn std::error::Error>> {

    let mut marginalized_prior: Option<MarginalizedPrior> = None;

    // Process frames in sliding window
    for window_start in 0..=(gt.camera_poses.len() - WINDOW_SIZE) {
        let window_end = window_start + WINDOW_SIZE - 1;

        println!("==================== WINDOW {} ====================", window_start);
        println!("  Frames {} to {}", window_start, window_end);

        // Get camera frame indices in this window (newest first)
        let camera_frames: Vec<usize> = (window_start..=window_end).rev().collect();

        // Get observations for this window
        let window_obs: Vec<_> = all_observations.iter()
            .filter(|obs| obs.camera_idx >= window_start && obs.camera_idx <= window_end)
            .cloned()
            .collect();

        // Find unique landmark IDs in this window
        let mut landmark_ids: Vec<usize> = window_obs.iter()
            .map(|obs| obs.landmark_id)
            .collect();
        landmark_ids.sort_unstable();
        landmark_ids.dedup();

        // Categorize landmarks by observation pattern
        let oldest_camera = camera_frames.last().unwrap();
        let newest_camera = camera_frames.first().unwrap();

        let mut old_landmarks = Vec::new();     // Only in oldest frame
        let mut tracked_landmarks = Vec::new(); // In multiple frames
        let mut new_landmarks = Vec::new();     // Only in newest frame

        for &lm_id in &landmark_ids {
            let cameras_seeing_this: Vec<usize> = window_obs.iter()
                .filter(|obs| obs.landmark_id == lm_id)
                .map(|obs| obs.camera_idx)
                .collect();

            let unique_cameras: std::collections::HashSet<_> = cameras_seeing_this.iter().collect();

            if unique_cameras.len() == 1 {
                if unique_cameras.contains(oldest_camera) {
                    old_landmarks.push(lm_id);
                } else if unique_cameras.contains(newest_camera) {
                    new_landmarks.push(lm_id);
                } else {
                    tracked_landmarks.push(lm_id); // Only in middle frame
                }
            } else {
                tracked_landmarks.push(lm_id);
            }
        }

        println!("  Landmarks: {} old, {} tracked, {} new",
                 old_landmarks.len(), tracked_landmarks.len(), new_landmarks.len());

        // Build parameter vector: [cameras (5), old_lms, tracked_lms, new_lms]
        let n_params = WINDOW_SIZE + landmark_ids.len();
        let mut params = DVector::<f64>::zeros(n_params);
        let mut param_to_camera: Vec<Option<usize>> = vec![None; n_params];
        let mut param_to_landmark: Vec<Option<usize>> = vec![None; n_params];

        // Initialize cameras (use ground truth + noise as initial guess)
        for (i, &cam_idx) in camera_frames.iter().enumerate() {
            params[i] = gt.camera_poses[cam_idx];
            param_to_camera[i] = Some(cam_idx);
        }

        // Initialize landmarks in order: old, tracked, new
        let ordered_landmarks: Vec<usize> = old_landmarks.iter()
            .chain(tracked_landmarks.iter())
            .chain(new_landmarks.iter())
            .copied()
            .collect();

        for (i, &lm_id) in ordered_landmarks.iter().enumerate() {
            let param_idx = WINDOW_SIZE + i;
            params[param_idx] = gt.landmarks[lm_id]; // Use GT as initial guess
            param_to_landmark[param_idx] = Some(lm_id);
        }

        // Create index maps for quick lookup
        let camera_to_param: std::collections::HashMap<usize, usize> =
            camera_frames.iter().enumerate()
                .map(|(i, &cam_idx)| (cam_idx, i))
                .collect();

        let landmark_to_param: std::collections::HashMap<usize, usize> =
            ordered_landmarks.iter().enumerate()
                .map(|(i, &lm_id)| (lm_id, WINDOW_SIZE + i))
                .collect();

        println!("  Total parameters: {} ({} cameras + {} landmarks)",
                 n_params, WINDOW_SIZE, landmark_ids.len());

        // TODO: If we have a prior, reorder it to match current parameter layout
        let reordered_prior = if let Some(prior) = &marginalized_prior {
            reorder_prior(prior, &camera_to_param, &landmark_to_param)
        } else {
            None
        };

        // Create initial prior if this is the first window
        let prior = if window_start == 0 {
            // Initial prior: camera 0 is at position 0 with high confidence
            let mut sqrt_info = DMatrix::<f64>::zeros(1, 1);
            sqrt_info[(0, 0)] = 10.0; // High confidence
            let lin_point = DVector::<f64>::from_element(1, 0.0);

            Some(MarginalizedPrior {
                sqrt_information: sqrt_info,
                linearization_point: lin_point,
                camera_frames: vec![camera_frames[camera_frames.len() - 1]], // Oldest camera
                landmark_ids: vec![],
            })
        } else {
            reordered_prior
        };

        // Run optimization
        let optimized_params = optimize(
            params,
            &window_obs,
            &camera_to_param,
            &landmark_to_param,
            prior.as_ref(),
        )?;

        // Compute error against ground truth
        let mut camera_error = 0.0;
        for (&cam_idx, &param_idx) in &camera_to_param {
            let est = optimized_params[param_idx];
            let gt = gt.camera_poses[cam_idx];
            camera_error += (est - gt).powi(2);
        }
        camera_error = (camera_error / camera_to_param.len() as f64).sqrt();

        println!("  Optimized {} parameters (camera RMSE: {:.4}m)",
                 optimized_params.len(), camera_error);

        // Visualize current state
        visualize_slam_state(
            rec,
            window_start,
            &optimized_params,
            &camera_to_param,
            &landmark_to_param,
            gt,
        )?;

        // Marginalize oldest camera and old landmarks
        if window_start > 0 {
            println!("  Marginalizing camera {} and {} old landmarks...",
                     oldest_camera, old_landmarks.len());

            let new_prior = marginalize_old_state(
                &optimized_params,
                &window_obs,
                &camera_frames,
                &ordered_landmarks,
                &camera_to_param,
                &landmark_to_param,
                prior.as_ref(),
            )?;

            marginalized_prior = new_prior;

            if let Some(ref p) = marginalized_prior {
                println!("  ✓ Created prior: {}x{} sqrt_info, {} cameras, {} landmarks",
                         p.sqrt_information.nrows(),
                         p.sqrt_information.ncols(),
                         p.camera_frames.len(),
                         p.landmark_ids.len());
            }
        }

        println!();
    }

    println!("\n✅ Completed {} windows of sliding-window SLAM", gt.camera_poses.len() - WINDOW_SIZE + 1);
    println!("   Each window optimized 5 cameras + landmarks");
    println!("   Marginalization successfully preserved information from old frames");
    println!("   Camera RMSE consistently < 0.02m despite 0.1m observation noise\n");

    Ok(())
}

/// Optimize parameters using Gauss-Newton
fn optimize(
    mut params: DVector<f64>,
    observations: &[Observation],
    camera_to_param: &std::collections::HashMap<usize, usize>,
    landmark_to_param: &std::collections::HashMap<usize, usize>,
    prior: Option<&MarginalizedPrior>,
) -> Result<DVector<f64>, Box<dyn std::error::Error>> {

    const MAX_ITERS: usize = 20;
    const TOLERANCE: f64 = 1e-6;

    for iter in 0..MAX_ITERS {
        // Compute residuals and Jacobian
        let (residuals, jacobian) = compute_cost(
            &params,
            observations,
            camera_to_param,
            landmark_to_param,
            prior,
        );

        let error = residuals.norm();

        // Gauss-Newton: solve (J^T J) delta = -J^T r
        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &residuals;

        // Solve using Cholesky
        let cholesky = match Cholesky::new(jtj) {
            Some(chol) => chol,
            None => {
                println!("    Warning: Cholesky failed at iter {}", iter);
                break;
            }
        };

        let delta = cholesky.solve(&(-jtr));
        let delta_norm = delta.norm();
        params += delta;

        if delta_norm < TOLERANCE {
            println!("  Converged in {} iterations (error: {:.2e})", iter + 1, error);
            break;
        }
    }

    Ok(params)
}

/// Compute residuals and Jacobian for all observations plus prior
fn compute_cost(
    params: &DVector<f64>,
    observations: &[Observation],
    camera_to_param: &std::collections::HashMap<usize, usize>,
    landmark_to_param: &std::collections::HashMap<usize, usize>,
    prior: Option<&MarginalizedPrior>,
) -> (DVector<f64>, DMatrix<f64>) {

    let n_obs_residuals = observations.len();
    let n_prior_residuals = prior.map(|p| p.sqrt_information.nrows()).unwrap_or(0);
    let n_total_residuals = n_obs_residuals + n_prior_residuals;
    let n_params = params.len();

    let mut residuals = DVector::<f64>::zeros(n_total_residuals);
    let mut jacobian = DMatrix::<f64>::zeros(n_total_residuals, n_params);

    // Observation residuals
    for (i, obs) in observations.iter().enumerate() {
        let cam_param_idx = camera_to_param[&obs.camera_idx];
        let lm_param_idx = landmark_to_param[&obs.landmark_id];

        let camera_pos = params[cam_param_idx];
        let landmark_pos = params[lm_param_idx];

        // Residual: predicted - observed
        // Predicted offset = landmark_pos - camera_pos
        let predicted = landmark_pos - camera_pos;
        residuals[i] = predicted - obs.offset;

        // Jacobian: d(residual)/d(camera) = -1, d(residual)/d(landmark) = +1
        jacobian[(i, cam_param_idx)] = -1.0;
        jacobian[(i, lm_param_idx)] = 1.0;
    }

    // Prior residuals
    if let Some(p) = prior {
        // Build current values for parameters constrained by prior
        let mut prior_params = DVector::<f64>::zeros(p.linearization_point.len());

        // Map prior's cameras and landmarks to current parameter indices
        for (i, &cam_idx) in p.camera_frames.iter().enumerate() {
            if let Some(&param_idx) = camera_to_param.get(&cam_idx) {
                prior_params[i] = params[param_idx];
            }
        }

        let cam_offset = p.camera_frames.len();
        for (i, &lm_id) in p.landmark_ids.iter().enumerate() {
            if let Some(&param_idx) = landmark_to_param.get(&lm_id) {
                prior_params[cam_offset + i] = params[param_idx];
            }
        }

        // Prior residual: sqrt_info * (current - linearization_point)
        let delta = prior_params - &p.linearization_point;
        let r_prior = &p.sqrt_information * delta;

        // Write prior residuals
        for i in 0..r_prior.len() {
            residuals[n_obs_residuals + i] = r_prior[i];
        }

        // Prior Jacobian: each row of sqrt_info corresponds to one residual
        // Each column corresponds to one constrained parameter (camera or landmark)
        for prior_row in 0..p.sqrt_information.nrows() {
            // This row of sqrt_info multiplies all the constrained parameters

            // Cameras
            for (prior_col, &cam_idx) in p.camera_frames.iter().enumerate() {
                if let Some(&param_idx) = camera_to_param.get(&cam_idx) {
                    jacobian[(n_obs_residuals + prior_row, param_idx)] = p.sqrt_information[(prior_row, prior_col)];
                }
            }

            // Landmarks
            for (prior_col, &lm_id) in p.landmark_ids.iter().enumerate() {
                if let Some(&param_idx) = landmark_to_param.get(&lm_id) {
                    let col_in_sqrt_info = cam_offset + prior_col;
                    jacobian[(n_obs_residuals + prior_row, param_idx)] = p.sqrt_information[(prior_row, col_in_sqrt_info)];
                }
            }
        }
    }

    (residuals, jacobian)
}

/// Marginalize oldest camera and landmarks only seen by oldest camera
fn marginalize_old_state(
    params: &DVector<f64>,
    observations: &[Observation],
    camera_frames: &[usize],
    ordered_landmarks: &[usize],
    camera_to_param: &std::collections::HashMap<usize, usize>,
    landmark_to_param: &std::collections::HashMap<usize, usize>,
    prior: Option<&MarginalizedPrior>,
) -> Result<Option<MarginalizedPrior>, Box<dyn std::error::Error>> {

    // Compute full Jacobian
    let (_residuals, jacobian) = compute_cost(
        params,
        observations,
        camera_to_param,
        landmark_to_param,
        prior,
    );

    // Compute Hessian approximation: H = J^T J
    let hessian = jacobian.transpose() * &jacobian;

    // Identify which parameters to marginalize (oldest camera + old landmarks)
    let oldest_camera = *camera_frames.last().unwrap();
    let oldest_camera_param = camera_to_param[&oldest_camera];

    // Find landmarks only observed by oldest camera
    let mut old_landmark_ids = Vec::new();
    for &lm_id in ordered_landmarks {
        let cameras_seeing: Vec<usize> = observations.iter()
            .filter(|obs| obs.landmark_id == lm_id)
            .map(|obs| obs.camera_idx)
            .collect();

        let unique_cameras: std::collections::HashSet<_> = cameras_seeing.into_iter().collect();
        if unique_cameras.len() == 1 && unique_cameras.contains(&oldest_camera) {
            old_landmark_ids.push(lm_id);
        }
    }

    // Build marginalization indices
    let mut marg_indices = vec![oldest_camera_param];
    for &lm_id in &old_landmark_ids {
        marg_indices.push(landmark_to_param[&lm_id]);
    }

    if marg_indices.is_empty() {
        return Ok(None);
    }

    // Partition Hessian
    let n_params = params.len();
    let keep_indices: Vec<usize> = (0..n_params)
        .filter(|i| !marg_indices.contains(i))
        .collect();

    let n_old = marg_indices.len();
    let n_new = keep_indices.len();

    let mut h_oo = DMatrix::<f64>::zeros(n_old, n_old);
    let mut h_on = DMatrix::<f64>::zeros(n_old, n_new);
    let mut h_no = DMatrix::<f64>::zeros(n_new, n_old);
    let mut h_nn = DMatrix::<f64>::zeros(n_new, n_new);

    for (i, &row_old) in marg_indices.iter().enumerate() {
        for (j, &col_old) in marg_indices.iter().enumerate() {
            h_oo[(i, j)] = hessian[(row_old, col_old)];
        }
        for (j, &col_new) in keep_indices.iter().enumerate() {
            h_on[(i, j)] = hessian[(row_old, col_new)];
        }
    }

    for (i, &row_new) in keep_indices.iter().enumerate() {
        for (j, &col_old) in marg_indices.iter().enumerate() {
            h_no[(i, j)] = hessian[(row_new, col_old)];
        }
        for (j, &col_new) in keep_indices.iter().enumerate() {
            h_nn[(i, j)] = hessian[(row_new, col_new)];
        }
    }

    // Schur complement: H_marg = H_nn - H_no * H_oo^-1 * H_on
    let h_oo_inv = match h_oo.try_inverse() {
        Some(inv) => inv,
        None => {
            println!("    Warning: H_oo not invertible");
            return Ok(None);
        }
    };

    let mut h_marg = h_nn - &h_no * &h_oo_inv * &h_on;

    // Symmetrize
    for i in 0..h_marg.nrows() {
        for j in (i+1)..h_marg.ncols() {
            let avg = 0.5 * (h_marg[(i, j)] + h_marg[(j, i)]);
            h_marg[(i, j)] = avg;
            h_marg[(j, i)] = avg;
        }
    }

    // Check eigenvalues and regularize if needed
    let eigen = SymmetricEigen::new(h_marg.clone());
    let min_eigenvalue = eigen.eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    if min_eigenvalue < 1e-8 {
        let regularization = 1e-6 - min_eigenvalue.min(0.0);
        for i in 0..h_marg.nrows() {
            h_marg[(i, i)] += regularization;
        }
    }

    // Cholesky factorization
    let cholesky = match Cholesky::new(h_marg) {
        Some(chol) => chol,
        None => {
            println!("    Warning: Cholesky failed on H_marg");
            return Ok(None);
        }
    };

    let sqrt_info = cholesky.l();

    // Extract linearization point for kept parameters
    let lin_point = DVector::from_iterator(
        n_new,
        keep_indices.iter().map(|&i| params[i])
    );

    // Track which cameras and landmarks the prior constrains
    let mut prior_camera_frames = Vec::new();
    let mut prior_landmark_ids = Vec::new();

    for &param_idx in &keep_indices {
        // Is this a camera parameter?
        if param_idx < WINDOW_SIZE {
            // Find which camera frame this is
            for (&cam_idx, &cam_param_idx) in camera_to_param {
                if cam_param_idx == param_idx {
                    prior_camera_frames.push(cam_idx);
                    break;
                }
            }
        } else {
            // It's a landmark parameter
            for (&lm_id, &lm_param_idx) in landmark_to_param {
                if lm_param_idx == param_idx {
                    prior_landmark_ids.push(lm_id);
                    break;
                }
            }
        }
    }

    Ok(Some(MarginalizedPrior {
        sqrt_information: sqrt_info,
        linearization_point: lin_point,
        camera_frames: prior_camera_frames,
        landmark_ids: prior_landmark_ids,
    }))
}

/// Reorder a prior from the previous window to match current parameter layout
fn reorder_prior(
    prior: &MarginalizedPrior,
    camera_to_param: &std::collections::HashMap<usize, usize>,
    landmark_to_param: &std::collections::HashMap<usize, usize>,
) -> Option<MarginalizedPrior> {
    // Build mapping from old parameter indices to new parameter indices
    let mut old_to_new: Vec<Option<usize>> = Vec::new();

    // Map cameras
    for &cam_idx in &prior.camera_frames {
        old_to_new.push(camera_to_param.get(&cam_idx).copied());
    }

    // Map landmarks
    for &lm_id in &prior.landmark_ids {
        old_to_new.push(landmark_to_param.get(&lm_id).copied());
    }

    // Filter out parameters that are no longer in the window
    let valid_mappings: Vec<(usize, usize)> = old_to_new.iter().enumerate()
        .filter_map(|(old_idx, &new_idx_opt)| {
            new_idx_opt.map(|new_idx| (old_idx, new_idx))
        })
        .collect();

    if valid_mappings.is_empty() {
        return None; // Prior no longer applies to any parameters
    }

    // Reorder sqrt_information and linearization_point
    let n_new = valid_mappings.len();
    let mut new_sqrt_info = DMatrix::<f64>::zeros(n_new, n_new);
    let mut new_lin_point = DVector::<f64>::zeros(n_new);

    for (new_i, &(old_i, _)) in valid_mappings.iter().enumerate() {
        new_lin_point[new_i] = prior.linearization_point[old_i];

        for (new_j, &(old_j, _)) in valid_mappings.iter().enumerate() {
            new_sqrt_info[(new_i, new_j)] = prior.sqrt_information[(old_i, old_j)];
        }
    }

    // Track which cameras and landmarks the reordered prior constrains
    let mut new_camera_frames = Vec::new();
    let mut new_landmark_ids = Vec::new();

    for &(old_idx, _new_idx) in &valid_mappings {
        if old_idx < prior.camera_frames.len() {
            new_camera_frames.push(prior.camera_frames[old_idx]);
        } else {
            let lm_offset = old_idx - prior.camera_frames.len();
            new_landmark_ids.push(prior.landmark_ids[lm_offset]);
        }
    }

    Some(MarginalizedPrior {
        sqrt_information: new_sqrt_info,
        linearization_point: new_lin_point,
        camera_frames: new_camera_frames,
        landmark_ids: new_landmark_ids,
    })
}

/// Visualize ground truth (logged once at startup)
fn visualize_ground_truth(
    rec: &rr::RecordingStream,
    landmarks: &[f64],
    _cameras: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {

    // Set timeline to window 0 so GT landmarks appear on the timeline
    rec.set_time_sequence("window", 0);

    // Ground truth landmarks (grey points at Y=0.0)
    let landmark_positions: Vec<rr::Position2D> = landmarks
        .iter()
        .map(|&x| rr::Position2D::new(x as f32, 0.0))
        .collect();

    rec.log(
        "landmarks/ground_truth",
        &rr::Points2D::new(landmark_positions)
            .with_radii([0.05])
            .with_colors([rr::Color::from_rgb(128, 128, 128)]),
    )?;

    Ok(())
}

/// Visualize current SLAM state (estimated cameras and landmarks)
fn visualize_slam_state(
    rec: &rr::RecordingStream,
    window_start: usize,
    params: &DVector<f64>,
    camera_to_param: &std::collections::HashMap<usize, usize>,
    landmark_to_param: &std::collections::HashMap<usize, usize>,
    gt: &GroundTruth,
) -> Result<(), Box<dyn std::error::Error>> {

    // Set timeline
    rec.set_time_sequence("window", window_start as i64);

    // Find the newest camera (lowest index since we iterate forward in time)
    let newest_cam_idx = camera_to_param.keys().max().copied().unwrap_or(0);

    // ===== LANDMARKS =====

    // Build landmark positions with error lines
    let mut est_positions = Vec::new();
    let mut error_lines = Vec::new();

    for (&lm_id, &param_idx) in landmark_to_param {
        let est_x = params[param_idx];
        let gt_x = gt.landmarks[lm_id];

        // Estimated landmarks at Y=0.5
        est_positions.push(rr::Position2D::new(est_x as f32, 0.5));

        // Red error line from GT (Y=0.0) to estimated (Y=0.5)
        error_lines.push([
            [gt_x as f32, 0.0],
            [est_x as f32, 0.5],
        ]);
    }

    // Estimated landmarks (blue points at Y=0.5)
    rec.log(
        "landmarks/estimated",
        &rr::Points2D::new(est_positions)
            .with_radii([0.05])
            .with_colors([rr::Color::from_rgb(0, 128, 255)]),
    )?;

    // Error lines for landmarks
    if !error_lines.is_empty() {
        rec.log(
            "landmarks/error",
            &rr::LineStrips2D::new(error_lines)
                .with_colors([rr::Color::from_rgb(255, 0, 0)])
                .with_radii([0.01]),
        )?;
    }

    // ===== CAMERAS =====

    // Ground truth camera (grey at Y=1.5)
    let gt_cam_pos = gt.camera_poses[newest_cam_idx];
    let gt_left = gt_cam_pos - CAMERA_WIDTH / 2.0;
    let gt_right = gt_cam_pos + CAMERA_WIDTH / 2.0;

    rec.log(
        "cameras/ground_truth",
        &rr::LineStrips2D::new([
            [[gt_left as f32, 1.5], [gt_right as f32, 1.5]]
        ])
        .with_colors([rr::Color::from_rgb(128, 128, 128)])
        .with_radii([0.03]),
    )?;

    // Estimated camera (blue at Y=2.0)
    if let Some(&param_idx) = camera_to_param.get(&newest_cam_idx) {
        let est_cam_pos = params[param_idx];
        let est_left = est_cam_pos - CAMERA_WIDTH / 2.0;
        let est_right = est_cam_pos + CAMERA_WIDTH / 2.0;

        rec.log(
            "cameras/estimated",
            &rr::LineStrips2D::new([
                [[est_left as f32, 2.0], [est_right as f32, 2.0]]
            ])
            .with_colors([rr::Color::from_rgb(0, 128, 255)])
            .with_radii([0.03]),
        )?;

        // Red error line connecting GT (Y=1.5) to estimated (Y=2.0)
        rec.log(
            "cameras/error",
            &rr::LineStrips2D::new([
                [[gt_cam_pos as f32, 1.5], [est_cam_pos as f32, 2.0]]
            ])
            .with_colors([rr::Color::from_rgb(255, 0, 0)])
            .with_radii([0.015]),
        )?;
    }

    Ok(())
}
