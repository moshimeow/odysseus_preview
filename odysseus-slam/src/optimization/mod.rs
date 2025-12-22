//! Optimization and bundle adjustment

use crate::camera::StereoCamera;
use crate::geometry::StereoObservation;
use crate::math::SE3;
use crate::frame_graph::{FrameGraph, OptimizationState};
use crate::world_state::WorldState;
use nalgebra::{SymmetricEigen, Cholesky, DMatrix, DVector};
use odysseus_solver::math3d::Vec3;
use odysseus_solver::{Jet, Real};
use odysseus_solver::{SparseLevenbergMarquardt, build_slam_entries, sparse_solver::build_jacobian};
use std::collections::{HashMap, HashSet};
use sprs::CsMat;



#[derive(Clone)]
pub struct MarginalizedPrior {
    /// Cholesky factor of marginalized Hessian: H_marg = L * L^T
    pub sqrt_information: DMatrix<f64>,
    /// Linearization point for the kept parameters
    pub linearization_point: DVector<f64>,
    /// Camera frame IDs constrained by this prior (in order)
    pub camera_frame_ids: Vec<usize>,
    /// Point IDs constrained by this prior (in order, after cameras)
    pub point_ids: Vec<usize>,
}

/// Configuration for bundle adjustment
#[derive(Debug, Clone)]
pub struct BundleAdjustmentConfig {
    /// Maximum solver iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum number of map points to include
    pub max_active_points: usize,
    /// Huber loss threshold (pixels)
    pub huber_delta: f64,
}

impl Default for BundleAdjustmentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            tolerance: 1e-6,
            max_active_points: 600,
            huber_delta: 1.0,
        }
    }
}

impl BundleAdjustmentConfig {
    /// Configuration for Local Bundle Adjustment (fast, fewer iterations)
    pub fn lba() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 1e-6,
            max_active_points: 600,
            huber_delta: 1.0,
        }
    }

    /// Configuration for Global Bundle Adjustment (thorough, more iterations)
    pub fn gba() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-8,
            max_active_points: 800,
            huber_delta: 1.0,
        }
    }
}

/// Result of bundle adjustment optimization
pub struct BundleAdjustmentResult {
    /// Time spent in the solver (milliseconds)
    pub solve_time_ms: f64,
    /// Number of observations used
    pub n_observations: usize,
    /// Number of optimized poses
    pub n_poses: usize,
    /// Number of optimized points
    pub n_points: usize,
    /// New marginalized prior (if marginalization was performed)
    pub new_prior: Option<MarginalizedPrior>,
}

/// Run bundle adjustment with support for fixed (GBA-refined) points
///
/// Points in `fixed_point_ids` contribute to residuals but are not optimized.
/// This is useful for LBA where GBA-refined points should act as stable landmarks.
///
/// # Arguments
/// * `stereo_camera` - Stereo camera model
/// * `frame_graph` - Frame graph specifying which frames are optimized/fixed
/// * `world` - World state (poses and points) - modified in place
/// * `frame_observations` - Observations per frame, indexed by frame_idx
/// * `fixed_point_ids` - Set of point IDs that should be fixed (not optimized)
/// * `config` - Solver configuration
pub fn run_bundle_adjustment(
    stereo_camera: &StereoCamera<f64>,
    frame_graph: &FrameGraph,
    world: &mut WorldState,
    frame_observations: &[Vec<StereoObservation>],
    prior: Option<&MarginalizedPrior>,
    fixed_point_ids: &HashSet<usize>,
    config: &BundleAdjustmentConfig,
) -> BundleAdjustmentResult {
    // ========== 1. Collect observations and select active points ==========
    
    let mut all_obs: Vec<StereoObservation> = Vec::new();
    let mut score_map: HashMap<usize, f64> = HashMap::new();

    for (frame_idx, frame_state) in frame_graph.states.iter().enumerate() {
        if !frame_state.is_active_in_ba() {
            continue;
        }
        for obs in &frame_observations[frame_idx] {
            if world.get_point(obs.point_id).is_some() {
                all_obs.push(*obs);
                if frame_state.is_active_in_ba() {
                    *score_map.entry(obs.point_id).or_insert(0.0) += 1.0;
                }
            }
        }
    }
    
    // Sort points by score descending, truncate to max
    let mut scored_points: Vec<_> = score_map.clone().into_iter().collect();
    scored_points.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored_points.truncate(config.max_active_points);
    let all_point_ids: Vec<usize> = scored_points.into_iter().map(|(id, _)| id).collect();

    // Partition into optimized and fixed points
    let optimized_point_ids: Vec<usize> = all_point_ids.iter()
        .copied()
        .filter(|id| !fixed_point_ids.contains(id))
        .collect();
    
    // All points we care about (for filtering observations)
    let all_points_set: HashSet<_> = all_point_ids.iter().copied().collect();
    let observations: Vec<_> = all_obs
        .into_iter()
        .filter(|obs| all_points_set.contains(&obs.point_id))
        .collect();

    if observations.is_empty() {
        return BundleAdjustmentResult {
            solve_time_ms: 0.0,
            n_observations: 0,
            n_poses: 0,
            n_points: 0,
            new_prior: prior.cloned(),
        };
    }

    // ========== 2. Build parameter mappings ==========
    
    // Pose parameters
    let mut pose_to_param_idx: HashMap<usize, usize> = HashMap::new();
    let mut offset = 0;
    
    let optimized_frame_indices: Vec<_> = frame_graph.states.iter().enumerate()
        .filter(|(_, s)| s.is_optimized())
        .map(|(idx, _)| idx)
        .collect();
    let mut marginalized_frame_indices: Vec<usize> = Vec::new();
    for frame_idx in optimized_frame_indices {
        pose_to_param_idx.insert(frame_idx, offset);
        offset += 6;
        if frame_graph.states[frame_idx].state == OptimizationState::Marginalize {
            marginalized_frame_indices.push(frame_idx);
        }
    }
    let n_poses = pose_to_param_idx.len();

    // Point parameters (only for non-fixed points)
    let mut point_to_param_idx: HashMap<usize, usize> = HashMap::new();
    for &point_id in &optimized_point_ids {
        point_to_param_idx.insert(point_id, offset);
        offset += 3;
    }
    let n_optimized_points = optimized_point_ids.len();
    let _n_fixed_points = all_point_ids.len() - n_optimized_points;
    let n_observations = observations.len();

    // ========== 3. Build sparsity entries ==========

    let sparsity_obs: Vec<_> = observations.iter().enumerate().map(|(i, obs)| {
        let pose_optimized = frame_graph.states[obs.camera_id].is_optimized();
        let point_optimized = point_to_param_idx.contains_key(&obs.point_id);

        let pose_start = pose_to_param_idx.get(&obs.camera_id).copied().unwrap_or(0);
        let point_start = point_to_param_idx.get(&obs.point_id).copied().unwrap_or(0);

        (i * 4, pose_start, point_start, pose_optimized, point_optimized)
    }).collect();

    let n_params = n_poses * 6 + n_optimized_points * 3;
    let n_obs_residuals = n_observations * 4;
    let mut entries = build_slam_entries(&sparsity_obs);

    // Add prior residual entries (dense rows - each prior residual depends on all prior-constrained params)
    let n_prior_residuals = prior.map(|p| p.sqrt_information.nrows()).unwrap_or(0);
    if let Some(p) = prior {
        // Build list of parameter indices that this prior constrains
        let mut prior_param_indices: Vec<usize> = Vec::new();
        for &frame_id in &p.camera_frame_ids {
            if let Some(&param_start) = pose_to_param_idx.get(&frame_id) {
                for i in 0..6 {
                    prior_param_indices.push(param_start + i);
                }
            }
        }
        for &point_id in &p.point_ids {
            if let Some(&param_start) = point_to_param_idx.get(&point_id) {
                for i in 0..3 {
                    prior_param_indices.push(param_start + i);
                }
            }
        }

        // Each prior residual row connects to all prior-constrained parameters
        for prior_row in 0..n_prior_residuals {
            let residual_idx = n_obs_residuals + prior_row;
            for &param_idx in &prior_param_indices {
                entries.push((residual_idx, param_idx));
            }
        }
        // Re-sort entries for CSR format
        entries.sort_by_key(|&(row, col)| (row, col));
    }

    let n_residuals = n_obs_residuals + n_prior_residuals;

    // ========== 4. Pack parameters ==========

    let mut initial_params = DVector::<f64>::zeros(n_params);

    for (&frame_idx, &param_idx) in &pose_to_param_idx {
        // pose.rotation is the delta (Vec3), pose.translation is world coords
        let pose = &world.frames[frame_idx].pose;
        initial_params[param_idx + 0] = pose.rotation.x;
        initial_params[param_idx + 1] = pose.rotation.y;
        initial_params[param_idx + 2] = pose.rotation.z;
        initial_params[param_idx + 3] = pose.translation.x;
        initial_params[param_idx + 4] = pose.translation.y;
        initial_params[param_idx + 5] = pose.translation.z;
    }
    
    for &point_id in &optimized_point_ids {
        let param_idx = point_to_param_idx[&point_id];
        let point = world.get_point(point_id).unwrap();
        initial_params[param_idx + 0] = point.x;
        initial_params[param_idx + 1] = point.y;
        initial_params[param_idx + 2] = point.z;
    }

    // ========== 5. Solve ==========
    
    // Handle case where there are no parameters to optimize
    if n_params == 0 {
        return BundleAdjustmentResult {
            solve_time_ms: 0.0,
            n_observations,
            n_poses: 0,
            n_points: 0,
            new_prior: prior.cloned(),
        };
    }

    let mut solver = SparseLevenbergMarquardt::<f64>::new(n_residuals, n_params, &entries)
        .with_tolerance(config.tolerance)
        .with_max_iterations(config.max_iterations)
        .with_verbose(false);

    let start = std::time::Instant::now();

    let optimized_params = solver.solve(
        initial_params,
        |params, residuals, jacobian_data| {
            compute_cost(
                params,
                residuals,
                jacobian_data,
                &observations,
                &pose_to_param_idx,
                &point_to_param_idx,
                world,
                stereo_camera,
                config.huber_delta,
                prior,
                n_obs_residuals,
            );
        },
        |_iter, _result, _params| {},
    );

    let solve_time = start.elapsed();

    // ========== 6. Marginalize frames ==========
    let new_prior = if !marginalized_frame_indices.is_empty() {
        // Compute full Jacobian at optimized params
        let mut jacobian = build_jacobian::<f64>(&entries, n_residuals, n_params);
        let mut residuals = DVector::zeros(n_residuals);
        compute_cost(
            &optimized_params,
            residuals.as_mut_slice(),
            jacobian.data_mut(),
            &observations,
            &pose_to_param_idx,
            &point_to_param_idx,
            world,
            stereo_camera,
            config.huber_delta,
            prior,
            n_obs_residuals,
        );

        // Compute Hessian approximation: H = J^T * J
        let jt: CsMat<f64> = jacobian.clone().transpose_into();
        let sprs_hessian = &jt * &jacobian;
        let mut hessian = DMatrix::<f64>::zeros(n_params, n_params);
        for (value, (row, col)) in sprs_hessian.iter() {
            hessian[(row, col)] = *value;
        }

        // Build set of marginalized frame indices for quick lookup
        let marg_frame_set: HashSet<usize> = marginalized_frame_indices.iter().copied().collect();

        // Find points that are ONLY observed by marginalized frames
        // A point is "unique to marginalized frames" if all its observations come from marginalized frames
        let mut unique_point_ids: Vec<usize> = Vec::new();
        for (&point_id, _) in &point_to_param_idx {
            let observing_frames: HashSet<usize> = observations.iter()
                .filter(|obs| obs.point_id == point_id)
                .map(|obs| obs.camera_id)
                .collect();

            // Check if ALL observing frames are in the marginalized set
            if observing_frames.iter().all(|f| marg_frame_set.contains(f)) {
                unique_point_ids.push(point_id);
            }
        }

        // Build list of ALL parameter indices to marginalize
        let mut marg_indices: Vec<usize> = Vec::new();

        // Add all 6 camera parameters for each marginalized frame
        for &frame_idx in &marginalized_frame_indices {
            let camera_param_start = pose_to_param_idx[&frame_idx];
            for i in 0..6 {
                marg_indices.push(camera_param_start + i);
            }
        }

        // Add all 3 parameters for each unique point
        for &point_id in &unique_point_ids {
            if let Some(&point_param_start) = point_to_param_idx.get(&point_id) {
                for i in 0..3 {
                    marg_indices.push(point_param_start + i);
                }
            }
        }

        // Partition Hessian into [old | keep] blocks
        let keep_indices: Vec<usize> = (0..n_params)
            .filter(|i| !marg_indices.contains(i))
            .collect();
        let n_old = marg_indices.len();
        let n_new = keep_indices.len();

        if n_old == 0 || n_new == 0 {
            None // Nothing to marginalize or nothing left
        } else {
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
                    eprintln!("Warning: H_oo not invertible during marginalization");
                    return BundleAdjustmentResult {
                        solve_time_ms: solve_time.as_secs_f64() * 1000.0,
                        n_observations,
                        n_poses,
                        n_points: n_optimized_points,
                        new_prior: prior.cloned(),
                    };
                }
            };
            let mut h_marg = h_nn - &h_no * &h_oo_inv * &h_on;

            // Symmetrize
            for i in 0..h_marg.nrows() {
                for j in (i + 1)..h_marg.ncols() {
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

            // Cholesky factorization to get sqrt_information
            let cholesky = match Cholesky::new(h_marg.clone()) {
                Some(chol) => chol,
                None => {
                    eprintln!("Warning: Cholesky failed during marginalization");
                    return BundleAdjustmentResult {
                        solve_time_ms: solve_time.as_secs_f64() * 1000.0,
                        n_observations,
                        n_poses,
                        n_points: n_optimized_points,
                        new_prior: prior.cloned(),
                    };
                }
            };
            let sqrt_information = cholesky.l();
            let linearization_point = DVector::from_iterator(
                n_new,
                keep_indices.iter().map(|&i| optimized_params[i]),
            );

            // Track which camera frame IDs and point IDs are in the kept parameters
            // Build reverse mappings: param_start -> frame_id or point_id
            let param_to_frame: HashMap<usize, usize> = pose_to_param_idx
                .iter()
                .map(|(&frame_id, &param_start)| (param_start, frame_id))
                .collect();
            let param_to_point: HashMap<usize, usize> = point_to_param_idx
                .iter()
                .map(|(&point_id, &param_start)| (param_start, point_id))
                .collect();

            let mut kept_camera_frame_ids: Vec<usize> = Vec::new();
            let mut kept_point_ids: Vec<usize> = Vec::new();

            for &keep_idx in &keep_indices {
                if let Some(&frame_id) = param_to_frame.get(&keep_idx) {
                    if !kept_camera_frame_ids.contains(&frame_id) {
                        kept_camera_frame_ids.push(frame_id);
                    }
                }
                if let Some(&point_id) = param_to_point.get(&keep_idx) {
                    if !kept_point_ids.contains(&point_id) {
                        kept_point_ids.push(point_id);
                    }
                }
            }

            // NOTE: Do NOT sort! The order must match keep_indices order,
            // which is what sqrt_information and linearization_point use.

            Some(MarginalizedPrior {
                sqrt_information,
                linearization_point,
                camera_frame_ids: kept_camera_frame_ids,
                point_ids: kept_point_ids,
            })
        }
    } else {
        None
    };

    // ========== 7. Unpack results ==========

    for (&frame_idx, &param_idx) in &pose_to_param_idx {
        let rot_delta = Vec3::new(
            optimized_params[param_idx + 0],
            optimized_params[param_idx + 1],
            optimized_params[param_idx + 2],
        );
        let translation = Vec3::new(
            optimized_params[param_idx + 3],
            optimized_params[param_idx + 4],
            optimized_params[param_idx + 5],
        );
        world.frames[frame_idx].pose.set_from_params(rot_delta, translation);
    }
    
    for &point_id in &optimized_point_ids {
        let param_idx = point_to_param_idx[&point_id];
        let new_position = Vec3::new(
            optimized_params[param_idx + 0],
            optimized_params[param_idx + 1],
            optimized_params[param_idx + 2],
        );
        world.update_point(point_id, new_position);
    }

    BundleAdjustmentResult {
        solve_time_ms: solve_time.as_secs_f64() * 1000.0,
        n_observations,
        n_poses,
        n_points: n_optimized_points,
        new_prior,
    }
}

// ========== Helper functions ==========

/// Dispatch code over different Jet sizes based on two runtime conditions.
/// 
/// Handles all 4 combinations of (cond1, cond2) with appropriate Jet sizes.
/// The Jet size is computed as: size1 * cond1 + size2 * cond2
/// Outputs constant flags to placate the array bounds checker
///
/// Example for BA where pose has 6 params and point has 3:
/// ```ignore
/// with_jet_size!(pose_active, point_active, { 6, 3 }, |JetN, POSE, POINT| {
///     // POSE and POINT are const bools
///     // JetN is Jet<f64, 0/3/6/9> depending on combination
/// });
/// ```
macro_rules! with_jet_size {
    ($cond1:expr, $cond2:expr, { $size1:literal, $size2:literal }, |$jet_ty:ident, $flag1:ident, $flag2:ident| $body:expr) => {{
        // Inner helper captures the pattern for each arm
        macro_rules! arm {
            ($v1:literal, $v2:literal, $size:expr) => {{
                #[allow(non_upper_case_globals)]
                const $flag1: bool = $v1;
                #[allow(non_upper_case_globals)]
                const $flag2: bool = $v2;
                type $jet_ty = Jet<f64, $size>;
                $body
            }};
        }
        match ($cond1, $cond2) {
            (false, false) => arm!(false, false, 0),
            (false, true)  => arm!(false, true, $size2),
            (true, false)  => arm!(true, false, $size1),
            (true, true)   => arm!(true, true, { $size1 + $size2 }),
        }
    }};
}

/// Convert an f64 array to constant jets (no derivatives)
fn jet_constants<const N: usize, const D: usize>(arr: &[f64; N]) -> [Jet<f64, D>; N] {
    std::array::from_fn(|i| Jet::constant(arr[i]))
}

/// Create variable jets from consecutive params with sequential derivative indices
fn jet_variables<const N: usize, const D: usize, P: std::ops::Index<usize, Output = f64>>(
    params: &P,
    param_offset: usize,
    deriv_offset: usize,
) -> [Jet<f64, D>; N] {
    std::array::from_fn(|i| Jet::variable(params[param_offset + i], deriv_offset + i))
}

/// Compute stereo reprojection residual with host-relative rotation parameterization
///
/// The pose is parameterized as:
/// - rotation = q_host * exp(rotation_delta)
/// - translation = direct world coordinates
///
/// This keeps rotation parameters small, avoiding the rotation vector singularity at 2π.
pub fn stereo_reprojection_residual_host_relative<T: Real>(
    rotation_host: &odysseus_solver::math3d::Quat<f64>,
    pose_params: &[T; 6],  // [rotation_delta (3), translation (3)]
    world_point: &[T; 3],
    stereo_camera: &StereoCamera<T>,
    observed_left_u: T,
    observed_left_v: T,
    observed_right_u: T,
    observed_right_v: T,
) -> (T, T, T, T) {
    // Build rotation: q_host * exp(delta)
    let rot_delta = Vec3::new(pose_params[0], pose_params[1], pose_params[2]);
    let q_delta = odysseus_solver::math3d::Quat::from_axis_angle(rot_delta);

    // Compose with host (host is f64, delta is T)
    // q_new = q_host * q_delta
    let q_host_t = odysseus_solver::math3d::Quat::new(
        T::from_literal(rotation_host.w),
        T::from_literal(rotation_host.x),
        T::from_literal(rotation_host.y),
        T::from_literal(rotation_host.z),
    );
    let q_new = q_host_t * q_delta;

    // Build world_T_camera pose
    let translation = Vec3::new(pose_params[3], pose_params[4], pose_params[5]);
    let world_t_camera = SE3::from_rotation_translation(
        crate::math::SO3 { quat: q_new },
        translation,
    );

    // Transform world point to camera frame
    let camera_t_world = world_t_camera.inverse();
    let point_world = Vec3::new(world_point[0], world_point[1], world_point[2]);
    let point_camera = camera_t_world.transform_point(point_world);

    // Project to image
    let (pred_lu, pred_lv, pred_ru, pred_rv) = stereo_camera.project_stereo(point_camera);
    (
        observed_left_u - pred_lu,
        observed_left_v - pred_lv,
        observed_right_u - pred_ru,
        observed_right_v - pred_rv,
    )
}

/// Apply Huber loss to a single residual and its Jacobian row
#[inline]
fn apply_huber_loss(huber_delta: f64, residual: &mut f64, jacobian_row: &mut [f64]) {
    let abs_r = residual.abs();
    if abs_r > huber_delta {
        let weight = (huber_delta / abs_r).sqrt();
        *residual *= weight;
        for j in jacobian_row.iter_mut() {
            *j *= weight;
        }
    }
}

/// Get a point's XYZ position - either from params (if optimized) or from world (if fixed)
fn get_point_xyz(
    point_id: usize,
    params: &DVector<f64>,
    point_to_param_idx: &HashMap<usize, usize>,
    world: &WorldState,
) -> [f64; 3] {
    if let Some(&param_idx) = point_to_param_idx.get(&point_id) {
        [params[param_idx], params[param_idx + 1], params[param_idx + 2]]
    } else {
        let point = world.get_point(point_id).unwrap();
        [point.x, point.y, point.z]
    }
}

/// Compute cost function with support for fixed points and marginalized prior
///
/// The jacobian_data slice is written in row-major order matching the CSR structure:
/// - Rows are in observation order (4 residuals per observation)
/// - Within each row, pose params come first (if active), then point params (if active)
/// - Prior residuals come after observation residuals (starting at n_obs_residuals)
pub fn compute_cost(
    params: &DVector<f64>,
    residuals: &mut [f64],
    jacobian_data: &mut [f64],
    observations: &[StereoObservation],
    pose_to_param_idx: &HashMap<usize, usize>,
    point_to_param_idx: &HashMap<usize, usize>,
    world: &WorldState,
    stereo_camera: &StereoCamera<f64>,
    huber_delta: f64,
    prior: Option<&MarginalizedPrior>,
    n_obs_residuals: usize,
) {
    // Cursor into jacobian_data - we write linearly in row-major order
    let mut jac_cursor = 0;

    for (obs_idx, obs) in observations.iter().enumerate() {
        let res_offset = obs_idx * 4;
        let pose_optimized = pose_to_param_idx.contains_key(&obs.camera_id);
        let point_optimized = point_to_param_idx.contains_key(&obs.point_id);

        with_jet_size!(pose_optimized, point_optimized, { 6, 3 }, |JetN, POSE_ACTIVE, POINT_ACTIVE| {
            // Derivative layout: pose[0:6] if active, then point[0:3 or 6:9] if active
            let point_deriv_offset: usize = if POSE_ACTIVE { 6 } else { 0 };

            // Get the host quaternion for this camera
            let rotation_host = &world.frames[obs.camera_id].pose.rotation_host;

            // Pose params: [rotation_delta, translation] - pose stores delta directly
            let pose_params: [JetN; 6] = if POSE_ACTIVE {
                let idx = pose_to_param_idx[&obs.camera_id];
                jet_variables(params, idx, 0)
            } else {
                // Fixed pose: read delta directly from pose (it's Vec3 now)
                let pose = &world.frames[obs.camera_id].pose;
                jet_constants(&[
                    pose.rotation.x, pose.rotation.y, pose.rotation.z,
                    pose.translation.x, pose.translation.y, pose.translation.z,
                ])
            };

            // Point - variable if optimized, constant if fixed
            let world_point: [JetN; 3] = if POINT_ACTIVE {
                let idx = point_to_param_idx[&obs.point_id];
                jet_variables(params, idx, point_deriv_offset)
            } else {
                let xyz = get_point_xyz(obs.point_id, params, point_to_param_idx, world);
                jet_constants(&xyz)
            };

            // Camera as jets
            let camera_jet = StereoCamera::new(
                crate::camera::PinholeCamera::new(
                    JetN::constant(stereo_camera.left.fx),
                    JetN::constant(stereo_camera.left.fy),
                    JetN::constant(stereo_camera.left.cx),
                    JetN::constant(stereo_camera.left.cy),
                ),
                JetN::constant(stereo_camera.baseline),
            );

            // Compute residuals using host-relative parameterization
            let (r1, r2, r3, r4) = stereo_reprojection_residual_host_relative(
                rotation_host,
                &pose_params,
                &world_point,
                &camera_jet,
                JetN::constant(obs.left_u),
                JetN::constant(obs.left_v),
                JetN::constant(obs.right_u),
                JetN::constant(obs.right_v),
            );

            // Write residuals and jacobian with Huber loss
            // Jacobian is written linearly - pose entries first, then point entries
            let residual_jets = [r1, r2, r3, r4];
            for (i, r) in residual_jets.iter().enumerate() {
                let mut r_val = r.value;

                // Build jacobian row based on what's active
                match (POSE_ACTIVE, POINT_ACTIVE) {
                    (true, true) => {
                        let mut combined = [
                            r.derivs[0], r.derivs[1], r.derivs[2], r.derivs[3], r.derivs[4], r.derivs[5],
                            r.derivs[6], r.derivs[7], r.derivs[8],
                        ];
                        apply_huber_loss(huber_delta, &mut r_val, &mut combined);
                        // Write pose params (6), then point params (3)
                        jacobian_data[jac_cursor..jac_cursor + 9].copy_from_slice(&combined);
                        jac_cursor += 9;
                    }
                    (true, false) => {
                        let mut pose_jac = [
                            r.derivs[0], r.derivs[1], r.derivs[2], r.derivs[3], r.derivs[4], r.derivs[5],
                        ];
                        apply_huber_loss(huber_delta, &mut r_val, &mut pose_jac);
                        jacobian_data[jac_cursor..jac_cursor + 6].copy_from_slice(&pose_jac);
                        jac_cursor += 6;
                    }
                    (false, true) => {
                        let mut point_jac = [r.derivs[0], r.derivs[1], r.derivs[2]];
                        apply_huber_loss(huber_delta, &mut r_val, &mut point_jac);
                        jacobian_data[jac_cursor..jac_cursor + 3].copy_from_slice(&point_jac);
                        jac_cursor += 3;
                    }
                    (false, false) => {
                        // No jacobian entries, just apply Huber to residual
                        let mut dummy = [0.0_f64; 0];
                        apply_huber_loss(huber_delta, &mut r_val, &mut dummy);
                    }
                }

                residuals[res_offset + i] = r_val;
            }
        });
    }

    // ========== Prior residuals ==========
    if let Some(p) = prior {
        // Build vector of current parameter values for prior-constrained params
        // Order: cameras first (6 params each), then points (3 params each)
        let n_prior_params = p.linearization_point.len();
        let mut current_prior_params = DVector::<f64>::zeros(n_prior_params);

        let mut prior_param_idx = 0;

        // Map prior's camera params to current window
        for &frame_id in &p.camera_frame_ids {
            if let Some(&param_start) = pose_to_param_idx.get(&frame_id) {
                for i in 0..6 {
                    current_prior_params[prior_param_idx] = params[param_start + i];
                    prior_param_idx += 1;
                }
            } else {
                // Camera not in current window - use linearization point values
                for _ in 0..6 {
                    current_prior_params[prior_param_idx] = p.linearization_point[prior_param_idx];
                    prior_param_idx += 1;
                }
            }
        }

        // Map prior's point params to current window
        for &point_id in &p.point_ids {
            if let Some(&param_start) = point_to_param_idx.get(&point_id) {
                for i in 0..3 {
                    current_prior_params[prior_param_idx] = params[param_start + i];
                    prior_param_idx += 1;
                }
            } else {
                // Point not in current window - use linearization point values
                for _ in 0..3 {
                    current_prior_params[prior_param_idx] = p.linearization_point[prior_param_idx];
                    prior_param_idx += 1;
                }
            }
        }

        // Compute prior residuals: r_prior = sqrt_info * (current - linearization)
        let delta = &current_prior_params - &p.linearization_point;
        let r_prior = &p.sqrt_information * &delta;

        // Write prior residuals
        for i in 0..r_prior.len() {
            residuals[n_obs_residuals + i] = r_prior[i];
        }

        // Write prior Jacobian
        // Each row of sqrt_info is the Jacobian for one prior residual
        // Columns are ordered: cameras (6 params each), then points (3 params each)
        let n_prior_residuals = p.sqrt_information.nrows();

        for prior_row in 0..n_prior_residuals {
            let mut prior_col = 0;

            // Camera columns
            for &frame_id in &p.camera_frame_ids {
                if pose_to_param_idx.contains_key(&frame_id) {
                    // This camera is in current window - write 6 Jacobian entries
                    for i in 0..6 {
                        jacobian_data[jac_cursor] = p.sqrt_information[(prior_row, prior_col + i)];
                        jac_cursor += 1;
                    }
                }
                prior_col += 6;
            }

            // Point columns
            for &point_id in &p.point_ids {
                if point_to_param_idx.contains_key(&point_id) {
                    // This point is in current window - write 3 Jacobian entries
                    for i in 0..3 {
                        jacobian_data[jac_cursor] = p.sqrt_information[(prior_row, prior_col + i)];
                        jac_cursor += 1;
                    }
                }
                prior_col += 3;
            }
        }
    }
}