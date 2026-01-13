//! Bundle Adjustment for visual SLAM
//!
//! This module contains the bundle adjustment optimization for stereo visual SLAM.

use crate::camera::StereoCamera;
use crate::frame_graph::{FrameGraph, OptimizationState};
use crate::geometry::StereoObservation;
use crate::world_state::WorldState;
use nalgebra::DVector;
use odysseus_solver::math3d::Vec3;
use odysseus_solver::Jet;
use odysseus_solver::{
    build_slam_entries, sparse_solver::build_jacobian, SparseLevenbergMarquardt,
};
use std::collections::{HashMap, HashSet};

use super::marginalization::{compute_marginalization, SlamMarginalization};
use super::{
    apply_huber_loss, get_point_xyz, jet_constants, jet_variables,
    stereo_reprojection_residual_host_relative, with_jet_size,
};
use super::graph_visualization::{build_graph_info, OptimizationGraphInfo};

/// Type alias for backward compatibility
pub type MarginalizedPrior = SlamMarginalization;

/// Prior information for a single 3D point from GBA
///
/// Contains the point's optimized position and its information matrix (inverse covariance).
/// LBA uses this to treat GBA points as soft constraints rather than fixed.
#[derive(Debug, Clone)]
pub struct PointPrior {
    /// Point ID
    pub point_id: usize,
    /// Optimized position from GBA [x, y, z]
    pub position: [f64; 3],
    /// 3x3 information matrix (inverse covariance)
    /// Stored as [row0, row1, row2] = 9 elements in row-major order
    pub information: nalgebra::SMatrix<f64, 3, 3>,
}

/// Collection of point priors from GBA
#[derive(Debug, Clone, Default)]
pub struct PointPriors {
    /// Prior for each point, keyed by point_id
    pub priors: HashMap<usize, PointPrior>,
}

impl PointPriors {
    pub fn new() -> Self {
        Self { priors: HashMap::new() }
    }

    /// Get the prior for a specific point
    pub fn get(&self, point_id: usize) -> Option<&PointPrior> {
        self.priors.get(&point_id)
    }

    /// Insert a point prior
    pub fn insert(&mut self, prior: PointPrior) {
        self.priors.insert(prior.point_id, prior);
    }

    /// Number of point priors
    pub fn len(&self) -> usize {
        self.priors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.priors.is_empty()
    }
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
    /// Visual observation standard deviation (pixels)
    pub obs_std_dev: f64,
    /// Enable optimization graph visualization
    pub enable_graph_viz: bool,
}

impl Default for BundleAdjustmentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            tolerance: 1e-6,
            max_active_points: 600,
            huber_delta: 1.0,
            obs_std_dev: 0.5, // Standard observation noise (pixels)
            enable_graph_viz: false,
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
            obs_std_dev: 0.5,
            enable_graph_viz: false,
        }
    }

    /// Configuration for Global Bundle Adjustment (thorough, more iterations)
    pub fn gba() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-8,
            max_active_points: 800,
            huber_delta: 1.0,
            obs_std_dev: 0.5,
            enable_graph_viz: false,
        }
    }

    /// Enable optimization graph visualization
    pub fn with_graph_viz(mut self, enable: bool) -> Self {
        self.enable_graph_viz = enable;
        self
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
    /// Optimization graph visualization data
    pub graph_info: Option<OptimizationGraphInfo>,
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
    let optimized_point_ids: Vec<usize> = all_point_ids
        .iter()
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
            graph_info: None,
        };
    }

    // ========== 2. Build parameter mappings ==========

    // Pose parameters
    let mut pose_to_param_idx: HashMap<usize, usize> = HashMap::new();
    let mut offset = 0;

    let optimized_frame_indices: Vec<_> = frame_graph
        .states
        .iter()
        .enumerate()
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

    let sparsity_obs: Vec<_> = observations
        .iter()
        .enumerate()
        .map(|(i, obs)| {
            let pose_optimized = frame_graph.states[obs.camera_id].is_optimized();
            let point_optimized = point_to_param_idx.contains_key(&obs.point_id);

            let pose_start = pose_to_param_idx.get(&obs.camera_id).copied().unwrap_or(0);
            let point_start = point_to_param_idx.get(&obs.point_id).copied().unwrap_or(0);

            (
                i * 4,
                pose_start,
                point_start,
                pose_optimized,
                point_optimized,
            )
        })
        .collect();

    let n_params = n_poses * 6 + n_optimized_points * 3;
    let n_obs_residuals = n_observations * 4;
    let mut entries = build_slam_entries(&sparsity_obs);

    // Add prior residual entries (dense rows - each prior residual depends on all prior-constrained params)
    let n_prior_residuals = prior.map(|p| p.sqrt_information.nrows()).unwrap_or(0);
    if let Some(p) = prior {
        // Build list of parameter indices that this prior constrains
        let mut prior_param_indices: Vec<usize> = Vec::new();
        for &frame_id in &p.pose_ids {
            if let Some(&param_start) = pose_to_param_idx.get(&frame_id) {
                for i in 0..p.params_per_pose {
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
            graph_info: None,
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
            compute_ba_cost(
                params,
                residuals,
                jacobian_data,
                &observations,
                &pose_to_param_idx,
                &point_to_param_idx,
                world,
                stereo_camera,
                config.huber_delta,
                config.obs_std_dev,
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
        compute_ba_cost(
            &optimized_params,
            residuals.as_mut_slice(),
            jacobian.data_mut(),
            &observations,
            &pose_to_param_idx,
            &point_to_param_idx,
            world,
            stereo_camera,
            config.huber_delta,
            config.obs_std_dev,
            prior,
            n_obs_residuals,
        );

        // Build lookup for which poses observe each point
        let get_observing_poses = |point_id: usize| -> HashSet<usize> {
            observations
                .iter()
                .filter(|obs| obs.point_id == point_id)
                .map(|obs| obs.camera_id)
                .collect()
        };

        // Use shared marginalization implementation
        compute_marginalization(
            &jacobian,
            &optimized_params,
            6, // params_per_pose for BA
            n_params,
            &pose_to_param_idx,
            &point_to_param_idx,
            &marginalized_frame_indices,
            get_observing_poses,
        )
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
        world.frames[frame_idx]
            .pose
            .set_from_params(rot_delta, translation);
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

    // Extract graph info for visualization if enabled
    let graph_info = if config.enable_graph_viz {
        Some(build_graph_info(
            &observations,
            frame_graph,
            &pose_to_param_idx,
            &point_to_param_idx,
            fixed_point_ids,
        ))
    } else {
        None
    };

    BundleAdjustmentResult {
        solve_time_ms: solve_time.as_secs_f64() * 1000.0,
        n_observations,
        n_poses,
        n_points: n_optimized_points,
        new_prior,
        graph_info,
    }
}

/// Compute BA cost function with support for fixed points and marginalized prior
fn compute_ba_cost(
    params: &DVector<f64>,
    residuals: &mut [f64],
    jacobian_data: &mut [f64],
    observations: &[StereoObservation],
    pose_to_param_idx: &HashMap<usize, usize>,
    point_to_param_idx: &HashMap<usize, usize>,
    world: &WorldState,
    stereo_camera: &StereoCamera<f64>,
    huber_delta: f64,
    obs_std_dev: f64,
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
                    Jet::constant(stereo_camera.left.fx),
                    Jet::constant(stereo_camera.left.fy),
                    Jet::constant(stereo_camera.left.cx),
                    Jet::constant(stereo_camera.left.cy),
                ),
                Jet::constant(stereo_camera.baseline),
            );

            // Compute residuals using host-relative parameterization
            let (r1, r2, r3, r4) = stereo_reprojection_residual_host_relative(
                rotation_host,
                &pose_params,
                &world_point,
                &camera_jet,
                Jet::constant(obs.left_u),
                Jet::constant(obs.left_v),
                Jet::constant(obs.right_u),
                Jet::constant(obs.right_v),
            );

            // Write residuals and jacobian with Huber loss and observation weighting
            // Jacobian is written linearly - pose entries first, then point entries
            let obs_weight = 1.0 / (obs_std_dev * obs_std_dev);
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
                        // Apply observation weighting
                        r_val *= obs_weight;
                        for d in &mut combined {
                            *d *= obs_weight;
                        }
                        // Write pose params (6), then point params (3)
                        jacobian_data[jac_cursor..jac_cursor + 9].copy_from_slice(&combined);
                        jac_cursor += 9;
                    }
                    (true, false) => {
                        let mut pose_jac = [
                            r.derivs[0], r.derivs[1], r.derivs[2], r.derivs[3], r.derivs[4], r.derivs[5],
                        ];
                        apply_huber_loss(huber_delta, &mut r_val, &mut pose_jac);
                        r_val *= obs_weight;
                        for d in &mut pose_jac {
                            *d *= obs_weight;
                        }
                        jacobian_data[jac_cursor..jac_cursor + 6].copy_from_slice(&pose_jac);
                        jac_cursor += 6;
                    }
                    (false, true) => {
                        let mut point_jac = [r.derivs[0], r.derivs[1], r.derivs[2]];
                        apply_huber_loss(huber_delta, &mut r_val, &mut point_jac);
                        r_val *= obs_weight;
                        for d in &mut point_jac {
                            *d *= obs_weight;
                        }
                        jacobian_data[jac_cursor..jac_cursor + 3].copy_from_slice(&point_jac);
                        jac_cursor += 3;
                    }
                    (false, false) => {
                        // No jacobian entries, just apply Huber to residual
                        let mut dummy = [0.0_f64; 0];
                        apply_huber_loss(huber_delta, &mut r_val, &mut dummy);
                        r_val *= obs_weight;
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

        // Map prior's pose params to current window
        for &frame_id in &p.pose_ids {
            if let Some(&param_start) = pose_to_param_idx.get(&frame_id) {
                for i in 0..p.params_per_pose {
                    current_prior_params[prior_param_idx] = params[param_start + i];
                    prior_param_idx += 1;
                }
            } else {
                // Pose not in current window - use linearization point values
                for _ in 0..p.params_per_pose {
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
        // Columns are ordered: poses (params_per_pose each), then points (3 params each)
        let n_prior_residuals = p.sqrt_information.nrows();

        for prior_row in 0..n_prior_residuals {
            let mut prior_col = 0;

            // Pose columns
            for &frame_id in &p.pose_ids {
                if pose_to_param_idx.contains_key(&frame_id) {
                    // This pose is in current window - write Jacobian entries
                    for i in 0..p.params_per_pose {
                        jacobian_data[jac_cursor] = p.sqrt_information[(prior_row, prior_col + i)];
                        jac_cursor += 1;
                    }
                }
                prior_col += p.params_per_pose;
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

// ========== GBA with Relative Pose Constraints ==========

use crate::imu::{ImuFrameState, VisuallyInformedPreintegration, imu_preintegration_residual, bias_residual};
use nalgebra::Vector3;

/// Result of GBA with IMU constraints
pub struct GbaWithImuResult {
    /// Time spent in the solver (milliseconds)
    pub solve_time_ms: f64,
    /// Number of visual observations used
    pub n_observations: usize,
    /// Number of optimized poses
    pub n_poses: usize,
    /// Number of optimized points
    pub n_points: usize,
    /// Number of relative pose constraints used
    pub n_constraints: usize,
    /// Point priors (positions + information matrices) for LBA to use
    pub point_priors: PointPriors,
}

/// Run GBA with visually-informed IMU preintegrations from VIO
///
/// This extends bundle adjustment to use 15 DOF per frame (pose + velocity + biases)
/// and adds IMU preintegration residuals using pose-invariant measurements.
///
/// Unlike the old RelativePoseConstraint approach, this maintains IMU dynamics
/// when GBA adjusts poses based on visual residuals.
///
/// # Arguments
/// * `stereo_camera` - Stereo camera model
/// * `frame_graph` - Frame graph specifying which frames are optimized/fixed
/// * `world` - World state (poses and points) - modified in place
/// * `frame_observations` - Observations per frame
/// * `imu_states` - IMU states (velocity, biases) per frame - modified in place
/// * `preintegrations` - Visually-informed preintegrations (one per frame, None for first)
/// * `gravity` - Gravity vector in world frame
/// * `config` - Solver configuration
pub fn run_gba_with_imu_constraints(
    stereo_camera: &StereoCamera<f64>,
    frame_graph: &FrameGraph,
    world: &mut WorldState,
    frame_observations: &[Vec<StereoObservation>],
    imu_states: &mut [Option<ImuFrameState>],
    preintegrations: &[Option<VisuallyInformedPreintegration>],
    gravity: [f64; 3],
    config: &BundleAdjustmentConfig,
) -> GbaWithImuResult {
    // ========== 1. Collect observations and select active points ==========

    let mut all_obs: Vec<StereoObservation> = Vec::new();
    let mut score_map: HashMap<usize, f64> = HashMap::new();

    for (frame_idx, frame_state) in frame_graph.states.iter().enumerate() {
        if !frame_state.is_active_in_ba() {
            continue;
        }
        if frame_idx >= frame_observations.len() {
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
    let mut scored_points: Vec<_> = score_map.into_iter().collect();
    scored_points.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored_points.truncate(config.max_active_points);
    let all_point_ids: Vec<usize> = scored_points.into_iter().map(|(id, _)| id).collect();

    let all_points_set: HashSet<_> = all_point_ids.iter().copied().collect();
    let observations: Vec<_> = all_obs
        .into_iter()
        .filter(|obs| all_points_set.contains(&obs.point_id))
        .collect();

    // ========== 2. Build parameter mappings ==========
    // 15 DOF per frame: rotation_delta (3), translation (3), velocity (3), gyro_bias (3), accel_bias (3)
    // We now optimize biases in GBA to maintain IMU dynamics consistency

    let mut pose_to_param_idx: HashMap<usize, usize> = HashMap::new();
    let mut offset = 0;

    let optimized_frame_indices: Vec<_> = frame_graph
        .states
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_optimized())
        .map(|(idx, _)| idx)
        .collect();

    for frame_idx in &optimized_frame_indices {
        pose_to_param_idx.insert(*frame_idx, offset);
        offset += 15; // 15 DOF per frame
    }
    let n_poses = pose_to_param_idx.len();

    // Point parameters
    let mut point_to_param_idx: HashMap<usize, usize> = HashMap::new();
    for &point_id in &all_point_ids {
        point_to_param_idx.insert(point_id, offset);
        offset += 3;
    }
    let n_points = all_point_ids.len();
    let n_observations = observations.len();

    // Extract preintegrations from visually-informed preintegrations (where both frames are optimized)
    let valid_preintegrations: Vec<&VisuallyInformedPreintegration> = preintegrations
        .iter()
        .filter_map(|p| p.as_ref())
        .filter(|p| {
            pose_to_param_idx.contains_key(&p.frame_i)
                && pose_to_param_idx.contains_key(&p.frame_j)
        })
        .collect();
    let n_preintegrations = valid_preintegrations.len();

    // ========== 3. Build sparsity entries ==========

    let n_params = n_poses * 15 + n_points * 3;
    let n_obs_residuals = n_observations * 4;
    let n_imu_residuals = n_preintegrations * 9;  // rotation(3), velocity(3), position(3)
    let n_bias_residuals = n_preintegrations * 6;  // gyro_bias(3), accel_bias(3)
    let n_residuals = n_obs_residuals + n_imu_residuals + n_bias_residuals;

    if n_params == 0 || n_residuals == 0 {
        return GbaWithImuResult {
            solve_time_ms: 0.0,
            n_observations: 0,
            n_poses: 0,
            n_points: 0,
            n_constraints: 0,
            point_priors: PointPriors::new(),
        };
    }

    let mut entries: Vec<(usize, usize)> = Vec::new();

    // Visual observation entries (4 residuals each, depend on 6 pose params + 3 point params)
    for (obs_idx, obs) in observations.iter().enumerate() {
        let res_base = obs_idx * 4;
        if let Some(&pose_start) = pose_to_param_idx.get(&obs.camera_id) {
            // Pose params (only first 6: rotation, translation)
            for r in 0..4 {
                for p in 0..6 {
                    entries.push((res_base + r, pose_start + p));
                }
            }
        }
        if let Some(&point_start) = point_to_param_idx.get(&obs.point_id) {
            for r in 0..4 {
                for p in 0..3 {
                    entries.push((res_base + r, point_start + p));
                }
            }
        }
    }

    // IMU preintegration residual entries (9 residuals each, depend on 15 params of each frame)
    for (p_idx, preint) in valid_preintegrations.iter().enumerate() {
        let res_base = n_obs_residuals + p_idx * 9;
        let p_i = pose_to_param_idx[&preint.frame_i];
        let p_j = pose_to_param_idx[&preint.frame_j];

        for r in 0..9 {
            for p in 0..15 {
                entries.push((res_base + r, p_i + p));
                entries.push((res_base + r, p_j + p));
            }
        }
    }

    // Bias residual entries (6 residuals each, depend on bias params of both frames)
    let bias_res_base = n_obs_residuals + n_imu_residuals;
    for (p_idx, preint) in valid_preintegrations.iter().enumerate() {
        let res_base = bias_res_base + p_idx * 6;
        let p_i = pose_to_param_idx[&preint.frame_i];
        let p_j = pose_to_param_idx[&preint.frame_j];

        for r in 0..6 {
            // Biases are params 9-14
            for p in 9..15 {
                entries.push((res_base + r, p_i + p));
                entries.push((res_base + r, p_j + p));
            }
        }
    }

    entries.sort();
    entries.dedup();

    // ========== 4. Pack parameters ==========

    let mut initial_params = DVector::<f64>::zeros(n_params);

    for (&frame_idx, &param_idx) in &pose_to_param_idx {
        let pose = &world.frames[frame_idx].pose;
        initial_params[param_idx + 0] = pose.rotation.x;
        initial_params[param_idx + 1] = pose.rotation.y;
        initial_params[param_idx + 2] = pose.rotation.z;
        initial_params[param_idx + 3] = pose.translation.x;
        initial_params[param_idx + 4] = pose.translation.y;
        initial_params[param_idx + 5] = pose.translation.z;

        // Velocity and biases from ImuFrameState (or zero if not available)
        if let Some(Some(imu_state)) = imu_states.get(frame_idx) {
            initial_params[param_idx + 6] = imu_state.velocity.x;
            initial_params[param_idx + 7] = imu_state.velocity.y;
            initial_params[param_idx + 8] = imu_state.velocity.z;
            initial_params[param_idx + 9] = imu_state.gyro_bias.x;
            initial_params[param_idx + 10] = imu_state.gyro_bias.y;
            initial_params[param_idx + 11] = imu_state.gyro_bias.z;
            initial_params[param_idx + 12] = imu_state.accel_bias.x;
            initial_params[param_idx + 13] = imu_state.accel_bias.y;
            initial_params[param_idx + 14] = imu_state.accel_bias.z;
        }
    }

    for &point_id in &all_point_ids {
        let param_idx = point_to_param_idx[&point_id];
        let point = world.get_point(point_id).unwrap();
        initial_params[param_idx + 0] = point.x;
        initial_params[param_idx + 1] = point.y;
        initial_params[param_idx + 2] = point.z;
    }

    // ========== 5. Solve ==========

    let mut solver = SparseLevenbergMarquardt::<f64>::new(n_residuals, n_params, &entries)
        .with_tolerance(config.tolerance)
        .with_max_iterations(config.max_iterations)
        .with_verbose(false);

    let start = std::time::Instant::now();

    let optimized_params = solver.solve(
        initial_params,
        |params, residuals, jacobian_data| {
            compute_gba_imu_cost(
                params,
                residuals,
                jacobian_data,
                &observations,
                &valid_preintegrations,
                &pose_to_param_idx,
                &point_to_param_idx,
                world,
                stereo_camera,
                config.huber_delta,
                config.obs_std_dev,
                gravity,
                n_obs_residuals,
            );
        },
        |_iter, _result, _params| {},
    );

    let solve_time = start.elapsed();

    // ========== 6. Unpack results ==========

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
        world.frames[frame_idx]
            .pose
            .set_from_params(rot_delta, translation);

        // Update velocity and biases in imu_states
        if let Some(imu_state) = imu_states.get_mut(frame_idx).and_then(|s| s.as_mut()) {
            imu_state.velocity = Vector3::new(
                optimized_params[param_idx + 6],
                optimized_params[param_idx + 7],
                optimized_params[param_idx + 8],
            );
            imu_state.gyro_bias = Vector3::new(
                optimized_params[param_idx + 9],
                optimized_params[param_idx + 10],
                optimized_params[param_idx + 11],
            );
            imu_state.accel_bias = Vector3::new(
                optimized_params[param_idx + 12],
                optimized_params[param_idx + 13],
                optimized_params[param_idx + 14],
            );
        }
    }

    for &point_id in &all_point_ids {
        let param_idx = point_to_param_idx[&point_id];
        let new_position = Vec3::new(
            optimized_params[param_idx + 0],
            optimized_params[param_idx + 1],
            optimized_params[param_idx + 2],
        );
        world.update_point(point_id, new_position);
    }

    // ========== 7. Extract point information matrices ==========
    // Recompute Jacobian at optimized solution and extract point Hessian blocks
    let point_priors = extract_point_priors(
        &optimized_params,
        &entries,
        n_residuals,
        n_params,
        &observations,
        &valid_preintegrations,
        &pose_to_param_idx,
        &point_to_param_idx,
        &all_point_ids,
        world,
        stereo_camera,
        config.huber_delta,
        config.obs_std_dev,
        gravity,
        n_obs_residuals,
    );

    GbaWithImuResult {
        solve_time_ms: solve_time.as_secs_f64() * 1000.0,
        n_observations,
        n_poses,
        n_points,
        n_constraints: n_preintegrations,  // Now using preintegrations instead of constraints
        point_priors,
    }
}

/// Compute GBA cost function with visual and IMU preintegration residuals
fn compute_gba_imu_cost(
    params: &DVector<f64>,
    residuals: &mut [f64],
    jacobian_data: &mut [f64],
    observations: &[StereoObservation],
    preintegrations: &[&VisuallyInformedPreintegration],
    pose_to_param_idx: &HashMap<usize, usize>,
    point_to_param_idx: &HashMap<usize, usize>,
    world: &WorldState,
    stereo_camera: &StereoCamera<f64>,
    huber_delta: f64,
    obs_std_dev: f64,
    gravity: [f64; 3],
    n_obs_residuals: usize,
) {
    let mut jac_cursor = 0;

    // ========== Visual observation residuals ==========
    for (obs_idx, obs) in observations.iter().enumerate() {
        let res_offset = obs_idx * 4;
        let pose_optimized = pose_to_param_idx.contains_key(&obs.camera_id);
        let point_optimized = point_to_param_idx.contains_key(&obs.point_id);

        with_jet_size!(pose_optimized, point_optimized, { 6, 3 }, |JetN, POSE_ACTIVE, POINT_ACTIVE| {
            let point_deriv_offset: usize = if POSE_ACTIVE { 6 } else { 0 };

            let rotation_host = &world.frames[obs.camera_id].pose.rotation_host;

            // Pose params: only first 6 (rotation_delta, translation) affect visual residuals
            let pose_params: [JetN; 6] = if POSE_ACTIVE {
                let idx = pose_to_param_idx[&obs.camera_id];
                jet_variables(params, idx, 0)
            } else {
                let pose = &world.frames[obs.camera_id].pose;
                jet_constants(&[
                    pose.rotation.x, pose.rotation.y, pose.rotation.z,
                    pose.translation.x, pose.translation.y, pose.translation.z,
                ])
            };

            let world_point: [JetN; 3] = if POINT_ACTIVE {
                let idx = point_to_param_idx[&obs.point_id];
                jet_variables(params, idx, point_deriv_offset)
            } else {
                let xyz = get_point_xyz(obs.point_id, params, point_to_param_idx, world);
                jet_constants(&xyz)
            };

            let camera_jet = StereoCamera::new(
                crate::camera::PinholeCamera::new(
                    Jet::constant(stereo_camera.left.fx),
                    Jet::constant(stereo_camera.left.fy),
                    Jet::constant(stereo_camera.left.cx),
                    Jet::constant(stereo_camera.left.cy),
                ),
                Jet::constant(stereo_camera.baseline),
            );

            let (err_lu, err_lv, err_ru, err_rv) = stereo_reprojection_residual_host_relative(
                rotation_host,
                &pose_params,
                &world_point,
                &camera_jet,
                Jet::constant(obs.left_u),
                Jet::constant(obs.left_v),
                Jet::constant(obs.right_u),
                Jet::constant(obs.right_v),
            );

            let errs = [err_lu, err_lv, err_ru, err_rv];
            let obs_weight = 1.0 / (obs_std_dev * obs_std_dev);

            for (i, err) in errs.iter().enumerate() {
                let mut r_val = err.value;
                let n_active_params = (if POSE_ACTIVE { 6 } else { 0 }) + (if POINT_ACTIVE { 3 } else { 0 });

                if n_active_params > 0 {
                    let mut local_jac: Vec<f64> = err.derivs[..n_active_params].to_vec();
                    apply_huber_loss(huber_delta, &mut r_val, &mut local_jac);
                    // Apply observation weighting
                    r_val *= obs_weight;
                    for j in &mut local_jac {
                        *j *= obs_weight;
                    }
                    for j in local_jac {
                        jacobian_data[jac_cursor] = j;
                        jac_cursor += 1;
                    }
                } else {
                    let mut dummy = [0.0_f64; 0];
                    apply_huber_loss(huber_delta, &mut r_val, &mut dummy);
                    r_val *= obs_weight;
                }

                residuals[res_offset + i] = r_val;
            }
        });
    }

    // ========== IMU preintegration residuals ==========
    let n_imu_factors = preintegrations.len();
    for (p_idx, preint_data) in preintegrations.iter().enumerate() {
        let res_offset = n_obs_residuals + p_idx * 9;
        
        let p_i = pose_to_param_idx[&preint_data.frame_i];
        let p_j = pose_to_param_idx[&preint_data.frame_j];
        
        let rotation_host_i = &world.frames[preint_data.frame_i].pose.rotation_host;
        let rotation_host_j = &world.frames[preint_data.frame_j].pose.rotation_host;
        
        // Use Jet<f64, 30> for derivatives w.r.t. both frames (15 params each)
        type Jet30 = Jet<f64, 30>;
        
        // Pack params for frame i (15 DOF)
        let pose_params_i: [Jet30; 15] = std::array::from_fn(|k| {
            Jet30::variable(params[p_i + k], k)
        });
        
        // Pack params for frame j (15 DOF, derivatives at offset 15)
        let pose_params_j: [Jet30; 15] = std::array::from_fn(|k| {
            Jet30::variable(params[p_j + k], 15 + k)
        });
        
        let gravity_jet: [Jet30; 3] = [
            Jet30::constant(gravity[0]),
            Jet30::constant(gravity[1]),
            Jet30::constant(gravity[2]),
        ];
        
        // Compute IMU residual using the pose-invariant preintegration
        let imu_residual = imu_preintegration_residual(
            rotation_host_i,
            &pose_params_i,
            rotation_host_j,
            &pose_params_j,
            &preint_data.preintegration,
            &gravity_jet,
        );
        
        // Extract residuals and Jacobians
        for r in 0..9 {
            residuals[res_offset + r] = imu_residual[r].value;
            
            // Write Jacobian entries (frame_i params then frame_j params)
            for d in 0..30 {
                jacobian_data[jac_cursor] = imu_residual[r].derivs[d];
                jac_cursor += 1;
            }
        }
    }
    
    // ========== Bias residuals ==========
    let bias_res_base = n_obs_residuals + n_imu_factors * 9;
    for (p_idx, preint_data) in preintegrations.iter().enumerate() {
        let res_offset = bias_res_base + p_idx * 6;
        
        let p_i = pose_to_param_idx[&preint_data.frame_i];
        let p_j = pose_to_param_idx[&preint_data.frame_j];
        
        // Extract bias params
        let bg_i = [params[p_i + 9], params[p_i + 10], params[p_i + 11]];
        let ba_i = [params[p_i + 12], params[p_i + 13], params[p_i + 14]];
        let bg_j = [params[p_j + 9], params[p_j + 10], params[p_j + 11]];
        let ba_j = [params[p_j + 12], params[p_j + 13], params[p_j + 14]];
        
        let dt = preint_data.preintegration.delta_time;
        
        // Compute bias residuals with random walk model
        // We use the posterior covariance from VIO to scale these appropriately
        // For now, use standard random walk noise parameters
        const GYRO_SIGMA: f64 = 0.001;  // rad/s/sqrt(Hz)
        const ACCEL_SIGMA: f64 = 0.01;   // m/s^2/sqrt(Hz)
        
        let bias_res = bias_residual::<f64>(
            &bg_i,
            &ba_i,
            &bg_j,
            &ba_j,
            dt,
            GYRO_SIGMA,
            ACCEL_SIGMA,
        );
        
        // Compute weights
        let dt_sqrt = dt.sqrt();
        let gw = 1.0 / (GYRO_SIGMA * dt_sqrt);
        let aw = 1.0 / (ACCEL_SIGMA * dt_sqrt);
        
        for r_idx in 0..6 {
            residuals[res_offset + r_idx] = bias_res[r_idx];
            let w = if r_idx < 3 { gw } else { aw };
            
            // Jacobian for frame i biases (params 9-14)
            for p in 0..6 {
                jacobian_data[jac_cursor] = if p == r_idx { -w } else { 0.0 };
                jac_cursor += 1;
            }
            
            // Jacobian for frame j biases (params 9-14)
            for p in 0..6 {
                jacobian_data[jac_cursor] = if p == r_idx { w } else { 0.0 };
                jac_cursor += 1;
            }
        }
    }
}

/// Extract point priors (positions + information matrices) from the GBA Hessian
///
/// After GBA optimization, this computes J^T * J at the solution and extracts
/// the 3x3 diagonal blocks for each point as information matrices.
#[allow(clippy::too_many_arguments)]
fn extract_point_priors(
    optimized_params: &DVector<f64>,
    entries: &[(usize, usize)],
    n_residuals: usize,
    n_params: usize,
    observations: &[StereoObservation],
    preintegrations: &[&VisuallyInformedPreintegration],
    pose_to_param_idx: &HashMap<usize, usize>,
    point_to_param_idx: &HashMap<usize, usize>,
    all_point_ids: &[usize],
    world: &WorldState,
    stereo_camera: &StereoCamera<f64>,
    huber_delta: f64,
    obs_std_dev: f64,
    gravity: [f64; 3],
    n_obs_residuals: usize,
) -> PointPriors {
    // Build Jacobian at optimized solution
    let mut jacobian = build_jacobian::<f64>(entries, n_residuals, n_params);
    let mut residuals = DVector::zeros(n_residuals);

    compute_gba_imu_cost(
        optimized_params,
        residuals.as_mut_slice(),
        jacobian.data_mut(),
        observations,
        preintegrations,
        pose_to_param_idx,
        point_to_param_idx,
        world,
        stereo_camera,
        huber_delta,
        obs_std_dev,
        gravity,
        n_obs_residuals,
    );

    // Compute Hessian: H = J^T * J
    let jt: sprs::CsMat<f64> = jacobian.transpose_view().to_csr();
    let sprs_hessian = &jt * &jacobian;

    // Extract point priors
    let mut point_priors = PointPriors::new();

    for &point_id in all_point_ids {
        let param_idx = point_to_param_idx[&point_id];

        // Extract position from optimized params
        let position = [
            optimized_params[param_idx],
            optimized_params[param_idx + 1],
            optimized_params[param_idx + 2],
        ];

        // Extract 3x3 diagonal block from Hessian
        // The Hessian is sparse, so we need to look up each element
        let mut information = nalgebra::SMatrix::<f64, 3, 3>::zeros();
        for i in 0..3 {
            for j in 0..3 {
                let row = param_idx + i;
                let col = param_idx + j;
                // Look up value in sparse Hessian
                if let Some(&val) = sprs_hessian.get(row, col) {
                    information[(i, j)] = val;
                }
            }
        }

        // Scale down the information matrix to loosen the constraint
        // Similar to COVARIANCE_SCALE_FACTOR for RelativePoseConstraints,
        // we need to allow VIO/LBA room to move points if visual evidence
        // contradicts GBA's estimates.
        const INFORMATION_SCALE_FACTOR: f64 = 0.5;
        information *= INFORMATION_SCALE_FACTOR;

        // Add small regularization to ensure invertibility
        const MIN_INFO_DIAG: f64 = 1e-8; // Reduced from 1e-6 since we scaled down
        for i in 0..3 {
            if information[(i, i)] < MIN_INFO_DIAG {
                information[(i, i)] = MIN_INFO_DIAG;
            }
        }

        point_priors.insert(PointPrior {
            point_id,
            position,
            information,
        });
    }

    point_priors
}
