use crate::camera::StereoCamera;
use crate::frame_graph::{FrameGraph, OptimizationState};
use crate::geometry::StereoObservation;
use crate::imu::preintegration::{Matrix9, PreintegratedImu};
use crate::imu::residuals::{bias_residual, imu_preintegration_residual};
use crate::imu::types::ImuFrameState;
use crate::optimization::select_active_points;
use crate::world_state::WorldState;
use nalgebra::{DVector, Vector3};
use odysseus_solver::math3d::Vec3;
use odysseus_solver::{Jet, SparseLevenbergMarquardt};
use std::collections::{HashMap, HashSet};

use super::apply_huber_loss;
use super::graph_visualization::{build_graph_info, OptimizationGraphInfo};

/// Configuration for VIO bundle adjustment
#[derive(Debug, Clone)]
pub struct VioConfig {
    /// Maximum solver iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Huber loss threshold (pixels)
    pub huber_delta: f64,
    /// Maximum number of map points to include
    pub max_active_points: usize,
    /// Gyroscope random walk noise (rad/s/sqrt(Hz))
    pub gyro_sigma: f64,
    /// Accelerometer random walk noise (m/s^2/sqrt(Hz))
    pub accel_sigma: f64,
    /// Visual observation standard deviation (pixels)
    pub obs_std_dev: f64,
    /// Enable optimization graph visualization
    pub enable_graph_viz: bool,
}

impl Default for VioConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 1e-6,
            huber_delta: 1.0,
            max_active_points: 600,
            gyro_sigma: 0.001,
            accel_sigma: 0.01,
            obs_std_dev: 0.5, // Standard observation noise for VIO (pixels)
            enable_graph_viz: false,
        }
    }
}

impl VioConfig {
    /// Enable optimization graph visualization
    pub fn with_graph_viz(mut self, enable: bool) -> Self {
        self.enable_graph_viz = enable;
        self
    }
}

pub struct VioResult {
    pub final_error: f64,
    pub iterations: usize,
    pub converged: bool,
    pub solve_time_ms: f64,
    /// Optimization graph visualization data
    pub graph_info: Option<OptimizationGraphInfo>,
    /// New marginalized prior (if frames were marginalized)
    pub new_prior: Option<super::VioMarginalizedPrior>,
}

/// Run simple VIO bundle adjustment (no GBA, no inter-keyframe priors)
///
/// This is a simplified version for basic VIO without the complexity of
/// inter-keyframe priors used when integrating with GBA.
pub fn run_simple_vio_bundle_adjustment(
    stereo_camera: &StereoCamera<f64>,
    frame_graph: &FrameGraph,
    world: &mut WorldState,
    frame_observations: &[Vec<StereoObservation>],
    imu_states: &mut Vec<ImuFrameState>,
    preintegrations: &[PreintegratedImu],
    gravity: [f64; 3],
    prior: Option<&super::VioMarginalizedPrior>,
    config: &VioConfig,
) -> VioResult {
    let n_frames = world.frames.len();
    assert_eq!(imu_states.len(), n_frames);
    // Preintegrations should be between consecutive frames
    assert_eq!(preintegrations.len(), n_frames - 1);

    // ========== 1. Collect observations and select active points ==========
    let active_frame_indices: Vec<_> = frame_graph
        .states
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_optimized())
        .map(|(idx, _)| idx)
        .collect();

    // Collect all observations from active frames
    let all_obs: Vec<_> = active_frame_indices
        .iter()
        .flat_map(|&frame_idx| {
            frame_observations[frame_idx]
                .iter()
                .filter(|obs| world.get_point(obs.point_id).is_some())
                .copied()
        })
        .collect();

    let (optimized_point_ids, all_point_ids) = select_active_points(
        &all_obs,
        |obs| obs.point_id,
        |obs| obs.camera_id,
        |frame_id| {
            frame_graph
                .states
                .get(frame_id)
                .map(|s| s.is_optimized())
                .unwrap_or(false)
        },
        &HashSet::new(),
        config.max_active_points,
    );

    let active_points_set: HashSet<_> = all_point_ids.into_iter().collect();

    // ========== 2. Build parameter mappings ==========

    let mut pose_to_param_idx = HashMap::new();
    let mut point_to_param_idx = HashMap::new();
    let mut offset = 0;

    for &idx in &active_frame_indices {
        pose_to_param_idx.insert(idx, offset);
        offset += 15; // [rot_delta(3), trans(3), vel(3), bg(3), ba(3)]
    }

    // Only optimized points get parameter indices (fixed points use world state directly)
    for &id in &optimized_point_ids {
        point_to_param_idx.insert(id, offset);
        offset += 3;
    }

    let n_params = offset;
    if n_params == 0 {
        return VioResult {
            final_error: 0.0,
            iterations: 0,
            converged: true,
            solve_time_ms: 0.0,
            graph_info: None,
            new_prior: None,
        };
    }

    // ========== 3. Build sparsity entries ==========
    let mut visual_obs_filtered = Vec::new();
    let mut entries = Vec::new();

    // Visual labels and entries
    for frame_idx in 0..n_frames {
        let pose_opt = pose_to_param_idx.get(&frame_idx);
        for obs in &frame_observations[frame_idx] {
            // Skip if point doesn't exist or isn't in our active set
            if world.get_point(obs.point_id).is_none() || !active_points_set.contains(&obs.point_id)
            {
                continue;
            }
            let point_opt = point_to_param_idx.get(&obs.point_id);
            if pose_opt.is_some() || point_opt.is_some() {
                let res_idx = visual_obs_filtered.len() * 4;
                visual_obs_filtered.push(*obs);

                if let Some(&p_idx) = pose_opt {
                    for i in 0..6 {
                        for r in 0..4 {
                            entries.push((res_idx + r, p_idx + i));
                        }
                    }
                }
                if let Some(&pt_idx) = point_opt {
                    for i in 0..3 {
                        for r in 0..4 {
                            entries.push((res_idx + r, pt_idx + i));
                        }
                    }
                }
            }
        }
    }

    let n_imu_factors = n_frames - 1;
    let n_visual_residuals = visual_obs_filtered.len() * 4;
    let n_imu_residuals = n_imu_factors * 9;
    let n_bias_residuals = n_imu_factors * 6;
    
    let n_prior_residuals = prior.map(|p| p.sqrt_information.nrows()).unwrap_or(0);
    
    let n_residuals = n_visual_residuals + n_imu_residuals + n_bias_residuals + n_prior_residuals;

    // IMU and Bias entries
    let imu_res_start = n_visual_residuals;
    for i in 0..n_imu_factors {
        let res_base = imu_res_start + i * 9;
        let bias_res_base = imu_res_start + n_imu_residuals + i * 6;

        let opt_i = pose_to_param_idx.get(&i);
        let opt_j = pose_to_param_idx.get(&(i + 1));

        if let Some(&p_i) = opt_i {
            // IMU residual depends on all 15 params of frame i
            for p in 0..15 {
                for r in 0..9 {
                    entries.push((res_base + r, p_i + p));
                }
            }
            // Bias residual depends on 6 bias params of frame i (indices 9-14)
            for p in 9..15 {
                for r in 0..6 {
                    entries.push((bias_res_base + r, p_i + p));
                }
            }
        }
        if let Some(&p_j) = opt_j {
            for p in 0..15 {
                for r in 0..9 {
                    entries.push((res_base + r, p_j + p));
                }
            }
            for p in 9..15 {
                for r in 0..6 {
                    entries.push((bias_res_base + r, p_j + p));
                }
            }
        }
    }

    // Prior entries
    if let Some(p) = prior {
        let prior_res_start = n_visual_residuals + n_imu_residuals + n_bias_residuals;
        let entries_for_prior = p.sparsity_entries(
            prior_res_start,
            &pose_to_param_idx,
            &point_to_param_idx,
        );
        entries.extend(entries_for_prior);
    }

    entries.sort();
    entries.dedup();

    // ========== 4. Pack parameters ==========

    let mut initial_params = DVector::zeros(n_params);
    for (&idx, &p_idx) in &pose_to_param_idx {
        let pose = &world.frames[idx].pose;
        let imu = &imu_states[idx];
        initial_params[p_idx + 0] = pose.rotation.x;
        initial_params[p_idx + 1] = pose.rotation.y;
        initial_params[p_idx + 2] = pose.rotation.z;
        initial_params[p_idx + 3] = pose.translation.x;
        initial_params[p_idx + 4] = pose.translation.y;
        initial_params[p_idx + 5] = pose.translation.z;
        initial_params[p_idx + 6] = imu.velocity.x;
        initial_params[p_idx + 7] = imu.velocity.y;
        initial_params[p_idx + 8] = imu.velocity.z;
        initial_params[p_idx + 9] = imu.gyro_bias.x;
        initial_params[p_idx + 10] = imu.gyro_bias.y;
        initial_params[p_idx + 11] = imu.gyro_bias.z;
        initial_params[p_idx + 12] = imu.accel_bias.x;
        initial_params[p_idx + 13] = imu.accel_bias.y;
        initial_params[p_idx + 14] = imu.accel_bias.z;
    }
    for &id in &optimized_point_ids {
        let p_idx = point_to_param_idx[&id];
        let pt_info = world.get_point_info(id).unwrap();
        initial_params[p_idx + 0] = pt_info.direction.0; // direction_u
        initial_params[p_idx + 1] = pt_info.direction.1; // direction_v
        initial_params[p_idx + 2] = pt_info.inv_depth;   // inverse depth
    }

    // ========== 5. Solve ==========

    let start_time = std::time::Instant::now();
    let mut solver = SparseLevenbergMarquardt::<f64>::new(n_residuals, n_params, &entries)
        .with_tolerance(config.tolerance)
        .with_max_iterations(config.max_iterations);

    let mut iteration_count = 0;
    let mut final_error = 0.0;
    let mut converged = false;

    // Convert single prior to slice for cost function
    let prior_slice: &[super::VioMarginalizedPrior] = if let Some(p) = prior {
        std::slice::from_ref(p)
    } else {
        &[]
    };

    let optimized = solver.solve(
        initial_params,
        |params, residuals, jacobian_data| {
            compute_simple_vio_cost(
                params,
                residuals,
                jacobian_data,
                world,
                &visual_obs_filtered,
                imu_states,
                preintegrations,
                &pose_to_param_idx,
                &point_to_param_idx,
                stereo_camera,
                &gravity,
                config,
                prior_slice,
                n_visual_residuals,
                n_imu_residuals,
                n_bias_residuals,
            );
        },
        |iter, res, _| {
            iteration_count = iter + 1;
            final_error = res.error;
            converged = res.converged;
        },
    );
    let solve_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    // ========== 6. Unpack results ==========

    for (&idx, &p_idx) in &pose_to_param_idx {
        let rot_delta = Vec3::new(
            optimized[p_idx + 0],
            optimized[p_idx + 1],
            optimized[p_idx + 2],
        );
        let trans = Vec3::new(
            optimized[p_idx + 3],
            optimized[p_idx + 4],
            optimized[p_idx + 5],
        );
        world.frames[idx].pose.set_from_params(rot_delta, trans);

        imu_states[idx].velocity = Vector3::new(
            optimized[p_idx + 6],
            optimized[p_idx + 7],
            optimized[p_idx + 8],
        );
        imu_states[idx].gyro_bias = Vector3::new(
            optimized[p_idx + 9],
            optimized[p_idx + 10],
            optimized[p_idx + 11],
        );
        imu_states[idx].accel_bias = Vector3::new(
            optimized[p_idx + 12],
            optimized[p_idx + 13],
            optimized[p_idx + 14],
        );
    }
    for &id in &optimized_point_ids {
        let p_idx = point_to_param_idx[&id];
        let direction = (
            optimized[p_idx + 0],  // direction_u
            optimized[p_idx + 1],  // direction_v
        );
        let inv_depth = optimized[p_idx + 2];
        world.update_point_params(id, direction, inv_depth);
    }

    // Extract graph info for visualization if enabled
    let graph_info = if config.enable_graph_viz {
        Some(build_graph_info(
            &visual_obs_filtered,
            frame_graph,
            &pose_to_param_idx,
            &point_to_param_idx,
            &HashSet::new(), // No fixed points in simple VIO
        ))
    } else {
        None
    };

    // ========== 7. Marginalize frames if requested ==========
    let marginalized_frame_indices: Vec<usize> = frame_graph
        .states
        .iter()
        .enumerate()
        .filter(|(_, s)| s.state == OptimizationState::Marginalize)
        .map(|(idx, _)| idx)
        .collect();

    let new_prior = if !marginalized_frame_indices.is_empty() {
        // Need to compute Jacobian at optimized params for marginalization
        use odysseus_solver::sparse_solver::build_jacobian;
        
        // Rebuild sparsity entries (same as used in optimization, but without prior)
        let mut entries = Vec::new();
        
        // Visual entries
        for (obs_idx, obs) in visual_obs_filtered.iter().enumerate() {
            let res_idx = obs_idx * 4;
            let opt_pose = pose_to_param_idx.contains_key(&obs.camera_id);
            let opt_point = point_to_param_idx.contains_key(&obs.point_id);
            
            if let Some(&p_idx) = pose_to_param_idx.get(&obs.camera_id) {
                if opt_pose {
                    for i in 0..6 {
                        for r in 0..4 {
                            entries.push((res_idx + r, p_idx + i));
                        }
                    }
                }
            }
            if let Some(&pt_idx) = point_to_param_idx.get(&obs.point_id) {
                if opt_point {
                    for i in 0..3 {
                        for r in 0..4 {
                            entries.push((res_idx + r, pt_idx + i));
                        }
                    }
                }
            }
        }
        
        // IMU entries
        let n_imu_factors = n_frames - 1;
        let n_visual_residuals = visual_obs_filtered.len() * 4;
        let imu_res_start = n_visual_residuals;
        
        for i in 0..n_imu_factors {
            let res_base = imu_res_start + i * 9;
            let bias_res_base = imu_res_start + (n_frames - 1) * 9 + i * 6;
            
            if let Some(&p_i) = pose_to_param_idx.get(&i) {
                for p in 0..15 {
                    for r in 0..9 {
                        entries.push((res_base + r, p_i + p));
                    }
                }
                for p in 9..15 {
                    for r in 0..6 {
                        entries.push((bias_res_base + r, p_i + p));
                    }
                }
            }
            if let Some(&p_j) = pose_to_param_idx.get(&(i + 1)) {
                for p in 0..15 {
                    for r in 0..9 {
                        entries.push((res_base + r, p_j + p));
                    }
                }
                for p in 9..15 {
                    for r in 0..6 {
                        entries.push((bias_res_base + r, p_j + p));
                    }
                }
            }
        }
        
        entries.sort();
        entries.dedup();
        
        let marg_n_imu_residuals = n_imu_factors * 9;
        let marg_n_bias_residuals = n_imu_factors * 6;
        let marg_n_residuals = n_visual_residuals + marg_n_imu_residuals + marg_n_bias_residuals;
        let mut jacobian = build_jacobian::<f64>(&entries, marg_n_residuals, n_params);
        let mut residuals = vec![0.0; marg_n_residuals];
        
        // Recompute cost at optimized params to get Jacobian (no prior for fresh marginalization)
        compute_simple_vio_cost(
            &optimized,
            &mut residuals,
            jacobian.data_mut(),
            world,
            &visual_obs_filtered,
            imu_states,
            preintegrations,
            &pose_to_param_idx,
            &point_to_param_idx,
            stereo_camera,
            &gravity,
            config,
            &[], // No prior when computing fresh marginalization
            n_visual_residuals,
            marg_n_imu_residuals,
            marg_n_bias_residuals,
        );
        
        // Get observing poses for each point (for marginalization logic)
        let get_observing_poses = |point_id: usize| -> HashSet<usize> {
            frame_observations
                .iter()
                .enumerate()
                .filter(|(_, obs_list)| obs_list.iter().any(|o| o.point_id == point_id))
                .map(|(frame_idx, _)| frame_idx)
                .collect()
        };
        
        // Compute marginalization using 15 params per pose
        super::compute_marginalization(
            &jacobian,
            &optimized,
            15, // params_per_pose for VIO
            n_params,
            &pose_to_param_idx,
            &point_to_param_idx,
            &marginalized_frame_indices,
            get_observing_poses,
        )
    } else {
        None
    };

    VioResult {
        final_error,
        iterations: iteration_count,
        converged,
        solve_time_ms,
        graph_info,
        new_prior,
    }
}

/// Compute the sqrt-information matrix for an IMU preintegration factor.
///
/// Returns L^{-1} where cov = L L^T (Cholesky). Weighting residuals by this
/// matrix is equivalent to premultiplying by sqrt(Sigma^{-1}), which correctly
/// balances IMU residuals against visual residuals in the joint cost.
/// Falls back to identity when the covariance is singular (e.g. zero delta_time).
fn imu_sqrt_information(cov: &Matrix9<f64>) -> Matrix9<f64> {
    if let Some(chol) = nalgebra::Cholesky::new(*cov) {
        if let Some(l_inv) = chol.l().try_inverse() {
            return l_inv;
        }
    }
    Matrix9::identity()
}


#[allow(clippy::too_many_arguments)]
fn compute_simple_vio_cost(
    params: &DVector<f64>,
    residuals: &mut [f64],
    jacobian_data: &mut [f64],
    world: &WorldState,
    visual_obs: &[StereoObservation],
    imu_states_host: &[ImuFrameState],
    preintegrations: &[PreintegratedImu],
    pose_to_param_idx: &HashMap<usize, usize>,
    point_to_param_idx: &HashMap<usize, usize>,
    stereo_camera: &StereoCamera<f64>,
    gravity: &[f64; 3],
    config: &VioConfig,
    priors: &[super::VioMarginalizedPrior],
    n_visual_residuals: usize,
    n_imu_residuals: usize,
    n_bias_residuals: usize,
) {
    let mut jac_cursor = 0;

    // 1. Visual Cost
    for (obs_idx, obs) in visual_obs.iter().enumerate() {
        let opt_pose = pose_to_param_idx.contains_key(&obs.camera_id);
        let opt_point = point_to_param_idx.contains_key(&obs.point_id);

        // Use Jet for autodiff
        type JetV = Jet<f64, 9>;

        let rot_host = &world.frames[obs.camera_id].pose.rotation_host;

        let pose_params: [JetV; 6] = if opt_pose {
            let base = pose_to_param_idx[&obs.camera_id];
            std::array::from_fn(|i| JetV::variable(params[base + i], i))
        } else {
            let p = &world.frames[obs.camera_id].pose;
            [
                JetV::constant(p.rotation.x),
                JetV::constant(p.rotation.y),
                JetV::constant(p.rotation.z),
                JetV::constant(p.translation.x),
                JetV::constant(p.translation.y),
                JetV::constant(p.translation.z),
            ]
        };

        // Get point info for host pose and parameters
        let pt_info = world.get_point_info(obs.point_id).unwrap();
        
        let pt_params: [JetV; 3] = if opt_point {
            let base = point_to_param_idx[&obs.point_id];
            let offset = if opt_pose { 6 } else { 0 };
            std::array::from_fn(|i| JetV::variable(params[base + i], offset + i))
        } else {
            [
                JetV::constant(pt_info.direction.0),
                JetV::constant(pt_info.direction.1),
                JetV::constant(pt_info.inv_depth),
            ]
        };

        let camera_jet = StereoCamera::new(
            crate::camera::PinholeCamera::new(
                JetV::constant(stereo_camera.left.fx),
                JetV::constant(stereo_camera.left.fy),
                JetV::constant(stereo_camera.left.cx),
                JetV::constant(stereo_camera.left.cy),
            ),
            JetV::constant(stereo_camera.baseline),
        );

        let (r1, r2, r3, r4) = crate::optimization::stereo_reprojection_residual_inverse_depth(
            rot_host,
            &pose_params,
            &pt_params,
            &pt_info.host_pose,
            &camera_jet,
            JetV::constant(obs.left_u),
            JetV::constant(obs.left_v),
            JetV::constant(obs.right_u),
            JetV::constant(obs.right_v),
        );

        let res_jets = [r1, r2, r3, r4];
        for (i, r) in res_jets.iter().enumerate() {
            let mut r_val = r.value;
            let mut deriv_slice = if opt_pose && opt_point {
                [
                    r.derivs[0], r.derivs[1], r.derivs[2], r.derivs[3], r.derivs[4], r.derivs[5],
                    r.derivs[6], r.derivs[7], r.derivs[8],
                ]
            } else if opt_pose {
                [
                    r.derivs[0], r.derivs[1], r.derivs[2], r.derivs[3], r.derivs[4], r.derivs[5],
                    0.0, 0.0, 0.0,
                ]
            } else {
                [r.derivs[0], r.derivs[1], r.derivs[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            };

            let n_active = (if opt_pose { 6 } else { 0 }) + (if opt_point { 3 } else { 0 });
            apply_huber_loss(config.huber_delta, &mut r_val, &mut deriv_slice[..n_active]);

            // Apply observation noise weighting
            let obs_weight = 1.0 / (config.obs_std_dev * config.obs_std_dev);
            r_val *= obs_weight;
            for d in &mut deriv_slice[..n_active] {
                *d *= obs_weight;
            }

            residuals[obs_idx * 4 + i] = r_val;
            if opt_pose {
                for p in 0..6 {
                    jacobian_data[jac_cursor] = deriv_slice[p];
                    jac_cursor += 1;
                }
            }
            if opt_point {
                let off = if opt_pose { 6 } else { 0 };
                for p in 0..3 {
                    jacobian_data[jac_cursor] = deriv_slice[off + p];
                    jac_cursor += 1;
                }
            }
        }
    }

    // 2. IMU Cost
    let imu_res_base = visual_obs.len() * 4;
    let n_frames = world.frames.len();
    for i in 0..n_frames - 1 {
        let res_idx = imu_res_base + i * 9;
        let opt_i = pose_to_param_idx.get(&i);
        let opt_j = pose_to_param_idx.get(&(i + 1));

        type Jet30 = Jet<f64, 30>; // 15 + 15

        let p_i: [Jet30; 15] = if let Some(&idx) = opt_i {
            std::array::from_fn(|p| Jet30::variable(params[idx + p], p))
        } else {
            let imu = &imu_states_host[i];
            let pose = &world.frames[i].pose;
            [
                Jet30::constant(pose.rotation.x),
                Jet30::constant(pose.rotation.y),
                Jet30::constant(pose.rotation.z),
                Jet30::constant(pose.translation.x),
                Jet30::constant(pose.translation.y),
                Jet30::constant(pose.translation.z),
                Jet30::constant(imu.velocity.x),
                Jet30::constant(imu.velocity.y),
                Jet30::constant(imu.velocity.z),
                Jet30::constant(imu.gyro_bias.x),
                Jet30::constant(imu.gyro_bias.y),
                Jet30::constant(imu.gyro_bias.z),
                Jet30::constant(imu.accel_bias.x),
                Jet30::constant(imu.accel_bias.y),
                Jet30::constant(imu.accel_bias.z),
            ]
        };

        let p_j: [Jet30; 15] = if let Some(&idx) = opt_j {
            let offset = if opt_i.is_some() { 15 } else { 0 };
            std::array::from_fn(|p| Jet30::variable(params[idx + p], offset + p))
        } else {
            let imu = &imu_states_host[i + 1];
            let pose = &world.frames[i + 1].pose;
            [
                Jet30::constant(pose.rotation.x),
                Jet30::constant(pose.rotation.y),
                Jet30::constant(pose.rotation.z),
                Jet30::constant(pose.translation.x),
                Jet30::constant(pose.translation.y),
                Jet30::constant(pose.translation.z),
                Jet30::constant(imu.velocity.x),
                Jet30::constant(imu.velocity.y),
                Jet30::constant(imu.velocity.z),
                Jet30::constant(imu.gyro_bias.x),
                Jet30::constant(imu.gyro_bias.y),
                Jet30::constant(imu.gyro_bias.z),
                Jet30::constant(imu.accel_bias.x),
                Jet30::constant(imu.accel_bias.y),
                Jet30::constant(imu.accel_bias.z),
            ]
        };

        let gravity_jet = [
            Jet30::constant(gravity[0]),
            Jet30::constant(gravity[1]),
            Jet30::constant(gravity[2]),
        ];

        let res = imu_preintegration_residual(
            &world.frames[i].pose.rotation_host,
            &p_i,
            &world.frames[i + 1].pose.rotation_host,
            &p_j,
            &preintegrations[i],
            &gravity_jet,
        );

        // Collect raw Jacobian blocks before applying information weighting
        let jac_i: Option<[[f64; 15]; 9]> = opt_i.map(|_| {
            std::array::from_fn(|k| std::array::from_fn(|p| res[k].derivs[p]))
        });
        let jac_j: Option<[[f64; 15]; 9]> = opt_j.map(|_| {
            let off = if opt_i.is_some() { 15 } else { 0 };
            std::array::from_fn(|k| std::array::from_fn(|p| res[k].derivs[off + p]))
        });

        // Weight residuals and Jacobians by sqrt-information (L^{-1} where cov = L L^T)
        let sqrt_info = imu_sqrt_information(&preintegrations[i].covariance);

        for r_idx in 0..9 {
            let mut weighted_r = 0.0;
            for k in 0..9 {
                weighted_r += sqrt_info[(r_idx, k)] * res[k].value;
            }
            residuals[res_idx + r_idx] = weighted_r;

            if let Some(ref ji) = jac_i {
                for p in 0..15 {
                    let mut wj = 0.0;
                    for k in 0..9 {
                        wj += sqrt_info[(r_idx, k)] * ji[k][p];
                    }
                    jacobian_data[jac_cursor] = wj;
                    jac_cursor += 1;
                }
            }
            if let Some(ref jj) = jac_j {
                for p in 0..15 {
                    let mut wj = 0.0;
                    for k in 0..9 {
                        wj += sqrt_info[(r_idx, k)] * jj[k][p];
                    }
                    jacobian_data[jac_cursor] = wj;
                    jac_cursor += 1;
                }
            }
        }
    }

    // 3. Bias Cost
    let bias_res_base = imu_res_base + (n_frames - 1) * 9;
    for i in 0..n_frames - 1 {
        let res_idx = bias_res_base + i * 6;
        let opt_i = pose_to_param_idx.get(&i);
        let opt_j = pose_to_param_idx.get(&(i + 1));

        let bg_i = if let Some(&idx) = opt_i {
            [params[idx + 9], params[idx + 10], params[idx + 11]]
        } else {
            let b = imu_states_host[i].gyro_bias;
            [b.x, b.y, b.z]
        };
        let ba_i = if let Some(&idx) = opt_i {
            [params[idx + 12], params[idx + 13], params[idx + 14]]
        } else {
            let b = imu_states_host[i].accel_bias;
            [b.x, b.y, b.z]
        };
        let bg_j = if let Some(&idx) = opt_j {
            [params[idx + 9], params[idx + 10], params[idx + 11]]
        } else {
            let b = imu_states_host[i + 1].gyro_bias;
            [b.x, b.y, b.z]
        };
        let ba_j = if let Some(&idx) = opt_j {
            [params[idx + 12], params[idx + 13], params[idx + 14]]
        } else {
            let b = imu_states_host[i + 1].accel_bias;
            [b.x, b.y, b.z]
        };

        let dt = preintegrations[i].delta_time;

        let res = bias_residual::<f64>(
            &bg_i,
            &ba_i,
            &bg_j,
            &ba_j,
            dt,
            config.gyro_sigma,
            config.accel_sigma,
        );

        let dt_sqrt = dt.sqrt();
        let gw = 1.0 / (config.gyro_sigma * dt_sqrt);
        let aw = 1.0 / (config.accel_sigma * dt_sqrt);

        for r_idx in 0..6 {
            residuals[res_idx + r_idx] = res[r_idx];
            let w = if r_idx < 3 { gw } else { aw };

            if opt_i.is_some() {
                for p in 0..6 {
                    jacobian_data[jac_cursor] = if p == r_idx { -w } else { 0.0 };
                    jac_cursor += 1;
                }
            }
            if opt_j.is_some() {
                for p in 0..6 {
                    jacobian_data[jac_cursor] = if p == r_idx { w } else { 0.0 };
                    jac_cursor += 1;
                }
            }
        }
    }

    // 4. Prior Cost (single prior if any)
    if !priors.is_empty() {
        let prior_res_start = n_visual_residuals + n_imu_residuals + n_bias_residuals;
        let mut current_prior_res_offset = prior_res_start;
        
        for prior in priors {
            jac_cursor = prior.apply_to_cost(
                params,
                residuals,
                jacobian_data,
                jac_cursor,
                current_prior_res_offset,
                pose_to_param_idx,
                point_to_param_idx,
            );
            current_prior_res_offset += prior.sqrt_information.nrows();
        }
    }
}
