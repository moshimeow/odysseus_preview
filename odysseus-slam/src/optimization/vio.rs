use crate::camera::StereoCamera;
use crate::frame_graph::FrameGraph;
use crate::geometry::StereoObservation;
use crate::imu::preintegration::PreintegratedImu;
use crate::imu::residuals::{bias_residual, imu_preintegration_residual};
use crate::imu::types::ImuFrameState;
use crate::optimization::select_active_points;
use crate::optimization::PointPriors;
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
}

/// Run tightly-coupled VIO bundle adjustment
///
/// Points with priors (from GBA) are kept fixed for performance, but their
/// uncertainty is used to scale visual residuals - uncertain points contribute
/// less to constraining poses.
pub fn run_vio_bundle_adjustment(
    stereo_camera: &StereoCamera<f64>,
    frame_graph: &FrameGraph,
    world: &mut WorldState,
    frame_observations: &[Vec<StereoObservation>],
    imu_states: &mut Vec<ImuFrameState>,
    preintegrations: &[PreintegratedImu],
    gravity: [f64; 3],
    point_priors: &PointPriors,
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

    // Points with priors from GBA are kept fixed for performance
    // Their uncertainty will be used to scale visual residuals instead
    let fixed_point_ids: HashSet<usize> = point_priors.priors.keys().copied().collect();

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
        &fixed_point_ids,
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
    let n_residuals = n_visual_residuals + n_imu_residuals + n_bias_residuals;

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

    let optimized = solver.solve(
        initial_params,
        |params, residuals, jacobian_data| {
            compute_vio_cost(
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
                point_priors,
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
            &fixed_point_ids,
        ))
    } else {
        None
    };

    VioResult {
        final_error,
        iterations: iteration_count,
        converged,
        solve_time_ms,
        graph_info,
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_vio_cost(
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
    point_priors: &PointPriors,
) {
    let mut jac_cursor = 0;

    // 1. Visual Cost
    // For fixed points with priors, we scale residuals by uncertainty.
    // The weight is derived from the point's information matrix:
    // Higher information (more certain) → weight closer to 1
    // Lower information (uncertain) → weight closer to 0
    for (obs_idx, obs) in visual_obs.iter().enumerate() {
        let opt_pose = pose_to_param_idx.contains_key(&obs.camera_id);
        let opt_point = point_to_param_idx.contains_key(&obs.point_id);

        // Compute uncertainty weight for this observation
        // For optimized points: weight = 1 (full contribution)
        // For fixed points with priors: weight based on information
        let uncertainty_weight = if !opt_point {
            if let Some(prior) = point_priors.get(obs.point_id) {
                // Use trace of information matrix as a measure of certainty
                // Normalize by expected information scale to get reasonable weights
                let info_trace = prior.information[(0, 0)]
                    + prior.information[(1, 1)]
                    + prior.information[(2, 2)];
                // Expected trace for a well-observed point (roughly)
                const EXPECTED_INFO_TRACE: f64 = 1000.0;
                // Weight: saturates at 1 for high information, falls off for low
                (info_trace / (info_trace + EXPECTED_INFO_TRACE)).sqrt()
            } else {
                // Fixed point without prior - full weight (it's trusted)
                1.0
            }
        } else {
            // Optimized point - full weight
            1.0
        };

        // Use Jet for autodiff
        // Visual residual only depends on 6 pose params + 3 point params
        // We'll use Jet9 to match bundle_adjustment's structure
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
            // Point params are [direction_u, direction_v, inv_depth]
            std::array::from_fn(|i| JetV::variable(params[base + i], offset + i))
        } else {
            // Fixed point - use current inverse depth parameters
            [
                JetV::constant(pt_info.direction.0),  // direction_u
                JetV::constant(pt_info.direction.1),  // direction_v
                JetV::constant(pt_info.inv_depth),    // inverse depth
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

        // Use inverse depth residual
        let (r1, r2, r3, r4) = crate::optimization::stereo_reprojection_residual_inverse_depth(
            rot_host,
            &pose_params,
            &pt_params,
            &pt_info.host_pose,  // Fixed host pose from point
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
                    r.derivs[0],
                    r.derivs[1],
                    r.derivs[2],
                    r.derivs[3],
                    r.derivs[4],
                    r.derivs[5],
                    r.derivs[6],
                    r.derivs[7],
                    r.derivs[8],
                ]
            } else if opt_pose {
                [
                    r.derivs[0],
                    r.derivs[1],
                    r.derivs[2],
                    r.derivs[3],
                    r.derivs[4],
                    r.derivs[5],
                    0.0,
                    0.0,
                    0.0,
                ]
            } else {
                [
                    r.derivs[0],
                    r.derivs[1],
                    r.derivs[2],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            };

            let n_active = (if opt_pose { 6 } else { 0 }) + (if opt_point { 3 } else { 0 });
            apply_huber_loss(config.huber_delta, &mut r_val, &mut deriv_slice[..n_active]);

            // Apply observation noise weighting (information matrix = 1/variance)
            // This ensures residuals are properly weighted relative to IMU residuals
            let obs_weight = 1.0 / (config.obs_std_dev * config.obs_std_dev);
            
            // Combined weight: observation noise * point uncertainty
            let combined_weight = obs_weight * uncertainty_weight;
            
            r_val *= combined_weight;
            for d in &mut deriv_slice[..n_active] {
                *d *= combined_weight;
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

        for r_idx in 0..9 {
            residuals[res_idx + r_idx] = res[r_idx].value;
            if opt_i.is_some() {
                for p in 0..15 {
                    jacobian_data[jac_cursor] = res[r_idx].derivs[p];
                    jac_cursor += 1;
                }
            }
            if opt_j.is_some() {
                let off = if opt_i.is_some() { 15 } else { 0 };
                for p in 0..15 {
                    jacobian_data[jac_cursor] = res[r_idx].derivs[off + p];
                    jac_cursor += 1;
                }
            }
        }
    }

    // 3. Bias Cost
    let bias_res_base = imu_res_base + (n_frames - 1) * 9;
    // For bias residual we use f64 directly as it's simple linear
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
                    // Only biases (9-14)
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
}

use crate::imu::preintegration::{VisuallyInformedPreintegration, Matrix15};
use nalgebra::DMatrix;
use sprs::TriMat;

/// Extract visually-informed preintegrations from VIO optimization results
///
/// This function recomputes the Jacobian at the converged solution, builds the Hessian,
/// and extracts the posterior covariance for relative motion between consecutive frames.
///
/// Unlike the old RelativePoseConstraint approach, this returns the pose-invariant
/// preintegrations along with posterior estimates, allowing GBA to maintain IMU dynamics
/// when adjusting poses.
///
/// # Arguments
/// * All the same arguments as run_vio_bundle_adjustment, plus:
/// * The converged world state and imu_states after optimization
///
/// # Returns
/// A vector of VisuallyInformedPreintegration, one for each consecutive frame pair
pub fn extract_relative_pose_constraints(
    stereo_camera: &StereoCamera<f64>,
    frame_graph: &FrameGraph,
    world: &WorldState,
    frame_observations: &[Vec<StereoObservation>],
    imu_states: &[ImuFrameState],
    preintegrations: &[PreintegratedImu],
    gravity: [f64; 3],
    point_priors: &PointPriors,
    config: &VioConfig,
) -> Vec<VisuallyInformedPreintegration> {
    let n_frames = world.frames.len();
    if n_frames < 2 {
        return Vec::new();
    }

    // ========== 1. Collect observations and build parameter mappings ==========
    // (Same setup as run_vio_bundle_adjustment)

    let active_frame_indices: Vec<_> = frame_graph
        .states
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_optimized())
        .map(|(idx, _)| idx)
        .collect();

    let all_obs: Vec<_> = active_frame_indices
        .iter()
        .flat_map(|&frame_idx| {
            frame_observations[frame_idx]
                .iter()
                .filter(|obs| world.get_point(obs.point_id).is_some())
                .copied()
        })
        .collect();

    // Points with priors from GBA are kept fixed (consistent with VIO optimization)
    let fixed_point_ids: HashSet<usize> = point_priors.priors.keys().copied().collect();

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
        &fixed_point_ids,
        config.max_active_points,
    );

    let active_points_set: HashSet<_> = all_point_ids.into_iter().collect();

    // Build parameter mappings
    let mut pose_to_param_idx = HashMap::new();
    let mut point_to_param_idx = HashMap::new();
    let mut offset = 0;

    for &idx in &active_frame_indices {
        pose_to_param_idx.insert(idx, offset);
        offset += 15;
    }
    for &id in &optimized_point_ids {
        point_to_param_idx.insert(id, offset);
        offset += 3;
    }
    let n_params = offset;

    // ========== 2. Build sparsity pattern ==========

    let mut entries = Vec::new();
    let visual_obs_filtered: Vec<_> = all_obs
        .iter()
        .filter(|obs| {
            (pose_to_param_idx.contains_key(&obs.camera_id)
                || point_to_param_idx.contains_key(&obs.point_id))
                && active_points_set.contains(&obs.point_id)
        })
        .copied()
        .collect();

    for (obs_idx, obs) in visual_obs_filtered.iter().enumerate() {
        let res_idx = obs_idx * 4;
        if let Some(&p_idx) = pose_to_param_idx.get(&obs.camera_id) {
            for i in 0..6 {
                for r in 0..4 {
                    entries.push((res_idx + r, p_idx + i));
                }
            }
        }
        if let Some(&pt_idx) = point_to_param_idx.get(&obs.point_id) {
            for i in 0..3 {
                for r in 0..4 {
                    entries.push((res_idx + r, pt_idx + i));
                }
            }
        }
    }

    let n_imu_factors = n_frames - 1;
    let n_visual_residuals = visual_obs_filtered.len() * 4;
    let n_imu_residuals = n_imu_factors * 9;
    let n_bias_residuals = n_imu_factors * 6;
    let n_residuals = n_visual_residuals + n_imu_residuals + n_bias_residuals;

    let imu_res_start = n_visual_residuals;
    for i in 0..n_imu_factors {
        let res_base = imu_res_start + i * 9;
        let bias_res_base = imu_res_start + n_imu_residuals + i * 6;

        let opt_i = pose_to_param_idx.get(&i);
        let opt_j = pose_to_param_idx.get(&(i + 1));

        if let Some(&p_i) = opt_i {
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

    entries.sort();
    entries.dedup();

    // ========== 3. Pack current (converged) parameters ==========

    let mut params = DVector::zeros(n_params);
    for (&idx, &p_idx) in &pose_to_param_idx {
        let pose = &world.frames[idx].pose;
        let imu = &imu_states[idx];
        params[p_idx + 0] = pose.rotation.x;
        params[p_idx + 1] = pose.rotation.y;
        params[p_idx + 2] = pose.rotation.z;
        params[p_idx + 3] = pose.translation.x;
        params[p_idx + 4] = pose.translation.y;
        params[p_idx + 5] = pose.translation.z;
        params[p_idx + 6] = imu.velocity.x;
        params[p_idx + 7] = imu.velocity.y;
        params[p_idx + 8] = imu.velocity.z;
        params[p_idx + 9] = imu.gyro_bias.x;
        params[p_idx + 10] = imu.gyro_bias.y;
        params[p_idx + 11] = imu.gyro_bias.z;
        params[p_idx + 12] = imu.accel_bias.x;
        params[p_idx + 13] = imu.accel_bias.y;
        params[p_idx + 14] = imu.accel_bias.z;
    }
    for &id in &optimized_point_ids {
        if let Some(pt) = world.get_point(id) {
            let p_idx = point_to_param_idx[&id];
            params[p_idx + 0] = pt.x;
            params[p_idx + 1] = pt.y;
            params[p_idx + 2] = pt.z;
        }
    }

    // ========== 4. Recompute Jacobian at converged solution ==========

    // Build Jacobian using TriMat (COO format) then convert to CSR
    let mut tri = TriMat::new((n_residuals, n_params));
    for &(row, col) in &entries {
        tri.add_triplet(row, col, 0.0f64);
    }
    let mut jacobian = tri.to_csr::<usize>();

    let mut residuals = vec![0.0; n_residuals];

    compute_vio_cost(
        &params,
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
        point_priors,
    );

    // ========== 5. Compute Hessian H = J^T * J ==========

    let jt = jacobian.transpose_view();
    let hessian_sparse = &jt * &jacobian;

    // Convert to dense for block extraction
    let mut hessian = DMatrix::<f64>::zeros(n_params, n_params);
    for (value, (row, col)) in hessian_sparse.iter() {
        hessian[(row, col)] = *value;
    }

    // ========== 6. Extract constraints for consecutive frame pairs ==========

    let gravity_vec = Vector3::new(gravity[0], gravity[1], gravity[2]);
    let mut constraints = Vec::new();

    // Only extract for consecutive frames that are both in the optimization
    for frame_i in 0..(n_frames - 1) {
        let frame_j = frame_i + 1;

        // Both frames must be in the pose_to_param_idx
        let param_i = match pose_to_param_idx.get(&frame_i) {
            Some(&p) => p,
            None => continue,
        };
        let param_j = match pose_to_param_idx.get(&frame_j) {
            Some(&p) => p,
            None => continue,
        };

        // Extract 30x30 block for frames i and j
        let mut h_block = DMatrix::<f64>::zeros(30, 30);
        for ii in 0..15 {
            for jj in 0..15 {
                h_block[(ii, jj)] = hessian[(param_i + ii, param_i + jj)];
                h_block[(ii, 15 + jj)] = hessian[(param_i + ii, param_j + jj)];
                h_block[(15 + ii, jj)] = hessian[(param_j + ii, param_i + jj)];
                h_block[(15 + ii, 15 + jj)] = hessian[(param_j + ii, param_j + jj)];
            }
        }

        // Regularize and invert to get covariance
        // Add small regularization to diagonal for numerical stability
        for k in 0..30 {
            h_block[(k, k)] += 1e-6;
        }

        let cov_block = match h_block.try_inverse() {
            Some(c) => c,
            None => {
                eprintln!("Warning: Failed to invert Hessian block for frames {} and {}", frame_i, frame_j);
                continue;
            }
        };

        // Compute posterior estimates of relative motion (for initialization/warm start)
        let pose_i = &world.frames[frame_i].pose;
        let pose_j = &world.frames[frame_j].pose;
        let rot_i = pose_i.world_rotation();
        let rot_j = pose_j.world_rotation();
        let trans_i = Vector3::new(pose_i.translation.x, pose_i.translation.y, pose_i.translation.z);
        let trans_j = Vector3::new(pose_j.translation.x, pose_j.translation.y, pose_j.translation.z);
        let vel_i = imu_states[frame_i].velocity;
        let vel_j = imu_states[frame_j].velocity;
        
        // Compute posterior deltas (incorporating visual information)
        let r_i_inv = rot_i.conjugate();
        let delta_rotation_posterior = r_i_inv * rot_j;
        
        let dt = preintegrations[frame_i].delta_time;
        let vel_diff = vel_j - vel_i - gravity_vec * dt;
        let delta_velocity_posterior = rotate_vec_by_quat(&r_i_inv, &vel_diff);
        
        let pos_diff = trans_j - trans_i - vel_i * dt - 0.5 * gravity_vec * dt * dt;
        let delta_position_posterior = rotate_vec_by_quat(&r_i_inv, &pos_diff);
        
        // Extract posterior biases from frame i (used for preintegration correction)
        let gyro_bias_posterior = imu_states[frame_i].gyro_bias;
        let accel_bias_posterior = imu_states[frame_i].accel_bias;
        
        // Compute relative motion covariance including biases (15x15)
        // Transform from joint 30x30 to relative 15x15 using Jacobian
        let cov_rel = compute_relative_motion_covariance_with_biases(
            &world.frames[frame_i].pose,
            &imu_states[frame_i].velocity,
            &world.frames[frame_j].pose,
            &imu_states[frame_j].velocity,
            &gravity_vec,
            dt,
            &cov_block,
        );
        
        // Create visually-informed preintegration
        let visually_informed = VisuallyInformedPreintegration {
            preintegration: preintegrations[frame_i].clone(),
            delta_rotation_posterior,
            delta_velocity_posterior,
            delta_position_posterior,
            covariance_posterior: cov_rel,
            gyro_bias_posterior,
            accel_bias_posterior,
            frame_i,
            frame_j,
        };
        
        constraints.push(visually_informed);
    }

    constraints
}

/// Helper to rotate a Vector3 by a Quat
fn rotate_vec_by_quat(q: &odysseus_solver::math3d::Quat<f64>, v: &Vector3<f64>) -> Vector3<f64> {
    let v_solver = odysseus_solver::math3d::Vec3::new(v.x, v.y, v.z);
    let rotated = q.rotate_vec(v_solver);
    Vector3::new(rotated.x, rotated.y, rotated.z)
}

/// Compute relative motion covariance including biases from joint frame covariance
///
/// Transforms 30x30 joint covariance [state_i, state_j] to 15x15 relative covariance
/// [delta_R, delta_v, delta_p, delta_bg, delta_ba]
///
/// Uses numerical differentiation to compute J_rel, then transforms:
/// Cov_rel = J_rel * Cov_joint * J_rel^T
fn compute_relative_motion_covariance_with_biases(
    pose_i: &crate::world_state::Pose,
    velocity_i: &Vector3<f64>,
    pose_j: &crate::world_state::Pose,
    velocity_j: &Vector3<f64>,
    gravity: &Vector3<f64>,
    delta_time: f64,
    cov_joint: &DMatrix<f64>,
) -> Matrix15<f64> {
    use odysseus_solver::math3d::Quat;

    // Compute J_rel (15x30) numerically
    // Output: [delta_R(3), delta_v(3), delta_p(3), delta_bg(3), delta_ba(3)] = 15
    // Input: [R_i(3), p_i(3), v_i(3), bg_i(3), ba_i(3), R_j(3), p_j(3), v_j(3), bg_j(3), ba_j(3)] = 30
    let eps = 1e-6;
    let mut j_rel = DMatrix::<f64>::zeros(15, 30);

    // Helper to compute relative motion given states
    let compute_rel = |rot_i: &Quat<f64>, trans_i: &Vector3<f64>, vel_i: &Vector3<f64>, 
                       bg_i: &Vector3<f64>, ba_i: &Vector3<f64>,
                       rot_j: &Quat<f64>, trans_j: &Vector3<f64>, vel_j: &Vector3<f64>,
                       bg_j: &Vector3<f64>, ba_j: &Vector3<f64>| -> [f64; 15] {
        let r_i_inv = rot_i.conjugate();
        let delta_rot = r_i_inv * *rot_j;
        let delta_rot_vec = delta_rot.to_axis_angle();

        let vel_diff = vel_j - vel_i - gravity * delta_time;
        let delta_vel = rotate_vec_by_quat(&r_i_inv, &vel_diff);

        let pos_diff = trans_j - trans_i - vel_i * delta_time - 0.5 * gravity * delta_time * delta_time;
        let delta_pos = rotate_vec_by_quat(&r_i_inv, &pos_diff);
        
        // Bias deltas
        let delta_bg = bg_j - bg_i;
        let delta_ba = ba_j - ba_i;

        [
            delta_rot_vec.x, delta_rot_vec.y, delta_rot_vec.z,
            delta_vel.x, delta_vel.y, delta_vel.z,
            delta_pos.x, delta_pos.y, delta_pos.z,
            delta_bg.x, delta_bg.y, delta_bg.z,
            delta_ba.x, delta_ba.y, delta_ba.z,
        ]
    };

    // Base values
    let rot_i = pose_i.world_rotation();
    let trans_i = Vector3::new(pose_i.translation.x, pose_i.translation.y, pose_i.translation.z);
    let rot_j = pose_j.world_rotation();
    let trans_j = Vector3::new(pose_j.translation.x, pose_j.translation.y, pose_j.translation.z);
    
    // Get biases from the joint covariance (we'll use zeros as base, derivatives will be identity for biases)
    let bg_i = Vector3::zeros();
    let ba_i = Vector3::zeros();
    let bg_j = Vector3::zeros();
    let ba_j = Vector3::zeros();

    let base = compute_rel(&rot_i, &trans_i, velocity_i, &bg_i, &ba_i,
                          &rot_j, &trans_j, velocity_j, &bg_j, &ba_j);

    // Frame i: rotation (0-2), translation (3-5), velocity (6-8), gyro_bias (9-11), accel_bias (12-14)
    // Numerical differentiation for frame i rotation (params 0-2)
    for p in 0..3 {
        let mut delta = Vector3::zeros();
        delta[p] = eps;
        let rot_i_pert = rot_i * Quat::from_axis_angle(odysseus_solver::math3d::Vec3::new(delta.x, delta.y, delta.z));
        let pert = compute_rel(&rot_i_pert, &trans_i, velocity_i, &bg_i, &ba_i,
                              &rot_j, &trans_j, velocity_j, &bg_j, &ba_j);
        for r in 0..15 {
            j_rel[(r, p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Frame i translation (params 3-5)
    for p in 0..3 {
        let mut trans_i_pert = trans_i;
        trans_i_pert[p] += eps;
        let pert = compute_rel(&rot_i, &trans_i_pert, velocity_i, &bg_i, &ba_i,
                              &rot_j, &trans_j, velocity_j, &bg_j, &ba_j);
        for r in 0..15 {
            j_rel[(r, 3 + p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Frame i velocity (params 6-8)
    for p in 0..3 {
        let mut vel_i_pert = *velocity_i;
        vel_i_pert[p] += eps;
        let pert = compute_rel(&rot_i, &trans_i, &vel_i_pert, &bg_i, &ba_i,
                              &rot_j, &trans_j, velocity_j, &bg_j, &ba_j);
        for r in 0..15 {
            j_rel[(r, 6 + p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Frame i gyro bias (params 9-11)
    for p in 0..3 {
        let mut bg_i_pert = bg_i;
        bg_i_pert[p] += eps;
        let pert = compute_rel(&rot_i, &trans_i, velocity_i, &bg_i_pert, &ba_i,
                              &rot_j, &trans_j, velocity_j, &bg_j, &ba_j);
        for r in 0..15 {
            j_rel[(r, 9 + p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Frame i accel bias (params 12-14)
    for p in 0..3 {
        let mut ba_i_pert = ba_i;
        ba_i_pert[p] += eps;
        let pert = compute_rel(&rot_i, &trans_i, velocity_i, &bg_i, &ba_i_pert,
                              &rot_j, &trans_j, velocity_j, &bg_j, &ba_j);
        for r in 0..15 {
            j_rel[(r, 12 + p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Frame j: rotation (15-17), translation (18-20), velocity (21-23), gyro_bias (24-26), accel_bias (27-29)
    // Frame j rotation (params 15-17)
    for p in 0..3 {
        let mut delta = Vector3::zeros();
        delta[p] = eps;
        let rot_j_pert = rot_j * Quat::from_axis_angle(odysseus_solver::math3d::Vec3::new(delta.x, delta.y, delta.z));
        let pert = compute_rel(&rot_i, &trans_i, velocity_i, &bg_i, &ba_i,
                              &rot_j_pert, &trans_j, velocity_j, &bg_j, &ba_j);
        for r in 0..15 {
            j_rel[(r, 15 + p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Frame j translation (params 18-20)
    for p in 0..3 {
        let mut trans_j_pert = trans_j;
        trans_j_pert[p] += eps;
        let pert = compute_rel(&rot_i, &trans_i, velocity_i, &bg_i, &ba_i,
                              &rot_j, &trans_j_pert, velocity_j, &bg_j, &ba_j);
        for r in 0..15 {
            j_rel[(r, 18 + p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Frame j velocity (params 21-23)
    for p in 0..3 {
        let mut vel_j_pert = *velocity_j;
        vel_j_pert[p] += eps;
        let pert = compute_rel(&rot_i, &trans_i, velocity_i, &bg_i, &ba_i,
                              &rot_j, &trans_j, &vel_j_pert, &bg_j, &ba_j);
        for r in 0..15 {
            j_rel[(r, 21 + p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Frame j gyro bias (params 24-26)
    for p in 0..3 {
        let mut bg_j_pert = bg_j;
        bg_j_pert[p] += eps;
        let pert = compute_rel(&rot_i, &trans_i, velocity_i, &bg_i, &ba_i,
                              &rot_j, &trans_j, velocity_j, &bg_j_pert, &ba_j);
        for r in 0..15 {
            j_rel[(r, 24 + p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Frame j accel bias (params 27-29)
    for p in 0..3 {
        let mut ba_j_pert = ba_j;
        ba_j_pert[p] += eps;
        let pert = compute_rel(&rot_i, &trans_i, velocity_i, &bg_i, &ba_i,
                              &rot_j, &trans_j, velocity_j, &bg_j, &ba_j_pert);
        for r in 0..15 {
            j_rel[(r, 27 + p)] = (pert[r] - base[r]) / eps;
        }
    }

    // Transform covariance: Cov_rel = J_rel * Cov_joint * J_rel^T
    let cov_rel_dense = &j_rel * cov_joint * j_rel.transpose();

    // Convert to Matrix15
    let mut cov_rel = Matrix15::<f64>::zeros();
    for i in 0..15 {
        for j in 0..15 {
            cov_rel[(i, j)] = cov_rel_dense[(i, j)];
        }
    }

    // Scale covariance to loosen the constraint (same philosophy as before)
    const COVARIANCE_SCALE_FACTOR: f64 = 100.0;
    cov_rel *= COVARIANCE_SCALE_FACTOR;

    // Ensure minimum diagonal values
    const MIN_COVARIANCE_DIAG: f64 = 1e-6;
    for i in 0..15 {
        if cov_rel[(i, i)] < MIN_COVARIANCE_DIAG {
            cov_rel[(i, i)] = MIN_COVARIANCE_DIAG;
        }
    }

    cov_rel
}
