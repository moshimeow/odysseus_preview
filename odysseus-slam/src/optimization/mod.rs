//! Optimization utilities and bundle adjustment
//!
//! This module provides shared optimization utilities used by both
//! bundle adjustment and VIO.

use crate::camera::StereoCamera;
use crate::math::SE3;
use crate::world_state::WorldState;
use nalgebra::DVector;
use odysseus_solver::math3d::Vec3;
use odysseus_solver::{Jet, Real};
use std::collections::{HashMap, HashSet};

pub mod graph_visualization;
pub mod marginalization;
pub mod slam;
pub mod vio;
pub mod vio_slam;

// Re-export BA types for backward compatibility
pub use slam::{
    run_bundle_adjustment, BundleAdjustmentConfig,
    BundleAdjustmentResult, MarginalizedPrior, PointPrior, PointPriors,
};

// Re-export simple VIO types
pub use vio::{run_simple_vio_bundle_adjustment, VioConfig, VioResult};

// Re-export shared problem types
pub use marginalization::{compute_marginalization, SlamMarginalization, VioMarginalizedPrior};

// Re-export graph visualization types
pub use graph_visualization::{visualize_optimization_graph, OptimizationGraphInfo, PointState};

// ========== Shared helper functions ==========

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
            (false, true) => arm!(false, true, $size2),
            (true, false) => arm!(true, false, $size1),
            (true, true) => arm!(true, true, { $size1 + $size2 }),
        }
    }};
}

pub(crate) use with_jet_size;

/// Convert an f64 array to constant jets (no derivatives)
pub fn jet_constants<const N: usize, const D: usize>(arr: &[f64; N]) -> [Jet<f64, D>; N] {
    std::array::from_fn(|i| Jet::constant(arr[i]))
}

/// Create variable jets from consecutive params with sequential derivative indices
pub fn jet_variables<
    const N: usize,
    const D: usize,
    P: std::ops::Index<usize, Output = f64>,
>(
    params: &P,
    param_offset: usize,
    deriv_offset: usize,
) -> [Jet<f64, D>; N] {
    std::array::from_fn(|i| Jet::variable(params[param_offset + i], deriv_offset + i))
}

/// Create a Vec3 of variable jets from a nalgebra Vector3
pub fn jet_vec3_variable<const D: usize>(
    v: &nalgebra::Vector3<f64>,
    deriv_offset: usize,
) -> Vec3<Jet<f64, D>> {
    Vec3::new(
        Jet::variable(v.x, deriv_offset),
        Jet::variable(v.y, deriv_offset + 1),
        Jet::variable(v.z, deriv_offset + 2),
    )
}

/// Create a Vec3 of constant jets from a nalgebra Vector3
pub fn jet_vec3_constant<const D: usize>(v: &nalgebra::Vector3<f64>) -> Vec3<Jet<f64, D>> {
    Vec3::new(
        Jet::constant(v.x),
        Jet::constant(v.y),
        Jet::constant(v.z),
    )
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
    pose_params: &[T; 6], // [rotation_delta (3), translation (3)]
    world_point: &[T; 3],
    stereo_camera: &StereoCamera<crate::camera::PinholeCamera<T>, T>,
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
        T::from_f64(rotation_host.w),
        T::from_f64(rotation_host.x),
        T::from_f64(rotation_host.y),
        T::from_f64(rotation_host.z),
    );
    let q_new = q_host_t * q_delta;

    // Build world_T_camera pose
    let translation = Vec3::new(pose_params[3], pose_params[4], pose_params[5]);
    let world_t_camera =
        SE3::from_rotation_translation(crate::math::SO3 { quat: q_new }, translation);

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

pub use odysseus_solver::apply_huber_loss;

/// Get a point's XYZ position - either from params (if optimized) or from world (if fixed)
pub(crate) fn get_point_xyz(
    point_id: usize,
    params: &DVector<f64>,
    point_to_param_idx: &HashMap<usize, usize>,
    world: &WorldState,
) -> [f64; 3] {
    if let Some(&param_idx) = point_to_param_idx.get(&point_id) {
        // Point is optimized - get from params (inverse depth representation)
        let pt_info = world.get_point_info(point_id).unwrap();
        
        // Unproject 2D direction to 3D unit bearing
        let direction_u = params[param_idx];
        let direction_v = params[param_idx + 1];
        let inv_depth = params[param_idx + 2];
        
        let bearing = crate::math::stereographic::unproject(direction_u, direction_v);
        
        // Scale by distance
        let distance = 1.0 / inv_depth;
        let point_host = bearing * distance;
        
        // Transform from host to world
        let point_world = pt_info.host_pose.transform_point(point_host);
        [point_world.x, point_world.y, point_world.z]
    } else {
        // Point is fixed - get from world state
        let point = world.get_point(point_id).unwrap();
        [point.x, point.y, point.z]
    }
}

/// Compute stereo reprojection residual for inverse depth parameterized point
///
/// The point is parameterized as:
/// - direction: 2D stereographic projection of unit bearing from host pose
/// - inv_depth: inverse distance from host pose
/// - host_pose: fixed anchor pose (from PointInfo)
///
/// # Arguments
/// * `camera_rotation_host` - Host quaternion for camera pose (host-relative parameterization)
/// * `camera_pose_params` - Camera pose parameters [rotation_delta(3), translation(3)]
/// * `point_params` - Point parameters [direction_u, direction_v, inv_depth]
/// * `point_host_pose` - Host pose for the point (fixed, stored in PointInfo)
/// * `stereo_camera` - Stereo camera model
/// * `observed_*` - Observed pixel coordinates
///
/// # Returns
/// Tuple of 4 residuals (left_u, left_v, right_u, right_v)
pub fn stereo_reprojection_residual_inverse_depth<T: Real>(
    camera_rotation_host: &odysseus_solver::math3d::Quat<f64>,
    camera_pose_params: &[T; 6], // [rotation_delta (3), translation (3)]
    point_params: &[T; 3],        // [direction_u, direction_v, inv_depth]
    point_host_pose: &SE3<f64>,   // Fixed host pose from PointInfo
    stereo_camera: &StereoCamera<crate::camera::PinholeCamera<T>, T>,
    observed_left_u: T,
    observed_left_v: T,
    observed_right_u: T,
    observed_right_v: T,
) -> (T, T, T, T) {
    // 1. Unproject 2D direction to 3D unit bearing in host frame
    let bearing_host = crate::math::stereographic::unproject_jet::<T, 9>([point_params[0], point_params[1]]);
    
    // 2. Scale by distance (1 / inv_depth) to get 3D point in host frame
    let distance = T::one() / point_params[2];
    let point_host_x = bearing_host[0] * distance;
    let point_host_y = bearing_host[1] * distance;
    let point_host_z = bearing_host[2] * distance;
    
    // 3. Transform from point's host frame to world frame
    // point_host_pose components (f64) -> T
    let host_quat_t = odysseus_solver::math3d::Quat::new(
        T::from_f64(point_host_pose.rotation.quat.w),
        T::from_f64(point_host_pose.rotation.quat.x),
        T::from_f64(point_host_pose.rotation.quat.y),
        T::from_f64(point_host_pose.rotation.quat.z),
    );
    let host_trans_t = Vec3::new(
        T::from_f64(point_host_pose.translation.x),
        T::from_f64(point_host_pose.translation.y),
        T::from_f64(point_host_pose.translation.z),
    );
    
    let point_host_vec = Vec3::new(point_host_x, point_host_y, point_host_z);
    let point_world = host_quat_t.rotate_vec(point_host_vec) + host_trans_t;
    
    // 4. Now use standard residual computation (world point -> camera -> image)
    let world_point = [point_world.x, point_world.y, point_world.z];
    
    stereo_reprojection_residual_host_relative(
        camera_rotation_host,
        camera_pose_params,
        &world_point,
        stereo_camera,
        observed_left_u,
        observed_left_v,
        observed_right_u,
        observed_right_v,
    )
}

/// Select active points based on observation count in active frames
///
/// Returns (optimized_point_ids, all_active_point_ids)
/// - optimized_point_ids: points that will have their position optimized
/// - all_active_point_ids: all points used in residuals (including fixed)
pub fn select_active_points<O>(
    observations: &[O],
    get_point_id: impl Fn(&O) -> usize,
    get_frame_id: impl Fn(&O) -> usize,
    is_frame_active: impl Fn(usize) -> bool,
    fixed_point_ids: &HashSet<usize>,
    max_points: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut score_map: HashMap<usize, f64> = HashMap::new();

    for obs in observations {
        let frame_id = get_frame_id(obs);
        if is_frame_active(frame_id) {
            let point_id = get_point_id(obs);
            *score_map.entry(point_id).or_insert(0.0) += 1.0;
        }
    }

    // Sort by score descending, truncate to max
    let mut scored_points: Vec<_> = score_map.into_iter().collect();
    scored_points.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored_points.truncate(max_points);

    let all_point_ids: Vec<usize> = scored_points.into_iter().map(|(id, _)| id).collect();

    // Partition into optimized and fixed
    let optimized_point_ids: Vec<usize> = all_point_ids
        .iter()
        .copied()
        .filter(|id| !fixed_point_ids.contains(id))
        .collect();

    (optimized_point_ids, all_point_ids)
}
