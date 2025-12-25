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

pub mod marginalization;
pub mod slam;
pub mod vio;

// Re-export BA types for backward compatibility
pub use slam::{
    run_bundle_adjustment, BundleAdjustmentConfig, BundleAdjustmentResult, MarginalizedPrior,
};

// Re-export shared problem types
pub use marginalization::{compute_marginalization, SlamMarginalization};

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
pub(crate) fn jet_constants<const N: usize, const D: usize>(arr: &[f64; N]) -> [Jet<f64, D>; N] {
    std::array::from_fn(|i| Jet::constant(arr[i]))
}

/// Create variable jets from consecutive params with sequential derivative indices
pub(crate) fn jet_variables<
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

/// Apply Huber loss to a single residual and its Jacobian row
#[inline]
pub(crate) fn apply_huber_loss(huber_delta: f64, residual: &mut f64, jacobian_row: &mut [f64]) {
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
pub(crate) fn get_point_xyz(
    point_id: usize,
    params: &DVector<f64>,
    point_to_param_idx: &HashMap<usize, usize>,
    world: &WorldState,
) -> [f64; 3] {
    if let Some(&param_idx) = point_to_param_idx.get(&point_id) {
        [
            params[param_idx],
            params[param_idx + 1],
            params[param_idx + 2],
        ]
    } else {
        let point = world.get_point(point_id).unwrap();
        [point.x, point.y, point.z]
    }
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
