//! Shared SLAM problem abstractions
//!
//! This module provides common types and utilities for both bundle adjustment and VIO.

use nalgebra::{Cholesky, DMatrix, DVector, SymmetricEigen};
use sprs::CsMat;
use std::collections::{HashMap, HashSet};

/// Marginalized prior from Schur complement
///
/// This stores the information from marginalized frames/points as a dense
/// prior constraint. Works for both BA (6 params/pose) and VIO (15 params/pose).
#[derive(Clone)]
pub struct SlamMarginalization {
    /// Cholesky factor of marginalized Hessian: H_marg = L * L^T
    pub sqrt_information: DMatrix<f64>,
    /// Linearization point for the kept parameters
    pub linearization_point: DVector<f64>,
    /// Number of parameters per pose (6 for BA, 15 for VIO)
    pub params_per_pose: usize,
    /// Pose frame IDs constrained by this prior (in order)
    pub pose_ids: Vec<usize>,
    /// Point IDs constrained by this prior (in order, after poses)
    pub point_ids: Vec<usize>,
}

impl SlamMarginalization {
    /// Compute the number of parameters in this prior
    pub fn n_params(&self) -> usize {
        self.linearization_point.len()
    }

    /// Get sparsity entries for prior residuals starting at `residual_offset`
    ///
    /// Each prior residual row connects to all prior-constrained parameters.
    pub fn sparsity_entries(
        &self,
        residual_offset: usize,
        pose_to_param_idx: &HashMap<usize, usize>,
        point_to_param_idx: &HashMap<usize, usize>,
    ) -> Vec<(usize, usize)> {
        let mut entries = Vec::new();
        let n_prior_residuals = self.sqrt_information.nrows();

        // Build list of parameter indices that this prior constrains
        let mut prior_param_indices: Vec<usize> = Vec::new();
        for &pose_id in &self.pose_ids {
            if let Some(&param_start) = pose_to_param_idx.get(&pose_id) {
                for i in 0..self.params_per_pose {
                    prior_param_indices.push(param_start + i);
                }
            }
        }
        for &point_id in &self.point_ids {
            if let Some(&param_start) = point_to_param_idx.get(&point_id) {
                for i in 0..3 {
                    prior_param_indices.push(param_start + i);
                }
            }
        }

        // Each prior residual row connects to all prior-constrained parameters
        for prior_row in 0..n_prior_residuals {
            let residual_idx = residual_offset + prior_row;
            for &param_idx in &prior_param_indices {
                entries.push((residual_idx, param_idx));
            }
        }

        entries
    }

    /// Apply prior residuals and Jacobian to the cost function
    ///
    /// Writes to residuals and jacobian_data starting at the given cursor.
    /// Returns the new jacobian cursor position.
    pub fn apply_to_cost(
        &self,
        params: &DVector<f64>,
        residuals: &mut [f64],
        jacobian_data: &mut [f64],
        jac_cursor: usize,
        residual_offset: usize,
        pose_to_param_idx: &HashMap<usize, usize>,
        point_to_param_idx: &HashMap<usize, usize>,
    ) -> usize {
        let mut cursor = jac_cursor;
        let n_prior_params = self.linearization_point.len();
        let mut current_prior_params = DVector::<f64>::zeros(n_prior_params);

        let mut prior_param_idx = 0;

        // Map prior's pose params to current window
        for &pose_id in &self.pose_ids {
            if let Some(&param_start) = pose_to_param_idx.get(&pose_id) {
                for i in 0..self.params_per_pose {
                    current_prior_params[prior_param_idx] = params[param_start + i];
                    prior_param_idx += 1;
                }
            } else {
                // Pose not in current window - use linearization point values
                for _ in 0..self.params_per_pose {
                    current_prior_params[prior_param_idx] =
                        self.linearization_point[prior_param_idx];
                    prior_param_idx += 1;
                }
            }
        }

        // Map prior's point params to current window
        for &point_id in &self.point_ids {
            if let Some(&param_start) = point_to_param_idx.get(&point_id) {
                for i in 0..3 {
                    current_prior_params[prior_param_idx] = params[param_start + i];
                    prior_param_idx += 1;
                }
            } else {
                // Point not in current window - use linearization point values
                for _ in 0..3 {
                    current_prior_params[prior_param_idx] =
                        self.linearization_point[prior_param_idx];
                    prior_param_idx += 1;
                }
            }
        }

        // Compute prior residuals: r_prior = sqrt_info * (current - linearization)
        let delta = &current_prior_params - &self.linearization_point;
        let r_prior = &self.sqrt_information * &delta;

        // Write prior residuals
        for i in 0..r_prior.len() {
            residuals[residual_offset + i] = r_prior[i];
        }

        // Write prior Jacobian
        let n_prior_residuals = self.sqrt_information.nrows();

        for prior_row in 0..n_prior_residuals {
            let mut prior_col = 0;

            // Pose columns
            for &pose_id in &self.pose_ids {
                if pose_to_param_idx.contains_key(&pose_id) {
                    for i in 0..self.params_per_pose {
                        jacobian_data[cursor] = self.sqrt_information[(prior_row, prior_col + i)];
                        cursor += 1;
                    }
                }
                prior_col += self.params_per_pose;
            }

            // Point columns
            for &point_id in &self.point_ids {
                if point_to_param_idx.contains_key(&point_id) {
                    for i in 0..3 {
                        jacobian_data[cursor] = self.sqrt_information[(prior_row, prior_col + i)];
                        cursor += 1;
                    }
                }
                prior_col += 3;
            }
        }

        cursor
    }
}

/// Result of marginalization computation
pub struct MarginalizationResult {
    /// New prior from marginalized parameters
    pub prior: SlamMarginalization,
}

/// Compute marginalization using Schur complement
///
/// This extracts the common marginalization logic used by both BA and VIO.
///
/// # Arguments
/// * `jacobian` - Full Jacobian at the linearization point
/// * `optimized_params` - Optimized parameters (linearization point)
/// * `params_per_pose` - 6 for BA, 15 for VIO
/// * `pose_to_param_idx` - Mapping from pose ID to parameter index
/// * `point_to_param_idx` - Mapping from point ID to parameter index
/// * `marginalized_pose_ids` - Pose IDs to marginalize out
/// * `get_observing_poses` - Closure that returns the set of pose IDs observing a given point
pub fn compute_marginalization<F>(
    jacobian: &CsMat<f64>,
    optimized_params: &DVector<f64>,
    params_per_pose: usize,
    n_params: usize,
    pose_to_param_idx: &HashMap<usize, usize>,
    point_to_param_idx: &HashMap<usize, usize>,
    marginalized_pose_ids: &[usize],
    get_observing_poses: F,
) -> Option<SlamMarginalization>
where
    F: Fn(usize) -> HashSet<usize>, // point_id -> set of observing pose_ids
{
    if marginalized_pose_ids.is_empty() {
        return None;
    }

    // Compute Hessian approximation: H = J^T * J
    let jt: CsMat<f64> = jacobian.clone().transpose_into();
    let sprs_hessian = &jt * jacobian;
    let mut hessian = DMatrix::<f64>::zeros(n_params, n_params);
    for (value, (row, col)) in sprs_hessian.iter() {
        hessian[(row, col)] = *value;
    }

    // Build set of marginalized pose indices for quick lookup
    let marg_pose_set: HashSet<usize> = marginalized_pose_ids.iter().copied().collect();

    // Find points that are ONLY observed by marginalized poses
    let mut unique_point_ids: Vec<usize> = Vec::new();
    for (&point_id, _) in point_to_param_idx {
        let observing_poses = get_observing_poses(point_id);
        if observing_poses.iter().all(|p| marg_pose_set.contains(p)) {
            unique_point_ids.push(point_id);
        }
    }

    // Build list of ALL parameter indices to marginalize
    let mut marg_indices: Vec<usize> = Vec::new();

    // Add pose parameters for each marginalized pose
    for &pose_id in marginalized_pose_ids {
        let pose_param_start = pose_to_param_idx[&pose_id];
        for i in 0..params_per_pose {
            marg_indices.push(pose_param_start + i);
        }
    }

    // Add point parameters for each unique point
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
        return None;
    }

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
    let h_oo_inv = h_oo.try_inverse()?;
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
    let min_eigenvalue = eigen
        .eigenvalues
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    if min_eigenvalue < 1e-8 {
        let regularization = 1e-6 - min_eigenvalue.min(0.0);
        for i in 0..h_marg.nrows() {
            h_marg[(i, i)] += regularization;
        }
    }

    // Cholesky factorization to get sqrt_information
    let cholesky = Cholesky::new(h_marg)?;
    let sqrt_information = cholesky.l();
    let linearization_point =
        DVector::from_iterator(n_new, keep_indices.iter().map(|&i| optimized_params[i]));

    // Track which pose and point IDs are in the kept parameters
    let param_to_pose: HashMap<usize, usize> = pose_to_param_idx
        .iter()
        .map(|(&pose_id, &param_start)| (param_start, pose_id))
        .collect();
    let param_to_point: HashMap<usize, usize> = point_to_param_idx
        .iter()
        .map(|(&point_id, &param_start)| (param_start, point_id))
        .collect();

    let mut kept_pose_ids: Vec<usize> = Vec::new();
    let mut kept_point_ids: Vec<usize> = Vec::new();

    for &keep_idx in &keep_indices {
        if let Some(&pose_id) = param_to_pose.get(&keep_idx) {
            if !kept_pose_ids.contains(&pose_id) {
                kept_pose_ids.push(pose_id);
            }
        }
        if let Some(&point_id) = param_to_point.get(&keep_idx) {
            if !kept_point_ids.contains(&point_id) {
                kept_point_ids.push(point_id);
            }
        }
    }

    // NOTE: Do NOT sort! The order must match keep_indices order.

    Some(SlamMarginalization {
        sqrt_information,
        linearization_point,
        params_per_pose,
        pose_ids: kept_pose_ids,
        point_ids: kept_point_ids,
    })
}
