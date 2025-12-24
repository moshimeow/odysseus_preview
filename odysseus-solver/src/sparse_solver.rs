//! Sparse Levenberg-Marquardt solver using sprs
//!
//! This module provides a sparse solver that leverages the sparsity of SLAM Jacobians
//! for dramatic memory and computation savings. The Jacobians are ~98%+ sparse.

use nalgebra::DVector;
use sprs::{CsMat, SymmetryCheck, TriMat};
use sprs_ldl::Ldl;

/// Result of one optimization iteration
pub struct IterationResult<T> {
    pub error: T,
    pub step_norm: T,
    pub lambda: T,
    pub gradient_norm: T,
    pub converged: bool,
}

pub fn build_jacobian<T>(entries: &[(usize, usize)], n_rows: usize, n_cols: usize) -> CsMat<T>
where
    T: num_traits::Float
        + nalgebra::RealField
        + std::fmt::Display
        + std::iter::Sum
        + Default
        + for<'r> std::ops::DivAssign<&'r T>
        + for<'a> std::ops::Mul<&'a T, Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T>,
{
    let mut tri = TriMat::new((n_rows, n_cols));
    for &(row, col) in entries {
        tri.add_triplet(row, col, T::zero());
    }
    tri.to_csr()
}

/// Sparse Levenberg-Marquardt solver
///
/// Uses sparse CSR matrices for Jacobians, with LDL factorization for solving.
/// The Jacobian structure is fixed at construction time; only values change during optimization.
pub struct SparseLevenbergMarquardt<T> {
    // Solver parameters
    pub tolerance: T,
    pub max_iterations: usize,
    pub initial_lambda: T,
    pub lambda_scale_up: T,
    pub lambda_scale_down: T,
    pub verbose: bool,

    // Sparse Jacobian in CSR format (structure fixed, values updated each iteration)
    jacobian: CsMat<T>,

    // Dense workspace vectors
    jtr: DVector<T>,            // J^T * r (dense, always full)
    residuals: DVector<T>,      // Dense residual vector
    temp_residuals: DVector<T>, // For trial step evaluation
}

impl<T> SparseLevenbergMarquardt<T>
where
    T: num_traits::Float
        + nalgebra::RealField
        + std::fmt::Display
        + std::iter::Sum
        + Default
        + for<'r> std::ops::DivAssign<&'r T>
        + for<'a> std::ops::Mul<&'a T, Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T>,
{
    /// Create a new sparse solver
    ///
    /// # Arguments
    /// * `n_rows` - Number of residuals
    /// * `n_cols` - Number of parameters
    /// * `entries` - List of (row, col) pairs defining the sparsity structure.
    ///   MUST be sorted by (row, col) for CSR format - i.e., row-major order.
    pub fn new(n_rows: usize, n_cols: usize, entries: &[(usize, usize)]) -> Self {
        // Build CSR matrix from entries using TriMat
        let jacobian = build_jacobian::<T>(entries, n_rows, n_cols);

        // Pre-allocate dense vectors
        let jtr = DVector::zeros(n_cols);
        let residuals = DVector::zeros(n_rows);
        let temp_residuals = DVector::zeros(n_rows);

        Self {
            tolerance: T::from(1e-10).unwrap(),
            max_iterations: 50,
            initial_lambda: T::from(1e-4).unwrap(),
            lambda_scale_up: T::from(10.0).unwrap(),
            lambda_scale_down: T::from(0.1).unwrap(),
            verbose: true,
            jacobian,
            jtr,
            residuals,
            temp_residuals,
        }
    }

    pub fn with_tolerance(mut self, tolerance: T) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_lambda_params(mut self, initial: T, scale_up: T, scale_down: T) -> Self {
        self.initial_lambda = initial;
        self.lambda_scale_up = scale_up;
        self.lambda_scale_down = scale_down;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Get the number of non-zero entries in the Jacobian
    pub fn nnz(&self) -> usize {
        self.jacobian.nnz()
    }

    /// Solve the optimization problem
    ///
    /// # Arguments
    /// * `params` - Initial parameter guess
    /// * `cost_fn` - Function that computes residuals and Jacobian.
    ///   Receives (params, residuals_slice, jacobian_data_slice).
    ///   Must write values to jacobian_data in the same order as the entries
    ///   were provided to the constructor (row-major order).
    /// * `callback` - Optional callback called after each iteration
    pub fn solve<F, C>(
        &mut self,
        mut params: DVector<T>,
        mut cost_fn: F,
        mut callback: C,
    ) -> DVector<T>
    where
        F: FnMut(&DVector<T>, &mut [T], &mut [T]),
        C: FnMut(usize, &IterationResult<T>, &DVector<T>),
    {
        let mut lambda = self.initial_lambda;

        for iteration in 0..self.max_iterations {
            // Zero out residuals (jacobian values will be overwritten by cost_fn)
            self.residuals.fill(T::zero());

            // Compute residuals and Jacobian
            cost_fn(
                &params,
                self.residuals.as_mut_slice(),
                self.jacobian.data_mut(),
            );
            let error = self.residuals.norm();

            // Compute J^T * J using sprs
            // Clone jacobian since transpose_into consumes self
            let jt: CsMat<T> = self.jacobian.clone().transpose_into();
            let jtj: CsMat<T> = &jt * &self.jacobian;

            // Compute J^T * r
            // jt is CSC, so outer_iterator iterates over columns of J^T
            // J^T * r [i] = sum_j J^T[i,j] * r[j]
            // For CSC column j, we add J^T[i,j] * r[j] to result[i] for each non-zero at row i
            self.jtr.fill(T::zero());
            for (col_j, col) in jt.outer_iterator().enumerate() {
                let r_j = self.residuals[col_j];
                for (row_i, &val) in col.iter() {
                    self.jtr[row_i] = self.jtr[row_i] + val * r_j;
                }
            }

            let gradient_norm = self.jtr.norm();

            // Add damping: JtJ[i,i] += lambda * max(JtJ[i,i], 1.0)
            let jtj_damped = add_damping(&jtj, lambda);

            // Sparse LDL factorization and solve
            let ldl_result = Ldl::new()
                .check_symmetry(SymmetryCheck::DontCheckSymmetry)
                .numeric(jtj_damped.view());

            let ldl = match ldl_result {
                Ok(ldl) => ldl,
                Err(e) => {
                    if self.verbose {
                        println!("❌ LDL factorization failed: {:?}", e);
                    }
                    lambda = lambda * self.lambda_scale_up;
                    continue;
                }
            };

            // Solve the system
            let jtr_vec: Vec<T> = self.jtr.iter().cloned().collect();
            let step_vec: Vec<T> = ldl.solve(&jtr_vec);
            let step = DVector::from_vec(step_vec);

            // Try the step
            let new_params = &params - &step;

            // Evaluate at new params
            self.temp_residuals.fill(T::zero());
            cost_fn(
                &new_params,
                self.temp_residuals.as_mut_slice(),
                self.jacobian.data_mut(),
            );
            let new_error = self.temp_residuals.norm();

            let step_norm = step.norm();
            let converged = step_norm < self.tolerance;

            // Accept or reject step
            if new_error < error {
                params = new_params;
                lambda = lambda * self.lambda_scale_down;

                let result = IterationResult {
                    error,
                    step_norm,
                    lambda,
                    gradient_norm,
                    converged,
                };

                callback(iteration, &result, &params);

                if converged {
                    if self.verbose {
                        println!("✅ Converged after {} iterations", iteration + 1);
                    }
                    break;
                }
            } else {
                lambda = lambda * self.lambda_scale_up;

                let gradient_threshold = T::from(1e-6).unwrap();
                if gradient_norm < gradient_threshold {
                    if self.verbose {
                        println!(
                            "✅ Local minimum detected at iteration {} (gradient={})",
                            iteration, gradient_norm
                        );
                    }
                    break;
                }

                if !lambda.is_finite() || lambda > T::from(1e12).unwrap() {
                    if self.verbose {
                        println!(
                            "❌ Lambda diverged at iteration {} (lambda={}, gradient={})",
                            iteration, lambda, gradient_norm
                        );
                    }
                    break;
                }

                let result = IterationResult {
                    error,
                    step_norm,
                    lambda,
                    gradient_norm,
                    converged: false,
                };

                callback(iteration, &result, &params);
            }
        }

        params
    }
}

/// Add Levenberg-Marquardt damping to diagonal
fn add_damping<T>(jtj: &CsMat<T>, lambda: T) -> CsMat<T>
where
    T: num_traits::Float + Default + Clone,
{
    let n = jtj.cols();
    let mut tri = TriMat::new((n, n));

    // Copy existing entries, modifying diagonal
    let indptr_storage = jtj.indptr();
    let indptr: &[usize] = indptr_storage.as_slice().unwrap();
    for col in 0..n {
        for idx in indptr[col]..indptr[col + 1] {
            let row = jtj.indices()[idx];
            let mut val = jtj.data()[idx];
            if row == col {
                val = val + lambda * T::max(val, T::one());
            }
            tri.add_triplet(row, col, val);
        }
    }

    tri.to_csc()
}

/// Build the list of (row, col) entries for a SLAM-style Jacobian in row-major order.
///
/// # Arguments
/// * `observations` - Vec of (residual_start, pose_param_start, point_param_start, has_pose, has_point)
///   Each observation contributes 4 residuals (stereo).
///
/// Returns entries sorted by (row, col) suitable for CSR construction.
pub fn build_slam_entries(
    observations: &[(usize, usize, usize, bool, bool)],
) -> Vec<(usize, usize)> {
    let mut entries = Vec::new();

    for &(residual_start, pose_start, point_start, has_pose, has_point) in observations {
        for res_offset in 0..4 {
            let row = residual_start + res_offset;

            // Pose parameters first (lower column indices)
            if has_pose {
                for pose_offset in 0..6 {
                    entries.push((row, pose_start + pose_offset));
                }
            }

            // Point parameters second (higher column indices)
            if has_point {
                for point_offset in 0..3 {
                    entries.push((row, point_start + point_offset));
                }
            }
        }
    }

    // Sort by (row, col) for CSR format
    entries.sort_by_key(|&(row, col)| (row, col));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_slam_entries() {
        // Simple test: 2 observations
        let observations = vec![
            (0, 0, 6, true, true), // residual 0-3, pose 0-5, point 6-8
            (4, 0, 9, true, true), // residual 4-7, pose 0-5, point 9-11
        ];

        let entries = build_slam_entries(&observations);

        // Each observation: 4 residuals × (6 pose + 3 point) = 36 entries
        assert_eq!(entries.len(), 72);

        // Check sorting: entries should be in (row, col) order
        for i in 1..entries.len() {
            assert!(entries[i - 1] <= entries[i]);
        }

        // Check first row has pose params 0-5, then point params 6-8
        let row0_entries: Vec<_> = entries.iter().filter(|&&(r, _)| r == 0).collect();
        assert_eq!(row0_entries.len(), 9);
        assert_eq!(row0_entries[0], &(0, 0));
        assert_eq!(row0_entries[5], &(0, 5));
        assert_eq!(row0_entries[6], &(0, 6));
        assert_eq!(row0_entries[8], &(0, 8));
    }

    #[test]
    fn test_sparse_solver_simple() {
        // Simple linear regression: y = a*x + b
        // Data: y = 2x + 1
        let data = [
            (1.0_f64, 3.0),
            (2.0, 5.0),
            (3.0, 7.0),
            (4.0, 9.0),
            (5.0, 11.0),
        ];

        // Each residual depends on both params (a and b)
        let entries: Vec<_> = (0..5).flat_map(|i| vec![(i, 0), (i, 1)]).collect();

        let mut solver = SparseLevenbergMarquardt::<f64>::new(5, 2, &entries).with_verbose(false);

        let initial = DVector::from_vec(vec![0.0, 0.0]);

        let cost_fn = |params: &DVector<f64>, residuals: &mut [f64], jacobian: &mut [f64]| {
            let a = params[0];
            let b = params[1];

            for (i, &(x, y_true)) in data.iter().enumerate() {
                let y_pred = a * x + b;
                residuals[i] = y_pred - y_true;

                // Jacobian: d(residual)/da = x, d(residual)/db = 1
                // Stored in row-major order: (i,0), (i,1)
                jacobian[i * 2 + 0] = x;
                jacobian[i * 2 + 1] = 1.0;
            }
        };

        let result = solver.solve(initial, cost_fn, |_, _, _| {});

        // Should find a ≈ 2, b ≈ 1
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_fixed_entries() {
        // Test with some fixed parameters (no jacobian entries)
        let observations = vec![
            (0, 0, 6, true, true),   // Both active
            (4, 0, 9, false, true),  // Only point active
            (8, 0, 12, true, false), // Only pose active
        ];

        let entries = build_slam_entries(&observations);

        // Obs 0: 4 × 9 = 36
        // Obs 1: 4 × 3 = 12 (no pose)
        // Obs 2: 4 × 6 = 24 (no point)
        assert_eq!(entries.len(), 72);
    }
}
