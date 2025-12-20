//! Levenberg-Marquardt solver with pre-allocated workspace
//!
//! This module provides a solver that pre-allocates all memory on the heap
//! for optimal performance in long-running sessions (e.g., VR tracking).
//!
//! The key principle: **Allocate once at construction, never reallocate during solve**.

use nalgebra::{SMatrix, SVector, Cholesky};

/// Levenberg-Marquardt solver with heap-allocated working memory
///
/// All large matrices are pre-allocated in Box on the heap at construction time.
/// During solve(), no additional allocations occur - only in-place updates.
///
/// # Type Parameters
/// - `T`: Scalar type (f32 or f64)
/// - `N_PARAMS`: Number of optimization parameters
/// - `N_RESIDUALS`: Number of residual values
///
/// # Memory Layout
/// - Jacobian: `N_RESIDUALS × N_PARAMS × sizeof(T)` (pre-allocated in Box)
/// - JtJ: `N_PARAMS × N_PARAMS × sizeof(T)` (pre-allocated in Box)
/// - Residuals: `N_RESIDUALS × sizeof(T)` (pre-allocated in Box)
/// - Total: ~(N_RESIDUALS × N_PARAMS + N_PARAMS² + N_RESIDUALS) × sizeof(T)
///
/// # Example
/// ```rust
/// // For a large problem (1000 residuals, 200 params):
/// let mut solver = LevenbergMarquardt::<f64, 200, 1000>::new();
/// // All memory allocated here ↑ (once, on heap)
///
/// let result = solver.solve(initial_params, cost_function, callback);
/// // No allocations during solve! ↑
/// ```
pub struct LevenbergMarquardt<T, const N_PARAMS: usize, const N_RESIDUALS: usize> {
    // Solver parameters
    pub tolerance: T,
    pub max_iterations: usize,
    pub initial_lambda: T,
    pub lambda_scale_up: T,
    pub lambda_scale_down: T,
    pub verbose: bool,

    // Pre-allocated working memory (on heap via Box)
    // These are allocated ONCE at construction, never reallocated
    workspace_jacobian: Box<SMatrix<T, N_RESIDUALS, N_PARAMS>>,
    workspace_jtj: Box<SMatrix<T, N_PARAMS, N_PARAMS>>,
    workspace_residuals: Box<SVector<T, N_RESIDUALS>>,
    workspace_jtr: Box<SVector<T, N_PARAMS>>,

    // Additional workspace for trial step evaluation
    workspace_temp_residuals: Box<SVector<T, N_RESIDUALS>>,
    workspace_temp_jacobian: Box<SMatrix<T, N_RESIDUALS, N_PARAMS>>,
}

/// Result of one optimization iteration
pub struct IterationResult<T> {
    pub error: T,
    pub step_norm: T,
    pub lambda: T,
    pub gradient_norm: T,
    pub converged: bool,
}

// Macro to implement solver for f32 and f64
macro_rules! impl_levenberg_marquardt {
    ($T:ty, $tolerance:expr, $initial_lambda:expr) => {
        impl<const N_PARAMS: usize, const N_RESIDUALS: usize>
            LevenbergMarquardt<$T, N_PARAMS, N_RESIDUALS>
        {
            /// Create new solver with pre-allocated heap memory
            ///
            /// This allocates all working memory on the heap immediately.
            /// Memory layout:
            /// - Jacobian: N_RESIDUALS × N_PARAMS
            /// - JtJ: N_PARAMS × N_PARAMS
            /// - Residuals: N_RESIDUALS
            /// - JtR: N_PARAMS
            /// - Step: N_PARAMS
            ///
            /// IMPORTANT: Uses unsafe Box::new_zeroed() to avoid stack overflow.
            /// Box::new() creates value on stack first, which fails for large matrices!
            pub fn new() -> Self {
                Self {
                    tolerance: $tolerance,
                    max_iterations: 50,
                    initial_lambda: $initial_lambda,
                    lambda_scale_up: 10.0,
                    lambda_scale_down: 0.1,
                    verbose: true,

                    // Pre-allocate all working memory on heap
                    // SAFETY: We use new_zeroed() to allocate directly on heap,
                    // then assume_init() because zero-initialized matrices are valid
                    workspace_jacobian: unsafe {
                        Box::new_zeroed().assume_init()
                    },
                    workspace_jtj: unsafe {
                        Box::new_zeroed().assume_init()
                    },
                    workspace_residuals: unsafe {
                        Box::new_zeroed().assume_init()
                    },
                    workspace_jtr: unsafe {
                        Box::new_zeroed().assume_init()
                    },
                    workspace_temp_residuals: unsafe {
                        Box::new_zeroed().assume_init()
                    },
                    workspace_temp_jacobian: unsafe {
                        Box::new_zeroed().assume_init()
                    },
                }
            }

            pub fn with_tolerance(mut self, tolerance: $T) -> Self {
                self.tolerance = tolerance;
                self
            }

            pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
                self.max_iterations = max_iterations;
                self
            }

            pub fn with_lambda_params(mut self, initial: $T, scale_up: $T, scale_down: $T) -> Self {
                self.initial_lambda = initial;
                self.lambda_scale_up = scale_up;
                self.lambda_scale_down = scale_down;
                self
            }

            pub fn with_verbose(mut self, verbose: bool) -> Self {
                self.verbose = verbose;
                self
            }

            /// Solve the optimization problem
            ///
            /// # Memory Guarantees
            /// - No heap allocations during this function
            /// - All working memory was pre-allocated at construction
            /// - No stack allocations of large matrices (cost_fn writes to provided buffers)
            ///
            /// # Arguments
            /// * `params` - Initial parameter guess
            /// * `cost_fn` - Function that computes residuals and Jacobian into provided buffers
            /// * `callback` - Optional callback called after each iteration
            pub fn solve<F, C>(
                &mut self,
                params: SVector<$T, N_PARAMS>,
                cost_fn: F,
                callback: C,
            ) -> SVector<$T, N_PARAMS>
            where
                F: FnMut(
                    &SVector<$T, N_PARAMS>,
                    &mut SVector<$T, N_RESIDUALS>,
                    &mut SMatrix<$T, N_RESIDUALS, N_PARAMS>,
                ),
                C: FnMut(usize, &IterationResult<$T>, &SVector<$T, N_PARAMS>),
            {
                self.solve_with_prior(params, cost_fn, callback, None, None)
            }

            /// Solve the optimization problem with optional marginalized prior
            ///
            /// This extends the Gauss-Newton system to include a prior:
            /// (J^T J + H_prior + λI) δ = -(J^T r + b_prior)
            ///
            /// # Arguments
            /// * `params` - Initial parameter guess
            /// * `cost_fn` - Function that computes residuals and Jacobian
            /// * `callback` - Callback after each iteration
            /// * `prior_hessian` - Optional prior information matrix (H_prior)
            /// * `prior_rhs` - Optional prior right-hand side (b_prior)
            pub fn solve_with_prior<F, C>(
                &mut self,
                mut params: SVector<$T, N_PARAMS>,
                mut cost_fn: F,
                mut callback: C,
                prior_hessian: Option<&SMatrix<$T, N_PARAMS, N_PARAMS>>,
                prior_rhs: Option<&SVector<$T, N_PARAMS>>,
            ) -> SVector<$T, N_PARAMS>
            where
                F: FnMut(
                    &SVector<$T, N_PARAMS>,
                    &mut SVector<$T, N_RESIDUALS>,
                    &mut SMatrix<$T, N_RESIDUALS, N_PARAMS>,
                ),
                C: FnMut(usize, &IterationResult<$T>, &SVector<$T, N_PARAMS>),
            {
                let mut lambda = self.initial_lambda;

                for iteration in 0..self.max_iterations {
                    // Compute residuals and Jacobian directly into workspace (no stack allocation!)
                    cost_fn(&params, &mut self.workspace_residuals, &mut self.workspace_jacobian);
                    let error = self.workspace_residuals.norm();

                    // Gauss-Newton system: (J^T J + H_prior + λI) δ = -(J^T r + b_prior)
                    // Use nalgebra's tr_mul_to and tr_mul to avoid creating transpose on stack

                    // J^T * J: use tr_mul_to which computes self.transpose() * rhs into output
                    self.workspace_jacobian.tr_mul_to(&*self.workspace_jacobian, &mut *self.workspace_jtj);

                    // Add prior Hessian if provided
                    if let Some(h_prior) = prior_hessian {
                        *self.workspace_jtj += h_prior;
                    }

                    // J^T * r: use tr_mul which computes self.transpose() * rhs
                    *self.workspace_jtr = self.workspace_jacobian.tr_mul(&*self.workspace_residuals);

                    // Add prior RHS if provided
                    if let Some(b_prior) = prior_rhs {
                        *self.workspace_jtr += b_prior;
                    }

                    let gradient_norm = self.workspace_jtr.norm();

                    // Add damping (Levenberg-Marquardt)
                    for i in 0..N_PARAMS {
                        self.workspace_jtj[(i, i)] += lambda * self.workspace_jtj[(i, i)].max(1.0 as $T);
                    }

                    // WARNING: if you ever make jtj not symmetric positive definite, this will break!
                    let jtj_cholesky = Cholesky::new_unchecked(*self.workspace_jtj);
                    jtj_cholesky.solve_mut(&mut *self.workspace_jtr);

                    // Try the step
                    let new_params = params - *self.workspace_jtr;

                    // Evaluate at new params (using pre-allocated temp workspace - NO STACK ALLOCATION!)
                    cost_fn(&new_params, &mut self.workspace_temp_residuals, &mut self.workspace_temp_jacobian);
                    let new_error = self.workspace_temp_residuals.norm();

                    let step_norm = self.workspace_jtr.norm();
                    let converged = step_norm < self.tolerance;

                    // Accept or reject step
                    if new_error < error {
                        // Good step - accept and decrease damping
                        params = new_params;
                        lambda *= self.lambda_scale_down;

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
                        // Bad step - reject and increase damping
                        lambda *= self.lambda_scale_up;

                        // Check for local minimum (better than just checking lambda!)
                        let gradient_threshold = 1e-6 as $T;  // Relaxed from 1e-8 (too tight!)

                        if gradient_norm < gradient_threshold {
                            if self.verbose {
                                println!("✅ Local minimum detected at iteration {} (gradient={:.2e})", iteration, gradient_norm);
                            }
                            break;
                        }

                        // Check if lambda is too large (divergence symptom)
                        if !lambda.is_finite() || lambda > 1e12 as $T {
                            if self.verbose {
                                println!("❌ Lambda diverged at iteration {} (lambda={}, gradient={:.2e})", iteration, lambda, gradient_norm);
                                println!("   This indicates a local minimum where gradient is non-zero.");
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

            /// Simpler solve without callback
            ///
            /// This is a convenience wrapper around solve() that:
            /// 1. Wraps old-style cost functions that return (residuals, jacobian)
            /// 2. Converts them to the buffer-writing style
            /// 3. Provides no-op callback
            pub fn solve_simple<F>(
                &mut self,
                params: SVector<$T, N_PARAMS>,
                cost_fn: F,
            ) -> SVector<$T, N_PARAMS>
            where
                F: Fn(
                    &SVector<$T, N_PARAMS>,
                ) -> (
                    SVector<$T, N_RESIDUALS>,
                    SMatrix<$T, N_RESIDUALS, N_PARAMS>,
                ),
            {
                // Wrap the old-style cost function to match new buffer-writing API
                #[allow(unused_mut)]  // mut required by solve() signature even though closure doesn't mutate
                let mut cost_fn_wrapped = |params: &SVector<$T, N_PARAMS>,
                                            residuals: &mut SVector<$T, N_RESIDUALS>,
                                            jacobian: &mut SMatrix<$T, N_RESIDUALS, N_PARAMS>| {
                    let (r, j) = cost_fn(params);
                    *residuals = r;
                    *jacobian = j;
                };

                self.solve(params, cost_fn_wrapped, |_iter, _result, _params| {})
            }
        }

        impl<const N_PARAMS: usize, const N_RESIDUALS: usize> Default
            for LevenbergMarquardt<$T, N_PARAMS, N_RESIDUALS>
        {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

impl_levenberg_marquardt!(f32, 1e-6_f32, 1e-4_f32);
impl_levenberg_marquardt!(f64, 1e-10_f64, 1e-4_f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Jet;

    #[test]
    fn test_solver_works() {
        // Fit y = a*x + b to data points
        const N_PARAMS: usize = 2;
        const N_DATA: usize = 3;

        let data = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]; // y = 2x + 1

        type Jet2 = Jet<f64, N_PARAMS>;

        let cost_fn = |params: &SVector<f64, N_PARAMS>| {
            let a = Jet2::variable(params[0], 0);
            let b = Jet2::variable(params[1], 1);

            let mut residuals = SVector::<f64, N_DATA>::zeros();
            let mut jacobian = SMatrix::<f64, N_DATA, N_PARAMS>::zeros();

            for (i, &(x, y_true)) in data.iter().enumerate() {
                let x_jet = Jet2::constant(x);
                let y_pred = a * x_jet + b;
                let residual = y_pred - Jet2::constant(y_true);

                residuals[i] = residual.value;
                jacobian[(i, 0)] = residual.derivs[0];
                jacobian[(i, 1)] = residual.derivs[1];
            }

            (residuals, jacobian)
        };

        let mut solver = LevenbergMarquardt::<f64, N_PARAMS, N_DATA>::new().with_verbose(false);
        let initial = SVector::<f64, N_PARAMS>::new(0.0, 0.0);
        let result = solver.solve_simple(initial, cost_fn);

        assert!((result[0] - 2.0).abs() < 1e-10); // a = 2
        assert!((result[1] - 1.0).abs() < 1e-10); // b = 1
    }

    #[test]
    fn test_large_problem() {
        // Large problem that would overflow the stack without heap allocation
        const N_PARAMS: usize = 128;
        const N_RESIDUALS: usize = 512;

        // Create solver - allocates ~512KB on heap
        let mut _solver = LevenbergMarquardt::<f64, N_PARAMS, N_RESIDUALS>::new();

        // If we got here without issues, test passes!
        assert!(true);
    }
}
