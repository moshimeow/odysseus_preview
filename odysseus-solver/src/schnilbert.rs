//! Schnilbert: A tiny dense Gauss-Newton solver for KLT-style problems
//!
//! This solver is designed for problems where:
//! - Many residuals (e.g., ~900 pixels in a 31x31 window)
//! - Very few parameters (2 for translation, 3 for SE2)
//! - J^T*J is tiny (2x2 or 3x3)
//!
//! Unlike LevenbergMarquardt, this solver:
//! - Uses direct inversion for the tiny normal equations
//! - Computes the Jacobian once (linearized approach, like classic Lucas-Kanade)
//! - Has minimal overhead - no lambda tuning, no line search

/// Result of solving with Schnilbert
#[derive(Debug, Clone)]
pub struct SchnilbertResult<T, const N: usize> {
    /// The optimized parameters
    pub params: [T; N],
    /// Final sum of squared residuals
    pub residual_norm_sq: T,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the solver converged
    pub converged: bool,
}

/// A tiny dense Gauss-Newton solver for KLT-style problems
///
/// # Type Parameters
/// - `T`: Scalar type (f32 or f64)
/// - `N`: Number of parameters (typically 2 or 3)
///
/// # Example
/// ```ignore
/// use odysseus_solver::schnilbert::Schnilbert;
/// use odysseus_solver::Jet;
///
/// // 2-parameter translation problem
/// let solver = Schnilbert::<f64, 2>::new();
///
/// let result = solver.solve(
///     [0.0, 0.0],  // initial guess
///     |params| {
///         // Compute residuals using Jets for autodiff
///         let dx = Jet::<f64, 2>::variable(params[0], 0);
///         let dy = Jet::<f64, 2>::variable(params[1], 1);
///
///         // Return vector of (residual_value, [jacobian_row])
///         vec![
///             (some_residual_1.value, some_residual_1.derivs),
///             (some_residual_2.value, some_residual_2.derivs),
///             // ...
///         ]
///     },
/// );
/// ```
#[derive(Debug, Clone)]
pub struct Schnilbert<T, const N: usize> {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold for step norm
    pub step_threshold: T,
    /// Minimum pivot value for rejecting degenerate solutions
    pub min_pivot: T,
}

// Implement for f32
impl<const N: usize> Schnilbert<f32, N> {
    pub fn new() -> Self {
        Self {
            max_iterations: 10,
            step_threshold: 1e-4,
            min_pivot: 1e-6,
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_step_threshold(mut self, threshold: f32) -> Self {
        self.step_threshold = threshold;
        self
    }

    pub fn with_min_pivot(mut self, threshold: f32) -> Self {
        self.min_pivot = threshold;
        self
    }

    /// Solve a tiny optimization problem using linearized Gauss-Newton
    ///
    /// The residual function should return a vector of (residual, jacobian_row) pairs.
    /// The Jacobian is computed once using forward-mode autodiff (Jets).
    ///
    /// # Arguments
    /// * `initial` - Initial parameter guess
    /// * `residual_fn` - Function that computes residuals and their Jacobians
    ///
    /// # Returns
    /// A `SchnilbertResult` containing the optimized parameters and convergence info
    pub fn solve<F>(&self, initial: [f32; N], residual_fn: F) -> SchnilbertResult<f32, N>
    where
        F: Fn(&[f32; N]) -> Vec<(f32, [f32; N])>,
    {
        let mut params = initial;

        for iteration in 0..self.max_iterations {
            // Compute residuals and Jacobian
            let residuals = residual_fn(&params);

            if residuals.is_empty() {
                return SchnilbertResult {
                    params,
                    residual_norm_sq: 0.0,
                    iterations: iteration,
                    converged: true,
                };
            }

            // Build J^T * J and J^T * r
            let mut jtj = [[0.0f32; N]; N];
            let mut jtr = [0.0f32; N];
            let mut residual_norm_sq = 0.0f32;

            for (r, j) in &residuals {
                residual_norm_sq += r * r;

                // J^T * r
                for i in 0..N {
                    jtr[i] += j[i] * r;
                }

                // J^T * J (symmetric, but we fill the whole thing for simplicity)
                for i in 0..N {
                    for k in 0..N {
                        jtj[i][k] += j[i] * j[k];
                    }
                }
            }

            // Solve the normal equations: (J^T J) * delta = -J^T r
            let delta = solve_nxn_f32::<N>(jtj, jtr, self.min_pivot);

            let Some(delta) = delta else {
                // Degenerate system - return current params
                return SchnilbertResult {
                    params,
                    residual_norm_sq,
                    iterations: iteration,
                    converged: false,
                };
            };

            // Update parameters
            let mut step_norm_sq = 0.0f32;
            for i in 0..N {
                params[i] -= delta[i];
                step_norm_sq += delta[i] * delta[i];
            }

            // Check convergence
            if step_norm_sq.sqrt() < self.step_threshold {
                // Recompute residual at final params
                let final_residuals = residual_fn(&params);
                let final_norm_sq: f32 = final_residuals.iter().map(|(r, _)| r * r).sum();

                return SchnilbertResult {
                    params,
                    residual_norm_sq: final_norm_sq,
                    iterations: iteration + 1,
                    converged: true,
                };
            }
        }

        // Didn't converge - compute final residual
        let final_residuals = residual_fn(&params);
        let final_norm_sq: f32 = final_residuals.iter().map(|(r, _)| r * r).sum();

        SchnilbertResult {
            params,
            residual_norm_sq: final_norm_sq,
            iterations: self.max_iterations,
            converged: false,
        }
    }

    /// Solve with a pre-linearized problem (Jacobian computed once)
    ///
    /// This is even more efficient when you can compute the Jacobian once
    /// and reuse it across iterations (like in classic Lucas-Kanade).
    ///
    /// # Arguments
    /// * `initial` - Initial parameter guess
    /// * `residual_fn` - Function that computes just the residuals (no Jacobian)
    /// * `jacobian` - Pre-computed Jacobian rows (one per residual)
    pub fn solve_linearized<F>(
        &self,
        initial: [f32; N],
        mut residual_fn: F,
        jacobian: &[[f32; N]],
    ) -> SchnilbertResult<f32, N>
    where
        F: FnMut(&[f32; N]) -> Vec<f32>,
    {
        let mut params = initial;

        // Pre-compute J^T * J (doesn't change across iterations)
        let mut jtj = [[0.0f32; N]; N];
        for j in jacobian {
            for i in 0..N {
                for k in 0..N {
                    jtj[i][k] += j[i] * j[k];
                }
            }
        }

        for iteration in 0..self.max_iterations {
            // Compute residuals only
            let residuals = residual_fn(&params);

            if residuals.len() != jacobian.len() {
                panic!(
                    "Residual count {} doesn't match Jacobian rows {}",
                    residuals.len(),
                    jacobian.len()
                );
            }

            // Build J^T * r
            let mut jtr = [0.0f32; N];
            let mut residual_norm_sq = 0.0f32;

            for (r, j) in residuals.iter().zip(jacobian.iter()) {
                residual_norm_sq += r * r;
                for i in 0..N {
                    jtr[i] += j[i] * r;
                }
            }

            // Solve the normal equations
            let delta = solve_nxn_f32::<N>(jtj, jtr, self.min_pivot);

            let Some(delta) = delta else {
                return SchnilbertResult {
                    params,
                    residual_norm_sq,
                    iterations: iteration,
                    converged: false,
                };
            };

            // Update parameters
            let mut step_norm_sq = 0.0f32;
            for i in 0..N {
                params[i] -= delta[i];
                step_norm_sq += delta[i] * delta[i];
            }

            // Check convergence
            if step_norm_sq.sqrt() < self.step_threshold {
                let final_residuals = residual_fn(&params);
                let final_norm_sq: f32 = final_residuals.iter().map(|r| r * r).sum();

                return SchnilbertResult {
                    params,
                    residual_norm_sq: final_norm_sq,
                    iterations: iteration + 1,
                    converged: true,
                };
            }
        }

        let final_residuals = residual_fn(&params);
        let final_norm_sq: f32 = final_residuals.iter().map(|r| r * r).sum();

        SchnilbertResult {
            params,
            residual_norm_sq: final_norm_sq,
            iterations: self.max_iterations,
            converged: false,
        }
    }
}

// Implement for f64
impl<const N: usize> Schnilbert<f64, N> {
    pub fn new() -> Self {
        Self {
            max_iterations: 10,
            step_threshold: 1e-6,
            min_pivot: 1e-10,
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_step_threshold(mut self, threshold: f64) -> Self {
        self.step_threshold = threshold;
        self
    }

    pub fn with_min_pivot(mut self, threshold: f64) -> Self {
        self.min_pivot = threshold;
        self
    }

    pub fn solve<F>(&self, initial: [f64; N], residual_fn: F) -> SchnilbertResult<f64, N>
    where
        F: Fn(&[f64; N]) -> Vec<(f64, [f64; N])>,
    {
        let mut params = initial;

        for iteration in 0..self.max_iterations {
            let residuals = residual_fn(&params);

            if residuals.is_empty() {
                return SchnilbertResult {
                    params,
                    residual_norm_sq: 0.0,
                    iterations: iteration,
                    converged: true,
                };
            }

            let mut jtj = [[0.0f64; N]; N];
            let mut jtr = [0.0f64; N];
            let mut residual_norm_sq = 0.0f64;

            for (r, j) in &residuals {
                residual_norm_sq += r * r;

                for i in 0..N {
                    jtr[i] += j[i] * r;
                }

                for i in 0..N {
                    for k in 0..N {
                        jtj[i][k] += j[i] * j[k];
                    }
                }
            }

            let delta = solve_nxn_f64::<N>(jtj, jtr, self.min_pivot);

            let Some(delta) = delta else {
                return SchnilbertResult {
                    params,
                    residual_norm_sq,
                    iterations: iteration,
                    converged: false,
                };
            };

            let mut step_norm_sq = 0.0f64;
            for i in 0..N {
                params[i] -= delta[i];
                step_norm_sq += delta[i] * delta[i];
            }

            if step_norm_sq.sqrt() < self.step_threshold {
                let final_residuals = residual_fn(&params);
                let final_norm_sq: f64 = final_residuals.iter().map(|(r, _)| r * r).sum();

                return SchnilbertResult {
                    params,
                    residual_norm_sq: final_norm_sq,
                    iterations: iteration + 1,
                    converged: true,
                };
            }
        }

        let final_residuals = residual_fn(&params);
        let final_norm_sq: f64 = final_residuals.iter().map(|(r, _)| r * r).sum();

        SchnilbertResult {
            params,
            residual_norm_sq: final_norm_sq,
            iterations: self.max_iterations,
            converged: false,
        }
    }

    pub fn solve_linearized<F>(
        &self,
        initial: [f64; N],
        mut residual_fn: F,
        jacobian: &[[f64; N]],
    ) -> SchnilbertResult<f64, N>
    where
        F: FnMut(&[f64; N]) -> Vec<f64>,
    {
        let mut params = initial;

        let mut jtj = [[0.0f64; N]; N];
        for j in jacobian {
            for i in 0..N {
                for k in 0..N {
                    jtj[i][k] += j[i] * j[k];
                }
            }
        }

        for iteration in 0..self.max_iterations {
            let residuals = residual_fn(&params);

            if residuals.len() != jacobian.len() {
                panic!(
                    "Residual count {} doesn't match Jacobian rows {}",
                    residuals.len(),
                    jacobian.len()
                );
            }

            let mut jtr = [0.0f64; N];
            let mut residual_norm_sq = 0.0f64;

            for (r, j) in residuals.iter().zip(jacobian.iter()) {
                residual_norm_sq += r * r;
                for i in 0..N {
                    jtr[i] += j[i] * r;
                }
            }

            let delta = solve_nxn_f64::<N>(jtj, jtr, self.min_pivot);

            let Some(delta) = delta else {
                return SchnilbertResult {
                    params,
                    residual_norm_sq,
                    iterations: iteration,
                    converged: false,
                };
            };

            let mut step_norm_sq = 0.0f64;
            for i in 0..N {
                params[i] -= delta[i];
                step_norm_sq += delta[i] * delta[i];
            }

            if step_norm_sq.sqrt() < self.step_threshold {
                let final_residuals = residual_fn(&params);
                let final_norm_sq: f64 = final_residuals.iter().map(|r| r * r).sum();

                return SchnilbertResult {
                    params,
                    residual_norm_sq: final_norm_sq,
                    iterations: iteration + 1,
                    converged: true,
                };
            }
        }

        let final_residuals = residual_fn(&params);
        let final_norm_sq: f64 = final_residuals.iter().map(|r| r * r).sum();

        SchnilbertResult {
            params,
            residual_norm_sq: final_norm_sq,
            iterations: self.max_iterations,
            converged: false,
        }
    }
}

impl<const N: usize> Default for Schnilbert<f32, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Default for Schnilbert<f64, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Direct solvers for tiny systems using Gaussian elimination
// ============================================================================

/// Solve an NxN system using Gaussian elimination with partial pivoting
/// For tiny N (2 or 3), this is still very fast and avoids code duplication
fn solve_nxn_f32<const N: usize>(
    mut a: [[f32; N]; N],
    mut b: [f32; N],
    min_pivot: f32,
) -> Option<[f32; N]> {
    // Forward elimination with partial pivoting
    for col in 0..N {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..N {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }

        if max_val < min_pivot {
            return None;
        }

        // Swap rows
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        // Eliminate
        let pivot = a[col][col];
        for row in (col + 1)..N {
            let factor = a[row][col] / pivot;
            a[row][col] = 0.0;
            for k in (col + 1)..N {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = [0.0f32; N];
    for i in (0..N).rev() {
        let mut sum = b[i];
        for j in (i + 1)..N {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    Some(x)
}

fn solve_nxn_f64<const N: usize>(
    mut a: [[f64; N]; N],
    mut b: [f64; N],
    min_pivot: f64,
) -> Option<[f64; N]> {
    for col in 0..N {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..N {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }

        if max_val < min_pivot {
            return None;
        }

        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        let pivot = a[col][col];
        for row in (col + 1)..N {
            let factor = a[row][col] / pivot;
            a[row][col] = 0.0;
            for k in (col + 1)..N {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0f64; N];
    for i in (0..N).rev() {
        let mut sum = b[i];
        for j in (i + 1)..N {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    Some(x)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Jet;

    #[test]
    fn test_solve_2x2_identity() {
        // Solve I * x = [3, 4]
        let a = [[1.0, 0.0], [0.0, 1.0]];
        let b = [3.0, 4.0];
        let x = solve_nxn_f64::<2>(a, b, 1e-10).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_2x2_general() {
        // Solve [[2, 1], [1, 3]] * x = [5, 5]
        // Solution: x = [2, 1]
        let a = [[2.0, 1.0], [1.0, 3.0]];
        let b = [5.0, 5.0];
        let x = solve_nxn_f64::<2>(a, b, 1e-10).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_3x3_identity() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [1.0, 2.0, 3.0];
        let x = solve_nxn_f64::<3>(a, b, 1e-10).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_little_guy_linear_fit() {
        // Fit y = a*x + b to points (1,3), (2,5), (3,7) -> a=2, b=1
        let data = [(1.0f64, 3.0), (2.0, 5.0), (3.0, 7.0)];

        let solver = Schnilbert::<f64, 2>::new();

        let result = solver.solve([0.0, 0.0], |params| {
            let a = Jet::<f64, 2>::variable(params[0], 0);
            let b = Jet::<f64, 2>::variable(params[1], 1);

            data.iter()
                .map(|&(x, y)| {
                    let x_jet = Jet::<f64, 2>::constant(x);
                    let y_jet = Jet::<f64, 2>::constant(y);
                    let residual = a * x_jet + b - y_jet;
                    (residual.value, residual.derivs)
                })
                .collect()
        });

        assert!(result.converged);
        assert!((result.params[0] - 2.0).abs() < 1e-6); // a = 2
        assert!((result.params[1] - 1.0).abs() < 1e-6); // b = 1
    }

    #[test]
    fn test_little_guy_quadratic() {
        // Find minimum of f(x,y) = (x-3)^2 + (y-4)^2
        // Residuals: [x-3, y-4]
        let solver = Schnilbert::<f64, 2>::new();

        let result = solver.solve([0.0, 0.0], |params| {
            let x = Jet::<f64, 2>::variable(params[0], 0);
            let y = Jet::<f64, 2>::variable(params[1], 1);

            let r1 = x - Jet::<f64, 2>::constant(3.0);
            let r2 = y - Jet::<f64, 2>::constant(4.0);

            vec![(r1.value, r1.derivs), (r2.value, r2.derivs)]
        });

        assert!(result.converged);
        assert!((result.params[0] - 3.0).abs() < 1e-6);
        assert!((result.params[1] - 4.0).abs() < 1e-6);
        assert!(result.residual_norm_sq < 1e-10);
    }

    #[test]
    fn test_little_guy_linearized() {
        // Same linear fit but using the linearized interface
        let data = [(1.0f64, 3.0), (2.0, 5.0), (3.0, 7.0)];

        // Pre-compute Jacobian: d/da(a*x + b - y) = x, d/db(a*x + b - y) = 1
        let jacobian: Vec<[f64; 2]> = data.iter().map(|&(x, _)| [x, 1.0]).collect();

        let solver = Schnilbert::<f64, 2>::new();

        let result = solver.solve_linearized(
            [0.0, 0.0],
            |params| {
                data.iter()
                    .map(|&(x, y)| params[0] * x + params[1] - y)
                    .collect()
            },
            &jacobian,
        );

        assert!(result.converged);
        assert!((result.params[0] - 2.0).abs() < 1e-6);
        assert!((result.params[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f32_solver() {
        // Same test but with f32
        let data = [(1.0f32, 3.0), (2.0, 5.0), (3.0, 7.0)];

        let solver = Schnilbert::<f32, 2>::new();

        let result = solver.solve([0.0, 0.0], |params| {
            let a = Jet::<f32, 2>::variable(params[0], 0);
            let b = Jet::<f32, 2>::variable(params[1], 1);

            data.iter()
                .map(|&(x, y)| {
                    let x_jet = Jet::<f32, 2>::constant(x);
                    let y_jet = Jet::<f32, 2>::constant(y);
                    let residual = a * x_jet + b - y_jet;
                    (residual.value, residual.derivs)
                })
                .collect()
        });

        assert!(result.converged);
        assert!((result.params[0] - 2.0).abs() < 1e-4);
        assert!((result.params[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_degenerate_system() {
        // Singular system should return None
        let solver = Schnilbert::<f64, 2>::new();

        let result = solver.solve([0.0, 0.0], |_params| {
            // Two identical residuals -> rank-deficient J^T*J
            vec![(1.0, [1.0, 1.0]), (1.0, [1.0, 1.0])]
        });

        // Should fail due to singular J^T*J
        assert!(!result.converged);
    }

    #[test]
    fn test_3_parameter_problem() {
        // 3-parameter problem: fit y = a*x^2 + b*x + c
        // Points: (0,1), (1,2), (2,5), (3,10) -> a=1, b=0, c=1
        let data = [(0.0f64, 1.0), (1.0, 2.0), (2.0, 5.0), (3.0, 10.0)];

        let solver = Schnilbert::<f64, 3>::new();

        let result = solver.solve([0.0, 0.0, 0.0], |params| {
            let a = Jet::<f64, 3>::variable(params[0], 0);
            let b = Jet::<f64, 3>::variable(params[1], 1);
            let c = Jet::<f64, 3>::variable(params[2], 2);

            data.iter()
                .map(|&(x, y)| {
                    let x_jet = Jet::<f64, 3>::constant(x);
                    let y_jet = Jet::<f64, 3>::constant(y);
                    let residual = a * x_jet * x_jet + b * x_jet + c - y_jet;
                    (residual.value, residual.derivs)
                })
                .collect()
        });

        assert!(result.converged);
        assert!((result.params[0] - 1.0).abs() < 1e-6); // a = 1
        assert!((result.params[1] - 0.0).abs() < 1e-6); // b = 0
        assert!((result.params[2] - 1.0).abs() < 1e-6); // c = 1
    }
}
