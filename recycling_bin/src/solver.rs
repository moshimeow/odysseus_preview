use crate::jet::{JetArena, JetHandle, ParameterVector};
use crate::math_context::MathContext;

/// Trait for cost functors that compute residuals for optimization problems
///
/// Generic over `Ctx: MathContext` to support:
/// - `JetArena<f64>` for computing residuals with Jacobians (derivatives)
/// - `F64Ctx` for computing residuals without Jacobians (values only)
/// - `F32Ctx` for f32 precision
pub trait CostFunctor<Ctx: MathContext = JetArena<f64>> {
    /// Compute residuals for the given parameters
    /// Returns the number of residuals written to the ResidualWriter
    fn residuals(&self, params: &[Ctx::Value], ctx: &mut Ctx, writer: ResidualWriter<Ctx::Value>) -> Result<usize, String>;
}

/// Safe writer interface for filling residual buffers
pub struct ResidualWriter<'a, T> {
    buffer: &'a mut [T],
    count: usize,
}

impl<'a, T: Copy> ResidualWriter<'a, T> {
    pub(crate) fn new(buffer: &'a mut [T]) -> Self {
        Self { buffer, count: 0 }
    }

    /// Add a residual to the buffer. Returns error if buffer is full.
    pub fn push(&mut self, residual: T) -> Result<(), &'static str> {
        if self.count >= self.buffer.len() {
            return Err("Residual buffer full");
        }
        self.buffer[self.count] = residual;
        self.count += 1;
        Ok(())
    }

    /// Try to add a residual, panicking with helpful message if buffer is full
    pub fn push_unchecked(&mut self, residual: T) {
        if let Err(e) = self.push(residual) {
            panic!("ResidualWriter: {}", e);
        }
    }

    /// Get the number of residuals written so far
    pub fn count(&self) -> usize {
        self.count
    }

    /// Finish writing and return the final count
    pub fn finish(self) -> usize {
        self.count
    }
}

/// Simple non-linear least squares solver using Gauss-Newton or Levenberg-Marquardt method
pub struct TinySolver {
    max_iterations: usize,
    tolerance: f64,
    use_levenberg_marquardt: bool,
    initial_lambda: f64,
    lambda_scale_up: f64,
    lambda_scale_down: f64,
    arena_capacity: usize,
}

/// Pre-allocated working buffers for zero-allocation solving
pub struct TinySolverBuffers {
    pub max_params: usize,
    pub max_residuals: usize,
    // Pre-allocated working memory - reused every iteration (public for custom solvers)
    pub residual_handles: Vec<JetHandle>,
    pub residuals: Vec<f64>,
    pub jacobian: Vec<f64>,  // flattened: [residual0_derivs..., residual1_derivs...]
    pub jtj: Vec<f64>,       // flattened: symmetric matrix storage
    pub jtr: Vec<f64>,
    pub param_step: Vec<f64>,
    // Levenberg-Marquardt line search buffers
    pub candidate_params: Vec<f64>,
    pub candidate_residuals_f64: Vec<f64>,
}

impl Default for TinySolver {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-8,
            use_levenberg_marquardt: false,
            initial_lambda: 1e-3,
            lambda_scale_up: 10.0,
            lambda_scale_down: 0.1,
            arena_capacity: 32,  // Default capacity, can be increased for complex problems
        }
    }
}

impl TinySolverBuffers {
    /// Create buffers for problems with up to max_params parameters and max_residuals residuals
    pub fn new(max_params: usize, max_residuals: usize) -> Self {
        Self {
            max_params,
            max_residuals,
            // Allocate once, reuse forever
            residual_handles: vec![JetHandle::from_index(0); max_residuals],
            residuals: vec![0.0; max_residuals],
            jacobian: vec![0.0; max_residuals * max_params],  // flattened
            jtj: vec![0.0; max_params * max_params],          // flattened symmetric
            jtr: vec![0.0; max_params],
            param_step: vec![0.0; max_params],
            candidate_params: vec![0.0; max_params],
            candidate_residuals_f64: vec![0.0; max_residuals],
        }
    }

    /// Check if buffers are large enough for this problem size
    pub fn can_handle(&self, n_params: usize, n_residuals: usize) -> bool {
        n_params <= self.max_params && n_residuals <= self.max_residuals
    }

    /// Get mutable slice for residual handles (up to max_residuals long)
    pub fn residual_handles_mut(&mut self) -> &mut [JetHandle] {
        &mut self.residual_handles
    }

    /// Create a ResidualWriter for safe residual buffer access
    pub fn residual_writer(&mut self) -> ResidualWriter<'_, JetHandle> {
        ResidualWriter::new(&mut self.residual_handles)
    }
}

impl TinySolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_levenberg_marquardt(mut self, enable: bool) -> Self {
        self.use_levenberg_marquardt = enable;
        self
    }

    pub fn with_lambda_params(mut self, initial: f64, scale_up: f64, scale_down: f64) -> Self {
        self.initial_lambda = initial;
        self.lambda_scale_up = scale_up;
        self.lambda_scale_down = scale_down;
        self
    }

    pub fn with_arena_capacity(mut self, capacity: usize) -> Self {
        self.arena_capacity = capacity;
        self
    }

    /// Perform a single iteration of optimization using a CostFunctor
    /// Returns Ok((step_norm, converged)) or Err(error_message)
    ///
    /// If Levenberg-Marquardt is enabled, this will evaluate the cost functor twice:
    /// - Once with JetArena to get Jacobian
    /// - Once with F64Ctx to check if the step improves the cost (line search)
    pub fn solve_iteration_with_functor<F>(
        &self,
        params: &mut [f64],
        buffers: &mut TinySolverBuffers,
        cost_functor: &F,
        lambda: &mut f64,
    ) -> Result<(f64, bool), String>
    where
        F: CostFunctor<JetArena<f64>> + CostFunctor<crate::f64_ctx::F64Ctx>,
    {
        let num_params = params.len();
        if !buffers.can_handle(num_params, buffers.max_residuals) {
            return Err(format!("Buffers too small: need {} params, have {}",
                              num_params, buffers.max_params));
        }

        let mut arena = JetArena::with_capacity(num_params, self.arena_capacity);

        // Create parameter vector with pre-filled derivatives
        let param_vector = ParameterVector::from_params(params, &mut arena);

        // Evaluate cost function into pre-allocated buffer
        let writer = buffers.residual_writer();
        let actual_residuals = cost_functor.residuals(param_vector.handles(), &mut arena, writer)?;

        // Extract residuals and jacobian into pre-allocated buffers
        for i in 0..actual_residuals {
            let jet = buffers.residual_handles[i];
            buffers.residuals[i] = *arena.value(jet);
            let derivs = arena.derivatives(jet);
            for j in 0..num_params {
                buffers.jacobian[i * num_params + j] = derivs[j];
            }
        }

        // Compute step using normal equations: J^T J delta = -J^T r
        // Zero out working arrays
        for i in 0..num_params * num_params {
            buffers.jtj[i] = 0.0;
        }
        for i in 0..num_params {
            buffers.jtr[i] = 0.0;
        }

        // J^T J and J^T r
        for i in 0..actual_residuals {
            for j in 0..num_params {
                let ji = buffers.jacobian[i * num_params + j];
                buffers.jtr[j] += ji * buffers.residuals[i];
                for k in 0..num_params {
                    let jk = buffers.jacobian[i * num_params + k];
                    buffers.jtj[j * num_params + k] += ji * jk;
                }
            }
        }

        // Solve linear system: J^T J * delta = -J^T r
        for i in 0..num_params {
            buffers.jtr[i] = -buffers.jtr[i];
        }

        // Apply Levenberg-Marquardt damping or simple regularization
        let damping = if self.use_levenberg_marquardt {
            *lambda
        } else {
            1e-6
        };

        for i in 0..num_params {
            buffers.jtj[i * num_params + i] += damping;
        }

        // Solve using Gaussian elimination (in-place in jtj and jtr)
        if let Err(e) = solve_linear_system_inplace(&mut buffers.jtj, &mut buffers.jtr, num_params) {
            return Err(e);
        }

        // buffers.jtr now contains the solution (delta)
        for i in 0..num_params {
            buffers.param_step[i] = buffers.jtr[i];
        }

        // Check convergence
        let step_norm: f64 = buffers.param_step[..num_params].iter().map(|x| x * x).sum::<f64>().sqrt();
        let converged = step_norm < self.tolerance;

        if self.use_levenberg_marquardt {
            // Levenberg-Marquardt: test step before accepting
            // Compute current cost
            let current_cost: f64 = buffers.residuals[..actual_residuals].iter().map(|r| r * r).sum::<f64>() * 0.5;

            // Compute candidate parameters
            for i in 0..num_params {
                buffers.candidate_params[i] = params[i] + buffers.param_step[i];
            }

            // Evaluate cost at candidate parameters using F64Ctx (no Jacobian needed)
            use crate::f64_ctx::F64Ctx;
            let mut f64_ctx = F64Ctx;
            let f64_writer = ResidualWriter::new(&mut buffers.candidate_residuals_f64[..actual_residuals]);
            let _ = cost_functor.residuals(&buffers.candidate_params[..num_params], &mut f64_ctx, f64_writer)?;

            // Compute candidate cost
            let candidate_cost: f64 = buffers.candidate_residuals_f64[..actual_residuals].iter().map(|r| r * r).sum::<f64>() * 0.5;

            // Compute predicted cost change (linear model)
            let mut predicted_reduction = 0.0;
            for i in 0..num_params {
                predicted_reduction += buffers.param_step[i] * buffers.jtr[i]; // Already negated
            }
            predicted_reduction *= 0.5;

            let actual_reduction = current_cost - candidate_cost;
            let rho = if predicted_reduction.abs() < 1e-12 {
                0.0
            } else {
                actual_reduction / predicted_reduction
            };

            if rho > 0.0 {
                // Accept step
                for i in 0..num_params {
                    params[i] = buffers.candidate_params[i];
                }
                // Decrease damping
                *lambda *= self.lambda_scale_down;
                *lambda = lambda.max(1e-12);
            } else {
                // Reject step
                *lambda *= self.lambda_scale_up;
                *lambda = lambda.min(1e12);
            }

            Ok((step_norm, converged))
        } else {
            // Gauss-Newton: always accept step
            for i in 0..num_params {
                params[i] += buffers.param_step[i];
            }
            Ok((step_norm, converged))
        }
    }

    /// Solve a non-linear least squares problem using a CostFunctor
    /// Zero-allocation solve with pre-allocated buffers
    pub fn solve_with_functor<F>(
        &self,
        initial_params: &[f64],
        buffers: &mut TinySolverBuffers,
        cost_functor: &F,
    ) -> Result<Vec<f64>, String>
    where
        F: CostFunctor<JetArena<f64>> + CostFunctor<crate::f64_ctx::F64Ctx>,
    {
        let num_params = initial_params.len();
        if !buffers.can_handle(num_params, buffers.max_residuals) {
            return Err(format!("Buffers too small: need {} params, have {}",
                              num_params, buffers.max_params));
        }

        let mut params = initial_params.to_vec(); // TODO: eliminate this allocation too
        let mut lambda = self.initial_lambda;

        for _iteration in 0..self.max_iterations {
            let (_step_norm, converged) = self.solve_iteration_with_functor(&mut params, buffers, cost_functor, &mut lambda)?;
            if converged {
                return Ok(params);
            }
        }

        Err("Failed to converge".to_string())
    }
}

/// In-place Gaussian elimination solver for A x = b
/// a: flattened square matrix (modified in place)
/// b: right-hand side (modified in place to contain solution)
/// n: matrix dimension
fn solve_linear_system_inplace(a: &mut [f64], b: &mut [f64], n: usize) -> Result<(), String> {
    if a.len() != n * n || b.len() != n {
        return Err("Invalid matrix dimensions".to_string());
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[k * n + i].abs() > a[max_row * n + i].abs() {
                max_row = k;
            }
        }

        // Swap rows in A and b
        if max_row != i {
            for j in 0..n {
                a.swap(i * n + j, max_row * n + j);
            }
            b.swap(i, max_row);
        }

        // Check for near-singular matrix
        if a[i * n + i].abs() < 1e-12 {
            return Err("Matrix is near singular".to_string());
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = a[k * n + i] / a[i * n + i];
            for j in i..n {
                a[k * n + j] -= factor * a[i * n + j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            b[i] -= a[i * n + j] * b[j];
        }
        b[i] /= a[i * n + i];
    }

    Ok(())
}
