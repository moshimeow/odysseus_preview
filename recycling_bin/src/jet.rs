/// Handle to a dual number (Jet) stored in the arena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JetHandle(pub(crate) usize);

impl JetHandle {
    /// Get the internal index (for internal use only)
    pub fn index(self) -> usize {
        self.0
    }

    /// Create a handle from an index (for internal use only)
    pub fn from_index(index: usize) -> Self {
        Self(index)
    }
}

/// Arena usage statistics
#[derive(Debug)]
pub struct ArenaStats {
    pub current_used: usize,
    pub max_used: usize,
    pub capacity: usize,
    pub initial_capacity: usize,
}

/// Arena-based storage for dual numbers with dynamic derivative dimensions
pub struct JetArena<T> {
    /// Scalar values for all jets
    pub(crate) values: Vec<T>,
    /// Derivative components, stored as [jet0_derivs..., jet1_derivs..., ...]
    /// Each jet has `derivative_dim` consecutive entries
    pub(crate) derivatives: Vec<T>,
    /// Number of derivative dimensions (N in Jet<T, N>)
    pub(crate) derivative_dim: usize,
    /// Next available slot for allocation
    pub(crate) next_handle: usize,
    /// Initial capacity for preallocation
    pub(crate) initial_capacity: usize,
    /// Track maximum usage for statistics
    pub(crate) max_used: usize,
}

impl<T: Clone + Default> JetArena<T> {
    /// Create new arena with specified derivative dimension
    pub fn new(derivative_dim: usize) -> Self {
        Self::with_capacity(derivative_dim, 32) // Default capacity of 32 jets
    }

    /// Create new arena with specified derivative dimension and initial capacity
    pub fn with_capacity(derivative_dim: usize, capacity: usize) -> Self {
        let mut values = Vec::with_capacity(capacity);
        let mut derivatives = Vec::with_capacity(capacity * derivative_dim);

        // Pre-fill with default values
        values.resize(capacity, T::default());
        derivatives.resize(capacity * derivative_dim, T::default());

        Self {
            values,
            derivatives,
            derivative_dim,
            next_handle: 0,
            initial_capacity: capacity,
            max_used: 0,
        }
    }

    /// Reset arena for reuse (keeps allocated memory)
    pub fn reset(&mut self) {
        self.next_handle = 0;
        // Don't deallocate, just reset usage
    }

    /// Get current usage statistics
    pub fn usage_stats(&self) -> ArenaStats {
        ArenaStats {
            current_used: self.next_handle,
            max_used: self.max_used,
            capacity: self.values.len(),
            initial_capacity: self.initial_capacity,
        }
    }

    /// Ensure we have capacity for more jets, growing by 2x if needed
    fn ensure_capacity(&mut self) {
        if self.next_handle >= self.values.len() {
            let old_capacity = self.values.len();
            let new_capacity = (old_capacity * 2).max(16);

            eprintln!("WARNING: Arena capacity {} exceeded, growing to {}",
                     old_capacity, new_capacity);

            // Resize value storage
            self.values.resize(new_capacity, T::default());

            // Resize derivative storage
            self.derivatives.resize(new_capacity * self.derivative_dim, T::default());
        }
    }
}

impl JetArena<f64> {
    /// Check for NaN in value and derivatives (debug helper)
    #[inline]
    fn check_nan(&self, handle: JetHandle, operation: &str) {
        let value = &self.values[handle.0];
        if value.is_nan() {
            panic!("NaN detected in {} operation! Value is NaN. Handle: {:?}", operation, handle);
        }

        let derivs = self.derivatives(handle);
        for (i, &deriv) in derivs.iter().enumerate() {
            if deriv.is_nan() {
                panic!("NaN detected in {} operation! Derivative {} is NaN. Handle: {:?}, Value: {:?}",
                       operation, i, handle, value);
            }
        }
    }

    /// Allocate a new jet in the arena
    fn allocate(&mut self) -> JetHandle {
        self.ensure_capacity();
        let handle = JetHandle(self.next_handle);
        self.next_handle += 1;
        self.max_used = self.max_used.max(self.next_handle);
        handle
    }

    /// Create a constant (zero derivative) jet
    pub fn constant(&mut self, value: f64) -> JetHandle {
        let handle = self.allocate();
        self.values[handle.0] = value;
        // Derivatives are already zero-initialized
        handle
    }

    /// Create a variable (unit derivative) jet
    pub fn variable(&mut self, value: f64, derivative_index: usize) -> JetHandle {
        assert!(derivative_index < self.derivative_dim,
                "Derivative index {} out of bounds for dimension {}",
                derivative_index, self.derivative_dim);

        let handle = self.allocate();
        self.values[handle.0] = value;

        // Set unit derivative at the specified index
        let deriv_start = handle.0 * self.derivative_dim;
        self.derivatives[deriv_start + derivative_index] = 1.0;

        handle
    }

    /// Get value of a jet
    pub fn value(&self, jet: JetHandle) -> &f64 {
        &self.values[jet.0]
    }

    /// Get derivatives of a jet
    pub fn derivatives(&self, jet: JetHandle) -> &[f64] {
        let start = jet.0 * self.derivative_dim;
        let end = start + self.derivative_dim;
        &self.derivatives[start..end]
    }

    /// Get mutable reference to derivatives (for internal use)
    pub(crate) fn derivatives_mut(&mut self, jet: JetHandle) -> &mut [f64] {
        let start = jet.0 * self.derivative_dim;
        let end = start + self.derivative_dim;
        &mut self.derivatives[start..end]
    }

    /// Addition: (a + da) + (b + db) = (a + b) + (da + db)
    pub fn add(&mut self, lhs: JetHandle, rhs: JetHandle) -> JetHandle {
        let result = self.allocate();

        self.values[result.0] = self.values[lhs.0] + self.values[rhs.0];

        // Collect derivatives before getting mutable reference
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let lhs_derivs = self.derivatives(lhs);
        let rhs_derivs = self.derivatives(rhs);

        for i in 0..self.derivative_dim {
            result_derivatives.push(lhs_derivs[i] + rhs_derivs[i]);
        }

        let result_derivs = self.derivatives_mut(result);
        result_derivs.copy_from_slice(&result_derivatives);

        self.check_nan(result, "add");
        result
    }

    /// Subtraction: (a + da) - (b + db) = (a - b) + (da - db)
    pub fn sub(&mut self, lhs: JetHandle, rhs: JetHandle) -> JetHandle {
        let result = self.allocate();

        self.values[result.0] = self.values[lhs.0] - self.values[rhs.0];

        // Collect derivatives before getting mutable reference
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let lhs_derivs = self.derivatives(lhs);
        let rhs_derivs = self.derivatives(rhs);

        for i in 0..self.derivative_dim {
            result_derivatives.push(lhs_derivs[i] - rhs_derivs[i]);
        }

        let result_derivs = self.derivatives_mut(result);
        result_derivs.copy_from_slice(&result_derivatives);

        self.check_nan(result, "sub");
        result
    }

    /// Multiplication: (a + da) * (b + db) = ab + a*db + b*da
    pub fn mul(&mut self, lhs: JetHandle, rhs: JetHandle) -> JetHandle {
        let result = self.allocate();
        let a = self.values[lhs.0];
        let b = self.values[rhs.0];

        self.values[result.0] = a * b;

        // Collect derivatives before getting mutable reference
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let lhs_derivs = self.derivatives(lhs);
        let rhs_derivs = self.derivatives(rhs);

        for i in 0..self.derivative_dim {
            result_derivatives.push(a * rhs_derivs[i] + b * lhs_derivs[i]);
        }

        let result_derivs = self.derivatives_mut(result);
        result_derivs.copy_from_slice(&result_derivatives);

        self.check_nan(result, "mul");
        result
    }

    /// Division: (a + da) / (b + db) = a/b + (da*b - a*db)/b²
    pub fn div(&mut self, lhs: JetHandle, rhs: JetHandle) -> JetHandle {
        let result = self.allocate();
        let a = self.values[lhs.0];
        let b = self.values[rhs.0];

        self.values[result.0] = a / b;

        // Collect derivatives before getting mutable reference
        let b_squared = b * b;
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let lhs_derivs = self.derivatives(lhs);
        let rhs_derivs = self.derivatives(rhs);

        for i in 0..self.derivative_dim {
            result_derivatives.push((lhs_derivs[i] * b - a * rhs_derivs[i]) / b_squared);
        }

        let result_derivs = self.derivatives_mut(result);
        result_derivs.copy_from_slice(&result_derivatives);

        self.check_nan(result, "div");
        result
    }

    /// Sine function: sin(a + da) ≈ sin(a) + cos(a) * da
    pub fn sin(&mut self, jet: JetHandle) -> JetHandle {
        let a = self.values[jet.0];
        let sin_a = a.sin();
        let cos_a = a.cos();

        // Collect derivative values before creating result
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let jet_derivs = self.derivatives(jet);

        for i in 0..self.derivative_dim {
            result_derivatives.push(cos_a * jet_derivs[i]);
        }

        let handle = self.constant(sin_a);
        let result_derivs = self.derivatives_mut(handle);
        result_derivs.copy_from_slice(&result_derivatives);

        self.check_nan(handle, "sin");
        handle
    }

    /// Cosine function: cos(a + da) ≈ cos(a) - sin(a) * da
    pub fn cos(&mut self, jet: JetHandle) -> JetHandle {
        let a = self.values[jet.0];
        let cos_a = a.cos();
        let sin_a = a.sin();

        // Collect derivative values before creating result
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let jet_derivs = self.derivatives(jet);

        for i in 0..self.derivative_dim {
            result_derivatives.push(-sin_a * jet_derivs[i]);
        }

        let handle = self.constant(cos_a);
        let result_derivs = self.derivatives_mut(handle);
        result_derivs.copy_from_slice(&result_derivatives);

        self.check_nan(handle, "cos");
        handle
    }

    /// Square root: sqrt(a + da) ≈ sqrt(a) + da/(2*sqrt(a))
    pub fn sqrt(&mut self, jet: JetHandle) -> JetHandle {
        let a = self.values[jet.0];
        let sqrt_a = a.sqrt();

        // Safe derivative: add small epsilon to avoid division by zero
        // When sqrt_a is very small, the derivative becomes very large,
        // which can cause numerical issues. We clamp it for stability.
        let eps = 1e-16;
        let derivative_factor = 1.0 / (2.0 * (sqrt_a + eps));

        // Collect derivative values before creating result
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let jet_derivs = self.derivatives(jet);

        for i in 0..self.derivative_dim {
            result_derivatives.push(derivative_factor * jet_derivs[i]);
        }

        let handle = self.constant(sqrt_a);
        let result_derivs = self.derivatives_mut(handle);
        result_derivs.copy_from_slice(&result_derivatives);

        self.check_nan(handle, "sqrt");
        handle
    }

    /// atan2 function: computes arctan(y/x) with correct quadrant
    /// Derivatives: ∂atan2(y,x)/∂y = x/(x²+y²), ∂atan2(y,x)/∂x = -y/(x²+y²)
    pub fn atan2(&mut self, y_jet: JetHandle, x_jet: JetHandle) -> JetHandle {
        let y = self.values[y_jet.0];
        let x = self.values[x_jet.0];
        let result_value = y.atan2(x);

        let denom = x * x + y * y;
        let dy_factor = x / denom;  // ∂/∂y
        let dx_factor = -y / denom; // ∂/∂x

        // Collect derivative values before creating result
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let y_derivs = self.derivatives(y_jet);
        let x_derivs = self.derivatives(x_jet);

        for i in 0..self.derivative_dim {
            result_derivatives.push(dy_factor * y_derivs[i] + dx_factor * x_derivs[i]);
        }

        let handle = self.constant(result_value);
        let result_derivs = self.derivatives_mut(handle);
        result_derivs.copy_from_slice(&result_derivatives);

        handle
    }

    /// Power function: (a + da)^p ≈ a^p + p*a^(p-1)*da
    pub fn pow(&mut self, jet: JetHandle, exponent: f64) -> JetHandle {
        let a = self.values[jet.0];
        let result_value = a.powf(exponent);
        let derivative_factor = exponent * a.powf(exponent - 1.0);

        // Collect derivative values before creating result
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let jet_derivs = self.derivatives(jet);

        for i in 0..self.derivative_dim {
            result_derivatives.push(derivative_factor * jet_derivs[i]);
        }

        let handle = self.constant(result_value);
        let result_derivs = self.derivatives_mut(handle);
        for (i, val) in result_derivatives.into_iter().enumerate() {
            result_derivs[i] = val;
        }

        handle
    }

    /// Exponential: exp(a + da) ≈ exp(a) + exp(a) * da
    pub fn exp(&mut self, jet: JetHandle) -> JetHandle {
        let a = self.values[jet.0];
        let exp_a = a.exp();

        // Collect derivative values before creating result
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let jet_derivs = self.derivatives(jet);

        for i in 0..self.derivative_dim {
            result_derivatives.push(exp_a * jet_derivs[i]);
        }

        let handle = self.constant(exp_a);
        let result_derivs = self.derivatives_mut(handle);
        result_derivs.copy_from_slice(&result_derivatives);

        handle
    }

    /// Natural logarithm: log(a + da) ≈ log(a) + da/a
    pub fn ln(&mut self, jet: JetHandle) -> JetHandle {
        let a = self.values[jet.0];
        let ln_a = a.ln();
        let derivative_factor = 1.0 / a;

        // Collect derivative values before creating result
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let jet_derivs = self.derivatives(jet);

        for i in 0..self.derivative_dim {
            result_derivatives.push(derivative_factor * jet_derivs[i]);
        }

        let handle = self.constant(ln_a);
        let result_derivs = self.derivatives_mut(handle);
        result_derivs.copy_from_slice(&result_derivatives);

        handle
    }

    /// Square function: (a + da)^2 ≈ a^2 + 2*a*da (optimized version of pow(jet, 2.0))
    pub fn square(&mut self, jet: JetHandle) -> JetHandle {
        let a = self.values[jet.0];
        let result_value = a * a;
        let derivative_factor = 2.0 * a;

        // Collect derivative values before creating result
        let mut result_derivatives = Vec::with_capacity(self.derivative_dim);
        let jet_derivs = self.derivatives(jet);

        for i in 0..self.derivative_dim {
            result_derivatives.push(derivative_factor * jet_derivs[i]);
        }

        let handle = self.constant(result_value);
        let result_derivs = self.derivatives_mut(handle);
        for (i, val) in result_derivatives.into_iter().enumerate() {
            result_derivs[i] = val;
        }

        handle
    }
}

/// A vector of parameter JetHandles with pre-filled derivatives for convenience
pub struct ParameterVector {
    handles: Vec<JetHandle>,
}

impl ParameterVector {
    /// Create a new parameter vector from raw parameter values
    pub fn from_params(params: &[f64], arena: &mut JetArena<f64>) -> Self {
        let handles = params
            .iter()
            .enumerate()
            .map(|(i, &val)| arena.variable(val, i))
            .collect();

        Self { handles }
    }

    /// Get the parameter handles as a slice
    pub fn handles(&self) -> &[JetHandle] {
        &self.handles
    }

    /// Get a specific parameter handle by index
    pub fn get(&self, index: usize) -> Option<JetHandle> {
        self.handles.get(index).copied()
    }

    /// Get the number of parameters
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Check if the parameter vector is empty
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }
}

// Implement MathContext for JetArena<f64>
impl crate::math_context::MathContext for JetArena<f64> {
    type Value = JetHandle;

    #[inline]
    fn constant(&mut self, val: f64) -> JetHandle {
        self.constant(val)
    }
    #[inline]
    fn add(&mut self, a: JetHandle, b: JetHandle) -> JetHandle {
        self.add(a, b)
    }
    #[inline]
    fn sub(&mut self, a: JetHandle, b: JetHandle) -> JetHandle {
        self.sub(a, b)
    }
    #[inline]
    fn mul(&mut self, a: JetHandle, b: JetHandle) -> JetHandle {
        self.mul(a, b)
    }
    #[inline]
    fn div(&mut self, a: JetHandle, b: JetHandle) -> JetHandle {
        self.div(a, b)
    }
    #[inline]
    fn sin(&mut self, a: JetHandle) -> JetHandle {
        self.sin(a)
    }
    #[inline]
    fn cos(&mut self, a: JetHandle) -> JetHandle {
        self.cos(a)
    }
    #[inline]
    fn sqrt(&mut self, a: JetHandle) -> JetHandle {
        self.sqrt(a)
    }
    #[inline]
    fn atan2(&mut self, y: JetHandle, x: JetHandle) -> JetHandle {
        self.atan2(y, x)
    }
}
