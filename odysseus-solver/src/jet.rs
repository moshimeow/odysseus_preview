//! Dual number (Jet) type for forward-mode automatic differentiation
//!
//! A Jet<T, N> represents a value along with its derivatives with respect to N parameters.

use std::ops::{Add, Sub, Mul, Div, Neg};

/// A dual number containing a value and its derivatives
///
/// Generic over:
/// - T: the scalar type (usually f64 or f32)
/// - N: the number of parameters (compile-time constant)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jet<T, const N: usize> {
    /// The scalar value
    pub value: T,
    /// Derivatives with respect to each parameter
    pub derivs: [T; N],
}

impl<T: Copy + Default, const N: usize> Jet<T, N> {
    /// Create a constant (zero derivatives)
    pub fn constant(value: T) -> Self {
        Self {
            value,
            derivs: [T::default(); N],
        }
    }

    /// Create a variable with unit derivative at the given index
    pub fn variable(value: T, index: usize) -> Self
    where
        T: num_traits::One,
    {
        let mut derivs = [T::default(); N];
        derivs[index] = T::one();
        Self { value, derivs }
    }
}

// Macro to avoid code duplication for f32 and f64
macro_rules! impl_jet_nan_check {
    ($T:ty) => {
        impl<const N: usize> Jet<$T, N> {
            /// Check for NaN in value and all derivatives (debug builds only)
            #[inline]
            #[allow(unused_variables)]
            fn check_nan(&self, operation: &str) {
                #[cfg(debug_assertions)]
                {
                    if self.value.is_nan() {
                        panic!("NaN detected in {} operation! Value is NaN", operation);
                    }
                    for (i, &deriv) in self.derivs.iter().enumerate() {
                        if deriv.is_nan() {
                            panic!(
                                "NaN detected in {} operation! Derivative {} is NaN (value: {})",
                                operation, i, self.value
                            );
                        }
                    }
                }
            }
        }
    };
}

impl_jet_nan_check!(f32);
impl_jet_nan_check!(f64);

// ============================================================================
// Arithmetic Operations (specialized for f32 and f64)
// ============================================================================

macro_rules! impl_jet_arithmetic {
    ($T:ty) => {
        /// Addition: (a + da) + (b + db) = (a + b) + (da + db)
        impl<const N: usize> Add for Jet<$T, N> {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                let result = Self {
                    value: self.value + rhs.value,
                    derivs: std::array::from_fn(|i| self.derivs[i] + rhs.derivs[i]),
                };
                result.check_nan("add");
                result
            }
        }

        /// Subtraction: (a + da) - (b + db) = (a - b) + (da - db)
        impl<const N: usize> Sub for Jet<$T, N> {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                let result = Self {
                    value: self.value - rhs.value,
                    derivs: std::array::from_fn(|i| self.derivs[i] - rhs.derivs[i]),
                };
                result.check_nan("sub");
                result
            }
        }

        /// Multiplication: (a + da) * (b + db) = ab + a*db + b*da
        impl<const N: usize> Mul for Jet<$T, N> {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                let result = Self {
                    value: self.value * rhs.value,
                    derivs: std::array::from_fn(|i| {
                        self.value * rhs.derivs[i] + rhs.value * self.derivs[i]
                    }),
                };
                result.check_nan("mul");
                result
            }
        }

        /// Division: (a + da) / (b + db) = a/b + (da*b - a*db)/b²
        impl<const N: usize> Div for Jet<$T, N> {
            type Output = Self;

            fn div(self, rhs: Self) -> Self {
                let b_squared = rhs.value * rhs.value;
                let result = Self {
                    value: self.value / rhs.value,
                    derivs: std::array::from_fn(|i| {
                        (self.derivs[i] * rhs.value - self.value * rhs.derivs[i]) / b_squared
                    }),
                };
                result.check_nan("div");
                result
            }
        }

        /// Negation: -(a + da) = -a + (-da)
        impl<const N: usize> Neg for Jet<$T, N> {
            type Output = Self;

            fn neg(self) -> Self {
                let result = Self {
                    value: -self.value,
                    derivs: std::array::from_fn(|i| -self.derivs[i]),
                };
                result.check_nan("neg");
                result
            }
        }
    };
}

impl_jet_arithmetic!(f32);
impl_jet_arithmetic!(f64);

// ============================================================================
// Mathematical Functions (specialized for f32 and f64)
// ============================================================================

macro_rules! impl_jet_math {
    ($T:ty, $epsilon:expr) => {
        impl<const N: usize> Jet<$T, N> {
            /// Sine: sin(a + da) = sin(a) + cos(a) * da
            pub fn sin(self) -> Self {
                let sin_a = self.value.sin();
                let cos_a = self.value.cos();
                let result = Self {
                    value: sin_a,
                    derivs: std::array::from_fn(|i| cos_a * self.derivs[i]),
                };
                result.check_nan("sin");
                result
            }

            /// Cosine: cos(a + da) = cos(a) - sin(a) * da
            pub fn cos(self) -> Self {
                let sin_a = self.value.sin();
                let cos_a = self.value.cos();
                let result = Self {
                    value: cos_a,
                    derivs: std::array::from_fn(|i| -sin_a * self.derivs[i]),
                };
                result.check_nan("cos");
                result
            }

            /// Square root: sqrt(a + da) = sqrt(a) + da/(2*sqrt(a))
            pub fn sqrt(self) -> Self {
                let sqrt_a = self.value.sqrt();
                // Safe derivative: add epsilon to avoid division by zero
                let deriv_factor = 1.0 / (2.0 * (sqrt_a + $epsilon));
                let result = Self {
                    value: sqrt_a,
                    derivs: std::array::from_fn(|i| deriv_factor * self.derivs[i]),
                };
                result.check_nan("sqrt");
                result
            }

            /// Power: (a + da)^n ≈ a^n + n*a^(n-1) * da
            pub fn powi(self, n: i32) -> Self {
                let value = self.value.powi(n);
                let n_float = n as $T;
                let deriv_factor = n_float * self.value.powi(n - 1);
                let result = Self {
                    value,
                    derivs: std::array::from_fn(|i| deriv_factor * self.derivs[i]),
                };
                result.check_nan("powi");
                result
            }

            /// Arc cosine: acos(a + da) = acos(a) - da / sqrt(1 - a^2)
            pub fn acos(self) -> Self {
                let value = self.value.acos();
                // Derivative: -1 / sqrt(1 - x^2)
                let one_minus_x_sq = 1.0 - self.value * self.value;
                let deriv_factor = -1.0 / (one_minus_x_sq.sqrt() + $epsilon);
                let result = Self {
                    value,
                    derivs: std::array::from_fn(|i| deriv_factor * self.derivs[i]),
                };
                result.check_nan("acos");
                result
            }

            /// Absolute value (non-differentiable at 0, uses sign)
            pub fn abs(self) -> Self {
                let sign = self.value.signum();
                let result = Self {
                    value: self.value.abs(),
                    derivs: std::array::from_fn(|i| sign * self.derivs[i]),
                };
                result.check_nan("abs");
                result
            }
        }
    };
}

impl_jet_math!(f32, 1e-10_f32);  // f32 has ~7 decimal digits of precision
impl_jet_math!(f64, 1e-16_f64);

// ============================================================================
// Real trait for generic programming
// ============================================================================

/// Trait for types that support real number operations
///
/// Implemented for both f64 and Jet<f64, N>, allowing generic code
/// that works with or without automatic differentiation.
pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Sized
{
    type Scalar: Copy;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn acos(self) -> Self;

    fn constant(value: Self::Scalar) -> Self; //
    fn from_literal(value: f64) -> Self;  // Convert f64 to Self (works for literals and variables)
    fn from_f64(value: f64) -> Self;
    fn from_f32(value: f32) -> Self;

    fn zero() -> Self;
    fn one() -> Self;
}

/// Real implementation for f64 (no autodiff)
impl Real for f64 {
    type Scalar = f64;

    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    fn acos(self) -> Self {
        f64::acos(self)
    }
    fn constant(value: f64) -> Self {
        value
    }
    fn from_literal(value: f64) -> Self {
        value
    }
    fn from_f64(value: f64) -> Self {
        value
    }
    fn from_f32(value: f32) -> Self {
        value as f64
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

/// Real implementation for f32 (no autodiff)
impl Real for f32 {
    type Scalar = f32;

    fn sin(self) -> Self {
        f32::sin(self)
    }
    fn cos(self) -> Self {
        f32::cos(self)
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }
    fn acos(self) -> Self {
        f32::acos(self)
    }
    fn constant(value: f32) -> Self {
        value
    }
    fn from_literal(value: f64) -> Self {
        value as f32
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn from_f32(value: f32) -> Self {
        value
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

/// Real implementation for Jet - specialized for f32 and f64
macro_rules! impl_real_for_jet {
    ($T:ty) => {
        impl<const N: usize> Real for Jet<$T, N> {
            type Scalar = $T;

            fn sin(self) -> Self {
                self.sin()
            }
            fn cos(self) -> Self {
                self.cos()
            }
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn abs(self) -> Self {
                self.abs()
            }
            fn powi(self, n: i32) -> Self {
                self.powi(n)
            }
            fn acos(self) -> Self {
                self.acos()
            }
            fn constant(value: $T) -> Self {
                Jet::constant(value)
            }
            fn from_literal(value: f64) -> Self {
                Jet::constant(value as $T)
            }
            fn from_f64(value: f64) -> Self {
                Jet::constant(value as $T)
            }
            fn from_f32(value: f32) -> Self {
                Jet::constant(value as $T)
            }
            fn zero() -> Self {
                Jet::constant(0.0)
            }
            fn one() -> Self {
                Jet::constant(1.0)
            }
        }
    };
}

impl_real_for_jet!(f32);
impl_real_for_jet!(f64);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let x = Jet::<f64, 2>::variable(3.0, 0);
        let y = Jet::<f64, 2>::variable(4.0, 1);

        let sum = x + y;
        assert_eq!(sum.value, 7.0);
        assert_eq!(sum.derivs, [1.0, 1.0]);
    }

    #[test]
    fn test_multiplication() {
        let x = Jet::<f64, 2>::variable(3.0, 0);
        let y = Jet::<f64, 2>::variable(4.0, 1);

        let product = x * y;
        assert_eq!(product.value, 12.0);
        assert_eq!(product.derivs, [4.0, 3.0]); // d/dx(xy) = y, d/dy(xy) = x
    }

    #[test]
    fn test_chain_rule() {
        let x = Jet::<f64, 1>::variable(2.0, 0);

        // f(x) = x^2
        let result = x * x;
        assert_eq!(result.value, 4.0);
        assert_eq!(result.derivs[0], 4.0); // d/dx(x^2) = 2x = 4
    }

    #[test]
    fn test_sin() {
        use std::f64::consts::PI;
        let x = Jet::<f64, 1>::variable(PI / 4.0, 0);

        let result = x.sin();
        assert!((result.value - (PI / 4.0).sin()).abs() < 1e-10);
        assert!((result.derivs[0] - (PI / 4.0).cos()).abs() < 1e-10);
    }

    #[test]
    fn test_generic_function() {
        // Generic function that works with both f64 and Jet
        fn quadratic<T: Real>(x: T) -> T {
            x * x + x + T::from_literal(1.0)
        }

        // Test with f64
        let result_f64 = quadratic(2.0);
        assert_eq!(result_f64, 7.0);

        // Test with Jet
        let x_jet = Jet::<f64, 1>::variable(2.0, 0);
        let result_jet = quadratic(x_jet);
        assert_eq!(result_jet.value, 7.0);
        assert_eq!(result_jet.derivs[0], 5.0); // d/dx(x^2 + x + 1) = 2x + 1 = 5
    }
}
