// ============================================================================
// Module Declarations
// ============================================================================

mod jet;
mod f64_ctx;
mod math_context;
mod solver;
pub mod math3d;

// Re-export the proc macro from arena-autodiff-macros
pub use arena_autodiff_macros::expr;

// ============================================================================
// Public Re-exports
// ============================================================================

// Jet and Arena types
pub use jet::{ArenaStats, JetArena, JetHandle, ParameterVector};

// Math context abstraction
pub use math_context::MathContext;

// Concrete context types
pub use f64_ctx::{F64Ctx, F32Ctx};

// Solver types
pub use solver::{CostFunctor, ResidualWriter, TinySolver, TinySolverBuffers};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let mut arena = JetArena::new(2);

        let x = arena.variable(3.0, 0);
        let y = arena.variable(4.0, 1);

        // Basic operations
        let sum = arena.add(x, y);
        assert_eq!(*arena.value(sum), 7.0);
        assert_eq!(arena.derivatives(sum), &[1.0, 1.0]);

        let product = arena.mul(x, y);
        assert_eq!(*arena.value(product), 12.0);
        assert_eq!(arena.derivatives(product), &[4.0, 3.0]); // d/dx(xy) = y, d/dy(xy) = x

        let diff = arena.sub(x, y);
        assert_eq!(*arena.value(diff), -1.0);
        assert_eq!(arena.derivatives(diff), &[1.0, -1.0]);
    }

    #[test]
    fn test_expr_macro_with_jets() {
        let mut arena = JetArena::new(2);

        let x = arena.variable(2.0, 0);
        let y = arena.variable(3.0, 1);

        // Test: operator precedence works correctly, type inferred from arena!
        let result = expr!(arena, x * x + y * y);

        assert_eq!(*arena.value(result), 13.0);
        let derivs = arena.derivatives(result);
        assert_eq!(derivs[0], 4.0);  // d/dx(x^2 + y^2) = 2x = 4
        assert_eq!(derivs[1], 6.0);  // d/dy(x^2 + y^2) = 2y = 6
    }

    #[test]
    fn test_expr_macro_with_f64() {
        let a = 3.0;
        let b = 4.0;

        // Test: same expression works with f64 (zero-cost!)
        let result = expr!(F64Ctx, a * a + b * b);

        assert_eq!(result, 25.0);  // 3^2 + 4^2 = 9 + 16 = 25
    }

    #[test]
    fn test_expr_macro_with_functions() {
        use std::f64::consts::PI;

        let mut arena = JetArena::new(2);

        let x = arena.variable(PI / 4.0, 0);
        let y = arena.variable(PI / 6.0, 1);

        // Test: function calls in ONE expression with operators!
        let result = expr!(arena, sin(x) + cos(y));

        let expected = (PI / 4.0).sin() + (PI / 6.0).cos();
        assert!((*arena.value(result) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_expr_macro_comprehensive() {
        let mut arena = JetArena::new(2);

        let x = arena.variable(2.0, 0);
        let y = arena.variable(3.0, 1);

        // Test: complex expression with precedence, functions, and mixed operations
        let result = expr!(arena, sin(x) * x + cos(y) / y);

        let x_val: f64 = 2.0;
        let y_val: f64 = 3.0;
        let expected = x_val.sin() * x_val + y_val.cos() / y_val;

        assert!((*arena.value(result) - expected).abs() < 1e-10);

        // Verify derivatives are computed
        let derivs = arena.derivatives(result);
        assert!(derivs[0] != 0.0);
        assert!(derivs[1] != 0.0);
    }
}
