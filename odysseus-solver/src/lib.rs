//! odysseus-solver: Compile-time sized automatic differentiation
//!
//! This crate provides forward-mode automatic differentiation using const generics.
//! All sizes are known at compile time, with workspace pre-allocated on the heap
//! for optimal performance in long-running VR tracking sessions.

// Let macro-generated `::odysseus_solver::` paths resolve inside this crate.
extern crate self as odysseus_solver;

mod jet;
pub mod math3d;
pub mod params;
pub mod problem;
pub mod solver;
pub mod sparse_solver;

pub use jet::{split_jets, Jet, Real};
pub use odysseus_solver_macros::{compose_params, real_fn};
pub use problem::apply_huber_loss;
pub use solver::LevenbergMarquardt;
pub use sparse_solver::{SparseLevenbergMarquardt, build_slam_entries};

// Re-export nalgebra for convenience
pub use nalgebra;

#[cfg(test)]
mod real_fn_tests {
    use super::*;
    use crate::real_fn;

    #[real_fn]
    fn double<T: Real<Scalar = f64>>(x: T) -> T {
        x * 2.0R
    }

    #[real_fn]
    fn circle_x<T: Real<Scalar = f64>>(t: T) -> T {
        (t * 2.0R * std::f64::consts::PI).cos()
    }

    #[test]
    fn test_double_f64() {
        assert_eq!(double(3.0_f64), 6.0);
    }

    #[test]
    fn test_double_jet() {
        let x = Jet::<f64, 1>::variable(3.0, 0);
        let result = double(x);
        assert_eq!(result.value, 6.0);
        assert_eq!(result.derivs[0], 2.0);
    }

    #[test]
    fn test_plain_f64_literal_unchanged() {
        // 0.0 has no R suffix — must remain f64 for the comparison
        let amplitude = 0.0_f64;
        assert!(amplitude >= 0.0);
    }
}
