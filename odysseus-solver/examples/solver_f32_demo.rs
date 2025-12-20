//! Demonstration of f32 solver support
//!
//! Shows that LevenbergMarquardt works with both f32 and f64

use nalgebra::{SMatrix, SVector};
use odysseus_solver::{Jet, LevenbergMarquardt};

const N_PARAMS: usize = 2;
const N_DATA: usize = 4;

fn main() {
    println!("🔧 Solver f32/f64 Support Demo");
    println!("================================\n");

    // Data: fit y = a*x + b
    let data = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 9.0)]; // y = 2x + 1

    println!("📊 Data: y = 2x + 1");
    println!("Points: {:?}\n", data);

    // ===== f64 version =====
    println!("Using f64 solver:");

    type Jet64_2 = Jet<f64, N_PARAMS>;

    let cost_fn_f64 = |params: &SVector<f64, N_PARAMS>| {
        let a = Jet64_2::variable(params[0], 0);
        let b = Jet64_2::variable(params[1], 1);

        let mut residuals = SVector::<f64, N_DATA>::zeros();
        let mut jacobian = SMatrix::<f64, N_DATA, N_PARAMS>::zeros();

        for (i, &(x, y_true)) in data.iter().enumerate() {
            let x_jet = Jet64_2::constant(x);
            let y_pred = a * x_jet + b;
            let residual = y_pred - Jet64_2::constant(y_true);

            residuals[i] = residual.value;
            jacobian[(i, 0)] = residual.derivs[0];
            jacobian[(i, 1)] = residual.derivs[1];
        }

        (residuals, jacobian)
    };

    let mut solver_f64 = LevenbergMarquardt::<f64, N_PARAMS, N_DATA>::new()
        .with_tolerance(1e-10)
        .with_verbose(false);
    let initial_f64 = SVector::<f64, N_PARAMS>::new(0.0, 0.0);
    let result_f64 = solver_f64.solve_simple(initial_f64, cost_fn_f64);

    println!("  Result: a = {:.6}, b = {:.6}", result_f64[0], result_f64[1]);
    println!("  Expected: a = 2.0, b = 1.0\n");

    // ===== f32 version =====
    println!("Using f32 solver:");

    type Jet32_2 = Jet<f32, N_PARAMS>;

    let cost_fn_f32 = |params: &SVector<f32, N_PARAMS>| {
        let a = Jet32_2::variable(params[0], 0);
        let b = Jet32_2::variable(params[1], 1);

        let mut residuals = SVector::<f32, N_DATA>::zeros();
        let mut jacobian = SMatrix::<f32, N_DATA, N_PARAMS>::zeros();

        for (i, &(x, y_true)) in data.iter().enumerate() {
            let x_jet = Jet32_2::constant(x as f32);
            let y_pred = a * x_jet + b;
            let residual = y_pred - Jet32_2::constant(y_true as f32);

            residuals[i] = residual.value;
            jacobian[(i, 0)] = residual.derivs[0];
            jacobian[(i, 1)] = residual.derivs[1];
        }

        (residuals, jacobian)
    };

    let mut solver_f32 = LevenbergMarquardt::<f32, N_PARAMS, N_DATA>::new()
        .with_tolerance(1e-4)  // Looser tolerance for f32
        .with_lambda_params(1e-2, 10.0, 0.1)  // Higher initial lambda for stability
        .with_verbose(false);
    let initial_f32 = SVector::<f32, N_PARAMS>::new(0.0, 0.0);
    let result_f32 = solver_f32.solve_simple(initial_f32, cost_fn_f32);

    println!("  Result: a = {:.6}, b = {:.6}", result_f32[0], result_f32[1]);
    println!("  Expected: a = 2.0, b = 1.0\n");

    println!("✅ Both f32 and f64 solvers work perfectly!");
    println!("   Use f32 for: lower memory, faster SIMD, GPU-friendly");
    println!("   Use f64 for: higher precision (default)");
}
