//! Simple curve fitting example showing the clean API
//!
//! This demonstrates how to use the const generic approach for optimization
//! without any macros - just natural Rust syntax.

use nalgebra::{SMatrix, SVector};
use odysseus_solver::{Jet, Real};

/// Fit a parabola y = ax² + bx + c to data points
fn main() {
    println!("🎯 odysseus-solver: Curve Fitting Example");
    println!("=========================================\n");

    // Data points to fit (noisy parabola with a=2, b=1, c=3)
    let data = vec![
        (0.0, 3.0),
        (1.0, 6.1),
        (2.0, 13.0),
        (3.0, 23.9),
        (4.0, 38.8),
    ];

    println!("Data points:");
    for (x, y) in &data {
        println!("  ({:.1}, {:.1})", x, y);
    }

    // Initial guess for parameters [a, b, c]
    let mut params = SVector::<f64, 3>::new(1.0, 1.0, 1.0);
    println!("\n🚀 Initial guess: a={:.3}, b={:.3}, c={:.3}", params[0], params[1], params[2]);

    // Gauss-Newton optimization
    for iteration in 0..10 {
        // Compute Jacobian using autodiff
        let (residuals, jacobian) = compute_residuals_and_jacobian(&data, params);

        // Compute step: (J^T J)^-1 J^T r
        let jtj = jacobian.transpose() * jacobian;
        let jtr = jacobian.transpose() * residuals;

        let step = match jtj.try_inverse() {
            Some(inv) => inv * jtr,
            None => {
                println!("❌ Singular matrix!");
                break;
            }
        };

        // Update parameters
        params -= step;

        let error: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        println!(
            "  Iter {:2}: error={:.6}, params=[{:.6}, {:.6}, {:.6}]",
            iteration, error, params[0], params[1], params[2]
        );

        if step.norm() < 1e-10 {
            println!("✅ Converged!");
            break;
        }
    }

    println!("\n🎉 Final parameters:");
    println!("  a = {:.6} (true: 2.0)", params[0]);
    println!("  b = {:.6} (true: 1.0)", params[1]);
    println!("  c = {:.6} (true: 3.0)", params[2]);
}

/// Evaluate parabola: y = ax² + bx + c
/// Works with both f64 and Jet thanks to Real trait
fn parabola<T: Real>(x: f64, a: T, b: T, c: T) -> T {
    let x_t = T::from_literal(x);
    a * x_t * x_t + b * x_t + c
}

/// Compute residuals and Jacobian using automatic differentiation
fn compute_residuals_and_jacobian(
    data: &[(f64, f64)],
    params: SVector<f64, 3>,
) -> (SVector<f64, 5>, SMatrix<f64, 5, 3>) {
    const N_PARAMS: usize = 3;
    const N_DATA: usize = 5;

    type Jet3 = Jet<f64, N_PARAMS>;

    // Create parameter Jets
    let a = Jet3::variable(params[0], 0);
    let b = Jet3::variable(params[1], 1);
    let c = Jet3::variable(params[2], 2);

    // Compute residuals with autodiff
    let mut residuals = SVector::<f64, N_DATA>::zeros();
    let mut jacobian = SMatrix::<f64, N_DATA, N_PARAMS>::zeros();

    for (i, &(x, y_true)) in data.iter().enumerate() {
        // Evaluate parabola with Jets - just natural syntax!
        let y_pred = parabola(x, a, b, c);

        // Residual = predicted - true
        let residual = y_pred.value - y_true;
        residuals[i] = residual;

        // Jacobian row = derivatives of residual w.r.t. parameters
        for j in 0..N_PARAMS {
            jacobian[(i, j)] = y_pred.derivs[j];
        }
    }

    (residuals, jacobian)
}
