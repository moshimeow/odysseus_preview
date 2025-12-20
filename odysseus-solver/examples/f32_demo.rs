//! Demonstration of f32 support in odysseus-solver
//!
//! Shows that Jets work with both f32 and f64 scalar types

use odysseus_solver::{Jet, Real};

fn main() {
    println!("🔢 odysseus-solver: f32 and f64 Support Demo");
    println!("=============================================\n");

    // Generic function that works with any Real type
    fn quadratic<T: Real>(x: T, a: T, b: T, c: T) -> T {
        a * x * x + b * x + c
    }

    // ===== f64 version =====
    println!("📊 Using f64:");

    // Without autodiff
    let result_f64 = quadratic(2.0_f64, 1.0_f64, 3.0_f64, 5.0_f64);
    println!("  f(2) = {}", result_f64); // 1*4 + 3*2 + 5 = 15

    // With autodiff
    type Jet64_3 = Jet<f64, 3>;
    let x = Jet64_3::variable(2.0, 0);
    let a = Jet64_3::variable(1.0, 1);
    let b = Jet64_3::variable(3.0, 2);
    let c = Jet64_3::constant(5.0);

    let result_jet64 = quadratic(x, a, b, c);
    println!("  f(2) = {}", result_jet64.value);
    println!("  ∂f/∂x = {} (should be 2ax + b = 2*1*2 + 3 = 7)", result_jet64.derivs[0]);
    println!("  ∂f/∂a = {} (should be x² = 4)", result_jet64.derivs[1]);
    println!("  ∂f/∂b = {} (should be x = 2)", result_jet64.derivs[2]);

    println!();

    // ===== f32 version =====
    println!("📊 Using f32:");

    // Without autodiff
    let result_f32 = quadratic(2.0_f32, 1.0_f32, 3.0_f32, 5.0_f32);
    println!("  f(2) = {}", result_f32);

    // With autodiff
    type Jet32_3 = Jet<f32, 3>;
    let x32 = Jet32_3::variable(2.0, 0);
    let a32 = Jet32_3::variable(1.0, 1);
    let b32 = Jet32_3::variable(3.0, 2);
    let c32 = Jet32_3::constant(5.0);

    let result_jet32 = quadratic(x32, a32, b32, c32);
    println!("  f(2) = {}", result_jet32.value);
    println!("  ∂f/∂x = {}", result_jet32.derivs[0]);
    println!("  ∂f/∂a = {}", result_jet32.derivs[1]);
    println!("  ∂f/∂b = {}", result_jet32.derivs[2]);

    println!();

    // ===== Transcendental functions =====
    println!("📐 Transcendental Functions:");

    let angle_f64 = std::f64::consts::PI / 4.0;
    let angle_f32 = std::f32::consts::PI / 4.0;

    println!("  f64: sin(π/4) = {:.6}", angle_f64.sin());
    println!("  f32: sin(π/4) = {:.6}", angle_f32.sin());

    // With autodiff
    let x_f64 = Jet::<f64, 1>::variable(angle_f64, 0);
    let x_f32 = Jet::<f32, 1>::variable(angle_f32, 0);

    let sin_f64 = x_f64.sin();
    let sin_f32 = x_f32.sin();

    println!("\n  With autodiff:");
    println!("  f64: sin(π/4) = {:.6}, d/dx = {:.6} (should be cos(π/4))",
             sin_f64.value, sin_f64.derivs[0]);
    println!("  f32: sin(π/4) = {:.6}, d/dx = {:.6}",
             sin_f32.value, sin_f32.derivs[0]);

    println!("\n✅ Both f32 and f64 work seamlessly with Jets!");
    println!("   Use f32 for lower memory usage and faster computation.");
    println!("   Use f64 for higher precision (default).");
}
