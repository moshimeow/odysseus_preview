//! Quickstart example showing the clean API

use odysseus_solver::{Jet, Real};

fn main() {
    println!("🚀 odysseus-solver Quickstart\n");

    // Example 1: Basic autodiff
    println!("Example 1: Basic Autodiff");
    println!("--------------------------");

    type Jet2 = Jet<f64, 2>;
    let x = Jet2::variable(3.0, 0);  // x = 3, ∂/∂x
    let y = Jet2::variable(4.0, 1);  // y = 4, ∂/∂y

    // NO MACROS! Just write normal math
    let result = x * x + y * y;

    println!("f(x,y) = x² + y²");
    println!("f(3, 4) = {}", result.value);           // 25
    println!("∂f/∂x = {}", result.derivs[0]);         // 2x = 6
    println!("∂f/∂y = {}", result.derivs[1]);         // 2y = 8

    // Example 2: Math functions
    println!("\nExample 2: Math Functions");
    println!("--------------------------");

    type Jet1 = Jet<f64, 1>;
    let x = Jet1::variable(2.0, 0);

    let result = (x * x).sin() + x.sqrt();

    println!("f(x) = sin(x²) + √x");
    println!("f(2) = {:.6}", result.value);
    println!("f'(2) = {:.6}", result.derivs[0]);

    // Example 3: Generic functions
    println!("\nExample 3: Generic Functions");
    println!("-----------------------------");

    fn squared<T: Real>(x: T) -> T {
        x * x  // Works for both f64 and Jet!
    }

    println!("With f64:");
    let result_f64 = squared(5.0);
    println!("  5² = {}", result_f64);

    println!("With Jet (autodiff):");
    let x_jet = Jet1::variable(5.0, 0);
    let result_jet = squared(x_jet);
    println!("  5² = {}", result_jet.value);
    println!("  d/dx(x²) at x=5 = {}", result_jet.derivs[0]);  // 2*5 = 10

    // Example 4: Chain rule in action
    println!("\nExample 4: Chain Rule");
    println!("----------------------");

    let x = Jet1::variable(1.0, 0);

    // Compose functions - chain rule handled automatically
    let y = x + x;           // y = 2x
    let z = y * y;           // z = y² = (2x)²
    let w = z.sin();         // w = sin(z) = sin((2x)²)

    println!("w = sin((2x)²)");
    println!("w(1) = {:.6}", w.value);
    println!("dw/dx at x=1 = {:.6}", w.derivs[0]);

    // Manual calculation: dw/dx = cos(z) * 2y * 2 = cos(4) * 4 * 2 = 8*cos(4)
    let expected = 8.0 * 4.0_f64.cos();
    println!("Expected:      {:.6}", expected);
    println!("Match: {}", (w.derivs[0] - expected).abs() < 1e-10);

    println!("\n✨ All without a single macro!");
}
