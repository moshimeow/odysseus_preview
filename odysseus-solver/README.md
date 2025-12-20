# odysseus-solver

**Compile-time sized automatic differentiation for Rust**

A clean, macro-free automatic differentiation library using const generics. Perfect for computer vision, robotics, and optimization problems where parameter counts are known at compile time.

## 🎯 Key Features

- ✅ **Zero macros** - Just natural Rust syntax with operator overloading
- ✅ **Stack-allocated** - All memory on the stack, zero heap allocations
- ✅ **Type-safe** - Parameter count is part of the type system
- ✅ **NaN detection** - Automatic checks in debug mode (zero overhead in release)
- ✅ **nalgebra integration** - Works seamlessly with nalgebra's static matrices
- ✅ **Generic code** - Write once, works with or without autodiff
- ✅ **Levenberg-Marquardt solver** - Production-ready nonlinear optimizer
- ✅ **Rerun visualization** - Interactive 3D visualization support
- ✅ **Compile-time optimization** - Aggressive inlining and optimization

## 🚀 Quick Start

```bash
# Run examples
cargo run --example quickstart
cargo run --example simple_optimization
cargo run --example robot_arm_2d
cargo run --example pointcloud_alignment

# With Rerun visualization
cargo run --example robot_arm_2d --features visualization
cargo run --example pointcloud_alignment_rerun --features visualization
```

## 📖 Philosophy

`odysseus-solver` uses const generics for compile-time sized differentiation:

- Number of parameters must be known at compile time
- Everything is stack-allocated with zero heap allocations
- Perfect for CV/robotics where you know the problem size (e.g., 6-DOF pose)
- Zero runtime overhead for parameter management

## 💡 Example

```rust
use odysseus_solver::{Jet, Real};

// Define parameter count at compile time
const N_PARAMS: usize = 3;
type Jet3 = Jet<f64, N_PARAMS>;

// Create variables
let a = Jet3::variable(2.0, 0);  // ∂/∂a
let b = Jet3::variable(3.0, 1);  // ∂/∂b
let c = Jet3::variable(1.0, 2);  // ∂/∂c

// Just write normal math - no macros!
let result = a * a + b.sin() + c.sqrt();

println!("Value: {}", result.value);        // 7.798...
println!("∂/∂a: {}", result.derivs[0]);    // 2*a = 4.0
println!("∂/∂b: {}", result.derivs[1]);    // cos(b)
println!("∂/∂c: {}", result.derivs[2]);    // 1/(2√c)
```

## 🎯 f32 and f64 Support

Jets work with both `f32` and `f64` scalar types - just change the type parameter:

```rust
// Use f64 for high precision (default)
type Jet64 = Jet<f64, 3>;
let x_f64 = Jet64::variable(2.0, 0);

// Use f32 for lower memory usage and faster computation
type Jet32 = Jet<f32, 3>;
let x_f32 = Jet32::variable(2.0, 0);
```

**When to use f32:**
- Lower memory usage (half the size)
- Faster SIMD operations
- GPU computation
- Mobile/embedded platforms

**When to use f64:**
- Higher numerical precision needed
- Scientific computing (default choice)
- Avoiding accumulation of floating point errors

See `examples/f32_demo.rs` for a complete demonstration.

## 🔧 Optimization with Built-in Solver

```rust
use odysseus_solver::{Jet, LevenbergMarquardt};
use nalgebra::{SVector, SMatrix};

const N_PARAMS: usize = 2;
const N_RESIDUALS: usize = 5;

// Define cost function
let cost_fn = |params: &SVector<f64, N_PARAMS>| {
    // Compute residuals and Jacobian using Jets
    // ... (see examples for full code)
    (residuals, jacobian)
};

// Create solver
let solver = LevenbergMarquardt::<N_PARAMS, N_RESIDUALS>::new()
    .with_tolerance(1e-10)
    .with_max_iterations(50);

// Solve!
let initial = SVector::<f64, N_PARAMS>::zeros();
let solution = solver.solve_simple(initial, cost_fn);
```

## 🎨 Generic Functions

Write functions that work with **both** `f64` and `Jet<f64, N>`:

```rust
use odysseus_solver::Real;

fn my_function<T: Real>(x: T, y: T) -> T {
    x * x + y.sin()  // Works for both f64 and Jet!
}

// Use with f64 (no autodiff)
let result_f64 = my_function(2.0, 3.0);

// Use with Jet (with autodiff)
type Jet2 = Jet<f64, 2>;
let x = Jet2::variable(2.0, 0);
let y = Jet2::variable(3.0, 1);
let result_jet = my_function(x, y);
// Now result_jet.derivs contains ∂f/∂x and ∂f/∂y
```

## 🤖 3D Math with Autodiff

The `math3d` module provides `Vec3` and `Mat3` types that work generically:

```rust
use odysseus_solver::math3d::{Vec3, Mat3, rodrigues_to_matrix};

// Works with f64
let rvec_f64 = Vec3::new(0.1, 0.2, 0.3);
let rotation_f64 = rodrigues_to_matrix(rvec_f64);

// Works with Jet - same code, now with derivatives!
type Jet6 = Jet<f64, 6>;
let rvec_jet = Vec3::new(
    Jet6::variable(0.1, 3),
    Jet6::variable(0.2, 4),
    Jet6::variable(0.3, 5),
);
let rotation_jet = rodrigues_to_matrix(rvec_jet);
// rotation_jet now contains derivatives w.r.t. rotation parameters!
```

## 🛡️ NaN Detection

In debug builds, all Jet operations automatically check for NaN values:

```rust
let x = Jet::<f64, 1>::variable(0.0, 0);
let bad = x.sqrt();  // In debug: panics with "NaN detected in sqrt operation!"
                     // In release: no overhead
```

This makes debugging numerical issues much easier - you'll know exactly which operation produces NaN!

## 📊 Examples

### Simple Optimization
Fit a parabola `y = ax² + bx + c` to noisy data:
```bash
cargo run --example simple_optimization
```

### 2D Robot Arm IK
Inverse kinematics for a 2-DOF planar robot arm:
```bash
cargo run --example robot_arm_2d --features visualization
```

### 3D Pointcloud Alignment
6-DOF pose estimation between two pointclouds:
```bash
cargo run --example pointcloud_alignment_rerun --features visualization
```
## 🏗️ Implementation Details

### Jet Type

```rust
pub struct Jet<T, const N: usize> {
    pub value: T,           // The scalar value
    pub derivs: [T; N],     // Derivatives w.r.t. N parameters
}
```

### Operator Implementations

All standard operators are implemented using the chain rule:

```rust
impl<const N: usize> Mul for Jet<f64, N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let result = Self {
            value: self.value * rhs.value,
            derivs: std::array::from_fn(|i| {
                self.value * rhs.derivs[i] + rhs.value * self.derivs[i]
            }),
        };
        result.check_nan("mul");  // Debug-only check
        result
    }
}
```

### Math Functions

Math functions are implemented as methods on `Jet`:

```rust
impl<const N: usize> Jet<f64, N> {
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
}
```

## 🔗 Integration with nalgebra

Perfect integration with nalgebra's compile-time sized types:

```rust
use nalgebra::{SVector, SMatrix};
use odysseus_solver::Jet;

type Jet6 = Jet<f64, 6>;

// Vector of Jets (parameters)
let params: SVector<Jet6, 6> = SVector::from_fn(|i, _| Jet6::variable(0.0, i));

// Matrix of derivatives (Jacobian) - extracted from Jets
let mut jacobian = SMatrix::<f64, 3, 6>::zeros();
// ... fill from jet.derivs
```

