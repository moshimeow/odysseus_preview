# Marginalization in Sliding Window Optimization

## Overview

This document explains **marginalization** (also called **variable elimination** or **Schur complement**) in the context of sliding window optimization for SLAM and state estimation.

## The Problem

In VR controller tracking (or SLAM), we want to estimate the trajectory of an object over time. We have:
- **States**: Position and velocity at discrete time steps: `x₀, x₁, x₂, ..., xₙ`
- **Observations**: Noisy measurements of position/velocity at each time step
- **Motion model**: How the state evolves (e.g., Brownian motion, constant velocity)

As we collect more observations over time, the state vector grows unbounded. We need a **sliding window** approach:
1. Keep only the last `W` states in our optimization (e.g., `W=5`)
2. When a new observation arrives, **marginalize out** the oldest state
3. This keeps memory constant while preserving information from old measurements

## Key Insight: Information Preservation

When we marginalize out old states, we must preserve the **information** they provided about the remaining states. This is done through the **Schur complement** of the approximate Hessian.

## Mathematical Derivation

### 1. The Optimization Problem

We're solving a nonlinear least squares problem:

```
minimize: ½ ||r(x)||²
```

where `r(x)` is the vector of residuals (prediction errors).

### 2. Gauss-Newton Approximation

At each iteration, we linearize around the current estimate `x₀`:

```
r(x₀ + δx) ≈ r(x₀) + J δx
```

where `J` is the Jacobian. The update step solves:

```
minimize: ½ ||r₀ + J δx||²
```

Expanding and taking the derivative:

```
J^T J δx = -J^T r₀
```

This is the **Gauss-Newton normal equations**. The matrix `H = J^T J` is the **approximate Hessian**, and `b = -J^T r₀` is the right-hand side.

### 3. Block Structure

Suppose we have states `[x_old, x_new]` where:
- `x_old` = states to marginalize out (oldest time step)
- `x_new` = states to keep (recent time steps)

The Hessian has block structure:

```
H = [H_oo  H_on]
    [H_no  H_nn]

b = [b_o]
    [b_n]
```

The normal equations are:

```
H_oo δx_o + H_on δx_n = b_o
H_no δx_o + H_nn δx_n = b_n
```

### 4. Schur Complement

From the first equation, solve for `δx_o`:

```
δx_o = H_oo⁻¹ (b_o - H_on δx_n)
```

Substitute into the second equation:

```
H_no H_oo⁻¹ (b_o - H_on δx_n) + H_nn δx_n = b_n
```

Rearrange:

```
(H_nn - H_no H_oo⁻¹ H_on) δx_n = b_n - H_no H_oo⁻¹ b_o
```

**This is the marginalized system!** We define:

```
H_marg = H_nn - H_no H_oo⁻¹ H_on    (Schur complement)
b_marg = b_n - H_no H_oo⁻¹ b_o
```

The marginalized system only involves `x_new`:

```
H_marg δx_n = b_marg
```

### 5. Why This Works

The Schur complement `S = H_nn - H_no H_oo⁻¹ H_on` captures:
- **Direct information** about `x_new` from observations (via `H_nn`)
- **Indirect information** about `x_new` that came through `x_old` (via the correction term)

By marginalizing instead of just deleting old states, we preserve information about how old measurements constrained the new states.

## Practical Implementation

### Algorithm: Sliding Window with Marginalization

```
1. Initialize window with first W observations
2. Optimize all W states
3. For each new observation:
   a. Build Hessian H and right-hand side b from current estimate
   b. Partition H into blocks: H_oo (oldest), H_on, H_no, H_nn (rest)
   c. Compute Schur complement:
      H_marg = H_nn - H_no * H_oo⁻¹ * H_on
      b_marg = b_n - H_no * H_oo⁻¹ * b_o
   d. Use (H_marg, b_marg) as a prior for the next optimization
   e. Shift window: drop oldest state, add new state
   f. Optimize with new observation + marginalized prior
```

### Representing the Marginalized Prior

After marginalization, we have:
- `H_marg`: A dense matrix encoding correlations between remaining states
- `b_marg`: The linear term

In the next optimization, we add this as a **prior cost**:

```
cost = ½ ||r_obs(x)||² + ½ δx^T H_marg δx + b_marg^T δx
```

where `δx = x - x_lin` (deviation from linearization point).

## Implementation in nalgebra

```rust
// Block structure for 1D example:
// State: [pos, vel] at each time step
// Window size W=5 → total params = 2*5 = 10

const STATE_DIM: usize = 2;  // [position, velocity]
const WINDOW_SIZE: usize = 5;
const N_PARAMS: usize = STATE_DIM * WINDOW_SIZE;

// After optimization, extract blocks from Hessian
let H: SMatrix<f64, N_PARAMS, N_PARAMS> = /* from solver */;
let b: SVector<f64, N_PARAMS> = /* from solver */;

// Partition: marginalize out first state (index 0..STATE_DIM)
let H_oo = H.fixed_view::<STATE_DIM, STATE_DIM>(0, 0);
let H_on = H.fixed_view::<STATE_DIM, {N_PARAMS - STATE_DIM}>(0, STATE_DIM);
let H_no = H.fixed_view::<{N_PARAMS - STATE_DIM}, STATE_DIM>(STATE_DIM, 0);
let H_nn = H.fixed_view::<{N_PARAMS - STATE_DIM}, {N_PARAMS - STATE_DIM}>(STATE_DIM, STATE_DIM);

// Schur complement
let H_oo_inv = H_oo.try_inverse().expect("H_oo must be invertible");
let H_marg = H_nn - H_no * H_oo_inv * H_on;
let b_marg = b_n - H_no * H_oo_inv * b_o;

// Store (H_marg, b_marg) and linearization point x_lin
// Use in next optimization as prior
```

## Challenges & Caveats

1. **Linearization Point**: The marginalized prior is only valid near the linearization point. If the estimate drifts too far, you may need to re-linearize.

2. **Dense Hessian**: After marginalization, `H_marg` is typically dense (even if original Jacobian was sparse). This is okay for small windows but becomes expensive for large state spaces.

3. **Numerical Stability**: Inverting `H_oo` can be numerically unstable if it's poorly conditioned. Use robust linear algebra (LU, Cholesky) instead of direct matrix inverse.

4. **Information Loss**: Marginalization is approximate (because we linearized). Over many marginalizations, small errors can accumulate.

## 1D Controller Tracking Example

- **State**: `[position, velocity]` at each time step
- **Observations**: Noisy GPS-like position measurements, noisy IMU-like velocity measurements
- **Motion Model**: Brownian motion (velocity changes randomly, position integrates velocity)
- **Window Size**: 5 time steps
- **Marginalization**: When time step 6 arrives, marginalize out time step 0, keep 1-5, optimize with step 6

This gives us constant memory while still using information from all past measurements!

## References

- Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics.
- Sibley, G., Matthies, L., & Sukhatme, G. (2010). "Sliding Window Filter with Application to Planetary Landing". JFR.
