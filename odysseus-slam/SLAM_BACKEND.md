# odysseus SLAM Backend - Technical Summary

## Overview

odysseus-slam is a **stereo SLAM backend** built with odysseus's autodiff & solver. It performs bundle adjustment to jointly optimize camera poses and 3D point positions from stereo observations.

## Key Features

### 1. **Automatic Differentiation**
- Uses `Jet` numbers for forward-mode autodiff
- Jacobians computed automatically - no manual derivatives!
- Generic over `Real` trait (works with `f64` and `f32`)

### 2. **Stereo Vision**
- Stereo camera model with baseline
- 4 residuals per observation: `(left_u, left_v, right_u, right_v)`
- Scale is directly observable (unlike monocular SLAM)

### 3. **Gauge Fixing**
- First camera fixed at identity to anchor coordinate system
- Prevents solution from floating in space
- Implemented by zeroing first 6 Jacobian columns

### 4. **Compile-Time Optimization**
- All matrices use const generics (stack-allocated)
- Zero heap allocations during optimization
- Jacobian is `SMatrix<f64, N_RESIDUALS, N_PARAMS>`

## Current Performance

### Accuracy (with proper setup)
- **Pose error: 0.98cm** (sub-centimeter!)
- **Point error: 5.6cm**
- Converges in ~27 iterations

### Problem Size
- 10 camera frames
- 15 3D points
- 107 stereo observations
- 428 residuals (4 per observation)
- Jacobian: 1024×128 (padded) ≈ 1MB on stack

## Critical Setup Parameters

### 1. **Room Position**
**CRITICAL:** Points must be in front of camera!

```rust
// ❌ BAD: Random points around origin
[-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]  // Only ~11 observations

// ✅ GOOD: Points 1m-4m in front
[-1.5, 1.5, -1.5, 1.5, 1.0, 4.0]   // ~107 observations!
```

This single change improved accuracy from **13cm → 0.98cm**!

### 2. **Random Perturbations**
Use truly random perturbations, not constant offsets:

```rust
// ❌ BAD: Constant perturbation
tangent[0] + 0.05  // All poses perturbed identically

// ✅ GOOD: Random perturbation
tangent[0] + rng.gen_range(-0.05..0.05)  // Each pose different
```

### 3. **Solver Parameters**
Different tolerances for f32 vs f64:

```rust
// f64: High precision
LevenbergMarquardt::<f64, N, M>::new()
    .with_tolerance(1e-10)
    .with_max_iterations(500)

// f32: Lower precision
LevenbergMarquardt::<f32, N, M>::new()
    .with_tolerance(1e-6)   // Looser!
    .with_max_iterations(500)
```

### 4. **Lambda Divergence Detection**
Early termination when optimization diverges:

```rust
if !lambda.is_finite() || lambda > 1e12 {
    println!("❌ Lambda diverged");
    break;
}
```

## Stack Size Limitations

### The Problem
Const-generic matrices live on the stack:
- Jacobian: `N_RESIDUALS × N_PARAMS × 8` bytes
- J^T J: `N_PARAMS × N_PARAMS × 8` bytes
- Plus temporaries during matrix inversion

### Limits Discovered
- ✅ 512×128 ≈ 512KB - Works
- ✅ 1024×128 ≈ 1MB - Works in release
- ❌ 2048×256 ≈ 4MB - Stack overflow!

### Debug vs Release
**Debug mode has smaller stack!** Use `--release` for larger problems.

```bash
# Debug mode: Often stack overflows
cargo run --example stereo_slam_demo

# Release mode: Much larger stack
cargo run --example stereo_slam_demo --release
```

## Basin of Attraction

From GAUGE_FIXING_RESULTS.md:

### ✅ What Works
- **Small perturbations (0.1m, 3°)**: Perfect convergence
- **Large perturbations (1.0m, 30°)**: Still converges!
- Basin of attraction is HUGE for stereo

### ❌ What Fails
- **Random initialization**: Gets stuck in local minima
- Need reasonable initial guess (within ~1m of truth)

## Visibility and Point Recovery

### Question: Can points come back into frustum?

**Answer: YES!**

During optimization, if:
- True point is inside frustum (observation exists)
- Estimated point is outside frustum (bad projection)

Then the large residual will pull the estimated point back inside. No visibility caching - observations are fixed at initialization.

## Architecture

### Core Types

```rust
// Camera models
pub struct PinholeCamera<T: Real> {
    fx, fy, cx, cy: T
}

pub struct StereoCamera<T: Real> {
    left: PinholeCamera<T>,
    baseline: T,
}

// Observations
pub struct StereoObservation {
    point_id: usize,
    camera_id: usize,
    left_u, left_v: f64,
    right_u, right_v: f64,
}

// Optimization problem
pub struct StereoBundleAdjustmentProblem<
    const N_PARAMS: usize,
    const N_RESIDUALS: usize
> {
    stereo_camera: StereoCamera<f64>,
    observations: Vec<StereoObservation>,
    n_cameras: usize,
    n_points: usize,
}
```

### Cost Function

```rust
pub fn stereo_reprojection_residual<T: Real>(
    pose_tangent: &[T; 6],          // SE3 in tangent space
    point: &Vec3<T>,                 // 3D point
    stereo_camera: &StereoCamera<T>,
    observed_left_u, observed_left_v: T,
    observed_right_u, observed_right_v: T,
) -> (T, T, T, T)  // 4 residuals
```

## What's Next?

### Production Requirements

For real-world SLAM, we still need:

1. **Initialization Strategy**
   - Can't start from ground truth!
   - Options: Stereo triangulation + PnP, Visual odometry, Essential matrix
   - See INITIALIZATION.md

2. **Feature Detection/Matching**
   - ORB features
   - BRIEF descriptors
   - Hamming distance matching

3. **Outlier Rejection**
   - RANSAC for robust estimation
   - Huber loss for bundle adjustment

4. **Incremental Processing**
   - Add frames one at a time
   - Sliding window optimization
   - Marginalization of old frames

5. **Loop Closure**
   - Detect revisited locations
   - Add loop closure constraints
   - Pose graph optimization

### Current Limitations

1. **Stack size** limits problem size
   - Max ~1000 residuals, ~100 parameters
   - For larger problems, need heap-allocated dynamic version

2. **No outlier handling**
   - Assumes all observations are inliers
   - Needs robust cost functions

3. **Batch processing only**
   - Optimizes all frames at once
   - Needs incremental/sliding window version

4. **Fixed camera intrinsics**
   - Camera parameters not optimized
   - Could add to parameter vector if needed

## Lessons Learned

### 1. Observation Count Matters
More observations = better constraints = better accuracy

- 11 observations → 13cm error
- 107 observations → 0.98cm error

### 2. Stack Size is Real
Const generics are fast but limited by stack size. For production, may need hybrid approach.

### 3. Gauge Fixing is Essential
Without anchoring first camera, solution floats arbitrarily. Gauge fixing is not optional!

### 4. Debug Mode is Strict
Debug builds have:
- Smaller stack (more overflows)
- NaN checks (good for catching bugs!)
- Slower execution

Always test in release for realistic performance.

### 5. Stereo Helps Tremendously
Having depth from stereo:
- Fixes scale ambiguity
- Provides strong constraints
- Enables large basin of attraction

Much easier than monocular SLAM!

## Code Quality

### What's Good
- ✅ Zero heap allocations (as designed)
- ✅ Generic over f32/f64
- ✅ Automatic Jacobians via autodiff
- ✅ Clean separation of concerns
- ✅ Comprehensive test coverage

### What Could Improve
- ⚠️ Stack overflow handling (better error messages?)
- ⚠️ No runtime size checking (all compile-time)
- ⚠️ Debug mode performance (very slow)

## Running the Demo

```bash
# Release mode (recommended)
cargo run --example stereo_slam_demo --release

# View in Rerun
# Blue = ground truth, Orange = optimized estimate
# Gray lines = error between GT and estimate
```

Expected output:
```
📊 Accuracy:
  Average pose error: 0.0098 m  (< 1cm!)
  Average point error: 0.0563 m

✅ Tests passed! Recovered trajectory is accurate.
```

## References

- GAUGE_FIXING_RESULTS.md - Experiments with initialization
- INITIALIZATION.md - Production initialization strategies
- examples/stereo_slam_demo.rs - Full working example

---

**Bottom line:** Odysseus SLAM demonstrates that compile-time autodiff + stack-allocated optimization can achieve sub-centimeter accuracy for small-scale SLAM problems. For production, we'd need incremental processing and feature matching, but the core optimizer works beautifully!
