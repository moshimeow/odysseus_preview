# Problem builder design (Phase 1 of the systematic refactor)

Replaces hand-rolled sparse-LM bookkeeping (column offsets, sorted `(row,col)`
entry lists, value-write ordering, inline sigma/Huber weighting) with a
declarative API. Designed against the three real consumers:
`batch_joint_ba`, `ControllerFilter`, and `uwb_fusion` — plus the smaller
dense users (`refine_pose_pairs`, `validate_leds_*`).

## Goals

1. A residual is written **once** as a `#[real_fn]` closure; the library owns
   rows, sparsity, Jacobian layout, and weighting.
2. **No allocation during solve.** Structure is built once (`Problem` →
   `PreparedProblem`); iterations only overwrite value storage, matching
   today's `data_mut()` pattern and the ~4.5 ms filter-update budget
   (BENCHMARKS.md).
3. Factors carry their decorations: sigma or sqrt-info weighting, Huber loss,
   enabled/disabled (today's `use_* = sigma > 0.0` boolean forest).
4. Marginalization and dense priors are first-class, so fixed-lag consumers
   (ControllerFilter, uwb_fusion) stop reimplementing Schur complements.
5. Per-factor-kind diagnostics for free (per-kind RMS, JtJ block diagonal),
   replacing hand-maintained parallel bookkeeping in `batch_led_opt`.

## API sketch

```rust
use odysseus_solver::problem::{Problem, BlockKey, FactorKind};

let mut pb = Problem::new();

// 1. Parameter blocks. Key carries the tangent dim as a const generic;
//    the builder owns column offsets. Initial values live with the block.
let leds: Vec<BlockKey<3>> =
    (0..n_leds).map(|i| pb.add_block(current_leds[i])).collect();
let frames: Vec<BlockKey<15>> = ...;              // xi(6)+vel(3)+bg(3)+ba(3)
let gravity: BlockKey<3> = pb.add_block(gravity_init);

// Blocks can be frozen (anchor frames, fixed extrinsics): no columns
// allocated, values injected as constants into every factor that uses them.
pb.freeze(frames[0]);

// 2. Factors. The residual closure is generic over T: Real, written once.
//    N_IN = sum of block dims (compile-time), N_RES = residual dim.
//    `kind` groups factors for diagnostics and bulk enable/disable.
let visual = pb.kind("visual").sigma(visual_obs_sigma).huber(huber_delta_px);
for obs in &observations {
    pb.add_factor::<_, 9, 4>(          // 9 input dims (3 led + 6 pose), 4 residuals
        visual,
        (leds[obs.led_idx], pose_of(frames[obs.frame_idx])),
        move |(led, pose): (Vec3<T>, SE3Tangent<T>)| -> [T; 4] {
            stereo_reproj_residual(led, pose, &host[obs.frame_idx], &camera, &obs.uv)
        },
    );
}

// IMU pair factors: two 15-dim blocks + gravity, 9 residuals, sqrt-info weighted.
let imu = pb.kind("imu");
for pair in &imu_pairs {
    pb.add_factor_sqrt_info::<_, 33, 9>(
        imu,
        (frames[pair.prev], frames[pair.curr], gravity),
        pair.sqrt_info,
        move |(prev, curr, g)| imu_motion_residual(prev, curr, g, &pair.factor),
    );
}

// Stock factor helpers (thin wrappers over add_factor):
pb.prior(led_prior_kind, leds[i], led_prior[i], led_abs_sigma);   // absolute prior
pb.random_walk(bias_rw_kind, sub(frames[p], 9..12), sub(frames[c], 9..12), rw_sigma);
pb.dense_prior(marginalized_prior);                                // H, b from marginalization
// sigma == 0.0 on a kind ⇒ all its factors disabled (rows dropped at prepare()).

// 3. Prepare once, solve many. prepare() computes row layout + sorted CSR
//    entries and allocates all storage. Iterations only write values.
let mut prepared = pb.prepare();
let result = prepared.solve(&SolveOptions { max_iterations, ..Default::default() });

// 4. Structured access back out (replaces flat-vector unpacking and the
//    7-tuple returns).
let led_out: Vec3<f64> = result.block(leds[i]);
let per_kind_rms = result.kind_rms(visual);          // diagnostics for free
let jtj_diag = result.jtj_block_diagonal(frames[k]); // conditioning stats
```

### Marginalization

```rust
// Schur-complement the given blocks out of the current linearization;
// returns a DensePrior over the remaining (specified) blocks.
let prior: DensePrior = prepared.marginalize(&[frames[0]], &kept_blocks);
// Next window: pb.dense_prior(prior) — promoted from
// controller_filter::marginalize_front_odysseus + the 4 solver examples.
```

### Fixed-lag reuse (ControllerFilter / uwb_fusion, later phase)

Windows have identical structure each step; rebuilding `Problem` per frame
would allocate. `PreparedProblem::rebind` re-points block initial values and
factor payloads (observations, preintegration results) without recomputing
structure, as long as the shape (counts, sparsity) is unchanged. Shape changes
(LED dropout, variable obs count) fall back to `prepare()` — same behavior as
today's per-solve `SparseLevenbergMarquardt::new`, so this is an optimization,
not a correctness requirement, and can land after the ports.

## Implementation notes

- **Jet sizing.** Each factor's total input dim is a const generic (`N_IN`),
  so the residual runs with `Jet<f64, N_IN>` — same monomorphization pattern
  as today (9 for visual, 33 for IMU). Inputs are built with `to_variable`
  per block at its local offset; frozen blocks via `to_constant`. Output goes
  through `split_jets`; the M×N_IN local Jacobian scatters into CSR value
  storage through a precomputed index map (per-factor `Vec<usize>` of data
  indices, built at prepare()).
- **Tuple-of-blocks input.** Implement `add_factor` via a `BlockTuple` trait
  for 1..=4 blocks (macro-generated), each impl knowing how to build the jet
  input tuple and scatter columns. `sub(key, range)` gives a sub-block view
  (bias random walks touch 3 of 15 dims) — a `BlockKey<3>` borrowing parent
  columns.
- **Weighting/loss** are applied by the library after the closure returns:
  residual *= w, row *= w (sigma scalar or triangular sqrt-info multiply;
  Huber reweight from the unweighted residual norm — semantics copied from
  `apply_huber_loss` + batch_led_opt's inline version).
- **Backend split.** `prepare()` targets the sparse LDL backend. A dense
  backend (for ≤ ~200-param problems like `refine_pose_pairs`) reuses the
  same Problem API with an `SMatrix` system — this is where the unified LM
  core (one loop, two `LinearSystem` impls) pays off.
- **Solver core reuse.** `PreparedProblem::solve` wraps the existing LM loop
  (`solve_with_accept` semantics preserved — IRLS/proximal state hooks are
  used by batch_led_opt today).
- **Determinism.** Row order = factor insertion order; entries sorted
  (row, col) exactly as today, so ports should reproduce current results to
  FP-reassociation level. Golden tests (controller_tracking/tests/golden.rs)
  gate each port.

## Migration order

1. Land `problem` module + unit tests (line-fit, small BA) + a ported solver
   example (numberline_slam_marg) exercising marginalization.
2. Port `batch_joint_ba` (exercises every factor kind). Golden:
   `golden_validate_led_calibration`, `golden_validate_tracking`.
3. Port `ControllerFilter` solve + `optimize_led_positions` + marginalization.
   Golden: `golden_validate_tracking(_dual)` + filter-update timing
   (BENCHMARKS.md).
4. Port `uwb_fusion` (drops its `SlamMarginalization` dependency).
5. Port dense users (`refine_pose_pairs`, `validate_leds_*`).
6. Delete dead bookkeeping; odysseus-slam's optimization/ can follow
   opportunistically (not required to stay functional).

## Compile-time structure (discussed 2026-07-07)

Counts (n_obs, n_frames, n_leds) are runtime facts, so layout/sparsity stay
runtime-built-once; a type-level factor graph (HList-style) was considered and
rejected for ergonomics. The worthwhile compile-time upgrade, if profiling the
ControllerFilter port shows dispatch cost against the 4.5 ms budget, is typed
homogeneous factor storage (`FactorVec<F>` per factor type — static kinds,
dynamic counts) replacing `Box<dyn FnMut>` per factor, behind the same API.
Dimension literals get fixed separately by Phase 2's `derive(ParamBlock)`.
