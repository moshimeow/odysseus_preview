//! Declarative nonlinear least-squares problem builder.
//!
//! Replaces hand-rolled sparse-LM bookkeeping (column offsets, sorted
//! `(row, col)` entry lists, Jacobian value-write ordering, inline
//! sigma/Huber weighting) with a small API:
//!
//! 1. Declare parameter blocks (`add_block`); the builder owns column layout.
//! 2. Group factors into named kinds carrying weighting decorations
//!    (`kind("visual").sigma(..).huber(..).key()`).
//! 3. Add factors: a residual closure generic over the jet, plus the block
//!    refs it touches (`add_factor1/2/3`). Rows, sparsity, and Jacobian
//!    scatter are derived.
//! 4. `prepare()` once — all layout computed, all storage allocated.
//!    `solve()` iterations only overwrite values (no allocation).
//!
//! Blocks are plain ℝⁿ vectors. Manifold states (SE3 poses) follow the
//! existing convention: the block holds a tangent increment (initialized at
//! zero) and the residual closure composes it with a host value captured in
//! the closure payload.
//!
//! Weighting order per factor: Huber first (per-element, on raw residual
//! units — e.g. pixels), then sigma scaling (`r/σ`) or sqrt-info multiply.
//! This matches `batch_led_opt`'s inline Huber-then-sigma and
//! odysseus-slam's `apply_huber_loss`.
//!
//! Setting a kind's sigma to `0.0` disables all its factors (rows dropped at
//! `prepare()`), replacing the `use_x = sigma > 0.0` boolean forests.

use crate::sparse_solver::{IterationResult, SparseLevenbergMarquardt};
use crate::Jet;
use nalgebra::DVector;


// ── Blocks ───────────────────────────────────────────────────────────────────

/// Handle to a parameter block of tangent dimension `D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockKey<const D: usize> {
    block: usize,
}

/// A view of `S` consecutive dims of a block, starting at `offset`.
/// Factors take `BlockRef`s; a whole-block `BlockKey` converts implicitly.
#[derive(Clone, Copy, Debug)]
pub struct BlockRef<const S: usize> {
    block: usize,
    offset: usize,
}

impl<const D: usize> BlockKey<D> {
    /// View a sub-range of this block (e.g. the gyro-bias dims of a
    /// 15-dim frame state).
    pub fn sub<const S: usize>(self, offset: usize) -> BlockRef<S> {
        assert!(offset + S <= D, "sub-block {}..{} exceeds dim {}", offset, offset + S, D);
        BlockRef { block: self.block, offset }
    }
}

impl<const D: usize> From<BlockKey<D>> for BlockRef<D> {
    fn from(key: BlockKey<D>) -> Self {
        BlockRef { block: key.block, offset: 0 }
    }
}

struct BlockInfo {
    dim: usize,
    values: Vec<f64>,
    frozen: bool,
    /// Column of the first dim, or usize::MAX when frozen. Set at prepare().
    col: usize,
}

// ── Kinds ────────────────────────────────────────────────────────────────────

/// Handle to a factor kind: a named group sharing weighting decorations,
/// reported together in diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactorKind(usize);

struct KindInfo {
    name: String,
    /// `Some(0.0)` disables the kind's factors entirely.
    sigma: Option<f64>,
    huber_delta: Option<f64>,
}

impl KindInfo {
    fn enabled(&self) -> bool {
        self.sigma != Some(0.0)
    }
}

/// Builder for a factor kind's decorations; finish with [`KindBuilder::key`].
pub struct KindBuilder<'p> {
    problem: &'p mut Problem,
    index: usize,
}

impl KindBuilder<'_> {
    /// Scale this kind's residuals by `1/sigma`. `0.0` disables the kind.
    pub fn sigma(self, sigma: f64) -> Self {
        self.problem.kinds[self.index].sigma = Some(sigma);
        self
    }

    /// Huber loss kink, in raw residual units, applied per element before
    /// sigma scaling. `0.0` means pure L2 (no-op).
    pub fn huber(self, delta: f64) -> Self {
        self.problem.kinds[self.index].huber_delta =
            if delta > 0.0 { Some(delta) } else { None };
        self
    }

    pub fn key(self) -> FactorKind {
        FactorKind(self.index)
    }
}

// ── Factors ──────────────────────────────────────────────────────────────────

/// Type-erased factor evaluator: `(stacked_input, residuals_out, local_jac_out)`.
/// `local_jac_out` is row-major `n_res × n_in`. Produced by the typed
/// `add_factor*` wrappers, which monomorphize the jet plumbing.
type ErasedEval = Box<dyn FnMut(&[f64], &mut [f64], &mut [f64])>;

struct FactorInfo {
    kind: usize,
    /// (block index, offset within block, dim) per input, in closure order.
    inputs: Vec<(usize, usize, usize)>,
    n_res: usize,
    eval: ErasedEval,
    /// Row-major `n_res × n_res` square-root information matrix; applied
    /// instead of the kind's sigma when present.
    sqrt_info: Option<Vec<f64>>,
}

// ── Problem ──────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct Problem {
    blocks: Vec<BlockInfo>,
    kinds: Vec<KindInfo>,
    factors: Vec<FactorInfo>,
}

impl Problem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a parameter block with initial values.
    pub fn add_block<const D: usize>(&mut self, initial: impl Into<[f64; D]>) -> BlockKey<D> {
        let block = self.blocks.len();
        self.blocks.push(BlockInfo {
            dim: D,
            values: initial.into().to_vec(),
            frozen: false,
            col: usize::MAX,
        });
        BlockKey { block }
    }

    /// Freeze a block: no columns are allocated for it and its values enter
    /// factors as constants (e.g. anchor frames, fixed extrinsics).
    pub fn freeze<const D: usize>(&mut self, key: BlockKey<D>) {
        self.blocks[key.block].frozen = true;
    }

    /// Start declaring a factor kind (see [`KindBuilder`]).
    pub fn kind(&mut self, name: &str) -> KindBuilder<'_> {
        let index = self.kinds.len();
        self.kinds.push(KindInfo {
            name: name.to_string(),
            sigma: None,
            huber_delta: None,
        });
        KindBuilder { problem: self, index }
    }

    fn push_factor(
        &mut self,
        kind: FactorKind,
        inputs: Vec<(usize, usize, usize)>,
        n_res: usize,
        eval: ErasedEval,
        sqrt_info: Option<Vec<f64>>,
    ) {
        for &(block, offset, dim) in &inputs {
            assert!(
                offset + dim <= self.blocks[block].dim,
                "factor input exceeds block dims"
            );
        }
        self.factors.push(FactorInfo {
            kind: kind.0,
            inputs,
            n_res,
            eval,
            sqrt_info,
        });
    }

}

/// Generates `add_factor{1,2,3}` (+ `_sqrt_info` variants): the typed wrappers
/// that build jet inputs at local indices, run the residual closure, and write
/// the local Jacobian row-major.
macro_rules! impl_add_factor {
    ($name:ident, $name_si:ident, $($bn:ident : $Dn:ident),+) => {
        impl Problem {
            #[allow(clippy::too_many_arguments)]
            pub fn $name<const N_IN: usize, const N_RES: usize, $(const $Dn: usize,)+ F>(
                &mut self,
                kind: FactorKind,
                $($bn: impl Into<BlockRef<$Dn>>,)+
                f: F,
            ) where
                F: Fn($([Jet<f64, N_IN>; $Dn],)+) -> [Jet<f64, N_IN>; N_RES] + 'static,
            {
                assert_eq!(0 $(+ $Dn)+, N_IN, "block dims must sum to N_IN");
                $(let $bn: BlockRef<$Dn> = $bn.into();)+
                let inputs = vec![$(($bn.block, $bn.offset, $Dn),)+];
                let eval: ErasedEval = Box::new(move |input, res, jac| {
                    let mut base = 0usize;
                    $(
                        let $bn: [Jet<f64, N_IN>; $Dn] = std::array::from_fn(|i| {
                            Jet::variable(input[base + i], base + i)
                        });
                        base += $Dn;
                    )+
                    let _ = base;
                    let out = f($($bn,)+);
                    for r in 0..N_RES {
                        res[r] = out[r].value;
                        jac[r * N_IN..(r + 1) * N_IN].copy_from_slice(&out[r].derivs);
                    }
                });
                self.push_factor(kind, inputs, N_RES, eval, None);
            }

            #[allow(clippy::too_many_arguments)]
            pub fn $name_si<const N_IN: usize, const N_RES: usize, $(const $Dn: usize,)+ F>(
                &mut self,
                kind: FactorKind,
                $($bn: impl Into<BlockRef<$Dn>>,)+
                sqrt_info: nalgebra::SMatrix<f64, N_RES, N_RES>,
                f: F,
            ) where
                F: Fn($([Jet<f64, N_IN>; $Dn],)+) -> [Jet<f64, N_IN>; N_RES] + 'static,
            {
                assert_eq!(0 $(+ $Dn)+, N_IN, "block dims must sum to N_IN");
                $(let $bn: BlockRef<$Dn> = $bn.into();)+
                let inputs = vec![$(($bn.block, $bn.offset, $Dn),)+];
                let si: Vec<f64> = (0..N_RES)
                    .flat_map(|r| (0..N_RES).map(move |c| sqrt_info[(r, c)]))
                    .collect();
                let eval: ErasedEval = Box::new(move |input, res, jac| {
                    let mut base = 0usize;
                    $(
                        let $bn: [Jet<f64, N_IN>; $Dn] = std::array::from_fn(|i| {
                            Jet::variable(input[base + i], base + i)
                        });
                        base += $Dn;
                    )+
                    let _ = base;
                    let out = f($($bn,)+);
                    for r in 0..N_RES {
                        res[r] = out[r].value;
                        jac[r * N_IN..(r + 1) * N_IN].copy_from_slice(&out[r].derivs);
                    }
                });
                self.push_factor(kind, inputs, N_RES, eval, Some(si));
            }
        }
    };
}

impl_add_factor!(add_factor1, add_factor1_sqrt_info, b1: D1);
impl_add_factor!(add_factor2, add_factor2_sqrt_info, b1: D1, b2: D2);
impl_add_factor!(add_factor3, add_factor3_sqrt_info, b1: D1, b2: D2, b3: D3);

// ── Stock factor helpers ─────────────────────────────────────────────────────

impl Problem {
    /// Absolute prior: `r = (x - target) / sigma` (sigma from the kind).
    pub fn prior<const D: usize>(
        &mut self,
        kind: FactorKind,
        b: impl Into<BlockRef<D>>,
        target: [f64; D],
    ) {
        self.add_factor1::<D, D, D, _>(kind, b, move |x: [Jet<f64, D>; D]| {
            std::array::from_fn(|i| x[i] - target[i])
        });
    }

    /// Random-walk pair factor: `r = curr - prev` (weighted by the kind).
    /// The Jacobian is constant (±1), so this skips the jet plumbing — and
    /// avoids needing `N_IN = 2*D` const arithmetic, which stable Rust lacks.
    pub fn random_walk<const D: usize>(
        &mut self,
        kind: FactorKind,
        prev: impl Into<BlockRef<D>>,
        curr: impl Into<BlockRef<D>>,
    ) {
        let p: BlockRef<D> = prev.into();
        let c: BlockRef<D> = curr.into();
        let inputs = vec![(p.block, p.offset, D), (c.block, c.offset, D)];
        let eval: ErasedEval = Box::new(move |input, res, jac| {
            let n_in = 2 * D;
            jac[..D * n_in].fill(0.0);
            for i in 0..D {
                res[i] = input[D + i] - input[i];
                jac[i * n_in + i] = -1.0;
                jac[i * n_in + D + i] = 1.0;
            }
        });
        self.push_factor(kind, inputs, D, eval, None);
    }
}

/// Apply Huber loss to a single residual and its Jacobian row, in place.
/// Below `huber_delta` the residual is untouched (L2); above it both residual
/// and row are reweighted by sqrt(delta/|r|) (L1 tail). This is the same
/// per-element convention the problem builder applies via `KindBuilder::huber`;
/// exposed for hand-rolled cost functions.
#[inline]
pub fn apply_huber_loss(huber_delta: f64, residual: &mut f64, jacobian_row: &mut [f64]) {
    let abs_r = residual.abs();
    if abs_r > huber_delta {
        let weight = (huber_delta / abs_r).sqrt();
        *residual *= weight;
        for j in jacobian_row.iter_mut() {
            *j *= weight;
        }
    }
}

// ── Erased block refs, dyn factors, marginal priors ─────────────────────────

/// Type-erased block handle (dimension checked at runtime). Used where block
/// sets are dynamic: marginalization and marginal-prior attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockId(usize);

impl<const D: usize> From<BlockKey<D>> for BlockId {
    fn from(key: BlockKey<D>) -> Self {
        BlockId(key.block)
    }
}

/// Dense prior produced by [`PreparedProblem::marginalize`]: constrains the
/// kept blocks with `r = sqrt_info · (x − linearization_point)`. Matches the
/// existing `SlamMarginalization` / controller-filter `DensePrior` convention
/// (the gradient term vanishes because marginalization runs at the post-solve
/// optimum).
#[derive(Clone)]
pub struct MarginalPrior {
    /// Dimensions of the constrained blocks, in attachment order.
    pub block_dims: Vec<usize>,
    /// Upper-triangular Cholesky factor of the Schur-complement Hessian.
    pub sqrt_info: nalgebra::DMatrix<f64>,
    pub linearization_point: DVector<f64>,
}

impl Problem {
    /// Fully type-erased factor: caller provides the weighted-input layout and
    /// an evaluator writing raw residuals + row-major local Jacobian. The
    /// typed `add_factor*` wrappers are preferred; this exists for factors
    /// whose shape is only known at runtime (e.g. marginal priors).
    pub fn add_factor_dyn(
        &mut self,
        kind: FactorKind,
        blocks: &[(BlockId, usize, usize)], // (block, offset, dim)
        n_res: usize,
        eval: impl FnMut(&[f64], &mut [f64], &mut [f64]) + 'static,
    ) {
        let inputs = blocks.iter().map(|&(b, o, d)| (b.0, o, d)).collect();
        self.push_factor(kind, inputs, n_res, Box::new(eval), None);
    }

    /// Attach a marginal prior to `blocks` (same order and dims as the
    /// `kept` blocks at [`PreparedProblem::marginalize`] time). The prior's
    /// own sqrt-info supplies the weighting, superseding the kind's sigma;
    /// don't put a Huber decoration on the kind.
    pub fn add_marginal_prior(
        &mut self,
        kind: FactorKind,
        prior: MarginalPrior,
        blocks: &[BlockId],
    ) {
        assert_eq!(
            blocks.len(),
            prior.block_dims.len(),
            "marginal prior block count mismatch"
        );
        let mut inputs = Vec::with_capacity(blocks.len());
        for (&b, &dim) in blocks.iter().zip(&prior.block_dims) {
            assert_eq!(
                self.blocks[b.0].dim, dim,
                "marginal prior block dim mismatch"
            );
            inputs.push((b.0, 0usize, dim));
        }
        let n = prior.linearization_point.len();
        let si: Vec<f64> = (0..n)
            .flat_map(|r| (0..n).map(|c| prior.sqrt_info[(r, c)]).collect::<Vec<_>>())
            .collect();
        let x0 = prior.linearization_point;
        // Raw residual is the plain deviation with identity Jacobian; the
        // stored sqrt-info is applied by the generic weighting path.
        let eval: ErasedEval = Box::new(move |input, res, jac| {
            jac[..n * n].fill(0.0);
            for r in 0..n {
                res[r] = input[r] - x0[r];
                jac[r * n + r] = 1.0;
            }
        });
        self.push_factor(kind, inputs, n, eval, Some(si));
    }
}

// ── Prepared problem ─────────────────────────────────────────────────────────

struct FactorLayout {
    row_start: usize,
    /// CSR data index for each (res_row, local_col), row-major;
    /// `usize::MAX` for frozen columns (value not scattered).
    scatter: Vec<usize>,
    /// Effective per-factor weighting resolved from the kind at prepare time.
    sigma_inv: f64,
    huber_delta: Option<f64>,
}

pub struct SolveOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub verbose: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-10,
            verbose: false,
        }
    }
}

pub struct SolveSummary {
    pub iterations: usize,
    pub final_error: f64,
}

/// Read-only view of block values at a solver iterate, handed to the
/// `on_accept` callback of [`PreparedProblem::solve_with_accept`].
pub struct BlockValues<'a> {
    params: &'a DVector<f64>,
    blocks: &'a [BlockInfo],
}

impl BlockValues<'_> {
    pub fn get<const D: usize>(&self, key: BlockKey<D>) -> [f64; D] {
        let b = &self.blocks[key.block];
        std::array::from_fn(|i| {
            if b.frozen {
                b.values[i]
            } else {
                self.params[b.col + i]
            }
        })
    }
}

pub struct PreparedProblem {
    blocks: Vec<BlockInfo>,
    kinds: Vec<KindInfo>,
    factors: Vec<FactorInfo>,
    layouts: Vec<FactorLayout>,
    /// Indices of enabled factors, in row order.
    active: Vec<usize>,
    solver: SparseLevenbergMarquardt<f64>,
    n_rows: usize,
    n_cols: usize,
    /// Scratch: stacked factor inputs, raw and weighted residuals/Jacobians.
    scratch_in: Vec<f64>,
    scratch_res: Vec<f64>,
    scratch_jac: Vec<f64>,
    scratch_wres: Vec<f64>,
    scratch_wjac: Vec<f64>,
    /// Final residual vector from the last solve (for diagnostics).
    final_residuals: Vec<f64>,
}

/// Evaluate one factor at `params` (or at stored block values when `None`),
/// applying Huber then sigma/sqrt-info weighting. Writes weighted residuals
/// into `wres` and the weighted dense local Jacobian (row-major
/// `n_res × n_in`) into `wjac`. Returns `n_in`.
///
/// Shared by the solve cost closure, the post-solve diagnostics pass, and
/// marginalization, so weighting semantics can't drift between them.
#[allow(clippy::too_many_arguments)]
fn eval_factor_weighted(
    blocks: &[BlockInfo],
    factor: &mut FactorInfo,
    layout: &FactorLayout,
    params: Option<&DVector<f64>>,
    scratch_in: &mut [f64],
    scratch_res: &mut [f64],
    scratch_jac: &mut [f64],
    wres: &mut [f64],
    wjac: &mut [f64],
) -> usize {
    let n_res = factor.n_res;
    let n_in: usize = factor.inputs.iter().map(|&(_, _, d)| d).sum();

    // Gather inputs: unfrozen from params (when given), frozen from storage.
    let mut local = 0usize;
    for &(block, offset, dim) in &factor.inputs {
        let b = &blocks[block];
        for d in 0..dim {
            scratch_in[local] = match (b.frozen, params) {
                (false, Some(p)) => p[b.col + offset + d],
                _ => b.values[offset + d],
            };
            local += 1;
        }
    }

    let res = &mut scratch_res[..n_res];
    let jac = &mut scratch_jac[..n_res * n_in];
    (factor.eval)(&scratch_in[..n_in], res, jac);

    // Huber (per element, raw units).
    if let Some(delta) = layout.huber_delta {
        for r in 0..n_res {
            let abs_r = res[r].abs();
            if abs_r > delta {
                let w = (delta / abs_r).sqrt();
                res[r] *= w;
                for c in 0..n_in {
                    jac[r * n_in + c] *= w;
                }
            }
        }
    }

    // Sigma scaling or sqrt-info multiply.
    if let Some(si) = &factor.sqrt_info {
        for r in 0..n_res {
            let mut acc = 0.0;
            for k in 0..n_res {
                acc += si[r * n_res + k] * res[k];
            }
            wres[r] = acc;
            for c in 0..n_in {
                let mut jacc = 0.0;
                for k in 0..n_res {
                    jacc += si[r * n_res + k] * jac[k * n_in + c];
                }
                wjac[r * n_in + c] = jacc;
            }
        }
    } else {
        let w = layout.sigma_inv;
        for r in 0..n_res {
            wres[r] = res[r] * w;
            for c in 0..n_in {
                wjac[r * n_in + c] = jac[r * n_in + c] * w;
            }
        }
    }

    n_in
}

impl Problem {
    /// Compute row/column layout and sparsity, allocate all solver storage.
    pub fn prepare(mut self) -> PreparedProblem {
        // Column layout.
        let mut n_cols = 0usize;
        for b in &mut self.blocks {
            if !b.frozen {
                b.col = n_cols;
                n_cols += b.dim;
            }
        }

        // Row layout over enabled factors.
        let mut active = Vec::new();
        let mut n_rows = 0usize;
        let mut max_in = 0usize;
        let mut max_res = 0usize;
        let mut row_starts = Vec::with_capacity(self.factors.len());
        for (fi, f) in self.factors.iter().enumerate() {
            if !self.kinds[f.kind].enabled() {
                row_starts.push(usize::MAX);
                continue;
            }
            active.push(fi);
            row_starts.push(n_rows);
            n_rows += f.n_res;
            let n_in: usize = f.inputs.iter().map(|&(_, _, d)| d).sum();
            max_in = max_in.max(n_in);
            max_res = max_res.max(f.n_res);
        }

        // Sparsity entries, sorted (row, col); duplicates are a caller bug.
        let mut entries: Vec<(usize, usize)> = Vec::new();
        for &fi in &active {
            let f = &self.factors[fi];
            let row0 = row_starts[fi];
            for r in 0..f.n_res {
                for &(block, offset, dim) in &f.inputs {
                    let b = &self.blocks[block];
                    if b.frozen {
                        continue;
                    }
                    for d in 0..dim {
                        entries.push((row0 + r, b.col + offset + d));
                    }
                }
            }
        }
        entries.sort_unstable();
        for w in entries.windows(2) {
            assert!(
                w[0] != w[1],
                "duplicate Jacobian entry {:?}: a factor references overlapping block dims",
                w[0]
            );
        }

        // Columns untouched by any factor are legal (the sparse damping
        // inserts pure-λ diagonals for them, so they simply don't move —
        // e.g. a first-frame velocity before any motion factor exists).

        // CSR data-index lookup: entries sorted by (row, col) *is* CSR data
        // order, so the position in `entries` is the data index.
        let scatter_index = |row: usize, col: usize| -> usize {
            entries
                .binary_search(&(row, col))
                .expect("scatter target must be a declared entry")
        };

        let mut layouts = Vec::with_capacity(self.factors.len());
        for (fi, f) in self.factors.iter().enumerate() {
            let kind = &self.kinds[f.kind];
            if row_starts[fi] == usize::MAX {
                layouts.push(FactorLayout {
                    row_start: usize::MAX,
                    scatter: Vec::new(),
                    sigma_inv: 1.0,
                    huber_delta: None,
                });
                continue;
            }
            let row0 = row_starts[fi];
            let n_in: usize = f.inputs.iter().map(|&(_, _, d)| d).sum();
            let mut scatter = vec![usize::MAX; f.n_res * n_in];
            for r in 0..f.n_res {
                let mut local = 0usize;
                for &(block, offset, dim) in &f.inputs {
                    let b = &self.blocks[block];
                    for d in 0..dim {
                        if !b.frozen {
                            scatter[r * n_in + local] =
                                scatter_index(row0 + r, b.col + offset + d);
                        }
                        local += 1;
                    }
                }
            }
            let sigma_inv = match (f.sqrt_info.is_some(), kind.sigma) {
                (true, _) => 1.0, // sqrt-info supersedes the kind sigma
                (false, Some(s)) => 1.0 / s,
                (false, None) => 1.0,
            };
            layouts.push(FactorLayout {
                row_start: row0,
                scatter,
                sigma_inv,
                huber_delta: kind.huber_delta,
            });
        }

        let solver = SparseLevenbergMarquardt::<f64>::new(n_rows, n_cols, &entries);

        PreparedProblem {
            blocks: self.blocks,
            kinds: self.kinds,
            factors: self.factors,
            layouts,
            active,
            solver,
            n_rows,
            n_cols,
            scratch_in: vec![0.0; max_in],
            scratch_res: vec![0.0; max_res],
            scratch_jac: vec![0.0; max_in * max_res],
            scratch_wres: vec![0.0; max_res],
            scratch_wjac: vec![0.0; max_in * max_res],
            final_residuals: vec![0.0; n_rows],
        }
    }
}

impl PreparedProblem {
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Current values of a block (initial values before solve, optimized after).
    pub fn block<const D: usize>(&self, key: BlockKey<D>) -> [f64; D] {
        let b = &self.blocks[key.block];
        std::array::from_fn(|i| b.values[i])
    }

    /// Overwrite a block's current values (e.g. rebinding a sliding window).
    pub fn set_block<const D: usize>(&mut self, key: BlockKey<D>, values: [f64; D]) {
        self.blocks[key.block].values.copy_from_slice(&values);
    }

    /// RMS of a kind's residuals from the last solve, in weighted units.
    pub fn kind_rms(&self, kind: FactorKind) -> f64 {
        let mut sum_sq = 0.0;
        let mut n = 0usize;
        for &fi in &self.active {
            let f = &self.factors[fi];
            if f.kind != kind.0 {
                continue;
            }
            let row0 = self.layouts[fi].row_start;
            for r in 0..f.n_res {
                sum_sq += self.final_residuals[row0 + r].powi(2);
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            (sum_sq / n as f64).sqrt()
        }
    }

    pub fn kind_name(&self, kind: FactorKind) -> &str {
        &self.kinds[kind.0].name
    }

    /// Look up a kind by its declared name (first match).
    pub fn kind_by_name(&self, name: &str) -> Option<FactorKind> {
        self.kinds
            .iter()
            .position(|k| k.name == name)
            .map(FactorKind)
    }

    /// Diagonal of J^T J from the last accepted iteration, per block dim.
    pub fn jtj_block_diagonal<const D: usize>(&self, key: BlockKey<D>) -> [f64; D] {
        let b = &self.blocks[key.block];
        assert!(!b.frozen, "frozen blocks have no JtJ diagonal");
        let diag = self.solver.last_jtj_diagonal();
        std::array::from_fn(|i| diag[b.col + i])
    }

    /// Full J^T J diagonal from the last accepted iteration, in column order
    /// (block declaration order).
    pub fn jtj_diagonal(&self) -> Vec<f64> {
        self.solver.last_jtj_diagonal().as_slice().to_vec()
    }

    /// Dense weighted Gauss-Newton Hessian J^T J over all columns, evaluated
    /// at the current block values. For covariance extraction and
    /// conditioning diagnostics on small problems — O(n_cols²) memory.
    pub fn dense_hessian(&mut self) -> nalgebra::DMatrix<f64> {
        let mut h = nalgebra::DMatrix::<f64>::zeros(self.n_cols, self.n_cols);
        for &fi in &self.active {
            let f = &mut self.factors[fi];
            let lay = &self.layouts[fi];
            let n_in = eval_factor_weighted(
                &self.blocks, f, lay, None,
                &mut self.scratch_in, &mut self.scratch_res, &mut self.scratch_jac,
                &mut self.scratch_wres, &mut self.scratch_wjac,
            );
            let mut cols = Vec::with_capacity(n_in);
            for &(b, offset, dim) in &f.inputs {
                for d in 0..dim {
                    cols.push(if self.blocks[b].frozen {
                        usize::MAX
                    } else {
                        self.blocks[b].col + offset + d
                    });
                }
            }
            for r in 0..f.n_res {
                let row = &self.scratch_wjac[r * n_in..(r + 1) * n_in];
                for (ci, &gci) in cols.iter().enumerate() {
                    if gci == usize::MAX {
                        continue;
                    }
                    for (cj, &gcj) in cols.iter().enumerate() {
                        if gcj == usize::MAX {
                            continue;
                        }
                        h[(gci, gcj)] += row[ci] * row[cj];
                    }
                }
            }
        }
        h
    }

    /// Run LM to convergence. Block values are updated in place; call
    /// [`PreparedProblem::block`] to read results.
    pub fn solve(&mut self, opts: &SolveOptions) -> SolveSummary {
        self.solve_with_accept(opts, |_| {})
    }

    /// Like [`solve`], with a callback fired exactly when LM accepts a step
    /// (mirroring `SparseLevenbergMarquardt::solve_with_accept`). Use it to
    /// refresh cost-function state that should track the current
    /// linearization point — proximal anchors, IRLS reweighting — via shared
    /// state (`Rc<RefCell<..>>`) captured by both the factor closures and
    /// this callback.
    ///
    /// [`solve`]: PreparedProblem::solve
    pub fn solve_with_accept(
        &mut self,
        opts: &SolveOptions,
        mut on_accept: impl FnMut(&BlockValues<'_>),
    ) -> SolveSummary {
        // Stack unfrozen block values into the parameter vector.
        let mut x0 = DVector::zeros(self.n_cols);
        for b in &self.blocks {
            if !b.frozen {
                for i in 0..b.dim {
                    x0[b.col + i] = b.values[i];
                }
            }
        }

        self.solver.tolerance = opts.tolerance;
        self.solver.max_iterations = opts.max_iterations;
        self.solver.verbose = opts.verbose;

        // Split borrows so the cost closure can use factor state while the
        // solver owns itself.
        let blocks = &self.blocks;
        let factors = &mut self.factors;
        let layouts = &self.layouts;
        let active = &self.active;
        let scratch_in = &mut self.scratch_in;
        let scratch_res = &mut self.scratch_res;
        let scratch_jac = &mut self.scratch_jac;
        let scratch_wres = &mut self.scratch_wres;
        let scratch_wjac = &mut self.scratch_wjac;

        let mut iterations = 0usize;
        let mut final_error = 0.0f64;

        let params = self.solver.solve_with_accept(
            x0,
            |params: &DVector<f64>, residuals: &mut [f64], jac_data: &mut [f64]| {
                for &fi in active {
                    let f = &mut factors[fi];
                    let lay = &layouts[fi];
                    let n_in = eval_factor_weighted(
                        blocks, f, lay, Some(params),
                        scratch_in, scratch_res, scratch_jac,
                        scratch_wres, scratch_wjac,
                    );
                    let row0 = lay.row_start;
                    for r in 0..f.n_res {
                        residuals[row0 + r] = scratch_wres[r];
                        for c in 0..n_in {
                            let idx = lay.scatter[r * n_in + c];
                            if idx != usize::MAX {
                                jac_data[idx] = scratch_wjac[r * n_in + c];
                            }
                        }
                    }
                }
            },
            |iter: usize, result: &IterationResult<f64>, _params: &DVector<f64>| {
                iterations = iter + 1;
                final_error = result.error;
            },
            |accepted: &DVector<f64>| {
                on_accept(&BlockValues {
                    params: accepted,
                    blocks,
                });
            },
        );

        // Write optimized values back into blocks.
        for b in &mut self.blocks {
            if !b.frozen {
                for i in 0..b.dim {
                    b.values[i] = params[b.col + i];
                }
            }
        }

        // Re-evaluate residuals at the solution for diagnostics.
        // (One extra evaluation; cheap relative to the solve.)
        let mut final_res = std::mem::take(&mut self.final_residuals);
        for &fi in &self.active {
            let f = &mut self.factors[fi];
            let lay = &self.layouts[fi];
            eval_factor_weighted(
                &self.blocks, f, lay, None,
                &mut self.scratch_in, &mut self.scratch_res, &mut self.scratch_jac,
                &mut self.scratch_wres, &mut self.scratch_wjac,
            );
            let row0 = lay.row_start;
            final_res[row0..row0 + f.n_res].copy_from_slice(&self.scratch_wres[..f.n_res]);
        }
        self.final_residuals = final_res;

        SolveSummary {
            iterations,
            final_error,
        }
    }

    /// Marginalize the given blocks out of the problem, returning a dense
    /// prior over the kept blocks connected to them (their Markov blanket),
    /// ordered by block declaration order. Evaluates the factors touching the
    /// dropped blocks at the current block values — call after [`solve`],
    /// matching the existing fixed-lag convention where the gradient term
    /// vanishes at the optimum.
    ///
    /// Returns the prior and the kept block ids (attachment order for
    /// [`Problem::add_marginal_prior`] in the next window's problem).
    ///
    /// [`solve`]: PreparedProblem::solve
    pub fn marginalize(&mut self, drop: &[BlockId]) -> Option<(MarginalPrior, Vec<BlockId>)> {
        let dropped = |b: usize| drop.iter().any(|id| id.0 == b);
        for id in drop {
            assert!(!self.blocks[id.0].frozen, "cannot marginalize a frozen block");
        }

        // Factors touching any dropped block, and the involved unfrozen blocks.
        let mut involved_factors = Vec::new();
        let mut involved_blocks: Vec<usize> = Vec::new();
        for &fi in &self.active {
            let f = &self.factors[fi];
            if !f.inputs.iter().any(|&(b, _, _)| dropped(b)) {
                continue;
            }
            involved_factors.push(fi);
            for &(b, _, _) in &f.inputs {
                if !self.blocks[b].frozen && !involved_blocks.contains(&b) {
                    involved_blocks.push(b);
                }
            }
        }
        involved_blocks.sort_unstable();
        let kept: Vec<usize> = involved_blocks
            .iter()
            .copied()
            .filter(|&b| !dropped(b))
            .collect();
        if involved_factors.is_empty() || kept.is_empty() {
            return None;
        }

        // Dense column layout: dropped blocks first, then kept.
        let mut dense_col = vec![usize::MAX; self.blocks.len()];
        let mut n_marg = 0usize;
        for id in drop {
            dense_col[id.0] = n_marg;
            n_marg += self.blocks[id.0].dim;
        }
        let mut n_total = n_marg;
        for &b in &kept {
            dense_col[b] = n_total;
            n_total += self.blocks[b].dim;
        }
        let n_keep = n_total - n_marg;

        // Accumulate H = Σ Jᵀ J over the involved factors (weighted, at the
        // current values).
        let mut h = nalgebra::DMatrix::<f64>::zeros(n_total, n_total);
        for &fi in &involved_factors {
            let f = &mut self.factors[fi];
            let lay = &self.layouts[fi];
            let n_in = eval_factor_weighted(
                &self.blocks, f, lay, None,
                &mut self.scratch_in, &mut self.scratch_res, &mut self.scratch_jac,
                &mut self.scratch_wres, &mut self.scratch_wjac,
            );
            // Dense column of each local input dim (usize::MAX for frozen).
            let mut cols = Vec::with_capacity(n_in);
            for &(b, offset, dim) in &f.inputs {
                for d in 0..dim {
                    cols.push(if self.blocks[b].frozen {
                        usize::MAX
                    } else {
                        dense_col[b] + offset + d
                    });
                }
            }
            for r in 0..f.n_res {
                let row = &self.scratch_wjac[r * n_in..(r + 1) * n_in];
                for (ci, &dci) in cols.iter().enumerate() {
                    if dci == usize::MAX {
                        continue;
                    }
                    for (cj, &dcj) in cols.iter().enumerate() {
                        if dcj == usize::MAX {
                            continue;
                        }
                        h[(dci, dcj)] += row[ci] * row[cj];
                    }
                }
            }
        }

        // Schur complement onto the kept blocks.
        let h_oo = h.view((0, 0), (n_marg, n_marg)).into_owned();
        let h_ok = h.view((0, n_marg), (n_marg, n_keep)).into_owned();
        let h_kk = h.view((n_marg, n_marg), (n_keep, n_keep)).into_owned();
        let h_oo_inv = h_oo.try_inverse()?;
        let mut h_marg = &h_kk - h_ok.transpose() * &h_oo_inv * &h_ok;

        // Symmetrize and regularize (same recipe as controller_filter's
        // marginalize_front: bump the diagonal if the min eigenvalue dips).
        for i in 0..n_keep {
            for j in (i + 1)..n_keep {
                let a = 0.5 * (h_marg[(i, j)] + h_marg[(j, i)]);
                h_marg[(i, j)] = a;
                h_marg[(j, i)] = a;
            }
        }
        let eig = h_marg.clone().symmetric_eigen();
        let min_eig = eig.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        const MIN_EIGENVALUE: f64 = 1e-8;
        const EIGENVALUE_REGULARIZATION: f64 = 1e-6;
        if min_eig < MIN_EIGENVALUE {
            let reg = EIGENVALUE_REGULARIZATION - min_eig.min(0.0);
            for i in 0..n_keep {
                h_marg[(i, i)] += reg;
            }
        }
        let chol = nalgebra::Cholesky::new(h_marg)?;
        let sqrt_info = chol.l().transpose();

        let mut linearization_point = DVector::zeros(n_keep);
        let mut block_dims = Vec::with_capacity(kept.len());
        let mut at = 0usize;
        for &b in &kept {
            let info = &self.blocks[b];
            for i in 0..info.dim {
                linearization_point[at + i] = info.values[i];
            }
            at += info.dim;
            block_dims.push(info.dim);
        }

        Some((
            MarginalPrior {
                block_dims,
                sqrt_info,
                linearization_point,
            },
            kept.into_iter().map(BlockId).collect(),
        ))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// y = a*x + b line fit through one 2-dim block.
    #[test]
    fn line_fit() {
        let data = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]; // y = 2x + 1

        let mut pb = Problem::new();
        let ab = pb.add_block([0.0, 0.0]);
        let k = pb.kind("data").key();
        for (x, y) in data {
            pb.add_factor1::<2, 1, 2, _>(k, ab, move |p: [Jet<f64, 2>; 2]| {
                [p[0] * x + p[1] - y]
            });
        }

        let mut prepared = pb.prepare();
        prepared.solve(&SolveOptions::default());
        let [a, b] = prepared.block(ab);
        assert!((a - 2.0).abs() < 1e-9, "a = {a}");
        assert!((b - 1.0).abs() < 1e-9, "b = {b}");
    }

    // NOTE: sprs's LDL asserts n > 1, so every test problem needs at least
    // two parameter columns. Real problems are far past that; only these toy
    // fixtures care.

    /// Two blocks + a prior + a frozen block; checks column layout and
    /// frozen-value injection.
    #[test]
    fn two_blocks_prior_frozen() {
        let mut pb = Problem::new();
        let anchor = pb.add_block([10.0, 20.0]);
        let x = pb.add_block([0.0, 0.0]);
        pb.freeze(anchor);

        let k_rel = pb.kind("relative").sigma(1.0).key();
        // x should sit 3 above the frozen anchor, per dim.
        pb.add_factor2::<4, 2, 2, 2, _>(k_rel, anchor, x, |a, xv| {
            [xv[0] - a[0] - 3.0, xv[1] - a[1] - 3.0]
        });

        let k_prior = pb.kind("prior").sigma(10.0).key(); // weak prior at 0
        pb.prior(k_prior, x, [0.0, 0.0]);

        let mut prepared = pb.prepare();
        assert_eq!(prepared.n_cols(), 2); // anchor contributes no columns
        prepared.solve(&SolveOptions::default());
        let xv = prepared.block(x);
        // Weak prior pulls slightly below the exact 13 / 23.
        assert!((xv[0] - 13.0).abs() < 0.2 && xv[0] < 13.0, "x = {xv:?}");
        assert!((xv[1] - 23.0).abs() < 0.3 && xv[1] < 23.0, "x = {xv:?}");
        assert_eq!(prepared.block(anchor), [10.0, 20.0], "frozen block must not move");
    }

    /// sigma == 0.0 disables a kind's factors entirely.
    #[test]
    fn zero_sigma_disables_kind() {
        let mut pb = Problem::new();
        let x = pb.add_block([5.0, 5.0]);
        let k_on = pb.kind("on").sigma(1.0).key();
        pb.prior(k_on, x, [1.0, 2.0]);
        let k_off = pb.kind("off").sigma(0.0).key();
        pb.prior(k_off, x, [100.0, 100.0]);

        let mut prepared = pb.prepare();
        assert_eq!(prepared.n_rows(), 2); // disabled prior contributes no rows
        prepared.solve(&SolveOptions::default());
        let xv = prepared.block(x);
        assert!((xv[0] - 1.0).abs() < 1e-9 && (xv[1] - 2.0).abs() < 1e-9, "x = {xv:?}");
    }

    /// Huber loss caps an outlier's influence.
    #[test]
    fn huber_downweights_outlier() {
        // Fit a constant to {0, 0, 0, 100}. L2 mean is 25; Huber should land
        // near the inliers. (Second dim exists only to satisfy sprs's n > 1.)
        let mut pb = Problem::new();
        let x = pb.add_block([25.0, 0.0]);
        let k = pb.kind("obs").sigma(1.0).huber(1.0).key();
        for y in [0.0, 0.0, 0.0, 100.0] {
            pb.add_factor1::<1, 1, 1, _>(k, x.sub::<1>(0), move |p: [Jet<f64, 1>; 1]| {
                [p[0] - y]
            });
        }
        let k_pad = pb.kind("pad").sigma(1.0).key();
        pb.prior(k_pad, x.sub::<1>(1), [0.0]);

        let mut prepared = pb.prepare();
        prepared.solve(&SolveOptions {
            max_iterations: 200,
            ..Default::default()
        });
        let xv = prepared.block(x)[0];
        assert!(xv < 5.0, "huber fit pulled to {xv}, expected near inliers");
    }

    /// Sub-block refs: random walk over dims 1..2 of two 3-dim blocks.
    #[test]
    fn sub_block_random_walk() {
        let mut pb = Problem::new();
        let a = pb.add_block([0.0, 5.0, 0.0]);
        let b = pb.add_block([0.0, 9.0, 0.0]);

        let k_anchor = pb.kind("anchor").sigma(1.0).key();
        pb.prior(k_anchor, a, [0.0, 5.0, 0.0]);
        // Tight random walk on the middle dim pulls b's middle toward a's.
        let k_rw = pb.kind("rw").sigma(0.001).key();
        pb.random_walk::<1>(k_rw, a.sub::<1>(1), b.sub::<1>(1));
        // Weak prior keeps b's other dims determined.
        let k_weak = pb.kind("weak").sigma(100.0).key();
        pb.prior(k_weak, b, [0.0, 9.0, 0.0]);

        let mut prepared = pb.prepare();
        prepared.solve(&SolveOptions::default());
        let bv = prepared.block(b);
        assert!((bv[1] - 5.0).abs() < 0.1, "b[1] = {} should track a[1]", bv[1]);
    }

    /// sqrt-info weighting reproduces per-element sigma weighting when diagonal.
    #[test]
    fn sqrt_info_matches_sigma() {
        let solve_with = |use_si: bool| -> f64 {
            let mut pb = Problem::new();
            let x = pb.add_block([0.0, 0.0]);
            let k = if use_si {
                pb.kind("obs").key()
            } else {
                pb.kind("obs").sigma(2.0).key()
            };
            let target = [3.0, 4.0];
            if use_si {
                let si = nalgebra::SMatrix::<f64, 2, 2>::from_diagonal_element(0.5);
                pb.add_factor1_sqrt_info::<2, 2, 2, _>(k, x, si, move |p: [Jet<f64, 2>; 2]| {
                    [p[0] - target[0], p[1] - target[1]]
                });
            } else {
                pb.prior(k, x, target);
            }
            let k2 = pb.kind("pull").sigma(1.0).key();
            pb.prior(k2, x, [0.0, 0.0]);
            let mut prepared = pb.prepare();
            prepared.solve(&SolveOptions::default());
            prepared.block(x)[0]
        };
        let a = solve_with(false);
        let b = solve_with(true);
        assert!((a - b).abs() < 1e-9, "sigma {a} vs sqrt_info {b}");
    }

    /// Fixed-lag marginalization on a linear chain matches the full batch
    /// solve (exact for linear factors, up to the eigenvalue regularization).
    #[test]
    fn marginalization_matches_batch() {
        let odo_sigma = 0.1;
        let meas_sigma = 0.5;
        // Chain: p0 (prior at 0) -odo-> p1 -odo-> p2 (measured at 2.5).
        let odo = move |pb: &mut Problem, k: FactorKind, a: BlockKey<2>, b: BlockKey<2>| {
            pb.add_factor2::<4, 2, 2, 2, _>(k, a, b, |pa, pbv| {
                [pbv[0] - pa[0] - 1.0, pbv[1] - pa[1] - 1.0]
            });
        };

        // Full batch.
        let p2_full = {
            let mut pb = Problem::new();
            let p0 = pb.add_block([0.0, 0.0]);
            let p1 = pb.add_block([1.0, 1.0]);
            let p2 = pb.add_block([2.0, 2.0]);
            let k_prior = pb.kind("prior").sigma(1.0).key();
            pb.prior(k_prior, p0, [0.0, 0.0]);
            let k_odo = pb.kind("odo").sigma(odo_sigma).key();
            odo(&mut pb, k_odo, p0, p1);
            odo(&mut pb, k_odo, p1, p2);
            let k_meas = pb.kind("meas").sigma(meas_sigma).key();
            pb.prior(k_meas, p2, [2.5, 2.5]);
            let mut prepared = pb.prepare();
            prepared.solve(&SolveOptions::default());
            prepared.block(p2)
        };

        // Window 1: p0, p1 only; marginalize p0.
        let (prior, kept, p1_win1) = {
            let mut pb = Problem::new();
            let p0 = pb.add_block([0.0, 0.0]);
            let p1 = pb.add_block([1.0, 1.0]);
            let k_prior = pb.kind("prior").sigma(1.0).key();
            pb.prior(k_prior, p0, [0.0, 0.0]);
            let k_odo = pb.kind("odo").sigma(odo_sigma).key();
            odo(&mut pb, k_odo, p0, p1);
            let mut prepared = pb.prepare();
            prepared.solve(&SolveOptions::default());
            let (prior, kept) = prepared.marginalize(&[p0.into()]).expect("marginalize");
            (prior, kept, prepared.block(p1))
        };
        assert_eq!(kept.len(), 1);
        assert_eq!(prior.block_dims, vec![2]);

        // Window 2: p1 (under the marginal prior), p2.
        let p2_win = {
            let mut pb = Problem::new();
            let p1 = pb.add_block(p1_win1);
            let p2 = pb.add_block([2.0, 2.0]);
            let k_marg = pb.kind("marg").key();
            pb.add_marginal_prior(k_marg, prior, &[p1.into()]);
            let k_odo = pb.kind("odo").sigma(odo_sigma).key();
            odo(&mut pb, k_odo, p1, p2);
            let k_meas = pb.kind("meas").sigma(meas_sigma).key();
            pb.prior(k_meas, p2, [2.5, 2.5]);
            let mut prepared = pb.prepare();
            prepared.solve(&SolveOptions::default());
            prepared.block(p2)
        };

        for i in 0..2 {
            assert!(
                (p2_win[i] - p2_full[i]).abs() < 1e-3,
                "windowed {:?} vs batch {:?}",
                p2_win,
                p2_full
            );
        }
    }

    /// kind_rms reports weighted residual RMS at the solution.
    #[test]
    fn kind_rms_reports() {
        let mut pb = Problem::new();
        let x = pb.add_block([0.0, 0.0]);
        let k1 = pb.kind("a").sigma(1.0).key();
        pb.prior(k1, x, [1.0, 1.0]);
        let k2 = pb.kind("b").sigma(1.0).key();
        pb.prior(k2, x, [3.0, 3.0]);
        let mut prepared = pb.prepare();
        prepared.solve(&SolveOptions::default());
        // Solution is x = [2, 2]; each kind has |r| = 1 per element.
        assert!((prepared.kind_rms(k1) - 1.0).abs() < 1e-6);
        assert!((prepared.kind_rms(k2) - 1.0).abs() < 1e-6);
        assert_eq!(prepared.kind_name(k1), "a");
    }
}
