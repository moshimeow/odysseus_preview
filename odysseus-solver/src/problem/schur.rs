//! Levenberg-Marquardt by Schur complement, for problems whose unknowns split
//! into a long chain and a small dense rest.
//!
//! [`PreparedProblem::solve`] builds `JᵀJ` as one sparse matrix and hands it
//! to a general sparse LDL. That suits small windows. It does not suit a
//! trajectory with thousands of knots observing a few thousand shared
//! unknowns (calibration targets, landmarks, intrinsics): every shared unknown
//! couples to every knot, the factor fills in, and the general-purpose sparse
//! code spends its time on what is really a dense block.
//!
//! [`PreparedProblem::solve_schur`] takes the split from the caller instead.
//! The blocks listed in `eliminate` (the chain) must only meet each other
//! within a short distance in declaration order — true of trajectory knots,
//! where a factor touches a few neighbouring ones — so their part of `JᵀJ` is
//! banded and is factorised in time linear in its length. Everything else is
//! kept as one dense system:
//!
//! ```text
//! [ A  B ] [xa]   [ga]        A = L Lᵀ (banded),  Y = L⁻¹ B
//! [ Bᵀ C ] [xb] = [gb]   →    (C − YᵀY) xb = gb − Yᵀ L⁻¹ ga,   xa = L⁻ᵀ (L⁻¹ ga − Y xb)
//! ```
//!
//! The band's width is read off the factors, so a poor declaration order is
//! slow, not wrong. Damping, Huber and sigma weighting, frozen blocks and the
//! diagnostics afterwards behave as in [`PreparedProblem::solve`]; the normal
//! equations are accumulated factor by factor and no global Jacobian is formed.

use super::{eval_factor_weighted, BlockId, PreparedProblem, SolveOptions, SolveSummary};
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// Where a parameter column lives in the split system.
#[derive(Clone, Copy)]
enum Slot {
    Chain(usize),
    Rest(usize),
}

/// Symmetric positive-definite band matrix, lower half, row `i` holding
/// columns `i - width ..= i` contiguously (so row dot products are slices).
struct Band {
    n: usize,
    width: usize,
    data: Vec<f64>,
}

impl Band {
    fn zeros(n: usize, width: usize) -> Self {
        Self { n, width, data: vec![0.0; n * (width + 1)] }
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> usize {
        debug_assert!(j <= i && i - j <= self.width);
        i * (self.width + 1) + (j + self.width - i)
    }

    /// Cholesky factor in place; `false` if the matrix is not positive definite.
    fn factorize(&mut self) -> bool {
        let (n, w) = (self.n, self.width);
        for i in 0..n {
            let first = i.saturating_sub(w);
            for j in first..=i {
                // Columns both rows have in common to the left of j.
                let start = first.max(j.saturating_sub(w));
                let (ri, rj) = (self.index(i, start), self.index(j, start));
                let mut sum = self.data[self.index(i, j)];
                for k in 0..j - start {
                    sum -= self.data[ri + k] * self.data[rj + k];
                }
                let at = self.index(i, j);
                if j == i {
                    if !(sum > 0.0) {
                        return false;
                    }
                    self.data[at] = sum.sqrt();
                } else {
                    self.data[at] = sum / self.data[self.index(j, j)];
                }
            }
        }
        true
    }

    /// `x ← L⁻¹ x`.
    fn forward(&self, x: &mut [f64]) {
        for i in 0..self.n {
            let first = i.saturating_sub(self.width);
            let row = self.index(i, first);
            let mut sum = x[i];
            for k in 0..i - first {
                sum -= self.data[row + k] * x[first + k];
            }
            x[i] = sum / self.data[self.index(i, i)];
        }
    }

    /// `x ← L⁻ᵀ x`.
    fn backward(&self, x: &mut [f64]) {
        for i in (0..self.n).rev() {
            x[i] /= self.data[self.index(i, i)];
            let first = i.saturating_sub(self.width);
            let row = self.index(i, first);
            let xi = x[i];
            for k in 0..i - first {
                x[first + k] -= self.data[row + k] * xi;
            }
        }
    }
}

/// Cholesky factor of a dense symmetric positive-definite matrix, in place in its lower
/// triangle (the upper is left as scratch). Blocked, with the trailing update spread over
/// threads, because this is where a Schur solve spends its time once the chain is long.
/// Returns `false` if the matrix is not positive definite.
fn dense_cholesky(a: &mut DMatrix<f64>, workers: usize) -> bool {
    const BLOCK: usize = 96;
    let n = a.nrows();
    let mut k = 0;
    while k < n {
        let b = BLOCK.min(n - k);
        // Diagonal block, unblocked.
        for j in k..k + b {
            let mut d = a[(j, j)];
            for p in k..j {
                d -= a[(j, p)] * a[(j, p)];
            }
            if !(d > 0.0) {
                return false;
            }
            let d = d.sqrt();
            a[(j, j)] = d;
            for i in j + 1..k + b {
                let mut v = a[(i, j)];
                for p in k..j {
                    v -= a[(i, p)] * a[(j, p)];
                }
                a[(i, j)] = v / d;
            }
        }
        let below = n - k - b;
        if below == 0 {
            break;
        }
        // Panel under it: L21 = A21 L11⁻ᵀ, a column at a time (each is a saxpy down the panel).
        for j in k..k + b {
            for p in k..j {
                let factor = a[(j, p)];
                if factor != 0.0 {
                    let (left, right) = a.as_mut_slice().split_at_mut(j * n);
                    let source = &left[p * n + k + b..p * n + n];
                    let target = &mut right[k + b..n];
                    for (t, s) in target.iter_mut().zip(source) {
                        *t -= factor * s;
                    }
                }
            }
            let d = a[(j, j)];
            for value in &mut a.as_mut_slice()[j * n + k + b..j * n + n] {
                *value /= d;
            }
        }
        // Trailing block: A22 -= L21 L21ᵀ, in column strips across threads.
        let (left, right) = a.as_mut_slice().split_at_mut((k + b) * n);
        let panel = nalgebra::DMatrixView::from_slice(&left[k * n..], n, b);
        let panel = panel.rows(k + b, below);
        let strip = below.div_ceil(workers).max(1);
        std::thread::scope(|scope| {
            for (index, columns) in right.chunks_mut(strip * n).enumerate() {
                let panel = &panel;
                scope.spawn(move || {
                    let count = columns.len() / n;
                    let mut target = nalgebra::DMatrixViewMut::from_slice(columns, n, count);
                    let mut target = target.rows_mut(k + b, below);
                    let across = panel.rows(index * strip, count).transpose();
                    target.gemm(-1.0, panel, &across, 1.0);
                });
            }
        });
        k += b;
    }
    true
}

/// `x ← (L Lᵀ)⁻¹ x` for the factor [`dense_cholesky`] leaves in the lower triangle.
fn dense_cholesky_solve(l: &DMatrix<f64>, x: &mut DVector<f64>) {
    let n = l.nrows();
    for j in 0..n {
        x[j] /= l[(j, j)];
        let xj = x[j];
        for i in j + 1..n {
            x[i] -= l[(i, j)] * xj;
        }
    }
    for j in (0..n).rev() {
        let mut sum = x[j];
        for i in j + 1..n {
            sum -= l[(i, j)] * x[i];
        }
        x[j] = sum / l[(j, j)];
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16)
}

impl PreparedProblem {
    /// Solve like [`solve`](PreparedProblem::solve), eliminating the blocks in
    /// `eliminate` through a banded factorisation and solving the remaining
    /// blocks as one dense system. See the module docs for when this is the
    /// right shape. `opts.tolerance` is the relative cost decrease below
    /// which an accepted step ends the solve.
    pub fn solve_schur(&mut self, eliminate: &[BlockId], opts: &SolveOptions) -> SolveSummary {
        // Columns of the two parts, each in block declaration order.
        let mut slots = vec![Slot::Rest(0); self.n_cols];
        let (mut n_chain, mut n_rest) = (0usize, 0usize);
        for (index, block) in self.blocks.iter().enumerate() {
            if block.frozen {
                continue;
            }
            let in_chain = eliminate.iter().any(|id| id.0 == index);
            for d in 0..block.dim {
                slots[block.col + d] = if in_chain { Slot::Chain(n_chain + d) } else { Slot::Rest(n_rest + d) };
            }
            if in_chain {
                n_chain += block.dim;
            } else {
                n_rest += block.dim;
            }
        }

        // Half-bandwidth of the chain: the furthest apart two chain columns sit in one factor.
        let mut width = 0usize;
        for &fi in &self.active {
            let (mut low, mut high) = (usize::MAX, 0usize);
            for &(block, offset, dim) in &self.factors[fi].inputs {
                let b = &self.blocks[block];
                if b.frozen {
                    continue;
                }
                if let Slot::Chain(first) = slots[b.col + offset] {
                    low = low.min(first);
                    high = high.max(first + dim - 1);
                }
            }
            if low != usize::MAX {
                width = width.max(high - low);
            }
        }

        let mut x = DVector::zeros(self.n_cols);
        for b in &self.blocks {
            if !b.frozen {
                for i in 0..b.dim {
                    x[b.col + i] = b.values[i];
                }
            }
        }

        let mut chain = Band::zeros(n_chain, width);
        let mut coupling = DMatrix::<f64>::zeros(n_chain, n_rest);
        let mut rest = DMatrix::<f64>::zeros(n_rest, n_rest);
        let mut gradient_chain = vec![0.0; n_chain];
        let mut gradient_rest = DVector::<f64>::zeros(n_rest);
        let max_in = self.scratch_in.len();
        let mut local_cols = vec![usize::MAX; max_in];
        let mut local_normal = vec![0.0; max_in * max_in];

        // Sum of squared weighted residuals at `at`; with `into`, also the normal equations.
        let mut evaluate = |problem: &mut PreparedProblem,
                            at: &DVector<f64>,
                            mut into: Option<(&mut Band, &mut DMatrix<f64>, &mut DMatrix<f64>, &mut [f64], &mut DVector<f64>)>|
         -> f64 {
            if let Some((chain, coupling, rest, gradient_chain, gradient_rest)) = into.as_mut() {
                chain.data.fill(0.0);
                coupling.fill(0.0);
                rest.fill(0.0);
                gradient_chain.fill(0.0);
                gradient_rest.fill(0.0);
            }
            let mut cost = 0.0;
            for k in 0..problem.active.len() {
                let fi = problem.active[k];
                let n_in = eval_factor_weighted(
                    &problem.blocks,
                    &mut problem.factors[fi],
                    &problem.layouts[fi],
                    Some(at),
                    &mut problem.scratch_in,
                    &mut problem.scratch_res,
                    &mut problem.scratch_jac,
                    &mut problem.scratch_wres,
                    &mut problem.scratch_wjac,
                );
                let n_res = problem.factors[fi].n_res;
                let (res, jac) = (&problem.scratch_wres[..n_res], &problem.scratch_wjac[..n_res * n_in]);
                cost += res.iter().map(|r| r * r).sum::<f64>();
                let Some((chain, coupling, rest, gradient_chain, gradient_rest)) = into.as_mut() else { continue };
                let mut local = 0usize;
                for &(block, offset, dim) in &problem.factors[fi].inputs {
                    let b = &problem.blocks[block];
                    for d in 0..dim {
                        local_cols[local + d] = if b.frozen { usize::MAX } else { b.col + offset + d };
                    }
                    local += dim;
                }
                // This factor's JᵀJ (upper half) and Jᵀr, then scattered.
                for a in 0..n_in {
                    if local_cols[a] == usize::MAX {
                        continue;
                    }
                    let mut g = 0.0;
                    for r in 0..n_res {
                        g += jac[r * n_in + a] * res[r];
                    }
                    match slots[local_cols[a]] {
                        Slot::Chain(i) => gradient_chain[i] += g,
                        Slot::Rest(i) => gradient_rest[i] += g,
                    }
                    for b in a..n_in {
                        let mut h = 0.0;
                        for r in 0..n_res {
                            h += jac[r * n_in + a] * jac[r * n_in + b];
                        }
                        local_normal[a * n_in + b] = h;
                    }
                }
                for a in 0..n_in {
                    if local_cols[a] == usize::MAX {
                        continue;
                    }
                    for b in a..n_in {
                        if local_cols[b] == usize::MAX {
                            continue;
                        }
                        let h = local_normal[a * n_in + b];
                        match (slots[local_cols[a]], slots[local_cols[b]]) {
                            (Slot::Chain(i), Slot::Chain(j)) => {
                                let at = chain.index(i.max(j), i.min(j));
                                chain.data[at] += h;
                            }
                            (Slot::Chain(i), Slot::Rest(j)) | (Slot::Rest(j), Slot::Chain(i)) => coupling[(i, j)] += h,
                            (Slot::Rest(i), Slot::Rest(j)) => {
                                rest[(i.max(j), i.min(j))] += h;
                                if i != j {
                                    rest[(i.min(j), i.max(j))] += h;
                                }
                            }
                        }
                    }
                }
            }
            cost
        };

        let mut cost = evaluate(
            self,
            &x,
            Some((&mut chain, &mut coupling, &mut rest, &mut gradient_chain, &mut gradient_rest)),
        );
        let mut lambda = 1e-4;
        let mut iterations = 0usize;
        let workers = threads();
        let mut reduced_coupling = DMatrix::<f64>::zeros(n_chain, n_rest);
        'outer: for iteration in 0..opts.max_iterations {
            iterations = iteration + 1;
            let started = Instant::now();
            // Seconds in: band factor, reduction Y, Schur product, dense solve, trial evaluation.
            let mut spent = [0.0f64; 5];
            let mut mark = |slot: usize, lap: &mut Instant| {
                spent[slot] += lap.elapsed().as_secs_f64();
                *lap = Instant::now();
            };
            let accepted = loop {
                let mut lap = Instant::now();
                // Damp as the sparse solver does: diagonal += lambda * max(diagonal, 1).
                let mut factor = Band { n: chain.n, width: chain.width, data: chain.data.clone() };
                for i in 0..n_chain {
                    let at = factor.index(i, i);
                    factor.data[at] += lambda * factor.data[at].max(1.0);
                }
                let mut step = None;
                let factored = factor.factorize();
                mark(0, &mut lap);
                if factored {
                    // Y = L⁻¹ B, a column at a time across threads.
                    reduced_coupling.copy_from(&coupling);
                    if n_chain > 0 && n_rest > 0 {
                        let per = n_rest.div_ceil(workers).max(1) * n_chain;
                        std::thread::scope(|scope| {
                            for columns in reduced_coupling.as_mut_slice().chunks_mut(per) {
                                let factor = &factor;
                                scope.spawn(move || {
                                    for column in columns.chunks_mut(n_chain) {
                                        factor.forward(column);
                                    }
                                });
                            }
                        });
                    }
                    mark(1, &mut lap);
                    // S = C + damping − YᵀY, in column strips across threads.
                    let mut schur = rest.clone();
                    for i in 0..n_rest {
                        schur[(i, i)] += lambda * schur[(i, i)].max(1.0);
                    }
                    if n_chain > 0 && n_rest > 0 {
                        // Yᵀ is written out so that each strip is a plain matrix product, which
                        // nalgebra hands to a cache-blocked kernel (its `tr_mul` is not blocked).
                        let transposed = reduced_coupling.transpose();
                        let strip = n_rest.div_ceil(workers).max(1);
                        let products: Vec<(usize, DMatrix<f64>)> = std::thread::scope(|scope| {
                            let handles: Vec<_> = (0..n_rest)
                                .step_by(strip)
                                .map(|first| {
                                    let (y, yt) = (&reduced_coupling, &transposed);
                                    scope.spawn(move || {
                                        let count = strip.min(n_rest - first);
                                        (first, yt * y.columns(first, count))
                                    })
                                })
                                .collect();
                            handles.into_iter().map(|h| h.join().expect("Schur product thread")).collect()
                        });
                        for (first, product) in products {
                            let mut target = schur.columns_mut(first, product.ncols());
                            target -= product;
                        }
                    }
                    mark(2, &mut lap);
                    let mut reduced_chain = gradient_chain.clone();
                    factor.forward(&mut reduced_chain);
                    let reduced_chain = DVector::from_vec(reduced_chain);
                    let mut rhs = -(&gradient_rest - reduced_coupling.tr_mul(&reduced_chain));
                    let step_rest = dense_cholesky(&mut schur, workers).then(|| {
                        dense_cholesky_solve(&schur, &mut rhs);
                        rhs
                    });
                    if let Some(step_rest) = step_rest {
                        let mut step_chain = (-(reduced_chain + &reduced_coupling * &step_rest)).as_slice().to_vec();
                        factor.backward(&mut step_chain);
                        step = Some((step_chain, step_rest));
                    }
                }
                mark(3, &mut lap);
                if let Some((step_chain, step_rest)) = step {
                    let mut trial = x.clone();
                    for (column, slot) in slots.iter().enumerate() {
                        trial[column] += match *slot {
                            Slot::Chain(i) => step_chain[i],
                            Slot::Rest(i) => step_rest[i],
                        };
                    }
                    let trial_cost = evaluate(self, &trial, None);
                    mark(4, &mut lap);
                    if trial_cost.is_finite() && trial_cost < cost {
                        break Some((trial, trial_cost));
                    }
                    // A step that leaves the cost where it was, to within the tolerance, is the
                    // minimum showing through rounding; more damping would only repeat it.
                    if trial_cost.is_finite() && trial_cost - cost < opts.tolerance * cost {
                        break None;
                    }
                }
                lambda *= 10.0;
                if !(lambda < 1e12) {
                    break None;
                }
            };
            let Some((trial, trial_cost)) = accepted else { break 'outer };
            let gain = (cost - trial_cost) / cost.max(f64::MIN_POSITIVE);
            x = trial;
            lambda = (lambda * 0.1).max(f64::EPSILON);
            let relinearize = Instant::now();
            cost = evaluate(
                self,
                &x,
                Some((&mut chain, &mut coupling, &mut rest, &mut gradient_chain, &mut gradient_rest)),
            );
            if opts.verbose {
                println!(
                    "schur iteration {iteration}: error {:.6} (down {:.2e}), lambda {:.1e}, {:.2} s \
                     (band {:.2}, reduce {:.2}, product {:.2}, dense {:.2}, trial {:.2}, linearize {:.2})",
                    cost.sqrt(),
                    gain,
                    lambda,
                    started.elapsed().as_secs_f64(),
                    spent[0],
                    spent[1],
                    spent[2],
                    spent[3],
                    spent[4],
                    relinearize.elapsed().as_secs_f64()
                );
            }
            if gain < opts.tolerance {
                break;
            }
        }

        for b in &mut self.blocks {
            if !b.frozen {
                for i in 0..b.dim {
                    b.values[i] = x[b.col + i];
                }
            }
        }
        let mut final_res = std::mem::take(&mut self.final_residuals);
        for k in 0..self.active.len() {
            let fi = self.active[k];
            eval_factor_weighted(
                &self.blocks,
                &mut self.factors[fi],
                &self.layouts[fi],
                None,
                &mut self.scratch_in,
                &mut self.scratch_res,
                &mut self.scratch_jac,
                &mut self.scratch_wres,
                &mut self.scratch_wjac,
            );
            let n_res = self.factors[fi].n_res;
            let row0 = self.layouts[fi].row_start;
            final_res[row0..row0 + n_res].copy_from_slice(&self.scratch_wres[..n_res]);
        }
        self.final_residuals = final_res;
        SolveSummary { iterations, final_error: cost.sqrt() }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Problem;
    use super::*;
    use crate::Jet;

    #[test]
    fn band_cholesky_solves_a_banded_system() {
        // A = tridiagonal-ish SPD with half-bandwidth 2.
        let n = 9;
        let dense = DMatrix::from_fn(n, n, |i, j| match i.abs_diff(j) {
            0 => 6.0 + i as f64,
            1 => -1.5,
            2 => 0.7,
            _ => 0.0,
        });
        let mut band = Band::zeros(n, 2);
        for i in 0..n {
            for j in i.saturating_sub(2)..=i {
                let at = band.index(i, j);
                band.data[at] = dense[(i, j)];
            }
        }
        assert!(band.factorize());
        let b: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 2.0).collect();
        let mut x = b.clone();
        band.forward(&mut x);
        band.backward(&mut x);
        let back = &dense * DVector::from_vec(x);
        for i in 0..n {
            assert!((back[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn dense_cholesky_matches_nalgebra_on_a_matrix_larger_than_one_block() {
        let n = 250;
        let seed = DMatrix::from_fn(n, n, |i, j| ((i * 31 + j * 17) % 23) as f64 / 23.0 - 0.5);
        let spd = &seed * seed.transpose() + DMatrix::identity(n, n) * 3.0;
        let b = DVector::from_fn(n, |i, _| (i as f64 * 0.37).cos());
        let expected = nalgebra::Cholesky::new(spd.clone()).unwrap().solve(&b);
        let mut factor = spd.clone();
        assert!(dense_cholesky(&mut factor, 3));
        let mut x = b.clone();
        dense_cholesky_solve(&factor, &mut x);
        assert!((x - expected).amax() < 1e-9);
        // And an indefinite matrix is refused.
        let mut indefinite = spd;
        indefinite[(200, 200)] = -1.0;
        assert!(!dense_cholesky(&mut indefinite, 3));
    }

    /// A chain of positions tied to their neighbours, each seen through a shared offset and a
    /// shared gain: the Schur solve and the sparse solve must land on the same answer.
    fn chain_problem() -> (PreparedProblem, Vec<BlockId>) {
        let mut problem = Problem::new();
        let n = 40;
        let knots: Vec<_> = (0..n).map(|i| problem.add_block([0.1 * i as f64, 0.0])).collect();
        let shared = problem.add_block([0.3, -0.2]);
        let scale = problem.add_block([1.2]);
        let step = problem.kind("step").sigma(0.5).key();
        let seen = problem.kind("seen").sigma(0.1).huber(1.0).key();
        let direct = problem.kind("direct").sigma(0.2).key();
        for i in 0..n {
            let truth = [(0.2 * i as f64).sin() * 3.0, (0.13 * i as f64).cos() * 2.0];
            let measured = [1.5 * truth[0] + 0.7 + 0.01 * (i as f64 * 1.7).sin(), 1.5 * truth[1] - 0.4];
            // Each knot is also seen directly, which is what fixes the gain.
            problem.prior(direct, knots[i], truth);
            problem.add_factor3::<5, 2, _, _, _, _>(
                seen,
                knots[i],
                shared,
                scale,
                move |k: [Jet<f64, 5>; 2], s: [Jet<f64, 5>; 2], c: [Jet<f64, 5>; 1]| {
                    [c[0] * k[0] + s[0] - measured[0], c[0] * k[1] + s[1] - measured[1]]
                },
            );
            if i + 1 < n {
                problem.random_walk(step, knots[i], knots[i + 1]);
            }
        }
        (problem.prepare(), knots.into_iter().map(BlockId::from).collect())
    }

    #[test]
    fn schur_solve_matches_the_sparse_solve() {
        let options = SolveOptions { max_iterations: 200, tolerance: 1e-13, verbose: false };
        let (mut sparse, _) = chain_problem();
        let reference = sparse.solve(&options);
        let (mut schur, chain) = chain_problem();
        let summary = schur.solve_schur(&chain, &options);
        assert!(summary.iterations > 1);
        assert!(summary.iterations < 50, "took {} iterations", summary.iterations);
        assert!((summary.final_error - reference.final_error).abs() < 1e-6 * reference.final_error);
        for (a, b) in sparse.blocks.iter().zip(&schur.blocks) {
            for (x, y) in a.values.iter().zip(&b.values) {
                assert!((x - y).abs() < 1e-6, "sparse {x} vs schur {y}");
            }
        }
    }

    #[test]
    fn schur_solve_with_nothing_left_over_is_a_banded_solve() {
        // Freeze the shared blocks: only the chain is solved.
        let mut problem = Problem::new();
        let knots: Vec<_> = (0..10).map(|_| problem.add_block([0.0])).collect();
        let step = problem.kind("step").key();
        let pin = problem.kind("pin").key();
        for i in 0..9 {
            problem.random_walk(step, knots[i], knots[i + 1]);
        }
        problem.prior(pin, knots[0], [1.0]);
        problem.prior(pin, knots[9], [10.0]);
        let mut prepared = problem.prepare();
        let chain: Vec<BlockId> = knots.iter().map(|k| BlockId::from(*k)).collect();
        prepared.solve_schur(&chain, &SolveOptions { max_iterations: 50, tolerance: 1e-14, verbose: false });
        // Minimum of the quadratic: evenly spaced between the two pins, pulled in slightly.
        let values: Vec<f64> = knots.iter().map(|k| prepared.block(*k)[0]).collect();
        for w in values.windows(3) {
            assert!((w[0] - 2.0 * w[1] + w[2]).abs() < 1e-6, "interior knots lie on a line");
        }
        assert!(values[0] > 1.0 && values[9] < 10.0);
    }
}
