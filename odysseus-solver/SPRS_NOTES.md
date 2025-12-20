# sprs Usage Notes

This document captures lessons learned while using the `sprs` sparse matrix library in Rust.

## Trait Bounds for Generic Code

When using sprs with generic types `T`, you need several trait bounds beyond the obvious ones:

```rust
impl<T> MyStruct<T>
where
    T: num_traits::Float + nalgebra::RealField + std::fmt::Display + std::iter::Sum + Default
        + for<'r> std::ops::DivAssign<&'r T>      // Required for sprs_ldl::Ldl::solve()
        + for<'a> std::ops::Mul<&'a T, Output = T>, // May be needed for some operations
    for<'a> &'a T: std::ops::Mul<Output = T>,      // Required for CsMat multiplication
{
```

The key insight: `&CsMat * &CsMat` multiplication requires `for<'a> &'a T: Mul<Output = T>`.

## Matrix Formats: CSR vs CSC

- **CSR (Compressed Sparse Row)**: Efficient for row-wise iteration. Use `to_csr()`.
- **CSC (Compressed Sparse Column)**: Efficient for column-wise iteration. Use `to_csc()`.

The `outer_iterator()` method:
- On **CSR**: iterates over **rows**
- On **CSC**: iterates over **columns**

## Transpose

`transpose_into()` **consumes** self. If you need to keep the original matrix:

```rust
// WRONG - j_csr is moved
let jt = j_csr.transpose_into();
let jtj = &jt * &j_csr;  // ERROR: j_csr was moved

// CORRECT - clone first
let jt = j_csr.clone().transpose_into();
let jtj = &jt * &j_csr;  // OK
```

When you transpose:
- CSR matrix → CSC matrix (same data, reinterpreted)
- The resulting matrix has swapped dimensions

## Matrix-Vector Multiplication (J^T * r)

For computing `J^T * r` where J is stored as CSR:

```rust
// J is m×n CSR, r is m×1 vector, result is n×1 vector
// J^T is n×m, stored as CSC after transpose_into()

let jt: CsMat<T> = j_csr.clone().transpose_into();  // Now CSC

// J^T * r [i] = sum_j J^T[i,j] * r[j]
// For CSC, outer_iterator gives columns j
// Each column j contains non-zeros at rows i

let mut result = vec![T::zero(); n_cols];
for (col_j, col) in jt.outer_iterator().enumerate() {
    let r_j = r[col_j];
    for (row_i, &val) in col.iter() {
        result[row_i] = result[row_i] + val * r_j;
    }
}
```

**Common mistake**: Thinking `outer_iterator` on the transpose gives you rows. It gives you columns (because transpose of CSR is CSC).

## Matrix-Matrix Multiplication (J^T * J)

```rust
let jt: CsMat<T> = j_csr.clone().transpose_into();
let jtj: CsMat<T> = &jt * &j_csr;
```

This works but requires the trait bound `for<'a> &'a T: std::ops::Mul<Output = T>`.

## Matrix Addition

The `+` operator on `&CsMat + &CsMat` produces a **dense** `ndarray::Array2`, not a sparse matrix!

For sparse + sparse → sparse, use `csmat_binop`:

```rust
use sprs::binop::csmat_binop;

// WRONG - produces dense matrix
let c = &a + &b;

// CORRECT - produces sparse matrix
let c: CsMat<T> = csmat_binop(a.view(), b.view(), |x, y| *x + *y);
```

Note the dereference `*x + *y` - the closure receives references.

## Accessing indptr

The `indptr()` method returns an `IndPtrBase` which borrows from the matrix. To avoid lifetime issues:

```rust
// WRONG - temporary dropped
let indptr: &[usize] = jtj.indptr().as_slice().unwrap();
for col in 0..n {
    // ERROR: indptr is invalid here
}

// CORRECT - bind the IndPtrBase first
let indptr_storage = jtj.indptr();
let indptr: &[usize] = indptr_storage.as_slice().unwrap();
for col in 0..n {
    // OK
}
```

## Building Matrices with TriMat

`TriMat` (triplet format) is the easiest way to build sparse matrices:

```rust
let mut tri = TriMat::new((n_rows, n_cols));
tri.add_triplet(row, col, value);
tri.add_triplet(row, col, value);  // Duplicates are summed!

let csr: CsMat<T> = tri.to_csr();
let csc: CsMat<T> = tri.to_csc();
```

Duplicate entries at the same (row, col) are **summed together**.

## Modifying Values In-Place

If you have a pre-built `CsMat` and want to update values without changing structure:

```rust
// Get mutable access to the data array
let data: &mut [T] = matrix.data_mut();

// Write directly by index (you need to know the index)
data[idx] = new_value;
```

The indices in `data_mut()` correspond to the order entries appear in the CSR/CSC structure.

## sprs_ldl for LDL Factorization

```rust
use sprs_ldl::Ldl;
use sprs::SymmetryCheck;

// Matrix must be in CSC format for LDL
let jtj_csc: CsMat<T> = ...;

let ldl = Ldl::new()
    .check_symmetry(SymmetryCheck::DontCheckSymmetry)  // Skip if you know it's symmetric
    .numeric(jtj_csc.view())?;

let rhs: Vec<T> = ...;
let solution: Vec<T> = ldl.solve(&rhs);
```

Requirements for type T:
- `for<'r> std::ops::DivAssign<&'r T>` - the solve method requires this

## Performance Tips

1. **Reuse structure**: If only values change (not sparsity pattern), build the CsMat once and use `data_mut()` to update values.

2. **CSR for row iteration**: If your algorithm iterates over rows, use CSR format.

3. **Avoid cloning large matrices**: `transpose_into()` is more efficient than `transpose_view().to_owned()` but consumes the original.

4. **TriMat for construction**: Building via triplets is cleaner than manually constructing CSR/CSC arrays.

## Common Error Messages

| Error | Likely Cause |
|-------|--------------|
| `cannot multiply &CsMatBase... by &CsMatBase...` | Missing `for<'a> &'a T: Mul<Output = T>` bound |
| `cannot divide-assign T by &'r T` | Missing `for<'r> DivAssign<&'r T>` bound for LDL solve |
| `temporary value dropped while borrowed` | Need to bind `indptr()` result to a variable |
| `expected CsMatBase, found ArrayBase` | Used `+` operator instead of `csmat_binop` for addition |
