//! `compose_params!` — parameter structs with array conversions and
//! problem-builder block helpers.
//!
//! The macro itself is a proc-macro in `odysseus-solver-macros` (re-exported
//! from this crate's root); it emits the struct, `From<[T; DIM]>` both ways,
//! a `DIM` const, and one `BlockRef` helper per field so factor wiring never
//! repeats offset/dimension literals. See the macro's own docs for details.

#[cfg(test)]
mod tests {
    use crate::compose_params;
    use crate::math3d::Vec3;
    use crate::problem::{Problem, SolveOptions};

    compose_params! {
        struct TestParams<T> {
            position: Vec3<T> [3],
            velocity: Vec3<T> [3],
        }
    }

    #[test]
    fn roundtrip_and_dim() {
        assert_eq!(TestParams::<f64>::DIM, 6);
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = TestParams::from(arr);
        assert_eq!(p.position.y, 2.0);
        assert_eq!(p.velocity.z, 6.0);
        let back: [f64; 6] = p.into();
        assert_eq!(back, arr);
    }

    /// The generated block helpers wire priors to the right sub-ranges:
    /// a prior through `TestParams::velocity` must move dims 3..6 only.
    #[test]
    fn block_helpers_target_correct_dims() {
        let mut pb = Problem::new();
        let key = pb.add_block([0.0; 6]);
        let k_pos = pb.kind("pos").sigma(1.0).key();
        pb.prior(k_pos, TestParams::<f64>::position(key), [1.0, 2.0, 3.0]);
        let k_vel = pb.kind("vel").sigma(1.0).key();
        pb.prior(k_vel, TestParams::<f64>::velocity(key), [4.0, 5.0, 6.0]);
        let mut prepared = pb.prepare();
        prepared.solve(&SolveOptions::default());
        let solved = prepared.block(key);
        for (i, expect) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
            assert!((solved[i] - expect).abs() < 1e-9, "dim {i}: {solved:?}");
        }
    }
}
