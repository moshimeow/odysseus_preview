/// Defines a parameter struct whose fields each convert to/from a fixed-size array,
/// and generates `From<[T; N]>` and `From<Struct> for [T; N]` by concatenating chunks.
///
/// Each field must be a type that already implements `From<[T; n]>` and `[T; n]: From<FieldType>`.
///
/// Example:
/// ```
/// use odysseus_solver::compose_params;
/// use odysseus_solver::math3d::Vec3;
///
/// compose_params! {
///     pub struct MyParams<T> {
///         position: Vec3<T> [3],
///         velocity: Vec3<T> [3],
///     }
/// }
///
/// let arr: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let p = MyParams::from(arr);
/// let back: [f64; 6] = p.into();
/// ```
#[macro_export]
macro_rules! compose_params {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident<$T:ident> {
            $($field:ident : $ftype:ty [$n:literal]),* $(,)?
        }
    ) => {
        $(#[$attr])*
        $vis struct $name<$T> {
            $(pub $field: $ftype,)*
        }

        impl<$T: Copy + Default> From<[$T; { 0usize $(+ $n)* }]> for $name<$T>
        where
            $($ftype: From<[$T; $n]>,)*
        {
            fn from(arr: [$T; { 0usize $(+ $n)* }]) -> Self {
                let mut _offset = 0usize;
                $(
                    let $field = {
                        let chunk: [$T; $n] = std::array::from_fn(|i| arr[_offset + i]);
                        _offset += $n;
                        <$ftype>::from(chunk)
                    };
                )*
                Self { $($field,)* }
            }
        }

        impl<$T: Copy + Default> From<$name<$T>> for [$T; { 0usize $(+ $n)* }]
        where
            $([$T; $n]: From<$ftype>,)*
        {
            fn from(val: $name<$T>) -> Self {
                let mut out = [$T::default(); { 0usize $(+ $n)* }];
                let mut _offset = 0usize;
                $(
                    let chunk: [$T; $n] = <[$T; $n]>::from(val.$field);
                    out[_offset.._offset + $n].copy_from_slice(&chunk);
                    _offset += $n;
                )*
                out
            }
        }
    };
}
