/// Trait for mathematical context types that provide autodiff-compatible operations
///
/// This trait is implemented by:
/// - `F64Ctx` - zero-cost f64 operations
/// - `F32Ctx` - zero-cost f32 operations
/// - `JetArena<f64>` - automatic differentiation with f64
///
/// The associated `Value` type represents what the operations work on:
/// - For F64Ctx: Value = f64
/// - For F32Ctx: Value = f32
/// - For JetArena<f64>: Value = JetHandle
pub trait MathContext {
    /// The value type this context operates on (f64, f32, or JetHandle)
    type Value: Copy;

    fn constant(&mut self, val: f64) -> Self::Value;
    fn add(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    fn sub(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    fn mul(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    fn div(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    fn sin(&mut self, a: Self::Value) -> Self::Value;
    fn cos(&mut self, a: Self::Value) -> Self::Value;
    fn sqrt(&mut self, a: Self::Value) -> Self::Value;
    fn atan2(&mut self, y: Self::Value, x: Self::Value) -> Self::Value;
}
