use crate::math_context::MathContext;
use std::marker::PhantomData;

/// Unified math context that captures both the underlying context and the type
///
/// This allows writing generic code that works with both f64 and JetHandle
/// without having to specify the type parameter at every call site.
///
/// ## Usage:
/// ```ignore
/// // With JetHandle:
/// let mut arena = JetArena::new(2);
/// let mut ctx = MathCtx::<JetHandle>::new(&mut arena);
/// let result = expr!(ctx, sin(x) + y);
///
/// // With f64:
/// let mut unit = ();
/// let mut ctx = MathCtx::<f64>::new(&mut unit);
/// let result = expr!(ctx, a * a + b * b);
///
/// // Generic function:
/// fn my_function<T: MathContext>(ctx: &mut MathCtx<T>, x: T) -> T {
///     expr!(ctx, sin(x) + x)  // Type inferred from ctx!
/// }
/// ```
pub struct MathCtx<'a, T: MathContext> {
    pub(crate) ctx: &'a mut T::Context,
    _phantom: PhantomData<T>,
}

impl<'a, T: MathContext> MathCtx<'a, T> {
    /// Create a new math context wrapping the underlying context
    #[inline(always)]
    pub fn new(ctx: &'a mut T::Context) -> Self {
        Self {
            ctx,
            _phantom: PhantomData,
        }
    }

    /// Get a mutable reference to the underlying context
    ///
    /// Useful when you need to call arena methods directly
    #[inline(always)]
    pub fn inner(&mut self) -> &mut T::Context {
        self.ctx
    }
}

// Convenience methods that delegate to MathContext
impl<'a, T: MathContext> MathCtx<'a, T> {
    #[inline(always)]
    pub fn constant(&mut self, val: f64) -> T {
        T::constant(self.ctx, val)
    }

    #[inline(always)]
    pub fn add(&mut self, a: T, b: T) -> T {
        T::add(self.ctx, a, b)
    }

    #[inline(always)]
    pub fn sub(&mut self, a: T, b: T) -> T {
        T::sub(self.ctx, a, b)
    }

    #[inline(always)]
    pub fn mul(&mut self, a: T, b: T) -> T {
        T::mul(self.ctx, a, b)
    }

    #[inline(always)]
    pub fn div(&mut self, a: T, b: T) -> T {
        T::div(self.ctx, a, b)
    }

    #[inline(always)]
    pub fn sin(&mut self, a: T) -> T {
        T::sin(self.ctx, a)
    }

    #[inline(always)]
    pub fn cos(&mut self, a: T) -> T {
        T::cos(self.ctx, a)
    }

    #[inline(always)]
    pub fn sqrt(&mut self, a: T) -> T {
        T::sqrt(self.ctx, a)
    }

    #[inline(always)]
    pub fn atan2(&mut self, y: T, x: T) -> T {
        T::atan2(self.ctx, y, x)
    }
}
