use crate::math_context::MathContext;

/// Zero-sized context for f64 math operations
///
/// This provides the same interface as JetArena but for plain f64 values,
/// allowing generic code to work with both f64 and JetHandle.
///
/// ## Usage:
/// ```ignore
/// let result = expr!(F64Ctx, a * a + b * b);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct F64Ctx;

impl F64Ctx {
    #[inline(always)]
    pub fn constant(&mut self, val: f64) -> f64 {
        val
    }

    #[inline(always)]
    pub fn add(&mut self, a: f64, b: f64) -> f64 {
        a + b
    }

    #[inline(always)]
    pub fn sub(&mut self, a: f64, b: f64) -> f64 {
        a - b
    }

    #[inline(always)]
    pub fn mul(&mut self, a: f64, b: f64) -> f64 {
        a * b
    }

    #[inline(always)]
    pub fn div(&mut self, a: f64, b: f64) -> f64 {
        a / b
    }

    #[inline(always)]
    pub fn sin(&mut self, a: f64) -> f64 {
        a.sin()
    }

    #[inline(always)]
    pub fn cos(&mut self, a: f64) -> f64 {
        a.cos()
    }

    #[inline(always)]
    pub fn sqrt(&mut self, a: f64) -> f64 {
        a.sqrt()
    }

    #[inline(always)]
    pub fn atan2(&mut self, y: f64, x: f64) -> f64 {
        y.atan2(x)
    }
}

impl MathContext for F64Ctx {
    type Value = f64;

    #[inline(always)]
    fn constant(&mut self, val: f64) -> f64 {
        val
    }
    #[inline(always)]
    fn add(&mut self, a: f64, b: f64) -> f64 {
        a + b
    }
    #[inline(always)]
    fn sub(&mut self, a: f64, b: f64) -> f64 {
        a - b
    }
    #[inline(always)]
    fn mul(&mut self, a: f64, b: f64) -> f64 {
        a * b
    }
    #[inline(always)]
    fn div(&mut self, a: f64, b: f64) -> f64 {
        a / b
    }
    #[inline(always)]
    fn sin(&mut self, a: f64) -> f64 {
        a.sin()
    }
    #[inline(always)]
    fn cos(&mut self, a: f64) -> f64 {
        a.cos()
    }
    #[inline(always)]
    fn sqrt(&mut self, a: f64) -> f64 {
        a.sqrt()
    }
    #[inline(always)]
    fn atan2(&mut self, y: f64, x: f64) -> f64 {
        y.atan2(x)
    }
}

/// Zero-sized context for f32 math operations
#[derive(Debug, Clone, Copy)]
pub struct F32Ctx;

impl F32Ctx {
    #[inline(always)]
    pub fn constant(&mut self, val: f64) -> f32 {
        val as f32
    }

    #[inline(always)]
    pub fn add(&mut self, a: f32, b: f32) -> f32 {
        a + b
    }

    #[inline(always)]
    pub fn sub(&mut self, a: f32, b: f32) -> f32 {
        a - b
    }

    #[inline(always)]
    pub fn mul(&mut self, a: f32, b: f32) -> f32 {
        a * b
    }

    #[inline(always)]
    pub fn div(&mut self, a: f32, b: f32) -> f32 {
        a / b
    }

    #[inline(always)]
    pub fn sin(&mut self, a: f32) -> f32 {
        a.sin()
    }

    #[inline(always)]
    pub fn cos(&mut self, a: f32) -> f32 {
        a.cos()
    }

    #[inline(always)]
    pub fn sqrt(&mut self, a: f32) -> f32 {
        a.sqrt()
    }

    #[inline(always)]
    pub fn atan2(&mut self, y: f32, x: f32) -> f32 {
        y.atan2(x)
    }
}

impl MathContext for F32Ctx {
    type Value = f32;

    #[inline(always)]
    fn constant(&mut self, val: f64) -> f32 {
        val as f32
    }
    #[inline(always)]
    fn add(&mut self, a: f32, b: f32) -> f32 {
        a + b
    }
    #[inline(always)]
    fn sub(&mut self, a: f32, b: f32) -> f32 {
        a - b
    }
    #[inline(always)]
    fn mul(&mut self, a: f32, b: f32) -> f32 {
        a * b
    }
    #[inline(always)]
    fn div(&mut self, a: f32, b: f32) -> f32 {
        a / b
    }
    #[inline(always)]
    fn sin(&mut self, a: f32) -> f32 {
        a.sin()
    }
    #[inline(always)]
    fn cos(&mut self, a: f32) -> f32 {
        a.cos()
    }
    #[inline(always)]
    fn sqrt(&mut self, a: f32) -> f32 {
        a.sqrt()
    }
    #[inline(always)]
    fn atan2(&mut self, y: f32, x: f32) -> f32 {
        y.atan2(x)
    }
}
