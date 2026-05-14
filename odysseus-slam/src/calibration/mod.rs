//! Camera intrinsic calibration via AprilGrid boards.
//!
//! Layered into:
//! - [`board`] — calibration target geometry (AprilGrid layout → 3D corner positions
//!   in board frame). Always available.
//! - [`detector`] — wrapper over the `aprilgrid` crate that turns an image into
//!   a [`BoardObservation`]. Behind the `aprilgrid` cargo feature so non-image
//!   users (synthetic-data tests, headless) don't pull in image-processing deps.
//!
//! Phase 3 will add `optimization::calibration` which consumes a
//! `Vec<BoardObservation>` and recovers intrinsics + per-frame extrinsics.

pub mod board;

#[cfg(feature = "aprilgrid")]
pub mod detector;

pub use board::AprilGridLayout;

use odysseus_solver::math3d::Vec3;

/// Identifier of a single corner observation: `(tag_id, corner_idx)`.
///
/// `corner_idx ∈ 0..4` indexes the four corners of one tag, ordered to match
/// the [`detector`] output. See [`AprilGridLayout::corner_position`] for the
/// exact convention.
pub type CornerKey = (u32, u8);

/// One detected corner: a 2D pixel observation paired with its known 3D
/// position in the calibration board's local frame.
#[derive(Debug, Clone, Copy)]
pub struct CornerObservation {
    pub key: CornerKey,
    pub image_xy: [f64; 2],
    pub board_xyz: Vec3<f64>,
}

/// All corner observations for a single calibration image.
///
/// `frame_idx` is an opaque user-facing identifier — typically the frame
/// number. The optimizer attaches one SE(3) extrinsic block per unique
/// `frame_idx`.
#[derive(Debug, Clone)]
pub struct BoardObservation {
    pub frame_idx: usize,
    pub corners: Vec<CornerObservation>,
}

impl BoardObservation {
    pub fn new(frame_idx: usize) -> Self {
        Self { frame_idx, corners: Vec::new() }
    }
}
