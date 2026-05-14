//! AprilGrid detector wrapper.
//!
//! Thin layer over [`aprilgrid::detector::TagDetector`] that joins detected
//! tag corners with their known positions on an [`AprilGridLayout`] to produce
//! a [`BoardObservation`] suitable for the calibration optimizer.
//!
//! Re-exports [`TagFamily`] from the upstream crate for convenience.

use aprilgrid::detector::{DetectorParams, TagDetector};
pub use aprilgrid::TagFamily;
use image::DynamicImage;

use crate::calibration::{AprilGridLayout, BoardObservation, CornerObservation};

/// Detector tied to a specific [`AprilGridLayout`] and tag family.
pub struct AprilGridDetector {
    inner: TagDetector,
    layout: AprilGridLayout,
}

impl AprilGridDetector {
    pub fn new(layout: AprilGridLayout, family: TagFamily) -> Self {
        Self::with_params(layout, family, None)
    }

    /// Create with custom detector parameters. Pass `None` to use upstream defaults.
    pub fn with_params(
        layout: AprilGridLayout,
        family: TagFamily,
        params: Option<DetectorParams>,
    ) -> Self {
        // Forward the spacing ratio to the detector so its quad-search lines up
        // with our board geometry.
        let params = params.unwrap_or_else(|| {
            let mut p = DetectorParams::default_params();
            p.tag_spacing_ratio = layout.tag_spacing_ratio as f32;
            p
        });
        Self {
            inner: TagDetector::new(&family, Some(params)),
            layout,
        }
    }

    pub fn layout(&self) -> &AprilGridLayout {
        &self.layout
    }

    /// Run detection on one image and return the joined corner observations.
    /// Tags whose id is outside the layout (e.g. from a different board) are
    /// silently dropped.
    pub fn detect(&self, img: &DynamicImage, frame_idx: usize) -> BoardObservation {
        let raw = self.inner.detect(img);
        let mut observation = BoardObservation::new(frame_idx);

        for (tag_id, corners_xy) in raw {
            for (corner_idx, &(x, y)) in corners_xy.iter().enumerate() {
                let corner_idx = corner_idx as u8;
                let Some(board_xyz) = self.layout.corner_position(tag_id, corner_idx) else {
                    continue;
                };
                observation.corners.push(CornerObservation {
                    key: (tag_id, corner_idx),
                    image_xy: [x as f64, y as f64],
                    board_xyz,
                });
            }
        }

        observation
    }
}
