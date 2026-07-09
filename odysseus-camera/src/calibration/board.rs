//! Calibration target geometry — AprilGrid layout → 3D corner positions.
//!
//! Conventions match Kalibr's AprilGrid (the de facto standard the
//! `aprilgrid` crate is built around):
//!
//! - Board lies in the z = 0 plane of its local frame, with +x to the right
//!   and +y up.
//! - Tags are arranged in a `tag_cols × tag_rows` grid.
//! - Tag id = `row * tag_cols + col`, with row 0 at the bottom.
//! - Each tag has side length `tag_size_m`; gaps between adjacent tags are
//!   `tag_size_m * tag_spacing_ratio` (Kalibr default ratio: 0.3).
//! - Tag pitch (centre-to-centre) = `tag_size_m * (1 + tag_spacing_ratio)`.
//!
//! Within one tag, corner indices `0..4` are bottom-left, bottom-right,
//! top-right, top-left (counter-clockwise starting from BL when viewed
//! looking at the board from +z). This matches the canonical AprilTag
//! ordering. If a future integration test shows the `aprilgrid` crate orders
//! them differently, change `CORNER_OFFSETS` here.

use odysseus_solver::math3d::Vec3;

/// Counter-clockwise corner offsets within one tag, in tag-local units of
/// `tag_size_m`. Order: BL, BR, TR, TL.
const CORNER_OFFSETS: [(f64, f64); 4] =
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

/// Layout parameters for a planar AprilGrid calibration target.
#[derive(Debug, Clone, Copy)]
pub struct AprilGridLayout {
    pub tag_rows: u32,
    pub tag_cols: u32,
    pub tag_size_m: f64,
    /// Gap between adjacent tags as a fraction of `tag_size_m`. Kalibr default 0.3.
    pub tag_spacing_ratio: f64,
}

impl AprilGridLayout {
    pub fn new(tag_rows: u32, tag_cols: u32, tag_size_m: f64, tag_spacing_ratio: f64) -> Self {
        Self { tag_rows, tag_cols, tag_size_m, tag_spacing_ratio }
    }

    /// Centre-to-centre tag pitch (tag size + gap).
    pub fn tag_pitch_m(&self) -> f64 {
        self.tag_size_m * (1.0 + self.tag_spacing_ratio)
    }

    /// Total tag count.
    pub fn num_tags(&self) -> u32 {
        self.tag_rows * self.tag_cols
    }

    /// 3D position of one corner in the board's local frame.
    ///
    /// `tag_id` is `row * tag_cols + col` (row 0 at bottom).
    /// `corner_idx ∈ 0..4` per [`CORNER_OFFSETS`]. Out-of-range inputs return
    /// `None`.
    pub fn corner_position(&self, tag_id: u32, corner_idx: u8) -> Option<Vec3<f64>> {
        if tag_id >= self.num_tags() || corner_idx >= 4 {
            return None;
        }
        let row = tag_id / self.tag_cols;
        let col = tag_id % self.tag_cols;
        let pitch = self.tag_pitch_m();
        let (dx, dy) = CORNER_OFFSETS[corner_idx as usize];
        Some(Vec3::new(
            col as f64 * pitch + dx * self.tag_size_m,
            row as f64 * pitch + dy * self.tag_size_m,
            0.0,
        ))
    }

    /// Iterate every (`tag_id`, `corner_idx`, position) triple in the grid.
    pub fn iter_corners(&self) -> impl Iterator<Item = (u32, u8, Vec3<f64>)> + '_ {
        (0..self.num_tags()).flat_map(move |tid| {
            (0..4).map(move |c| (tid, c, self.corner_position(tid, c).unwrap()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_corner_position_origin() {
        let layout = AprilGridLayout::new(6, 6, 0.05, 0.3);
        let p = layout.corner_position(0, 0).unwrap();
        assert_abs_diff_eq!(p.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p.z, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_corner_position_top_right_of_first_tag() {
        let layout = AprilGridLayout::new(6, 6, 0.05, 0.3);
        let p = layout.corner_position(0, 2).unwrap(); // TR of tag (0, 0)
        assert_abs_diff_eq!(p.x, 0.05, epsilon = 1e-12);
        assert_abs_diff_eq!(p.y, 0.05, epsilon = 1e-12);
    }

    #[test]
    fn test_pitch_and_neighbour_tag() {
        // Tag (0, 1): bottom-left at (pitch, 0).
        let layout = AprilGridLayout::new(6, 6, 0.05, 0.3);
        let pitch = layout.tag_pitch_m();
        assert_abs_diff_eq!(pitch, 0.05 * 1.3, epsilon = 1e-12);

        let p = layout.corner_position(1, 0).unwrap();
        assert_abs_diff_eq!(p.x, pitch, epsilon = 1e-12);
        assert_abs_diff_eq!(p.y, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_corner_position_second_row() {
        let layout = AprilGridLayout::new(6, 6, 0.05, 0.3);
        // Tag id 6 is row 1, col 0.
        let p = layout.corner_position(6, 0).unwrap();
        assert_abs_diff_eq!(p.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p.y, layout.tag_pitch_m(), epsilon = 1e-12);
    }

    #[test]
    fn test_out_of_range_returns_none() {
        let layout = AprilGridLayout::new(6, 6, 0.05, 0.3);
        assert!(layout.corner_position(36, 0).is_none());
        assert!(layout.corner_position(0, 4).is_none());
    }

    #[test]
    fn test_iter_corners_count() {
        let layout = AprilGridLayout::new(6, 6, 0.05, 0.3);
        assert_eq!(layout.iter_corners().count(), 6 * 6 * 4);
    }
}
