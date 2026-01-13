//! SLAM Frontend: Feature detection and tracking for visual SLAM
//!
//! This crate provides ORB-like feature detection (FAST corners + BRIEF descriptors)
//! and Lucas-Kanade tracking for use with the odysseus-slam backend.

pub mod descriptor;
pub mod detector;
pub mod lk_tracker;
pub mod stereo_matcher;
pub mod tracker;

pub use descriptor::{BriefDescriptor, BriefExtractor};
pub use detector::{FastDetector, KeyPoint};
pub use lk_tracker::{LKConfig, LKTracker, TrackResult};
pub use stereo_matcher::{StereoMatch, StereoMatcher};
pub use tracker::{TrackedFeature, Tracker, TrackerConfig};

/// Pinhole camera model
#[derive(Debug, Clone, Copy)]
pub struct PinholeCamera {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// Stereo camera configuration
#[derive(Debug, Clone, Copy)]
pub struct StereoCamera {
    pub left: PinholeCamera,
    pub baseline: f32,  // meters
}

impl StereoCamera {
    /// Project left pixel to right pixel using depth
    pub fn project_left_to_right(&self, left_u: f32, left_v: f32, depth: f32) -> (f32, f32) {
        // Right camera is baseline distance along x-axis
        // disparity = baseline * fx / depth
        let disparity = self.baseline * self.left.fx / depth;
        (left_u - disparity, left_v)
    }
}
