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
