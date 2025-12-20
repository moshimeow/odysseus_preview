//! Feature tracker with stereo matching and temporal tracking
//!
//! Combines:
//! - Lucas-Kanade optical flow for temporal tracking
//! - Stereo matching for depth estimation
//! - Feature detection to add new features over time

use std::collections::HashMap;

use image::GrayImage;

use crate::{
    lk_tracker::{LKConfig, LKTracker},
    BriefDescriptor, BriefExtractor, FastDetector, KeyPoint, StereoMatch, StereoMatcher,
};

/// A tracked feature with persistent ID
#[derive(Debug, Clone)]
pub struct TrackedFeature {
    /// Unique ID for this feature (persistent across frames)
    pub id: usize,
    /// Current stereo match
    pub stereo: StereoMatch,
    /// Number of frames this feature has been tracked
    pub age: usize,
    /// Frame index when first detected
    pub first_frame: usize,
}

/// Configuration for the feature tracker
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Minimum number of features to maintain
    pub min_features: usize,
    /// Maximum number of features to track
    pub max_features: usize,
    /// Maximum age (frames without stereo match) before dropping a feature
    pub max_age_without_stereo: usize,
    /// Lucas-Kanade configuration
    pub lk_config: LKConfig,
    /// Grid cell size for ensuring spatial distribution of new features
    pub grid_size: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            min_features: 100,
            max_features: 300,
            max_age_without_stereo: 3,
            lk_config: LKConfig::default(),
            grid_size: 64,
        }
    }
}

/// Feature tracker that maintains consistent feature IDs across frames
pub struct Tracker {
    config: TrackerConfig,
    /// Currently tracked features (id -> feature)
    tracks: HashMap<usize, TrackedFeature>,
    /// Next available feature ID
    next_id: usize,
    /// Frame counter
    frame_idx: usize,
    /// LK optical flow tracker
    lk_tracker: LKTracker,
    /// Feature detector
    detector: FastDetector,
    /// Descriptor extractor
    extractor: BriefExtractor,
    /// Stereo matcher
    stereo_matcher: StereoMatcher,
    /// Previous left image (for LK tracking)
    prev_left: Option<GrayImage>,
}

impl Tracker {
    /// Create a new tracker with default configuration
    pub fn new() -> Self {
        Self::with_config(TrackerConfig::default())
    }

    /// Create a new tracker with custom configuration
    pub fn with_config(config: TrackerConfig) -> Self {
        Self {
            lk_tracker: LKTracker::with_config(config.lk_config.clone()),
            config,
            tracks: HashMap::new(),
            next_id: 0,
            frame_idx: 0,
            detector: FastDetector::default(),
            extractor: BriefExtractor::new(),
            stereo_matcher: StereoMatcher::default(),
            prev_left: None,
        }
    }

    /// Process a stereo frame pair
    ///
    /// Returns the currently tracked features with updated positions
    pub fn process_frame(
        &mut self,
        left_image: &GrayImage,
        right_image: &GrayImage,
    ) -> Vec<TrackedFeature> {
        let frame_idx = self.frame_idx;
        self.frame_idx += 1;

        // Step 1: Track existing features using LK optical flow
        let tracked_positions = if let Some(ref prev_left) = self.prev_left {
            self.track_existing_features(prev_left, left_image)
        } else {
            HashMap::new()
        };

        // Step 2: Update tracked feature positions
        for (id, new_pos) in &tracked_positions {
            if let Some(feature) = self.tracks.get_mut(id) {
                feature.stereo.left_kp.x = new_pos.0;
                feature.stereo.left_kp.y = new_pos.1;
            }
        }

        // Step 3: Perform stereo matching for all tracked features
        self.update_stereo_matches(left_image, right_image);

        // Step 4: Remove features that failed stereo matching for too long
        self.prune_dead_tracks();

        // Step 5: Detect and add new features if below minimum
        if self.tracks.len() < self.config.min_features {
            self.detect_new_features(left_image, right_image, frame_idx);
        }

        // Store current frame for next iteration
        self.prev_left = Some(left_image.clone());

        // Return current tracks
        self.tracks.values().cloned().collect()
    }

    /// Track existing features from previous frame to current frame
    fn track_existing_features(
        &self,
        prev_image: &GrayImage,
        curr_image: &GrayImage,
    ) -> HashMap<usize, (f32, f32)> {
        let mut result = HashMap::new();

        if self.tracks.is_empty() {
            return result;
        }

        // Collect points to track
        let ids: Vec<usize> = self.tracks.keys().copied().collect();
        let points: Vec<(f32, f32)> = ids
            .iter()
            .map(|id| {
                let f = &self.tracks[id];
                (f.stereo.left_kp.x, f.stereo.left_kp.y)
            })
            .collect();

        // Run LK tracking
        let track_results = self.lk_tracker.track(prev_image, curr_image, &points);

        // Collect successful tracks
        for (id, track_result) in ids.iter().zip(track_results.iter()) {
            if track_result.success {
                result.insert(*id, track_result.position);
            }
        }

        result
    }

    /// Update stereo matches for tracked features
    fn update_stereo_matches(&mut self, left_image: &GrayImage, right_image: &GrayImage) {
        // Detect features in right image for stereo matching
        let right_keypoints = self.detector.detect(right_image);
        let right_features = self.extractor.compute_all(right_image, &right_keypoints);

        // For each tracked feature, try to find stereo match
        let track_ids: Vec<usize> = self.tracks.keys().copied().collect();

        for id in track_ids {
            let track = self.tracks.get_mut(&id).unwrap();

            // Recompute descriptor at new position
            let kp = track.stereo.left_kp;
            if let Some(desc) = self.extractor.compute(left_image, &kp) {
                // Find best match in right image
                if let Some((right_kp, disparity)) =
                    find_stereo_match(&kp, &desc, &right_features)
                {
                    track.stereo.right_kp = right_kp;
                    track.stereo.descriptor = desc;
                    track.stereo.disparity = disparity;
                    track.age = 0; // Reset age on successful stereo match
                } else {
                    track.age += 1; // Increment age on failed stereo match
                }
            } else {
                track.age += 1;
            }
        }
    }

    /// Remove features that have failed stereo matching for too long
    fn prune_dead_tracks(&mut self) {
        let max_age = self.config.max_age_without_stereo;
        self.tracks.retain(|_, track| track.age <= max_age);
    }

    /// Detect new features in areas without existing tracks
    fn detect_new_features(
        &mut self,
        left_image: &GrayImage,
        right_image: &GrayImage,
        frame_idx: usize,
    ) {
        let (width, height) = left_image.dimensions();

        // Build occupancy grid of existing features
        let grid_size = self.config.grid_size;
        let grid_width = (width as usize + grid_size - 1) / grid_size;
        let grid_height = (height as usize + grid_size - 1) / grid_size;
        let mut occupied = vec![false; grid_width * grid_height];

        for track in self.tracks.values() {
            let gx = (track.stereo.left_kp.x as usize) / grid_size;
            let gy = (track.stereo.left_kp.y as usize) / grid_size;
            if gx < grid_width && gy < grid_height {
                occupied[gy * grid_width + gx] = true;
            }
        }

        // Detect features
        let left_keypoints = self.detector.detect(left_image);
        let left_features = self.extractor.compute_all(left_image, &left_keypoints);

        // Detect in right for stereo matching
        let right_keypoints = self.detector.detect(right_image);
        let right_features = self.extractor.compute_all(right_image, &right_keypoints);

        // Match and add new features in unoccupied cells
        let mut new_features = Vec::new();

        for (left_kp, left_desc) in &left_features {
            // Check if cell is occupied
            let gx = (left_kp.x as usize) / grid_size;
            let gy = (left_kp.y as usize) / grid_size;
            if gx >= grid_width || gy >= grid_height {
                continue;
            }
            if occupied[gy * grid_width + gx] {
                continue;
            }

            // Try to find stereo match
            if let Some((right_kp, disparity)) =
                find_stereo_match(left_kp, left_desc, &right_features)
            {
                new_features.push(TrackedFeature {
                    id: self.next_id,
                    stereo: StereoMatch {
                        left_kp: *left_kp,
                        right_kp,
                        descriptor: *left_desc,
                        disparity,
                    },
                    age: 0,
                    first_frame: frame_idx,
                });
                self.next_id += 1;

                // Mark cell as occupied
                occupied[gy * grid_width + gx] = true;

                // Check if we have enough features
                if self.tracks.len() + new_features.len() >= self.config.max_features {
                    break;
                }
            }
        }

        // Add new features to tracks
        for feature in new_features {
            self.tracks.insert(feature.id, feature);
        }
    }

    /// Get current number of tracked features
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Get current frame index
    pub fn frame_index(&self) -> usize {
        self.frame_idx
    }

    /// Get all current tracks
    pub fn get_tracks(&self) -> &HashMap<usize, TrackedFeature> {
        &self.tracks
    }
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Find stereo match for a single feature (free function to avoid borrow issues)
fn find_stereo_match(
    left_kp: &KeyPoint,
    left_desc: &BriefDescriptor,
    right_features: &[(KeyPoint, BriefDescriptor)],
) -> Option<(KeyPoint, f32)> {
    let max_vertical_diff = 2.0;
    let min_disparity = 1.0;
    let max_disparity = 200.0;
    let max_hamming = 64;

    let mut best_match: Option<(KeyPoint, f32, u32)> = None;
    let mut second_best_dist = u32::MAX;

    for (right_kp, right_desc) in right_features {
        // Epipolar constraint
        let vertical_diff = (left_kp.y - right_kp.y).abs();
        if vertical_diff > max_vertical_diff {
            continue;
        }

        // Disparity constraint
        let disparity = left_kp.x - right_kp.x;
        if disparity < min_disparity || disparity > max_disparity {
            continue;
        }

        let dist = left_desc.hamming_distance(right_desc);

        if let Some((_, _, best_dist)) = best_match {
            if dist < best_dist {
                second_best_dist = best_dist;
                best_match = Some((*right_kp, disparity, dist));
            } else if dist < second_best_dist {
                second_best_dist = dist;
            }
        } else {
            best_match = Some((*right_kp, disparity, dist));
        }
    }

    // Apply thresholds
    if let Some((right_kp, disparity, dist)) = best_match {
        if dist <= max_hamming {
            // Ratio test
            if second_best_dist == u32::MAX || (dist as f32 / second_best_dist as f32) < 0.8 {
                return Some((right_kp, disparity));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_empty_images() {
        let mut tracker = Tracker::new();
        let left = GrayImage::new(100, 100);
        let right = GrayImage::new(100, 100);

        let tracks = tracker.process_frame(&left, &right);
        // Uniform images should have no features
        assert!(tracks.is_empty());
    }
}
