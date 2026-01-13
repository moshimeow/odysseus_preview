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
    BriefDescriptor, BriefExtractor, FastDetector, KeyPoint, StereoCamera, StereoMatch, StereoMatcher,
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
    /// Default depth for stereo matching initialization (meters)
    /// Used when backend depth estimates are unavailable
    pub default_stereo_depth: f64,
    /// Epipolar error threshold (pixels)
    pub epipolar_error_threshold: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            min_features: 100,
            max_features: 300,
            max_age_without_stereo: 2, // Frames without a stereo match before dropping a feature
            lk_config: LKConfig {
                max_iterations: 30,
                epsilon: 0.01,
                num_levels: 3,          // More pyramid levels for larger motions
                min_eigenvalue: 20.0,   // Stricter: reject tracking if landed on weak texture
                forward_backward_threshold: 0.2, // Tighter consistency check (Basalt uses 0.2)
            },
            grid_size: 64,
            default_stereo_depth: 3.0,  // 3 meters reasonable for indoor/outdoor
            epipolar_error_threshold: 1.0,  // Tighter vertical alignment for rectified stereo
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
    /// Previous right image (for temporal tracking)
    prev_right: Option<GrayImage>,
    /// Per-feature depth estimates (feature_id -> depth in meters)
    feature_depths: HashMap<usize, f32>,
    /// Camera model for depth-based stereo initialization
    camera: Option<StereoCamera>,
}

impl Tracker {
    /// Create a new tracker with default configuration
    pub fn new() -> Self {
        Self::with_config(TrackerConfig::default())
    }

    /// Create a new tracker with custom configuration
    pub fn with_config(config: TrackerConfig) -> Self {
        // Configure detector - higher threshold for better quality features
        let detector = FastDetector::new(30, 32, config.max_features)
            .with_min_eigen_threshold(20.0)  // Match tracking threshold
            .with_subpixel_refinement(true);

        Self {
            lk_tracker: LKTracker::with_config(config.lk_config.clone()),
            config,
            tracks: HashMap::new(),
            next_id: 0,
            frame_idx: 0,
            detector,
            extractor: BriefExtractor::new(),
            stereo_matcher: StereoMatcher::default(),
            prev_left: None,
            prev_right: None,
            feature_depths: HashMap::new(),
            camera: None,
        }
    }

    /// Set the camera model for depth-based stereo initialization
    pub fn with_camera(mut self, camera: StereoCamera) -> Self {
        self.camera = Some(camera);
        self
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
        // Features that fail LK tracking keep their old positions
        // and will be pruned if stereo matching fails repeatedly
        for (id, new_pos) in &tracked_positions {
            if let Some(feature) = self.tracks.get_mut(id) {
                feature.stereo.left_kp.x = new_pos.0;
                feature.stereo.left_kp.y = new_pos.1;
            }
        }

        // Step 3: Perform stereo matching for all tracked features
        self.update_stereo_matches(left_image, right_image);

        // Step 3.5: Filter epipolar outliers
        self.filter_epipolar_outliers();

        // Step 4: Remove features that failed stereo matching for too long
        self.prune_dead_tracks();

        // Step 5: Detect and add new features if below minimum
        if self.tracks.len() < self.config.min_features {
            self.detect_new_features(left_image, right_image, frame_idx);
        }

        // Store current frames for next iteration
        self.prev_left = Some(left_image.clone());
        self.prev_right = Some(right_image.clone());

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

    /// Update stereo matches using temporal tracking with depth-based initialization
    fn update_stereo_matches(&mut self, left_image: &GrayImage, right_image: &GrayImage) {
        let track_ids: Vec<usize> = self.tracks.keys().copied().collect();

        // Step 1: Track right camera temporally (if we have previous right)
        let right_tracked = if let Some(ref prev_right) = self.prev_right {
            self.track_right_camera_temporal(prev_right, right_image, &track_ids)
        } else {
            HashMap::new()
        };

        // Step 2: For each left feature, get or compute right position
        // Collect data needed for tracking first to avoid borrow issues
        let track_data: Vec<(usize, KeyPoint, Option<(f32, f32)>)> = track_ids
            .iter()
            .map(|&id| {
                let track = &self.tracks[&id];
                let left_kp = track.stereo.left_kp;
                
                // Check if temporal tracking succeeded for right camera
                let temporal_right = right_tracked.get(&id).copied();
                
                (id, left_kp, temporal_right)
            })
            .collect();
        
        // Now perform tracking and update tracks
        for (id, left_kp, temporal_right) in track_data {
            // If temporal tracking succeeded, trust it! Otherwise fall back to stereo matching
            let right_kp_result = if let Some(temporal_pos) = temporal_right {
                // Temporal tracking succeeded - use it directly
                Some(KeyPoint {
                    x: temporal_pos.0,
                    y: temporal_pos.1,
                    response: left_kp.response,
                    angle: 0.0,
                })
            } else {
                // Temporal tracking failed - compute depth-based init and run stereo LK
                let depth = self.feature_depths.get(&id)
                    .copied()
                    .unwrap_or(self.config.default_stereo_depth as f32);
                
                let right_init = if let Some(ref cam) = self.camera {
                    cam.project_left_to_right(left_kp.x, left_kp.y, depth)
                } else {
                    // Fallback: assume rectified stereo, estimate disparity
                    let disparity = 50.0;
                    (left_kp.x - disparity, left_kp.y)
                };
                
                self.track_stereo_with_init(left_image, right_image, &left_kp, right_init)
            };
            
            if let Some(right_kp) = right_kp_result {
                let track = self.tracks.get_mut(&id).unwrap();
                
                // Update right position
                track.stereo.right_kp = right_kp;

                // Update disparity
                let disparity = left_kp.x - right_kp.x;
                track.stereo.disparity = disparity;

                // Update depth estimate from disparity
                if disparity > 1.0 {
                    if let Some(ref cam) = self.camera {
                        let new_depth = cam.baseline * cam.left.fx / disparity;
                        self.feature_depths.insert(id, new_depth);
                    }
                }

                // Recompute descriptor at new position
                if let Some(desc) = self.extractor.compute(left_image, &left_kp) {
                    track.stereo.descriptor = desc;
                }

                track.age = 0;  // Reset age on success
            } else {
                let track = self.tracks.get_mut(&id).unwrap();
                track.age += 1;  // Failed stereo match
            }
        }
    }

    /// Track right camera temporally from previous frame
    fn track_right_camera_temporal(
        &self,
        prev_right: &GrayImage,
        curr_right: &GrayImage,
        track_ids: &[usize],
    ) -> HashMap<usize, (f32, f32)> {
        let points: Vec<(f32, f32)> = track_ids
            .iter()
            .map(|id| {
                let track = &self.tracks[id];
                (track.stereo.right_kp.x, track.stereo.right_kp.y)
            })
            .collect();

        let results = self.lk_tracker.track(prev_right, curr_right, &points);

        track_ids
            .iter()
            .zip(results.iter())
            .filter_map(|(id, result)| {
                if result.success {
                    Some((*id, result.position))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Track stereo correspondence with depth-based initialization
    fn track_stereo_with_init(
        &self,
        left_image: &GrayImage,
        right_image: &GrayImage,
        left_kp: &KeyPoint,
        right_init: (f32, f32),
    ) -> Option<KeyPoint> {
        // Track from left image to right image
        // Source point: left_kp position in left_image
        // Initial guess: right_init position in right_image
        let result = self.lk_tracker.track_with_guess(
            left_image, 
            right_image, 
            &[(left_kp.x, left_kp.y)],
            Some(&[right_init])
        );

        if !result[0].success {
            return None;
        }

        let right_pos = result[0].position;

        // Basic epipolar constraint: vertical difference should be small
        let vertical_diff = (left_kp.y - right_pos.1).abs();
        if vertical_diff > 2.0 {
            return None;
        }

        // Disparity should be positive (right image shifted left)
        let disparity = left_kp.x - right_pos.0;
        if disparity < 1.0 || disparity > 200.0 {
            return None;
        }

        Some(KeyPoint {
            x: right_pos.0,
            y: right_pos.1,
            response: left_kp.response,
            angle: 0.0,  // Angle not used for stereo matching
        })
    }

    /// Filter out tracks with excessive epipolar error
    fn filter_epipolar_outliers(&mut self) {
        let threshold = self.config.epipolar_error_threshold;

        let mut to_remove = Vec::new();

        for (id, track) in &self.tracks {
            let left_kp = &track.stereo.left_kp;
            let right_kp = &track.stereo.right_kp;

            // Simple epipolar check: vertical alignment
            let vertical_error = (left_kp.y - right_kp.y).abs();

            if vertical_error > threshold {
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            self.tracks.remove(&id);
            self.feature_depths.remove(&id);
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

    /// Update depth estimates from backend
    ///
    /// # Arguments
    /// * `feature_depths` - Map from feature ID to depth in meters
    pub fn update_depth_estimates(&mut self, feature_depths: HashMap<usize, f32>) {
        for (id, depth) in feature_depths {
            if self.tracks.contains_key(&id) {
                self.feature_depths.insert(id, depth);
            }
        }
    }

    /// Get current depth estimates (for backend)
    pub fn get_depth_estimates(&self) -> &HashMap<usize, f32> {
        &self.feature_depths
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
    let max_hamming = 48; // Stricter descriptor matching (was 64)

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
            // Ratio test - stricter threshold (was 0.8)
            if second_best_dist == u32::MAX || (dist as f32 / second_best_dist as f32) < 0.7 {
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
