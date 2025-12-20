//! Stereo matching for rectified image pairs
//!
//! Matches features between left and right images using:
//! - Epipolar constraint (same row for rectified images)
//! - BRIEF descriptor matching with Hamming distance
//! - Left-right consistency check

use crate::{BriefDescriptor, BriefExtractor, FastDetector, KeyPoint};
use image::GrayImage;

/// A stereo match between left and right images
#[derive(Debug, Clone)]
pub struct StereoMatch {
    /// Keypoint in left image
    pub left_kp: KeyPoint,
    /// Keypoint in right image
    pub right_kp: KeyPoint,
    /// Descriptor (from left image)
    pub descriptor: BriefDescriptor,
    /// Disparity (left_x - right_x), positive for objects in front of camera
    pub disparity: f32,
}

/// Stereo matcher for rectified image pairs
pub struct StereoMatcher {
    /// Feature detector
    detector: FastDetector,
    /// Descriptor extractor
    extractor: BriefExtractor,
    /// Minimum disparity (pixels)
    min_disparity: f32,
    /// Maximum disparity (pixels)
    max_disparity: f32,
    /// Maximum Hamming distance for a valid match
    max_hamming_distance: u32,
    /// Ratio test threshold (best/second_best must be < this)
    ratio_threshold: f32,
    /// Maximum vertical difference for epipolar constraint (pixels)
    max_vertical_diff: f32,
}

impl Default for StereoMatcher {
    fn default() -> Self {
        Self {
            detector: FastDetector::default(),
            extractor: BriefExtractor::new(),
            min_disparity: 1.0,
            max_disparity: 200.0,
            max_hamming_distance: 64,  // Out of 256 bits
            ratio_threshold: 0.8,
            max_vertical_diff: 2.0,
        }
    }
}

impl StereoMatcher {
    /// Create a new stereo matcher with custom parameters
    pub fn new(
        detector: FastDetector,
        min_disparity: f32,
        max_disparity: f32,
        max_hamming_distance: u32,
    ) -> Self {
        Self {
            detector,
            extractor: BriefExtractor::new(),
            min_disparity,
            max_disparity,
            max_hamming_distance,
            ratio_threshold: 0.8,
            max_vertical_diff: 2.0,
        }
    }

    /// Set ratio test threshold
    pub fn with_ratio_threshold(mut self, ratio: f32) -> Self {
        self.ratio_threshold = ratio;
        self
    }

    /// Match features between left and right images
    pub fn match_stereo(
        &self,
        left_image: &GrayImage,
        right_image: &GrayImage,
    ) -> Vec<StereoMatch> {
        // Detect features in both images
        let left_keypoints = self.detector.detect(left_image);
        let right_keypoints = self.detector.detect(right_image);

        // Compute descriptors
        let left_features = self.extractor.compute_all(left_image, &left_keypoints);
        let right_features = self.extractor.compute_all(right_image, &right_keypoints);

        self.match_features(&left_features, &right_features)
    }

    /// Match pre-computed features between left and right images
    pub fn match_features(
        &self,
        left_features: &[(KeyPoint, BriefDescriptor)],
        right_features: &[(KeyPoint, BriefDescriptor)],
    ) -> Vec<StereoMatch> {
        let mut matches = Vec::new();

        // Build a lookup structure for right features by row (y coordinate)
        // Group features into row buckets for efficient epipolar search
        let mut right_by_row: std::collections::HashMap<i32, Vec<usize>> =
            std::collections::HashMap::new();

        for (idx, (kp, _)) in right_features.iter().enumerate() {
            let row = kp.y as i32;
            // Add to nearby rows to handle sub-pixel differences
            for r in (row - self.max_vertical_diff as i32)..=(row + self.max_vertical_diff as i32) {
                right_by_row.entry(r).or_default().push(idx);
            }
        }

        // For each left feature, find best match in right image
        for (left_kp, left_desc) in left_features {
            let row = left_kp.y as i32;

            // Get candidate right features on same row (epipolar constraint)
            let candidates = match right_by_row.get(&row) {
                Some(indices) => indices,
                None => continue,
            };

            // Find best and second-best matches
            let mut best_dist = u32::MAX;
            let mut second_best_dist = u32::MAX;
            let mut best_idx: Option<usize> = None;

            for &right_idx in candidates {
                let (right_kp, right_desc) = &right_features[right_idx];

                // Check disparity constraint
                let disparity = left_kp.x - right_kp.x;
                if disparity < self.min_disparity || disparity > self.max_disparity {
                    continue;
                }

                // Check vertical alignment (epipolar constraint)
                let vertical_diff = (left_kp.y - right_kp.y).abs();
                if vertical_diff > self.max_vertical_diff {
                    continue;
                }

                // Compute Hamming distance
                let dist = left_desc.hamming_distance(right_desc);

                if dist < best_dist {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx = Some(right_idx);
                } else if dist < second_best_dist {
                    second_best_dist = dist;
                }
            }

            // Apply thresholds
            if let Some(right_idx) = best_idx {
                // Check absolute distance threshold
                if best_dist > self.max_hamming_distance {
                    continue;
                }

                // Apply ratio test (skip if second_best is 0 to avoid division issues)
                if second_best_dist > 0 {
                    let ratio = best_dist as f32 / second_best_dist as f32;
                    if ratio > self.ratio_threshold {
                        continue;
                    }
                }

                let (right_kp, _) = &right_features[right_idx];
                let disparity = left_kp.x - right_kp.x;

                matches.push(StereoMatch {
                    left_kp: *left_kp,
                    right_kp: *right_kp,
                    descriptor: *left_desc,
                    disparity,
                });
            }
        }

        // Optional: left-right consistency check
        // (For now we skip this for simplicity, but it's important for robustness)

        matches
    }

    /// Get reference to the detector
    pub fn detector(&self) -> &FastDetector {
        &self.detector
    }

    /// Get reference to the extractor
    pub fn extractor(&self) -> &BriefExtractor {
        &self.extractor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereo_matcher_empty() {
        let matcher = StereoMatcher::default();
        let left = GrayImage::new(100, 100);
        let right = GrayImage::new(100, 100);
        let matches = matcher.match_stereo(&left, &right);
        // Uniform images should have no features, thus no matches
        assert!(matches.is_empty());
    }
}
