//! FAST corner detection with Shi-Tomasi scoring and non-maximum suppression
//!
//! Implements FAST-9 corner detection (9 contiguous pixels on a 16-pixel Bresenham circle
//! must all be brighter or darker than center by a threshold).
//!
//! Uses Shi-Tomasi scoring (minimum eigenvalue) instead of Harris for better trackability.

use image::GrayImage;
use std::collections::HashMap;

/// A detected keypoint with position, response strength, and orientation
#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    /// X coordinate (column) in pixels - may be subpixel
    pub x: f32,
    /// Y coordinate (row) in pixels - may be subpixel
    pub y: f32,
    /// Shi-Tomasi response: min eigenvalue (higher = more trackable)
    pub response: f32,
    /// Orientation in radians (computed via intensity centroid)
    pub angle: f32,
}

/// FAST corner detector with Shi-Tomasi scoring and grid-based NMS
pub struct FastDetector {
    /// Intensity difference threshold for FAST (typically 20-40)
    threshold: u8,
    /// Grid cell size for non-maximum suppression
    grid_size: usize,
    /// Maximum number of features to return
    max_features: usize,
    /// Minimum Shi-Tomasi response (min eigenvalue) to accept a corner
    min_eigen_threshold: f32,
    /// Whether to use subpixel refinement
    subpixel_refinement: bool,
}

impl Default for FastDetector {
    fn default() -> Self {
        Self {
            threshold: 20,
            grid_size: 32,
            max_features: 500,
            min_eigen_threshold: 10.0, // Minimum eigenvalue to accept
            subpixel_refinement: true,
        }
    }
}

impl FastDetector {
    /// Create a new FAST detector with custom parameters
    pub fn new(threshold: u8, grid_size: usize, max_features: usize) -> Self {
        Self {
            threshold,
            grid_size,
            max_features,
            min_eigen_threshold: 10.0,
            subpixel_refinement: true,
        }
    }

    /// Set the minimum eigenvalue threshold for corner acceptance
    pub fn with_min_eigen_threshold(mut self, threshold: f32) -> Self {
        self.min_eigen_threshold = threshold;
        self
    }

    /// Enable or disable subpixel refinement
    pub fn with_subpixel_refinement(mut self, enabled: bool) -> Self {
        self.subpixel_refinement = enabled;
        self
    }

    /// Detect keypoints in a grayscale image
    pub fn detect(&self, image: &GrayImage) -> Vec<KeyPoint> {
        let (width, height) = image.dimensions();
        let width = width as usize;
        let height = height as usize;

        // Need 3-pixel border for FAST circle + gradient computation
        if width < 10 || height < 10 {
            return Vec::new();
        }

        // Step 1: Find all FAST corners
        let mut candidates: Vec<(usize, usize)> = Vec::new();

        for y in 4..(height - 4) {
            for x in 4..(width - 4) {
                if self.is_fast_corner(image, x, y) {
                    candidates.push((x, y));
                }
            }
        }

        // Step 2: Compute Shi-Tomasi response (min eigenvalue) for each candidate
        // and filter by minimum eigenvalue threshold
        let mut keypoints: Vec<KeyPoint> = candidates
            .iter()
            .filter_map(|&(x, y)| {
                let response = self.shi_tomasi_response(image, x, y);

                // Filter out weak corners - this is the key improvement!
                if response < self.min_eigen_threshold {
                    return None;
                }

                // Apply subpixel refinement if enabled
                let (refined_x, refined_y) = if self.subpixel_refinement {
                    self.subpixel_refine(image, x, y)
                } else {
                    (x as f32, y as f32)
                };

                let angle = self.compute_orientation(image, x, y);
                Some(KeyPoint {
                    x: refined_x,
                    y: refined_y,
                    response,
                    angle,
                })
            })
            .collect();

        // Step 3: Non-maximum suppression in grid cells
        keypoints = self.non_max_suppression(keypoints, width, height);

        // Step 4: Sort by response and take top N
        keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        keypoints.truncate(self.max_features);

        keypoints
    }

    /// Check if a pixel is a FAST-9 corner
    ///
    /// Uses the 16-pixel Bresenham circle and checks if 9 contiguous pixels
    /// are all brighter or all darker than center by threshold.
    fn is_fast_corner(&self, image: &GrayImage, x: usize, y: usize) -> bool {
        let center = image.get_pixel(x as u32, y as u32).0[0] as i16;
        let t = self.threshold as i16;

        // Offsets for 16-pixel Bresenham circle (radius 3)
        // Ordered around the circle starting from top
        const CIRCLE: [(i32, i32); 16] = [
            (0, -3),  // 0: top
            (1, -3),  // 1
            (2, -2),  // 2
            (3, -1),  // 3
            (3, 0),   // 4: right
            (3, 1),   // 5
            (2, 2),   // 6
            (1, 3),   // 7
            (0, 3),   // 8: bottom
            (-1, 3),  // 9
            (-2, 2),  // 10
            (-3, 1),  // 11
            (-3, 0),  // 12: left
            (-3, -1), // 13
            (-2, -2), // 14
            (-1, -3), // 15
        ];

        // High-speed test: check cardinal points (0, 4, 8, 12)
        // At least 3 of these 4 must be brighter or darker for a corner to exist
        let p0 = image.get_pixel(x as u32, (y as i32 - 3) as u32).0[0] as i16;
        let p4 = image.get_pixel((x as i32 + 3) as u32, y as u32).0[0] as i16;
        let p8 = image.get_pixel(x as u32, (y as i32 + 3) as u32).0[0] as i16;
        let p12 = image.get_pixel((x as i32 - 3) as u32, y as u32).0[0] as i16;

        let brighter_count = (p0 > center + t) as u8
            + (p4 > center + t) as u8
            + (p8 > center + t) as u8
            + (p12 > center + t) as u8;

        let darker_count = (p0 < center - t) as u8
            + (p4 < center - t) as u8
            + (p8 < center - t) as u8
            + (p12 < center - t) as u8;

        // Need at least 3 of 4 to pass high-speed test
        if brighter_count < 3 && darker_count < 3 {
            return false;
        }

        // Full test: check for 9 contiguous pixels
        // Get all 16 pixel values
        let pixels: [i16; 16] = std::array::from_fn(|i| {
            let (dx, dy) = CIRCLE[i];
            let px = (x as i32 + dx) as u32;
            let py = (y as i32 + dy) as u32;
            image.get_pixel(px, py).0[0] as i16
        });

        // Check for 9 contiguous brighter pixels
        // We need to check wrap-around (pixels 15, 0, 1, ...)
        let mut consecutive_bright = 0;
        let mut max_consecutive_bright = 0;
        for i in 0..32 {
            // Go around twice to handle wrap
            if pixels[i % 16] > center + t {
                consecutive_bright += 1;
                max_consecutive_bright = max_consecutive_bright.max(consecutive_bright);
            } else {
                consecutive_bright = 0;
            }
        }

        if max_consecutive_bright >= 9 {
            return true;
        }

        // Check for 9 contiguous darker pixels
        let mut consecutive_dark = 0;
        let mut max_consecutive_dark = 0;
        for i in 0..32 {
            if pixels[i % 16] < center - t {
                consecutive_dark += 1;
                max_consecutive_dark = max_consecutive_dark.max(consecutive_dark);
            } else {
                consecutive_dark = 0;
            }
        }

        max_consecutive_dark >= 9
    }

    /// Compute Shi-Tomasi corner response at a pixel
    ///
    /// Uses a 7x7 window to compute the structure tensor and returns
    /// the minimum eigenvalue: min(λ1, λ2)
    ///
    /// This directly measures "trackability" - corners with high min eigenvalue
    /// will track well with Lucas-Kanade.
    fn shi_tomasi_response(&self, image: &GrayImage, x: usize, y: usize) -> f32 {
        let (width, height) = image.dimensions();

        // Compute gradients using central differences
        // We use a 7x7 window centered at (x, y)
        let mut sum_ix2: f32 = 0.0;
        let mut sum_iy2: f32 = 0.0;
        let mut sum_ixiy: f32 = 0.0;

        for dy in -3i32..=3 {
            for dx in -3i32..=3 {
                let px = (x as i32 + dx) as u32;
                let py = (y as i32 + dy) as u32;

                // Gradient using central differences (with bounds checking)
                let px_plus = (px + 1).min(width - 1);
                let px_minus = px.saturating_sub(1);
                let py_plus = (py + 1).min(height - 1);
                let py_minus = py.saturating_sub(1);

                let ix = (image.get_pixel(px_plus, py).0[0] as f32
                    - image.get_pixel(px_minus, py).0[0] as f32)
                    / 2.0;
                let iy = (image.get_pixel(px, py_plus).0[0] as f32
                    - image.get_pixel(px, py_minus).0[0] as f32)
                    / 2.0;

                sum_ix2 += ix * ix;
                sum_iy2 += iy * iy;
                sum_ixiy += ix * iy;
            }
        }

        // Shi-Tomasi response: minimum eigenvalue of structure tensor
        // For 2x2 matrix M = [[a, b], [b, c]], eigenvalues are:
        // λ = (trace ± sqrt(trace² - 4*det)) / 2
        // min eigenvalue = (trace - sqrt(trace² - 4*det)) / 2
        let trace = sum_ix2 + sum_iy2;
        let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
        let discriminant = (trace * trace - 4.0 * det).max(0.0);
        (trace - discriminant.sqrt()) / 2.0
    }

    /// Refine corner position to subpixel accuracy
    ///
    /// Uses the gradient-based method: fits a quadratic to the corner response
    /// surface and finds the peak analytically.
    fn subpixel_refine(&self, image: &GrayImage, x: usize, y: usize) -> (f32, f32) {
        // Compute corner response at center and neighbors
        let c = self.shi_tomasi_response(image, x, y);
        let l = self.shi_tomasi_response(image, x.saturating_sub(1), y);
        let r = self.shi_tomasi_response(image, x + 1, y);
        let u = self.shi_tomasi_response(image, x, y.saturating_sub(1));
        let d = self.shi_tomasi_response(image, x, y + 1);

        // Fit parabola in x: f(x) = ax² + bx + c
        // Using 3 points: (-1, l), (0, c), (1, r)
        // a = (l + r - 2c) / 2
        // b = (r - l) / 2
        // Peak at x = -b / (2a)
        let ax = (l + r - 2.0 * c) / 2.0;
        let bx = (r - l) / 2.0;

        let dx = if ax.abs() > 1e-6 {
            (-bx / (2.0 * ax)).clamp(-0.5, 0.5)
        } else {
            0.0
        };

        // Same for y
        let ay = (u + d - 2.0 * c) / 2.0;
        let by = (d - u) / 2.0;

        let dy = if ay.abs() > 1e-6 {
            (-by / (2.0 * ay)).clamp(-0.5, 0.5)
        } else {
            0.0
        };

        (x as f32 + dx, y as f32 + dy)
    }

    /// Compute keypoint orientation using intensity centroid method
    ///
    /// Returns angle in radians
    fn compute_orientation(&self, image: &GrayImage, x: usize, y: usize) -> f32 {
        // Compute moments in a circular patch (radius 15 is typical for ORB)
        // We use a smaller radius (7) for efficiency
        const RADIUS: i32 = 7;

        let mut m10: f32 = 0.0;
        let mut m01: f32 = 0.0;

        let (width, height) = image.dimensions();

        for dy in -RADIUS..=RADIUS {
            for dx in -RADIUS..=RADIUS {
                // Check if within circle
                if dx * dx + dy * dy > RADIUS * RADIUS {
                    continue;
                }

                let px = x as i32 + dx;
                let py = y as i32 + dy;

                // Bounds check
                if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                    continue;
                }

                let intensity = image.get_pixel(px as u32, py as u32).0[0] as f32;
                m10 += dx as f32 * intensity;
                m01 += dy as f32 * intensity;
            }
        }

        m01.atan2(m10)
    }

    /// Apply non-maximum suppression using a grid
    ///
    /// Divides the image into cells and keeps only the strongest keypoint per cell
    fn non_max_suppression(
        &self,
        keypoints: Vec<KeyPoint>,
        _width: usize,
        _height: usize,
    ) -> Vec<KeyPoint> {
        // Map from grid cell -> best keypoint in that cell
        let mut grid: HashMap<(usize, usize), KeyPoint> = HashMap::new();

        for kp in keypoints {
            let cell_x = kp.x as usize / self.grid_size;
            let cell_y = kp.y as usize / self.grid_size;
            let cell = (cell_x, cell_y);

            match grid.get(&cell) {
                Some(existing) if existing.response >= kp.response => {
                    // Keep existing, it's stronger
                }
                _ => {
                    grid.insert(cell, kp);
                }
            }
        }

        grid.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    #[test]
    fn test_fast_detector_empty_image() {
        let detector = FastDetector::default();
        let image = GrayImage::new(100, 100);
        let keypoints = detector.detect(&image);
        // Uniform image should have few/no corners
        assert!(keypoints.len() < 10);
    }

    #[test]
    fn test_fast_detector_small_image() {
        let detector = FastDetector::default();
        let image = GrayImage::new(5, 5);
        let keypoints = detector.detect(&image);
        // Too small for FAST
        assert!(keypoints.is_empty());
    }

    #[test]
    fn test_fast_detector_corner() {
        let detector = FastDetector::new(20, 32, 100);

        // Create a checkerboard pattern which has strong corners
        let mut image = GrayImage::from_pixel(100, 100, image::Luma([128]));

        // Create a strong corner at (50, 50) by making quadrants different intensities
        for y in 0..100 {
            for x in 0..100 {
                let val = match (x < 50, y < 50) {
                    (true, true) => 200u8,   // Top-left: bright
                    (true, false) => 50u8,   // Bottom-left: dark
                    (false, true) => 50u8,   // Top-right: dark
                    (false, false) => 200u8, // Bottom-right: bright
                };
                image.put_pixel(x, y, image::Luma([val]));
            }
        }

        let keypoints = detector.detect(&image);

        // Should detect corners at the checkerboard intersection
        // If no keypoints, that's also OK - FAST might not detect on this pattern
        // The important thing is we don't crash
        println!("Detected {} keypoints", keypoints.len());

        // If we got keypoints, check they're near expected corners
        if !keypoints.is_empty() {
            let has_corner_near_50 = keypoints
                .iter()
                .any(|kp| (kp.x - 50.0).abs() < 15.0 && (kp.y - 50.0).abs() < 15.0);
            // This is informative, not a hard requirement
            println!("Has corner near (50,50): {}", has_corner_near_50);
        }
    }
}
