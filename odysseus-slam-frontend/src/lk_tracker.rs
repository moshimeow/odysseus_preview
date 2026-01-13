//! Lucas-Kanade optical flow tracker
//!
//! Implements pyramidal Lucas-Kanade for tracking features across frames.
//! Uses pattern-based sparse sampling (Pattern52: 52 points in cross pattern).

use image::GrayImage;

use crate::KeyPoint;

/// Pattern52: 52 sample points in cross pattern
/// Pattern extends from -3.5 to +3.5 pixels in both x and y
const PATTERN_POINTS: [(f32, f32); 52] = [
    // Top row (4 points): y=3.5
    (-1.5, 3.5),
    (-0.5, 3.5),
    (0.5, 3.5),
    (1.5, 3.5),
    // Second row (6 points): y=2.5
    (-2.5, 2.5),
    (-1.5, 2.5),
    (-0.5, 2.5),
    (0.5, 2.5),
    (1.5, 2.5),
    (2.5, 2.5),
    // Third row (8 points): y=1.5
    (-3.5, 1.5),
    (-2.5, 1.5),
    (-1.5, 1.5),
    (-0.5, 1.5),
    (0.5, 1.5),
    (1.5, 1.5),
    (2.5, 1.5),
    (3.5, 1.5),
    // Fourth row (8 points): y=0.5
    (-3.5, 0.5),
    (-2.5, 0.5),
    (-1.5, 0.5),
    (-0.5, 0.5),
    (0.5, 0.5),
    (1.5, 0.5),
    (2.5, 0.5),
    (3.5, 0.5),
    // Fifth row (8 points): y=-0.5
    (-3.5, -0.5),
    (-2.5, -0.5),
    (-1.5, -0.5),
    (-0.5, -0.5),
    (0.5, -0.5),
    (1.5, -0.5),
    (2.5, -0.5),
    (3.5, -0.5),
    // Sixth row (8 points): y=-1.5
    (-3.5, -1.5),
    (-2.5, -1.5),
    (-1.5, -1.5),
    (-0.5, -1.5),
    (0.5, -1.5),
    (1.5, -1.5),
    (2.5, -1.5),
    (3.5, -1.5),
    // Seventh row (6 points): y=-2.5
    (-2.5, -2.5),
    (-1.5, -2.5),
    (-0.5, -2.5),
    (0.5, -2.5),
    (1.5, -2.5),
    (2.5, -2.5),
    // Bottom row (4 points): y=-3.5
    (-1.5, -3.5),
    (-0.5, -3.5),
    (0.5, -3.5),
    (1.5, -3.5),
];

/// Pattern margin: minimum distance from image border (ceil(3.5) = 4)
const PATTERN_MARGIN: i32 = 4;

/// Configuration for Lucas-Kanade tracker
#[derive(Debug, Clone)]
pub struct LKConfig {
    /// Maximum number of iterations per pyramid level
    pub max_iterations: usize,
    /// Convergence threshold (stop if motion < this)
    pub epsilon: f32,
    /// Number of pyramid levels (1 = no pyramid)
    pub num_levels: usize,
    /// Minimum eigenvalue threshold for valid tracking
    pub min_eigenvalue: f32,
    /// Forward-backward error threshold (pixels). If tracking A->B->A' gives
    /// |A - A'| > this threshold, reject the track. Set to 0 to disable.
    pub forward_backward_threshold: f32,
}

impl Default for LKConfig {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            epsilon: 0.01,
            num_levels: 3,
            min_eigenvalue: 0.001,
            forward_backward_threshold: 1.0, // Reject if round-trip error > 1 pixel
        }
    }
}

/// Lucas-Kanade optical flow tracker
pub struct LKTracker {
    config: LKConfig,
}

/// Result of tracking a single point
#[derive(Debug, Clone, Copy)]
pub struct TrackResult {
    /// New position after tracking
    pub position: (f32, f32),
    /// Whether tracking was successful
    pub success: bool,
    /// Tracking error (sum of squared differences)
    pub error: f32,
}

impl LKTracker {
    /// Create a new LK tracker with default configuration
    pub fn new() -> Self {
        Self {
            config: LKConfig::default(),
        }
    }

    /// Create a new LK tracker with custom configuration
    pub fn with_config(config: LKConfig) -> Self {
        Self { config }
    }

    /// Track a set of points from prev_image to next_image
    /// If forward_backward_threshold > 0, performs forward-backward consistency check
    pub fn track(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        points: &[(f32, f32)],
    ) -> Vec<TrackResult> {
        self.track_with_guess(prev_image, next_image, points, None)
    }

    /// Track a set of points with optional initial guesses for their positions in next_image
    /// 
    /// # Arguments
    /// * `prev_image` - Source image
    /// * `next_image` - Destination image  
    /// * `points` - Points to track in prev_image
    /// * `init_guesses` - Optional initial guesses for where each point is in next_image
    ///                    If None, uses the same position as in prev_image
    pub fn track_with_guess(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        points: &[(f32, f32)],
        init_guesses: Option<&[(f32, f32)]>,
    ) -> Vec<TrackResult> {
        // Build image pyramids
        let prev_pyramid = self.build_pyramid(prev_image);
        let next_pyramid = self.build_pyramid(next_image);

        // Forward tracking: prev -> next
        let forward_results: Vec<TrackResult> = points
            .iter()
            .enumerate()
            .map(|(i, &pt)| {
                let init_guess = init_guesses.map(|guesses| guesses[i]);
                self.track_point(&prev_pyramid, &next_pyramid, pt, init_guess)
            })
            .collect();

        // If forward-backward check is disabled, return forward results
        if self.config.forward_backward_threshold <= 0.0 {
            return forward_results;
        }

        // Backward tracking: next -> prev (only for successful forward tracks)
        let backward_points: Vec<(f32, f32)> = forward_results
            .iter()
            .map(|r| r.position)
            .collect();

        let backward_results: Vec<TrackResult> = backward_points
            .iter()
            .map(|&pt| self.track_point(&next_pyramid, &prev_pyramid, pt, None))
            .collect();

        // Check forward-backward consistency
        let fb_threshold_sq = self.config.forward_backward_threshold * self.config.forward_backward_threshold;

        points
            .iter()
            .zip(forward_results.iter())
            .zip(backward_results.iter())
            .map(|((&orig_pt, fwd), bwd)| {
                if !fwd.success || !bwd.success {
                    return TrackResult {
                        position: orig_pt,
                        success: false,
                        error: f32::MAX,
                    };
                }

                // Compute round-trip error: |original - backward_result|
                let dx = orig_pt.0 - bwd.position.0;
                let dy = orig_pt.1 - bwd.position.1;
                let fb_error_sq = dx * dx + dy * dy;

                if fb_error_sq > fb_threshold_sq {
                    // Forward-backward check failed
                    TrackResult {
                        position: orig_pt,
                        success: false,
                        error: fb_error_sq.sqrt(),
                    }
                } else {
                    // Passed! Return forward result
                    *fwd
                }
            })
            .collect()
    }

    /// Track keypoints, returning new keypoints with updated positions
    pub fn track_keypoints(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        keypoints: &[KeyPoint],
    ) -> Vec<(KeyPoint, bool)> {
        let points: Vec<(f32, f32)> = keypoints.iter().map(|kp| (kp.x, kp.y)).collect();
        let results = self.track(prev_image, next_image, &points);

        keypoints
            .iter()
            .zip(results.iter())
            .map(|(kp, result)| {
                let mut new_kp = *kp;
                new_kp.x = result.position.0;
                new_kp.y = result.position.1;
                (new_kp, result.success)
            })
            .collect()
    }

    /// Build image pyramid
    fn build_pyramid(&self, image: &GrayImage) -> Vec<GrayImage> {
        let mut pyramid = Vec::with_capacity(self.config.num_levels);
        pyramid.push(image.clone());

        for level in 1..self.config.num_levels {
            let prev = &pyramid[level - 1];
            let downsampled = self.downsample(prev);
            pyramid.push(downsampled);
        }

        pyramid
    }

    /// Downsample image by 2x with Gaussian blur
    fn downsample(&self, image: &GrayImage) -> GrayImage {
        let (width, height) = image.dimensions();
        let new_width = width / 2;
        let new_height = height / 2;

        if new_width == 0 || new_height == 0 {
            return image.clone();
        }

        let mut result = GrayImage::new(new_width, new_height);

        // Simple 2x2 box filter for downsampling
        for y in 0..new_height {
            for x in 0..new_width {
                let sx = x * 2;
                let sy = y * 2;

                // Average 2x2 block
                let p00 = image.get_pixel(sx, sy).0[0] as u32;
                let p10 = image.get_pixel((sx + 1).min(width - 1), sy).0[0] as u32;
                let p01 = image.get_pixel(sx, (sy + 1).min(height - 1)).0[0] as u32;
                let p11 = image
                    .get_pixel((sx + 1).min(width - 1), (sy + 1).min(height - 1))
                    .0[0] as u32;

                let avg = ((p00 + p10 + p01 + p11 + 2) / 4) as u8;
                result.put_pixel(x, y, image::Luma([avg]));
            }
        }

        result
    }

    /// Track a single point through the pyramid
    /// 
    /// # Arguments
    /// * `prev_pyramid` - Source image pyramid
    /// * `next_pyramid` - Destination image pyramid
    /// * `point` - Point position in source image
    /// * `init_guess` - Optional initial guess for position in destination image
    fn track_point(
        &self,
        prev_pyramid: &[GrayImage],
        next_pyramid: &[GrayImage],
        point: (f32, f32),
        init_guess: Option<(f32, f32)>,
    ) -> TrackResult {
        let num_levels = prev_pyramid.len();

        // Scale point to coarsest level
        let scale = (1 << (num_levels - 1)) as f32;
        
        // Use initial guess if provided, otherwise start at same position
        let init_pos = init_guess.unwrap_or(point);
        let mut guess = (init_pos.0 / scale, init_pos.1 / scale);
        let mut flow = (0.0f32, 0.0f32);

        // Coarse to fine
        for level in (0..num_levels).rev() {
            let level_scale = (1 << level) as f32;
            let prev_pt = (point.0 / level_scale, point.1 / level_scale);
            
            // Current guess at this level (incorporating flow from coarser levels)
            let current_guess = (guess.0 + flow.0, guess.1 + flow.1);

            // Refine flow at this level
            let result = self.track_at_level(
                &prev_pyramid[level],
                &next_pyramid[level],
                prev_pt,
                current_guess,
            );

            if !result.success {
                return TrackResult {
                    position: point,
                    success: false,
                    error: f32::MAX,
                };
            }

            // Update flow
            flow.0 = result.position.0 - prev_pt.0;
            flow.1 = result.position.1 - prev_pt.1;

            // Scale flow for next (finer) level
            if level > 0 {
                flow.0 *= 2.0;
                flow.1 *= 2.0;
                guess = (prev_pt.0 * 2.0, prev_pt.1 * 2.0);
            }
        }

        // Final position
        let final_pos = (point.0 + flow.0, point.1 + flow.1);

        // Bounds check
        let (width, height) = prev_pyramid[0].dimensions();
        if final_pos.0 < 0.0
            || final_pos.1 < 0.0
            || final_pos.0 >= width as f32
            || final_pos.1 >= height as f32
        {
            return TrackResult {
                position: point,
                success: false,
                error: f32::MAX,
            };
        }

        TrackResult {
            position: final_pos,
            success: true,
            error: 0.0, // Could compute actual error here
        }
    }

    /// Track at a single pyramid level using iterative Lucas-Kanade
    fn track_at_level(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        prev_pt: (f32, f32),
        init_guess: (f32, f32),
    ) -> TrackResult {
        let (width, height) = prev_image.dimensions();

        // Check bounds - pattern extends PATTERN_MARGIN pixels from center
        if prev_pt.0 < PATTERN_MARGIN as f32
            || prev_pt.1 < PATTERN_MARGIN as f32
            || prev_pt.0 >= (width as i32 - PATTERN_MARGIN) as f32
            || prev_pt.1 >= (height as i32 - PATTERN_MARGIN) as f32
        {
            return TrackResult {
                position: prev_pt,
                success: false,
                error: f32::MAX,
            };
        }

        // Compute image gradients and structure tensor using pattern sampling
        let (gxx, gyy, gxy, grad_x, grad_y) =
            self.compute_gradient_matrix_pattern(prev_image, prev_pt.0, prev_pt.1);

        // Check minimum eigenvalue
        let trace = gxx + gyy;
        let det = gxx * gyy - gxy * gxy;
        let discriminant = (trace * trace - 4.0 * det).max(0.0);
        let min_eig = (trace - discriminant.sqrt()) / 2.0;

        if min_eig < self.config.min_eigenvalue {
            return TrackResult {
                position: prev_pt,
                success: false,
                error: f32::MAX,
            };
        }

        // Iterative refinement
        let mut cur_pos = init_guess;

        for _iter in 0..self.config.max_iterations {
            // Check bounds for current guess
            if cur_pos.0 < PATTERN_MARGIN as f32
                || cur_pos.1 < PATTERN_MARGIN as f32
                || cur_pos.0 >= (width as i32 - PATTERN_MARGIN) as f32
                || cur_pos.1 >= (height as i32 - PATTERN_MARGIN) as f32
            {
                return TrackResult {
                    position: prev_pt,
                    success: false,
                    error: f32::MAX,
                };
            }

            // Compute image difference using pattern sampling
            let (bx, by) = self.compute_mismatch_pattern(
                prev_image,
                next_image,
                prev_pt,
                cur_pos,
                &grad_x,
                &grad_y,
            );

            // Solve 2x2 system: [gxx gxy; gxy gyy] * [dx; dy] = [bx; by]
            let det = gxx * gyy - gxy * gxy;
            if det.abs() < 1e-10 {
                return TrackResult {
                    position: prev_pt,
                    success: false,
                    error: f32::MAX,
                };
            }

            let dx = (gyy * bx - gxy * by) / det;
            let dy = (gxx * by - gxy * bx) / det;

            cur_pos.0 += dx;
            cur_pos.1 += dy;

            // Check convergence
            if dx * dx + dy * dy < self.config.epsilon * self.config.epsilon {
                break;
            }
        }

        TrackResult {
            position: cur_pos,
            success: true,
            error: 0.0,
        }
    }

    /// Compute gradient matrix (structure tensor) at a point using pattern sampling
    fn compute_gradient_matrix_pattern(
        &self,
        image: &GrayImage,
        cx: f32,
        cy: f32,
    ) -> (f32, f32, f32, Vec<f32>, Vec<f32>) {
        let mut gxx = 0.0f32;
        let mut gyy = 0.0f32;
        let mut gxy = 0.0f32;
        let mut grad_x = Vec::with_capacity(PATTERN_POINTS.len());
        let mut grad_y = Vec::with_capacity(PATTERN_POINTS.len());

        for &(px, py) in &PATTERN_POINTS {
            let sample_x = cx + px;
            let sample_y = cy + py;
            // Sample gradient at pattern point
            let (ix, iy) = self.sample_gradient(image, sample_x, sample_y);
            grad_x.push(ix);
            grad_y.push(iy);
            gxx += ix * ix;
            gyy += iy * iy;
            gxy += ix * iy;
        }

        (gxx, gyy, gxy, grad_x, grad_y)
    }

    /// Sample gradient at subpixel location using central differences
    /// This computes the gradient by sampling the image at offset locations
    fn sample_gradient(&self, image: &GrayImage, x: f32, y: f32) -> (f32, f32) {
        // Use bilinear interpolation to sample gradient
        // For gradient in x: sample at (x+1, y) and (x-1, y)
        // For gradient in y: sample at (x, y+1) and (x, y-1)
        // Note: sample_bilinear handles bounds clamping internally
        let ix_plus = self.sample_bilinear(image, x + 1.0, y);
        let ix_minus = self.sample_bilinear(image, x - 1.0, y);
        let iy_plus = self.sample_bilinear(image, x, y + 1.0);
        let iy_minus = self.sample_bilinear(image, x, y - 1.0);

        let ix = (ix_plus - ix_minus) / 2.0;
        let iy = (iy_plus - iy_minus) / 2.0;

        (ix, iy)
    }

    /// Compute mismatch vector (temporal gradient weighted by spatial gradient) using pattern sampling
    fn compute_mismatch_pattern(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        prev_pt: (f32, f32),
        cur_pt: (f32, f32),
        grad_x: &[f32],
        grad_y: &[f32],
    ) -> (f32, f32) {
        let mut bx = 0.0f32;
        let mut by = 0.0f32;

        for (i, &(px, py)) in PATTERN_POINTS.iter().enumerate() {
            // Sample from prev image at pattern offset
            let prev_val = self.sample_bilinear(prev_image, prev_pt.0 + px, prev_pt.1 + py);

            // Sample from next image at pattern offset relative to current position
            let next_val = self.sample_bilinear(next_image, cur_pt.0 + px, cur_pt.1 + py);

            let dt = prev_val - next_val;
            bx += grad_x[i] * dt;
            by += grad_y[i] * dt;
        }

        (bx, by)
    }

    /// Bilinear interpolation for subpixel sampling
    fn sample_bilinear(&self, image: &GrayImage, x: f32, y: f32) -> f32 {
        let (width, height) = image.dimensions();

        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        // Clamp to image bounds
        let x0 = x0.max(0).min(width as i32 - 1) as u32;
        let y0 = y0.max(0).min(height as i32 - 1) as u32;
        let x1 = x1.max(0).min(width as i32 - 1) as u32;
        let y1 = y1.max(0).min(height as i32 - 1) as u32;

        let fx = x - x.floor();
        let fy = y - y.floor();

        let p00 = image.get_pixel(x0, y0).0[0] as f32;
        let p10 = image.get_pixel(x1, y0).0[0] as f32;
        let p01 = image.get_pixel(x0, y1).0[0] as f32;
        let p11 = image.get_pixel(x1, y1).0[0] as f32;

        // Bilinear interpolation
        let top = p00 * (1.0 - fx) + p10 * fx;
        let bottom = p01 * (1.0 - fx) + p11 * fx;

        top * (1.0 - fy) + bottom * fy
    }
}

impl Default for LKTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lk_tracker_stationary() {
        let tracker = LKTracker::new();

        // Create a test image with a corner that has gradient
        // The corner at (50, 50) will have texture for tracking
        let mut image = GrayImage::from_pixel(100, 100, image::Luma([128]));
        // Add a bright square - track point near the corner where there's gradient
        for y in 50..80 {
            for x in 50..80 {
                image.put_pixel(x, y, image::Luma([200]));
            }
        }

        // Track a point near the corner (50, 50) where the pattern will see gradient
        // Pattern52 extends ±3.5 pixels, so point at (50, 50) samples (46.5-53.5)
        // which crosses the edge at x=50 and y=50
        let points = vec![(50.0, 50.0)];
        let results = tracker.track(&image, &image, &points);

        assert!(results[0].success, "Tracking failed - point at ({}, {})", results[0].position.0, results[0].position.1);
        assert!((results[0].position.0 - 50.0).abs() < 1.0);
        assert!((results[0].position.1 - 50.0).abs() < 1.0);
    }
}
