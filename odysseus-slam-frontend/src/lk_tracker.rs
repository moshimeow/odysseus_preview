//! Lucas-Kanade optical flow tracker
//!
//! Implements pyramidal Lucas-Kanade for tracking features across frames.
//! Uses iterative refinement with a local window.

use image::GrayImage;

use crate::KeyPoint;

/// Configuration for Lucas-Kanade tracker
#[derive(Debug, Clone)]
pub struct LKConfig {
    /// Size of the tracking window (half-width, so window is 2*win_size+1)
    pub win_size: usize,
    /// Maximum number of iterations per pyramid level
    pub max_iterations: usize,
    /// Convergence threshold (stop if motion < this)
    pub epsilon: f32,
    /// Number of pyramid levels (1 = no pyramid)
    pub num_levels: usize,
    /// Minimum eigenvalue threshold for valid tracking
    pub min_eigenvalue: f32,
}

impl Default for LKConfig {
    fn default() -> Self {
        Self {
            win_size: 11,
            max_iterations: 30,
            epsilon: 0.01,
            num_levels: 3,
            min_eigenvalue: 0.001,
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
    pub fn track(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        points: &[(f32, f32)],
    ) -> Vec<TrackResult> {
        // Build image pyramids
        let prev_pyramid = self.build_pyramid(prev_image);
        let next_pyramid = self.build_pyramid(next_image);

        // Track each point
        points
            .iter()
            .map(|&pt| self.track_point(&prev_pyramid, &next_pyramid, pt))
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
    fn track_point(
        &self,
        prev_pyramid: &[GrayImage],
        next_pyramid: &[GrayImage],
        point: (f32, f32),
    ) -> TrackResult {
        let num_levels = prev_pyramid.len();

        // Scale point to coarsest level
        let scale = (1 << (num_levels - 1)) as f32;
        let mut guess = (point.0 / scale, point.1 / scale);
        let mut flow = (0.0f32, 0.0f32);

        // Coarse to fine
        for level in (0..num_levels).rev() {
            let level_scale = (1 << level) as f32;
            let prev_pt = (point.0 / level_scale, point.1 / level_scale);

            // Refine flow at this level
            let result = self.track_at_level(
                &prev_pyramid[level],
                &next_pyramid[level],
                prev_pt,
                (guess.0 + flow.0, guess.1 + flow.1),
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
        let win = self.config.win_size as i32;

        // Check bounds
        if prev_pt.0 < win as f32
            || prev_pt.1 < win as f32
            || prev_pt.0 >= (width as i32 - win) as f32
            || prev_pt.1 >= (height as i32 - win) as f32
        {
            return TrackResult {
                position: prev_pt,
                success: false,
                error: f32::MAX,
            };
        }

        // Compute image gradients and structure tensor in the window
        let (gxx, gyy, gxy, grad_x, grad_y) =
            self.compute_gradient_matrix(prev_image, prev_pt.0 as i32, prev_pt.1 as i32);

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
            if cur_pos.0 < win as f32
                || cur_pos.1 < win as f32
                || cur_pos.0 >= (width as i32 - win) as f32
                || cur_pos.1 >= (height as i32 - win) as f32
            {
                return TrackResult {
                    position: prev_pt,
                    success: false,
                    error: f32::MAX,
                };
            }

            // Compute image difference
            let (bx, by) = self.compute_mismatch(
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

    /// Compute gradient matrix (structure tensor) at a point
    fn compute_gradient_matrix(
        &self,
        image: &GrayImage,
        cx: i32,
        cy: i32,
    ) -> (f32, f32, f32, Vec<f32>, Vec<f32>) {
        let win = self.config.win_size as i32;
        let win_pixels = ((2 * win + 1) * (2 * win + 1)) as usize;

        let mut gxx = 0.0f32;
        let mut gyy = 0.0f32;
        let mut gxy = 0.0f32;
        let mut grad_x = Vec::with_capacity(win_pixels);
        let mut grad_y = Vec::with_capacity(win_pixels);

        let (width, height) = image.dimensions();

        for dy in -win..=win {
            for dx in -win..=win {
                let px = (cx + dx) as u32;
                let py = (cy + dy) as u32;

                // Central differences for gradient
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

                grad_x.push(ix);
                grad_y.push(iy);

                gxx += ix * ix;
                gyy += iy * iy;
                gxy += ix * iy;
            }
        }

        (gxx, gyy, gxy, grad_x, grad_y)
    }

    /// Compute mismatch vector (temporal gradient weighted by spatial gradient)
    fn compute_mismatch(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        prev_pt: (f32, f32),
        cur_pt: (f32, f32),
        grad_x: &[f32],
        grad_y: &[f32],
    ) -> (f32, f32) {
        let win = self.config.win_size as i32;
        let mut bx = 0.0f32;
        let mut by = 0.0f32;

        let mut idx = 0;
        for dy in -win..=win {
            for dx in -win..=win {
                // Sample from prev image at integer location
                let prev_val = self.sample_bilinear(
                    prev_image,
                    prev_pt.0 + dx as f32,
                    prev_pt.1 + dy as f32,
                );

                // Sample from next image at subpixel location
                let next_val =
                    self.sample_bilinear(next_image, cur_pt.0 + dx as f32, cur_pt.1 + dy as f32);

                let dt = prev_val - next_val;
                bx += grad_x[idx] * dt;
                by += grad_y[idx] * dt;
                idx += 1;
            }
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

        // Create a simple test image with a corner
        let mut image = GrayImage::from_pixel(100, 100, image::Luma([128]));
        // Add a bright square
        for y in 40..60 {
            for x in 40..60 {
                image.put_pixel(x, y, image::Luma([200]));
            }
        }

        // Track with identical images - should find same position
        let points = vec![(50.0, 50.0)];
        let results = tracker.track(&image, &image, &points);

        assert!(results[0].success);
        assert!((results[0].position.0 - 50.0).abs() < 1.0);
        assert!((results[0].position.1 - 50.0).abs() < 1.0);
    }
}
