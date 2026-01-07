//! Lucas-Kanade optical flow tracker
//!
//! Implements pyramidal Lucas-Kanade for tracking features across frames.
//! Uses Schnilbert solver with Jets for automatic differentiation.

use image::GrayImage;
use odysseus_solver::schnilbert::Schnilbert;

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
    /// Forward-backward error threshold (pixels). If tracking A->B->A' gives
    /// |A - A'| > this threshold, reject the track. Set to 0 to disable.
    pub forward_backward_threshold: f32,
}

impl Default for LKConfig {
    fn default() -> Self {
        Self {
            win_size: 11,
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
    solver: Schnilbert<f32, 2>,
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
        Self::with_config(LKConfig::default())
    }

    /// Create a new LK tracker with custom configuration
    pub fn with_config(config: LKConfig) -> Self {
        let solver = Schnilbert::<f32, 2>::new()
            .with_max_iterations(config.max_iterations)
            .with_step_threshold(config.epsilon);

        Self { config, solver }
    }

    /// Track a set of points from prev_image to next_image
    /// If forward_backward_threshold > 0, performs forward-backward consistency check
    pub fn track(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        points: &[(f32, f32)],
    ) -> Vec<TrackResult> {
        // Build image pyramids
        let prev_pyramid = self.build_pyramid(prev_image);
        let next_pyramid = self.build_pyramid(next_image);

        // Forward tracking: prev -> next
        let forward_results: Vec<TrackResult> = points
            .iter()
            .map(|&pt| self.track_point(&prev_pyramid, &next_pyramid, pt))
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
            .map(|&pt| self.track_point(&next_pyramid, &prev_pyramid, pt))
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

    /// Track at a single pyramid level using Schnilbert solver
    ///
    /// The LK tracking problem is: find (dx, dy) that minimizes
    ///   sum_window (I(x,y) - J(x+dx, y+dy))^2
    ///
    /// This is a linearized problem where:
    /// - Residual r_i = I(x_i, y_i) - J(x_i + dx, y_i + dy)
    /// - Jacobian row = [-dJ/dx, -dJ/dy] ≈ [-dI/dx, -dI/dy] (using template gradients)
    ///
    /// We use solve_linearized since the Jacobian (image gradients) is constant.
    fn track_at_level(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        prev_pt: (f32, f32),
        init_guess: (f32, f32),
    ) -> TrackResult {
        let (width, height) = prev_image.dimensions();
        let win = self.config.win_size as i32;

        // Check bounds for template extraction
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

        // Compute image gradients (this is the Jacobian)
        // For residual r = I_prev - I_next, dr/d(dx) = -dI_next/dx ≈ -dI_prev/dx
        let (jacobian, min_eig) = self.compute_jacobian(prev_image, prev_pt);

        // Check minimum eigenvalue (trackability)
        if min_eig < self.config.min_eigenvalue {
            return TrackResult {
                position: prev_pt,
                success: false,
                error: f32::MAX,
            };
        }

        // Initial offset from prev_pt
        let initial_offset = [init_guess.0 - prev_pt.0, init_guess.1 - prev_pt.1];

        // Create residual function that computes I_prev(x,y) - I_next(x+dx, y+dy)
        let prev_pt_copy = prev_pt;
        let residual_fn = |params: &[f32; 2]| -> Vec<f32> {
            self.compute_residuals(prev_image, next_image, prev_pt_copy, params, win)
        };

        // Solve using Schnilbert's linearized solver
        let result = self.solver.solve_linearized(initial_offset, residual_fn, &jacobian);

        // Check if we went out of bounds during iteration
        let final_pos = (prev_pt.0 + result.params[0], prev_pt.1 + result.params[1]);

        if final_pos.0 < win as f32
            || final_pos.1 < win as f32
            || final_pos.0 >= (width as i32 - win) as f32
            || final_pos.1 >= (height as i32 - win) as f32
        {
            return TrackResult {
                position: prev_pt,
                success: false,
                error: f32::MAX,
            };
        }

        TrackResult {
            position: final_pos,
            success: result.converged,
            error: result.residual_norm_sq,
        }
    }

    /// Compute Jacobian (image gradients) for all pixels in the window
    /// Returns (jacobian_rows, min_eigenvalue)
    ///
    /// Each row of the Jacobian is [-dI/dx, -dI/dy] for one pixel.
    /// The min eigenvalue of J^T*J tells us if the point is trackable.
    fn compute_jacobian(&self, image: &GrayImage, center: (f32, f32)) -> (Vec<[f32; 2]>, f32) {
        let win = self.config.win_size as i32;
        let cx = center.0 as i32;
        let cy = center.1 as i32;
        let (width, height) = image.dimensions();

        let win_pixels = ((2 * win + 1) * (2 * win + 1)) as usize;
        let mut jacobian = Vec::with_capacity(win_pixels);

        // Also accumulate J^T*J for eigenvalue check
        let mut gxx = 0.0f32;
        let mut gyy = 0.0f32;
        let mut gxy = 0.0f32;

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

                // Jacobian row: negative gradient (since residual = prev - next)
                jacobian.push([-ix, -iy]);

                gxx += ix * ix;
                gyy += iy * iy;
                gxy += ix * iy;
            }
        }

        // Compute minimum eigenvalue of J^T*J = [[gxx, gxy], [gxy, gyy]]
        let trace = gxx + gyy;
        let det = gxx * gyy - gxy * gxy;
        let discriminant = (trace * trace - 4.0 * det).max(0.0);
        let min_eig = (trace - discriminant.sqrt()) / 2.0;

        (jacobian, min_eig)
    }

    /// Compute residuals: I_prev(x,y) - I_next(x+dx, y+dy) for all pixels in window
    fn compute_residuals(
        &self,
        prev_image: &GrayImage,
        next_image: &GrayImage,
        prev_pt: (f32, f32),
        offset: &[f32; 2],
        win: i32,
    ) -> Vec<f32> {
        let cur_pt = (prev_pt.0 + offset[0], prev_pt.1 + offset[1]);
        let win_pixels = ((2 * win + 1) * (2 * win + 1)) as usize;
        let mut residuals = Vec::with_capacity(win_pixels);

        for dy in -win..=win {
            for dx in -win..=win {
                // Sample from prev image at template location
                let prev_val = self.sample_bilinear(
                    prev_image,
                    prev_pt.0 + dx as f32,
                    prev_pt.1 + dy as f32,
                );

                // Sample from next image at current guess
                let next_val = self.sample_bilinear(
                    next_image,
                    cur_pt.0 + dx as f32,
                    cur_pt.1 + dy as f32,
                );

                residuals.push(prev_val - next_val);
            }
        }

        residuals
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
