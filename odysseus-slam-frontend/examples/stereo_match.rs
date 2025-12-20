//! Example: Stereo matching between left and right images
//!
//! Usage:
//!   cargo run --example stereo_match -- <left_image> <right_image> [output_image]
//!
//! If no output path is given, saves to "output_stereo.png"

use image::{GrayImage, Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_circle_mut, draw_line_segment_mut};
use odysseus_slam_frontend::{StereoMatch, StereoMatcher};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <left_image> <right_image> [output_image]", args[0]);
        eprintln!("\nThis example performs stereo matching between a left and right image pair");
        eprintln!("and visualizes the matched features with disparity coloring.");
        std::process::exit(1);
    }

    let left_path = &args[1];
    let right_path = &args[2];
    let output_path = args.get(3).map(|s| s.as_str()).unwrap_or("output_stereo.png");

    println!("Loading left image: {}", left_path);
    println!("Loading right image: {}", right_path);

    // Load images
    let left_img = image::open(left_path).expect("Failed to load left image");
    let right_img = image::open(right_path).expect("Failed to load right image");

    let left_gray: GrayImage = left_img.to_luma8();
    let right_gray: GrayImage = right_img.to_luma8();

    let (width, height) = left_gray.dimensions();
    println!("Image size: {}x{}", width, height);

    // Perform stereo matching
    println!("\nPerforming stereo matching...");
    let matcher = StereoMatcher::default();

    let start = std::time::Instant::now();
    let matches = matcher.match_stereo(&left_gray, &right_gray);
    let match_time = start.elapsed();

    println!("  Found {} stereo matches in {:?}", matches.len(), match_time);

    // Print disparity statistics
    if !matches.is_empty() {
        let disparities: Vec<f32> = matches.iter().map(|m| m.disparity).collect();
        let min_disp = disparities.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_disp = disparities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let avg_disp: f32 = disparities.iter().sum::<f32>() / disparities.len() as f32;

        println!("\nDisparity statistics:");
        println!("  Min: {:.1} px", min_disp);
        println!("  Max: {:.1} px", max_disp);
        println!("  Avg: {:.1} px", avg_disp);
    }

    // Create side-by-side visualization
    println!("\nSaving visualization to: {}", output_path);
    let output = visualize_stereo_matches(&left_img.to_rgb8(), &right_img.to_rgb8(), &matches);
    output.save(output_path).expect("Failed to save output image");

    println!("\nDone!");
}

/// Create a side-by-side visualization of stereo matches
fn visualize_stereo_matches(
    left: &RgbImage,
    right: &RgbImage,
    matches: &[StereoMatch],
) -> RgbImage {
    let (width, height) = left.dimensions();

    // Create side-by-side image
    let mut output = RgbImage::new(width * 2, height);

    // Copy left image
    for y in 0..height {
        for x in 0..width {
            output.put_pixel(x, y, *left.get_pixel(x, y));
        }
    }

    // Copy right image (offset by width)
    for y in 0..height {
        for x in 0..width {
            output.put_pixel(x + width, y, *right.get_pixel(x, y));
        }
    }

    // Find disparity range for color mapping
    let disparities: Vec<f32> = matches.iter().map(|m| m.disparity).collect();
    let min_disp = disparities.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_disp = disparities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let disp_range = (max_disp - min_disp).max(1.0);

    // Draw matches
    for m in matches {
        // Color by disparity (blue = far/low disparity, red = close/high disparity)
        let norm_disp = (m.disparity - min_disp) / disp_range;
        let color = disparity_color(norm_disp);

        // Draw circle on left keypoint
        let lx = m.left_kp.x as i32;
        let ly = m.left_kp.y as i32;
        draw_hollow_circle_mut(&mut output, (lx, ly), 4, color);

        // Draw circle on right keypoint (offset by width)
        let rx = m.right_kp.x as i32 + width as i32;
        let ry = m.right_kp.y as i32;
        draw_hollow_circle_mut(&mut output, (rx, ry), 4, color);

        // Draw connecting line
        draw_line_segment_mut(
            &mut output,
            (m.left_kp.x, m.left_kp.y),
            (m.right_kp.x + width as f32, m.right_kp.y),
            color,
        );
    }

    output
}

/// Map normalized disparity [0, 1] to color (blue = far, red = close)
fn disparity_color(norm_disp: f32) -> Rgb<u8> {
    // Simple blue-to-red gradient
    let r = (norm_disp * 255.0) as u8;
    let b = ((1.0 - norm_disp) * 255.0) as u8;
    let g = ((1.0 - (2.0 * norm_disp - 1.0).abs()) * 255.0) as u8; // Peak green in middle
    Rgb([r, g, b])
}
