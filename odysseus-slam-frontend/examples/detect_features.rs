//! Example: Detect ORB-like features on a single image
//!
//! Usage:
//!   cargo run --example detect_features -- <input_image> [output_image]
//!
//! If no output path is given, saves to "output_features.png"

use image::{GrayImage, Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_circle_mut, draw_line_segment_mut};
use odysseus_slam_frontend::{BriefExtractor, FastDetector, KeyPoint};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input_image> [output_image]", args[0]);
        eprintln!("\nThis example detects ORB-like features (FAST corners + BRIEF descriptors)");
        eprintln!("and saves an annotated image showing the detected keypoints.");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = args.get(2).map(|s| s.as_str()).unwrap_or("output_features.png");

    println!("Loading image: {}", input_path);

    // Load and convert to grayscale
    let img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            std::process::exit(1);
        }
    };

    let gray: GrayImage = img.to_luma8();
    let (width, height) = gray.dimensions();
    println!("Image size: {}x{}", width, height);

    // Detect FAST corners
    println!("\nDetecting FAST corners...");
    let detector = FastDetector::default();
    let start = std::time::Instant::now();
    let keypoints = detector.detect(&gray);
    let detection_time = start.elapsed();
    println!("  Found {} keypoints in {:?}", keypoints.len(), detection_time);

    // Compute BRIEF descriptors
    println!("\nComputing BRIEF descriptors...");
    let extractor = BriefExtractor::new();
    let start = std::time::Instant::now();
    let features = extractor.compute_all(&gray, &keypoints);
    let descriptor_time = start.elapsed();
    println!(
        "  Computed {} descriptors in {:?}",
        features.len(),
        descriptor_time
    );
    println!(
        "  ({} keypoints too close to border)",
        keypoints.len() - features.len()
    );

    // Print some statistics
    if !features.is_empty() {
        let avg_response: f32 =
            features.iter().map(|(kp, _)| kp.response).sum::<f32>() / features.len() as f32;
        let max_response = features
            .iter()
            .map(|(kp, _)| kp.response)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_response = features
            .iter()
            .map(|(kp, _)| kp.response)
            .fold(f32::INFINITY, f32::min);

        println!("\nKeypoint statistics:");
        println!("  Response: min={:.1}, max={:.1}, avg={:.1}", min_response, max_response, avg_response);
    }

    // Create output visualization
    println!("\nSaving visualization to: {}", output_path);
    println!("  Converting to RGB...");
    let rgb = img.to_rgb8();
    println!("  Drawing keypoints...");
    let output = visualize_keypoints(&rgb, &features);
    println!("  Writing file...");
    output.save(output_path).expect("Failed to save output image");

    println!("\nDone!");
}

/// Draw keypoints on the image with orientation indicators
fn visualize_keypoints(image: &RgbImage, features: &[(KeyPoint, odysseus_slam_frontend::BriefDescriptor)]) -> RgbImage {
    let mut output = image.clone();

    for (kp, _desc) in features {
        let x = kp.x as i32;
        let y = kp.y as i32;

        // Choose color based on response (green = strong, yellow = weak)
        let color = if kp.response > 1000.0 {
            Rgb([0, 255, 0]) // Strong corner: green
        } else if kp.response > 100.0 {
            Rgb([255, 255, 0]) // Medium: yellow
        } else {
            Rgb([255, 128, 0]) // Weak: orange
        };

        // Draw circle at keypoint location
        draw_hollow_circle_mut(&mut output, (x, y), 5, color);

        // Draw orientation line from center outward
        let line_len = 12.0;
        let ex = kp.x + line_len * kp.angle.cos();
        let ey = kp.y + line_len * kp.angle.sin();

        draw_line_segment_mut(
            &mut output,
            (kp.x, kp.y),
            (ex, ey),
            Rgb([255, 0, 0]),
        );
    }

    output
}
