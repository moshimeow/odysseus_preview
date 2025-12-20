//! Example: Track features across a stereo image sequence
//!
//! Usage:
//!   cargo run --example track_sequence -- <image_dir> [output_dir]
//!
//! Expects images named like: 0001_L.jpg, 0001_R.jpg, 0002_L.jpg, etc.
//! If no output dir is given, saves to "tracking_output/"

use image::{GrayImage, Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_circle_mut, draw_line_segment_mut};
use odysseus_slam_frontend::{TrackedFeature, Tracker, TrackerConfig};
use std::collections::HashMap;
use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <image_dir> [output_dir]", args[0]);
        eprintln!("\nThis example tracks features across a stereo image sequence.");
        eprintln!("Images should be named: 0001_L.jpg, 0001_R.jpg, 0002_L.jpg, etc.");
        std::process::exit(1);
    }

    let image_dir = Path::new(&args[1]);
    let output_dir = args
        .get(2)
        .map(|s| Path::new(s).to_path_buf())
        .unwrap_or_else(|| Path::new("tracking_output").to_path_buf());

    // Create output directory
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Find all stereo pairs
    let mut frame_numbers: Vec<u32> = Vec::new();
    for entry in std::fs::read_dir(image_dir).expect("Failed to read image directory") {
        let entry = entry.expect("Failed to read directory entry");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with("_L.jpg") || name.ends_with("_L.png") {
            if let Ok(num) = name[..4].parse::<u32>() {
                frame_numbers.push(num);
            }
        }
    }
    frame_numbers.sort();

    if frame_numbers.is_empty() {
        eprintln!("No stereo pairs found in {}", image_dir.display());
        std::process::exit(1);
    }

    println!("Found {} stereo pairs", frame_numbers.len());

    // Create tracker
    let config = TrackerConfig {
        min_features: 150,
        max_features: 400,
        ..Default::default()
    };
    let mut tracker = Tracker::with_config(config);

    // Track feature history for visualization (id -> list of positions)
    let mut track_history: HashMap<usize, Vec<(f32, f32)>> = HashMap::new();

    // Process each frame
    let total_frames = frame_numbers.len().min(239); // Limit for demo
    for (i, &frame_num) in frame_numbers.iter().take(total_frames).enumerate() {
        let left_path = image_dir.join(format!("{:04}_L.jpg", frame_num));
        let right_path = image_dir.join(format!("{:04}_R.jpg", frame_num));

        // Try .png if .jpg doesn't exist
        let left_path = if left_path.exists() {
            left_path
        } else {
            image_dir.join(format!("{:04}_L.png", frame_num))
        };
        let right_path = if right_path.exists() {
            right_path
        } else {
            image_dir.join(format!("{:04}_R.png", frame_num))
        };

        println!(
            "\nFrame {}/{}: {}",
            i + 1,
            total_frames,
            left_path.display()
        );

        // Load images
        let left_img = image::open(&left_path).expect("Failed to load left image");
        let right_img = image::open(&right_path).expect("Failed to load right image");

        let left_gray: GrayImage = left_img.to_luma8();
        let right_gray: GrayImage = right_img.to_luma8();

        // Process frame
        let start = std::time::Instant::now();
        let tracks = tracker.process_frame(&left_gray, &right_gray);
        let process_time = start.elapsed();

        // Update track history
        for track in &tracks {
            track_history
                .entry(track.id)
                .or_default()
                .push((track.stereo.left_kp.x, track.stereo.left_kp.y));
        }

        // Print statistics
        let long_tracks = tracks.iter().filter(|t| t.age == 0).count();
        println!(
            "  Tracks: {} total, {} with stereo match, processed in {:?}",
            tracks.len(),
            long_tracks,
            process_time
        );

        // Visualize
        let output = visualize_tracks(&left_img.to_rgb8(), &tracks, &track_history);
        let output_path = output_dir.join(format!("frame_{:04}.png", frame_num));
        output.save(&output_path).expect("Failed to save output");
    }

    // Print final statistics
    println!("\n=== Tracking Summary ===");
    println!("Total frames processed: {}", total_frames);

    let track_lengths: Vec<usize> = track_history.values().map(|v| v.len()).collect();
    if !track_lengths.is_empty() {
        let max_len = *track_lengths.iter().max().unwrap();
        let avg_len: f32 = track_lengths.iter().sum::<usize>() as f32 / track_lengths.len() as f32;
        let long_tracks = track_lengths.iter().filter(|&&l| l >= 5).count();

        println!("Total unique tracks: {}", track_history.len());
        println!("Max track length: {} frames", max_len);
        println!("Avg track length: {:.1} frames", avg_len);
        println!("Tracks >= 5 frames: {}", long_tracks);
    }

    println!("\nOutput saved to: {}", output_dir.display());
}

/// Visualize tracked features with their trajectories
fn visualize_tracks(
    image: &RgbImage,
    current_tracks: &[TrackedFeature],
    history: &HashMap<usize, Vec<(f32, f32)>>,
) -> RgbImage {
    let mut output = image.clone();

    // Draw track trajectories
    for track in current_tracks {
        if let Some(positions) = history.get(&track.id) {
            if positions.len() >= 2 {
                // Color based on track age (more frames = more green)
                let track_len = positions.len();
                let color = track_color(track_len);

                // Draw trajectory
                for i in 1..positions.len() {
                    let (x0, y0) = positions[i - 1];
                    let (x1, y1) = positions[i];
                    draw_line_segment_mut(&mut output, (x0, y0), (x1, y1), color);
                }
            }
        }

        // Draw current position
        let x = track.stereo.left_kp.x as i32;
        let y = track.stereo.left_kp.y as i32;

        // Color circle by disparity
        let disp_color = disparity_color(track.stereo.disparity);
        draw_hollow_circle_mut(&mut output, (x, y), 5, disp_color);

        // Smaller inner circle if track has good stereo match
        if track.age == 0 {
            draw_hollow_circle_mut(&mut output, (x, y), 2, Rgb([0, 255, 0]));
        }
    }

    output
}

/// Color based on track length (short = red, long = green)
fn track_color(length: usize) -> Rgb<u8> {
    let norm = (length as f32 / 20.0).min(1.0);
    let r = ((1.0 - norm) * 255.0) as u8;
    let g = (norm * 255.0) as u8;
    Rgb([r, g, 100])
}

/// Color based on disparity (low/far = blue, high/close = red)
fn disparity_color(disparity: f32) -> Rgb<u8> {
    let norm = ((disparity - 10.0) / 150.0).clamp(0.0, 1.0);
    let r = (norm * 255.0) as u8;
    let b = ((1.0 - norm) * 255.0) as u8;
    let g = ((1.0 - (2.0 * norm - 1.0).abs()) * 128.0) as u8;
    Rgb([r, g, b])
}
