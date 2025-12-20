//! Example: Track features across a stereo image sequence with Rerun visualization
//!
//! Usage:
//!   cargo run --release --example track_sequence_rerun -- <image_dir>
//!
//! Expects images named like: 0001_L.jpg, 0001_R.jpg, 0002_L.jpg, etc.

use odysseus_slam_frontend::{TrackedFeature, Tracker, TrackerConfig};
use rerun as rr;
use std::collections::HashMap;
use std::env;
use std::path::Path;

use image::{GrayImage, RgbImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <image_dir>", args[0]);
        eprintln!("\nThis example tracks features across a stereo image sequence");
        eprintln!("and visualizes using Rerun.");
        eprintln!("Images should be named: 0001_L.jpg, 0001_R.jpg, 0002_L.jpg, etc.");
        std::process::exit(1);
    }

    let image_dir = Path::new(&args[1]);

    // Initialize Rerun
    let rec = rr::RecordingStreamBuilder::new("odysseus_slam_frontend").spawn()?;

    println!("Rerun viewer spawned successfully");

    // Find all stereo pairs
    let mut frame_numbers: Vec<u32> = Vec::new();
    for entry in std::fs::read_dir(image_dir)? {
        let entry = entry?;
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
    let total_frames = frame_numbers.len();
    for (i, &frame_num) in frame_numbers.iter().enumerate() {
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
            "Frame {}/{}: {}",
            i + 1,
            total_frames,
            left_path.display()
        );

        // Load images
        let left_img = image::open(&left_path)?;
        let right_img = image::open(&right_path)?;

        let left_gray: GrayImage = left_img.to_luma8();
        let right_gray: GrayImage = right_img.to_luma8();
        let (width, height) = left_gray.dimensions();

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

        // Set timeline
        rec.set_time_sequence("frame", i as i64);

        // Create side-by-side composite image
        let left_rgb = left_img.to_rgb8();
        let right_rgb = right_img.to_rgb8();
        let composite = create_side_by_side(&left_rgb, &right_rgb);

        // Log the composite stereo image
        rec.log(
            "stereo_view",
            &rr::Image::from_rgb24(composite.into_raw(), [width * 2, height]),
        )?;

        // Log keypoints on left side of composite
        log_left_keypoints(&rec, "stereo_view/left_keypoints", &tracks, &track_history)?;

        // Log keypoints on right side of composite
        log_right_keypoints(&rec, "stereo_view/right_keypoints", &tracks, width)?;

        // Log stereo match lines (colored by inlier/outlier status)
        log_stereo_matches(&rec, "stereo_view/matches", &tracks, width)?;

        // Log IDs as floating text above points
        log_keypoint_labels(&rec, "stereo_view/labels", &tracks)?;

        // Log track trajectories on left image
        log_track_trajectories(&rec, "stereo_view/tracks", &tracks, &track_history)?;

        // Log disparity statistics
        log_disparity_info(&rec, "stats/disparity", &tracks)?;

        // Print statistics
        let with_stereo = tracks.iter().filter(|t| t.age == 0).count();
        println!(
            "  Tracks: {} total, {} with stereo, {:?}",
            tracks.len(),
            with_stereo,
            process_time
        );
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

    println!("\nRerun visualization active. Close the viewer to exit.");

    Ok(())
}

/// Create a side-by-side composite of left and right images
fn create_side_by_side(left: &RgbImage, right: &RgbImage) -> RgbImage {
    let (width, height) = left.dimensions();
    let mut composite = RgbImage::new(width * 2, height);

    // Copy left image
    for y in 0..height {
        for x in 0..width {
            composite.put_pixel(x, y, *left.get_pixel(x, y));
        }
    }

    // Copy right image (offset by width)
    for y in 0..height {
        for x in 0..width {
            composite.put_pixel(x + width, y, *right.get_pixel(x, y));
        }
    }

    composite
}

/// Log keypoints on left side with color based on track age
fn log_left_keypoints(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    history: &HashMap<usize, Vec<(f32, f32)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let positions: Vec<[f32; 2]> = tracks
        .iter()
        .map(|t| [t.stereo.left_kp.x, t.stereo.left_kp.y])
        .collect();

    // Color based on track length (longer = greener)
    let point_colors: Vec<[u8; 3]> = tracks
        .iter()
        .map(|t| {
            let track_len = history.get(&t.id).map(|h| h.len()).unwrap_or(1);
            track_age_color(track_len)
        })
        .collect();

    // Size based on whether stereo match is valid
    let radii: Vec<f32> = tracks
        .iter()
        .map(|t| if t.age == 0 { 6.0 } else { 4.0 })
        .collect();

    rec.log(
        path,
        &rr::Points2D::new(positions)
            .with_colors(point_colors)
            .with_radii(radii),
    )?;

    Ok(())
}

/// Log keypoints on right side of composite (offset by image width)
fn log_right_keypoints(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    image_width: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let valid_tracks: Vec<_> = tracks.iter().filter(|t| t.age == 0).collect();

    let positions: Vec<[f32; 2]> = valid_tracks
        .iter()
        .map(|t| [t.stereo.right_kp.x + image_width as f32, t.stereo.right_kp.y])
        .collect();

    // Green for valid stereo matches
    let num_points = positions.len();
    rec.log(
        path,
        &rr::Points2D::new(positions)
            .with_colors(vec![[100u8, 255, 100]; num_points])
            .with_radii(vec![5.0; num_points]),
    )?;

    Ok(())
}

/// Log stereo matches as lines between left and right images
/// Color: green for good matches (inliers), red for questionable ones (outliers)
fn log_stereo_matches(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    image_width: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut strips: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut line_colors: Vec<[u8; 3]> = Vec::new();

    for track in tracks.iter().filter(|t| t.age == 0) {
        let left_pt = [track.stereo.left_kp.x, track.stereo.left_kp.y];
        let right_pt = [
            track.stereo.right_kp.x + image_width as f32,
            track.stereo.right_kp.y,
        ];
        strips.push(vec![left_pt, right_pt]);

        // Determine if this is an inlier or outlier based on epipolar constraint
        // For rectified stereo, left and right y-coordinates should be nearly equal
        let vertical_diff = (track.stereo.left_kp.y - track.stereo.right_kp.y).abs();
        let disparity = track.stereo.disparity;

        // Inlier criteria:
        // - Small vertical difference (good epipolar alignment)
        // - Reasonable disparity range (not too small or too large)
        let is_inlier = vertical_diff < 1.5 && disparity > 5.0 && disparity < 150.0;

        if is_inlier {
            line_colors.push([50, 255, 50]); // Green for inliers
        } else {
            line_colors.push([255, 100, 100]); // Red for outliers
        }
    }

    if !strips.is_empty() {
        rec.log(
            path,
            &rr::LineStrips2D::new(strips)
                .with_colors(line_colors)
                .with_radii(vec![1.5]),
        )?;
    }

    Ok(())
}

/// Log keypoint IDs as floating text above points
fn log_keypoint_labels(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
) -> Result<(), Box<dyn std::error::Error>> {
    // Only show labels for tracks with valid stereo matches
    let valid_tracks: Vec<_> = tracks.iter().filter(|t| t.age == 0).collect();

    if valid_tracks.is_empty() {
        return Ok(());
    }

    // Position labels slightly above the keypoints
    let label_offset_y = -12.0;

    let positions: Vec<[f32; 2]> = valid_tracks
        .iter()
        .map(|t| [t.stereo.left_kp.x, t.stereo.left_kp.y + label_offset_y])
        .collect();

    let labels: Vec<String> = valid_tracks
        .iter()
        .map(|t| format!("{}", t.id))
        .collect();

    rec.log(
        path,
        &rr::Points2D::new(positions)
            .with_labels(labels)
            .with_radii(vec![0.0; valid_tracks.len()]), // Invisible points, just show labels
    )?;

    Ok(())
}

/// Log track trajectories on left image
fn log_track_trajectories(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    history: &HashMap<usize, Vec<(f32, f32)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut strips: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut line_colors: Vec<[u8; 3]> = Vec::new();

    for track in tracks {
        if let Some(positions) = history.get(&track.id) {
            if positions.len() >= 2 {
                let strip: Vec<[f32; 2]> = positions.iter().map(|&(x, y)| [x, y]).collect();
                let track_len = positions.len();
                strips.push(strip);
                line_colors.push(track_age_color(track_len));
            }
        }
    }

    if !strips.is_empty() {
        rec.log(
            path,
            &rr::LineStrips2D::new(strips)
                .with_colors(line_colors)
                .with_radii(vec![2.0]),
        )?;
    }

    Ok(())
}

/// Log disparity statistics as scalar timelines
fn log_disparity_info(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
) -> Result<(), Box<dyn std::error::Error>> {
    let valid_tracks: Vec<&TrackedFeature> = tracks.iter().filter(|t| t.age == 0).collect();

    if valid_tracks.is_empty() {
        return Ok(());
    }

    let disparities: Vec<f32> = valid_tracks.iter().map(|t| t.stereo.disparity).collect();
    let avg_disparity: f32 = disparities.iter().sum::<f32>() / disparities.len() as f32;
    let min_disparity = disparities.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_disparity = disparities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    rec.log(
        format!("{}/avg", path),
        &rr::Scalars::new([avg_disparity as f64]),
    )?;
    rec.log(
        format!("{}/min", path),
        &rr::Scalars::new([min_disparity as f64]),
    )?;
    rec.log(
        format!("{}/max", path),
        &rr::Scalars::new([max_disparity as f64]),
    )?;
    rec.log(
        format!("{}/count", path),
        &rr::Scalars::new([valid_tracks.len() as f64]),
    )?;

    Ok(())
}

/// Color based on track age (short = red, long = green)
fn track_age_color(length: usize) -> [u8; 3] {
    let norm = (length as f32 / 20.0).min(1.0);
    let r = ((1.0 - norm) * 255.0) as u8;
    let g = (norm * 255.0) as u8;
    [r, g, 100]
}
