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

use image::GrayImage;

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

    // Initialize Rerun (same pattern as incremental_slam_demo.rs)
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

    // Generate colors for track IDs
    let colors: Vec<[u8; 3]> = (0..1000)
        .map(|i| {
            let hue = (i as f32 * 0.618033988749895) % 1.0; // Golden ratio for good distribution
            hsv_to_rgb(hue, 0.8, 0.9)
        })
        .collect();

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

        // Log left image
        let left_rgb = left_img.to_rgb8();
        rec.log(
            "world/left_image",
            &rr::Image::from_rgb24(left_rgb.into_raw(), [width, height]),
        )?;

        // Log right image
        let right_rgb = right_img.to_rgb8();
        rec.log(
            "world/right_image",
            &rr::Image::from_rgb24(right_rgb.into_raw(), [width, height]),
        )?;

        // Log keypoints on left image
        log_keypoints(&rec, "world/left_image/keypoints", &tracks, &colors)?;

        // Log keypoints on right image (using right_kp positions)
        log_right_keypoints(&rec, "world/right_image/keypoints", &tracks, &colors)?;

        // Log stereo matches as lines between left and right
        log_stereo_matches(&rec, "world/stereo_matches", &tracks, &colors, width)?;

        // Log orientation arrows
        log_orientations(&rec, "world/left_image/orientations", &tracks)?;

        // Log track trajectories
        log_track_trajectories(&rec, "world/left_image/tracks", &tracks, &track_history, &colors)?;

        // Log keypoint IDs as text annotations
        log_keypoint_ids(&rec, "world/left_image/ids", &tracks)?;

        // Log disparity information
        log_disparity_info(&rec, "world/disparity", &tracks)?;

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

/// Log keypoints as 2D points
fn log_keypoints(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    colors: &[[u8; 3]],
) -> Result<(), Box<dyn std::error::Error>> {
    let positions: Vec<[f32; 2]> = tracks
        .iter()
        .map(|t| [t.stereo.left_kp.x, t.stereo.left_kp.y])
        .collect();

    let point_colors: Vec<[u8; 3]> = tracks
        .iter()
        .map(|t| colors[t.id % colors.len()])
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

/// Log keypoints on right image
fn log_right_keypoints(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    colors: &[[u8; 3]],
) -> Result<(), Box<dyn std::error::Error>> {
    let valid_tracks: Vec<_> = tracks.iter().filter(|t| t.age == 0).collect();

    let positions: Vec<[f32; 2]> = valid_tracks
        .iter()
        .map(|t| [t.stereo.right_kp.x, t.stereo.right_kp.y])
        .collect();

    let point_colors: Vec<[u8; 3]> = valid_tracks
        .iter()
        .map(|t| colors[t.id % colors.len()])
        .collect();

    let num_points = positions.len();
    rec.log(
        path,
        &rr::Points2D::new(positions)
            .with_colors(point_colors)
            .with_radii(vec![5.0; num_points]),
    )?;

    Ok(())
}

/// Log stereo matches as line segments
fn log_stereo_matches(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    colors: &[[u8; 3]],
    image_width: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a side-by-side view coordinate system
    // Left image at x=0, right image offset by width
    let mut strips: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut line_colors: Vec<[u8; 3]> = Vec::new();

    for track in tracks.iter().filter(|t| t.age == 0) {
        let left_pt = [track.stereo.left_kp.x, track.stereo.left_kp.y];
        let right_pt = [
            track.stereo.right_kp.x + image_width as f32,
            track.stereo.right_kp.y,
        ];
        strips.push(vec![left_pt, right_pt]);
        line_colors.push(colors[track.id % colors.len()]);
    }

    if !strips.is_empty() {
        rec.log(
            path,
            &rr::LineStrips2D::new(strips)
                .with_colors(line_colors)
                .with_radii(vec![1.0]),
        )?;
    }

    Ok(())
}

/// Log keypoint orientations as arrows
fn log_orientations(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
) -> Result<(), Box<dyn std::error::Error>> {
    let arrow_len = 15.0;

    let vectors: Vec<[f32; 2]> = tracks
        .iter()
        .map(|t| {
            let angle = t.stereo.left_kp.angle;
            [arrow_len * angle.cos(), arrow_len * angle.sin()]
        })
        .collect();

    let origins: Vec<[f32; 2]> = tracks
        .iter()
        .map(|t| [t.stereo.left_kp.x, t.stereo.left_kp.y])
        .collect();

    let num_arrows = tracks.len();
    rec.log(
        path,
        &rr::Arrows2D::from_vectors(vectors)
            .with_origins(origins)
            .with_colors(vec![[255u8, 100, 100]; num_arrows])
            .with_radii(vec![1.5; num_arrows]),
    )?;

    Ok(())
}

/// Log track trajectories
fn log_track_trajectories(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    history: &HashMap<usize, Vec<(f32, f32)>>,
    colors: &[[u8; 3]],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut strips: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut line_colors: Vec<[u8; 3]> = Vec::new();

    for track in tracks {
        if let Some(positions) = history.get(&track.id) {
            if positions.len() >= 2 {
                let strip: Vec<[f32; 2]> = positions.iter().map(|&(x, y)| [x, y]).collect();
                strips.push(strip);
                line_colors.push(colors[track.id % colors.len()]);
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

/// Log keypoint IDs as text
fn log_keypoint_ids(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
) -> Result<(), Box<dyn std::error::Error>> {
    // Only show IDs for long-lived tracks to avoid clutter
    let long_tracks: Vec<&TrackedFeature> = tracks
        .iter()
        .filter(|t| {
            // Show ID if track has been alive for a while
            t.age == 0 // Has valid stereo match
        })
        .collect();

    if long_tracks.is_empty() {
        return Ok(());
    }

    // Log as text annotations - one per keypoint
    for track in long_tracks.iter().take(50) {
        // Limit to avoid clutter
        let label_path = format!("{}/id_{}", path, track.id);
        rec.log(
            label_path.as_str(),
            &rr::TextDocument::new(format!("{}", track.id))
        )?;
    }

    Ok(())
}

/// Log disparity as a scalar timeline
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

/// Convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}
