//! Example: Track features across a stereo image sequence with Rerun visualization
//!
//! Usage:
//!   cargo run --release --example track_sequence_rerun -- <data_dir>
//!
//! Expects:
//! - images/ImageXXXX_L.jpg, images/ImageXXXX_R.jpg
//! - images/depthXXXX_L.exr, images/depthXXXX_R.exr
//! - camera_poses.bin (in data_dir)

use odysseus_slam::math::SE3;
use odysseus_slam::utils::load_camera_poses;
use odysseus_slam_frontend::{TrackedFeature, Tracker, TrackerConfig};
use odysseus_solver::math3d::Vec3;
use rerun as rr;
use std::collections::HashMap;
use std::env;
use std::path::Path;

use image::{GrayImage, RgbImage};

/// Camera intrinsics (adjust these to match your Blender camera)
struct CameraIntrinsics {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

impl CameraIntrinsics {
    /// Project a 3D point to 2D pixel coordinates
    fn project(&self, p: &Vec3<f64>) -> Option<(f64, f64)> {
        if p.z <= 0.0 {
            return None;
        }
        let x = self.fx * p.x / p.z + self.cx;
        let y = self.fy * p.y / p.z + self.cy;
        Some((x, y))
    }

    /// Unproject a 2D pixel with depth to 3D point
    fn unproject(&self, u: f64, v: f64, depth: f64) -> Vec3<f64> {
        let x = (u - self.cx) * depth / self.fx;
        let y = (v - self.cy) * depth / self.fy;
        Vec3::new(x, y, depth)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <data_dir>", args[0]);
        eprintln!("\nThis example tracks features across a stereo image sequence");
        eprintln!("and visualizes using Rerun.");
        eprintln!("\nExpected structure:");
        eprintln!("  <data_dir>/images/ImageXXXX_L.jpg");
        eprintln!("  <data_dir>/images/depthXXXX_L.exr");
        eprintln!("  <data_dir>/camera_poses.bin");
        std::process::exit(1);
    }

    let data_dir = Path::new(&args[1]);
    let image_dir = data_dir.join("images");

    // Load camera poses
    let poses_path = data_dir.join("camera_poses.bin");
    let poses = if poses_path.exists() {
        println!("Loading camera poses from {}", poses_path.display());
        Some(load_camera_poses(poses_path.to_str().unwrap())?)
    } else {
        println!("No camera_poses.bin found, GT flow disabled");
        None
    };

    // Initialize Rerun
    let rec = rr::RecordingStreamBuilder::new("odysseus_slam_frontend").spawn()?;

    println!("Rerun viewer spawned successfully");

    // Find all stereo pairs (new naming: ImageXXXX_L.jpg)
    let mut frame_numbers: Vec<u32> = Vec::new();
    for entry in std::fs::read_dir(&image_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("Image") && name.ends_with("_L.jpg") {
            // Extract frame number: Image0001_L.jpg -> 0001
            if let Some(num_str) = name.strip_prefix("Image").and_then(|s| s.strip_suffix("_L.jpg")) {
                if let Ok(num) = num_str.parse::<u32>() {
                    frame_numbers.push(num);
                }
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

    // Track feature history for visualization (id -> list of (frame_idx, left_pos, right_pos option))
    let mut track_history: HashMap<usize, Vec<(usize, (f32, f32), Option<(f32, f32)>)>> = HashMap::new();

    // GT track storage: track_id -> (initial_frame_idx, 3D point in world coordinates, list of projected 2D positions)
    // We store the 3D point in world coords so we can reproject it into each frame
    let mut gt_tracks: HashMap<usize, (usize, Vec3<f64>, Vec<(f32, f32)>)> = HashMap::new();

    // Camera intrinsics - 1024x1024 image with 90 deg FOV
    // fx = fy = width / (2 * tan(fov/2)) = 1024 / (2 * tan(45deg)) = 1024 / 2 = 512
    let intrinsics = CameraIntrinsics {
        fx: 512.0,
        fy: 512.0,
        cx: 512.0,
        cy: 512.0,
    };

    // Flow error accumulators
    let mut cumulative_errors: Vec<f64> = Vec::new(); // Error from track start to current
    let mut per_frame_errors: Vec<f64> = Vec::new();  // Error between consecutive frames

    // Process each frame
    let total_frames = frame_numbers.len();
    for (i, &frame_num) in frame_numbers.iter().enumerate() {
        let left_path = image_dir.join(format!("Image{:04}_L.jpg", frame_num));
        let right_path = image_dir.join(format!("Image{:04}_R.jpg", frame_num));
        let depth_path = image_dir.join(format!("depth{:04}_L.exr", frame_num));

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

        // Load depth map if available
        let depth_map = if depth_path.exists() {
            match load_exr_depth(&depth_path) {
                Ok(d) => Some(d),
                Err(e) => {
                    eprintln!("  Warning: Failed to load depth: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Process frame
        let start = std::time::Instant::now();
        let tracks = tracker.process_frame(&left_gray, &right_gray);
        let process_time = start.elapsed();

        // Update GT tracks and compute errors
        if let (Some(ref poses), Some(ref curr_depth)) = (&poses, &depth_map) {
            if i < poses.len() {
                let curr_pose = &poses[i];

                // Initialize GT tracks for new features (age == 0 means just detected)
                for track in &tracks {
                    if track.age == 0 && !gt_tracks.contains_key(&track.id) {
                        // Get depth at feature location
                        let px = track.stereo.left_kp.x.round() as usize;
                        let py = track.stereo.left_kp.y.round() as usize;
                        if py < curr_depth.len() && px < curr_depth[0].len() {
                            let depth = curr_depth[py][px] as f64;
                            if depth > 0.0 && depth < 100.0 {
                                // Unproject to 3D in camera frame
                                let p3d_cam = intrinsics.unproject(
                                    track.stereo.left_kp.x as f64,
                                    track.stereo.left_kp.y as f64,
                                    depth,
                                );
                                // Transform to world coordinates
                                // Try: poses are camera-to-world, so apply directly
                                let p3d_world = curr_pose.transform_point(p3d_cam);

                                // Store initial position
                                let init_pos = (track.stereo.left_kp.x, track.stereo.left_kp.y);
                                gt_tracks.insert(track.id, (i, p3d_world, vec![init_pos]));
                            }
                        }
                    }
                }

                // Update all GT tracks by reprojecting into current frame
                let active_track_ids: std::collections::HashSet<usize> =
                    tracks.iter().map(|t| t.id).collect();

                for (track_id, (_, p3d_world, positions)) in gt_tracks.iter_mut() {
                    // Only update if this track is still active
                    if active_track_ids.contains(track_id) {
                        // Transform world point to current camera frame
                        // Try: poses are camera-to-world, so use inverse to go world-to-camera
                        let p3d_cam = curr_pose.inverse().transform_point(*p3d_world);

                        // Project to 2D
                        if let Some((u, v)) = intrinsics.project(&p3d_cam) {
                            if u >= 0.0 && u < width as f64 && v >= 0.0 && v < height as f64 {
                                positions.push((u as f32, v as f32));
                            }
                        }
                    }
                }

                // Compute flow errors comparing tracked positions to GT positions
                for track in &tracks {
                    if let Some((_, _, gt_positions)) = gt_tracks.get(&track.id) {
                        if gt_positions.len() >= 2 {
                            // Cumulative error: current tracked vs current GT
                            let gt_curr = gt_positions.last().unwrap();
                            let tracked_curr = (track.stereo.left_kp.x, track.stereo.left_kp.y);
                            let cumulative_error = ((gt_curr.0 - tracked_curr.0).powi(2)
                                       + (gt_curr.1 - tracked_curr.1).powi(2)).sqrt() as f64;
                            cumulative_errors.push(cumulative_error);

                            // Per-frame error: compare frame-to-frame motion
                            // GT motion: gt_positions[-1] - gt_positions[-2]
                            // Tracked motion: we need history, use track_history
                            if let Some(hist) = track_history.get(&track.id) {
                                if hist.len() >= 2 {
                                    let prev_tracked = hist[hist.len() - 1].1; // previous tracked pos
                                    let gt_prev = &gt_positions[gt_positions.len() - 2];

                                    // GT flow this frame
                                    let gt_flow = (gt_curr.0 - gt_prev.0, gt_curr.1 - gt_prev.1);
                                    // Tracked flow this frame
                                    let tracked_flow = (
                                        tracked_curr.0 - prev_tracked.0,
                                        tracked_curr.1 - prev_tracked.1,
                                    );
                                    // Per-frame error is difference in flow vectors
                                    let frame_error = ((tracked_flow.0 - gt_flow.0).powi(2)
                                                     + (tracked_flow.1 - gt_flow.1).powi(2)).sqrt() as f64;
                                    per_frame_errors.push(frame_error);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update track history
        for track in &tracks {
            let left_pos = (track.stereo.left_kp.x, track.stereo.left_kp.y);
            let right_pos = if track.age == 0 {
                Some((track.stereo.right_kp.x, track.stereo.right_kp.y))
            } else {
                None
            };
            track_history
                .entry(track.id)
                .or_default()
                .push((i, left_pos, right_pos));
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

        // Log keypoints on left side of composite (view_idx=0)
        log_keypoints(&rec, "stereo_view/left_keypoints", &tracks, &track_history, 0, width)?;
        log_keypoint_labels(&rec, "stereo_view/left_keypoint_labels", &tracks, 0, width)?;

        // Log keypoints on right side of composite (view_idx=1)
        log_keypoints(&rec, "stereo_view/right_keypoints", &tracks, &track_history, 1, width)?;
        log_keypoint_labels(&rec, "stereo_view/right_keypoint_labels", &tracks, 1, width)?;

        // Log stereo match lines (colored by inlier/outlier status)
        log_stereo_matches(&rec, "stereo_view/matches", &tracks, width)?;

        // Log track trajectories on both views
        log_track_trajectories(&rec, "stereo_view/left_tracks", &tracks, &track_history, 0, width)?;
        log_track_trajectories(&rec, "stereo_view/right_tracks", &tracks, &track_history, 1, width)?;

        // Log disparity statistics
        log_disparity_info(&rec, "stats/disparity", &tracks)?;

        // Log GT trajectories alongside tracked trajectories
        log_gt_trajectories(&rec, "stereo_view/gt_tracks", &tracks, &gt_tracks)?;

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
        let total_tracks = track_lengths.len();
        let max_len = *track_lengths.iter().max().unwrap();
        let avg_len: f32 = track_lengths.iter().sum::<usize>() as f32 / total_tracks as f32;

        // Count tracks above various thresholds
        let above_5 = track_lengths.iter().filter(|&&l| l >= 5).count();
        let above_10 = track_lengths.iter().filter(|&&l| l >= 10).count();
        let above_20 = track_lengths.iter().filter(|&&l| l >= 20).count();
        let above_40 = track_lengths.iter().filter(|&&l| l >= 40).count();

        println!("Total unique tracks: {}", total_tracks);
        println!("Max track length: {} frames", max_len);
        println!("Avg track length: {:.1} frames", avg_len);
        println!();
        println!("Track length distribution:");
        println!("  >= 5 frames:  {:4} ({:5.1}%)", above_5, 100.0 * above_5 as f32 / total_tracks as f32);
        println!("  >= 10 frames: {:4} ({:5.1}%)", above_10, 100.0 * above_10 as f32 / total_tracks as f32);
        println!("  >= 20 frames: {:4} ({:5.1}%)", above_20, 100.0 * above_20 as f32 / total_tracks as f32);
        println!("  >= 40 frames: {:4} ({:5.1}%)", above_40, 100.0 * above_40 as f32 / total_tracks as f32);
    }

    // Print flow accuracy statistics
    if !per_frame_errors.is_empty() {
        per_frame_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_error: f64 = per_frame_errors.iter().sum::<f64>() / per_frame_errors.len() as f64;
        let median_error = per_frame_errors[per_frame_errors.len() / 2];
        let p90_error = per_frame_errors[(per_frame_errors.len() as f64 * 0.9) as usize];
        let p99_error = per_frame_errors[(per_frame_errors.len() as f64 * 0.99).min(per_frame_errors.len() as f64 - 1.0) as usize];

        println!();
        println!("=== Per-Frame Flow Accuracy (vs Ground Truth) ===");
        println!("Measurements: {}", per_frame_errors.len());
        println!("Mean error:   {:.2} pixels", mean_error);
        println!("Median error: {:.2} pixels", median_error);
        println!("90th %%ile:   {:.2} pixels", p90_error);
        println!("99th %%ile:   {:.2} pixels", p99_error);
    }

    if !cumulative_errors.is_empty() {
        cumulative_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_error: f64 = cumulative_errors.iter().sum::<f64>() / cumulative_errors.len() as f64;
        let median_error = cumulative_errors[cumulative_errors.len() / 2];
        let p90_error = cumulative_errors[(cumulative_errors.len() as f64 * 0.9) as usize];
        let p99_error = cumulative_errors[(cumulative_errors.len() as f64 * 0.99).min(cumulative_errors.len() as f64 - 1.0) as usize];

        println!();
        println!("=== Cumulative Track Drift (from start) ===");
        println!("Measurements: {}", cumulative_errors.len());
        println!("Mean error:   {:.2} pixels", mean_error);
        println!("Median error: {:.2} pixels", median_error);
        println!("90th %%ile:   {:.2} pixels", p90_error);
        println!("99th %%ile:   {:.2} pixels", p99_error);
    }

    println!("\nRerun visualization active. Close the viewer to exit.");

    Ok(())
}

/// Load depth map from EXR file
/// Blender exports depth as a single channel named "Depth.V"
fn load_exr_depth(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    use exr::prelude::*;

    // Read all channels from the first layer
    let image = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .from_file(path)?;

    let layer = &image.layer_data;
    let size = image.layer_data.size;
    let width = size.width();
    let height = size.height();

    // Find the depth channel (Blender names it "Depth.V")
    // Just use the first channel since we know there's only one
    let depth_channel = layer
        .channel_data
        .list
        .first()
        .ok_or("No depth channel found in EXR")?;

    // Extract pixel values
    let mut depth_map = vec![vec![0.0f32; width]; height];
    match &depth_channel.sample_data {
        FlatSamples::F32(samples) => {
            for y in 0..height {
                for x in 0..width {
                    depth_map[y][x] = samples[y * width + x];
                }
            }
        }
        FlatSamples::F16(samples) => {
            for y in 0..height {
                for x in 0..width {
                    depth_map[y][x] = samples[y * width + x].to_f32();
                }
            }
        }
        FlatSamples::U32(samples) => {
            for y in 0..height {
                for x in 0..width {
                    depth_map[y][x] = samples[y * width + x] as f32;
                }
            }
        }
    }

    Ok(depth_map)
}

/// Log GT trajectories as cyan lines alongside the tracked trajectories
/// GT tracks show where the 3D point *should* project based on camera poses
fn log_gt_trajectories(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    gt_tracks: &HashMap<usize, (usize, Vec3<f64>, Vec<(f32, f32)>)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut gt_strips: Vec<Vec<[f32; 2]>> = Vec::new();

    // Only show GT tracks for currently active features
    for track in tracks {
        if let Some((_, _, positions)) = gt_tracks.get(&track.id) {
            if positions.len() >= 2 {
                let strip: Vec<[f32; 2]> = positions.iter().map(|p| [p.0, p.1]).collect();
                gt_strips.push(strip);
            }
        }
    }

    // Always log to clear stale data from previous frames
    if gt_strips.is_empty() {
        rec.log(path, &rr::LineStrips2D::new(Vec::<Vec<[f32; 2]>>::new()))?;
    } else {
        rec.log(
            path,
            &rr::LineStrips2D::new(gt_strips)
                .with_colors(vec![[0u8, 255, 255]]) // Cyan for GT
                .with_radii(vec![2.0]),
        )?;
    }

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

/// Log keypoints without labels
/// view_idx: 0 = left image, 1 = right image (offset by image_width)
fn log_keypoints(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    history: &HashMap<usize, Vec<(usize, (f32, f32), Option<(f32, f32)>)>>,
    view_idx: u32,
    image_width: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let x_offset = view_idx as f32 * image_width as f32;

    // For right view, only show tracks with valid stereo matches
    let filtered_tracks: Vec<_> = if view_idx == 0 {
        tracks.iter().collect()
    } else {
        tracks.iter().filter(|t| t.age == 0).collect()
    };

    // Always log to clear stale data from previous frames
    if filtered_tracks.is_empty() {
        rec.log(path, &rr::Points2D::new(Vec::<[f32; 2]>::new()))?;
        return Ok(());
    }

    let positions: Vec<[f32; 2]> = filtered_tracks
        .iter()
        .map(|t| {
            if view_idx == 0 {
                [t.stereo.left_kp.x + x_offset, t.stereo.left_kp.y]
            } else {
                [t.stereo.right_kp.x + x_offset, t.stereo.right_kp.y]
            }
        })
        .collect();

    // Color based on track length (longer = greener)
    let point_colors: Vec<[u8; 3]> = filtered_tracks
        .iter()
        .map(|t| {
            let track_len = history.get(&t.id).map(|h| h.len()).unwrap_or(1);
            track_age_color(track_len)
        })
        .collect();

    // Size based on whether stereo match is valid
    let radii: Vec<f32> = filtered_tracks
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

/// Log keypoint track ID labels as a separate entity
/// view_idx: 0 = left image, 1 = right image (offset by image_width)
fn log_keypoint_labels(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    view_idx: u32,
    image_width: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let x_offset = view_idx as f32 * image_width as f32;

    // For right view, only show tracks with valid stereo matches
    let filtered_tracks: Vec<_> = if view_idx == 0 {
        tracks.iter().collect()
    } else {
        tracks.iter().filter(|t| t.age == 0).collect()
    };

    // Always log to clear stale data from previous frames
    if filtered_tracks.is_empty() {
        rec.log(path, &rr::Points2D::new(Vec::<[f32; 2]>::new()))?;
        return Ok(());
    }

    let positions: Vec<[f32; 2]> = filtered_tracks
        .iter()
        .map(|t| {
            if view_idx == 0 {
                [t.stereo.left_kp.x + x_offset, t.stereo.left_kp.y]
            } else {
                [t.stereo.right_kp.x + x_offset, t.stereo.right_kp.y]
            }
        })
        .collect();

    let labels: Vec<String> = filtered_tracks
        .iter()
        .map(|t| format!("{}", t.id))
        .collect();

    rec.log(
        path,
        &rr::Points2D::new(positions)
            .with_labels(labels)
            .with_show_labels(true),
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

    // Always log to clear stale data from previous frames
    if strips.is_empty() {
        rec.log(path, &rr::LineStrips2D::new(Vec::<Vec<[f32; 2]>>::new()))?;
    } else {
        rec.log(
            path,
            &rr::LineStrips2D::new(strips)
                .with_colors(line_colors)
                .with_radii(vec![1.5]),
        )?;
    }

    Ok(())
}

/// Log track trajectories
/// view_idx: 0 = left image, 1 = right image (offset by image_width)
fn log_track_trajectories(
    rec: &rr::RecordingStream,
    path: &str,
    tracks: &[TrackedFeature],
    history: &HashMap<usize, Vec<(usize, (f32, f32), Option<(f32, f32)>)>>,
    view_idx: u32,
    image_width: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let x_offset = view_idx as f32 * image_width as f32;
    let mut strips: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut line_colors: Vec<[u8; 3]> = Vec::new();

    for track in tracks {
        if let Some(positions) = history.get(&track.id) {
            // Extract positions for this view
            let view_positions: Vec<[f32; 2]> = positions
                .iter()
                .filter_map(|(_, left, right)| {
                    if view_idx == 0 {
                        Some([left.0 + x_offset, left.1])
                    } else {
                        right.map(|(x, y)| [x + x_offset, y])
                    }
                })
                .collect();

            if view_positions.len() >= 2 {
                let track_len = view_positions.len();
                strips.push(view_positions);
                line_colors.push(track_age_color(track_len));
            }
        }
    }

    // Always log to clear stale data from previous frames
    if strips.is_empty() {
        rec.log(path, &rr::LineStrips2D::new(Vec::<Vec<[f32; 2]>>::new()))?;
    } else {
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
/// Uses HSV interpolation: red (0°) -> yellow (60°) -> green (120°)
fn track_age_color(length: usize) -> [u8; 3] {
    let norm = (length as f32 / 20.0).min(1.0);
    // Hue: 0 (red) to 120 (green) degrees, normalized to 0-1
    let hue = norm * (120.0 / 360.0);
    let saturation = 1.0;
    let value = 1.0;
    hsv_to_rgb(hue, saturation, value)
}

/// Convert HSV to RGB
/// h: 0-1 (0=red, 1/6=yellow, 1/3=green, etc.)
/// s, v: 0-1
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let h_prime = h * 6.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    ]
}

