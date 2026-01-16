//! Integrated Stereo SLAM Demo
//!
//! Combines the visual frontend (feature tracking) with the backend (bundle adjustment)
//! to run SLAM on real stereo image sequences.
//!
//! DIAGNOSTIC MODE: If room_mesh.bin exists in the data directory, the demo will
//! automatically use synthetic observations instead of the visual frontend. This
//! allows isolating backend performance from frontend tracking errors.
//!
//! Usage:
//!   cd odysseus-slam
//!   cargo run --release --example integrated_slam_demo -- <data_dir>
//!   cargo run --release --example integrated_slam_demo -- <data_dir> --noise 2.0
//!
//! Expected data structure:
//!   <data_dir>/images/Image{:04}_L.jpg, Image{:04}_R.jpg
//!   <data_dir>/camera_poses.bin (ground truth, optional)
//!   <data_dir>/room_mesh.bin (optional, enables diagnostic mode)

use backtrace_on_stack_overflow;
use clap::Parser;
use odysseus_slam::{
    camera::StereoCamera,
    frame_graph::{FrameGraph, FrameRole, OptimizationState},
    geometry::StereoObservation,
    math::SE3,
    optimization::{run_bundle_adjustment, visualize_optimization_graph, BundleAdjustmentConfig, MarginalizedPrior},
    simulation::{add_noise_to_stereo_observations, generate_stereo_observations},
    utils::{get_peak_rss_mb, get_rss_mb, load_camera_poses, load_point_cloud_vec3},
    visualization::{visualize_estimate, visualize_estimate_with_gt_points, visualize_gba_update, visualize_ground_truth},
    SlamSystemDynamic, WorldState,
};
use odysseus_slam_frontend::{TrackedFeature, Tracker, TrackerConfig};
use odysseus_solver::math3d::Vec3;
use rerun as rr;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use image::RgbImage;

/// Integrated Stereo SLAM Demo
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to data directory containing images/ and camera_poses.bin
    #[arg(default_value = "./blender_stuff/moshi_room_easy")]
    data_dir: String,

    /// Stereo baseline in meters (distance between left and right cameras)
    #[arg(long, default_value_t = 0.1)]
    baseline: f64,

    /// Focal length in pixels (for 90° FOV on 1024 image: 512)
    #[arg(long, default_value_t = 512.0)]
    focal_length: f64,

    /// Observation noise stddev in pixels (for synthetic observations from mesh)
    #[arg(long, default_value_t = 2.0)]
    noise: f64,
}

// SLAM parameters
const WINDOW_SIZE: usize = 5;
const BASELINE_RATIO: f64 = 0.1;

/// Convert tracked features to stereo observations
fn features_to_observations(features: &[TrackedFeature], frame_idx: usize) -> Vec<StereoObservation> {
    features
        .iter()
        .filter(|f| f.age == 0) // Only features with valid stereo match this frame
        .map(|f| {
            StereoObservation::new(
                f.id,
                frame_idx,
                f.stereo.left_kp.x as f64,
                f.stereo.left_kp.y as f64,
                f.stereo.right_kp.x as f64,
                f.stereo.right_kp.y as f64,
            )
        })
        .collect()
}

/// Compute median depth of observed points from current camera pose
fn compute_median_depth(
    observations: &[StereoObservation],
    world: &WorldState,
    current_pose: &SE3<f64>,
) -> Option<f64> {
    let inverse_pose = current_pose.inverse();
    let mut depths: Vec<f64> = observations
        .iter()
        .filter_map(|obs| world.get_point(obs.point_id))
        .map(|world_pt| {
            let cam_pt = inverse_pose.transform_point(world_pt);
            cam_pt.z
        })
        .filter(|&z| z > 0.0)
        .collect();

    if depths.is_empty() {
        return None;
    }
    depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(depths[depths.len() / 2])
}

/// Find stereo image pairs in the data directory
fn find_stereo_pairs(image_dir: &Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut frame_numbers: Vec<u32> = Vec::new();

    for entry in std::fs::read_dir(image_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();

        // New naming convention: Image0001_L.jpg
        if name.starts_with("Image") && name.ends_with("_L.jpg") {
            if let Some(num_str) = name
                .strip_prefix("Image")
                .and_then(|s| s.strip_suffix("_L.jpg"))
            {
                if let Ok(num) = num_str.parse::<u32>() {
                    frame_numbers.push(num);
                }
            }
        }
    }

    frame_numbers.sort();
    Ok(frame_numbers)
}

/// Load depth map from EXR file
/// Blender exports depth as a single channel
fn load_exr_depth(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    use exr::prelude::*;

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

    let depth_channel = layer
        .channel_data
        .list
        .first()
        .ok_or("No depth channel found in EXR")?;

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

/// Convert pixel coordinates and depth to 3D point in world frame
fn pixel_to_world_point(
    x: f64,
    y: f64,
    depth: f32,
    camera_pose: &SE3<f64>,
    stereo_camera: &StereoCamera<f64>,
) -> Vec3<f64> {
    // Convert pixel to normalized camera coordinates
    let cam_x = (x - stereo_camera.left.cx) / stereo_camera.left.fx;
    let cam_y = (y - stereo_camera.left.cy) / stereo_camera.left.fy;

    // Create 3D point in camera frame
    let depth_f64 = depth as f64;
    let point_camera = Vec3::new(cam_x * depth_f64, cam_y * depth_f64, depth_f64);

    // Transform to world frame
    camera_pose.transform_point(point_camera)
}

/// Load all depth maps for the given frame numbers
fn load_all_depth_maps(
    image_dir: &Path,
    frame_numbers: &[u32],
) -> Vec<Option<Vec<Vec<f32>>>> {
    println!("📊 Loading depth maps...");
    let mut depth_maps = Vec::new();
    let mut loaded_count = 0;

    for &frame_num in frame_numbers {
        let depth_path = image_dir.join(format!("depth{:04}_L.exr", frame_num));
        if depth_path.exists() {
            match load_exr_depth(&depth_path) {
                Ok(depth_map) => {
                    depth_maps.push(Some(depth_map));
                    loaded_count += 1;
                }
                Err(_) => {
                    depth_maps.push(None);
                }
            }
        } else {
            depth_maps.push(None);
        }
    }

    println!("  Loaded {}/{} depth maps\n", loaded_count, frame_numbers.len());
    depth_maps
}

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


fn main() {
    let args = Args::parse();
    unsafe {
        let _ = backtrace_on_stack_overflow::enable(|| run_slam(&args));
    }
}

fn run_slam(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Integrated Stereo SLAM Demo");
    println!("==============================\n");
    println!("📊 Memory at startup: {:.1} MB", get_rss_mb());

    let data_dir = Path::new(&args.data_dir);
    let image_dir = data_dir.join("images");

    // Initialize Rerun
    let rec = rr::RecordingStreamBuilder::new("integrated_slam").spawn()?;
    rec.log_static("world", &rr::ViewCoordinates::RDF())?;

    // Camera setup
    let image_width = 1024.0;
    let image_height = 1024.0;
    let stereo_camera = StereoCamera::simple(
        args.focal_length,
        image_width,
        image_height,
        args.baseline,
    );

    println!("📷 Stereo Camera:");
    println!("  Focal length: {} px", args.focal_length);
    println!("  Baseline: {} m", args.baseline);
    println!("  Image size: {}x{}\n", image_width, image_height);

    // Load ground truth poses if available
    let poses_path = data_dir.join("camera_poses.bin");
    let gt_poses = if poses_path.exists() {
        println!("📍 Loading ground truth poses from {}", poses_path.display());
        let poses = load_camera_poses(poses_path.to_str().unwrap())?;
        println!("  {} poses loaded\n", poses.len());
        Some(poses)
    } else {
        println!("⚠️  No camera_poses.bin found, running without ground truth\n");
        None
    };

    // Check for diagnostic mode: if room_mesh.bin exists, use synthetic observations
    let mesh_path = data_dir.join("room_mesh.bin");
    let use_synthetic_observations = mesh_path.exists();

    if use_synthetic_observations {
        println!("🔬 DIAGNOSTIC MODE: room_mesh.bin detected");
        println!("   Using synthetic observations instead of visual frontend\n");
    }

    // Determine frame count and frame numbers
    let (total_frames, frame_numbers) = if use_synthetic_observations {
        // Synthetic mode: use sequential indices matching GT poses
        if gt_poses.is_none() {
            return Err("Diagnostic mode requires camera_poses.bin".into());
        }
        let count = gt_poses.as_ref().unwrap().len();
        (count, (0..count as u32).collect())
    } else {
        // Tracker mode: use actual frame numbers from image files
        let numbers = find_stereo_pairs(&image_dir)?;
        if numbers.is_empty() {
            return Err("No stereo pairs found in images directory".into());
        }
        let count = numbers.len();

        // Verify frame count matches GT poses if available
        if let Some(ref poses) = gt_poses {
            if count != poses.len() {
                eprintln!("⚠️  Warning: Found {} image pairs but {} GT poses", count, poses.len());
            }
        }

        (count, numbers)
    };

    if use_synthetic_observations {
        println!("📹 Using {} frames from camera poses\n", total_frames);
    } else {
        println!("📹 Processing {} frames\n", total_frames);
    }

    // Generate observations and ground truth points based on mode
    let (mut frame_observations_arc, mut gt_points_vec): (Arc<Vec<Vec<StereoObservation>>>, Vec<Vec3<f64>>);
    let depth_maps_for_viz: Vec<Option<Vec<Vec<f32>>>>;

    if use_synthetic_observations {
        // DIAGNOSTIC MODE: Generate synthetic observations from mesh
        println!("🔍 Generating synthetic observations from mesh...");

        if gt_poses.is_none() {
            return Err("Synthetic observation mode requires camera_poses.bin".into());
        }

        let gt_points = load_point_cloud_vec3(mesh_path.to_str().unwrap())?;
        println!("  Loaded {} 3D points from mesh", gt_points.len());

        let perfect_observations = generate_stereo_observations(
            &gt_points,
            gt_poses.as_ref().unwrap(),
            &stereo_camera,
            image_width,
            image_height,
        );

        let observations = if args.noise > 0.0 {
            println!("  Adding noise with stddev = {} pixels", args.noise);
            add_noise_to_stereo_observations(&perfect_observations, args.noise, 123)
        } else {
            println!("  Using perfect observations (no noise)");
            perfect_observations
        };

        let frame_obs: Vec<Vec<StereoObservation>> = (0..total_frames)
            .map(|frame_idx| {
                observations
                    .iter()
                    .filter(|obs| obs.camera_id == frame_idx)
                    .cloned()
                    .collect()
            })
            .collect();

        println!("  Generated {} total observations\n", observations.len());

        frame_observations_arc = Arc::new(frame_obs);
        gt_points_vec = gt_points;
        depth_maps_for_viz = Vec::new();

    } else {
        // TRACKER MODE: Use visual frontend with depth maps for ground truth
        println!("📷 TRACKER MODE: Using visual frontend");
        println!("   Loading depth maps for ground truth points...\n");

        let depth_maps = load_all_depth_maps(&image_dir, &frame_numbers);

        // Start with empty GT points, will populate during tracking
        gt_points_vec = Vec::new();

        // In tracker mode, observations will be generated frame-by-frame
        // Create a placeholder that will be populated in the main loop
        frame_observations_arc = Arc::new(Vec::new());

        println!("🔍 Tracker initialized for SLAM (min: 150, max: 400 features)\n");
        
        // Store depth maps for GT track initialization in main loop
        depth_maps_for_viz = depth_maps;
    } // End of tracker mode else block

    // Initialize 2D tracking visualization state (only used in tracker mode)
    // Track feature history for visualization (id -> list of (frame_idx, left_pos, right_pos option))
    let mut track_history: HashMap<usize, Vec<(usize, (f32, f32), Option<(f32, f32)>)>> = HashMap::new();
    
    // GT track storage: track_id -> (initial_frame_idx, 3D point in world coordinates, list of projected 2D positions)
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

    // Create WorldState and FrameGraph
    let mut world = WorldState::new();
    let mut frame_graph = FrameGraph::new();

    // Storage for observations (needed for GBA thread and LBA in tracker mode)
    let mut all_observations: Vec<Vec<StereoObservation>> = Vec::new();

    // Initialize tracker for main loop if in tracker mode
    let mut tracker_opt = if use_synthetic_observations {
        None
    } else {
        let config = TrackerConfig {
            min_features: 150,
            max_features: 400,
            ..Default::default()
        };
        
        // Create frontend camera model from backend stereo camera
        use odysseus_slam_frontend::{PinholeCamera, StereoCamera as FrontendStereoCamera};
        let frontend_camera = FrontendStereoCamera {
            left: PinholeCamera {
                fx: stereo_camera.left.fx as f32,
                fy: stereo_camera.left.fy as f32,
                cx: stereo_camera.left.cx as f32,
                cy: stereo_camera.left.cy as f32,
            },
            baseline: stereo_camera.baseline as f32,
        };
        
        Some(Tracker::with_config(config).with_camera(frontend_camera))
    };

    // Process first frame to initialize
    println!("🚀 Processing frame 0 for initialization...");

    let frame0_obs = if use_synthetic_observations {
        // Get observations from pre-generated data
        frame_observations_arc[0].clone()
        } else {
            // Use tracker to get observations
            let first_frame_num = frame_numbers[0];
            let left_path = image_dir.join(format!("Image{:04}_L.jpg", first_frame_num));
            let right_path = image_dir.join(format!("Image{:04}_R.jpg", first_frame_num));

            let left_img = image::open(&left_path)?.to_luma8();
            let right_img = image::open(&right_path)?.to_luma8();

            let tracks = tracker_opt.as_mut().unwrap().process_frame(&left_img, &right_img);
            
            // No depth feedback for first frame (no prior world state)
            
            let obs = features_to_observations(&tracks, 0);
            println!("  Frame 0: {} tracks, {} with stereo", tracks.len(), obs.len());
            obs
        };

    if use_synthetic_observations {
        println!("  Frame 0: {} observations from synthetic data", frame0_obs.len());
    }

    // Initialize with ground truth pose if available, otherwise identity
    let initial_pose = gt_poses
        .as_ref()
        .and_then(|p| p.first().cloned())
        .unwrap_or_else(SE3::identity);

    frame_graph.add_frame(FrameRole::Keyframe, OptimizationState::Fixed);
    world.add_pose(initial_pose);

    // Triangulate initial points
    for obs in &frame0_obs {
        world.triangulate_and_add_point(obs, &stereo_camera, 0);
    }

    // In tracker mode, we need to build all_observations incrementally and update frame_observations_arc
    // In synthetic mode, frame_observations_arc is already fully populated
    if !use_synthetic_observations {
        all_observations.push(frame0_obs.clone());
        frame_observations_arc = Arc::new(all_observations.clone());
    }

    println!(
        "  Initialized {} points from triangulation\n",
        world.num_points()
    );

    // Create SLAM system (spawns GBA thread)
    // Use dynamic system which receives observations with each frame
    let mut slam_system = SlamSystemDynamic::new(stereo_camera.clone());

    // Visualize ground truth trajectory and tracked feature points
    if let Some(ref poses) = gt_poses {
        if use_synthetic_observations {
            visualize_ground_truth(&rec, Some(&gt_points_vec), poses, &stereo_camera)?;
        } else {
            visualize_ground_truth(&rec, None, poses, &stereo_camera)?;
        }
    }

    // Visualize initial state
    if use_synthetic_observations {
        visualize_estimate(
            &rec,
            0,
            &world,
            &frame_graph,
            &gt_points_vec,
            &stereo_camera,
            None,
        )?;
    } else {
        visualize_estimate_with_gt_points(
            &rec,
            0,
            &world,
            &frame_graph,
            &gt_points_vec,
            &stereo_camera,
            None,
        )?;
    }

    slam_system.send_to_gba(0, &world, frame_observations_arc.clone());

    println!("📊 Memory before frame processing: {:.1} MB\n", get_rss_mb());

    // Tracking variables
    let mut total_frame_time = 0.0;
    let mut total_lba_time = 0.0;
    let mut gba_update_count = 0;
    let mut gba_last_optimized_frame = 0;
    let mut prev_gba_frame_graph: Option<FrameGraph> = None;
    let mut prev_frame_graph: Option<FrameGraph> = None;
    let mut marginalized_prior: Option<MarginalizedPrior> = None;
    let mut last_keyframe_position: Vec3<f64> = initial_pose.translation;

    // MAIN LOOP
    for (i, &frame_num) in frame_numbers.iter().enumerate().skip(1) {
        let frame_idx = i;
        let frame_start = std::time::Instant::now();

        // Check for GBA results
        if let Some(gba_result) = slam_system.try_recv_from_gba() {
            let gba_world = &gba_result.world_state;
            let n_gba_frames = gba_world.frames.len();
            gba_update_count += 1;

            visualize_gba_update(
                &rec,
                gba_update_count,
                gba_world,
                &gba_result.frame_graph,
                &gt_points_vec,
                &stereo_camera,
                prev_gba_frame_graph.as_ref(),
            )?;

            prev_gba_frame_graph = Some(gba_result.frame_graph.clone());
            world.replace_frames_from(gba_world);
            gba_last_optimized_frame = gba_result.last_optimized_frame;

            println!(
                "  📥 GBA update #{} (frame {}, {} poses, {} points)",
                gba_update_count,
                gba_result.last_optimized_frame,
                n_gba_frames,
                gba_world.num_points()
            );
        }

        // Update frame graph based on GBA
        if let Some(frame_state) = frame_graph.get(gba_last_optimized_frame) {
            if frame_state.role != FrameRole::Keyframe {
                frame_graph.set_role(gba_last_optimized_frame, FrameRole::Stored);
            }
        }

        // Memory checkpoint
        if frame_idx % 10 == 0 {
            println!(
                "📊 Memory at frame {}: {:.1} MB (peak: {:.1} MB)",
                frame_idx,
                get_rss_mb(),
                get_peak_rss_mb()
            );
        }

        // Get observations for current frame
        let (current_frame_obs, tracks_opt, left_rgb_opt, right_rgb_opt, image_width_opt) = if use_synthetic_observations {
            // Use pre-generated observations
            (frame_observations_arc[frame_idx].clone(), None, None, None, None)
        } else {
            // Load images and track features
            let left_path = image_dir.join(format!("Image{:04}_L.jpg", frame_num));
            let right_path = image_dir.join(format!("Image{:04}_R.jpg", frame_num));

            // Load both RGB and grayscale images
            let left_img_full = image::open(&left_path)?;
            let right_img_full = image::open(&right_path)?;
            let left_rgb = left_img_full.to_rgb8();
            let right_rgb = right_img_full.to_rgb8();
            let left_gray = left_img_full.to_luma8();
            let right_gray = right_img_full.to_luma8();
            let (width, _height) = left_gray.dimensions();

            let tracks = tracker_opt.as_mut().unwrap().process_frame(&left_gray, &right_gray);
            
            // Extract depth from optimized world state and feed back to tracker
            let mut depth_updates = HashMap::new();
            for track in &tracks {
                if let Some(point) = world.get_point_xyz(track.id) {
                    // Transform to camera frame to get depth
                    let cam_point = world.frames[frame_idx - 1].world_to_camera(point);
                    if cam_point.z > 0.0 {
                        depth_updates.insert(track.id, cam_point.z as f32);
                    }
                }
            }
            
            // Update tracker with backend depths
            if !depth_updates.is_empty() {
                tracker_opt.as_mut().unwrap().update_depth_estimates(depth_updates);
            }
            
            // Update GT tracks and compute errors (only in tracker mode with GT poses and depth maps)
            if let (Some(ref poses), Some(ref depth_map)) = (&gt_poses, depth_maps_for_viz.get(frame_idx).and_then(|d| d.as_ref())) {
                if frame_idx < poses.len() {
                    let curr_pose = &poses[frame_idx];

                    // Initialize GT tracks for new features (age == 0 means just detected)
                    for track in &tracks {
                        if track.age == 0 && !gt_tracks.contains_key(&track.id) {
                            // Get depth at feature location
                            let px = track.stereo.left_kp.x.round() as usize;
                            let py = track.stereo.left_kp.y.round() as usize;
                            if py < depth_map.len() && px < depth_map[0].len() {
                                let depth = depth_map[py][px] as f64;
                                if depth > 0.0 && depth < 100.0 {
                                    // Unproject to 3D in camera frame
                                    let p3d_cam = intrinsics.unproject(
                                        track.stereo.left_kp.x as f64,
                                        track.stereo.left_kp.y as f64,
                                        depth,
                                    );
                                    // Transform to world coordinates
                                    let p3d_world = curr_pose.transform_point(p3d_cam);

                                    // Store initial position
                                    let init_pos = (track.stereo.left_kp.x, track.stereo.left_kp.y);
                                    gt_tracks.insert(track.id, (frame_idx, p3d_world, vec![init_pos]));
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
                            let p3d_cam = curr_pose.inverse().transform_point(*p3d_world);

                            // Project to 2D
                            if let Some((u, v)) = intrinsics.project(&p3d_cam) {
                                if u >= 0.0 && u < width as f64 && v >= 0.0 && v < width as f64 {
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
                    .push((frame_idx, left_pos, right_pos));
            }
            
            (features_to_observations(&tracks, frame_idx), Some(tracks), Some(left_rgb), Some(right_rgb), Some(width))
        };

        // Get last pose for initialization
        let last_pose = world.get_pose(frame_idx - 1).unwrap();

        // Keyframe selection
        let new_points_count = current_frame_obs
            .iter()
            .filter(|obs| world.get_point(obs.point_id).is_none())
            .count();
        let novelty_ratio = if current_frame_obs.is_empty() {
            0.0
        } else {
            new_points_count as f64 / current_frame_obs.len() as f64
        };

        let current_position = last_pose.translation;
        let translation_since_keyframe = (current_position - last_keyframe_position).norm();

        let sufficient_baseline =
            if let Some(median_depth) = compute_median_depth(&current_frame_obs, &world, &last_pose)
            {
                let min_baseline = median_depth * BASELINE_RATIO;
                translation_since_keyframe >= min_baseline
            } else {
                novelty_ratio >= 0.3
            };

        let should_create_keyframe = sufficient_baseline || novelty_ratio >= 0.3;
        let frame_role = if should_create_keyframe {
            FrameRole::Keyframe
        } else {
            FrameRole::Transient
        };

        // Add pose (initialize with last pose, LBA will refine)
        world.add_pose(last_pose);

        // Triangulate new points if keyframe
        if should_create_keyframe {
            println!(
                "  [KF] Frame {} (novelty: {:.1}%, baseline: {:.3}m)",
                frame_idx,
                novelty_ratio * 100.0,
                translation_since_keyframe
            );

            for obs in &current_frame_obs {
                if world.get_point(obs.point_id).is_none() {
                    world.triangulate_and_add_point(obs, &stereo_camera, frame_idx);
                    
                    // Compute GT point in tracker mode
                    if !use_synthetic_observations {
                        if let Some(ref depth_map) = depth_maps_for_viz.get(frame_idx).and_then(|d| d.as_ref()) {
                            let px = obs.left_u.round() as usize;
                            let py = obs.left_v.round() as usize;
                            if py < depth_map.len() && px < depth_map[0].len() {
                                let depth = depth_map[py][px];
                                if depth > 0.0 && depth < 100.0 {
                                    let gt_point = pixel_to_world_point(
                                        obs.left_u,
                                        obs.left_v,
                                        depth,
                                        &world.frames[frame_idx].world_pose(),
                                        &stereo_camera,
                                    );
                                    
                                    // Grow gt_points_vec if needed (use NAN as sentinel for uninitialized)
                                    if obs.point_id >= gt_points_vec.len() {
                                        gt_points_vec.resize(obs.point_id + 1, Vec3::new(f64::NAN, f64::NAN, f64::NAN));
                                    }
                                    gt_points_vec[obs.point_id] = gt_point;
                                }
                            }
                        }
                    }
                }
            }

            last_keyframe_position = current_position;
        }

        // Store observations and update Arc
        if !use_synthetic_observations {
            all_observations.push(current_frame_obs.clone());
            frame_observations_arc = Arc::new(all_observations.clone());
        } else {
            // In synthetic mode, ensure we have observations for this frame
            // (frame_observations_arc already contains all frames)
        }

        // Add frame to graph
        frame_graph.add_frame(frame_role, OptimizationState::Optimized);

        // Manage window
        let mut optimized_indices: Vec<usize> = frame_graph
            .states
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state == OptimizationState::Optimized)
            .map(|(idx, _)| idx)
            .collect();

        while optimized_indices.len() > WINDOW_SIZE {
            frame_graph.set_state(optimized_indices[0], OptimizationState::Marginalize);
            optimized_indices.remove(0);
        }

        // Fix GBA-optimized keyframes
        for j in 0..frame_graph.len() {
            let is_gba_optimized = j <= gba_last_optimized_frame;
            let is_keyframe = frame_graph.states[j].role == FrameRole::Keyframe;

            if j == 0 || (is_keyframe && is_gba_optimized) {
                frame_graph.set_state(j, OptimizationState::Fixed);
            }
        }

        if gba_last_optimized_frame > 0 {
            let gba_frame_role = frame_graph.states[gba_last_optimized_frame].role;
            if gba_frame_role != FrameRole::Keyframe {
                frame_graph.set_state(gba_last_optimized_frame, OptimizationState::Optimized);
            }
        }

        // Fixed points
        let fixed_point_ids: std::collections::HashSet<usize> = world
            .get_all_points()
            .iter()
            .filter_map(|(point_id, _)| {
                world
                    .get_point_keyframe(*point_id)
                    .filter(|&kf| kf <= gba_last_optimized_frame)
                    .map(|_| *point_id)
            })
            .collect();

        // Run LBA (frame_observations_arc is kept up to date above)
        let result = run_bundle_adjustment(
            &stereo_camera,
            &frame_graph,
            &mut world,
            &frame_observations_arc,
            marginalized_prior.as_ref(),
            &fixed_point_ids,
            &BundleAdjustmentConfig::lba().with_graph_viz(true),
        );
        let lba_time = result.solve_time_ms;
        total_lba_time += lba_time;

        // Visualize optimization graph
        if let Some(ref graph_info) = result.graph_info {
            visualize_optimization_graph(&rec, frame_idx, graph_info)?;
        }

        // Update prior
        marginalized_prior = result.new_prior;
        for j in 0..frame_graph.len() {
            if frame_graph.states[j].state == OptimizationState::Marginalize {
                frame_graph.set_state(j, OptimizationState::Inactive);
            }
        }

        // Send to GBA with updated observations
        slam_system.send_to_gba(frame_idx, &world, frame_observations_arc.clone());

        // Compute pose error if GT available
        let optimized_pose = world.frames[frame_idx].world_pose();
        let error_str = if let Some(ref poses) = gt_poses {
            if frame_idx < poses.len() {
                let pos_error = (optimized_pose.translation - poses[frame_idx].translation).norm();
                let q_err =
                    poses[frame_idx].rotation.inverse().quat * optimized_pose.rotation.quat;
                let angle_deg = 2.0 * q_err.w.abs().acos().to_degrees();
                format!(" | Err: {:.3}m, {:.2}°", pos_error, angle_deg)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Visualize
        if use_synthetic_observations {
            visualize_estimate(
                &rec,
                frame_idx,
                &world,
                &frame_graph,
                &gt_points_vec,
                &stereo_camera,
                prev_frame_graph.as_ref(),
            )?;
        } else {
            visualize_estimate_with_gt_points(
                &rec,
                frame_idx,
                &world,
                &frame_graph,
                &gt_points_vec,
                &stereo_camera,
                prev_frame_graph.as_ref(),
            )?;
        }
        prev_frame_graph = Some(frame_graph.clone());

        // 2D tracking visualization (tracker mode only)
        if !use_synthetic_observations {
            if let (Some(ref tracks), Some(ref left_rgb), Some(ref right_rgb), Some(width)) = 
                (tracks_opt, left_rgb_opt, right_rgb_opt, image_width_opt) {
                // Set timeline for 2D view
                rec.set_time_sequence("frame", frame_idx as i64);

                // Create and log side-by-side composite
                let composite = create_side_by_side(left_rgb, right_rgb);
                rec.log(
                    "stereo_view",
                    &rr::Image::from_rgb24(composite.into_raw(), [width * 2, width]),
                )?;

                // Log keypoints, labels, matches, trajectories
                log_keypoints(&rec, "stereo_view/left_keypoints", tracks, &track_history, 0, width)?;
                log_keypoint_labels(&rec, "stereo_view/left_keypoint_labels", tracks, 0, width)?;
                log_keypoints(&rec, "stereo_view/right_keypoints", tracks, &track_history, 1, width)?;
                log_keypoint_labels(&rec, "stereo_view/right_keypoint_labels", tracks, 1, width)?;
                log_stereo_matches(&rec, "stereo_view/matches", tracks, width)?;
                log_track_trajectories(&rec, "stereo_view/left_tracks", tracks, &track_history, 0, width)?;
                log_track_trajectories(&rec, "stereo_view/right_tracks", tracks, &track_history, 1, width)?;
                log_gt_trajectories(&rec, "stereo_view/gt_tracks", tracks, &gt_tracks)?;
                log_disparity_info(&rec, "stats/disparity", tracks)?;
            }
        }

        let frame_duration = frame_start.elapsed();
        total_frame_time += frame_duration.as_secs_f64() * 1000.0;

        let n_optimized = frame_graph
            .states
            .iter()
            .filter(|s| s.state == OptimizationState::Optimized)
            .count();
        let n_fixed = frame_graph
            .states
            .iter()
            .filter(|s| s.state == OptimizationState::Fixed)
            .count();

        // Print frame info
        if use_synthetic_observations {
            println!(
                "Frame {}: {} obs, {} opt, {} fixed, Map: {} pts, LBA: {:.2}ms{}{}",
                frame_idx,
                current_frame_obs.len(),
                n_optimized,
                n_fixed,
                world.num_points(),
                lba_time,
                if should_create_keyframe { " [KF]" } else { "" },
                error_str
            );
        } else {
            // In tracker mode, we don't have access to tracks here anymore
            // Just print observations count
            println!(
                "Frame {}: {} obs, {} opt, {} fixed, Map: {} pts, LBA: {:.2}ms{}{}",
                frame_idx,
                current_frame_obs.len(),
                n_optimized,
                n_fixed,
                world.num_points(),
                lba_time,
                if should_create_keyframe { " [KF]" } else { "" },
                error_str
            );
        }
    }

    println!("\n✅ Processed {} frames", total_frames);
    println!("   Final map: {} points", world.num_points());
    println!("   GBA updates received: {}", gba_update_count);
    println!(
        "\n📊 Final memory: {:.1} MB, Peak: {:.1} MB",
        get_rss_mb(),
        get_peak_rss_mb()
    );
    println!(
        "Average LBA time: {:.2} ms",
        total_lba_time / total_frames as f64
    );
    println!(
        "Average frame time: {:.2} ms",
        total_frame_time / total_frames as f64
    );

    // Print flow accuracy statistics (tracker mode only)
    if !use_synthetic_observations {
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
    }

    println!("\n📺 Open Rerun to see the SLAM visualization!");

    drop(slam_system);

    Ok(())
}
