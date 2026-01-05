//! Integrated Stereo SLAM Demo
//!
//! Combines the visual frontend (feature tracking) with the backend (bundle adjustment)
//! to run SLAM on real stereo image sequences.
//!
//! Usage:
//!   cd odysseus-slam
//!   cargo run --release --example integrated_slam_demo -- <data_dir>
//!
//! Expected data structure:
//!   <data_dir>/images/Image{:04}_L.jpg, Image{:04}_R.jpg
//!   <data_dir>/camera_poses.bin (ground truth, optional)

use backtrace_on_stack_overflow;
use clap::Parser;
use odysseus_slam::{
    camera::StereoCamera,
    frame_graph::{FrameGraph, FrameRole, OptimizationState},
    geometry::StereoObservation,
    math::SE3,
    optimization::{run_bundle_adjustment, BundleAdjustmentConfig, MarginalizedPrior},
    utils::{get_peak_rss_mb, get_rss_mb, load_camera_poses},
    visualization::{visualize_estimate, visualize_gba_update, visualize_ground_truth},
    SlamSystem, WorldState,
};
use odysseus_slam_frontend::{TrackedFeature, Tracker, TrackerConfig};
use odysseus_solver::math3d::Vec3;
use rerun as rr;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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

    // Find stereo pairs
    let frame_numbers = find_stereo_pairs(&image_dir)?;
    if frame_numbers.is_empty() {
        return Err("No stereo pairs found in images directory".into());
    }
    let total_frames = frame_numbers.len();
    println!("📹 Found {} stereo image pairs\n", total_frames);

    // Load all depth maps upfront for performance benchmarking
    let depth_maps = load_all_depth_maps(&image_dir, &frame_numbers);

    // Pre-run tracker to determine which features will be tracked
    println!("🔍 Pre-running tracker to determine tracked features...");
    let config = TrackerConfig {
        min_features: 150,
        max_features: 400,
        ..Default::default()
    };
    let mut pre_tracker = Tracker::with_config(config.clone());

    // Track through all frames to build GT point map
    let mut gt_points_map: HashMap<usize, Vec3<f64>> = HashMap::new();
    for (i, &frame_num) in frame_numbers.iter().enumerate() {
        let left_path = image_dir.join(format!("Image{:04}_L.jpg", frame_num));
        let right_path = image_dir.join(format!("Image{:04}_R.jpg", frame_num));

        let left_img = image::open(&left_path)?.to_luma8();
        let right_img = image::open(&right_path)?.to_luma8();

        let tracks = pre_tracker.process_frame(&left_img, &right_img);
        let frame_obs = features_to_observations(&tracks, i);

        // Build GT points for tracked features using cached depth map
        if let Some(ref depth_map) = depth_maps[i] {
            let pose = gt_poses.as_ref()
                .and_then(|p| p.get(i))
                .cloned()
                .unwrap_or_else(SE3::identity);

            for obs in &frame_obs {
                if !gt_points_map.contains_key(&obs.point_id) {
                    let x = obs.left_u as usize;
                    let y = obs.left_v as usize;
                    if y < depth_map.len() && x < depth_map[0].len() {
                        let depth = depth_map[y][x];
                        if depth > 0.0 && depth.is_finite() {
                            let gt_point = pixel_to_world_point(
                                obs.left_u,
                                obs.left_v,
                                depth,
                                &pose,
                                &stereo_camera,
                            );
                            gt_points_map.insert(obs.point_id, gt_point);
                        }
                    }
                }
            }
        }

        // Print progress every 5 frames
        if (i + 1) % 5 == 0 || i + 1 == frame_numbers.len() {
            println!(
                "  Progress: {}/{} frames, {} GT points so far",
                i + 1,
                frame_numbers.len(),
                gt_points_map.len()
            );
        }
    }
    println!("  Completed! Built {} ground truth points for tracked features\n", gt_points_map.len());

    // Convert to vector for visualization
    let mut gt_points_vec: Vec<Vec3<f64>> = vec![Vec3::new(0.0, 0.0, 0.0); gt_points_map.len()];
    for (&point_id, &point) in &gt_points_map {
        if point_id < gt_points_vec.len() {
            gt_points_vec[point_id] = point;
        }
    }

    // Create tracker for actual SLAM run
    let mut tracker = Tracker::with_config(config);
    println!("🔍 Tracker initialized for SLAM (min: 150, max: 400 features)\n");

    // Create WorldState and FrameGraph
    let mut world = WorldState::new();
    let mut frame_graph = FrameGraph::new();

    // Storage for observations (needed for GBA thread and LBA)
    let mut all_observations: Vec<Vec<StereoObservation>> = Vec::new();

    // Process first frame to initialize
    println!("🚀 Processing frame 0 for initialization...");
    let first_frame_num = frame_numbers[0];
    let left_path = image_dir.join(format!("Image{:04}_L.jpg", first_frame_num));
    let right_path = image_dir.join(format!("Image{:04}_R.jpg", first_frame_num));

    let left_img = image::open(&left_path)?.to_luma8();
    let right_img = image::open(&right_path)?.to_luma8();

    let tracks = tracker.process_frame(&left_img, &right_img);
    let frame0_obs = features_to_observations(&tracks, 0);
    println!("  Frame 0: {} tracks, {} with stereo", tracks.len(), frame0_obs.len());

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
    all_observations.push(frame0_obs.clone());

    println!(
        "  Initialized {} points from triangulation\n",
        world.num_points()
    );

    // Create SLAM system (spawns GBA thread)
    let frame_observations_arc = Arc::new(all_observations.clone());
    let mut slam_system = SlamSystem::new(stereo_camera.clone(), frame_observations_arc);

    // Visualize ground truth trajectory and tracked feature points
    if let Some(ref poses) = gt_poses {
        let _ = visualize_ground_truth(&rec, &gt_points_vec, poses, &stereo_camera);
    }

    // Visualize initial state
    visualize_estimate(
        &rec,
        0,
        &world,
        &frame_graph,
        &gt_points_vec,
        &stereo_camera,
        None,
    )?;

    slam_system.send_to_gba(0, &world);

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

            let _ = visualize_gba_update(
                &rec,
                gba_update_count,
                gba_world,
                &gba_result.frame_graph,
                &gt_points_vec,
                &stereo_camera,
                prev_gba_frame_graph.as_ref(),
            );

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

        // Load images
        let left_path = image_dir.join(format!("Image{:04}_L.jpg", frame_num));
        let right_path = image_dir.join(format!("Image{:04}_R.jpg", frame_num));

        let left_img = image::open(&left_path)?.to_luma8();
        let right_img = image::open(&right_path)?.to_luma8();

        // Track features
        let tracks = tracker.process_frame(&left_img, &right_img);
        let current_frame_obs = features_to_observations(&tracks, frame_idx);

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
                }
            }

            last_keyframe_position = current_position;
        }

        // Store observations
        all_observations.push(current_frame_obs.clone());

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

        // Update observations arc for LBA
        let frame_observations_arc = Arc::new(all_observations.clone());

        // Run LBA
        let result = run_bundle_adjustment(
            &stereo_camera,
            &frame_graph,
            &mut world,
            &frame_observations_arc,
            marginalized_prior.as_ref(),
            &fixed_point_ids,
            &BundleAdjustmentConfig::lba(),
        );
        let lba_time = result.solve_time_ms;
        total_lba_time += lba_time;

        // Update prior
        marginalized_prior = result.new_prior;
        for j in 0..frame_graph.len() {
            if frame_graph.states[j].state == OptimizationState::Marginalize {
                frame_graph.set_state(j, OptimizationState::Inactive);
            }
        }

        // Send to GBA
        slam_system.send_to_gba(frame_idx, &world);

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
        visualize_estimate(
            &rec,
            frame_idx,
            &world,
            &frame_graph,
            &gt_points_vec,
            &stereo_camera,
            prev_frame_graph.as_ref(),
        )?;
        prev_frame_graph = Some(frame_graph.clone());

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

        let with_stereo = tracks.iter().filter(|t| t.age == 0).count();

        println!(
            "Frame {}: {} tracks ({} stereo), {} opt, {} fixed, Map: {} pts, LBA: {:.2}ms{}{}",
            frame_idx,
            tracks.len(),
            with_stereo,
            n_optimized,
            n_fixed,
            world.num_points(),
            lba_time,
            if should_create_keyframe { " [KF]" } else { "" },
            error_str
        );
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
    println!("\n📺 Open Rerun to see the SLAM visualization!");

    drop(slam_system);

    Ok(())
}
