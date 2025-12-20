//! Incremental Stereo SLAM Demo with LBA/GBA
//!
//! Demonstrates parallel SLAM with:
//! - Local Bundle Adjustment (LBA) for real-time tracking
//! - Global Bundle Adjustment (GBA) running asynchronously  
//! - Triangulation to initialize new points
//! - Map that persists points even when not in window

use odysseus_slam::{
    camera::StereoCamera,
    geometry::{Point3D, StereoObservation},
    math::SE3,
    WorldState, SlamSystem,
    frame_graph::{FrameGraph, FrameRole, OptimizationState},
    simulation::{generate_stereo_observations, add_noise_to_stereo_observations},
    optimization::{run_bundle_adjustment, BundleAdjustmentConfig, MarginalizedPrior},
};
use rerun as rr;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::Arc;
use backtrace_on_stack_overflow;
use clap::Parser;

/// Incremental Stereo SLAM Demo
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Add noise to observations (default: 2.0 pixel stddev)
    #[arg(long, default_value_t = 0.0)]
    noise: f64,
}

// SLAM parameters
const WINDOW_SIZE: usize = 5;             // LBA window: last N frames

// Physical constants (meters)
const STEREO_BASELINE: f64 = 0.1;

/// Load point cloud from binary file
fn load_point_cloud(path: &str) -> Result<Vec<[f64; 3]>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read number of vertices
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let num_vertices = u32::from_le_bytes(buf) as usize;

    // Read vertices
    let mut vertices = Vec::with_capacity(num_vertices);
    for _ in 0..num_vertices {
        let mut x_buf = [0u8; 4];
        let mut y_buf = [0u8; 4];
        let mut z_buf = [0u8; 4];

        reader.read_exact(&mut x_buf)?;
        reader.read_exact(&mut y_buf)?;
        reader.read_exact(&mut z_buf)?;

        let x = f32::from_le_bytes(x_buf) as f64;
        let y = f32::from_le_bytes(y_buf) as f64;
        let z = f32::from_le_bytes(z_buf) as f64;

        vertices.push([x, y, z]);
    }

    Ok(vertices)
}

/// Load camera poses from binary file
fn load_camera_poses(path: &str) -> Result<Vec<SE3<f64>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read number of frames
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let num_frames = u32::from_le_bytes(buf) as usize;

    // Read 4x4 matrices (row-major)
    let mut poses = Vec::with_capacity(num_frames);
    for _ in 0..num_frames {
        let mut matrix = [[0.0f32; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                let mut val_buf = [0u8; 4];
                reader.read_exact(&mut val_buf)?;
                matrix[row][col] = f32::from_le_bytes(val_buf);
            }
        }

        // Convert 4x4 matrix to SE3
        // Extract rotation (top-left 3x3) as column vectors
        let x_axis = odysseus_solver::math3d::Vec3::new(
            matrix[0][0] as f64,
            matrix[1][0] as f64,
            matrix[2][0] as f64,
        );
        let y_axis = odysseus_solver::math3d::Vec3::new(
            matrix[0][1] as f64,
            matrix[1][1] as f64,
            matrix[2][1] as f64,
        );
        let z_axis = odysseus_solver::math3d::Vec3::new(
            matrix[0][2] as f64,
            matrix[1][2] as f64,
            matrix[2][2] as f64,
        );

        let rotation_matrix = odysseus_solver::math3d::Mat3::from_cols(x_axis, y_axis, z_axis);
        let rotation = odysseus_slam::math::SO3 { matrix: rotation_matrix };

        let translation = odysseus_solver::math3d::Vec3::new(
            matrix[0][3] as f64,
            matrix[1][3] as f64,
            matrix[2][3] as f64,
        );

        poses.push(SE3 { rotation, translation });
    }

    Ok(poses)
}

/// Get current resident set size (actual physical memory used) in MB
fn get_rss_mb() -> f64 {
    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

/// Get peak memory usage (high water mark) in MB
fn get_peak_rss_mb() -> f64 {
    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if line.starts_with("VmHWM:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

fn main() {
    let args = Args::parse();
    unsafe {
        let _ = backtrace_on_stack_overflow::enable(|| {run_slam(args.noise)});
    }
}

fn run_slam(noise_stddev: f64) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Incremental Stereo SLAM Demo (LBA/GBA)");
    println!("==========================================\n");
    println!("📊 Memory at startup: {:.1} MB", get_rss_mb());

    // Initialize Rerun
    let rec = rr::RecordingStreamBuilder::new("incremental_slam").spawn()?;

    // Configure view for OpenCV coordinates
    rec.log_static(
        "world",
        &rr::ViewCoordinates::RDF(),  // Right, Down, Forward (OpenCV)
    )?;

    // Stereo camera setup
    let focal_length = 500.0;
    let image_width = 640.0;
    let image_height = 480.0;
    let stereo_camera = StereoCamera::simple(focal_length, image_width, image_height, STEREO_BASELINE);

    println!("📷 Stereo Camera:");
    println!("  Focal length: {} px", focal_length);
    println!("  Baseline: {} m\n", STEREO_BASELINE);

    // Load GROUND TRUTH data from Blender export
    println!("🌍 Loading ground truth from greeble room...");
    let gt_points_raw = load_point_cloud("./blender_stuff/greeble_room/room_mesh.bin")?;
    let gt_poses = load_camera_poses("./blender_stuff/greeble_room/camera_poses.bin")?;

    // Convert points to Vec3 format
    let gt_points: Vec<_> = gt_points_raw.iter()
        .map(|p| odysseus_solver::math3d::Vec3::new(p[0], p[1], p[2]))
        .collect();

    println!("  {} 3D points from greeble room", gt_points.len());
    println!("  {} camera poses\n", gt_poses.len());
    
    println!("📊 Memory after loading data: {:.1} MB", get_rss_mb());

    // Generate ALL observations for all frames (simulating perfect tracking)
    println!("📹 Generating observations for all frames...");
    let perfect_observations = generate_stereo_observations(
        &gt_points,
        &gt_poses,
        &stereo_camera,
        image_width,
        image_height,
    );
    let observations = if noise_stddev > 0.0 {
        println!("  Adding noise with stddev = {} pixels", noise_stddev);
        add_noise_to_stereo_observations(&perfect_observations, noise_stddev, 123)
    } else {
        println!("  Using perfect observations (no noise)");
        perfect_observations
    };
    println!("  {} total stereo observations\n", observations.len());

    // Pre-organize observations by frame for sharing with GBA thread
    let total_frames = gt_poses.len();
    let frame_observations: Vec<Vec<StereoObservation>> = (0..total_frames)
        .map(|frame_idx| {
            observations
                .iter()
                .filter(|obs| obs.camera_id == frame_idx)
                .cloned()
                .collect()
        })
        .collect();
    let frame_observations_arc = Arc::new(frame_observations);
    
    // Create SLAM system (spawns GBA thread with shared observations)
    let mut slam_system = SlamSystem::new(stereo_camera.clone(), frame_observations_arc.clone());
    println!("🔧 SLAM System initialized (GBA thread spawned)\n");

    // Create WorldState (poses + map) and FrameGraph (state metadata)
    let mut world = WorldState::new();
    let mut frame_graph = FrameGraph::new();

    // Initialize SLAM from FIRST FRAME
    println!("🚀 Initializing SLAM from frame 0...");

    // Add frame 0 as a keyframe first (needed for storing points)
    frame_graph.add_frame(FrameRole::Keyframe, OptimizationState::Fixed);
    world.add_pose(gt_poses[0]);

    // Triangulate initial points from first frame observations
    let frame0_obs = &frame_observations_arc[0];

    println!("  Frame 0 has {} observations", frame0_obs.len());
    println!("  Triangulating initial points...");

    // Triangulate and store points directly in inverse depth form on keyframe 0
    for obs in frame0_obs.iter() {
        world.triangulate_and_add_point(obs, &stereo_camera, 0);
    }

    println!("  Initialized {} points from triangulation\n", world.num_points());
    println!("  Created initial keyframe from frame 0 (Fixed)\n");
    slam_system.send_to_gba(0, &world);

    // Visualize ground truth (static, shown in all timelines)
    let _ = visualize_ground_truth(&rec, &gt_points, &gt_poses)?;

    // Visualize initial state on trajectory timeline
    visualize_estimate(&rec, 0, &world, &frame_graph, &gt_points, &stereo_camera, &vec![(OptimizationState::Fixed, FrameRole::Keyframe)])?;
    
    println!("📊 Memory before frame processing: {:.1} MB", get_rss_mb());

    // Process frames sequentially
    let total_frames = gt_poses.len();
    // various tracking variables
    let mut total_frame_time = 0.0;
    let mut total_lba_time = 0.0;
    let mut gba_update_count = 0;
    let mut gba_last_optimized_frame = 0;
    let mut gba_states_before: Vec<(OptimizationState, FrameRole)> = Vec::new();
    let mut marginalized_prior: Option<MarginalizedPrior> = None;

    // MAIN LOOP
    for frame_idx in 1..total_frames {
        let frame_start = std::time::Instant::now();
        
        // Check for GBA results (non-blocking) and merge into world state
        if let Some(gba_result) = slam_system.try_recv_from_gba() {
            // Replace frames GBA optimized (it always has fewer frames than us)
            let gba_world = &gba_result.world_state;
            let n_gba_frames = gba_world.frames.len();
            gba_update_count += 1;

            // Visualize GBA result before merging (only updates changed cameras)
            let _ = visualize_gba_update(&rec, gba_update_count, gba_world, &gba_result.frame_graph, &gt_points, &stereo_camera, &gba_states_before);

            // Update GBA state tracking for next visualization
            gba_states_before = gba_result.frame_graph.states.iter()
                .map(|s| (s.state, s.role))
                .collect();

            world.replace_frames_from(gba_world);
            gba_last_optimized_frame = gba_result.last_optimized_frame;
            println!("  📥 Received GBA update #{} (frame {}, {} poses, {} points)",
                gba_update_count, gba_result.last_optimized_frame,
                n_gba_frames, gba_world.num_points());
        }
        
        // GBA usually optimizes extra frames that are more reliable than frames from LBA and could be used to gauge fix.
        if let Some(frame_state) = frame_graph.get(gba_last_optimized_frame) {
            if frame_state.role != FrameRole::Keyframe {
                frame_graph.set_role(gba_last_optimized_frame, FrameRole::Stored);
            }
        }


        // Memory checkpoint every 10 frames
        if frame_idx % 10 == 0 {
            println!("📊 Memory at frame {}: {:.1} MB (peak: {:.1} MB)", 
                     frame_idx, get_rss_mb(), get_peak_rss_mb());
        }

        // Get last estimated pose for initial guess
        let last_pose = *world.get_pose(frame_idx - 1).unwrap();

        // Get observations for current frame
        let current_frame_obs = &frame_observations_arc[frame_idx];

        // Determine if this should be a keyframe
        let new_points_count = current_frame_obs.iter()
            .filter(|obs| world.get_point(obs.point_id).is_none())
            .count();
        let novelty_ratio = if current_frame_obs.is_empty() {
            0.0
        } else {
            new_points_count as f64 / current_frame_obs.len() as f64
        };

        let should_create_keyframe = novelty_ratio >= 0.3;
        // LBA uses keyframes to figure out what points to gauge fix. Storing this information in the frame graph is slightly redundant with world state.
        let frame_role = if should_create_keyframe { FrameRole::Keyframe } else { FrameRole::Transient };
        
        // Capture frame states BEFORE any changes for visualization cleanup
        let states_before: Vec<(OptimizationState, FrameRole)> = frame_graph.states.iter()
            .map(|s| (s.state, s.role))
            .collect();
        
        // Add frame to world first (needed for storing points on keyframes)
        world.add_pose(last_pose);

        // Triangulate new points if this is a keyframe
        if should_create_keyframe {
            println!("  Creating keyframe from frame {} (novelty: {:.1}%)", frame_idx, novelty_ratio * 100.0);

            for obs in current_frame_obs.iter() {
                if world.get_point(obs.point_id).is_none() {
                    // Triangulate and store directly in inverse depth form on this keyframe
                    world.triangulate_and_add_point(obs, &stereo_camera, frame_idx);
                }
            }
        }
        let obs_count = current_frame_obs.len();

        // Add frame to graph as optimized
        frame_graph.add_frame(frame_role, OptimizationState::Optimized);
        
        // Manage window: mark old frames for marginalization
        let mut optimized_indices: Vec<usize> = frame_graph.states.iter().enumerate()
            .filter(|(_, s)| s.state == OptimizationState::Optimized)
            .map(|(idx, _)| idx)
            .collect();

        // Mark oldest frames for marginalization when window is full
        while optimized_indices.len() > WINDOW_SIZE {
            frame_graph.set_state(optimized_indices[0], OptimizationState::Marginalize);
            optimized_indices.remove(0);
        }

            
        // Fix frame 0 and all GBA-optimized keyframes
        for i in 0..frame_graph.len() {
            let is_gba_optimized = i <= gba_last_optimized_frame;
            let is_keyframe = frame_graph.states[i].role == FrameRole::Keyframe;
            
            if i == 0 || (is_keyframe && is_gba_optimized) {
                frame_graph.set_state(i, OptimizationState::Fixed);
            }
        }

        // If the last GBA-optimized frame is not a keyframe, include it as Fixed
        // This bridges the fixed GBA keyframes with the current LBA window
        if gba_last_optimized_frame > 0 {
            let gba_frame_role = frame_graph.states[gba_last_optimized_frame].role;
            if gba_frame_role != FrameRole::Keyframe {
                frame_graph.set_state(gba_last_optimized_frame, OptimizationState::Optimized);
            }
        }

        // Collect fixed point IDs: points whose reference keyframe has been GBA-optimized
        let fixed_point_ids: std::collections::HashSet<usize> = world
            .get_all_points()
            .iter()
            .filter_map(|(point_id, _)| {
                world.get_point_keyframe(*point_id)
                    .filter(|&kf| kf <= gba_last_optimized_frame)
                    .map(|_| *point_id)
            })
            .collect();

        // Run LBA optimization with marginalized prior
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

        // Update prior from result and mark marginalized frames as inactive
        marginalized_prior = result.new_prior;
        for i in 0..frame_graph.len() {
            if frame_graph.states[i].state == OptimizationState::Marginalize {
                frame_graph.set_state(i, OptimizationState::Inactive);
            }
        }

        // Send just this frame to GBA (incremental update)
        slam_system.send_to_gba(frame_idx, &world);

        // Get optimized pose for current frame
        let optimized_pose = world.frames[frame_idx].pose;

        let pos_error = (optimized_pose.translation - gt_poses[frame_idx].translation).norm();

        // Compute rotation error
        let r_error = gt_poses[frame_idx].rotation.matrix.transpose() * optimized_pose.rotation.matrix;
        let trace = r_error.m00() + r_error.m11() + r_error.m22();
        let angle_rad = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        let angle_deg = angle_rad.to_degrees();

        // Warn if error exceeds thresholds
        const MAX_POSITION_ERROR: f64 = 0.5;
        const MAX_ROTATION_ERROR: f64 = 10.0;
        if pos_error > MAX_POSITION_ERROR || angle_deg > MAX_ROTATION_ERROR {
            eprintln!("\n❌ ERROR: Pose error exceeded thresholds at frame {}!", frame_idx);
            eprintln!("  Position error: {:.4} m (max: {:.4} m)", pos_error, MAX_POSITION_ERROR);
            eprintln!("  Rotation error: {:.4} deg (max: {:.4} deg)", angle_deg, MAX_ROTATION_ERROR);
            eprintln!("  Frame {} had {} observations", frame_idx, obs_count);
        }

        // Check map points for catastrophic errors
        for (point_id, est_point) in world.get_all_points() {
            if point_id < gt_points.len() {
                let gt = gt_points[point_id];
                let error = (est_point - gt).norm();
                if error > 4.0 {
                    eprintln!("\n❌ ERROR: Point {} has catastrophic error: {:.4} m", point_id, error);
                    std::process::abort();
                }
            }
        }

        // Visualize final state
        visualize_estimate(&rec, frame_idx, &world, &frame_graph, &gt_points, &stereo_camera, &states_before)?;
        let frame_duration = frame_start.elapsed();
        total_frame_time += frame_duration.as_secs_f64() * 1000.0;

        // Count active frames for display
        let n_optimized = frame_graph.states.iter().filter(|s| s.state == OptimizationState::Optimized).count();
        let n_fixed = frame_graph.states.iter().filter(|s| s.state == OptimizationState::Fixed).count();

        println!("Frame {}: {} opt, {} fixed, {} obs, Map: {} pts, LBA: {:.2} ms, Total: {:.2} ms{}",
            frame_idx, n_optimized, n_fixed, obs_count, world.num_points(),
            lba_time, frame_duration.as_secs_f64() * 1000.0,
            if should_create_keyframe { " [KF]" } else { "" });
    }

    println!("\n✅ Processed {} frames", total_frames);
    println!("   Final map: {} points", world.num_points());
    println!("   GBA updates received: {}", gba_update_count);
    println!("\n📊 Final memory: {:.1} MB, Peak: {:.1} MB", get_rss_mb(), get_peak_rss_mb());
    println!("\n📺 Open Rerun to see the SLAM visualization!");
    println!("Average LBA time: {:.2} ms", total_lba_time / total_frames as f64);
    println!("Average frame time: {:.2} ms", total_frame_time / total_frames as f64);
    
    // Let GBA finish any pending work
    drop(slam_system);
    
    Ok(())
}

/// Visualize ground truth trajectory and map
fn visualize_ground_truth(
    rec: &rr::RecordingStream,
    points: &[odysseus_solver::math3d::Vec3<f64>],
    poses: &[SE3<f64>],
) -> Result<(), Box<dyn std::error::Error>> {
    // Set trajectory timeline so ground truth appears on it
    rec.set_time_sequence("trajectory", 0);

    // Ground truth points (gray)
    let point_positions: Vec<[f32; 3]> = points
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();

    rec.log(
        "world/ground_truth/points",
        &rr::Points3D::new(point_positions)
            .with_colors([[120, 120, 120]])
            .with_radii([0.03]),
    )?;

    // Ground truth trajectory (gray line)
    let trajectory: Vec<[f32; 3]> = poses
        .iter()
        .map(|pose| {
            let t = &pose.translation;
            [t.x as f32, t.y as f32, t.z as f32]
        })
        .collect();

    rec.log(
        "world/ground_truth/trajectory",
        &rr::LineStrips3D::new([trajectory])
            .with_colors([[100, 100, 100]])
            .with_radii([0.01]),
    )?;

    // Camera frustums for ground truth (subsample every 5 frames)
    let camera = StereoCamera::new(
        odysseus_slam::camera::PinholeCamera::new(500.0, 500.0, 320.0, 240.0),
        0.1,
    );
    for (i, pose) in poses.iter().enumerate().step_by(1) {
        visualize_stereo_camera(
            rec,
            &format!("world/ground_truth/cameras/cam_{}", i),
            pose,
            &camera,
            [100, 100, 100, 255],  // Gray for GT
        )?;
    }

    Ok(())
}

/// Get the camera path prefix for a given frame state and role
fn camera_path_prefix(state: OptimizationState, role: FrameRole) -> &'static str {
    match (state, role) {
        (OptimizationState::Fixed, _) => "world/estimate/fixed_cameras",
        (OptimizationState::Optimized, FrameRole::Keyframe) => "world/estimate/keyframe_cameras",
        (OptimizationState::Optimized, FrameRole::Transient) => "world/estimate/window_cameras",
        (OptimizationState::Optimized, FrameRole::Stored) => "world/estimate/window_cameras",
        (OptimizationState::Marginalize, FrameRole::Keyframe) => "world/estimate/keyframe_cameras",
        (OptimizationState::Marginalize, FrameRole::Transient) => "world/estimate/window_cameras",
        (OptimizationState::Marginalize, FrameRole::Stored) => "world/estimate/window_cameras",
        (OptimizationState::Inactive, FrameRole::Stored) => "world/estimate/stored_cameras",
        (OptimizationState::Inactive, _) => "world/estimate/marginalized_cameras",
    }
}

/// Visualize estimated trajectory and map on the "trajectory" timeline
fn visualize_estimate(
    rec: &rr::RecordingStream,
    frame_idx: usize,
    world: &WorldState,
    frame_graph: &FrameGraph,
    gt_points: &[Point3D<f64>],
    stereo_camera: &StereoCamera<f64>,
    states_before: &Vec<(OptimizationState, FrameRole)>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Set trajectory timeline
    rec.set_time_sequence("trajectory", frame_idx as i64);

    // Delete cameras that changed state (delete from old path before logging to new path)
    for idx in 0..(frame_idx+1) {
        let frame_state = &frame_graph.states[idx];
        let new_state = frame_state.state;
        let new_role = frame_state.role;
        let path_prefix = camera_path_prefix(new_state, new_role);
        
        // Determine if we need to update visualization (state changed or new frame)
        let should_update = if idx < states_before.len() {
            if (states_before[idx].0, states_before[idx].1) != (new_state, new_role) {
                // Delete from old path
                let old_prefix = camera_path_prefix(states_before[idx].0, states_before[idx].1);
                let old_path = format!("{}/cam_{:03}", old_prefix, idx);
                rec.log(old_path.as_str(), &rr::Clear::recursive())?;
                true
            } else {
                false
            }
        } else {
            true
        };

        if should_update {
            let color = match frame_state.role {
                FrameRole::Keyframe => [0, 255, 0, 255],  // Bright green for all keyframes
                FrameRole::Transient | FrameRole::Stored => match frame_state.state {
                    OptimizationState::Fixed => [255, 100, 100, 255],  // Red for fixed
                    OptimizationState::Optimized => [100, 200, 255, 255],  // Light blue for optimized transient
                    OptimizationState::Marginalize => [100, 200, 255, 255],  // Light blue for marginalize (same as Optimized)
                    OptimizationState::Inactive => [50, 80, 120, 255],  // Dark blue for inactive
                },
            };
            visualize_stereo_camera(
                rec,
                &format!("{}/cam_{:03}", path_prefix, idx),
                &world.frames[idx].pose,
                stereo_camera,
                color,
            )?;
        }
    }

    // Estimated points (blue)
    let all_points = world.get_all_points();
    let point_positions: Vec<[f32; 3]> = all_points
        .iter()
        .map(|(_, p)| [p.x as f32, p.y as f32, p.z as f32])
        .collect();

    rec.log(
        "world/estimate/points",
        &rr::Points3D::new(point_positions)
            .with_colors([[50, 150, 255]])
            .with_radii([0.04]),
    )?;

    // Draw error lines between estimated and GT points
    let mut error_lines = Vec::new();
    for (point_id, est_point) in &all_points {
        if *point_id < gt_points.len() {
            let gt = &gt_points[*point_id];
            error_lines.push(vec![
                [est_point.x as f32, est_point.y as f32, est_point.z as f32],
                [gt.x as f32, gt.y as f32, gt.z as f32],
            ]);
        }
    }

    if !error_lines.is_empty() {
        rec.log(
            "world/estimate/point_errors",
            &rr::LineStrips3D::new(error_lines)
                .with_colors([[255, 100, 100]])
                .with_radii([0.005]),
        )?;
    }

    // Estimated trajectory (blue line) - collect poses
    let trajectory: Vec<[f32; 3]> = world.frames.iter()
        .map(|frame| {
            let t = &frame.pose.translation;
            [t.x as f32, t.y as f32, t.z as f32]
        })
        .collect();

    rec.log(
        "world/estimate/trajectory",
        &rr::LineStrips3D::new([trajectory])
            .with_colors([[50, 150, 255]])
            .with_radii([0.02]),
    )?;

    Ok(())
}

/// Get the camera path prefix for GBA visualization (mirrors LBA's camera_path_prefix)
fn gba_camera_path_prefix(state: OptimizationState, role: FrameRole) -> &'static str {
    match (state, role) {
        (OptimizationState::Fixed, _) => "world/gba_estimate/fixed_cameras",
        (OptimizationState::Optimized, FrameRole::Keyframe) => "world/gba_estimate/keyframe_cameras",
        (OptimizationState::Optimized, FrameRole::Transient) => "world/gba_estimate/window_cameras",
        (OptimizationState::Optimized, FrameRole::Stored) => "world/gba_estimate/window_cameras",
        (OptimizationState::Marginalize, FrameRole::Keyframe) => "world/gba_estimate/keyframe_cameras",
        (OptimizationState::Marginalize, FrameRole::Transient) => "world/gba_estimate/window_cameras",
        (OptimizationState::Marginalize, FrameRole::Stored) => "world/gba_estimate/window_cameras",
        (OptimizationState::Inactive, FrameRole::Stored) => "world/gba_estimate/stored_cameras",
        (OptimizationState::Inactive, _) => "world/gba_estimate/marginalized_cameras",
    }
}

/// Visualize GBA results on a separate timeline and entity paths
/// Only updates cameras whose state changed (like visualize_estimate)
fn visualize_gba_update(
    rec: &rr::RecordingStream,
    gba_update_count: usize,
    gba_world: &WorldState,
    gba_frame_graph: &FrameGraph,
    gt_points: &[Point3D<f64>],
    stereo_camera: &StereoCamera<f64>,
    gba_states_before: &[(OptimizationState, FrameRole)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Use a separate timeline for GBA updates
    rec.set_time_sequence("gba_updates", gba_update_count as i64);

    // GBA trajectory (orange line)
    let trajectory: Vec<[f32; 3]> = gba_world.frames.iter()
        .map(|frame| {
            let t = &frame.pose.translation;
            [t.x as f32, t.y as f32, t.z as f32]
        })
        .collect();

    rec.log(
        "world/gba_estimate/trajectory",
        &rr::LineStrips3D::new([trajectory])
            .with_colors([[255, 165, 0]])  // Orange
            .with_radii([0.025]),
    )?;

    // GBA points (orange)
    let all_points = gba_world.get_all_points();
    let point_positions: Vec<[f32; 3]> = all_points
        .iter()
        .map(|(_, p)| [p.x as f32, p.y as f32, p.z as f32])
        .collect();

    rec.log(
        "world/gba_estimate/points",
        &rr::Points3D::new(point_positions.clone())
            .with_colors([[255, 165, 0]])  // Orange
            .with_radii([0.035]),
    )?;

    // GBA cameras - only update those whose state changed (like visualize_estimate)
    for (idx, frame) in gba_world.frames.iter().enumerate() {
        let frame_state = &gba_frame_graph.states[idx];
        let new_state = frame_state.state;
        let new_role = frame_state.role;
        let path_prefix = gba_camera_path_prefix(new_state, new_role);

        // Determine if we need to update visualization (state changed or new frame)
        let should_update = if idx < gba_states_before.len() {
            if (gba_states_before[idx].0, gba_states_before[idx].1) != (new_state, new_role) {
                // Delete from old path
                let old_prefix = gba_camera_path_prefix(gba_states_before[idx].0, gba_states_before[idx].1);
                let old_path = format!("{}/cam_{:03}", old_prefix, idx);
                rec.log(old_path.as_str(), &rr::Clear::recursive())?;
                true
            } else {
                false
            }
        } else {
            true  // New frame, always update
        };

        if should_update {
            let color = match frame_state.role {
                FrameRole::Keyframe => [0, 255, 0, 255],  // Bright green for keyframes
                FrameRole::Transient | FrameRole::Stored => match frame_state.state {
                    OptimizationState::Fixed => [255, 100, 100, 255],  // Red for fixed
                    OptimizationState::Optimized => [100, 200, 255, 255],  // Light blue for optimized
                    OptimizationState::Marginalize => [100, 200, 255, 255],  // Light blue for marginalize (same as Optimized)
                    OptimizationState::Inactive => [50, 80, 120, 255],  // Dark blue for inactive
                },
            };

            visualize_stereo_camera(
                rec,
                &format!("{}/cam_{:03}", path_prefix, idx),
                &frame.pose,
                stereo_camera,
                color,
            )?;
        }
    }

    // Draw error lines between GBA points and GT points
    let mut error_lines = Vec::new();
    for (point_id, est_point) in &all_points {
        if *point_id < gt_points.len() {
            let gt = &gt_points[*point_id];
            error_lines.push(vec![
                [est_point.x as f32, est_point.y as f32, est_point.z as f32],
                [gt.x as f32, gt.y as f32, gt.z as f32],
            ]);
        }
    }

    if !error_lines.is_empty() {
        rec.log(
            "world/gba_estimate/point_errors",
            &rr::LineStrips3D::new(error_lines)
                .with_colors([[255, 100, 50]])
                .with_radii([0.004]),
        )?;
    }

    Ok(())
}

/// Visualize optimization convergence on per-frame timeline
fn _visualize_optimization_step(
    rec: &rr::RecordingStream,
    frame_idx: usize,
    iteration: usize,
    error: f64,
    world: &WorldState,
    window_start: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Set per-frame optimization timeline
    rec.set_time_sequence(format!("optimization_frame_{}", frame_idx), iteration as i64);

    // Log error as scalar (separate from world/ hierarchy)
    rec.log(
        format!("optimization_view/frame_{}/error", frame_idx),
        &rr::Scalars::new([error]),
    )?;

    // Show current window points during optimization (orange)
    // Use separate entity path so it doesn't appear on trajectory timeline
    let all_points = world.get_all_points();
    let point_positions: Vec<[f32; 3]> = all_points
        .iter()
        .map(|(_, p)| [p.x as f32, p.y as f32, p.z as f32])
        .collect();

    rec.log(
        format!("optimization_view/frame_{}/points", frame_idx),
        &rr::Points3D::new(point_positions)
            .with_colors([[255, 150, 50]])
            .with_radii([0.035]),
    )?;

    // Show window trajectory during optimization (orange)
    let window_trajectory: Vec<[f32; 3]> = world.frames[window_start..=frame_idx]
        .iter()
        .map(|frame| {
            let t = &frame.pose.translation;
            [t.x as f32, t.y as f32, t.z as f32]
        })
        .collect();

    rec.log(
        format!("optimization_view/frame_{}/trajectory", frame_idx),
        &rr::LineStrips3D::new([window_trajectory])
            .with_colors([[255, 150, 50]])
            .with_radii([0.015]),
    )?;

    Ok(())
}

/// Visualize stereo camera using Rerun's native Pinhole
fn visualize_stereo_camera(
    rec: &rr::RecordingStream,
    entity_path: &str,
    pose: &SE3<f64>,
    stereo_camera: &StereoCamera<f64>,
    color: [u8; 4],  // RGBA color for the camera
) -> Result<(), Box<dyn std::error::Error>> {
    // Convert rotation matrix to 3x3 array

    let camera_rot = pose.rotation.matrix; //.inverse();

    let camera_rot_for_rerun = camera_rot.inverse();
    let rot_matrix = [
        [camera_rot_for_rerun.m00() as f32, camera_rot_for_rerun.m01() as f32, camera_rot_for_rerun.m02() as f32],
        [camera_rot_for_rerun.m10() as f32, camera_rot_for_rerun.m11() as f32, camera_rot_for_rerun.m12() as f32],
        [camera_rot_for_rerun.m20() as f32, camera_rot_for_rerun.m21() as f32, camera_rot_for_rerun.m22() as f32],
    ];

    // Left camera
    let left_path = format!("{}/left", entity_path);
    rec.log(
        left_path.as_str(),
        &rr::Transform3D::from_translation_mat3x3(
            [pose.translation.x as f32, pose.translation.y as f32, pose.translation.z as f32],
            rot_matrix,
        ),
    )?;

    // Log pinhole for left camera
    rec.log(
        left_path.as_str(),
        &rr::Pinhole::from_focal_length_and_resolution(
            [stereo_camera.left.fx as f32, stereo_camera.left.fy as f32],
            [640.0, 480.0],  // Resolution
        )
        .with_principal_point([stereo_camera.left.cx as f32, stereo_camera.left.cy as f32])
        .with_camera_xyz(rr::components::ViewCoordinates::RDF)
        .with_image_plane_distance(0.03)
        .with_color(color),
    )?;

    // Right camera (offset by baseline along X-axis in camera frame)
    let right_offset_world = camera_rot.mul_vec(odysseus_solver::math3d::Vec3::new(
        stereo_camera.baseline,
        0.0,
        0.0,
    ));
    let right_translation = pose.translation + right_offset_world;

    let right_path = format!("{}/right", entity_path);
    rec.log(
        right_path.as_str(),
        &rr::Transform3D::from_translation_mat3x3(
            [right_translation.x as f32, right_translation.y as f32, right_translation.z as f32],
            rot_matrix,
        ),
    )?;

    // Log pinhole for right camera (same intrinsics)
    rec.log(
        right_path.as_str(),
        &rr::Pinhole::from_focal_length_and_resolution(
            [stereo_camera.left.fx as f32, stereo_camera.left.fy as f32],
            [640.0, 480.0],
        )
        .with_principal_point([stereo_camera.left.cx as f32, stereo_camera.left.cy as f32])
        .with_camera_xyz(rr::components::ViewCoordinates::RDF)
        .with_image_plane_distance(0.03)
        .with_color(color),
    )?;

    Ok(())
}

/// OLD: Visualize a camera frustum (keeping for now, will remove later)
fn _visualize_camera_frustum(
    rec: &rr::RecordingStream,
    entity_path: String,
    pose: &SE3<f64>,
    color: [u8; 3],
    size: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let t = &pose.translation;
    let position = [t.x as f32, t.y as f32, t.z as f32];

    // Transform points to world frame
    let corners_camera = [
        [-size, -size, 2.0 * size],  // Bottom-left
        [size, -size, 2.0 * size],   // Bottom-right
        [size, size, 2.0 * size],    // Top-right
        [-size, size, 2.0 * size],   // Top-left
    ];

    let corners_world: Vec<[f32; 3]> = corners_camera
        .iter()
        .map(|&c| {
            let p = pose.transform_point(odysseus_solver::math3d::Vec3::new(c[0] as f64, c[1] as f64, c[2] as f64));
            [p.x as f32, p.y as f32, p.z as f32]
        })
        .collect();

    // Draw frustum lines
    let lines = vec![
        // From camera center to corners
        vec![position, corners_world[0]],
        vec![position, corners_world[1]],
        vec![position, corners_world[2]],
        vec![position, corners_world[3]],
        // Rectangle at image plane
        vec![corners_world[0], corners_world[1]],
        vec![corners_world[1], corners_world[2]],
        vec![corners_world[2], corners_world[3]],
        vec![corners_world[3], corners_world[0]],
    ];

    rec.log(
        entity_path,
        &rr::LineStrips3D::new(lines)
            .with_colors([color])
            .with_radii([0.005]),
    )?;

    Ok(())
}
