//! Incremental Stereo SLAM Demo with LBA/GBA
//!
//! Demonstrates parallel SLAM with:
//! - Local Bundle Adjustment (LBA) for real-time tracking
//! - Global Bundle Adjustment (GBA) running asynchronously  
//! - Triangulation to initialize new points
//! - Map that persists points even when not in window

use odysseus_slam::{
    camera::StereoCamera,
    geometry::StereoObservation,
    WorldState, SlamSystem,
    frame_graph::{FrameGraph, FrameRole, OptimizationState},
    simulation::{generate_stereo_observations, add_noise_to_stereo_observations},
    optimization::{run_bundle_adjustment, BundleAdjustmentConfig, MarginalizedPrior},
    utils::{get_rss_mb, get_peak_rss_mb, load_point_cloud, load_camera_poses},
    visualization::{visualize_ground_truth, visualize_estimate, visualize_gba_update},
};
use rerun as rr;
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
    let _ = visualize_ground_truth(&rec, &gt_points, &gt_poses, &stereo_camera)?;

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
        let last_pose = world.get_pose(frame_idx - 1).unwrap();

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
        let optimized_pose = world.frames[frame_idx].world_pose();

        let pos_error = (optimized_pose.translation - gt_poses[frame_idx].translation).norm();

        // Compute rotation error using quaternions
        // The rotation error quaternion q_err = q1^(-1) * q2
        // For unit quaternions, angle = 2 * acos(|w|) where w is the scalar part
        let q_err = gt_poses[frame_idx].rotation.inverse().quat * optimized_pose.rotation.quat;
        let angle_rad = 2.0 * q_err.w.abs().acos();
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
