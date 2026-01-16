//! Incremental Visual-Inertial Odometry (VIO) Demo
//!
//! Demonstrates incremental tightly-coupled VIO optimization with:
//! - Sliding window bundle adjustment with IMU constraints
//! - Visual reprojection residuals from stereo observations
//! - IMU preintegration residuals from high-rate IMU data
//! - Bias random walk constraints
//!
//! The demo uses a smooth Bezier spline trajectory exported from Blender
//! to generate ground truth poses, velocities, and simulated IMU measurements.

use backtrace_on_stack_overflow;
use clap::Parser;
use nalgebra::Vector3;
use odysseus_slam::{
    camera::StereoCamera,
    frame_graph::{FrameGraph, FrameRole, OptimizationState},
    geometry::{Point3D, StereoObservation},
    imu::{simulator::ImuNoiseParams, ImuFrameState, ImuSimulator, PreintegratedImu},
    math::SE3,
    optimization::{run_simple_vio_bundle_adjustment, VioConfig, VioMarginalizedPrior},
    simulation::{add_noise_to_stereo_observations, generate_stereo_observations},
    spline::BezierSplineTrajectory,
    trajectory::ContinuousTrajectory,
    utils::{get_peak_rss_mb, get_rss_mb, load_point_cloud},
    visualization::{visualize_estimate, visualize_ground_truth},WorldState,
};
use rerun as rr;
use std::sync::Arc;

/// VIO Demo with synthetic data
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Add noise to observations (default: 0.5 pixel stddev)
    #[arg(long, default_value_t = 0.5)]
    noise: f64,
}

// VIO Parameters
const IMU_RATE: f64 = 200.0; // Hz
const CAMERA_RATE: f64 = 30.0; // Hz
const STEREO_BASELINE: f64 = 0.1;
const WINDOW_SIZE: usize = 2; // VIO window size

// Noise parameters
const ACCEL_NOISE: f64 = 0.01; // m/s^2 / sqrt(Hz)
const GYRO_NOISE: f64 = 0.001; // rad/s / sqrt(Hz)

// Keyframe selection: require baseline >= BASELINE_RATIO * median_depth
const BASELINE_RATIO: f64 = 0.1;

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

fn main() {
    let args = Args::parse();
    unsafe {
        let _ = backtrace_on_stack_overflow::enable(|| {
            if let Err(e) = run_vio(args.noise) {
                eprintln!("Error: {}", e);
            }
        });
    }
}

fn run_vio(noise_stddev: f64) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Incremental Visual-Inertial Odometry (VIO) Demo");
    println!("   Observation Noise: {:.2} pixels", noise_stddev);
    println!("   Window Size: {} frames", WINDOW_SIZE);
    println!("=============================================\n");
    println!("📊 Memory at startup: {:.1} MB", get_rss_mb());

    // Initialize Rerun
    let rec = rr::RecordingStreamBuilder::new("vio_demo").spawn()?;
    rec.log_static("world", &rr::ViewCoordinates::RDF())?;

    // Stereo camera setup
    let focal_length = 500.0;
    let image_width = 640.0;
    let image_height = 480.0;
    let stereo_camera =
        StereoCamera::simple(focal_length, image_width, image_height, STEREO_BASELINE);

    println!("📷 Stereo Camera:");
    println!("  Focal length: {} px", focal_length);
    println!("  Baseline: {} m\n", STEREO_BASELINE);

    // Load ground truth spline trajectory
    println!("📈 Loading Bezier spline from Blender export...");
    let spline_path = "blender_stuff/greeble_room/camera_spline.bin";
    let trajectory = BezierSplineTrajectory::load(spline_path)?;
    let duration = trajectory.duration;
    println!(
        "   Spline loaded successfully ({:.2}s duration).\n",
        duration
    );

    // Sample ground truth IMU measurements from spline trajectory
    println!("🧪 Simulating IMU data at {} Hz...", IMU_RATE);
    let simulator = ImuSimulator::new(
        ImuNoiseParams {
            gyro_noise_density: GYRO_NOISE,
            accel_noise_density: ACCEL_NOISE,
            gyro_bias_random_walk: 0.0,
            accel_bias_random_walk: 0.0,
        },
        IMU_RATE,
    );

    let imu_measurements = simulator.generate_from_continuous_trajectory(&trajectory, duration, 42);
    println!("   Generated {} IMU measurements.", imu_measurements.len());

    // Sample ground truth camera poses from spline trajectory
    println!("📷 Sampling camera frames at {} Hz...", CAMERA_RATE);
    let mut gt_poses = Vec::new();
    let mut gt_velocities = Vec::new();
    let mut timestamps = Vec::new();

    let dt_cam = 1.0 / CAMERA_RATE;
    let mut t = 0.0;
    while t <= duration {
        gt_poses.push(trajectory.pose(t));
        gt_velocities.push(trajectory.linear_velocity(t));
        timestamps.push(t);
        t += dt_cam;
    }
    let n_frames = gt_poses.len();

    // Load points for observations
    let points_path = "blender_stuff/greeble_room/room_mesh.bin";
    let gt_points_raw = load_point_cloud(points_path)?;
    let gt_points: Vec<Point3D<f64>> = gt_points_raw
        .iter()
        .map(|p| Point3D {
            x: p[0],
            y: p[1],
            z: p[2],
        })
        .collect();
    // Also keep Vec3 version for observation generation
    let gt_points_vec3: Vec<_> = gt_points_raw
        .iter()
        .map(|p| odysseus_solver::math3d::Vec3::new(p[0], p[1], p[2]))
        .collect();

    // Generate ALL observations for all frames
    println!("📹 Generating observations for all frames...");
    let perfect_observations = generate_stereo_observations(
        &gt_points_vec3,
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

    let mut frame_observations: Vec<Vec<StereoObservation>> = vec![Vec::new(); n_frames];
    for obs in observations {
        frame_observations[obs.camera_id].push(obs);
    }
    // Wrap in Arc for sharing with GBA thread
    let frame_observations = Arc::new(frame_observations);

    println!(
        "  {} total stereo observations\n",
        frame_observations.iter().map(|f| f.len()).sum::<usize>()
    );

    // Visualize ground truth
    visualize_ground_truth(&rec, &gt_points, &gt_poses, &stereo_camera)?;

    // Initialize SLAM state
    println!("🏗️  Initializing VIO state...");
    let mut world = WorldState::new();
    let mut frame_graph = FrameGraph::new();
    let mut imu_states: Vec<ImuFrameState> = Vec::new();
    let mut preintegrations: Vec<PreintegratedImu> = Vec::new();

    // Initialize from first frame (fixed)
    println!("🚀 Initializing from frame 0...");
    world.add_pose(gt_poses[0]);
    imu_states.push(ImuFrameState::with_velocity(gt_velocities[0]));
    frame_graph.add_frame(FrameRole::Keyframe, OptimizationState::Fixed);

    // Triangulate initial points from first frame observations
    for obs in &frame_observations[0] {
        world.triangulate_and_add_point(obs, &stereo_camera, 0);
    }
    println!(
        "  Initialized {} points from triangulation\n",
        world.num_points()
    );
    
    // Simple marginalization prior (accumulates as frames are marginalized)
    let mut accumulated_prior: Option<VioMarginalizedPrior> = None;
    let mut last_keyframe_idx: usize = 0; // Track the most recent keyframe index

    // Metrics tracking for LBA/GBA interaction analysis
    let mut cumulative_position_error = 0.0;
    let mut cumulative_rotation_error = 0.0;
    let mut max_position_error = 0.0f64;
    let mut max_rotation_error = 0.0f64;

    // Track previous frame graph for efficient visualization updates
    let mut prev_frame_graph: Option<FrameGraph> = None;

    // Visualize initial state
    visualize_estimate(
        &rec,
        0,
        &world,
        &frame_graph,
        &gt_points,
        &stereo_camera,
        prev_frame_graph.as_ref(),
    )?;
    prev_frame_graph = Some(frame_graph.clone());

    println!(
        "📊 Memory before frame processing: {:.1} MB\n",
        get_rss_mb()
    );

    // Tracking variables
    let mut total_vio_time = 0.0;
    let gravity_vec = [0.0, 9.81, 0.0];
    let config = VioConfig::default();

    // Perturb state generator (to simulate tracking error)
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(1234);

    // MAIN LOOP - Process frames incrementally
    for frame_idx in 1..n_frames {
        let frame_start = std::time::Instant::now();

        // Memory checkpoint every 10 frames
        if frame_idx % 10 == 0 {
            println!(
                "📊 Memory at frame {}: {:.1} MB (peak: {:.1} MB)",
                frame_idx,
                get_rss_mb(),
                get_peak_rss_mb()
            );
        }

        // Get initial guess from previous pose (with simulated tracking error)
        let mut init_pose = world.get_pose(frame_idx - 1).unwrap();
        let mut init_vel = imu_states[frame_idx - 1].velocity;

        // Add some drift/noise to initial guess
        init_pose.translation.x += rng.gen_range(-0.05..0.05);
        init_pose.translation.y += rng.gen_range(-0.05..0.05);
        init_pose.translation.z += rng.gen_range(-0.05..0.05);
        init_vel.x += rng.gen_range(-0.02..0.02);
        init_vel.y += rng.gen_range(-0.02..0.02);
        init_vel.z += rng.gen_range(-0.02..0.02);

        // Get observations for keyframe decision
        let current_obs = &frame_observations[frame_idx];

        // Determine if this should be a keyframe (baseline-based selection)
        let new_points_count = current_obs
            .iter()
            .filter(|obs| world.get_point(obs.point_id).is_none())
            .count();
        let novelty_ratio = if current_obs.is_empty() {
            0.0
        } else {
            new_points_count as f64 / current_obs.len() as f64
        };

        // Compute baseline (translation) since last keyframe
        let current_position = init_pose.translation;
        let last_kf_position = world.frames[last_keyframe_idx].world_pose().translation;
        let translation_since_keyframe = (current_position - last_kf_position).norm();

        // Adaptive baseline threshold based on median depth of visible points
        let sufficient_baseline =
            if let Some(median_depth) = compute_median_depth(current_obs, &world, &init_pose) {
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

        // Add frame to world
        world.add_pose(init_pose);
        imu_states.push(ImuFrameState::with_velocity(init_vel));
        frame_graph.add_frame(frame_role, OptimizationState::Optimized);

        // Preintegrate IMU measurements for this frame
        let t_start = timestamps[frame_idx - 1];
        let t_end = timestamps[frame_idx];
        let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
        let frame_imu: Vec<_> = imu_measurements
            .iter()
            .filter(|m| m.timestamp >= t_start && m.timestamp < t_end)
            .cloned()
            .collect();
        preint.integrate_measurements(&frame_imu, GYRO_NOISE, ACCEL_NOISE);
        preintegrations.push(preint);

        // Triangulate new points only on keyframes
        let mut _new_points = 0;
        if should_create_keyframe {
            println!(
                "  Creating keyframe from frame {} (novelty: {:.1}%, baseline: {:.3}m)",
                frame_idx,
                novelty_ratio * 100.0,
                translation_since_keyframe
            );
            for obs in current_obs.iter() {
                if world.get_point(obs.point_id).is_none() {
                    world.triangulate_and_add_point(obs, &stereo_camera, frame_idx);
                    _new_points += 1;
                }
            }
        }
        
        // Marginalize intermediate (non-keyframe) frames to maintain window size
        // This only affects frames between keyframes, not the keyframes themselves
        let mut optimized_count = frame_graph
            .states
            .iter()
            .filter(|s| s.state == OptimizationState::Optimized)
            .count();

        while optimized_count > WINDOW_SIZE {
            // Find oldest optimized non-keyframe and mark for marginalization
            for i in 0..frame_graph.len() {
                if frame_graph.states[i].state == OptimizationState::Optimized {
                    frame_graph.set_state(i, OptimizationState::Marginalize);
                    break;
                }
            }
            optimized_count -= 1;
        }

        // Run simple VIO optimization on current window
        let result = run_simple_vio_bundle_adjustment(
            &stereo_camera,
            &frame_graph,
            &mut world,
            &frame_observations,
            &mut imu_states,
            &preintegrations,
            gravity_vec,
            accumulated_prior.as_ref(),
            &config,
        );

        let vio_time = result.solve_time_ms;
        total_vio_time += vio_time;

        // Handle marginalization result - accumulate priors
        if let Some(new_prior) = result.new_prior {
            accumulated_prior = Some(new_prior);
        }

        // Mark marginalized frames as inactive now that marginalization is done
        for i in 0..frame_graph.len() {
            if frame_graph.states[i].state == OptimizationState::Marginalize {
                frame_graph.set_state(i, OptimizationState::Inactive);
            }
        }

        // Get optimized pose for error checking
        let optimized_pose = world.frames[frame_idx].world_pose();
        let pos_error = (optimized_pose.translation - gt_poses[frame_idx].translation).norm();

        // Compute rotation error
        let q_err = gt_poses[frame_idx].rotation.inverse().quat * optimized_pose.rotation.quat;
        let angle_rad = 2.0 * q_err.w.abs().acos();
        let angle_deg = angle_rad.to_degrees();

        // Update cumulative error tracking
        cumulative_position_error += pos_error;
        cumulative_rotation_error += angle_deg;
        max_position_error = max_position_error.max(pos_error);
        max_rotation_error = max_rotation_error.max(angle_deg);

        // Log per-frame error metrics to Rerun
        rec.set_time_sequence("frame", frame_idx as i64);
        let _ = rec.log(
            "metrics/error/position_m",
            &rr::Scalars::new([pos_error]),
        );
        let _ = rec.log(
            "metrics/error/rotation_deg",
            &rr::Scalars::new([angle_deg]),
        );
        let _ = rec.log(
            "metrics/error/cumulative_position_m",
            &rr::Scalars::new([cumulative_position_error]),
        );
        let _ = rec.log(
            "metrics/error/avg_position_m",
            &rr::Scalars::new([cumulative_position_error / frame_idx as f64]),
        );
        let _ = rec.log(
            "metrics/error/avg_rotation_deg",
            &rr::Scalars::new([cumulative_rotation_error / frame_idx as f64]),
        );

        // Also log XYZ error components for debugging
        let pos_diff = optimized_pose.translation - gt_poses[frame_idx].translation;
        let _ = rec.log(
            "metrics/error/x_m",
            &rr::Scalars::new([pos_diff.x]),
        );
        let _ = rec.log(
            "metrics/error/y_m",
            &rr::Scalars::new([pos_diff.y]),
        );
        let _ = rec.log(
            "metrics/error/z_m",
            &rr::Scalars::new([pos_diff.z]),
        );

        // Warn if error exceeds thresholds
        const MAX_POSITION_ERROR: f64 = 0.5;
        const MAX_ROTATION_ERROR: f64 = 10.0;
        if pos_error > MAX_POSITION_ERROR || angle_deg > MAX_ROTATION_ERROR {
            eprintln!(
                "\n❌ ERROR: Pose error exceeded thresholds at frame {}!",
                frame_idx
            );
            eprintln!(
                "  Position error: {:.4} m (max: {:.4} m)",
                pos_error, MAX_POSITION_ERROR
            );
            eprintln!(
                "  Rotation error: {:.4} deg (max: {:.4} deg)",
                angle_deg, MAX_ROTATION_ERROR
            );
        }

        // Track keyframe index
        if should_create_keyframe {
            last_keyframe_idx = frame_idx;
        }

        // Visualize current state
        visualize_estimate(
            &rec,
            frame_idx,
            &world,
            &frame_graph,
            &gt_points,
            &stereo_camera,
            prev_frame_graph.as_ref(),
        )?;
        prev_frame_graph = Some(frame_graph.clone());

        let _frame_duration = frame_start.elapsed();
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

        println!(
            "Frame {}: {} opt, {} fixed, {} obs, err={:.3}m, VIO: {:.2} ms{}",
            frame_idx,
            n_optimized,
            n_fixed,
            current_obs.len(),
            pos_error,
            vio_time,
            if should_create_keyframe { " [KF]" } else { "" }
        );
    }

    println!("\n✅ Processed {} frames", n_frames);
    println!("   Final map: {} points", world.num_points());
    println!(
        "\n📊 Final memory: {:.1} MB, Peak: {:.1} MB",
        get_rss_mb(),
        get_peak_rss_mb()
    );
    println!(
        "   Average VIO time: {:.2} ms",
        total_vio_time / (n_frames - 1) as f64
    );

    // Print error summary
    let avg_pos_error = cumulative_position_error / (n_frames - 1) as f64;
    let avg_rot_error = cumulative_rotation_error / (n_frames - 1) as f64;
    println!("\n📏 Error Summary:");
    println!("   Average position error: {:.4} m", avg_pos_error);
    println!("   Maximum position error: {:.4} m", max_position_error);
    println!("   Average rotation error: {:.2}°", avg_rot_error);
    println!("   Maximum rotation error: {:.2}°", max_rotation_error);

    println!("\n📺 Open Rerun to see the SLAM visualization!");
    println!("   Metrics available under 'metrics/' in the Rerun viewer:");
    println!("   - metrics/error/*: Per-frame and cumulative errors");
    println!("   - metrics/gba_impact/*: How much GBA changes poses");
    println!("   - metrics/constraint/*: LBA constraint covariances sent to GBA");
    println!("   - metrics/point_priors/*: Point prior information from GBA");

    Ok(())
}
