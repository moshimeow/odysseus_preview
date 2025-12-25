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
    optimization::vio::{run_vio_bundle_adjustment, VioConfig},
    simulation::{add_noise_to_stereo_observations, generate_stereo_observations},
    spline::BezierSplineTrajectory,
    trajectory::ContinuousTrajectory,
    utils::{get_peak_rss_mb, get_rss_mb, load_point_cloud},
    visualization::{visualize_estimate, visualize_ground_truth},
    WorldState,
};
use rerun as rr;
use std::collections::HashSet;

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
    let fixed_point_ids: HashSet<usize> = HashSet::new();

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

        // Add frame to world
        world.add_pose(init_pose);
        imu_states.push(ImuFrameState::with_velocity(init_vel));
        frame_graph.add_frame(FrameRole::Transient, OptimizationState::Optimized);

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

        // Triangulate new points
        let current_obs = &frame_observations[frame_idx];
        let mut new_points = 0;
        for obs in current_obs {
            if world.get_point(obs.point_id).is_none() {
                world.triangulate_and_add_point(obs, &stereo_camera, frame_idx);
                new_points += 1;
            }
        }

        // Manage sliding window
        let mut optimized_count = frame_graph
            .states
            .iter()
            .filter(|s| s.state == OptimizationState::Optimized)
            .count();

        while optimized_count > WINDOW_SIZE {
            // Find oldest optimized frame and mark as inactive
            for i in 0..frame_graph.len() {
                if frame_graph.states[i].state == OptimizationState::Optimized {
                    frame_graph.set_state(i, OptimizationState::Inactive);
                    break;
                }
            }
            optimized_count -= 1;
        }

        // Run VIO optimization on current window
        let result = run_vio_bundle_adjustment(
            &stereo_camera,
            &frame_graph,
            &mut world,
            &frame_observations,
            &mut imu_states,
            &preintegrations,
            gravity_vec,
            &fixed_point_ids,
            &config,
        );

        let vio_time = result.solve_time_ms;
        total_vio_time += vio_time;

        // Get optimized pose for error checking
        let optimized_pose = world.frames[frame_idx].world_pose();
        let pos_error = (optimized_pose.translation - gt_poses[frame_idx].translation).norm();

        // Compute rotation error
        let q_err = gt_poses[frame_idx].rotation.inverse().quat * optimized_pose.rotation.quat;
        let angle_rad = 2.0 * q_err.w.abs().acos();
        let angle_deg = angle_rad.to_degrees();

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

        let frame_duration = frame_start.elapsed();
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
            "Frame {}: {} opt, {} fixed, {} obs, {} new pts, VIO: {:.2} ms, Total: {:.2} ms",
            frame_idx,
            n_optimized,
            n_fixed,
            current_obs.len(),
            new_points,
            vio_time,
            frame_duration.as_secs_f64() * 1000.0
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
    println!("\n📺 Open Rerun to see the SLAM visualization!");

    Ok(())
}
