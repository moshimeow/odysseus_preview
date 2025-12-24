//! Visual-Inertial Odometry (VIO) Demo
//!
//! Demonstrates tightly-coupled VIO optimization combining:
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
    geometry::StereoObservation,
    imu::{simulator::ImuNoiseParams, ImuFrameState, ImuSimulator, PreintegratedImu},
    math::SE3,
    optimization::vio::{run_vio_bundle_adjustment, VioConfig},
    simulation::{self, generate_stereo_observations},
    spline::BezierSplineTrajectory,
    trajectory::ContinuousTrajectory,
    WorldState,
};
use rerun as rr;
use std::fs::File;
use std::io::{BufReader, Read};

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
const DURATION: f64 = 5.0; // Seconds
const STEREO_BASELINE: f64 = 0.1;

// Noise parameters
const ACCEL_NOISE: f64 = 0.01; // m/s^2 / sqrt(Hz)
const GYRO_NOISE: f64 = 0.001; // rad/s / sqrt(Hz)

fn main() {
    let args = Args::parse();
    unsafe {
        let _ = backtrace_on_stack_overflow::enable(|| {
            if let Err(e) = run_demo(args.noise) {
                eprintln!("Error: {}", e);
            }
        });
    }
}

fn run_demo(pixel_noise: f64) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Visual-Inertial Odometry (VIO) Demo");
    println!("   Observation Noise: {:.2} pixels", pixel_noise);
    println!("======================================\n");

    // Initialize Rerun
    let rec = rr::RecordingStreamBuilder::new("vio_demo").spawn()?;
    rec.log_static("world", &rr::ViewCoordinates::RDF())?; // Right, Down, Forward (OpenCV)

    // 1. Load Spline and generating trajectory
    println!("📈 Loading Bezier spline from Blender export...");
    let spline_path = "blender_stuff/greeble_room/camera_spline.bin";
    let trajectory = BezierSplineTrajectory::load(spline_path)?;
    println!("   Spline loaded successfully.\n");

    // 2. Setup Camera
    let focal_length = 500.0;
    let width = 640.0;
    let height = 480.0;
    let stereo_camera = StereoCamera::simple(focal_length, width, height, STEREO_BASELINE);

    // 3. Generate Ground Truth and IMU data
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

    let imu_measurements = simulator.generate_from_continuous_trajectory(
        &trajectory,
        DURATION,
        42, // seed
    );
    println!("   Generated {} IMU measurements.", imu_measurements.len());

    // 4. Generate Camera Frames and Observations
    println!("📷 Sampling camera frames at {} Hz...", CAMERA_RATE);
    let mut gt_poses = Vec::new();
    let mut gt_velocities = Vec::new();
    let mut timestamps = Vec::new();

    let dt_cam = 1.0 / CAMERA_RATE;
    let mut t = 0.0;
    while t <= DURATION {
        gt_poses.push(trajectory.pose(t));
        gt_velocities.push(trajectory.linear_velocity(t));
        timestamps.push(t);
        t += dt_cam;
    }
    let n_frames = gt_poses.len();

    // Load points for observations
    let points_path = "blender_stuff/greeble_room/room_mesh.bin";
    let gt_points_raw = load_point_cloud(points_path)?;
    let gt_points: Vec<_> = gt_points_raw
        .iter()
        .map(|p| odysseus_solver::math3d::Vec3::new(p[0], p[1], p[2]))
        .collect();

    let observations_raw =
        generate_stereo_observations(&gt_points, &gt_poses, &stereo_camera, width, height);

    // Add pixel noise
    let observations =
        simulation::add_noise_to_stereo_observations(&observations_raw, pixel_noise, 42);

    let mut frame_observations: Vec<Vec<StereoObservation>> = vec![Vec::new(); n_frames];
    for obs in observations {
        frame_observations[obs.camera_id].push(obs);
    }

    // 5. Setup SLAM State
    println!("🏗️  Initializing VIO state...");
    let mut world = WorldState::new();
    let mut frame_graph = FrameGraph::new();
    let mut imu_states = Vec::new();

    // Perturb initial state slightly
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(1234);

    for i in 0..n_frames {
        let mut pose = gt_poses[i];
        let mut vel = gt_velocities[i];

        if i > 0 {
            // Add some drift/noise to initial guess (simulating tracking error)
            pose.translation.x += rng.gen_range(-0.1..0.1);
            pose.translation.y += rng.gen_range(-0.1..0.1);
            pose.translation.z += rng.gen_range(-0.1..0.1);

            vel.x += rng.gen_range(-0.05..0.05);
            vel.y += rng.gen_range(-0.05..0.05);
            vel.z += rng.gen_range(-0.05..0.05);
        }

        world.add_pose(pose);
        imu_states.push(ImuFrameState::with_velocity(vel));

        let state = if i == 0 {
            OptimizationState::Fixed
        } else {
            OptimizationState::Optimized
        };
        frame_graph.add_frame(FrameRole::Keyframe, state);
    }

    // Initialize all observed points
    for (i, frame_obs) in frame_observations.iter().enumerate() {
        for obs in frame_obs {
            if world.get_point(obs.point_id).is_none() {
                if let Some(pos) = gt_points.get(obs.point_id) {
                    world.add_point_with_id(*pos, i, obs.point_id);
                }
            }
        }
    }

    // 6. Preintegrate IMU measurements between frames
    println!("🔄 Preintegrating IMU data...");
    let mut preintegrations = Vec::new();
    for i in 0..n_frames - 1 {
        let t_start = timestamps[i];
        let t_end = timestamps[i + 1];

        let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
        let frame_measurements: Vec<_> = imu_measurements
            .iter()
            .filter(|m| m.timestamp >= t_start && m.timestamp < t_end)
            .cloned()
            .collect();

        preint.integrate_measurements(&frame_measurements, GYRO_NOISE, ACCEL_NOISE);
        preintegrations.push(preint);
    }

    // 7. Run VIO Optimization
    println!("🚀 Running tightly-coupled VIO Optimization...");
    let gravity_vec = [0.0, 9.81, 0.0];

    let config = VioConfig::default();

    let result = run_vio_bundle_adjustment(
        &stereo_camera,
        &frame_graph,
        &mut world,
        &frame_observations,
        &mut imu_states,
        &preintegrations,
        gravity_vec,
        &config,
    );

    println!(
        "\n✅ Optimization finished in {} iterations ({:.2} ms).",
        result.iterations, result.solve_time_ms
    );
    println!("   Final error: {:.4}", result.final_error);

    // 8. Visualization
    println!("📺 Visualizing in Rerun...");
    visualize_vio(
        &rec,
        &world,
        &gt_poses,
        &gt_points,
        &imu_states,
        &gt_velocities,
    )?;

    Ok(())
}

fn visualize_vio(
    rec: &rr::RecordingStream,
    world: &WorldState,
    gt_poses: &[SE3<f64>],
    _gt_points: &[odysseus_solver::math3d::Vec3<f64>],
    imu_states: &[ImuFrameState],
    gt_velocities: &[Vector3<f64>],
) -> Result<(), Box<dyn std::error::Error>> {
    // Log Ground Truth Trajectory
    let gt_path: Vec<[f32; 3]> = gt_poses
        .iter()
        .map(|p| {
            [
                p.translation.x as f32,
                p.translation.y as f32,
                p.translation.z as f32,
            ]
        })
        .collect();
    rec.log(
        "world/gt/trajectory",
        &rr::LineStrips3D::new([gt_path]).with_colors([[150, 150, 150]]),
    )?;

    // Log Estimated Trajectory
    let est_path: Vec<[f32; 3]> = world
        .frames
        .iter()
        .map(|f| {
            let p = f.world_pose().translation;
            [p.x as f32, p.y as f32, p.z as f32]
        })
        .collect();
    rec.log(
        "world/est/trajectory",
        &rr::LineStrips3D::new([est_path]).with_colors([[50, 150, 255]]),
    )?;

    // Log Points
    let points: Vec<[f32; 3]> = world
        .get_all_points()
        .iter()
        .map(|(_, p)| [p.x as f32, p.y as f32, p.z as f32])
        .collect();
    rec.log(
        "world/est/points",
        &rr::Points3D::new(points)
            .with_radii([0.02])
            .with_colors([[100, 200, 255]]),
    )?;

    // Log Velocity Comparison
    for i in 0..imu_states.len() {
        rec.set_time_sequence("frame", i as i64);
        let est_vel = imu_states[i].velocity;
        let gt_vel = gt_velocities[i];

        rec.log("plots/velocity/x/est", &rr::Scalars::new([est_vel.x]))?;
        rec.log("plots/velocity/x/gt", &rr::Scalars::new([gt_vel.x]))?;

        rec.log("plots/velocity/y/est", &rr::Scalars::new([est_vel.y]))?;
        rec.log("plots/velocity/y/gt", &rr::Scalars::new([gt_vel.y]))?;

        rec.log("plots/velocity/z/est", &rr::Scalars::new([est_vel.z]))?;
        rec.log("plots/velocity/z/gt", &rr::Scalars::new([gt_vel.z]))?;
    }

    Ok(())
}

fn load_point_cloud(path: &str) -> Result<Vec<[f64; 3]>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let num_vertices = u32::from_le_bytes(buf) as usize;

    let mut vertices = Vec::with_capacity(num_vertices);
    for _ in 0..num_vertices {
        let mut v_buf = [0u8; 4];
        let mut coords = [0.0f64; 3];
        for i in 0..3 {
            reader.read_exact(&mut v_buf)?;
            coords[i] = f32::from_le_bytes(v_buf) as f64;
        }
        vertices.push(coords);
    }
    Ok(vertices)
}
