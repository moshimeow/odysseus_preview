//! IMU Simulation Demo
//!
//! Demonstrates IMU simulation and preintegration:
//! - Generates a trajectory using CircularTrajectory
//! - Simulates IMU measurements from the trajectory
//! - Visualizes ground truth (green) vs IMU dead-reckoned (red) trajectory
//! - Tests preintegration accuracy by comparing against ground truth

use clap::Parser;
use nalgebra::Vector3;
use odysseus_slam::{
    imu::{simulator::ImuNoiseParams, ImuMeasurement, ImuSimulator, PreintegratedImu},
    math::{SE3, SO3},
    trajectory::{ContinuousCircularTrajectory, ContinuousTrajectory},
};
use rerun as rr;

/// IMU Simulation Demo
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of camera poses in trajectory
    #[arg(long, default_value_t = 100)]
    num_poses: usize,

    /// Trajectory duration in seconds
    #[arg(long, default_value_t = 10.0)]
    duration: f64,

    /// IMU rate in Hz
    #[arg(long, default_value_t = 200.0)]
    imu_rate: f64,

    /// Use noisy IMU (consumer-grade noise)
    #[arg(long, default_value_t = false)]
    noisy: bool,

    /// Circle radius in meters
    #[arg(long, default_value_t = 2.0)]
    radius: f64,

    /// Random seed for noise generation (omit for random seed each run)
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize Rerun
    let rec = rr::RecordingStreamBuilder::new("imu_simulation_demo").spawn()?;

    // Determine seed: use provided seed or generate random one
    let seed = args.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    println!("=== IMU Simulation Demo ===");
    println!(
        "Poses: {}, Duration: {}s, IMU Rate: {}Hz",
        args.num_poses, args.duration, args.imu_rate
    );
    println!(
        "Noisy: {}, Radius: {}m, Seed: {}",
        args.noisy, args.radius, seed
    );

    // Generate continuous trajectory with analytical derivatives
    let trajectory = ContinuousCircularTrajectory::with_oscillation(
        args.radius,
        args.duration,
        0.3, // 0.3 rad oscillation amplitude
    );

    // Sample discrete poses for visualization
    let poses: Vec<(f64, SE3<f64>)> = (0..args.num_poses)
        .map(|i| {
            let t_norm = if args.num_poses > 1 {
                i as f64 / (args.num_poses - 1) as f64
            } else {
                0.0
            };
            let t_real = t_norm * args.duration;
            (t_real, trajectory.pose(t_norm))
        })
        .collect();

    println!(
        "\nGenerated continuous trajectory, sampled {} poses over {}s",
        poses.len(),
        args.duration
    );

    // Log ground truth trajectory
    log_trajectory(&rec, &poses)?;

    // Create IMU simulator
    let noise_params = if args.noisy {
        ImuNoiseParams::consumer_grade()
    } else {
        ImuNoiseParams::zero()
    };
    let simulator = ImuSimulator::new(noise_params.clone(), args.imu_rate);

    // Generate IMU measurements using analytical derivatives
    let imu_measurements =
        simulator.generate_from_continuous_trajectory(&trajectory, args.duration, seed);
    println!(
        "Generated {} IMU measurements (using analytical Δv/Δt)",
        imu_measurements.len()
    );

    // Print some IMU data samples
    print_imu_samples(&imu_measurements);

    // Dead-reckon using IMU and visualize
    let gravity = Vector3::new(0.0, 0.0, -9.81);
    dead_reckon_and_visualize(
        &rec,
        &poses,
        &imu_measurements,
        gravity,
        &trajectory,
        args.duration,
    )?;

    // Test preintegration
    test_preintegration(&poses, &imu_measurements, &noise_params)?;

    println!("\nVisualization complete. Check Rerun viewer.");
    println!("  Green = Ground truth trajectory");
    println!("  Red   = IMU dead-reckoned trajectory");

    Ok(())
}

/// Log the ground truth trajectory to Rerun
fn log_trajectory(
    rec: &rr::RecordingStream,
    poses: &[(f64, SE3<f64>)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Extract positions
    let positions: Vec<[f32; 3]> = poses
        .iter()
        .map(|(_, pose)| {
            [
                pose.translation.x as f32,
                pose.translation.y as f32,
                pose.translation.z as f32,
            ]
        })
        .collect();

    // Log as line strip
    rec.log(
        "trajectory/path",
        &rr::LineStrips3D::new([positions.clone()]).with_colors([rr::Color::from_rgb(0, 200, 0)]),
    )?;

    // Log individual poses as points
    rec.log(
        "trajectory/poses",
        &rr::Points3D::new(positions)
            .with_colors([rr::Color::from_rgb(0, 255, 0)])
            .with_radii([0.02]),
    )?;

    // Log coordinate frames for a subset of poses
    for (i, (_, pose)) in poses.iter().enumerate().step_by(poses.len() / 10 + 1) {
        let translation = [
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        ];

        // Get rotation matrix
        let rot_mat = pose.rotation.to_matrix();
        let mat3x3 = [
            [
                rot_mat.x_axis.x as f32,
                rot_mat.y_axis.x as f32,
                rot_mat.z_axis.x as f32,
            ],
            [
                rot_mat.x_axis.y as f32,
                rot_mat.y_axis.y as f32,
                rot_mat.z_axis.y as f32,
            ],
            [
                rot_mat.x_axis.z as f32,
                rot_mat.y_axis.z as f32,
                rot_mat.z_axis.z as f32,
            ],
        ];

        rec.log(
            format!("trajectory/frames/{}", i),
            &rr::Transform3D::from_translation_mat3x3(translation, mat3x3),
        )?;
    }

    Ok(())
}

/// Dead-reckon trajectory using IMU measurements and visualize in Rerun
fn dead_reckon_and_visualize(
    rec: &rr::RecordingStream,
    gt_poses: &[(f64, SE3<f64>)],
    measurements: &[ImuMeasurement],
    gravity: Vector3<f64>,
    trajectory: &dyn ContinuousTrajectory,
    total_duration: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    if measurements.is_empty() || gt_poses.is_empty() {
        return Ok(());
    }

    println!("\n=== Dead Reckoning from IMU ===");

    // Start from first ground truth pose
    let start_pose = &gt_poses[0].1;
    let mut position = Vector3::new(
        start_pose.translation.x,
        start_pose.translation.y,
        start_pose.translation.z,
    );
    let mut rotation = start_pose.rotation;

    // Initial velocity should match the trajectory at t=0
    let mut velocity = if gt_poses.len() >= 2 {
        let p0 = gt_poses[0].1.translation;
        let p1 = gt_poses[1].1.translation;

        let theta0 = f64::atan2(p0.y, p0.x);
        let theta1 = f64::atan2(p1.y, p1.x);
        let dt = gt_poses[1].0 - gt_poses[0].0;
        let mut d_theta = theta1 - theta0;
        if d_theta > std::f64::consts::PI {
            d_theta -= 2.0 * std::f64::consts::PI;
        }
        if d_theta < -std::f64::consts::PI {
            d_theta += 2.0 * std::f64::consts::PI;
        }
        let omega = d_theta / dt;

        let radius = f64::sqrt(p0.x * p0.x + p0.y * p0.y);
        Vector3::new(
            -radius * omega * theta0.sin(),
            radius * omega * theta0.cos(),
            0.0,
        )
    } else {
        Vector3::zeros()
    };

    // Collect dead-reckoned positions
    let mut dr_positions: Vec<[f32; 3]> =
        vec![[position.x as f32, position.y as f32, position.z as f32]];

    // Process each IMU measurement
    let mut prev_time = 0.0; // The trajectory and first measurement interval start at 0.0

    for m in measurements.iter() {
        let dt = m.timestamp - prev_time;
        if dt <= 0.0 {
            prev_time = m.timestamp;
            continue;
        }

        // Midpoint integration: use rotation at middle of interval for acceleration transform
        // This correctly handles the fact that IMU reports average acceleration over the interval

        // Compute half rotation increment
        let omega = m.gyro * dt;
        let half_omega =
            odysseus_solver::math3d::Vec3::new(omega.x * 0.5, omega.y * 0.5, omega.z * 0.5);
        let half_delta_rot = SO3::exp(half_omega);

        // Rotation at midpoint of interval
        let rotation_mid = (rotation * half_delta_rot).normalize();

        // Transform acceleration using midpoint rotation
        // IMU measures specific force: a_imu = a_motion - g_body
        let accel_body = odysseus_solver::math3d::Vec3::new(m.accel.x, m.accel.y, m.accel.z);
        let accel_world_vec = rotation_mid.rotate(accel_body);
        let accel_world = Vector3::new(accel_world_vec.x, accel_world_vec.y, accel_world_vec.z);

        // Add gravity to get total acceleration
        // a_world = R_mid * a_imu + g
        let total_accel = accel_world + gravity;

        // Update velocity and position
        position += velocity * dt + 0.5 * total_accel * dt * dt;
        velocity += total_accel * dt;

        // Full rotation update
        let delta_rot = SO3::exp(odysseus_solver::math3d::Vec3::new(
            omega.x, omega.y, omega.z,
        ));
        rotation = (rotation * delta_rot).normalize();

        // Store position for visualization
        dr_positions.push([position.x as f32, position.y as f32, position.z as f32]);

        prev_time = m.timestamp;
    }

    // Log dead-reckoned trajectory as red line
    rec.log(
        "dead_reckoned/path",
        &rr::LineStrips3D::new([dr_positions.clone()])
            .with_colors([rr::Color::from_rgb(255, 50, 50)]),
    )?;

    // Log final positions as points (sampled)
    let sample_step = (dr_positions.len() / 100).max(1);
    let sampled_positions: Vec<_> = dr_positions.iter().step_by(sample_step).cloned().collect();
    rec.log(
        "dead_reckoned/poses",
        &rr::Points3D::new(sampled_positions)
            .with_colors([rr::Color::from_rgb(255, 100, 100)])
            .with_radii([0.015]),
    )?;

    // Compute final error at the exact time of the last integrated measurement
    let last_time = measurements.last().map(|m| m.timestamp).unwrap_or(0.0);
    let final_gt_pose = trajectory.pose(last_time / total_duration);

    let gt_final_pos = Vector3::new(
        final_gt_pose.translation.x,
        final_gt_pose.translation.y,
        final_gt_pose.translation.z,
    );
    let position_error = (position - gt_final_pos).norm();
    let gt_rotation = final_gt_pose.rotation;
    let rot_error = (rotation.inverse() * gt_rotation).log();
    let rotation_error_deg =
        (rot_error.x * rot_error.x + rot_error.y * rot_error.y + rot_error.z * rot_error.z)
            .sqrt()
            .to_degrees();

    println!("Dead reckoning results (at t={:.3}s):", last_time);
    println!(
        "  Final position: [{:.4}, {:.4}, {:.4}] m",
        position.x, position.y, position.z
    );
    println!(
        "  GT final pos:   [{:.4}, {:.4}, {:.4}] m",
        gt_final_pos.x, gt_final_pos.y, gt_final_pos.z
    );
    println!("  Position error: {:.4} m", position_error);
    println!("  Rotation error: {:.2} deg", rotation_error_deg);

    Ok(())
}

/// Print sample IMU measurements
fn print_imu_samples(measurements: &[ImuMeasurement]) {
    println!("\n=== IMU Measurement Samples ===");

    let samples = [
        0,
        measurements.len() / 4,
        measurements.len() / 2,
        3 * measurements.len() / 4,
        measurements.len() - 1,
    ];

    for &idx in &samples {
        if idx < measurements.len() {
            let m = &measurements[idx];
            println!(
                "t={:.3}s: gyro=[{:.4}, {:.4}, {:.4}] rad/s, accel=[{:.4}, {:.4}, {:.4}] m/s²",
                m.timestamp, m.gyro.x, m.gyro.y, m.gyro.z, m.accel.x, m.accel.y, m.accel.z
            );
        }
    }

    // Compute statistics
    let mut gyro_mean = Vector3::zeros();
    let mut accel_mean = Vector3::zeros();
    for m in measurements {
        gyro_mean += m.gyro;
        accel_mean += m.accel;
    }
    let n = measurements.len() as f64;
    gyro_mean /= n;
    accel_mean /= n;

    println!("\nStatistics over {} measurements:", measurements.len());
    println!(
        "  Mean gyro:  [{:.4}, {:.4}, {:.4}] rad/s",
        gyro_mean.x, gyro_mean.y, gyro_mean.z
    );
    println!(
        "  Mean accel: [{:.4}, {:.4}, {:.4}] m/s²",
        accel_mean.x, accel_mean.y, accel_mean.z
    );
    println!(
        "  Accel magnitude: {:.4} m/s² (expect ~9.81 from gravity)",
        accel_mean.norm()
    );
}

/// Test preintegration accuracy
fn test_preintegration(
    poses: &[(f64, SE3<f64>)],
    measurements: &[ImuMeasurement],
    noise_params: &ImuNoiseParams,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing Preintegration ===");

    // Find measurements between first and last pose
    let start_time = poses[0].0;
    let end_time = poses[poses.len() - 1].0;

    let relevant_measurements: Vec<_> = measurements
        .iter()
        .filter(|m| m.timestamp >= start_time && m.timestamp <= end_time)
        .cloned()
        .collect();

    println!(
        "Using {} measurements from t={:.3} to t={:.3}",
        relevant_measurements.len(),
        start_time,
        end_time
    );

    // Preintegrate
    let mut preint = PreintegratedImu::new(Vector3::zeros(), Vector3::zeros());
    preint.integrate_measurements(
        &relevant_measurements,
        noise_params.gyro_noise_density,
        noise_params.accel_noise_density,
    );

    println!("\nPreintegration results:");
    println!(
        "  Delta rotation (rotvec): [{:.4}, {:.4}, {:.4}] rad",
        preint.delta_rotation.x, preint.delta_rotation.y, preint.delta_rotation.z
    );
    println!(
        "  Delta rotation magnitude: {:.4} rad ({:.2} deg)",
        preint.delta_rotation.norm(),
        preint.delta_rotation.norm().to_degrees()
    );
    println!(
        "  Delta velocity: [{:.4}, {:.4}, {:.4}] m/s",
        preint.delta_velocity.x, preint.delta_velocity.y, preint.delta_velocity.z
    );
    println!(
        "  Delta position: [{:.4}, {:.4}, {:.4}] m",
        preint.delta_position.x, preint.delta_position.y, preint.delta_position.z
    );
    println!("  Delta time: {:.4} s", preint.delta_time);

    // Compare with ground truth pose change
    let pose_start = &poses[0].1;
    let pose_end = &poses[poses.len() - 1].1;

    // Ground truth relative pose
    let rel_pose = pose_start.inverse() * *pose_end;
    let gt_rotation = rel_pose.rotation.log();
    let gt_translation = rel_pose.translation;

    println!("\nGround truth (pose difference):");
    println!(
        "  Rotation: [{:.4}, {:.4}, {:.4}] rad ({:.2} deg)",
        gt_rotation.x,
        gt_rotation.y,
        gt_rotation.z,
        (gt_rotation.x * gt_rotation.x
            + gt_rotation.y * gt_rotation.y
            + gt_rotation.z * gt_rotation.z)
            .sqrt()
            .to_degrees()
    );
    println!(
        "  Translation: [{:.4}, {:.4}, {:.4}] m",
        gt_translation.x, gt_translation.y, gt_translation.z
    );

    // Rotation error
    let rot_error =
        preint.delta_rotation - Vector3::new(gt_rotation.x, gt_rotation.y, gt_rotation.z);
    println!(
        "\nRotation error: [{:.4}, {:.4}, {:.4}] rad ({:.2} deg)",
        rot_error.x,
        rot_error.y,
        rot_error.z,
        rot_error.norm().to_degrees()
    );

    println!(
        "\nNote: Position/velocity comparison requires gravity compensation in residual function."
    );
    println!("The preintegration values are in the start body frame and include accumulated gravity effects.");

    Ok(())
}
