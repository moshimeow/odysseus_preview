//! Visualize Jacobian Sparsity Pattern
//!
//! This example generates synthetic SLAM data, builds a SparsityPattern,
//! and visualizes its sparsity pattern as a PNG image.
//! Also visualizes the synthetic data in Rerun.

use odysseus_slam::{
    camera::StereoCamera,
    math::SE3,
    WorldState,
    frame_graph::{FrameGraph, FrameRole, OptimizationState},
    simulation::{generate_stereo_observations, generate_random_points, BrownianTrajectory, TrajectoryGenerator},
};
use odysseus_solver::build_slam_entries;
use std::collections::HashMap;
use rerun as rr;

// Problem size - small enough to visualize clearly
const N_CAMERAS: usize = 15;              // 15 camera poses
const N_POINTS: usize = 60;               // 60 3D points
const WINDOW_SIZE: usize = 3;             // Window of 3 frames

// Physical constants
const STEREO_BASELINE: f64 = 0.1;
const ROOM_SIZE: f64 = 3.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Jacobian Sparsity Visualization");
    println!("===================================\n");

    // Initialize Rerun
    println!("📺 Initializing Rerun...");
    let rec = rr::RecordingStreamBuilder::new("jacobian_sparsity").spawn()?;

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
    println!("  Image size: {}x{} px", image_width, image_height);
    println!("  Baseline: {} m\n", STEREO_BASELINE);

    // Generate random 3D points
    println!("🌍 Generating {} 3D points...", N_POINTS);
    let half_room = ROOM_SIZE / 2.0;
    let true_points = generate_random_points(
        N_POINTS,
        [-half_room, half_room, -half_room, half_room, 2.0, 2.0 + ROOM_SIZE],
        42,
    );
    println!("  ✓ Generated {} points\n", true_points.len());

    // Generate camera trajectory
    println!("📹 Generating {} camera poses...", N_CAMERAS);
    let trajectory = BrownianTrajectory::new(0.15, 0.08);
    let true_poses = trajectory.generate(N_CAMERAS, 42);
    println!("  ✓ Generated {} poses\n", true_poses.len());

    // Generate stereo observations
    println!("👁️  Generating stereo observations...");
    let observations = generate_stereo_observations(
        &true_points,
        &true_poses,
        &stereo_camera,
        image_width,
        image_height,
    );
    println!("  ✓ Generated {} observations\n", observations.len());

    // Visualize synthetic data in Rerun
    println!("📺 Visualizing synthetic data in Rerun...");
    visualize_synthetic_data(&rec, &true_points, &true_poses, &stereo_camera)?;

    // Create world state and add poses and points
    println!("🗺️  Creating world state...");
    let mut world = WorldState::new();
    // Add all poses first
    for pose in &true_poses {
        world.add_pose(*pose);
    }
    // Add points to keyframe 0
    for (point_id, point) in true_points.iter().enumerate() {
        world.add_point_with_id(*point, 0, point_id);
    }
    println!("  ✓ World state created with {} poses and {} points\n", world.num_frames(), world.num_points());

    // Build frame graph for window
    println!("🪟 Setting up window: frames 0 to {} (inclusive)\n", WINDOW_SIZE - 1);
    let mut frame_graph = FrameGraph::new();
    // First frame fixed (gauge anchor)
    frame_graph.add_frame(FrameRole::Keyframe, OptimizationState::Fixed);
    // Remaining window frames optimized
    for _ in 1..WINDOW_SIZE {
        frame_graph.add_frame(FrameRole::Transient, OptimizationState::Optimized);
    }
    // Add inactive frames for the rest
    for _ in WINDOW_SIZE..N_CAMERAS {
        frame_graph.add_frame(FrameRole::Transient, OptimizationState::Inactive);
    }

    // Filter observations to only those in the window
    let window_observations: Vec<_> = observations.iter()
        .filter(|obs| obs.camera_id < WINDOW_SIZE)
        .copied()
        .collect();

    println!("📊 Window statistics:");
    println!("  - Observations in window: {}", window_observations.len());
    println!("  - Points: {}", N_POINTS);
    println!("  - Window frames: {}\n", WINDOW_SIZE);

    // Build sparsity pattern (mirror run_bundle_adjustment logic)
    println!("🔧 Building sparsity pattern...");
    
    // Build pose parameter mappings (only for Optimized frames)
    let mut pose_to_param_idx: HashMap<usize, usize> = HashMap::new();
    let mut offset = 0;
    
    for (frame_idx, state) in frame_graph.states.iter().enumerate() {
        if state.state == OptimizationState::Optimized {
            pose_to_param_idx.insert(frame_idx, offset);
            offset += 6;
        }
    }
    let n_poses = pose_to_param_idx.len();

    // Build point parameter mappings
    let mut point_to_param_idx: HashMap<usize, usize> = HashMap::new();
    for point_id in 0..N_POINTS {
        point_to_param_idx.insert(point_id, offset);
        offset += 3;
    }
    let n_points = N_POINTS;
    let n_observations = window_observations.len();

    // Build sparsity observations (same format as optimization/mod.rs)
    let sparsity_obs: Vec<_> = window_observations.iter()
        .enumerate()
        .map(|(i, obs)| {
            let is_optimized = frame_graph.states[obs.camera_id].state == OptimizationState::Optimized;
            let pose_start = if is_optimized {
                pose_to_param_idx.get(&obs.camera_id).copied().unwrap_or(0)
            } else {
                0
            };
            let point_start = point_to_param_idx[&obs.point_id];
            // (residual_start, pose_start, point_start, has_pose, has_point)
            (i * 4, pose_start, point_start, is_optimized, true)
        })
        .collect();

    let n_params = offset;
    let n_residuals = n_observations * 4;
    let entries = build_slam_entries(&sparsity_obs);

    println!("  ✓ Sparsity pattern created");
    println!("  - Parameters: {} ({} poses × 6 + {} points × 3)", n_params, n_poses, n_points);
    println!("  - Residuals: {} ({} observations × 4)\n", n_residuals, n_observations);

    // Visualize sparsity pattern
    println!("📊 Jacobian dimensions: {} x {}", n_residuals, n_params);
    println!("🎨 Creating visualization...");

    // Create RGB image buffer (black background)
    let mut img_buf = image::RgbImage::new(n_params as u32, n_residuals as u32);

    // Fill image: white for non-zero entries
    for &(row, col) in &entries {
        img_buf.put_pixel(col as u32, row as u32, image::Rgb([255u8, 255u8, 255u8]));
    }

    let nnz = entries.len();
    let sparsity = 1.0 - (nnz as f64) / (n_residuals * n_params) as f64;
    println!("  Non-zero entries: {} / {} ({:.2}% sparse)\n", nnz, n_residuals * n_params, sparsity * 100.0);

    // Save image
    let output_path = "jacobian_sparsity.png";
    img_buf.save(output_path)?;
    println!("  ✓ Saved to {}\n", output_path);

    println!("✨ Done! The image shows:");
    println!("  - Black pixels = zero entries");
    println!("  - White pixels = non-zero entries");
    println!("  - Each row represents 4 residuals (one stereo observation)");
    println!("  - Each column represents one parameter (pose or point)");

    println!("\n📺 Check Rerun viewer for 3D visualization of synthetic data!");

    Ok(())
}

/// Visualize synthetic data: points, trajectory, and camera frustums
fn visualize_synthetic_data(
    rec: &rr::RecordingStream,
    points: &[odysseus_solver::math3d::Vec3<f64>],
    poses: &[SE3<f64>],
    stereo_camera: &StereoCamera<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // 3D points (blue)
    let point_positions: Vec<[f32; 3]> = points
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();

    rec.log(
        "world/points",
        &rr::Points3D::new(point_positions)
            .with_colors([[50, 150, 255]])
            .with_radii([0.04]),
    )?;

    // Camera trajectory (orange line)
    let trajectory: Vec<[f32; 3]> = poses
        .iter()
        .map(|pose| {
            let t = &pose.translation;
            [t.x as f32, t.y as f32, t.z as f32]
        })
        .collect();

    rec.log(
        "world/trajectory",
        &rr::LineStrips3D::new([trajectory])
            .with_colors([[255, 150, 50]])
            .with_radii([0.02]),
    )?;

    // Camera frustums
    for (i, pose) in poses.iter().enumerate() {
        visualize_stereo_camera(
            rec,
            &format!("world/cameras/cam_{:03}", i),
            pose,
            stereo_camera,
            [255, 150, 50, 255],  // Orange
        )?;
    }

    Ok(())
}

/// Visualize stereo camera using Rerun's native Pinhole
fn visualize_stereo_camera(
    rec: &rr::RecordingStream,
    entity_path: &str,
    pose: &SE3<f64>,
    stereo_camera: &StereoCamera<f64>,
    color: [u8; 4],
) -> Result<(), Box<dyn std::error::Error>> {
    let camera_rot = pose.rotation.to_matrix();
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

    rec.log(
        left_path.as_str(),
        &rr::Pinhole::from_focal_length_and_resolution(
            [stereo_camera.left.fx as f32, stereo_camera.left.fy as f32],
            [640.0, 480.0],
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

