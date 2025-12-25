//! Shared Rerun visualization functions for SLAM demos

use crate::camera::StereoCamera;
use crate::frame_graph::{FrameGraph, FrameRole, OptimizationState};
use crate::geometry::Point3D;
use crate::math::SE3;
use crate::world_state::WorldState;
use rerun as rr;

/// Get the camera path prefix for a given frame state and role
pub fn camera_path_prefix(state: OptimizationState, role: FrameRole) -> &'static str {
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

/// Get the camera path prefix for GBA visualization
pub fn gba_camera_path_prefix(state: OptimizationState, role: FrameRole) -> &'static str {
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

/// Visualize stereo camera using Rerun's native Pinhole
pub fn visualize_stereo_camera(
    rec: &rr::RecordingStream,
    entity_path: &str,
    pose: &SE3<f64>,
    stereo_camera: &StereoCamera<f64>,
    color: [u8; 4], // RGBA color for the camera
) -> Result<(), Box<dyn std::error::Error>> {
    // Convert rotation to 3x3 array for Rerun visualization
    let camera_rot = pose.rotation.to_matrix();
    let camera_rot_for_rerun = camera_rot.inverse();
    let rot_matrix = [
        [
            camera_rot_for_rerun.m00() as f32,
            camera_rot_for_rerun.m01() as f32,
            camera_rot_for_rerun.m02() as f32,
        ],
        [
            camera_rot_for_rerun.m10() as f32,
            camera_rot_for_rerun.m11() as f32,
            camera_rot_for_rerun.m12() as f32,
        ],
        [
            camera_rot_for_rerun.m20() as f32,
            camera_rot_for_rerun.m21() as f32,
            camera_rot_for_rerun.m22() as f32,
        ],
    ];

    // Left camera
    let left_path = format!("{}/left", entity_path);
    rec.log(
        left_path.as_str(),
        &rr::Transform3D::from_translation_mat3x3(
            [
                pose.translation.x as f32,
                pose.translation.y as f32,
                pose.translation.z as f32,
            ],
            rot_matrix,
        ),
    )?;

    // Log pinhole for left camera
    rec.log(
        left_path.as_str(),
        &rr::Pinhole::from_focal_length_and_resolution(
            [stereo_camera.left.fx as f32, stereo_camera.left.fy as f32],
            [640.0, 480.0], // Resolution
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
            [
                right_translation.x as f32,
                right_translation.y as f32,
                right_translation.z as f32,
            ],
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

/// Visualize ground truth trajectory and map
pub fn visualize_ground_truth(
    rec: &rr::RecordingStream,
    points: &[Point3D<f64>],
    poses: &[SE3<f64>],
    stereo_camera: &StereoCamera<f64>,
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

    // Camera frustums for ground truth
    for (i, pose) in poses.iter().enumerate() {
        visualize_stereo_camera(
            rec,
            &format!("world/ground_truth/cameras/cam_{}", i),
            pose,
            stereo_camera,
            [100, 100, 100, 255], // Gray for GT
        )?;
    }

    Ok(())
}

/// Visualize estimated trajectory and map on the "trajectory" timeline
pub fn visualize_estimate(
    rec: &rr::RecordingStream,
    frame_idx: usize,
    world: &WorldState,
    frame_graph: &FrameGraph,
    gt_points: &[Point3D<f64>],
    stereo_camera: &StereoCamera<f64>,
    states_before: &[(OptimizationState, FrameRole)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Set trajectory timeline
    rec.set_time_sequence("trajectory", frame_idx as i64);

    // Delete cameras that changed state (delete from old path before logging to new path)
    for idx in 0..(frame_idx + 1) {
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
                FrameRole::Keyframe => [0, 255, 0, 255], // Bright green for all keyframes
                FrameRole::Transient | FrameRole::Stored => match frame_state.state {
                    OptimizationState::Fixed => [255, 100, 100, 255], // Red for fixed
                    OptimizationState::Optimized => [100, 200, 255, 255], // Light blue for optimized transient
                    OptimizationState::Marginalize => [100, 200, 255, 255], // Light blue for marginalize
                    OptimizationState::Inactive => [50, 80, 120, 255], // Dark blue for inactive
                },
            };
            visualize_stereo_camera(
                rec,
                &format!("{}/cam_{:03}", path_prefix, idx),
                &world.frames[idx].world_pose(),
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
    let trajectory: Vec<[f32; 3]> = world
        .frames
        .iter()
        .map(|frame| {
            let t = frame.world_pose().translation;
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

/// Visualize GBA results on a separate timeline and entity paths
pub fn visualize_gba_update(
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
    let trajectory: Vec<[f32; 3]> = gba_world
        .frames
        .iter()
        .map(|frame| {
            let t = frame.world_pose().translation;
            [t.x as f32, t.y as f32, t.z as f32]
        })
        .collect();

    rec.log(
        "world/gba_estimate/trajectory",
        &rr::LineStrips3D::new([trajectory])
            .with_colors([[255, 165, 0]]) // Orange
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
            .with_colors([[255, 165, 0]]) // Orange
            .with_radii([0.035]),
    )?;

    // GBA cameras - only update those whose state changed
    for (idx, frame) in gba_world.frames.iter().enumerate() {
        let frame_state = &gba_frame_graph.states[idx];
        let new_state = frame_state.state;
        let new_role = frame_state.role;
        let path_prefix = gba_camera_path_prefix(new_state, new_role);

        // Determine if we need to update visualization (state changed or new frame)
        let should_update = if idx < gba_states_before.len() {
            if (gba_states_before[idx].0, gba_states_before[idx].1) != (new_state, new_role) {
                // Delete from old path
                let old_prefix =
                    gba_camera_path_prefix(gba_states_before[idx].0, gba_states_before[idx].1);
                let old_path = format!("{}/cam_{:03}", old_prefix, idx);
                rec.log(old_path.as_str(), &rr::Clear::recursive())?;
                true
            } else {
                false
            }
        } else {
            true // New frame, always update
        };

        if should_update {
            let color = match frame_state.role {
                FrameRole::Keyframe => [0, 255, 0, 255], // Bright green for keyframes
                FrameRole::Transient | FrameRole::Stored => match frame_state.state {
                    OptimizationState::Fixed => [255, 100, 100, 255],
                    OptimizationState::Optimized => [100, 200, 255, 255],
                    OptimizationState::Marginalize => [100, 200, 255, 255],
                    OptimizationState::Inactive => [50, 80, 120, 255],
                },
            };

            visualize_stereo_camera(
                rec,
                &format!("{}/cam_{:03}", path_prefix, idx),
                &frame.world_pose(),
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
