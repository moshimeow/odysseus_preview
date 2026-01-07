//! Optimization graph visualization
//!
//! Provides visualization of the optimization problem structure showing
//! which poses observe which points and their optimization states.

use crate::frame_graph::{FrameGraph, FrameRole, OptimizationState};
use crate::geometry::StereoObservation;
use rerun as rr;
use std::collections::HashMap;

/// Optimization state for a point
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointState {
    /// Point parameters are being optimized
    Optimized,
    /// Point parameters are held fixed (used as constraints)
    Fixed,
    /// Point is excluded from optimization
    Inactive,
}

/// Information about the optimization graph structure
#[derive(Debug, Clone)]
pub struct OptimizationGraphInfo {
    /// Which poses observe which points: (camera_id, point_id)
    pub observations: Vec<(usize, usize)>,

    /// Optimization state per pose
    pub pose_states: HashMap<usize, OptimizationState>,

    /// Frame role per pose (Keyframe, Transient, Stored)
    pub pose_roles: HashMap<usize, FrameRole>,

    /// Optimization state per point
    pub point_states: HashMap<usize, PointState>,

    /// Problem dimensions
    pub n_poses: usize,
    pub n_points: usize,
}

impl OptimizationGraphInfo {
    /// Create new optimization graph info
    pub fn new(
        observations: Vec<(usize, usize)>,
        pose_states: HashMap<usize, OptimizationState>,
        pose_roles: HashMap<usize, FrameRole>,
        point_states: HashMap<usize, PointState>,
    ) -> Self {
        let n_poses = pose_states.len();
        let n_points = point_states.len();

        Self {
            observations,
            pose_states,
            pose_roles,
            point_states,
            n_poses,
            n_points,
        }
    }
}

/// Build optimization graph info from optimization problem data
pub fn build_graph_info(
    observations: &[StereoObservation],
    frame_graph: &FrameGraph,
    _pose_to_param_idx: &HashMap<usize, usize>,
    point_to_param_idx: &HashMap<usize, usize>,
    fixed_point_ids: &std::collections::HashSet<usize>,
) -> OptimizationGraphInfo {
    // Extract observations as (camera_id, point_id) pairs
    let obs_pairs: Vec<(usize, usize)> = observations
        .iter()
        .map(|obs| (obs.camera_id, obs.point_id))
        .collect();

    // Build pose states from frame graph
    let mut pose_states = HashMap::new();
    let mut pose_roles = HashMap::new();
    for (frame_idx, frame_state) in frame_graph.states.iter().enumerate() {
        pose_states.insert(frame_idx, frame_state.state);
        pose_roles.insert(frame_idx, frame_state.role);
    }

    // Build point states
    // A point is Optimized if it's in point_to_param_idx and NOT in fixed_point_ids
    // A point is Fixed if it's in fixed_point_ids OR not in point_to_param_idx
    let mut point_states = HashMap::new();

    // Get all unique point IDs from observations
    let all_point_ids: std::collections::HashSet<usize> = observations
        .iter()
        .map(|obs| obs.point_id)
        .collect();

    for &point_id in &all_point_ids {
        let state = if point_to_param_idx.contains_key(&point_id) {
            // Point is being optimized
            PointState::Optimized
        } else if fixed_point_ids.contains(&point_id) {
            // Point is explicitly fixed (used as constraint in optimization)
            PointState::Fixed
        } else {
            // Point was excluded from optimization (e.g., by point selection)
            // or only observed by inactive frames
            PointState::Inactive
        };
        point_states.insert(point_id, state);
    }

    OptimizationGraphInfo::new(obs_pairs, pose_states, pose_roles, point_states)
}

/// Visualize the optimization graph in Rerun
///
/// Creates a 2D connectivity graph showing which poses observe which points,
/// color-coded by optimization state.
/// Only shows rows for points that are Optimized, Marginalized, or Fixed (not Inactive).
pub fn visualize_optimization_graph(
    rec: &rr::RecordingStream,
    frame_idx: usize,
    graph_info: &OptimizationGraphInfo,
) -> Result<(), Box<dyn std::error::Error>> {
    // Set timeline to sync with 3D visualization
    rec.set_time_sequence("trajectory", frame_idx as i64);

    // Build a mapping from point_id to compact row index
    // Only include points that are:
    // - Optimized (always shown), OR
    // - Fixed AND observed by at least one non-fixed frame

    // First, find which points are observed by non-fixed frames
    let mut points_with_nonfixed_observers: std::collections::HashSet<usize> =
        std::collections::HashSet::new();

    for &(camera_id, point_id) in &graph_info.observations {
        if let Some(&pose_state) = graph_info.pose_states.get(&camera_id) {
            if pose_state != OptimizationState::Fixed {
                points_with_nonfixed_observers.insert(point_id);
            }
        }
    }

    let mut active_point_ids: Vec<usize> = graph_info
        .point_states
        .iter()
        .filter_map(|(point_id, state)| {
            match state {
                PointState::Optimized => Some(*point_id),
                PointState::Fixed => {
                    // Only show fixed points if they have at least one non-fixed observer
                    if points_with_nonfixed_observers.contains(point_id) {
                        Some(*point_id)
                    } else {
                        None
                    }
                }
                PointState::Inactive => None,
            }
        })
        .collect();

    // Sort for consistent row ordering
    active_point_ids.sort_unstable();

    // Create mapping: point_id -> compact_row_index
    let point_to_row: HashMap<usize, usize> = active_point_ids
        .iter()
        .enumerate()
        .map(|(row_idx, &point_id)| (point_id, row_idx))
        .collect();

    // Convert observations to 2D points
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let mut radii = Vec::new();

    for &(camera_id, point_id) in &graph_info.observations {
        // Only visualize observations of active points
        let row_idx = match point_to_row.get(&point_id) {
            Some(&idx) => idx,
            None => continue, // Skip inactive points
        };

        // X-axis: camera/pose index
        // Y-axis: compact row index (not point ID)
        let x = camera_id as f32;
        let y = row_idx as f32;
        positions.push([x, y]);

        // Determine color based on pose and point states
        let pose_state = graph_info.pose_states.get(&camera_id);
        let point_state = graph_info.point_states.get(&point_id);

        let color = match (pose_state, point_state) {
            // Inactive pose or inactive point: gray
            (Some(OptimizationState::Inactive), _) | (_, Some(PointState::Inactive)) => [150, 150, 150, 255],

            // Pose being marginalized: orange
            (Some(OptimizationState::Marginalize), _) => [255, 165, 0, 255],

            // Point is fixed (constraining the optimization): red
            (_, Some(PointState::Fixed)) => [255, 100, 100, 255],

            // Both optimized: bright green
            (Some(OptimizationState::Optimized), Some(PointState::Optimized)) => [0, 255, 0, 255],

            // Fixed pose with optimized point: green (pose fixed but point being optimized)
            (Some(OptimizationState::Fixed), Some(PointState::Optimized)) => [0, 255, 0, 255],

            // Default: white
            _ => [255, 255, 255, 255],
        };
        colors.push(color);

        // Size based on frame role (keyframes larger)
        let pose_role = graph_info.pose_roles.get(&camera_id);
        let radius = match pose_role {
            Some(FrameRole::Keyframe) => 2.0,
            Some(FrameRole::Transient) => 1.0,
            Some(FrameRole::Stored) => 1.5,
            None => 1.0,
        };
        radii.push(radius);
    }

    // Log the observation points
    if !positions.is_empty() {
        rec.log(
            "optimization/graph/observations",
            &rr::Points2D::new(positions)
                .with_colors(colors)
                .with_radii(radii),
        )?;
    }

    // Add a row of pose state indicators at the bottom
    let mut pose_positions = Vec::new();
    let mut pose_colors = Vec::new();
    let mut pose_radii = Vec::new();

    for (&camera_id, &pose_state) in &graph_info.pose_states {
        let x = camera_id as f32;
        let y = -1.0; // Bottom row

        pose_positions.push([x, y]);

        // Color based on pose optimization state
        let color = match pose_state {
            OptimizationState::Optimized => [0, 255, 0, 255],      // Green
            OptimizationState::Fixed => [255, 100, 100, 255],      // Red
            OptimizationState::Marginalize => [255, 165, 0, 255],  // Orange
            OptimizationState::Inactive => [150, 150, 150, 255],   // Gray
        };
        pose_colors.push(color);

        // Size based on frame role (keyframes larger)
        let pose_role = graph_info.pose_roles.get(&camera_id);
        let radius = match pose_role {
            Some(FrameRole::Keyframe) => 3.0,  // Larger for visibility
            Some(FrameRole::Transient) => 2.0,
            Some(FrameRole::Stored) => 2.5,
            None => 2.0,
        };
        pose_radii.push(radius);
    }

    // Log the pose state indicators
    if !pose_positions.is_empty() {
        rec.log(
            "optimization/graph/pose_states",
            &rr::Points2D::new(pose_positions)
                .with_colors(pose_colors)
                .with_radii(pose_radii),
        )?;
    }

    Ok(())
}
