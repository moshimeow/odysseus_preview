//! World state containing poses and 3D points
//!
//! Bundles together the camera poses and 3D point map, which together
//! represent "our current estimate of the world". Both are optimized
//! together during bundle adjustment.

use crate::camera::StereoCamera;
use crate::geometry::{Point3D, StereoObservation};
use crate::math::SE3;
use std::collections::HashMap;
use odysseus_solver::math3d::Vec3;

/// Point information with observation tracking
#[derive(Debug, Clone)]
pub struct PointInfo {
    /// Point position in world coordinates
    pub position: Vec3<f64>,
    /// Last frame index where this point was observed
    pub last_observed_frame: usize,
    /// Number of times this point has been observed
    pub observation_count: usize,
}

impl PointInfo {
    /// Create a new point info
    pub fn new(position: Vec3<f64>, first_observed_frame: usize) -> Self {
        Self {
            position,
            last_observed_frame: first_observed_frame,
            observation_count: 1,
        }
    }

    /// Update when point is observed again
    pub fn observe(&mut self, frame_idx: usize) {
        self.last_observed_frame = frame_idx;
        self.observation_count += 1;
    }

    /// Check if point was observed recently (within N frames)
    pub fn is_recent(&self, current_frame: usize, window_size: usize) -> bool {
        if current_frame < self.last_observed_frame {
            return false;
        }
        (current_frame - self.last_observed_frame) <= window_size
    }
}

/// Camera pose with host-relative rotation parameterization
///
/// Stores rotation as a small delta from a host quaternion to keep parameters
/// small during optimization, avoiding the rotation vector singularity at 2π.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    /// Rotation delta as axis-angle (should be small)
    pub rotation: Vec3<f64>,
    /// Translation in world coordinates
    pub translation: Vec3<f64>,
    /// Host quaternion - world rotation is `host * exp(rotation)`
    pub rotation_host: odysseus_solver::math3d::Quat<f64>,
}

impl Pose {
    /// Create a new pose from a world-space SE3
    ///
    /// Sets rotation_host to the world rotation and rotation delta to zero.
    pub fn from_se3(world_pose: SE3<f64>) -> Self {
        Self {
            rotation: Vec3::new(0.0, 0.0, 0.0),
            translation: world_pose.translation,
            rotation_host: world_pose.rotation.quat,
        }
    }

    /// Create an identity pose at the origin
    pub fn identity() -> Self {
        Self {
            rotation: Vec3::new(0.0, 0.0, 0.0),
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation_host: odysseus_solver::math3d::Quat::identity(),
        }
    }

    /// Get the world rotation as a quaternion
    pub fn world_rotation(&self) -> odysseus_solver::math3d::Quat<f64> {
        let q_delta = odysseus_solver::math3d::Quat::from_axis_angle(self.rotation);
        (self.rotation_host * q_delta).normalize()
    }

    /// Get the full world-space pose as SE3
    pub fn to_se3(&self) -> SE3<f64> {
        SE3::from_rotation_translation(
            crate::math::SO3 { quat: self.world_rotation() },
            self.translation,
        )
    }

    /// Set from optimization parameters (rotation delta + world translation)
    pub fn set_from_params(&mut self, rotation_delta: Vec3<f64>, translation: Vec3<f64>) {
        self.rotation = rotation_delta;
        self.translation = translation;
    }

    /// Transform a point from camera coordinates to world coordinates
    pub fn camera_to_world(&self, camera_point: Vec3<f64>) -> Vec3<f64> {
        self.to_se3().transform_point(camera_point)
    }

    /// Transform a point from world coordinates to camera coordinates
    pub fn world_to_camera(&self, world_point: Vec3<f64>) -> Vec3<f64> {
        self.to_se3().inverse().transform_point(world_point)
    }
}

/// A single frame of the world state
#[derive(Clone)]
pub struct WorldFrame {
    /// Camera pose (stores rotation delta, translation, and host)
    pub pose: Pose,
    /// The points stored in world coordinates
    /// If this has points, it means this frame is a keyframe
    pub points: HashMap<usize, PointInfo>,
}

impl WorldFrame {
    /// Create a new world frame from a world-space pose
    pub fn new(world_pose: SE3<f64>, points: Option<HashMap<usize, PointInfo>>) -> Self {
        Self {
            pose: Pose::from_se3(world_pose),
            points: points.unwrap_or_default(),
        }
    }

    /// Get the pose in world coordinates as SE3
    pub fn world_pose(&self) -> SE3<f64> {
        self.pose.to_se3()
    }

    /// Convert a point from this frame's camera coordinates to world coordinates
    pub fn camera_to_world(&self, camera_point: Vec3<f64>) -> Vec3<f64> {
        self.pose.camera_to_world(camera_point)
    }

    /// Convert a point from world coordinates to this frame's camera coordinates
    pub fn world_to_camera(&self, world_point: Vec3<f64>) -> Vec3<f64> {
        self.pose.world_to_camera(world_point)
    }
}

/// The estimated state of the world
///
/// Contains camera poses and the 3D point map. Points are stored in world coordinates.
/// These are bundled together because:
/// 1. They're both system parameters that get optimized
/// 2. They're almost always passed together
#[derive(Clone)]
pub struct WorldState {
    /// Frames containing poses and points stored in world coordinates
    pub frames: Vec<WorldFrame>,
    /// Map from point ID to the keyframe index that owns it
    point_to_keyframe: HashMap<usize, usize>,
    /// Next available point ID
    next_point_id: usize,
}

impl WorldState {
    /// Create a new empty world state
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            point_to_keyframe: HashMap::new(),
            next_point_id: 0,
        }
    }

    // ========== Pose methods ==========

    /// Add a new pose, returns the frame index
    pub fn add_pose(&mut self, pose: SE3<f64>) -> usize {
        let idx = self.frames.len();
        self.frames.push(WorldFrame::new(pose, None));
        idx
    }

    /// Add a complete frame (pose + points), returns the frame index
    /// 
    /// This properly registers all points in the frame to point_to_keyframe.
    /// Use this when receiving a frame from another thread (e.g., LBA→GBA).
    pub fn add_frame(&mut self, frame: WorldFrame) -> usize {
        let idx = self.frames.len();
        // Register all points in this frame
        for &point_id in frame.points.keys() {
            self.point_to_keyframe.insert(point_id, idx);
            if point_id >= self.next_point_id {
                self.next_point_id = point_id + 1;
            }
        }
        self.frames.push(frame);
        idx
    }

    /// Get the current number of frames
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Get the world pose for a frame
    pub fn get_pose(&self, frame_idx: usize) -> Option<SE3<f64>> {
        self.frames.get(frame_idx).map(|f| f.world_pose())
    }

    // ========== Point methods (inverse depth) ==========

    /// Triangulate a stereo observation and add the point in inverse depth form
    ///
    /// This converts pixel coordinates directly to inverse depth without
    /// constructing intermediate 3D points.
    ///
    /// # Arguments
    /// * `obs` - Stereo observation to triangulate
    /// * `camera` - Stereo camera model
    /// * `keyframe_idx` - Index of the keyframe that owns this point
    ///
    /// # Returns
    /// * `true` if triangulation succeeded and point was added
    /// * `false` if triangulation failed (e.g., negative disparity)
    pub fn triangulate_and_add_point(
        &mut self,
        obs: &StereoObservation,
        camera: &StereoCamera<f64>,
        keyframe_idx: usize,
    ) -> bool {
        if keyframe_idx >= self.frames.len() {
            panic!("Keyframe index {} out of bounds ({} frames)", keyframe_idx, self.frames.len());
        }

        // Triangulate from pixel coordinates and disparity
        let disparity = obs.left_u - obs.right_u;
        if disparity <= 0.0 {
            return false;
        }

        // Normalized image coordinates
        let ray_x = (obs.left_u - camera.left.cx) / camera.left.fx;
        let ray_y = (obs.left_v - camera.left.cy) / camera.left.fy;
        let ray_z = 1.0;

        // Ray length factor
        let ray_length = (ray_x * ray_x + ray_y * ray_y + ray_z * ray_z).sqrt();

        // Z-depth from disparity
        let z_depth = camera.baseline * camera.left.fx / disparity;
        if z_depth <= 0.0 || !z_depth.is_finite() {
            return false;
        }

        // Actual distance along the ray
        let distance = z_depth * ray_length;

        // Point in camera frame
        let camera_point = Vec3::new(ray_x * distance / ray_length, ray_y * distance / ray_length, ray_z * distance / ray_length);

        // Transform to world coordinates
        let world_point = self.frames[keyframe_idx].camera_to_world(camera_point);

        // Store the point
        let point_info = PointInfo::new(world_point, keyframe_idx);
        self.frames[keyframe_idx].points.insert(obs.point_id, point_info);
        self.point_to_keyframe.insert(obs.point_id, keyframe_idx);
        if obs.point_id >= self.next_point_id {
            self.next_point_id = obs.point_id + 1;
        }
        true
    }

    // ========== Point methods ==========

    /// Get a point's position in world coordinates
    pub fn get_point_xyz(&self, point_id: usize) -> Option<Point3D<f64>> {
        let keyframe_idx = self.point_to_keyframe.get(&point_id)?;
        let frame = &self.frames[*keyframe_idx];
        let point_info = frame.points.get(&point_id)?;
        Some(point_info.position)
    }

    /// Update a point's position from world coordinates
    pub fn update_point_xyz(&mut self, point_id: usize, new_world_position: Point3D<f64>) {
        if let Some(&keyframe_idx) = self.point_to_keyframe.get(&point_id) {
            if let Some(point_info) = self.frames[keyframe_idx].points.get_mut(&point_id) {
                point_info.position = new_world_position;
            }
        }
    }

    /// Mark a point as observed in the given frame
    pub fn observe_point(&mut self, point_id: usize, frame_idx: usize) {
        if let Some(keyframe_idx) = self.point_to_keyframe.get(&point_id) {
            if let Some(point_info) = self.frames[*keyframe_idx].points.get_mut(&point_id) {
                point_info.observe(frame_idx);
            }
        }
    }

    /// Get the keyframe index for a point
    pub fn get_point_keyframe(&self, point_id: usize) -> Option<usize> {
        self.point_to_keyframe.get(&point_id).copied()
    }

    /// Get a point's position
    pub fn get_point(&self, point_id: usize) -> Option<Point3D<f64>> {
        self.get_point_xyz(point_id)
    }

    /// Add a new point to the map
    ///
    /// This stores the point on the frame where it was first observed.
    /// Returns the ID assigned to this point
    pub fn add_point(&mut self, position: Point3D<f64>, first_observed_frame: usize) -> usize {
        let point_id = self.next_point_id;
        self.next_point_id += 1;
        // Store on the frame where it was first observed
        if first_observed_frame >= self.frames.len() {
            panic!("Frame index {} out of bounds for add_point", first_observed_frame);
        }
        self.add_point_xyz(point_id, first_observed_frame, position);
        point_id
    }

    /// Add a new point to the map with a specific ID
    ///
    /// This is useful when point IDs come from observations (e.g., feature tracking)
    /// and need to match specific IDs.
    ///
    /// Returns the ID (should match requested_id)
    pub fn add_point_with_id(&mut self, position: Point3D<f64>, first_observed_frame: usize, requested_id: usize) -> usize {
        if first_observed_frame >= self.frames.len() {
            panic!("Frame index {} out of bounds for add_point_with_id", first_observed_frame);
        }
        self.add_point_xyz(requested_id, first_observed_frame, position);
        requested_id
    }
    
    /// Add a point from world coordinates
    ///
    /// This is a helper for backward compatibility methods.
    fn add_point_xyz(&mut self, point_id: usize, keyframe_idx: usize, world_point: Point3D<f64>) {
        // Store the point directly
        let point_info = PointInfo::new(world_point, keyframe_idx);
        self.frames[keyframe_idx].points.insert(point_id, point_info);
        self.point_to_keyframe.insert(point_id, keyframe_idx);
        if point_id >= self.next_point_id {
            self.next_point_id = point_id + 1;
        }
    }

    /// Update an existing point's position
    pub fn update_point(&mut self, point_id: usize, new_position: Point3D<f64>) {
        self.update_point_xyz(point_id, new_position);
    }

    /// Replace frames with those from another WorldState
    ///
    /// Replaces `self.frames[0..other.frames.len()]` with `other.frames`,
    /// updating poses and point_to_keyframe accordingly. Frames in self
    /// beyond other's length are kept unchanged.
    ///
    /// Use this when merging GBA results back into LBA state - GBA always
    /// has fewer frames (older snapshot), so this replaces all the frames
    /// GBA optimized while keeping newer frames LBA added.
    pub fn replace_frames_from(&mut self, other: &WorldState) {
        let n = other.frames.len();
        
        // Remove point_to_keyframe entries for points in frames we're replacing
        for i in 0..n.min(self.frames.len()) {
            for point_id in self.frames[i].points.keys() {
                self.point_to_keyframe.remove(point_id);
            }
        }
        
        // Replace frames
        for (i, frame) in other.frames.iter().enumerate() {
            if i < self.frames.len() {
                self.frames[i] = frame.clone();
            }
        }
        
        // Rebuild point_to_keyframe for replaced frames
        for (i, frame) in self.frames.iter().enumerate().take(n) {
            for &point_id in frame.points.keys() {
                self.point_to_keyframe.insert(point_id, i);
                if point_id >= self.next_point_id {
                    self.next_point_id = point_id + 1;
                }
            }
        }
    }

    /// Get all points that were observed within the sliding window
    ///
    /// # Arguments
    /// * `current_frame` - Current frame index
    /// * `window_size` - Number of frames to look back
    ///
    /// # Returns
    /// Vector of (point_id, position) for recent points
    pub fn get_recent_points(&self, current_frame: usize, window_size: usize) -> Vec<(usize, Point3D<f64>)> {
        let mut result = Vec::new();
        for (point_id, keyframe_idx) in &self.point_to_keyframe {
            if let Some(point_info) = self.frames[*keyframe_idx].points.get(point_id) {
                if point_info.is_recent(current_frame, window_size) {
                    if let Some(xyz) = self.get_point_xyz(*point_id) {
                        result.push((*point_id, xyz));
                    }
                }
            }
        }
        result
    }

    /// Get all points in the map
    pub fn get_all_points(&self) -> Vec<(usize, Point3D<f64>)> {
        let mut result = Vec::new();
        for (point_id, keyframe_idx) in &self.point_to_keyframe {
            let keyframe_idx = *keyframe_idx;
            if let Some(point_info) = self.frames[keyframe_idx].points.get(point_id) {
                result.push((*point_id, point_info.position));
            }
        }
        result
    }

    /// Number of points in the map
    pub fn num_points(&self) -> usize {
        self.point_to_keyframe.len()
    }

    /// Remove points that haven't been observed in a very long time
    ///
    /// This is for cleaning up the map when points are no longer relevant
    pub fn prune_old_points(&mut self, current_frame: usize, max_frames_unseen: usize) {
        let mut points_to_remove = Vec::new();
        for (point_id, keyframe_idx) in &self.point_to_keyframe {
            if let Some(point_info) = self.frames[*keyframe_idx].points.get(point_id) {
                let should_keep = if current_frame < point_info.last_observed_frame {
                    true  // Keep future observations (shouldn't happen but be safe)
                } else {
                    (current_frame - point_info.last_observed_frame) <= max_frames_unseen
                };
                if !should_keep {
                    points_to_remove.push(*point_id);
                }
            }
        }
        for point_id in points_to_remove {
            if let Some(keyframe_idx) = self.point_to_keyframe.remove(&point_id) {
                self.frames[keyframe_idx].points.remove(&point_id);
            }
        }
    }
}

impl Default for WorldState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use odysseus_solver::math3d::Vec3;

    #[test]
    fn test_add_and_get_point() {
        let mut world = WorldState::new();
        // Add frame 0 first
        world.add_pose(SE3::identity());
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let id = world.add_point(pos, 0);

        assert_eq!(id, 0);
        assert_eq!(world.num_points(), 1);

        let retrieved = world.get_point(id).unwrap();
        assert_eq!(retrieved.x, 1.0);
        assert_eq!(retrieved.y, 2.0);
        assert_eq!(retrieved.z, 3.0);
    }

    #[test]
    fn test_update_point() {
        let mut world = WorldState::new();
        // Add frame 0 first
        world.add_pose(SE3::identity());
        let id = world.add_point(Vec3::new(1.0, 2.0, 3.0), 0);

        world.update_point(id, Vec3::new(4.0, 5.0, 6.0));

        let updated = world.get_point(id).unwrap();
        assert_eq!(updated.x, 4.0);
        assert_eq!(updated.y, 5.0);
        assert_eq!(updated.z, 6.0);
    }

    #[test]
    fn test_recent_points() {
        let mut world = WorldState::new();
        // Add frames first
        for _ in 0..11 {
            world.add_pose(SE3::identity());
        }

        // Add points observed at different frames
        let _id0 = world.add_point(Vec3::new(0.0, 0.0, 0.0), 0);  // Frame 0
        let _id5 = world.add_point(Vec3::new(1.0, 1.0, 1.0), 5);  // Frame 5
        let id10 = world.add_point(Vec3::new(2.0, 2.0, 2.0), 10); // Frame 10

        // At frame 12 with window size 5:
        // - id0 (frame 0): not recent (12 - 0 = 12 > 5)
        // - id5 (frame 5): not recent (12 - 5 = 7 > 5)
        // - id10 (frame 10): recent (12 - 10 = 2 <= 5)
        let recent = world.get_recent_points(12, 5);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].0, id10);

        // At frame 12 with window size 10:
        // - id0 (frame 0): not recent (12 - 0 = 12 > 10)
        // - id5 (frame 5): recent (12 - 5 = 7 <= 10)
        // - id10 (frame 10): recent (12 - 10 = 2 <= 10)
        let recent = world.get_recent_points(12, 10);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_prune_old_points() {
        let mut world = WorldState::new();
        // Add frames first
        for _ in 0..46 {
            world.add_pose(SE3::identity());
        }

        let _id0 = world.add_point(Vec3::new(0.0, 0.0, 0.0), 0);   // Very old
        let _id10 = world.add_point(Vec3::new(1.0, 1.0, 1.0), 10); // Somewhat old
        let _id45 = world.add_point(Vec3::new(2.0, 2.0, 2.0), 45); // Recent

        assert_eq!(world.num_points(), 3);

        // At frame 50, prune points not seen in last 20 frames
        // - id0 (frame 0): 50 - 0 = 50 > 20 => PRUNE
        // - id10 (frame 10): 50 - 10 = 40 > 20 => PRUNE
        // - id45 (frame 45): 50 - 45 = 5 <= 20 => KEEP
        world.prune_old_points(50, 20);

        assert_eq!(world.num_points(), 1);
    }

    #[test]
    fn test_add_pose() {
        let mut world = WorldState::new();
        
        let pose = SE3::identity();
        let idx = world.add_pose(pose);
        
        assert_eq!(idx, 0);
        assert_eq!(world.num_frames(), 1);
        assert!(world.get_pose(0).is_some());
    }
}

