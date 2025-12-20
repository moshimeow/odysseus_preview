//! Frame graph for managing optimization states
//!
//! This module provides explicit state management for frames in SLAM,
//! separating structural roles (keyframe vs transient) from optimization
//! participation (optimized, fixed, stored, marginalized).
//!
//! Note: Poses and observations are stored separately from the frame graph.
//! The frame graph only tracks per-frame metadata about optimization state.

/// Structural role of a frame in the map
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameRole {
    /// Provides observation coverage, anchors map points
    Keyframe,
    /// Regular frame with no special structural role
    Transient,
    /// A reliable frame from GBA that isn't a keyframe
    Stored,
}

/// How a frame participates in bundle adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationState {
    /// Pose is actively optimized, observations used
    Optimized,
    /// Pose held constant (gauge anchor), observations used
    Fixed,
    /// Not currently being optimized, but observations retained for future use
    Inactive,
    /// Frame is marked to be marginalized
    Marginalize,
    // if a frame doesn't have any observations associate with it, it shouldn't be in the frame graph.

}

/// Per-frame state metadata
#[derive(Debug, Clone, Copy)]
pub struct FrameState {
    /// Structural role (keyframe or transient)
    pub role: FrameRole,
    /// Current optimization state
    pub state: OptimizationState,
}

impl FrameState {
    /// Create a new frame state
    pub fn new(role: FrameRole, state: OptimizationState) -> Self {
        Self { role, state }
    }

    /// Check if this frame contributes to BA (Optimized or Fixed)
    pub fn is_active_in_ba(&self) -> bool {
        matches!(self.state, OptimizationState::Optimized | OptimizationState::Fixed | OptimizationState::Marginalize)
    }

    /// Check if this frame should be optimized
    pub fn is_optimized(&self) -> bool {
        matches!(self.state, OptimizationState::Optimized | OptimizationState::Marginalize)
    }
}

/// Graph of frames with their optimization states
///
/// Index into the states vector corresponds to frame_idx.
/// Poses and observations are stored separately.
#[derive(Debug, Clone)]
pub struct FrameGraph {
    /// Per-frame state metadata. Index = frame_idx.
    pub states: Vec<FrameState>,
}

impl FrameGraph {
    /// Create a new empty frame graph
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
        }
    }

    /// Add a new frame state, returns the frame index
    pub fn add_frame(&mut self, role: FrameRole, state: OptimizationState) -> usize {
        let idx = self.states.len();
        self.states.push(FrameState::new(role, state));
        idx
    }

    /// Set the role of a frame
    pub fn set_role(&mut self, idx: usize, role: FrameRole) {
        self.states[idx].role = role;
    }

    /// Set the optimization state of a frame
    pub fn set_state(&mut self, idx: usize, state: OptimizationState) {
        self.states[idx].state = state;
    }

    /// Get frame state by index
    pub fn get(&self, idx: usize) -> Option<&FrameState> {
        self.states.get(idx)
    }

    /// Number of frames
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}
