//! SLAM System with parallel LBA/GBA
//!
//! Orchestrates Local Bundle Adjustment (LBA) running synchronously on the main thread
//! and Global Bundle Adjustment (GBA) running asynchronously on a background thread.
//!
//! - LBA: Fast, optimizes only window frames with keyframes Fixed
//! - GBA: Thorough, optimizes all keyframes + window, runs continuously

use crate::camera::StereoCamera;
use crate::frame_graph::{FrameGraph, FrameRole, OptimizationState};
use crate::geometry::StereoObservation;
use crate::world_state::WorldState;
use crate::optimization::{run_bundle_adjustment, BundleAdjustmentConfig};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::collections::HashSet;

use crate::world_state::WorldFrame;

/// Message from LBA to GBA: a single new frame
/// GBA derives keyframe information from which frames have points.
#[derive(Clone)]
pub struct LbaToGbaMsg {
    /// The new frame (pose + any points stored on it if keyframe)
    pub frame: WorldFrame,
    /// Index of this frame in the world state
    pub frame_index: usize,
}

/// Message from GBA to LBA: optimized world state
#[derive(Clone)]
pub struct GbaToLbaMsg {
    /// The fully optimized WorldState from GBA
    pub world_state: WorldState,
    /// GBA's frame graph (for visualization/debugging)
    pub frame_graph: FrameGraph,
    /// Index of the most recent frame GBA optimized (for gauge fixing)
    /// This is typically the window frame, not a keyframe
    pub last_optimized_frame: usize,
}

/// SLAM System managing GBA threads
pub struct SlamSystem {
    /// Sender to GBA thread
    to_gba: Sender<LbaToGbaMsg>,
    /// Receiver from GBA thread  
    from_gba: Receiver<GbaToLbaMsg>,
    /// GBA thread handle
    gba_handle: Option<JoinHandle<()>>,
    /// Index of most recent GBA-optimized frame (for LBA gauge fixing)
    /// This is typically the window frame GBA optimized, not a keyframe
    pub last_gba_frame: Option<usize>,
}

impl SlamSystem {
    /// Create a new SLAM system and spawn the GBA thread
    /// 
    /// # Arguments
    /// * `stereo_camera` - Camera model
    /// * `frame_observations` - Shared reference to all observations (read-only)
    pub fn new(
        stereo_camera: StereoCamera<f64>,
        frame_observations: Arc<Vec<Vec<StereoObservation>>>,
    ) -> Self {
        let (to_gba_tx, to_gba_rx) = mpsc::channel::<LbaToGbaMsg>();
        let (from_gba_tx, from_gba_rx) = mpsc::channel::<GbaToLbaMsg>();
        
        let gba_camera = stereo_camera.clone();
        let gba_handle = thread::spawn(move || {
            gba_thread_loop(to_gba_rx, from_gba_tx, gba_camera, frame_observations);
        });
        
        Self {
            to_gba: to_gba_tx,
            from_gba: from_gba_rx,
            gba_handle: Some(gba_handle),
            last_gba_frame: None,
        }
    }
    
    /// Send the latest frame to GBA for optimization
    /// 
    /// # Arguments
    /// * `frame_index` - Index of the frame to send
    /// * `world` - Current world state (only the specified frame is extracted)
    pub fn send_to_gba(&self, frame_index: usize, world: &WorldState) {
        let msg = LbaToGbaMsg {
            frame: world.frames[frame_index].clone(),
            frame_index,
        };
        // Ignore send errors (GBA thread may have exited)
        let _ = self.to_gba.send(msg);
    }
    
    /// Check for GBA results (non-blocking)
    /// Returns Some(msg) if GBA sent results, None otherwise
    pub fn try_recv_from_gba(&mut self) -> Option<GbaToLbaMsg> {
        match self.from_gba.try_recv() {
            Ok(msg) => {
                self.last_gba_frame = Some(msg.last_optimized_frame);
                Some(msg)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => None,
        }
    }
}

impl Drop for SlamSystem {
    fn drop(&mut self) {
        // Drop the sender to signal GBA thread to exit
        drop(std::mem::replace(&mut self.to_gba, mpsc::channel().0));
        
        // Wait for GBA thread to finish
        if let Some(handle) = self.gba_handle.take() {
            let _ = handle.join();
        }
    }
}

/// GBA window size (just the most recent frame beyond keyframes)
const GBA_WINDOW_SIZE: usize = 2;

/// GBA thread main loop
fn gba_thread_loop(
    from_lba: Receiver<LbaToGbaMsg>,
    to_lba: Sender<GbaToLbaMsg>,
    stereo_camera: StereoCamera<f64>,
    frame_observations: Arc<Vec<Vec<StereoObservation>>>,
) {
    // GBA state
    let mut gba_world: Option<WorldState> = None;
    let mut gba_graph = FrameGraph::new();
    
    loop {
        // Block for at least one message, then collect all pending
        // (can't skip messages since each contains only one frame)
        let mut messages = Vec::new();
        
        // Blocking wait for first message
        match from_lba.recv() {
            Ok(msg) => messages.push(msg),
            Err(_) => return, // LBA exited
        };
        
        // Drain any additional pending messages
        while let Ok(msg) = from_lba.try_recv() {
            messages.push(msg);
        }
        
        // Sort by frame index to ensure correct ordering
        messages.sort_by_key(|m| m.frame_index);
        
        // Process all messages - add new frames to GBA state
        for msg in messages {
            if let Some(ref mut world) = gba_world {
                // Only add if this is a new frame (skip duplicates)
                if msg.frame_index >= world.num_frames() {
                    // Frames should arrive in order
                    debug_assert_eq!(msg.frame_index, world.num_frames(), 
                        "Frame index mismatch: expected {}, got {}", 
                        world.num_frames(), msg.frame_index);
                    if !msg.frame.points.is_empty() {
                        gba_graph.add_frame(FrameRole::Keyframe, OptimizationState::Optimized);
                    } else {
                        gba_graph.add_frame(FrameRole::Transient, OptimizationState::Optimized);
                    }
                    world.add_frame(msg.frame);
                }
            } else {
                // First message - initialize GBA state
                debug_assert_eq!(msg.frame_index, 0, "First message should be frame 0");
                let mut world = WorldState::new();
                world.add_frame(msg.frame);
                gba_world = Some(world);
                gba_graph.add_frame(FrameRole::Keyframe, OptimizationState::Fixed);
            }
        }
        
        // Run GBA optimization (we always have state after processing messages)
        if let Some(world) = &mut gba_world {
            // The last optimized frame is the most recent frame in the graph
            // (GBA optimizes keyframes + window, window is the last frame)
            let optimized_frame = world.num_frames().saturating_sub(1);

            // Build the GBA-specific frame graph (all keyframes optimized, window optimized)
            // Keyframes are derived from world state: frames with points are keyframes
        
            // Manage window: marginalize old transient frames
            let mut transient_indices: Vec<usize> = gba_graph.states.iter().enumerate()
            .filter(|(_, s)| s.state == OptimizationState::Optimized && s.role == FrameRole::Transient)
            .map(|(idx, _)| idx)
            .collect();

            // cleanup the any bridging frames and remove the oldest frame in the window.
            while transient_indices.len() > GBA_WINDOW_SIZE {
                gba_graph.set_state(transient_indices[0], OptimizationState::Inactive);
                transient_indices.remove(0);
            }

            // currently no maximum number of keyframes.
            
            // Run one GBA optimization cycle (no marginalization for GBA)
            run_bundle_adjustment(
                &stereo_camera,
                &gba_graph,
                world,
                &frame_observations,
                None,  // No prior for GBA
                &HashSet::new(),
                &BundleAdjustmentConfig::gba(),
            );
            
            // Send results back to LBA
            let result = GbaToLbaMsg {
                world_state: world.clone(),
                frame_graph: gba_graph.clone(),
                last_optimized_frame: optimized_frame,
            };
            
            if to_lba.send(result).is_err() {
                return; // LBA exited
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_slam_system_creation() {
        let camera = StereoCamera::simple(500.0, 640.0, 480.0, 0.1);
        let _system = SlamSystem::new(camera, Arc::new(Vec::new()));
        // System should create and drop cleanly
    }
}

