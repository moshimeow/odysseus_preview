//! Shared utility functions for SLAM demos and tests

use crate::math::SE3;
use odysseus_solver::math3d::Vec3;
use std::fs::File;
use std::io::{BufReader, Read, Result};

/// Get current resident set size (actual physical memory used) in MB
pub fn get_rss_mb() -> f64 {
    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

/// Get peak memory usage (high water mark) in MB
pub fn get_peak_rss_mb() -> f64 {
    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if line.starts_with("VmHWM:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

/// Load point cloud from binary file (Blender export format)
///
/// File format:
/// - u32: number of vertices
/// - For each vertex: 3x f32 (x, y, z)
pub fn load_point_cloud(path: &str) -> Result<Vec<[f64; 3]>> {
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

/// Load point cloud and convert to Vec3 format
pub fn load_point_cloud_vec3(path: &str) -> Result<Vec<Vec3<f64>>> {
    let raw = load_point_cloud(path)?;
    Ok(raw
        .into_iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect())
}

/// Load camera poses from binary file (Blender export format)
///
/// File format:
/// - u32: number of frames
/// - For each frame: 4x4 f32 matrix (row-major)
pub fn load_camera_poses(path: &str) -> Result<Vec<SE3<f64>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let num_frames = u32::from_le_bytes(buf) as usize;

    let mut poses = Vec::with_capacity(num_frames);
    for _ in 0..num_frames {
        let mut matrix = [[0.0f32; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                let mut val_buf = [0u8; 4];
                reader.read_exact(&mut val_buf)?;
                matrix[row][col] = f32::from_le_bytes(val_buf);
            }
        }

        // Convert 4x4 matrix to SE3
        // Extract rotation (top-left 3x3) as column vectors
        let x_axis = Vec3::new(
            matrix[0][0] as f64,
            matrix[1][0] as f64,
            matrix[2][0] as f64,
        );
        let y_axis = Vec3::new(
            matrix[0][1] as f64,
            matrix[1][1] as f64,
            matrix[2][1] as f64,
        );
        let z_axis = Vec3::new(
            matrix[0][2] as f64,
            matrix[1][2] as f64,
            matrix[2][2] as f64,
        );

        let rotation_matrix = odysseus_solver::math3d::Mat3::from_cols(x_axis, y_axis, z_axis);
        let rotation = crate::math::SO3::from_matrix(rotation_matrix);

        let translation = Vec3::new(
            matrix[0][3] as f64,
            matrix[1][3] as f64,
            matrix[2][3] as f64,
        );

        poses.push(SE3 {
            rotation,
            translation,
        });
    }

    Ok(poses)
}
