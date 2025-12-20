//! Visualize greeble_room geometry and camera trajectory in Rerun
//!
//! First export from Blender:
//!   blender greeble_room.blend --background --python export_greeble_room.py
//!
//! Then run this:
//!   cargo run --example visualize_greeble_room

use rerun as rr;
use std::fs::File;
use std::io::{BufReader, Read};

/// Read point cloud (vertices only) from binary format
fn load_point_cloud(path: &str) -> Result<Vec<[f32; 3]>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read number of vertices
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let num_vertices = u32::from_le_bytes(buf) as usize;

    // Read vertices
    let mut vertices = Vec::with_capacity(num_vertices);
    for _ in 0..num_vertices {
        let mut x_buf = [0u8; 4];
        let mut y_buf = [0u8; 4];
        let mut z_buf = [0u8; 4];

        reader.read_exact(&mut x_buf)?;
        reader.read_exact(&mut y_buf)?;
        reader.read_exact(&mut z_buf)?;

        let x = f32::from_le_bytes(x_buf);
        let y = f32::from_le_bytes(y_buf);
        let z = f32::from_le_bytes(z_buf);

        vertices.push([x, y, z]);
    }

    Ok(vertices)
}

/// Read camera poses from binary format
fn load_camera_poses(path: &str) -> Result<Vec<[[f32; 4]; 4]>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read number of frames
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let num_frames = u32::from_le_bytes(buf) as usize;

    // Read 4x4 matrices (row-major)
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
        poses.push(matrix);
    }

    Ok(poses)
}

/// Extract position from 4x4 transform matrix
fn position_from_matrix(mat: &[[f32; 4]; 4]) -> [f32; 3] {
    [mat[0][3], mat[1][3], mat[2][3]]
}

/// Visualize camera frustum at given transform
fn visualize_camera_frustum(
    rec: &rr::RecordingStream,
    entity_path: String,
    matrix: &[[f32; 4]; 4],
    color: [u8; 3],
    size: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Camera position
    let pos = position_from_matrix(matrix);

    // Camera local corners (OpenCV camera: Z forward, Y down)
    let corners_local = [
        [-size, -size, 2.0 * size],  // Top-left (Y is down, so negative Y = top)
        [size, -size, 2.0 * size],   // Top-right
        [size, size, 2.0 * size],    // Bottom-right
        [-size, size, 2.0 * size],   // Bottom-left
    ];

    // Transform corners to world space
    let mut corners_world = Vec::new();
    for corner in &corners_local {
        let x = matrix[0][0] * corner[0] + matrix[0][1] * corner[1] + matrix[0][2] * corner[2] + matrix[0][3];
        let y = matrix[1][0] * corner[0] + matrix[1][1] * corner[1] + matrix[1][2] * corner[2] + matrix[1][3];
        let z = matrix[2][0] * corner[0] + matrix[2][1] * corner[1] + matrix[2][2] * corner[2] + matrix[2][3];
        corners_world.push([x, y, z]);
    }

    // Draw frustum lines
    let lines = vec![
        // From camera center to corners
        vec![pos, corners_world[0]],
        vec![pos, corners_world[1]],
        vec![pos, corners_world[2]],
        vec![pos, corners_world[3]],
        // Rectangle at image plane
        vec![corners_world[0], corners_world[1]],
        vec![corners_world[1], corners_world[2]],
        vec![corners_world[2], corners_world[3]],
        vec![corners_world[3], corners_world[0]],
    ];

    rec.log(
        entity_path,
        &rr::LineStrips3D::new(lines)
            .with_colors([color])
            .with_radii([0.005]),
    )?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Greeble Room Visualization");
    println!("=============================\n");

    // Initialize Rerun
    let rec = rr::RecordingStreamBuilder::new("greeble_room").spawn()?;

    // Configure view for OpenCV coordinates (Z-forward, Y-down)
    rec.log_static(
        "world",
        &rr::ViewCoordinates::RDF(),  // Right, Down, Forward (OpenCV standard)
    )?;

    // Load room point cloud
    println!("📦 Loading room point cloud...");
    let vertices = load_point_cloud("room_mesh.bin")?;
    println!("   {} vertices", vertices.len());

    // Debug: Print first few vertices
    println!("\n   First 5 vertices:");
    for (i, v) in vertices.iter().take(5).enumerate() {
        println!("     {}: [{:.3}, {:.3}, {:.3}]", i, v[0], v[1], v[2]);
    }

    // Debug: Print vertex bounds
    if !vertices.is_empty() {
        let mut min = vertices[0];
        let mut max = vertices[0];
        for v in &vertices {
            min[0] = min[0].min(v[0]);
            min[1] = min[1].min(v[1]);
            min[2] = min[2].min(v[2]);
            max[0] = max[0].max(v[0]);
            max[1] = max[1].max(v[1]);
            max[2] = max[2].max(v[2]);
        }
        println!("   Bounds: min=[{:.3}, {:.3}, {:.3}] max=[{:.3}, {:.3}, {:.3}]",
                 min[0], min[1], min[2], max[0], max[1], max[2]);
    }

    // Load camera poses
    println!("\n📷 Loading camera animation...");
    let poses = load_camera_poses("camera_poses.bin")?;
    println!("   {} camera poses", poses.len());

    // Debug: Print first camera position and rotation
    if !poses.is_empty() {
        let first_pos = position_from_matrix(&poses[0]);
        let last_pos = position_from_matrix(poses.last().unwrap());
        println!("   First camera pos: [{:.3}, {:.3}, {:.3}]", first_pos[0], first_pos[1], first_pos[2]);
        println!("   First camera rotation matrix:");
        println!("     [{:.4}, {:.4}, {:.4}]", poses[0][0][0], poses[0][0][1], poses[0][0][2]);
        println!("     [{:.4}, {:.4}, {:.4}]", poses[0][1][0], poses[0][1][1], poses[0][1][2]);
        println!("     [{:.4}, {:.4}, {:.4}]", poses[0][2][0], poses[0][2][1], poses[0][2][2]);
        println!("   Last camera pos:  [{:.3}, {:.3}, {:.3}]", last_pos[0], last_pos[1], last_pos[2]);
    }

    // Visualize room point cloud (static)
    println!("\n🎨 Visualizing...");
    println!("   Rendering room as point cloud");
    rec.log(
        "world/room/points",
        &rr::Points3D::new(vertices.clone())
            .with_colors([[180, 180, 200]])
            .with_radii([0.01]),
    )?;

    // Visualize camera trajectory as line
    let trajectory: Vec<[f32; 3]> = poses.iter().map(position_from_matrix).collect();
    rec.log(
        "world/camera/trajectory",
        &rr::LineStrips3D::new([trajectory.clone()])
            .with_colors([[255, 100, 100]])
            .with_radii([0.02]),
    )?;

    // Visualize camera poses on timeline
    for (frame_idx, matrix) in poses.iter().enumerate() {
        rec.set_time_sequence("animation", frame_idx as i64);

        // Draw camera frustum
        visualize_camera_frustum(
            &rec,
            format!("world/camera/frustum"),
            matrix,
            [255, 100, 100],
            0.15,
        )?;

        // Mark camera position
        let pos = position_from_matrix(matrix);
        rec.log(
            "world/camera/position",
            &rr::Points3D::new([pos])
                .with_colors([[255, 100, 100]])
                .with_radii([0.05]),
        )?;
    }

    println!("✅ Visualization complete!");
    println!("   Open Rerun and use the 'animation' timeline to scrub through frames");

    Ok(())
}
