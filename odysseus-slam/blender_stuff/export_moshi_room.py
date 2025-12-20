#!/usr/bin/env blender --python
# I AM SORRY this is just copied from export_greeble_room w/o the mesh part.
"""
Export greeble_room.blend geometry and camera animation to binary format.

Usage:
    blender greeble_room.blend --background --python export_greeble_room.py
"""

import bpy
import struct
import numpy as np
from mathutils import Matrix

def blender_to_opencv(co):
    """Convert Blender coords (Y forward, Z up) to OpenCV (Z forward, Y down)."""
    # Blender: X right, Y forward, Z up
    # OpenCV:  X right, Y down,    Z forward
    # Transformation: X' = X, Y' = -Z, Z' = Y
    return (co.x, -co.z, co.y)

def export_camera_animation(camera_name, output_path):
    """Export camera animation as SE3 transforms in OpenCV coordinate system."""
    camera = bpy.data.objects.get(camera_name)
    if not camera:
        raise ValueError(f"Camera '{camera_name}' not found")

    scene = bpy.context.scene
    start_frame = scene.frame_start
    end_frame = scene.frame_end

    # Conversion matrix from Blender world coords to OpenCV world coords
    # Blender: Y forward, Z up
    # OpenCV:  Z forward, Y down
    # This rotates: X→X, Y→Z, Z→-Y
    blender_to_opencv_mat = Matrix([
        [1,  0,  0, 0],
        [0,  0, -1, 0],
        [0,  1,  0, 0],
        [0,  0,  0, 1]
    ])

    # Local 180° rotation around camera's Y axis
    # Blender camera: -Z forward (OpenGL convention)
    # OpenCV camera: +Z forward (CV convention)
    # Rotation by 180° around Y: X→-X, Y→Y, Z→-Z
    camera_flip = Matrix([
        [ 1,  0,  0, 0],
        [ 0, -1,  0, 0],
        [ 0,  0, -1, 0],
        [ 0,  0,  0, 1]
    ])

    transforms = []

    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)

        # Get camera world matrix (4x4 transform) in Blender coords
        mat_blender = camera.matrix_world.copy()

        # Apply local camera flip (in camera's local space)
        mat_blender_flipped = mat_blender @ camera_flip

        # Convert to OpenCV coordinate system
        mat_opencv = blender_to_opencv_mat @ mat_blender_flipped

        transforms.append(mat_opencv)

    print(f"Exporting {len(transforms)} camera poses from frames {start_frame} to {end_frame}")

    # Write binary format:
    # u32: num_frames
    # transforms: [f32; 16] * num_frames (row-major 4x4 matrices)
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', len(transforms)))

        for mat in transforms:
            # Write row-major 4x4 matrix
            for row in range(4):
                for col in range(4):
                    f.write(struct.pack('f', mat[row][col]))

    print(f"Saved camera animation to {output_path}")

def main():
    # Export camera animation
    export_camera_animation('mapping_ish_easy_L', 'REDACTED/camera_poses.bin')

    print("Export complete!")

if __name__ == '__main__':
    main()
