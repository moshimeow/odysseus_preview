#!/usr/bin/env blender --python
"""
Export camera spline from greeble_room.blend to binary format.
Preserves original spline representation (Euler or Quat).
"""

import bpy
import struct
import numpy as np
from mathutils import Matrix

def main():
    camera = bpy.data.objects.get('camera')
    if not camera:
        print("Camera not found")
        return

    if not camera.animation_data or not camera.animation_data.action:
        print("No animation data on camera")
        return

    action = camera.animation_data.action
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base

    # Get fcurves (handles both legacy and layered actions in Blender 4.3+)
    if hasattr(action, "fcurves") and action.fcurves:
        fcurves_list = action.fcurves
    else:
        try:
            slot = action.slots[0]
            strip = action.layers[0].strips[0]
            fcurves_list = strip.channelbag(slot).fcurves
        except (AttributeError, IndexError):
            print("Could not find F-Curves in action")
            return

    curves_dict = {}
    for fcurve in fcurves_list:
        curves_dict[(fcurve.data_path, fcurve.array_index)] = fcurve

    loc_curves = [curves_dict.get(('location', i)) for i in range(3)]
    
    # Check rotation mode
    rot_mode = 0 # 0: Euler XYZ, 1: Quat, 2: Euler XZY, etc.
    rot_name = "rotation_euler"
    rot_channels = 3
    
    if camera.rotation_mode == 'QUATERNION':
        rot_name = "rotation_quaternion"
        rot_channels = 4
        rot_mode = 1
        print("Detected Quaternion rotation")
    else:
        # Map Euler modes to an integer for the binary format
        # For simplicity, we'll just handle XYZ and print a warning if different
        euler_modes = {'XYZ': 0, 'XZY': 2, 'YXZ': 3, 'YZX': 4, 'ZXY': 5, 'ZYX': 6}
        rot_mode = euler_modes.get(camera.rotation_mode, 0)
        print(f"Detected Euler rotation: {camera.rotation_mode}")

    rot_curves = [curves_dict.get((rot_name, i)) for i in range(rot_channels)]

    if any(c is None for c in loc_curves) or any(c is None for c in rot_curves):
        print("Missing some animation curves")
        return

    # Collect all keyframe times from all curves to find the full range
    times = set()
    for c in loc_curves + rot_curves:
        for k in c.keyframe_points:
            times.add(k.co.x)
    
    sorted_times = sorted(list(times))
    
    output_path = "/home/quantum1000/Documents/Programs/odysseus_preview/odysseus-slam/blender_stuff/greeble_room/camera_spline.bin"
    
    # Channels: 3 for pos, N for rot
    num_channels = 3 + rot_channels
    channels = [[] for _ in range(num_channels)]
    
    for t_frame in sorted_times:
        t_sec = t_frame / fps
        
        def get_key_data(curve, t):
            # Find keyframe exactly at this time
            kp = None
            for k in curve.keyframe_points:
                if abs(k.co.x - t) < 1e-4:
                    kp = k
                    break
            
            if kp:
                val = kp.co.y
                hl_x, hl_y = kp.handle_left[0], kp.handle_left[1]
                hr_x, hr_y = kp.handle_right[0], kp.handle_right[1]
            else:
                # If no keyframe, evaluate and use "auto" handles (slope=0 for simplicity at boundaries)
                val = curve.evaluate(t)
                hl_x, hl_y = t, val
                hr_x, hr_y = t, val
                
            return (t_sec, val, hl_x/fps, hl_y, hr_x/fps, hr_y)

        # Position transform: X_cv = X_b, Y_cv = -Z_b, Z_cv = Y_b
        # We transform the values and handles directly
        
        # X
        channels[0].append(get_key_data(loc_curves[0], t_frame))
        
        # Y (OpenCV) = -Z (Blender)
        t_sec, val, hlx, hly, hrx, hry = get_key_data(loc_curves[2], t_frame)
        channels[1].append((t_sec, -val, hlx, -hly, hrx, -hry))
        
        # Z (OpenCV) = Y (Blender)
        channels[2].append(get_key_data(loc_curves[1], t_frame))
        
        # Rotation channels (preserved as-is)
        for i in range(rot_channels):
            channels[3+i].append(get_key_data(rot_curves[i], t_frame))

    with open(output_path, 'wb') as f:
        # Binary format:
        # u32: total_channels
        # u32: rot_mode (0=EulerXYZ, 1=Quat, ...)
        # f64: fps
        # For each channel:
        #   u32: num_keyframes
        #   For each keyframe: [time, value, h_left_x, h_left_y, h_right_x, h_right_y] (f64 * 6)
        
        f.write(struct.pack('I', num_channels))
        f.write(struct.pack('I', rot_mode))
        f.write(struct.pack('d', fps))
        
        for chan in channels:
            f.write(struct.pack('I', len(chan)))
            for k in chan:
                f.write(struct.pack('dddddd', *k))

    print(f"Exported {len(sorted_times)} keyframes across {num_channels} channels to {output_path}")

if __name__ == "__main__":
    main()
