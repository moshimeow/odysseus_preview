import struct
with open('blender_stuff/greeble_room/camera_spline.bin', 'rb') as f:
    num_chan = struct.unpack('I', f.read(4))[0]
    rot_mode = struct.unpack('I', f.read(4))[0]
    fps = struct.unpack('d', f.read(8))[0]
    print(f"FPS: {fps}")
    for i in range(num_chan):
        num_keys = struct.unpack('I', f.read(4))[0]
        print(f"Chan {i}: {num_keys} keys")
        for j in range(num_keys):
            d = struct.unpack('dddddd', f.read(48))
            if j < 3: print(f"  Key {j}: t={d[0]:.4f}, val={d[1]:.4f}")
