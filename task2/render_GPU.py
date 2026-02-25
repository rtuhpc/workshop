import bpy
import os
import time

JOBID = os.environ.get("PBS_JOBID")
start_time = time.time()

# Enable Cycles render engine
bpy.context.scene.render.engine = 'CYCLES'

# Load cycles preferences
cycles_prefs = bpy.context.preferences.addons["cycles"].preferences

# Refresh device list
cycles_prefs.get_devices()

# Print available devices
for device in cycles_prefs.devices:
    print(f"Device: {device.name}, Enabled: {device.use}")

# Set Blender to use GPU rendering
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'

# Set to CPU rendering
# bpy.context.preferences.addons["cycles"].preferences.compute_device_type = 'NONE'
# bpy.context.scene.cycles.device = 'CPU'

my_dir = os.path.abspath("./output_" + JOBID + "/frame_")

bpy.context.scene.render.filepath = my_dir

bpy.ops.render.render(animation=True)  # Render animation

end_time = time.time()

print(f"Rendering time: {end_time - start_time} seconds")
