"""Render dense and top-two retargeted CesiumMan side by side with Blender.

Run: blender -b --factory-startup --python tools/render_skin_comparison.py
Images are a visual inspection aid, not an automatic quality benchmark.
"""
import bpy
import json
from pathlib import Path
from mathutils import Vector

root = Path.cwd()
out = root / 'artifacts/posed-deformation/render-comparison'
out.mkdir(parents=True, exist_ok=True)
inputs = [root/'artifacts/posed-deformation/cesiumman-v13-animated.glb',
          root/'artifacts/posed-deformation/cesiumman-v13-top2-animated.glb']
report = json.loads((root/'artifacts/posed-deformation/cesiumman-v13-posed-deformation.json').read_text())
times = [p['time'] for p in report['poses']]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 16
scene.render.resolution_x = 1200
scene.render.resolution_y = 700
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.world.color = (0.18, 0.18, 0.18)
scene.render.fps = 24
for index, path in enumerate(inputs):
    existing = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = set(bpy.data.objects) - existing
    roots = [o for o in objects if o.parent not in objects]
    for obj in roots:
        obj.location.x += (-0.7 if index == 0 else 0.7)
# Fixed camera in Blender's Y-forward, Z-up space; identical lighting for both.
bpy.ops.object.camera_add(location=(0, -4.8, 1.3))
cam = bpy.context.object
cam.rotation_euler = (Vector((0, 0, 0.35)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 3.1
scene.camera = cam
bpy.ops.object.light_add(type='AREA', location=(0, -3, 4))
bpy.context.object.data.energy = 650
bpy.context.object.data.shape = 'DISK'
bpy.context.object.data.size = 5
for index, time in enumerate(times):
    frame = time * scene.render.fps
    scene.frame_set(int(frame), subframe=frame-int(frame))
    scene.render.filepath = str(out/f'pose-{index:02d}.png')
    bpy.ops.render.render(write_still=True)
(out/'manifest.json').write_text(json.dumps({'scope':'Blender Cycles visual inspection; dense left, top-two right; not ground truth or HDRI benchmark','times_seconds':times,'frames_per_second':scene.render.fps,'inputs':[str(p.relative_to(root)) for p in inputs]},indent=2)+'\n')
