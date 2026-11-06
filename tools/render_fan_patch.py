"""Render source, dense and candidate v14 retargeted CesiumMan side by side with Blender.

Run: blender -b --factory-startup --python tools/render_fan_patch.py
Images are a visual inspection aid, not an automatic quality benchmark.
"""
import bpy
import json
from pathlib import Path
from mathutils import Vector

root = Path.cwd()
out = root / 'artifacts/posed-deformation/render-fan-patch'
out.mkdir(parents=True, exist_ok=True)
inputs = [root/'artifacts/rigged-inputs/CesiumMan/CesiumMan.glb',
          root/'artifacts/kaggle-textured-rig-export-v14/posed-comparison/candidate-animated.glb',
          root/'artifacts/kaggle-textured-rig-export-v14/fan-patch-posed/candidate-animated.glb']
report = json.loads((root/'artifacts/posed-deformation/cesiumman-v13-posed-deformation.json').read_text())
times = [p['time'] for p in report['poses']]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 16
scene.render.resolution_x = 1500
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
    if index == 0:
        # Normalize the source in glTF Y-up coordinates, then convert to Blender Z-up.
        center = Vector((-6.53480239920512e-08, -0.02497701394774722, 0.7532751040569674))
        scale = 0.6637680165409462
        parent = bpy.data.objects.new('source normalization', None)
        scene.collection.objects.link(parent)
        parent.scale = (scale, scale, scale)
        parent.location = -scale * center
        for obj in roots:
            obj.parent = parent
        parent.location.x -= 1.2
    else:
        for obj in roots:
            obj.location.x += (0 if index == 1 else 1.2)
# Fixed camera in Blender's Y-forward, Z-up space; identical lighting for both.
bpy.ops.object.camera_add(location=(0, -4.8, 1.3))
cam = bpy.context.object
cam.rotation_euler = (Vector((0, 0, 0.35)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 4.2
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
(out/'manifest.json').write_text(json.dumps({'scope':'Blender Cycles visual inspection; normalized source left, v14 center, fan-patch right; not an HDRI benchmark','times_seconds':times,'frames_per_second':scene.render.fps,'inputs':[str(p.relative_to(root)) for p in inputs]},indent=2)+'\n')
