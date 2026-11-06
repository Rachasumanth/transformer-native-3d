"""Render multi-view GLB color, depth and normal passes with Blender.

Usage: blender -b --factory-startup --python tools/render_curation_views.py --
       --input asset.glb --output output-directory [--resolution 128]
This produces images for later filtering; it does not classify or curate assets.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--azimuths', nargs='+', type=float, default=[0, 90, 180, 270])
    parser.add_argument('--fovs', nargs='+', type=float, default=[35, 55])
    parser.add_argument('--elevation', type=float, default=20)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    if not (16 <= args.resolution <= 2048) or not all(5 <= x <= 120 for x in args.fovs):
        parser.error('resolution or field of view outside supported range')
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').unlink(missing_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(source))
    bpy.context.view_layer.update()
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    if not points:
        raise ValueError('Asset has no mesh bounds')
    lower = Vector([min(p[i] for p in points) for i in range(3)])
    upper = Vector([max(p[i] for p in points) for i in range(3)])
    center = (lower + upper) * .5
    radius = max((p-center).length for p in points)
    if not math.isfinite(radius) or radius < 1e-8:
        raise ValueError('Asset bounds are degenerate')
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 8
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (.8, .8, .8, 1)
    background.inputs['Strength'].default_value = .8
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_normal = True
    scene.use_nodes = True
    if hasattr(scene, 'compositing_node_group'):
        tree = bpy.data.node_groups.new('curation_compositor', 'CompositorNodeTree')
        scene.compositing_node_group = tree
        tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        nodes = tree.nodes
        composite = nodes.new('NodeGroupOutput')
    else:
        tree = scene.node_tree
        nodes = tree.nodes
        nodes.clear()
        composite = nodes.new('CompositorNodeComposite')
    layer = nodes.new('CompositorNodeRLayers')
    tree.links.new(layer.outputs['Image'], composite.inputs['Image'])
    file_nodes = {}
    for label, socket in [('depth', 'Depth'), ('normal', 'Normal')]:
        node = nodes.new('CompositorNodeOutputFile')
        if hasattr(node, 'directory'):
            node.directory = str(output)
        else:
            node.base_path = str(output)
        if hasattr(node, 'file_output_items'):
            node.format.file_format = 'OPEN_EXR_MULTILAYER'
            node.file_output_items.new('RGBA', label)
        else:
            node.format.file_format = 'OPEN_EXR'
        node.format.color_depth = '32'
        tree.links.new(layer.outputs[socket], node.inputs[0])
        file_nodes[label] = node
    light_data = bpy.data.lights.new('curation_sun', 'SUN')
    light_data.energy = 2.0
    light = bpy.data.objects.new('curation_sun', light_data)
    scene.collection.objects.link(light)
    light.rotation_euler = (math.radians(25), math.radians(-30), math.radians(-25))
    camera_data = bpy.data.cameras.new('curation_camera')
    camera = bpy.data.objects.new('curation_camera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    results = []
    for fov in args.fovs:
        camera_data.angle = math.radians(fov)
        camera_data.clip_start = max(radius / 10000, 1e-6)
        camera_data.clip_end = radius * 100
        distance = radius / math.sin(math.radians(fov) / 2) * 1.1
        for azimuth in args.azimuths:
            az, el = math.radians(azimuth), math.radians(args.elevation)
            direction = Vector((math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)))
            camera.location = center + direction * distance
            camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
            name = f'view-{len(results):03d}'
            scene.render.filepath = str(output / (name + '-color.png'))
            for label, node in file_nodes.items():
                if hasattr(node, 'file_name'):
                    node.file_name = name + '-' + label + '-'
                else:
                    node.file_slots[0].path = name + '-' + label + '-'
            bpy.ops.render.render(write_still=True)
            suffix = '-.exr' if hasattr(file_nodes['depth'], 'file_name') else '-0001.exr'
            paths = [output / (name + '-color.png')] + [output / (name + '-' + label + suffix) for label in ('depth', 'normal')]
            if not all(path.is_file() for path in paths):
                raise RuntimeError('Expected render output missing')
            results.append(dict(azimuth=azimuth, elevation=args.elevation, fov_degrees=fov,
                                camera_position=list(camera.location), files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    manifest = dict(scope='Multi-view color/depth/normal render; no quality filtering or canonical pose prediction',
                    input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), blender_version=bpy.app.version_string,
                    resolution=args.resolution, samples=scene.cycles.samples, bounds=[list(lower), list(upper)], views=results)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('CURATION_RENDER_COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    main()
