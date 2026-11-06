import bpy,numpy as np
from pathlib import Path
root=Path.cwd(); scene=bpy.context.scene; scene.render.fps=24
files=['artifacts/rigged-inputs/CesiumMan/CesiumMan.glb','artifacts/posed-deformation/cesiumman-v13-animated.glb','artifacts/posed-deformation/cesiumman-v13-top2-animated.glb']
for i,p in enumerate(files):
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
 bpy.ops.import_scene.gltf(filepath=str(root/p))
 for frame in (1,2,25,49):
  scene.frame_set(frame); deps=bpy.context.evaluated_depsgraph_get();arrays=[]
  for o in scene.objects:
   if o.type!='MESH':continue
   ev=o.evaluated_get(deps); mesh=ev.to_mesh()
   arrays.extend([tuple(ev.matrix_world @ v.co) for v in mesh.vertices]);ev.to_mesh_clear()
  np.save('/tmp/t3d-blender-%d-%d.npy'%(i,frame),np.array(arrays))
 print('case',i,'fps',scene.render.fps,'vertices',len(arrays))
