"""CuMesh standard cleanup and UV preparation for the Step 6 export path."""
from dataclasses import dataclass
import math
import torch


@dataclass
class PreparedMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    uv: torch.Tensor
    normals: torch.Tensor
    vertex_map: torch.Tensor
    reference_bvh: object
    reference_vertices: torch.Tensor
    reference_faces: torch.Tensor
    audit: list


def build_reference_bvh(vertices,faces):
    """Return BVH and its exact triangle array, including small-input padding.

    CuMesh requires >8 triangles. Repeating existing triangles changes no closest
    surface, and only this reference index array is padded, not output geometry.
    """
    if not len(faces):raise ValueError('cannot build a BVH for empty geometry')
    import cumesh
    reference_faces=faces
    if len(faces)<9:
        reference_faces=faces.repeat(((9+len(faces)-1)//len(faces)),1).contiguous()
    return cumesh.cuBVH(vertices,reference_faces),reference_faces


def clean_and_unwrap(vertices, faces, *, target_faces=1000000, remesh=False,
                     aabb=None, grid_size=None):
    """Run blueprint cleanup and cone-clustered UVs on CUDA.

    The BVH retains the hole-filled input for later texture reprojection. The
    vertex map indexes cleaned geometry, not original sparse attribute rows.
    CuMesh's atlas packing uses CPU Xatlas; outputs are returned to input device.
    Strict remeshing requires aabb/grid_size; it uses a one-cell narrow band
    and 0.9 BVH projection before simplification. Rig transfer remains separate.
    """
    if not vertices.is_cuda or not faces.is_cuda or vertices.device != faces.device:
        raise ValueError('vertices and faces must share a CUDA device')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not vertices.is_floating_point():
        raise ValueError('vertices must be floating (V,3)')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype not in (torch.int32,torch.int64):
        raise ValueError('faces must be integer (F,3)')
    if not len(vertices) or not len(faces):
        raise ValueError('cannot unwrap empty geometry')
    if not torch.isfinite(vertices).all() or (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError('invalid mesh positions or indices')
    if type(target_faces) is not int or not 1 <= target_faces <= 1000000:
        raise ValueError('target_faces must be in [1,1000000]')
    if len(vertices) >= 2**31:
        raise ValueError('CuMesh requires int32 vertex indices')
    if type(remesh) is not bool:
        raise ValueError('remesh must be boolean')
    if not torch.isfinite(vertices.float()).all():
        raise ValueError('vertices overflow float32')
    if remesh:
        bounds=torch.as_tensor(aabb,device=vertices.device,dtype=torch.float32) if aabb is not None else None
        if bounds is None or bounds.shape!=(2,3) or not torch.isfinite(bounds).all() or not (bounds[1]>bounds[0]).all():
            raise ValueError('remeshing requires finite increasing aabb')
        if grid_size is None or len(grid_size)!=3 or any(type(x) is not int or x<=0 for x in grid_size):
            raise ValueError('remeshing requires positive integer grid_size')
    import cumesh
    device=vertices.device
    audit=[]
    with torch.cuda.device(device), torch.no_grad():
        mesh=cumesh.CuMesh()
        mesh.init(vertices.float().contiguous(),faces.int().contiguous())
        def record(stage):
            audit.append(dict(stage=stage,vertices=mesh.num_vertices,faces=mesh.num_faces))
        record('input')
        mesh.fill_holes(max_hole_perimeter=3e-2)
        reference_vertices,reference_faces=mesh.read()
        bvh,bvh_faces=build_reference_bvh(reference_vertices,reference_faces)
        record('initial_holes')
        if remesh:
            resolution=max(grid_size)
            mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
                reference_vertices,reference_faces,center=bounds.mean(0),
                scale=float((bounds[1]-bounds[0]).max())*(resolution+3)/resolution,
                resolution=resolution,band=1,project_back=.9,verbose=False,bvh=bvh))
            record('narrow_band_remesh')
            mesh.simplify(target_faces,verbose=False)
            record('remesh_simplify')
            mesh.remove_duplicate_faces()
            mesh.repair_non_manifold_edges()
            mesh.remove_small_connected_components(1e-5)
            mesh.fill_holes(max_hole_perimeter=3e-2)
            mesh.unify_face_orientations()
            record('remesh_cleanup')
            if not mesh.num_faces:raise ValueError('remeshing removed all faces')
        else:
            for stage,target in (('coarse',target_faces*3),('final',target_faces)):
                mesh.simplify(target,verbose=False)
                mesh.remove_duplicate_faces()
                mesh.repair_non_manifold_edges()
                mesh.remove_small_connected_components(1e-5)
                mesh.fill_holes(max_hole_perimeter=3e-2)
                record(stage)
                if not mesh.num_faces:
                    raise ValueError('cleanup removed all faces')
            mesh.unify_face_orientations()
        v,f,uv,vmap=mesh.uv_unwrap(compute_charts_kwargs={
            'threshold_cone_half_angle_rad':math.pi/2,
            'global_iterations':1,'smooth_strength':1},return_vmaps=True,verbose=False)
        mesh.compute_vertex_normals()
        vmap=vmap.to(device=device,dtype=torch.int64)
        normals=mesh.read_vertex_normals()[vmap]
        return PreparedMesh(v.to(device),f.to(device),uv.to(device),normals,vmap,bvh,
                            reference_vertices,bvh_faces,audit)
