"""GPU integration probe: voxel cube -> CuMesh UVs -> baked PBR -> saved GLB."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

PINS={
    'CuMesh':('https://github.com/JeffreyXiang/CuMesh.git','12289e1062f0603f2f0d0771b02e1395d247f26f'),
    'FlexGEMM':('https://github.com/JeffreyXiang/FlexGEMM.git','6dd94a859c26ee8246888502eada3dd8ad85532e'),
    'nvdiffrast':('https://github.com/NVlabs/nvdiffrast.git','253ac4fcea7de5f396371124af597e6cc957bfae'),
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--skinae-checkpoint',type=Path)
    parser.add_argument('--real-rig-source',type=Path)
    parser.add_argument('--real-grid-size',type=int,default=32)
    parser.add_argument('--real-texture-size',type=int,default=128)
    parser.add_argument('--max-skin-influences',type=int,default=16)
    parser.add_argument('--dependency-cache',type=Path,default=Path('/kaggle/input/transformer-native-3d-cuda-wheels'))
    args=parser.parse_args()
    if args.real_rig_source is not None and args.skinae_checkpoint is None:
        parser.error('--real-rig-source requires --skinae-checkpoint')
    if args.real_grid_size<16 or args.real_grid_size>256:
        parser.error('--real-grid-size must be in [16,256]')
    if args.real_texture_size<16 or args.real_texture_size>4096:
        parser.error('--real-texture-size must be in [16,4096]')
    if not 1<=args.max_skin_influences<=16:
        parser.error('--max-skin-influences must be in [1,16]')
    env=dict(os.environ,MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5')
    os.environ.update(MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5')
    from tools.build_cuda_dependency import build_and_install
    with open('/kaggle/working/export-build.log','w') as log:
        def run(cmd):subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        run([sys.executable,'-m','pip','install','ninja','opencv-python-headless'])
        for name,(url,commit) in PINS.items():
            build_and_install(name,url,commit,log,cache_dir=args.dependency_cache)
    import numpy as np
    import torch
    import trimesh
    from flex_gemm.ops.grid_sample import grid_sample_3d
    from t3d.voxelize import mesh_to_voxels
    from t3d.extraction_torch import voxels_to_mesh
    from t3d.mesh_postprocess import PreparedMesh,clean_and_unwrap,build_reference_bvh
    from t3d.texture_baking import bake_pbr
    from t3d.glb import pack_glb
    from t3d.topology import orient_faces
    from t3d.render_loss import shape_render_loss
    target=torch.tensor([[-.6,-.6,0.],[.6,-.6,0.],[.6,.6,0.],[-.6,.6,0.]],device='cuda')
    render_faces=torch.tensor([[0,1,2],[0,2,3]],device='cuda',dtype=torch.int32)
    camera=torch.eye(4,device='cuda')[None]
    same=shape_render_loss(target,target,render_faces,camera,resolution=32)
    assert sum(same[k].item() for k in ('silhouette','depth','normal'))<1e-5
    shifted=target.clone();shifted[:,0]+=.15;shifted[2,2]=.2;shifted.requires_grad_()
    changed=shape_render_loss(shifted,target,render_faces,camera,resolution=32)
    render_loss=sum(changed[k] for k in ('silhouette','depth','normal'))
    assert render_loss.item()>0
    render_loss.backward()
    assert shifted.grad is not None and torch.isfinite(shifted.grad).all() and shifted.grad.abs().sum()>0
    # Independent dense affine sampler oracle establishes coordinate convention.
    xyz=torch.cartesian_prod(*(torch.arange(4,device='cuda') for _ in range(3)))
    bcoords=torch.cat([torch.zeros_like(xyz[:,:1]),xyz],1).int()
    query=torch.tensor([[[.75,1.,1.25],[1.2,1.3,1.7],[2.4,2.2,1.1]]],device='cuda')
    sampled=grid_sample_3d(xyz.float()/3,bcoords,shape=torch.Size([1,3,4,4,4]),grid=query,mode='trilinear')
    torch.testing.assert_close(sampled,(query-.5)/3,atol=1e-5,rtol=1e-5)
    # A spatially varying material catches UV row orientation, BVH projection,
    # and global triangle IDs when a raster chunk starts after triangle zero.
    import cumesh
    plane=torch.tensor([[.125,.125,.5],[.875,.125,.5],[.875,.875,.5],[.125,.875,.5]],device='cuda')
    pf=torch.tensor([[0,1,2],[0,2,3]],device='cuda',dtype=torch.int32)
    puv=torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]],device='cuda')
    pn=torch.tensor([[0.,0.,1.]]*4,device='cuda')
    plane_bvh,reference_pf=build_reference_bvh(plane,pf)
    prepared_plane=PreparedMesh(plane,pf,puv,pn,torch.arange(4,device='cuda'),
                               plane_bvh,plane,reference_pf,[])
    colors=torch.cat([(xyz.float()+.5)/4,torch.tensor([[.2,.7,1.]],device='cuda').expand(len(xyz),3)],1)
    gradient,gm=bake_pbr(prepared_plane,xyz,colors,(4,)*3,[[0]*3,[1]*3],texture_size=32)
    chunked,cm=bake_pbr(prepared_plane,xyz,colors,(4,)*3,[[0]*3,[1]*3],texture_size=32,raster_chunk_size=1)
    np.testing.assert_array_equal(gm,cm)
    np.testing.assert_allclose(gradient,chunked,atol=1e-6)
    assert gm.all()
    py,px=np.mgrid[:32,:32]
    expected_gradient=np.stack([.125+.75*(px+.5)/32,.125+.75*(py+.5)/32,np.full((32,32),.5)],-1)
    gradient_error=float(np.abs(gradient[...,:3]-expected_gradient).max())
    assert gradient_error<1e-5,gradient_error
    v=np.array([[x,y,z] for x in (.2,.8) for y in (.2,.8) for z in (.2,.8)])
    faces=[]
    for axis in range(3):
        for side in (.2,.8):
            a,b,c,d=np.flatnonzero(v[:,axis]==side)
            faces.extend([[a,b,d],[a,d,c]])
    asset=mesh_to_voxels(v,faces,grid_size=(8,)*3,aabb=[[0]*3,[1]*3])
    tensor=lambda x:torch.tensor(np.array(x,copy=True),device='cuda')
    coords=tensor(asset.coords.astype(np.int64))
    vertices,triangles=voxels_to_mesh(coords,tensor(asset.dual_vertices).float(),
        tensor(asset.intersected_flag),(8,)*3,[[0]*3,[1]*3])
    oriented,_=orient_faces(vertices.cpu().numpy(),triangles.cpu().numpy(),outward=True)
    prepared=clean_and_unwrap(vertices,tensor(oriented),target_faces=200)
    expected=np.array([.25,.1,.8,.3,.65,.6],dtype=np.float32)
    pbr=torch.tensor(expected,device='cuda').expand(len(coords),6).contiguous()
    image,mask=bake_pbr(prepared,coords,pbr,(8,)*3,[[0]*3,[1]*3],texture_size=128)
    assert mask.any() and np.isfinite(image).all()
    max_error=float(np.abs(image[mask]-expected).max())
    assert max_error<1e-5, max_error
    padded_error=float(np.abs(image-expected).max())
    assert padded_error<=1/255, padded_error
    data=pack_glb(prepared.vertices.cpu().numpy(),prepared.faces.cpu().numpy(),
        prepared.uv.cpu().numpy(),prepared.normals.cpu().numpy(),image,alpha_mode='BLEND')
    path=Path('/kaggle/working/export-cube.glb');path.write_bytes(data)
    loaded=trimesh.load(io.BytesIO(data),file_type='glb',force='mesh',process=False)
    assert len(loaded.faces)==len(prepared.faces)
    np.testing.assert_allclose(loaded.bounds,[[.2,.2,-.8],[.8,.8,-.2]],atol=.03)
    assert loaded.visual.material.alphaMode=='BLEND'
    # Exercise the file/API boundary too: this uses raw provisional winding,
    # unlike the manually oriented fixture above, and must retain valid output.
    from dataclasses import replace
    from t3d.container import save
    material_asset=replace(asset,pbr=np.tile(expected,(len(asset.coords),1)))
    source=Path('/kaggle/working/export-input.t3d')
    output=Path('/kaggle/working/export-cli.glb')
    save(source,material_asset)
    subprocess.run([sys.executable,'-m','t3d.export',str(source),str(output),
        '--target-faces','200','--texture-size','128','--source-up','z'],check=True)
    audit=json.loads(output.with_suffix('.glb.json').read_text())
    assert audit['input_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert audit['glb_sha256']==hashlib.sha256(output.read_bytes()).hexdigest()
    assert audit['alpha_mode']=='BLEND' and audit['bpt_token_count']==0
    command_mesh=trimesh.load(output,force='mesh',process=False)
    assert len(command_mesh.faces)==audit['uv_faces'] and len(command_mesh.faces)>0
    positions,inverse=np.unique(command_mesh.vertices,axis=0,return_inverse=True)
    _,command_topology=orient_faces(positions,inverse[command_mesh.faces],outward=True)
    assert command_topology['components'] and all(c['closed'] for c in command_topology['components'])
    assert command_topology['flipped_faces']==0,command_topology
    rigged_report=None;real_rig_report=None
    if args.skinae_checkpoint is not None:
        from dataclasses import replace
        from t3d.articulation import Skeleton,attach_articulation,skeleton_displacements
        from t3d.export import export_raw
        from t3d.import_rig import load_rig_glb
        from t3d.skin_ae import SkinAE
        checkpoint=torch.load(args.skinae_checkpoint,map_location='cuda:0',weights_only=True)
        model=SkinAE(**checkpoint['config']).cuda();model.load_state_dict(checkpoint['model'],strict=True);model.freeze()
        rig=Skeleton([[.28,.28,.5],[.72,.28,.5],[.5,.72,.5]],[-1,0,0])
        centers=material_asset.aabb[0]+(material_asset.coords.astype(float)+.5)*material_asset.voxel_size
        _,joint_ids=skeleton_displacements(centers,rig)
        voxel_weights=np.eye(len(rig.joints),dtype=np.float32)[joint_ids]
        with torch.inference_mode():
            joint_embeddings=model.encode_joints(
                torch.as_tensor(rig.joints,device='cuda',dtype=torch.float32),
                torch.as_tensor(rig.parents,device='cuda',dtype=torch.int64))
        rigged=attach_articulation(material_asset,rig,voxel_weights=voxel_weights,
            joint_embeddings=joint_embeddings.cpu().numpy())
        rigged_data,rigged_audit=export_raw(rigged,device='cuda:0',target_faces=200,
            texture_size=128,source_up='z',skinae_checkpoint=args.skinae_checkpoint,
            joint_tolerance=1e-6,max_skin_influences=args.max_skin_influences)
        rigged_output=Path('/kaggle/working/export-cli-rigged.glb')
        rigged_output.write_bytes(rigged_data)
        restored=load_rig_glb(rigged_output)
        assert rigged_audit['skinned'] and rigged_audit['joint_count']==3
        assert restored.weights.shape==(len(restored.vertices),3)
        np.testing.assert_allclose(restored.weights.sum(axis=1),1.,atol=1e-6)
        assert np.isfinite(restored.weights).all() and np.all(restored.weights>=0)
        rigged_report=dict(audit=rigged_audit,vertices=len(restored.vertices),
            faces=len(restored.faces),joint_count=len(restored.skeleton.joints),
            maximum_weight_sum_error=float(np.max(np.abs(restored.weights.sum(1)-1))),
            glb_bytes=len(rigged_data),
            glb_sha256=hashlib.sha256(rigged_data).hexdigest(),
            scope='Synthetic skinned cube through extraction, CuMesh cleanup/UV, PBR bake, SkinAE decode and GLB reload; no source-rig quality claim')
        if args.real_rig_source is not None:
            from t3d.articulation import transfer_skin_weights
            from t3d.import_rig import load_rig_glb
            from t3d.metrics import surface_comparison
            from t3d.voxelize import mesh_to_voxels
            source_rig,normalization=load_rig_glb(args.real_rig_source,include_pbr=True).normalized()
            source_vertices=np.clip(source_rig.vertices,-.5,.5)
            geometry=mesh_to_voxels(source_vertices,source_rig.faces,grid_size=(args.real_grid_size,)*3,
                aabb=[[-.5]*3,[.5]*3])
            centers=geometry.world_vertices
            transferred=transfer_skin_weights(centers,source_vertices,source_rig.faces,source_rig.weights)
            with torch.inference_mode():
                source_joint_embeddings=model.encode_joints(
                    torch.as_tensor(source_rig.skeleton.joints,device='cuda',dtype=torch.float32),
                    torch.as_tensor(source_rig.skeleton.parents,device='cuda',dtype=torch.int64))
            real_asset=attach_articulation(geometry,source_rig.skeleton,
                voxel_weights=transferred,joint_embeddings=source_joint_embeddings.cpu().numpy())
            from t3d.materials import sample_pbr
            real_asset=sample_pbr(real_asset,source_vertices,source_rig.faces,
                source_rig.pbr_materials,source_rig.face_materials,
                corner_uv=source_rig.corner_uv,projection_method='bvh')
            source_pbr_report=dict(material_count=len(source_rig.pbr_materials),
                face_material_ids=np.unique(source_rig.face_materials).tolist(),
                voxel_pbr_min=real_asset.pbr.min(axis=0).tolist(),
                voxel_pbr_max=real_asset.pbr.max(axis=0).tolist(),
                materials=[dict(base_color=list(m.base_color),metallic=m.metallic,
                    roughness=m.roughness,alpha_mode=m.alpha_mode,
                    has_base_texture=m.base_texture is not None,
                    has_metallic_roughness_texture=m.metallic_roughness_texture is not None)
                    for m in source_rig.pbr_materials])
            real_data,real_audit=export_raw(real_asset,device='cuda:0',target_faces=max(1000,2*len(source_rig.faces)),
                texture_size=args.real_texture_size,source_up='y',skinae_checkpoint=args.skinae_checkpoint,
                joint_tolerance=1e-6,max_skin_influences=args.max_skin_influences)
            real_output=Path('/kaggle/working/export-real-rigged.glb');real_output.write_bytes(real_data)
            real_restored=load_rig_glb(real_output)
            from tools.audit_rig_pbr_textures import audit as audit_pbr_textures
            from tools.audit_material_fidelity import audit as audit_material_fidelity
            real_pbr_texture_audit=audit_pbr_textures(args.real_rig_source,real_output)
            real_material_fidelity=audit_material_fidelity(args.real_rig_source,real_output,samples=4096,seed=0)
            if real_audit['joint_count']!=len(source_rig.skeleton.joints):
                raise AssertionError('Real-rig joint count changed during export')
            np.testing.assert_allclose(real_restored.weights.sum(1),1.,atol=1e-6)
            if not np.isfinite(real_restored.weights).all():raise AssertionError('Real-rig output weights are nonfinite')
            distances=np.linalg.norm(real_restored.skeleton.joints[:,None]-source_rig.skeleton.joints[None],axis=-1)
            predicted_to_source=np.argmin(distances,axis=1)
            if len(set(predicted_to_source.tolist()))!=len(predicted_to_source):
                raise AssertionError('Recovered joints cannot be matched uniquely to source rig')
            aligned=np.zeros_like(real_restored.weights)
            for predicted_id,source_id in enumerate(predicted_to_source):
                aligned[:,source_id]=real_restored.weights[:,predicted_id]
            targets=transfer_skin_weights(real_restored.vertices,source_vertices,source_rig.faces,source_rig.weights)
            eps=1e-8
            weight_kl=float(np.mean(np.sum(np.where(targets>0,targets*np.log((targets+eps)/(aligned+eps)),0),axis=1)))
            geometry_quality=surface_comparison(source_vertices,source_rig.faces,
                real_restored.vertices,real_restored.faces,samples=512,seed=0,tolerance=.01)
            source_topology=surface_comparison(source_vertices,source_rig.faces,
                source_vertices,source_rig.faces,samples=512,seed=0,tolerance=.01)
            strict_remesh_report=None
            try:
                strict_data,strict_audit=export_raw(real_asset,device='cuda:0',
                    target_faces=max(1000,2*len(source_rig.faces)),texture_size=args.real_texture_size,
                    source_up='y',skinae_checkpoint=args.skinae_checkpoint,
                    joint_tolerance=1e-6,remesh=True,max_skin_influences=args.max_skin_influences)
                strict_output=Path('/kaggle/working/export-real-rigged-strict.glb')
                strict_output.write_bytes(strict_data)
                strict_restored=load_rig_glb(strict_output)
                strict_pbr_texture_audit=audit_pbr_textures(args.real_rig_source,strict_output)
                strict_material_fidelity=audit_material_fidelity(args.real_rig_source,strict_output,
                    samples=4096,seed=0)
                strict_quality=surface_comparison(source_vertices,source_rig.faces,
                    strict_restored.vertices,strict_restored.faces,samples=512,seed=0,tolerance=.01)
                strict_remesh_report=dict(audit=strict_audit,vertices=len(strict_restored.vertices),
                    faces=len(strict_restored.faces),joint_count=len(strict_restored.skeleton.joints),
                    surface_comparison=strict_quality,
                    pbr_texture_audit=strict_pbr_texture_audit,
                    material_fidelity_audit=strict_material_fidelity,
                    maximum_weight_sum_error=float(np.max(np.abs(strict_restored.weights.sum(1)-1))),
                    glb_bytes=len(strict_data),glb_sha256=hashlib.sha256(strict_data).hexdigest())
            except Exception as exc:
                strict_remesh_report=dict(error_type=type(exc).__name__,error=str(exc))
            # Exercise the actual checksummed .t3d read/CLI boundary on this
            # source rig, not only the in-memory export_raw API.
            from t3d.container import save as save_raw_container
            real_container=Path('/kaggle/working/export-real-rigged-input.t3d')
            real_cli_output=Path('/kaggle/working/export-real-rigged-container.glb')
            save_raw_container(real_container,real_asset)
            subprocess.run([sys.executable,'-m','t3d.export',str(real_container),
                str(real_cli_output),'--target-faces',str(max(1000,2*len(source_rig.faces))),
                '--texture-size',str(args.real_texture_size),'--source-up','y','--skinae-checkpoint',
                str(args.skinae_checkpoint)],check=True)
            real_cli_audit=json.loads(real_cli_output.with_suffix('.glb.json').read_text())
            if real_cli_audit['input_sha256']!=hashlib.sha256(real_container.read_bytes()).hexdigest():
                raise AssertionError('Real-rig CLI input checksum differs from saved .t3d')
            real_cli_restored=load_rig_glb(real_cli_output)
            real_cli_pbr_texture_audit=audit_pbr_textures(args.real_rig_source,real_cli_output)
            real_cli_material_fidelity=audit_material_fidelity(args.real_rig_source,real_cli_output,
                samples=4096,seed=0)
            if len(real_cli_restored.skeleton.joints)!=len(source_rig.skeleton.joints):
                raise AssertionError('Real-rig CLI joint count changed during export')
            np.testing.assert_allclose(real_cli_restored.weights.sum(1),1.,atol=1e-6)
            real_cli_surface=surface_comparison(source_vertices,source_rig.faces,
                real_cli_restored.vertices,real_cli_restored.faces,samples=512,seed=0,tolerance=.01)
            real_cli_report=dict(container_sha256=hashlib.sha256(real_container.read_bytes()).hexdigest(),
                audit=real_cli_audit,vertices=len(real_cli_restored.vertices),
                faces=len(real_cli_restored.faces),joint_count=len(real_cli_restored.skeleton.joints),
                pbr_texture_audit=real_cli_pbr_texture_audit,
                material_fidelity_audit=real_cli_material_fidelity,
                surface_comparison=real_cli_surface,
                maximum_weight_sum_error=float(np.max(np.abs(real_cli_restored.weights.sum(1)-1))),
                glb_bytes=real_cli_output.stat().st_size,
                glb_sha256=hashlib.sha256(real_cli_output.read_bytes()).hexdigest())
            real_rig_report=dict(source_sha256=source_rig.source_sha256,
                source_vertices=len(source_vertices),source_faces=len(source_rig.faces),
                source_joints=len(source_rig.skeleton.joints),normalization_scale=float(normalization.scale),
                active_voxels=len(geometry.coords),source_to_recovered_joint_ids=predicted_to_source.tolist(),
                vertices=len(real_restored.vertices),faces=len(real_restored.faces),
                weight_mae=float(np.mean(np.abs(aligned-targets))),weight_kl=weight_kl,
                source_topology=source_topology,surface_comparison=geometry_quality,
                strict_remesh=strict_remesh_report,audit=real_audit,
                checksummed_container_cli=real_cli_report,
                source_pbr=source_pbr_report,
                pbr_texture_audit=real_pbr_texture_audit,
                material_fidelity_audit=real_material_fidelity,
                requested_grid_size=args.real_grid_size,
                requested_texture_size=args.real_texture_size,
                glb_bytes=len(real_data),glb_sha256=hashlib.sha256(real_data).hexdigest(),
                scope='Real textured source rig through voxelization, sampled PBR rebaking, barycentric source-weight transfer, field/SkinAE export and bind-pose reload; structural texture audit only, no rendered material-fidelity or animation-transfer claim')
    report=dict(source_sha256=args.source_sha256,dependencies=PINS,device=torch.cuda.get_device_name(0),
        affine_sampling_passed=True,valid_texels=int(mask.sum()),valid_texel_max_error=max_error,
        full_atlas_max_error=padded_error,
        gradient_baking_max_error=gradient_error,raster_chunk_parity_passed=True,
        glb_bytes=len(data),glb_faces=len(loaded.faces),alpha_mode=loaded.visual.material.alphaMode,
        cli_audit=audit,cli_closed_outward_topology=command_topology,
        rigged_cli=rigged_report,real_rig_cli=real_rig_report,
        scope='analytic cube, constant sparse PBR and independent affine sampler oracle; no rendered fidelity claim')
    Path('/kaggle/working/export-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
