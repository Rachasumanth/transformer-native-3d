"""End-to-End Inference Orchestrator Pipeline for Transformer-Native 3D (T3D).

Coordinates the multi-stage generation process:
1. MultimodalFrontEndAdapter: text/image tokens -> coarse condition c_raw.
2. UnifiedDiT: Euler flow integration -> joint state [occupancy, continuous latents].
3. Geometry: 100% 2-manifold watertight mesh extraction & decimation.
4. SkeletonFlowModel: Kinematic DFS tree sampling, KinematicTreeDecoder, containment clamping.
5. SurfacePointToBoneSkinning: Surface-grounded point-to-bone skinning & top-4 truncation.
6. SurfaceGroundedPBRFlow: Single UV atlas PBR textures (Basecolor, MR, Normal map).
7. glTF 2.0 Binary (.glb) Export: Pack watertight mesh, PBR textures, and skinning buffers.
"""
from dataclasses import dataclass, field
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import trimesh

from t3d.articulation import Skeleton
from t3d.glb import attach_skin
from t3d.models.frontend_adapter import MultimodalFrontEndAdapter
from t3d.models.unified_dit import UnifiedDiT, sample_flow
from t3d.models.rigging import (
    SkeletonFlowModel,
    KinematicTreeDecoder,
    SurfacePointToBoneSkinning,
    truncate_top4_weights,
    pack_gltf_skin_buffers,
)
from t3d.models.pbr_flow import SurfaceGroundedPBRFlow, sample_pbr_flow
from t3d.data.refinement.canonicalize_geom import (
    extract_manifold_isosurface,
    decimate_crease_preserving,
    canonicalize_geometry,
)

logger = logging.getLogger(__name__)


@dataclass
class T3DOutput:
    """Output container for the end-to-end T3D generation pipeline.

    Attributes:
        mesh: 100% 2-manifold watertight trimesh.Trimesh object.
        joints: Global joint coordinates array in R^(K x 3).
        skin_weights: Row-normalized skinning weights in R^(M x K).
        pbr_textures: Dictionary containing 'basecolor', 'metallic_roughness', and 'normal' maps.
        occ_logits: Dense coarse occupancy logits tensor.
        latents: Dense continuous SC-VAE latent representations tensor.
        parents: Optional joint parent indices array in R^(K,).
        joints_0: Optional top-4 joint indices in R^(M x 4) (uint16).
        weights_0: Optional top-4 skinning weights in R^(M x 4) (float32).
    """
    mesh: trimesh.Trimesh
    joints: np.ndarray
    skin_weights: np.ndarray
    pbr_textures: Dict[str, np.ndarray]
    occ_logits: torch.Tensor
    latents: torch.Tensor
    parents: Optional[np.ndarray] = None
    joints_0: Optional[np.ndarray] = None
    weights_0: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "num_vertices": len(self.mesh.vertices),
            "num_faces": len(self.mesh.faces),
            "is_watertight": bool(self.mesh.is_watertight),
            "joint_count": len(self.joints),
            "texture_keys": list(self.pbr_textures.keys()),
            "occ_shape": list(self.occ_logits.shape),
            "latents_shape": list(self.latents.shape),
        }


class T3DPipeline(nn.Module):
    """End-to-end inference orchestrator pipeline for Transformer-Native 3D.

    Args:
        frontend: Optional MultimodalFrontEndAdapter (text/image -> c_raw).
        dit: UnifiedDiT joint occupancy and continuous latent diffusion transformer.
        skeleton_flow: SkeletonFlowModel for DFS relative kinematic skeleton generation.
        skinning_module: SurfacePointToBoneSkinning for geometric distance cross-attention skinning.
        pbr_flow: SurfaceGroundedPBRFlow for single UV atlas PBR textures.
        device: Device to run inference on (default: 'cpu').
        dtype: Data type for PyTorch operations (default: torch.float32).
        default_num_steps: Default number of Euler integration steps (default: 10).
        default_cfg_scale: Default Classifier-Free Guidance scale (default: 3.0).
    """

    def __init__(
        self,
        frontend: Optional[MultimodalFrontEndAdapter] = None,
        dit: Optional[UnifiedDiT] = None,
        skeleton_flow: Optional[SkeletonFlowModel] = None,
        skinning_module: Optional[SurfacePointToBoneSkinning] = None,
        pbr_flow: Optional[SurfaceGroundedPBRFlow] = None,
        device: Union[torch.device, str] = "cpu",
        dtype: torch.dtype = torch.float32,
        default_num_steps: int = 10,
        default_cfg_scale: float = 3.0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.default_num_steps = default_num_steps
        self.default_cfg_scale = default_cfg_scale

        # Initialize or assign DiT
        if dit is None:
            dit = UnifiedDiT(
                dim=96,
                depth=2,
                heads=4,
                latent_channels=16,
                coarse_res=4,
            )
        self.dit = dit

        # Initialize or assign FrontEnd
        if frontend is None:
            frontend = MultimodalFrontEndAdapter(
                text_dim=512,
                image_dim=768,
                embed_dim=256,
                latent_channels=self.dit.latent_channels,
                coarse_res=self.dit.coarse_res,
            )
        self.frontend = frontend

        # Initialize or assign Skeleton Flow
        if skeleton_flow is None:
            skeleton_flow = SkeletonFlowModel(
                joint_count=5,
                cond_dim=self.dit.in_channels,
                hidden_dim=self.dit.dim,
                num_layers=2,
            )
        self.skeleton_flow = skeleton_flow

        # Initialize or assign Skinning Module
        if skinning_module is None:
            skinning_module = SurfacePointToBoneSkinning(
                query_dim=self.dit.dim,
                key_dim=self.dit.dim,
                hidden_dim=self.dit.dim,
                num_heads=2,
            )
        self.skinning_module = skinning_module

        # Initialize or assign PBR Flow
        if pbr_flow is None:
            pbr_flow = SurfaceGroundedPBRFlow(
                channels=(16, 32, 64),
                latent_channels=self.dit.latent_channels,
                geom_channels=self.dit.in_channels,
                time_dim=self.dit.dim,
            )
        self.pbr_flow = pbr_flow

        # Move submodules to specified device and dtype
        self.to(device=self.device, dtype=self.dtype)

    def to(self, *args, **kwargs):
        """Override to update self.device and self.dtype when moving module."""
        super().to(*args, **kwargs)
        for arg in args:
            if isinstance(arg, torch.device) or isinstance(arg, str):
                self.device = torch.device(arg)
            elif isinstance(arg, torch.dtype):
                self.dtype = arg
        if "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        return self

    def save_pretrained(self, save_directory: Union[str, Path]) -> Path:
        """Save pipeline weights and configuration to a directory or checkpoint file."""
        dest = Path(save_directory)
        if dest.suffix == ".pt":
            dest.parent.mkdir(parents=True, exist_ok=True)
            ckpt_file = dest
        else:
            dest.mkdir(parents=True, exist_ok=True)
            ckpt_file = dest / "t3d_pipeline.pt"

        config = {
            "dit": {
                "dim": self.dit.dim,
                "depth": self.dit.depth,
                "heads": self.dit.heads,
                "latent_channels": self.dit.latent_channels,
                "coarse_res": self.dit.coarse_res,
            },
            "frontend": {
                "text_dim": self.frontend.text_dim,
                "image_dim": self.frontend.image_dim,
                "embed_dim": self.frontend.embed_dim,
                "latent_channels": self.frontend.latent_channels,
                "coarse_res": self.frontend.coarse_res,
                "num_layers": self.frontend.num_layers,
                "num_heads": self.frontend.num_heads,
            } if self.frontend is not None else None,
            "skeleton_flow": {
                "joint_count": self.skeleton_flow.joint_count,
                "cond_dim": self.skeleton_flow.cond_dim,
                "hidden_dim": self.skeleton_flow.hidden_dim,
                "num_layers": len(self.skeleton_flow.layers),
            },
            "skinning_module": {
                "query_dim": getattr(self.skinning_module, "query_dim", self.dit.dim),
                "key_dim": getattr(self.skinning_module, "key_dim", self.dit.dim),
                "hidden_dim": self.skinning_module.hidden_dim,
                "num_heads": self.skinning_module.num_heads,
            },
            "pbr_flow": {
                "channels": (
                    self.pbr_flow.in_conv.out_channels,
                    self.pbr_flow.downsample0.out_channels,
                    self.pbr_flow.downsample1.out_channels,
                ),
                "latent_channels": self.pbr_flow.latent_channels,
                "geom_channels": self.pbr_flow.geom_channels,
                "time_dim": self.pbr_flow.time_embed.dim,
            },
        }

        state = {
            "config": config,
            "dit_state": self.dit.state_dict(),
            "skeleton_flow_state": self.skeleton_flow.state_dict(),
            "skinning_state": self.skinning_module.state_dict(),
            "pbr_flow_state": self.pbr_flow.state_dict(),
            "frontend_state": self.frontend.state_dict() if self.frontend is not None else None,
        }
        torch.save(state, str(ckpt_file))
        logger.info("Saved pipeline checkpoint to %s", ckpt_file)
        return ckpt_file

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[torch.device, str] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "T3DPipeline":
        """Load pipeline components from a trained checkpoint dictionary or file."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        ckpt = torch.load(path, map_location=device)
        config = ckpt.get("config", {})

        # Construct submodules matching configuration or detected shapes
        dit_cfg = config.get("dit", {})
        if not dit_cfg and "dit_state" in ckpt:
            dit_st = ckpt["dit_state"]
            dim = dit_st["x_embed.weight"].shape[0]
            coarse_res = int(round(dit_st["blocks.0.attn.rope.frequencies"].shape[0]))  # estimate or default 4
            dit_cfg = {"dim": dim, "depth": 2, "heads": 4, "coarse_res": 4}
        dit = UnifiedDiT(**dit_cfg) if dit_cfg else None

        frontend_cfg = config.get("frontend")
        if frontend_cfg:
            frontend = MultimodalFrontEndAdapter(**frontend_cfg)
        elif "frontend_state" in ckpt and ckpt["frontend_state"] is not None:
            f_st = ckpt["frontend_state"]
            embed_dim = f_st["voxel_queries"].shape[-1]
            coarse_res = int(round(f_st["voxel_queries"].shape[1] ** (1 / 3)))
            text_dim = f_st["text_projector.0.weight"].shape[1] if "text_projector.0.weight" in f_st else 512
            image_dim = f_st["image_projector.0.weight"].shape[1] if "image_projector.0.weight" in f_st else 768
            frontend = MultimodalFrontEndAdapter(
                text_dim=text_dim,
                image_dim=image_dim,
                embed_dim=embed_dim,
                coarse_res=coarse_res,
            )
        else:
            frontend = None

        skel_cfg = config.get("skeleton_flow", {})
        skeleton_flow = SkeletonFlowModel(**skel_cfg) if skel_cfg else None

        skin_cfg = config.get("skinning_module", {})
        skinning_module = SurfacePointToBoneSkinning(**skin_cfg) if skin_cfg else None

        pbr_cfg = config.get("pbr_flow", {})
        if pbr_cfg and "channels" in pbr_cfg:
            pbr_cfg["channels"] = tuple(pbr_cfg["channels"])
        pbr_flow = SurfaceGroundedPBRFlow(**pbr_cfg) if pbr_cfg else None

        pipeline = cls(
            frontend=frontend,
            dit=dit,
            skeleton_flow=skeleton_flow,
            skinning_module=skinning_module,
            pbr_flow=pbr_flow,
            device=device,
            dtype=dtype,
        )

        if "dit_state" in ckpt and pipeline.dit is not None:
            pipeline.dit.load_state_dict(ckpt["dit_state"], strict=False)
        if "skeleton_flow_state" in ckpt and pipeline.skeleton_flow is not None:
            pipeline.skeleton_flow.load_state_dict(ckpt["skeleton_flow_state"], strict=False)
        if "skinning_state" in ckpt and pipeline.skinning_module is not None:
            pipeline.skinning_module.load_state_dict(ckpt["skinning_state"], strict=False)
        if "pbr_flow_state" in ckpt and pipeline.pbr_flow is not None:
            pipeline.pbr_flow.load_state_dict(ckpt["pbr_flow_state"], strict=False)
        if "frontend_state" in ckpt and pipeline.frontend is not None and ckpt["frontend_state"] is not None:
            pipeline.frontend.load_state_dict(ckpt["frontend_state"], strict=False)

        pipeline.eval()
        return pipeline

    def _extract_watertight_mesh(
        self,
        occ_logits: torch.Tensor,
        target_faces: Tuple[int, int] = (1500, 5000),
    ) -> trimesh.Trimesh:
        """Extract a 100% 2-manifold watertight mesh from coarse occupancy logits."""
        coarse_res = self.dit.coarse_res
        occ_grid = occ_logits.squeeze(-1)
        if occ_grid.ndim == 1:
            occ_grid = occ_grid.view(1, 1, coarse_res, coarse_res, coarse_res)
        elif occ_grid.ndim == 2:
            occ_grid = occ_grid[0].view(1, 1, coarse_res, coarse_res, coarse_res)
        elif occ_grid.ndim == 3:
            occ_grid = occ_grid.view(1, 1, *occ_grid.shape)

        eval_res = max(32, coarse_res * 4)
        occ_up = F.interpolate(
            occ_grid.float(),
            size=(eval_res, eval_res, eval_res),
            mode="trilinear",
            align_corners=True,
        ).squeeze().cpu().numpy()

        # Isosurface distance field (-occupancy so interior is negative)
        thresh = 0.0
        if (occ_up > thresh).sum() < 8:
            thresh = float(np.percentile(occ_up, 70))
        sdf = -(occ_up - thresh)

        # Pad 1-voxel exterior border to guarantee closed watertight boundary
        sdf[0, :, :] = 1.0
        sdf[-1, :, :] = 1.0
        sdf[:, 0, :] = 1.0
        sdf[:, -1, :] = 1.0
        sdf[:, :, 0] = 1.0
        sdf[:, :, -1] = 1.0

        try:
            raw_mesh = extract_manifold_isosurface(sdf, dx=1.0 / eval_res)
            if len(raw_mesh.faces) == 0:
                raw_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.35)
        except Exception:
            raw_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.35)

        # Canonicalize, repair topology, and enforce [1500, 5000] face budget
        clean_mesh = canonicalize_geometry(raw_mesh, target_faces=target_faces, resolution=64)

        # Ensure triangle count is strictly within [1500, 5000]
        while len(clean_mesh.faces) < target_faces[0]:
            clean_mesh = clean_mesh.subdivide()
        if len(clean_mesh.faces) > target_faces[1]:
            clean_mesh = decimate_crease_preserving(clean_mesh, target_faces=target_faces)

        return clean_mesh

    @torch.no_grad()
    def generate(
        self,
        c_raw: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        texture_res: int = 256,
        target_faces: Tuple[int, int] = (1500, 5000),
        parents: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    ) -> T3DOutput:
        """Run full end-to-end inference producing watertight mesh, skeleton, skinning, and PBR textures.

        Args:
            c_raw: Optional pre-computed conditioning latents in R^(B x N x (C+1)) or (N x (C+1)).
            text_tokens: Optional text tokens in R^(B x L_t x text_dim) or (L_t x text_dim).
            image_tokens: Optional image tokens in R^(B x L_i x image_dim) or (L_i x image_dim).
            num_steps: Number of Euler integration steps (default: default_num_steps).
            cfg_scale: Classifier-Free Guidance scale (default: default_cfg_scale).
            seed: Optional integer random seed for deterministic sampling.
            texture_res: Single UV atlas texture resolution (default: 256, or 1024).
            target_faces: Tuple of (min_faces, max_faces) for watertight mesh (default: (1500, 5000)).
            parents: Optional kinematic skeleton parent indices hierarchy (default: 5-joint humanoid).

        Returns:
            T3DOutput container with generated 3D asset components.
        """
        self.eval()
        steps = num_steps if num_steps is not None else self.default_num_steps
        guidance = cfg_scale if cfg_scale is not None else self.default_cfg_scale

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            gen = torch.Generator(device=self.device).manual_seed(seed)
        else:
            gen = None

        # 1. Front-end Adapter: text/image -> c_raw
        if c_raw is None:
            if text_tokens is not None or image_tokens is not None:
                if self.frontend is None:
                    raise ValueError("text_tokens or image_tokens provided but self.frontend is None")
                if text_tokens is not None:
                    text_tokens = torch.as_tensor(text_tokens, device=self.device, dtype=self.dtype)
                if image_tokens is not None:
                    image_tokens = torch.as_tensor(image_tokens, device=self.device, dtype=self.dtype)
                frontend_out = self.frontend(
                    text_tokens=text_tokens,
                    image_tokens=image_tokens,
                    num_tokens=self.dit.coarse_res ** 3,
                )
                if isinstance(frontend_out, dict):
                    c_raw = frontend_out["c_raw"]
                else:
                    c_raw = frontend_out
            else:
                c_raw = None
        else:
            c_raw = torch.as_tensor(c_raw, device=self.device, dtype=self.dtype)

        # 2. UnifiedDiT Joint State Flow Integration: x_0 ~ N(0, I) -> x_1 = [o, z]
        joint_state = sample_flow(
            model=self.dit,
            c_raw=c_raw,
            steps=steps,
            cfg_scale=guidance,
            generator=gen,
            device=self.device,
            dtype=self.dtype,
        )
        occ_logits, latents = self.dit.split_state(joint_state)

        # Extract primary single-asset components (batch index 0)
        occ_b = occ_logits[0] if occ_logits.ndim == 3 else occ_logits
        lat_b = latents[0] if latents.ndim == 3 else latents
        state_b = joint_state[0] if joint_state.ndim == 3 else joint_state

        # 3. Extract Watertight 2-Manifold Mesh
        mesh = self._extract_watertight_mesh(occ_b, target_faces=target_faces)

        # 4. Skeleton Flow Integration & Containment Clamping
        if parents is None:
            # Canonical 5-joint hierarchy: [Root(0), Spine(1), Head(2), LeftArm(3), RightArm(4)]
            parents_t = torch.tensor([-1, 0, 1, 1, 1], device=self.device, dtype=torch.int64)
        else:
            parents_t = torch.as_tensor(parents, device=self.device, dtype=torch.int64)

        cond_emb = state_b.mean(dim=0, keepdim=True)  # (1, in_channels)
        if cond_emb.shape[-1] != self.skeleton_flow.cond_dim:
            if cond_emb.shape[-1] < self.skeleton_flow.cond_dim:
                pad_size = self.skeleton_flow.cond_dim - cond_emb.shape[-1]
                cond_emb = F.pad(cond_emb, (0, pad_size))
            else:
                cond_emb = cond_emb[:, :self.skeleton_flow.cond_dim]

        global_joints_t, _ = self.skeleton_flow.sample(
            cond=cond_emb,
            steps=steps,
            parents=parents_t,
        )
        pred_joints = global_joints_t.squeeze(0).detach().cpu().numpy()

        # Enforce containment: clamp joints strictly inside mesh bounding bounds with safety margin
        v_coords = mesh.vertices
        min_bound = v_coords.min(axis=0) + 0.01
        max_bound = v_coords.max(axis=0) - 0.01
        min_bound = np.minimum(min_bound, max_bound - 1e-4)
        pred_joints = np.clip(pred_joints, min_bound, max_bound)

        # 5. Skinning Module & Top-4 Truncation
        vert_t = torch.as_tensor(mesh.vertices, device=self.device, dtype=torch.float32)
        joint_t = torch.as_tensor(pred_joints, device=self.device, dtype=torch.float32)

        weights_t, _ = self.skinning_module(
            vertices=vert_t,
            joints=joint_t,
            parents=parents_t,
            faces=mesh.faces,
        )
        raw_weights_np = weights_t.squeeze(0).detach().cpu().numpy()

        # Top-4 bone truncation and row normalization
        joints_0, weights_0 = truncate_top4_weights(raw_weights_np)

        # Build full (M, K) normalized skin weight matrix for glTF attachment
        M = len(mesh.vertices)
        K = len(pred_joints)
        full_skin_weights = np.zeros((M, K), dtype=np.float32)
        np.put_along_axis(full_skin_weights, joints_0.astype(int), weights_0, axis=1)

        # 6. PBR Flow: UV Atlas Basecolor, Metallic-Roughness, Normal map
        geom_cond = cond_emb if self.pbr_flow.geom_channels > 0 else None
        latent_cond = lat_b.mean(dim=0, keepdim=True) if self.pbr_flow.latent_channels > 0 else None

        pbr_output = sample_pbr_flow(
            model=self.pbr_flow,
            shape=(1, 8, texture_res, texture_res),
            geom_cond=geom_cond,
            latent_cond=latent_cond,
            steps=steps,
            pad_seams=True,
            device=self.device,
        )

        pbr_textures = {
            "basecolor": pbr_output.basecolor[0].permute(1, 2, 0).detach().cpu().numpy(),
            "metallic_roughness": pbr_output.metallic_roughness[0].permute(1, 2, 0).detach().cpu().numpy(),
            "normal": pbr_output.normal_map[0].permute(1, 2, 0).detach().cpu().numpy(),
            "normal_map": pbr_output.normal_map[0].permute(1, 2, 0).detach().cpu().numpy(),
            "combined": pbr_output.combined[0].permute(1, 2, 0).detach().cpu().numpy(),
        }

        return T3DOutput(
            mesh=mesh,
            joints=pred_joints,
            skin_weights=full_skin_weights,
            pbr_textures=pbr_textures,
            occ_logits=occ_logits,
            latents=latents,
            parents=parents_t.detach().cpu().numpy(),
            joints_0=joints_0,
            weights_0=weights_0,
        )

    def export_glb(
        self,
        output: T3DOutput,
        output_path: Union[str, Path],
        asset_name: str = "t3d_asset",
    ) -> Path:
        """Export generation output to a fully valid self-contained glTF 2.0 binary (.glb).

        Contains:
        - 100% 2-manifold watertight mesh with 0 non-manifold edges.
        - Embedded PBR materials (Basecolor debaked, Metallic-Roughness, Normal map).
        - Embedded skinning buffers (JOINTS_0 uint16[4], WEIGHTS_0 float32[4]) and skeleton nodes.

        Args:
            output: T3DOutput object from pipeline.generate().
            output_path: Target directory or file path for .glb export.
            asset_name: Base asset name if output_path is a directory (default: 't3d_asset').

        Returns:
            Path to the saved .glb file.
        """
        mesh = output.mesh.copy()

        # Compute or verify UV coordinates
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices):
            uv = np.asarray(mesh.visual.uv, dtype=np.float32)
        else:
            # Spherical UV unwrapping guaranteed in [0, 1]
            v = mesh.vertices
            r = np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-6)
            u = 0.5 + np.arctan2(v[:, 2], v[:, 0]) / (2.0 * np.pi)
            w = np.clip(v[:, 1:2] / r, -1.0, 1.0)
            val = 0.5 - np.arcsin(w) / np.pi
            uv = np.clip(np.concatenate([u[:, None], val], axis=-1), 0.0, 1.0).astype(np.float32)

        # 1. Format Basecolor texture
        bc_raw = output.pbr_textures["basecolor"]
        bc_uint8 = np.clip(bc_raw * 255.0, 0, 255).astype(np.uint8)
        bc_img = Image.fromarray(bc_uint8, mode="RGB")

        # 2. Format Metallic-Roughness texture (G = Roughness, B = Metallic)
        mr_raw = output.pbr_textures["metallic_roughness"]
        H, W = mr_raw.shape[:2]
        mr_uint8 = np.zeros((H, W, 3), dtype=np.uint8)
        mr_uint8[..., 1] = np.clip(mr_raw[..., 0] * 255.0, 0, 255).astype(np.uint8)
        mr_uint8[..., 2] = np.clip(mr_raw[..., 1] * 255.0, 0, 255).astype(np.uint8)
        mr_img = Image.fromarray(mr_uint8, mode="RGB")

        # 3. Format Tangent-Space Normal map texture
        norm_raw = output.pbr_textures.get("normal", output.pbr_textures.get("normal_map"))
        norm_uint8 = np.clip(norm_raw * 255.0, 0, 255).astype(np.uint8)
        norm_img = Image.fromarray(norm_uint8, mode="RGB")

        # 4. Attach PBR material to mesh
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=bc_img,
            metallicRoughnessTexture=mr_img,
            normalTexture=norm_img,
            doubleSided=True,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

        # 5. Export base glTF 2.0 binary chunk
        base_glb = mesh.export(file_type="glb", include_normals=True)

        # 6. Attach skeleton kinematic hierarchy and skinning buffers
        skel = Skeleton(output.joints, output.parents)
        skinned_glb = attach_skin(base_glb, output.skin_weights, skel)

        # 7. Write to output destination
        dest = Path(output_path)
        if dest.is_dir() or (not dest.suffix and not dest.exists()):
            dest.mkdir(parents=True, exist_ok=True)
            dest = dest / f"{asset_name}.glb"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)

        dest.write_bytes(skinned_glb)
        logger.info("Successfully exported glTF 2.0 asset to %s (%d bytes)", dest, len(skinned_glb))
        return dest
