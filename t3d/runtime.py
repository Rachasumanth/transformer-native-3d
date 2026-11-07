"""Long-lived inference runtime used by the HTTP service and integrations.

The CLI is optimized for one-shot jobs. The runtime keeps the text encoder,
Stage 1 occupancy flow, Stage 2 geometry flow, and SC-VAE resident so repeated
requests do not pay model-loading cost. It is intentionally small and explicit
about which checkpoints it loads.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

from .checkpoints import resolve_checkpoint, resolve_optional
from .conditioning import PromptConditioner
from .container import dumps
from .extraction import voxels_to_mesh
from .flow import SparseFlowDiT
from .generate import generate_asset_from_latents, generate_latents_from_prompt
from .glb import pack_static_glb, validate_glb_semantics
from .sc_vae import SparseCompressionVAE


def select_device(requested: str | None = None) -> str:
    value = (requested or os.environ.get('T3D_DEVICE') or '').lower()
    if value in ('cpu', 'cuda'):
        if value == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('T3D_DEVICE=cuda but CUDA is unavailable')
        return value
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class T3DRuntime:
    """Resident text-to-3D pipeline with an explicit shape/material surface."""

    def __init__(self, *, device: str | None = None, artifacts: str | Path | None = None,
                 material: bool = False, verify: bool = True):
        self.device = select_device(device)
        self.artifacts = Path(artifacts) if artifacts is not None else None
        self.lock = threading.RLock()
        self.request_count = 0
        self.error_count = 0

        self.occupancy_path = resolve_checkpoint('occupancy', resolution=64, root=self.artifacts, verify=verify)
        self.geometry_path = resolve_checkpoint('geometry', resolution=64, root=self.artifacts, verify=verify)
        self.vae_path = resolve_checkpoint('shape_vae', resolution=64, root=self.artifacts, verify=verify)
        self.resolution = 64
        self.grid_size = (64, 64, 64)
        self.latent_grid_size = 4
        self.coarse_grid_size = 4

        occ_data = torch.load(self.occupancy_path, map_location=self.device, weights_only=False)
        self.conditioner = PromptConditioner.from_env(output_channels=32).to(self.device)
        self.conditioner.text_projection.load_state_dict(occ_data['text_projection'])

        occ_blocks = [int(k.split('.')[1]) for k in occ_data['model'] if k.startswith('blocks.')]
        occ_width = occ_data['model']['input.weight'].shape[0]
        self.occupancy = SparseFlowDiT(
            channels=1, condition_channels=32, width=occ_width,
            heads=max(1, occ_width // 24), depth=max(occ_blocks) + 1, patch_size=64,
        ).to(self.device)
        self.occupancy.load_state_dict(occ_data['model'])

        geom_data = torch.load(self.geometry_path, map_location=self.device, weights_only=False)
        channels = int(geom_data.get('latent_channels', 16))
        blocks = [int(k.split('.')[1]) for k in geom_data['model'] if k.startswith('blocks.')]
        width = int(geom_data.get('width', geom_data['model']['input.weight'].shape[0]))
        self.geometry = SparseFlowDiT(
            channels=channels, condition_channels=32, width=width,
            heads=int(geom_data.get('heads', max(1, width // 24))),
            depth=int(geom_data.get('depth', max(blocks) + 1)), patch_size=64,
        ).to(self.device)
        self.geometry.load_state_dict(geom_data['model'])

        self.vae = self._load_vae(self.vae_path)

        self.material_flow = self.material_vae = None
        self.material_paths: dict[str, str] = {}
        if material:
            self._load_material()

        self.unified_model = None
        self.unified_stats = None
        self.unified_path = resolve_optional('unified_64', root=self.artifacts)
        if self.unified_path is not None:
            try:
                from .unified_dit import UnifiedFlowDiT
                u_data = torch.load(self.unified_path, map_location=self.device, weights_only=False)
                u_channels = int(u_data.get('latent_channels', 16))
                u_width = int(u_data.get('width', 144))
                u_heads = int(u_data.get('heads', 6))
                u_depth = int(u_data.get('depth', 6))
                self.unified_model = UnifiedFlowDiT(
                    latent_channels=u_channels, condition_channels=32,
                    width=u_width, heads=u_heads, depth=u_depth, patch_size=64
                ).to(self.device)
                self.unified_model.load_state_dict(u_data['model'])
                self.unified_model.eval()
                for p in self.unified_model.parameters():
                    p.requires_grad_(False)
                self.unified_stats = (u_data.get('latent_mean'), u_data.get('latent_std'))
            except Exception:
                self.unified_model = None

        for model in (self.occupancy, self.geometry, self.vae):
            model.eval()
        for parameter in self.occupancy.parameters():
            parameter.requires_grad_(False)
        for parameter in self.geometry.parameters():
            parameter.requires_grad_(False)
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        if self.material_flow is not None:
            self.material_flow.eval()
            for parameter in self.material_flow.parameters():
                parameter.requires_grad_(False)
        if self.material_vae is not None:
            self.material_vae.eval()
            for parameter in self.material_vae.parameters():
                parameter.requires_grad_(False)

    def _load_vae(self, path: Path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        config = {k: v for k, v in data['config'].items() if k in (
            'mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
        if 'input.weight' in data['model']:
            blocks = max(int(k.split('.')[1]) for k in data['model'] if k.startswith('encoder_blocks.'))
            config['channels'] = tuple(data['model'][f'encoder_blocks.{i}.scale'].shape[0] for i in range(blocks + 1))
        if self.device == 'cpu':
            config['convolution_backend'] = 'reference'
        model = SparseCompressionVAE(**config).to(self.device)
        model.load_state_dict(data['model'])
        return model

    def _load_material(self):
        flow_path = resolve_optional('material_flow', root=self.artifacts)
        vae_path = resolve_optional('material_vae', root=self.artifacts)
        if flow_path is None or vae_path is None:
            raise RuntimeError('Material inference requested but material_flow/material_vae checkpoints are unavailable')
        data = torch.load(flow_path, map_location=self.device, weights_only=False)
        cfg = data['config']
        self.material_flow = SparseFlowDiT(
            channels=int(cfg.get('channels', 16)), condition_channels=int(cfg.get('condition_channels', 32)),
            width=int(cfg.get('width', 96)), heads=int(cfg.get('heads', 4)),
            depth=int(cfg.get('depth', 4)), patch_size=int(cfg.get('patch_size', 64)),
            shape_channels=int(cfg.get('shape_channels', 16)),
        ).to(self.device)
        self.material_flow.load_state_dict(data['model'])
        self.material_vae = self._load_vae(vae_path)
        self.material_paths = {'material_flow': str(flow_path), 'material_vae': str(vae_path)}

    @property
    def checkpoints(self) -> dict:
        return {
            'occupancy': str(self.occupancy_path),
            'geometry': str(self.geometry_path),
            'shape_vae': str(self.vae_path),
            **self.material_paths,
        }

    def readiness(self) -> dict:
        return {
            'ready': True,
            'device': self.device,
            'resolution': self.resolution,
            'material': self.material_flow is not None,
            'checkpoints': self.checkpoints,
            'requests': self.request_count,
            'errors': self.error_count,
        }

    def generate(self, prompt: str, *, format: str = 'glb', steps: int = 20,
                 guidance: float = 2.0, occupancy_guidance: float = 1.0,
                 seed: int = 42, threshold: float = 0.5,
                 decode_threshold: float | None = None, method: str = 'midpoint') -> bytes:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must be a non-empty string')
        if len(prompt) > 500:
            raise ValueError('prompt must be at most 500 characters')
        if format not in ('glb', 't3d'):
            raise ValueError("format must be 'glb' or 't3d'")
        if not 1 <= steps <= 100:
            raise ValueError('steps must be in [1, 100]')
        if not 0.0 <= guidance <= 10.0:
            raise ValueError('guidance must be in [0, 10]')
        if method not in ('euler', 'midpoint'):
            raise ValueError("method must be 'euler' or 'midpoint'")

        self.request_count += 1
        with self.lock, torch.no_grad():
            try:
                if self.unified_model is not None:
                    from .generate import generate_latents_unified
                    active, shape, mat = generate_latents_unified(
                        prompt, self.unified_model, self.conditioner,
                        steps=steps, guidance=guidance, method=method,
                        seed=seed, device=self.device, latent_stats=self.unified_stats,
                    )
                else:
                    active, shape, mat = generate_latents_from_prompt(
                        prompt, self.occupancy, self.geometry, self.conditioner,
                    material_model=self.material_flow,
                    coarse_grid_size=self.coarse_grid_size,
                    latent_grid_size=self.latent_grid_size,
                    occupancy_steps=steps, geometry_steps=steps, material_steps=steps,
                    guidance=guidance, occupancy_guidance=occupancy_guidance,
                    occupancy_threshold=threshold, device=self.device, seed=seed,
                    return_material=True, method=method,
                )
                asset = generate_asset_from_latents(
                    active, shape, self.vae,
                    grid_size=self.grid_size,
                    threshold=0.46 if decode_threshold is None else decode_threshold,
                    material_latents=mat, material_vae_model=self.material_vae,
                )
                if format == 't3d':
                    return dumps(asset)
                vertices, faces = voxels_to_mesh(asset, resolve_nonmanifold=True)
                glb = pack_static_glb(vertices, faces, pbr=asset.pbr)
                validate_glb_semantics(glb)
                return glb
            except Exception:
                self.error_count += 1
                raise
