"""Experimental text-to-occupancy DiT research archive.

This directory houses the legacy Stage 1 text-conditioned coarse occupancy DiT
(2.8M parameters) and prompt conditioning projections.
The core production pipeline of T3D operates directly on raw 3D input geometry
(Raw 3D Mesh -> O-Voxel -> SC-VAE Refinement -> PBR Baking -> Rigging -> GLB),
bypassing unconstrained text-conditioned occupancy hallucinations.
"""
