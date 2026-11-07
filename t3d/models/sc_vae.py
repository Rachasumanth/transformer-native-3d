"""Multi-task SC-VAE with Hermite normal prediction head for continuous 3D surface normals."""
from t3d.sc_vae import (
    SparsePosterior,
    SparseCompressionVAE,
    HermiteNormalHead,
    HermiteHead,
    hermite_normal_loss,
    reconstruction_loss,
)

__all__ = [
    "SparsePosterior",
    "SparseCompressionVAE",
    "HermiteNormalHead",
    "HermiteHead",
    "hermite_normal_loss",
    "reconstruction_loss",
]
