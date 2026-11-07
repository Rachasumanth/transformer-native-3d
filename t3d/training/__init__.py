"""Training routines and optimization pipelines for T3D models."""
from .vae_trainer import VAETrainer, main
from .flow_trainer import (
    FlowTrainer,
    FlowTrainerConfig,
    sample_logit_normal,
    rectified_flow_pair,
    rectified_flow_loss,
    chamfer_distance,
    decoder_in_the_loop_loss,
    sample_ode,
)

__all__ = [
    "VAETrainer",
    "FlowTrainer",
    "FlowTrainerConfig",
    "sample_logit_normal",
    "rectified_flow_pair",
    "rectified_flow_loss",
    "chamfer_distance",
    "decoder_in_the_loop_loss",
    "sample_ode",
    "main",
]
