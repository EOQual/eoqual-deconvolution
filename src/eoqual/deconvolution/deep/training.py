"""Déprécié : voir :mod:`eoqual.deconvolution.training`."""
from ..training import (  # noqa: F401
    NoiseSampler, PatchSampler, PSFSampler, TrainingConfig, evaluate_usrnet, load_training_images, train_usrnet,
)

__all__ = ["PSFSampler", "NoiseSampler", "TrainingConfig", "PatchSampler", "load_training_images",
           "train_usrnet", "evaluate_usrnet"]
