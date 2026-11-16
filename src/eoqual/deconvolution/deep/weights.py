"""Déprécié : voir :mod:`eoqual.neural.weights`."""
from eoqual.neural.weights import KNOWN_WEIGHTS, resolve_weights, sha256, weights_dir  # noqa: F401

__all__ = ["KNOWN_WEIGHTS", "weights_dir", "resolve_weights", "sha256"]
