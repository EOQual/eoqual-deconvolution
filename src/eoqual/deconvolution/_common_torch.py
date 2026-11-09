"""Import paresseux de PyTorch (extra ``deep``) avec un message d'installation clair."""
from __future__ import annotations


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            "Cette méthode nécessite PyTorch : pip install 'eoqual-deconvolution[deep]'."
        ) from exc
    return torch


def no_grad():
    return require_torch().no_grad()
