"""Import paresseux de PyTorch — délégué à eoqual-neural (conservé pour compatibilité)."""
from __future__ import annotations


def require_torch():
    try:
        from eoqual.neural.torch_utils import require_torch as _require
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            "Cette méthode nécessite eoqual-neural (PyTorch) : pip install 'eoqual-deconvolution[deep]'."
        ) from exc
    return _require()


def no_grad():
    return require_torch().no_grad()
