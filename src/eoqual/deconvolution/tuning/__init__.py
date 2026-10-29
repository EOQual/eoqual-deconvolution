"""
Réglage des paramètres.

- :mod:`.criteria` : choix de λ des méthodes linéaires sans image de
  référence (GCV, discrépance, L-curve) ;
- :mod:`.optimizer` : optimisation bayésienne des paramètres de toute
  méthode (extra ``optim``), avec image de référence (PSNR, SSIM) ou
  sans (indice de netteté SI), métriques d'eoqual-metrics.
"""
from .criteria import DEFAULT_LAMBDAS, discrepancy_curve, gcv_curve, l_curve, select_lambda  # noqa: F401
from .optimizer import METRICS, OptimizationResult, optimize_parameters, score_image  # noqa: F401
from .search_spaces import SEARCH_SPACES, get_search_space, load_search_space, validate_search_space  # noqa: F401

__all__ = [
    "DEFAULT_LAMBDAS",
    "select_lambda",
    "gcv_curve",
    "discrepancy_curve",
    "l_curve",
    "optimize_parameters",
    "OptimizationResult",
    "METRICS",
    "score_image",
    "SEARCH_SPACES",
    "get_search_space",
    "load_search_space",
    "validate_search_space",
]
