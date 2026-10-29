"""
eoqual.deconvolution — méthodes de déconvolution d'images à PSF connue
ou estimée, pensées pour l'observation de la Terre.

>>> from eoqual.deconvolution import deconvolve, list_methods
>>> from eoqual.deconvolution.psf import gaussian_psf
>>> from eoqual.deconvolution.simulate import simulate_observation
>>> psf = gaussian_psf(2.0)
>>> blurred = simulate_observation(scene, psf, sigma=0.01, seed=0)
>>> restored = deconvolve(blurred, psf, "tv_primal_dual", sigma=0.01)

Voir ``README.md`` et ``REFERENCE_TECHNIQUE.md``.
"""

__version__ = "0.1.0"
__title__ = "eoqual.deconvolution"
__description__ = "Image deconvolution methods for Earth observation"
__url__ = "https://github.com/EOQual/eoqual-deconvolution"
__uri__ = __url__
__doc__ = __description__ + " <" + __uri__ + ">"
__author__ = "Olivier Amram"
__email__ = "olivier.amram@free.fr"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2026 Olivier Amram"

from .config import METHODS_CONFIGS  # noqa: F401
from .runner import deconvolve, get_method, list_methods  # noqa: F401

__all__ = ["deconvolve", "get_method", "list_methods", "METHODS_CONFIGS"]
