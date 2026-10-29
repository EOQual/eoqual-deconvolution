"""
Point d'entrée unique pour appeler n'importe quelle méthode de
déconvolution par son nom, et lister les méthodes disponibles.

Exemple
-------
>>> from eoqual.deconvolution.runner import deconvolve, list_methods
>>> list_methods()
>>> restored = deconvolve(blurred, psf, method="tv_primal_dual", sigma=0.01)
"""

from __future__ import annotations

import numpy as np

from .config import METHODS_CONFIGS
from .utils import apply_per_channel

__all__ = ["deconvolve", "list_methods", "get_method"]

# Import différé : eoqual-denoising (Plug-and-Play) et scikit-image
# (Wiener non supervisé) ne sont chargés qu'au premier appel.
_DISPATCH = {
    "inverse": "eoqual.deconvolution.backends.linear:inverse_deconvolve",
    "wiener": "eoqual.deconvolution.backends.linear:wiener_deconvolve",
    "wiener_unsupervised": "eoqual.deconvolution.backends.linear:wiener_unsupervised_deconvolve",
    "tikhonov": "eoqual.deconvolution.backends.linear:tikhonov_deconvolve",
    "richardson_lucy": "eoqual.deconvolution.backends.iterative:richardson_lucy_deconvolve",
    "landweber": "eoqual.deconvolution.backends.iterative:landweber_deconvolve",
    "fista_wavelet": "eoqual.deconvolution.backends.sparse:fista_wavelet_deconvolve",
    "tv_primal_dual": "eoqual.deconvolution.backends.sparse:tv_primal_dual_deconvolve",
    "pnp_hqs": "eoqual.deconvolution.backends.pnp:pnp_hqs_deconvolve",
    "pnp_admm": "eoqual.deconvolution.backends.pnp:pnp_admm_deconvolve",
    "blind_l0": "eoqual.deconvolution.backends.blind:blind_l0_deconvolve",
}


def get_method(method: str):
    """Fonction implémentant ``method`` (voir :func:`list_methods`)."""
    if method not in METHODS_CONFIGS:
        raise ValueError(
            f"Méthode '{method}' inconnue. Méthodes disponibles : {list(METHODS_CONFIGS.keys())}"
        )
    module_path, func_name = _DISPATCH[method].split(":")
    module = __import__(module_path, fromlist=[func_name])
    return getattr(module, func_name)


def list_methods() -> None:
    """Affiche toutes les méthodes de déconvolution, leur famille et leur licence."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Méthodes disponibles dans eoqual.deconvolution", header_style="bold magenta")
    table.add_column("Méthode", style="bold cyan", no_wrap=True)
    table.add_column("Famille")
    table.add_column("Extra")
    table.add_column("Licence")
    table.add_column("Restriction", style="yellow")
    table.add_column("Référence")
    for name, cfg in METHODS_CONFIGS.items():
        table.add_row(name, cfg["family"], cfg["extra"] or "—", cfg["license"],
                      cfg["restriction"] or "—", cfg["reference"])
    Console().print(table)


def deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    method: str,
    *,
    channel_axis: int | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Applique une méthode de déconvolution par son nom.

    Parameters
    ----------
    image : np.ndarray
        Image floue 2D, ou 3D avec ``channel_axis``.
    psf : np.ndarray
        PSF 2D (normalisée à une somme de 1 en interne), commune à toutes
        les bandes. Pour ``blind_l0``, PSF initiale (ou ``None``).
    method : str
        Nom de la méthode — une des clés de :data:`METHODS_CONFIGS`.
    channel_axis : int, optional
        Axe des bandes d'une image multi-bandes : la méthode est appliquée
        bande par bande (``sigma`` scalaire ou un σ par bande).
    **kwargs
        Transmis à la méthode (``sigma``, ``boundary``... voir sa docstring).

    Returns
    -------
    np.ndarray
        Image restaurée, ``float64``, même forme que ``image``.
    """
    func = get_method(method)
    if channel_axis is None:
        return func(image, psf, **kwargs)
    return apply_per_channel(lambda band, **kw: func(band, psf, **kw), image, channel_axis, **kwargs)
