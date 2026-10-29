#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrôles et conversions communs à toutes les méthodes : image, PSF,
écart-type du bruit, dynamique, images multi-bandes.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from eoqual.core.noise import estimate_noise_sigma

__all__ = [
    "check_image",
    "check_psf",
    "estimate_noise_sigma",
    "resolve_sigma",
    "to_unit_range",
    "from_unit_range",
    "apply_per_channel",
]


def check_image(image: np.ndarray) -> np.ndarray:
    """
    Copie ``float64`` d'une image 2D finie.

    Raises
    ------
    ValueError
        Si l'image n'est pas 2D ou contient des valeurs non finies.
    """
    arr = np.array(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"Image 2D attendue, reçu ndim={arr.ndim}. Pour une image "
            "multi-bandes, passer channel_axis à deconvolve()."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("L'image contient des valeurs non finies (NaN/inf).")
    return arr


def check_psf(psf: np.ndarray, image_shape: tuple[int, int] | None = None) -> np.ndarray:
    """
    Copie ``float64`` de la PSF, ramenée à une somme de 1.

    Le centre de la PSF est le pixel ``(kh // 2, kw // 2)`` (convention
    de :func:`eoqual.core.kernel.psf2otf`) : pour une taille paire, le pic
    doit être en ``(kh // 2, kw // 2)``, sinon l'image restaurée est
    décalée d'un pixel. Une PSF de taille impaire, centrée, évite toute
    ambiguïté (:func:`eoqual.deconvolution.psf.center_psf`).

    Raises
    ------
    ValueError
        Si la PSF n'est pas 2D, contient des valeurs non finies, a une
        somme nulle ou négative, ou est plus grande que l'image.
    """
    h = np.array(psf, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"PSF 2D attendue, reçu ndim={h.ndim}.")
    if not np.all(np.isfinite(h)):
        raise ValueError("La PSF contient des valeurs non finies (NaN/inf).")
    total = h.sum()
    if total <= 0:
        raise ValueError(f"La somme de la PSF doit être positive, reçu {total}.")
    if image_shape is not None and (h.shape[0] > image_shape[0] or h.shape[1] > image_shape[1]):
        raise ValueError(f"PSF {h.shape} plus grande que l'image {tuple(image_shape)}.")
    return h / total


def resolve_sigma(image: np.ndarray, sigma: float | None) -> float:
    """
    Renvoie ``sigma`` s'il est fourni, sinon son estimation sur ``image``
    (:func:`eoqual.core.noise.estimate_noise_sigma`).

    Sur une image floue, la sous-bande diagonale la plus fine contient peu
    de signal : l'estimation y est en général fiable. Fournir σ dès qu'il
    est connu (caractérisation du capteur).
    """
    if sigma is None:
        return estimate_noise_sigma(image)
    if sigma < 0:
        raise ValueError(f"sigma doit être positif ou nul, reçu {sigma}.")
    return float(sigma)


def to_unit_range(
    image: np.ndarray, data_range: tuple[float, float] | None = None
) -> tuple[np.ndarray, float, float]:
    """
    Ramène linéairement une image vers [0, 1].

    Returns
    -------
    (np.ndarray, float, float)
        Image normalisée, décalage ``lo`` et échelle ``scale`` tels que
        ``image = normalisée · scale + lo``.
    """
    if data_range is None:
        lo, hi = float(image.min()), float(image.max())
    else:
        lo, hi = float(data_range[0]), float(data_range[1])
    scale = hi - lo
    if scale <= 0:
        scale = 1.0
    return (image - lo) / scale, lo, scale


def from_unit_range(image: np.ndarray, lo: float, scale: float) -> np.ndarray:
    """Inverse de :func:`to_unit_range`."""
    return image * scale + lo


def apply_per_channel(
    func: Callable[..., np.ndarray], image: np.ndarray, channel_axis: int, **kwargs
) -> np.ndarray:
    """
    Applique ``func(bande_2D, **kwargs)`` à chaque bande d'une image 3D.

    ``sigma`` peut être un scalaire (même bruit pour toutes les bandes) ou
    une séquence d'un σ par bande.
    """
    arr = np.moveaxis(np.asarray(image), channel_axis, -1)
    if arr.ndim != 3:
        raise ValueError(f"channel_axis donné : image 3D attendue, reçu ndim={np.ndim(image)}.")
    sigma = kwargs.pop("sigma", None)
    sigmas = list(sigma) if np.ndim(sigma) == 1 else [sigma] * arr.shape[-1]
    if len(sigmas) != arr.shape[-1]:
        raise ValueError(f"{len(sigmas)} valeurs de sigma pour {arr.shape[-1]} bandes.")
    bands = [func(arr[..., c], sigma=sigmas[c], **kwargs) for c in range(arr.shape[-1])]
    return np.moveaxis(np.stack(bands, axis=-1), -1, channel_axis)
