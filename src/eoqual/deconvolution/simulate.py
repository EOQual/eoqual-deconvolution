#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observations synthétiques : flou par une PSF connue, puis bruit
gaussien ou poisson-gaussien. Sert aux tests, aux exemples et au
benchmark (vérité terrain connue).
"""
from __future__ import annotations

import numpy as np

from eoqual.core.kernel import convolve_circular

from .utils import check_image, check_psf

__all__ = ["blur", "add_noise", "simulate_observation"]


def blur(image: np.ndarray, psf: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    """
    Convolue ``image`` par ``psf`` (somme ramenée à 1).

    Parameters
    ----------
    boundary : {"periodic", "reflect"}
        ``"periodic"`` : convolution circulaire (exacte pour une
        déconvolution avec ``boundary="periodic"``). ``"reflect"`` :
        prolongement miroir, comme une scène qui se poursuit hors champ —
        plus réaliste, à déconvoluer avec ``boundary="reflect"``.
    """
    x = check_image(image)
    h = check_psf(psf, x.shape)
    if boundary == "periodic":
        return convolve_circular(x, h)
    if boundary == "reflect":
        # Prolongement miroir (« reflect » de scipy = « symmetric » de numpy),
        # puis convolution circulaire : même centrage que psf2otf, y compris
        # pour une PSF de taille paire ; la marge absorbe le repliement.
        my, mx = h.shape
        padded = np.pad(x, ((my, my), (mx, mx)), mode="symmetric")
        return convolve_circular(padded, h)[my:-my, mx:-mx]
    raise ValueError(f"boundary doit valoir 'periodic' ou 'reflect', reçu {boundary!r}.")


def add_noise(
    image: np.ndarray,
    sigma: float = 0.0,
    *,
    gain: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Ajoute un bruit gaussien d'écart-type ``sigma`` et, si ``gain`` est
    donné, un bruit photonique de Poisson.

    Parameters
    ----------
    sigma : float
        Écart-type du bruit gaussien additif (lecture), unités de l'image.
    gain : float, optional
        Nombre de photo-électrons par unité d'image : le signal ``x``
        devient ``Poisson(gain·x) / gain`` (variance ``x / gain``).
    seed : int or Generator, optional
        Graine du tirage.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(image, dtype=np.float64)
    if gain is not None:
        if gain <= 0:
            raise ValueError("gain doit être strictement positif.")
        x = rng.poisson(np.clip(x, 0, None) * gain) / gain
    if sigma > 0:
        x = x + rng.normal(0.0, sigma, x.shape)
    return x


def simulate_observation(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float = 0.0,
    *,
    gain: float | None = None,
    boundary: str = "periodic",
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Flou (:func:`blur`) puis bruit (:func:`add_noise`)."""
    return add_noise(blur(image, psf, boundary), sigma, gain=gain, seed=seed)
