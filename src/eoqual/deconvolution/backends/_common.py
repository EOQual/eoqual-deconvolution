#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Préparation commune : contrôles, domaine de travail, opérateur de flou."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..operators import BlurOperator, Domain, prepare_domain
from ..utils import check_image, check_psf

#: Rappel optionnel des méthodes itératives : ``callback(iteration, x)``,
#: ``x`` étant l'estimée courante **sur le domaine de travail** (prolongé).
Callback = Optional[Callable[[int, np.ndarray], None]]


@dataclass
class Problem:
    """Image de travail, PSF normalisée, opérateur et domaine."""

    y: np.ndarray
    psf: np.ndarray
    op: BlurOperator
    domain: Domain


def setup(image: np.ndarray, psf: np.ndarray, boundary: str, multiple: int = 1) -> Problem:
    """Contrôle ``image``/``psf``, prépare le domaine et l'opérateur de flou."""
    x = check_image(image)
    h = check_psf(psf, x.shape)
    domain = prepare_domain(x, h.shape, boundary, multiple)
    return Problem(domain.image, h, BlurOperator(h, domain.image.shape), domain)


def frequency_grid(shape: tuple[int, int]) -> np.ndarray:
    """``|ω|²`` en cycles/pixel², ordre ``np.fft.fftfreq``."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return fy**2 + fx**2


def regularizer_power(name: str, shape: tuple[int, int]) -> np.ndarray:
    """
    ``|L(ω)|²`` d'un opérateur de régularisation de Tikhonov, sur la
    grille FFT :

    - ``"identity"`` : ``L = I`` → 1 ;
    - ``"gradient"`` : différences avant → ``4 sin²(πf_y) + 4 sin²(πf_x)`` ;
    - ``"laplacian"`` : laplacien à 5 points → carré du précédent.
    """
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    grad2 = 4 * np.sin(np.pi * fy) ** 2 + 4 * np.sin(np.pi * fx) ** 2
    if name == "identity":
        return np.ones(shape)
    if name == "gradient":
        return grad2
    if name == "laplacian":
        return grad2**2
    raise ValueError(f"regularizer doit valoir identity/gradient/laplacian, reçu {name!r}.")
