#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Opérateur de flou et traitement des bords.

Toutes les méthodes résolvent le problème sur un domaine **périodique**,
où le flou est un produit dans le domaine de Fourier (``Y = H·X``). Une
image réelle n'est pas périodique : sans précaution, la discontinuité
entre bords opposés produit des oscillations (effet de Gibbs) sur toute
l'image restaurée. Le paramètre ``boundary`` choisit la parade :

- ``"reflect"`` (défaut) : prolongement miroir de l'image sur au moins
  une taille de PSF de chaque côté, déconvolution, puis recadrage. Les
  artefacts restent dans la marge ;
- ``"perdecomp"`` : décomposition périodique + lisse de Moisan
  (:func:`eoqual.core.fourier.perdecomp`) ; seule la composante
  périodique est déconvoluée, la composante lisse (basses fréquences,
  quasi insensible au flou) est rajoutée ensuite. Aucune marge, donc
  moins coûteux, mais approximatif près des bords ;
- ``"periodic"`` : aucun traitement — exact pour une image synthétique
  floutée par convolution circulaire (:func:`eoqual.deconvolution.simulate.blur`).

Référence : L. Moisan, "Periodic plus smooth image decomposition",
JMIV 39(2), 2011.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import fft as spfft

from eoqual.core.fourier import perdecomp
from eoqual.core.kernel import psf2otf

__all__ = ["BOUNDARIES", "BlurOperator", "Domain", "prepare_domain", "gradient", "divergence"]

BOUNDARIES = ("reflect", "perdecomp", "periodic")


class BlurOperator:
    """
    Convolution circulaire par une PSF sur une grille ``shape``.

    Attributes
    ----------
    otf : np.ndarray
        Fonction de transfert (complexe, ordre ``np.fft.fftfreq``).
    lipschitz : float
        ``max |H|²`` : constante de Lipschitz du gradient de
        ``½‖Hx − y‖²`` (vaut 1 pour une PSF positive de somme 1).
    """

    def __init__(self, psf: np.ndarray, shape: tuple[int, int]):
        self.shape = tuple(shape)
        self.otf = psf2otf(psf, self.shape)
        self.abs2 = np.abs(self.otf) ** 2
        self.lipschitz = float(self.abs2.max())

    def forward(self, x: np.ndarray) -> np.ndarray:
        """``H x``."""
        return spfft.ifft2(spfft.fft2(x) * self.otf).real

    def adjoint(self, x: np.ndarray) -> np.ndarray:
        """``Hᵀ x`` (corrélation par la PSF)."""
        return spfft.ifft2(spfft.fft2(x) * np.conj(self.otf)).real

    def gradient_data(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient de ``½‖Hx − y‖²`` : ``Hᵀ(Hx − y)``."""
        return self.adjoint(self.forward(x) - y)

    def solve_quadratic(self, rhs_fft: np.ndarray, denom: np.ndarray) -> np.ndarray:
        """``ifft(rhs_fft / denom)`` (partie réelle) — systèmes diagonaux en Fourier."""
        return spfft.ifft2(rhs_fft / denom).real


@dataclass
class Domain:
    """
    Image préparée pour la déconvolution sur domaine périodique.

    Attributes
    ----------
    image : np.ndarray
        Image de travail (prolongée, ou composante périodique).
    restore : callable
        Ramène une image de travail restaurée à la géométrie d'origine
        (recadrage, ajout de la composante lisse).
    """

    image: np.ndarray
    restore: Callable[[np.ndarray], np.ndarray]


def _fast_len(n: int, multiple: int) -> int:
    m = spfft.next_fast_len(n, real=True)
    while m % multiple:
        m = spfft.next_fast_len(m + 1, real=True)
    return m


def prepare_domain(
    image: np.ndarray,
    psf_shape: tuple[int, int],
    boundary: str = "reflect",
    multiple: int = 1,
) -> Domain:
    """
    Prépare ``image`` selon ``boundary`` (voir l'en-tête du module).

    Parameters
    ----------
    image : np.ndarray
        Image 2D ``float64``.
    psf_shape : tuple
        Forme de la PSF (fixe la marge de ``"reflect"``).
    boundary : str
        ``"reflect"``, ``"perdecomp"`` ou ``"periodic"``.
    multiple : int
        Avec ``"reflect"``, les dimensions de travail sont des multiples
        de ``multiple`` (transformées en ondelettes à plusieurs niveaux).
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary doit valoir {BOUNDARIES}, reçu {boundary!r}.")
    ny, nx = image.shape
    if boundary == "periodic":
        return Domain(image, lambda x: x)
    if boundary == "perdecomp":
        p, s = perdecomp(image)
        return Domain(np.asarray(p, dtype=np.float64), lambda x: x + s)
    # reflect : marge ≥ 2 tailles de PSF au total, puis taille rapide pour la FFT.
    ty = _fast_len(ny + 2 * psf_shape[0], multiple)
    tx = _fast_len(nx + 2 * psf_shape[1], multiple)
    padded = _periodic_mirror_extension(_periodic_mirror_extension(image, ty, 0), tx, 1)
    return Domain(padded, lambda x: x[:ny, :nx])


def _periodic_mirror_extension(x: np.ndarray, total: int, axis: int) -> np.ndarray:
    """
    Prolonge ``x`` jusqu'à ``total`` selon ``axis``, de façon **continue et
    périodique** : la marge (placée après l'image) part du miroir du bord
    de fin et rejoint le miroir du bord de début, avec une pondération en
    cosinus surélevé. L'image occupe les indices ``[0, n)``.

    Un simple miroir de chaque côté laisserait une discontinuité au
    raccord périodique des deux marges : son énergie haute fréquence n'est
    pas expliquée par le flou, ce qui fausse les critères de réglage
    (GCV, discrépance) et crée des oscillations dans la marge.
    """
    x = np.moveaxis(x, axis, 0)
    n = x.shape[0]
    gap = total - n
    if gap <= 0:
        return np.moveaxis(x, 0, axis)
    after = np.pad(x, [(0, gap)] + [(0, 0)] * (x.ndim - 1), mode="symmetric")[n:]
    before = np.pad(x, [(gap, 0)] + [(0, 0)] * (x.ndim - 1), mode="symmetric")[:gap]
    t = (np.arange(gap) + 0.5) / gap
    w = (0.5 - 0.5 * np.cos(np.pi * t)).reshape((-1,) + (1,) * (x.ndim - 1))
    margin = (1.0 - w) * after + w * before
    return np.moveaxis(np.concatenate([x, margin], axis=0), 0, axis)


def gradient(x: np.ndarray) -> np.ndarray:
    """Gradient discret périodique (différences avant), forme ``(2, H, W)``."""
    return np.stack([np.roll(x, -1, axis=0) - x, np.roll(x, -1, axis=1) - x])


def divergence(p: np.ndarray) -> np.ndarray:
    """Divergence discrète, adjointe négative de :func:`gradient` (``⟨∇x, p⟩ = −⟨x, div p⟩``)."""
    return (p[0] - np.roll(p[0], 1, axis=0)) + (p[1] - np.roll(p[1], 1, axis=1))
