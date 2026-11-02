#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Préparation commune : contrôles, domaine de travail, opérateur de flou,
pixels connus, inconnus et saturés.

**Données incomplètes.** Un pixel de l'image de travail est :

- *connu* : mesure valide, attache aux données ``(h ∗ x − y)²`` ;
- *inconnu* : pixel masqué (``mask``, no-data, NaN) ou marge ajoutée avec
  ``boundary="unknown"`` — aucune attache aux données ;
- *saturé* (``saturation``) : la mesure vaut le seuil ``L`` et la vraie
  valeur floue est seulement connue pour être ``≥ L`` ; l'attache aux
  données devient unilatérale, ``min(h ∗ x − L, 0)²``.

Toutes les méthodes traitent ces trois cas par la même opération,
:meth:`Problem.fill` : les données sont complétées par l'estimée courante
(``h ∗ x`` aux pixels inconnus, ``max(h ∗ x, L)`` aux pixels saturés).
Pour une méthode à gradient, ``Hᵀ(H x − fill(H x))`` est exactement le
gradient de l'attache masquée ; pour une méthode linéaire, le point fixe de
« compléter puis résoudre » est la solution exacte du problème masqué
(algorithme EM pour données manquantes, Dempster, Laird & Rubin 1977) ;
pour Richardson-Lucy, c'est l'EM du modèle de Poisson à données
manquantes. Références : Almeida & Figueiredo (2013) pour les bords
inconnus ; Bertero & Boccacci (2005) pour les données masquées.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

from ..operators import BlurOperator, Domain, prepare_domain
from ..utils import check_psf

#: Rappel optionnel des méthodes itératives : ``callback(iteration, x)``,
#: ``x`` étant l'estimée courante **sur le domaine de travail** (prolongé ;
#: l'image d'origine en occupe le coin ``[:H, :W]``).
Callback = Optional[Callable[[int, np.ndarray], None]]


@dataclass
class Problem:
    """
    Image de travail, PSF normalisée, opérateur, domaine et état des pixels.

    Attributes
    ----------
    y : np.ndarray
        Image de travail (pixels inconnus pré-remplis par le plus proche
        pixel connu, marge prolongée).
    known : np.ndarray or None
        Pixels connus et non saturés (``None`` : tous).
    saturated : np.ndarray or None
        Pixels saturés (``None`` : aucun).
    level : float or None
        Seuil de saturation, unités de l'image d'origine.
    """

    y: np.ndarray
    psf: np.ndarray
    op: BlurOperator
    domain: Domain
    known: Optional[np.ndarray] = None
    saturated: Optional[np.ndarray] = None
    level: Optional[float] = None

    @property
    def incomplete(self) -> bool:
        """Vrai si des pixels sont inconnus ou saturés."""
        return self.known is not None

    @property
    def n_known(self) -> int:
        """Nombre de pixels connus (pour le principe de discrépance)."""
        return int(self.y.size if self.known is None else self.known.sum())

    def level_in(self, lo: float, scale: float) -> Optional[float]:
        """Seuil de saturation dans les unités ``(y − lo) / scale``."""
        return None if self.level is None else (self.level - lo) / scale

    def fill(self, Hx: np.ndarray, y: np.ndarray, level: Optional[float] = None) -> np.ndarray:
        """Données complétées : ``y`` (connus), ``max(Hx, L)`` (saturés), ``Hx`` (inconnus)."""
        if self.known is None:
            return y
        out = np.where(self.known, y, Hx)
        if self.saturated is not None and level is not None:
            out = np.where(self.saturated, np.maximum(Hx, level), out)
        return out

    def residual(self, Hx: np.ndarray, y: np.ndarray, level: Optional[float] = None) -> np.ndarray:
        """``Hx − fill(Hx)`` : résidu de l'attache aux données (nul aux pixels inconnus)."""
        return Hx - self.fill(Hx, y, level)

    def gradient_data(self, x: np.ndarray, y: np.ndarray, level: Optional[float] = None) -> np.ndarray:
        """Gradient de l'attache aux données (masquée, unilatérale aux pixels saturés)."""
        return self.op.adjoint(self.residual(self.op.forward(x), y, level))


def _valid_mask(x: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        if not np.all(np.isfinite(x)):
            raise ValueError("L'image contient des valeurs non finies (NaN/inf) : les signaler par mask=.")
        return None
    m = np.asarray(mask, dtype=bool)
    if m.shape != x.shape:
        raise ValueError(f"mask {m.shape} doit avoir la forme de l'image {x.shape}.")
    m = m & np.isfinite(x)
    if not m.any():
        raise ValueError("mask ne contient aucun pixel valide.")
    return m


def _fill_nearest(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Remplit les pixels invalides par la valeur du plus proche pixel valide."""
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return x[tuple(idx)]


def setup(
    image: np.ndarray,
    psf: np.ndarray,
    boundary: str,
    multiple: int = 1,
    *,
    mask: Optional[np.ndarray] = None,
    saturation: Optional[float] = None,
) -> Problem:
    """
    Contrôle ``image``/``psf``, prépare le domaine, l'opérateur de flou et
    l'état des pixels.

    Parameters
    ----------
    mask : np.ndarray of bool, optional
        Pixels valides (``True``). Les pixels ``False`` et les NaN sont
        inconnus.
    saturation : float, optional
        Seuil de saturation (unités de l'image) : les pixels valides
        ``≥ saturation`` sont saturés.
    """
    x = np.array(image, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(
            f"Image 2D attendue, reçu ndim={x.ndim}. Pour une image "
            "multi-bandes, passer channel_axis à deconvolve()."
        )
    valid = _valid_mask(x, mask)
    if valid is not None:
        x = _fill_nearest(np.where(valid, x, 0.0), valid)
    h = check_psf(psf, x.shape)
    sat = None
    if saturation is not None:
        sat = x >= float(saturation)
        if valid is not None:
            sat &= valid
        if not sat.any():
            sat = None
    incomplete = valid is not None or sat is not None or boundary == "unknown"
    if incomplete and boundary == "perdecomp":
        raise ValueError("mask/saturation ne sont pas compatibles avec boundary='perdecomp'.")
    domain = prepare_domain(x, h.shape, boundary, multiple)
    known = sat_w = None
    if incomplete:
        ny, nx = x.shape
        known = np.zeros(domain.image.shape, dtype=bool)
        known[:ny, :nx] = True if valid is None else valid
        if boundary == "reflect":  # marge prolongée considérée comme mesurée
            known[ny:, :] = True
            known[:, nx:] = True
        if sat is not None:
            sat_w = np.zeros(domain.image.shape, dtype=bool)
            sat_w[:ny, :nx] = sat
            known &= ~sat_w
    return Problem(domain.image, h, BlurOperator(h, domain.image.shape), domain,
                   known, sat_w, None if sat_w is None else float(saturation))


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


def solve_with_fill(
    pb: Problem, solve: Callable[[np.ndarray], np.ndarray], n_fill_iter: int
) -> np.ndarray:
    """
    Méthode linéaire ``x = solve(y)`` sur données incomplètes : itère
    « compléter les données par ``H x``, puis résoudre » (EM). Le point fixe
    est la solution du problème à attache masquée.
    """
    x = solve(pb.y)
    if not pb.incomplete:
        return x
    for _ in range(n_fill_iter):
        x = solve(pb.fill(pb.op.forward(x), pb.y, pb.level))
    return x
