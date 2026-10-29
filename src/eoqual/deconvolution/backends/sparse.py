#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes variationnelles à a priori parcimonieux :

- :func:`fista_wavelet_deconvolve` : ``min ½‖Hx − y‖² + λ‖Wx‖₁``,
  ``W`` ondelettes orthogonales (coefficients de détail seulement),
  FISTA (Beck & Teboulle 2009) ;
- :func:`tv_primal_dual_deconvolve` : ``min ½‖Hx − y‖² + λ TV(x)
  [+ ι(x ≥ 0)]``, TV isotrope, algorithme primal-dual de Condat (2013) et
  Vũ (2013).

Les deux travaillent sur l'image ramenée à [0, 1] (``data_range``) : le
paramètre ``lam`` est ainsi exprimé en **multiples de σ**, sans
dépendre des unités de l'image.
"""
from __future__ import annotations

import numpy as np

from ._common import Callback, setup
from ..operators import divergence, gradient
from ..utils import from_unit_range, resolve_sigma, to_unit_range

__all__ = ["fista_wavelet_deconvolve", "tv_primal_dual_deconvolve", "tv_objective", "prox_tv_dual_projection"]


def _soft(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _wavelet_prox(x: np.ndarray, t: float, wavelet: str, levels: int, shift: tuple[int, int]) -> np.ndarray:
    """Seuillage doux des détails d'ondelettes orthogonales (périodisation), après décalage circulaire."""
    import pywt

    xs = np.roll(x, shift, axis=(0, 1))
    coeffs = pywt.wavedec2(xs, wavelet, mode="periodization", level=levels)
    coeffs = [coeffs[0]] + [tuple(_soft(c, t) for c in detail) for detail in coeffs[1:]]
    rec = pywt.waverec2(coeffs, wavelet, mode="periodization")[: x.shape[0], : x.shape[1]]
    return np.roll(rec, (-shift[0], -shift[1]), axis=(0, 1))


def fista_wavelet_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    lam: float = 0.2,
    wavelet: str = "db4",
    levels: int = 3,
    n_iter: int = 100,
    cycle_spin: bool = True,
    data_range: tuple[float, float] | None = None,
    seed: int = 0,
    boundary: str = "reflect",
    callback: Callback = None,
) -> np.ndarray:
    """
    FISTA sur ondelettes orthogonales.

    Parameters
    ----------
    sigma : float, optional
        Écart-type du bruit (unités de l'image) ; ``None`` : estimé.
    lam : float, default 0.2
        Poids de la parcimonie, **en multiples de σ** : seuil effectif
        ``lam · σ / max|H|²`` par itération.
    wavelet : str, default "db4"
        Ondelette orthogonale PyWavelets.
    levels : int, default 3
        Niveaux de décomposition.
    n_iter : int, default 100
    cycle_spin : bool, default True
        Décalage circulaire aléatoire de la grille d'ondelettes à chaque
        itération (Figueiredo & Nowak 2003) : supprime les artefacts en
        blocs d'une base décimée. Le problème résolu devient l'invariant
        par translation (au sens de la moyenne des décalages) ; la
        décroissance monotone de l'objectif n'est plus garantie.
    data_range : tuple, optional
        ``(min, max)`` de la dynamique ; défaut : celle de l'image.
    seed : int, default 0
        Graine des décalages.
    boundary : str, default "reflect"
    callback : callable, optional
    """
    if n_iter < 1:
        raise ValueError("n_iter doit être ≥ 1.")
    s = resolve_sigma(image, sigma)
    pb = setup(image, psf, boundary, multiple=2**levels)
    y, lo, scale = to_unit_range(pb.y, data_range)
    t = lam * (s / scale) / pb.op.lipschitz
    step = 1.0 / pb.op.lipschitz
    rng = np.random.default_rng(seed)
    block = 2**levels

    x = y.copy()
    z = x.copy()
    tk = 1.0
    for k in range(n_iter):
        shift = tuple(rng.integers(0, block, 2)) if cycle_spin else (0, 0)
        x_new = _wavelet_prox(z - step * pb.op.gradient_data(z, y), t, wavelet, levels, shift)
        tk_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        z = x_new + ((tk - 1.0) / tk_new) * (x_new - x)
        x, tk = x_new, tk_new
        if callback is not None:
            callback(k, from_unit_range(x, lo, scale))
    return pb.domain.restore(from_unit_range(x, lo, scale))


def prox_tv_dual_projection(p: np.ndarray, lam: float) -> np.ndarray:
    """Projection de ``p`` (forme ``(2, H, W)``) sur ``{‖p_ij‖₂ ≤ λ}`` (TV isotrope)."""
    norm = np.sqrt(np.sum(p**2, axis=0))
    return p / np.maximum(1.0, norm / max(lam, 1e-300))


def tv_objective(x: np.ndarray, y: np.ndarray, op, lam: float) -> float:
    """``½‖Hx − y‖² + λ TV(x)`` (TV isotrope, différences avant périodiques)."""
    r = op.forward(x) - y
    return float(0.5 * np.sum(r**2) + lam * np.sum(np.sqrt(np.sum(gradient(x) ** 2, axis=0))))


def tv_primal_dual_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    lam: float = 0.1,
    n_iter: int = 300,
    positivity: bool = False,
    data_range: tuple[float, float] | None = None,
    tol: float = 1e-5,
    boundary: str = "reflect",
    callback: Callback = None,
) -> np.ndarray:
    """
    Déconvolution par variation totale, algorithme primal-dual de
    Condat-Vũ, avec le terme d'attache aux données traité par son gradient :

    ``x ← P(x − τ(Hᵀ(Hx − y) − div p))`` ;
    ``p ← Proj_{‖·‖≤λ}(p + σ_d ∇(2x_new − x))``,

    avec ``1/τ − 8σ_d ≥ ‖H‖²/2`` (``‖∇‖² ≤ 8``). Ici ``τ = 1/‖H‖²``,
    ``σ_d = ‖H‖²/16``.

    Parameters
    ----------
    sigma : float, optional
        Écart-type du bruit (unités de l'image) ; ``None`` : estimé.
    lam : float, default 0.1
        Poids de la TV, **en multiples de σ** (image ramenée à [0, 1]).
    n_iter : int, default 300
        Nombre maximal d'itérations.
    positivity : bool, default False
        Contrainte ``x ≥ 0`` sur l'image ramenée à [0, 1] (donc
        ``x ≥ min`` de la dynamique).
    data_range : tuple, optional
    tol : float, default 1e-5
        Arrêt quand ``‖x_k+1 − x_k‖ / ‖x_k‖ < tol``.
    boundary : str, default "reflect"
    callback : callable, optional
    """
    if n_iter < 1:
        raise ValueError("n_iter doit être ≥ 1.")
    s = resolve_sigma(image, sigma)
    pb = setup(image, psf, boundary)
    y, lo, scale = to_unit_range(pb.y, data_range)
    weight = lam * s / scale
    lip = pb.op.lipschitz
    tau = 0.99 / lip
    sig = lip / 16.0

    x = y.copy()
    p = np.zeros((2,) + x.shape)
    for k in range(n_iter):
        x_new = x - tau * (pb.op.gradient_data(x, y) - divergence(p))
        if positivity:
            np.clip(x_new, 0.0, None, out=x_new)
        p = prox_tv_dual_projection(p + sig * gradient(2 * x_new - x), weight)
        delta = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-300)
        x = x_new
        if callback is not None:
            callback(k, from_unit_range(x, lo, scale))
        if delta < tol:
            break
    return pb.domain.restore(from_unit_range(x, lo, scale))
