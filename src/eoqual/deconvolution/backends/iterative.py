#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes itératives sans terme de régularisation explicite (le nombre
d'itérations régularise) : Richardson-Lucy et Landweber projeté.
"""
from __future__ import annotations

import numpy as np

from ._common import Callback, setup
from ..operators import divergence, gradient
from ..utils import resolve_sigma

__all__ = ["richardson_lucy_deconvolve", "landweber_deconvolve"]

_EPS = 1e-12


def richardson_lucy_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    n_iter: int = 12,
    acceleration: bool = True,
    tv_weight: float = 0.0,
    boundary: str = "reflect",
    callback: Callback = None,
) -> np.ndarray:
    """
    Richardson-Lucy (Richardson 1972, Lucy 1974) : maximum de
    vraisemblance pour un bruit de Poisson,
    ``x ← x · Hᵀ(y / Hx)`` (PSF de somme 1).

    L'image doit être positive (comptes) : si elle contient des valeurs
    négatives (fond soustrait, bruit de lecture), elle est décalée de son
    minimum avant traitement puis recalée. Le résultat reste positif.

    Parameters
    ----------
    sigma : float, optional
        Ignoré (le nombre d'itérations régularise) ; présent pour la
        signature commune.
    n_iter : int, default 12
        Nombre d'itérations : trop peu, l'image reste floue ; trop,
        le bruit est amplifié (semi-convergence). 12 itérations accélérées
        sont un bon compromis sur les flous du benchmark (BENCHMARK.md) ;
        un flou large en demande plus.
    acceleration : bool, default True
        Extrapolation vectorielle de Biggs & Andrews (1997), ordre 1 :
        résultat comparable en 3 à 5 fois moins d'itérations.
    tv_weight : float, default 0
        Régularisation par variation totale (Dey et al. 2006) :
        ``x ← x · Hᵀ(y / Hx) / (1 − λ div(∇x / |∇x|))``. Sans dimension ;
        valeurs utiles de l'ordre de 1e-3 à 1e-2.
    boundary : str, default "reflect"
    callback : callable, optional
        ``callback(k, x)`` après chaque itération.
    """
    if n_iter < 1:
        raise ValueError("n_iter doit être ≥ 1.")
    pb = setup(image, psf, boundary)
    y = pb.y
    offset = min(float(y.min()), 0.0)
    y = y - offset + _EPS
    op = pb.op

    x = np.full_like(y, y.mean())
    x_prev = x.copy()
    g_prev = g_prev2 = None
    for k in range(n_iter):
        if acceleration and g_prev is not None and g_prev2 is not None:
            alpha = float(np.sum(g_prev * g_prev2) / max(np.sum(g_prev2 * g_prev2), _EPS))
            alpha = min(max(alpha, 0.0), 1.0)
            z = np.clip(x + alpha * (x - x_prev), _EPS, None)
        else:
            z = x
        update = z * op.adjoint(y / np.maximum(op.forward(z), _EPS))
        if tv_weight > 0:
            gz = gradient(z)
            norm = np.sqrt(np.sum(gz**2, axis=0)) + 1e-8 * max(float(z.max()), _EPS)
            denom = 1.0 - tv_weight * divergence(gz / norm)
            update = update / np.maximum(denom, 0.1)
        update = np.clip(update, _EPS, None)
        g_new = update - z
        x_prev, x = x, update
        g_prev2, g_prev = g_prev, g_new
        if callback is not None:
            callback(k, x + offset)
    return pb.domain.restore(x + offset)


def landweber_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    n_iter: int = 500,
    step: float | None = None,
    positivity: bool = True,
    stop_at_discrepancy: bool = True,
    boundary: str = "reflect",
    callback: Callback = None,
) -> np.ndarray:
    """
    Landweber projeté (Bertero & Boccacci 1998) : descente de gradient
    sur ``½‖Hx − y‖²``, ``x ← P(x − τ Hᵀ(Hx − y))``, ``P`` projection sur
    les valeurs positives (si ``positivity``).

    Parameters
    ----------
    sigma : float, optional
        Écart-type du bruit, utilisé par ``stop_at_discrepancy`` (estimé
        si ``None``).
    n_iter : int, default 500
        Nombre maximal d'itérations (régularise, comme pour Richardson-Lucy).
    step : float, optional
        Pas ``τ`` ; défaut ``1.9 / max|H|²`` (convergence si ``τ < 2/max|H|²``).
    positivity : bool, default True
        Projection sur ``x ≥ 0``. À désactiver pour une image de moyenne
        nulle ou à fond soustrait.
    stop_at_discrepancy : bool, default True
        Arrêt dès que ``‖Hx − y‖² ≤ N σ²`` (principe de discrépance,
        évalué sur le domaine de travail) : le nombre d'itérations s'adapte
        au flou et au bruit. ``False`` : exactement ``n_iter`` itérations.
    boundary : str, default "reflect"
    callback : callable, optional
    """
    if n_iter < 1:
        raise ValueError("n_iter doit être ≥ 1.")
    pb = setup(image, psf, boundary)
    y, op = pb.y, pb.op
    tau = 1.9 / op.lipschitz if step is None else float(step)
    target = y.size * resolve_sigma(image, sigma) ** 2 if stop_at_discrepancy else None
    x = y.copy()
    for k in range(n_iter):
        r = op.forward(x) - y
        if target is not None and np.sum(r**2) <= target:
            break
        x = x - tau * op.adjoint(r)
        if positivity:
            np.clip(x, 0.0, None, out=x)
        if callback is not None:
            callback(k, x)
    return pb.domain.restore(x)
