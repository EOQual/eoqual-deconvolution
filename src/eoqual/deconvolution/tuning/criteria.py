#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Choix du paramètre de régularisation des méthodes linéaires (Tikhonov,
Wiener à rapport constant) sans image de référence.

Filtre ``X = H* Y / (|H|² + λ|L|²)``, matrice d'influence diagonale en
Fourier ``S_λ(ω) = |H|² / (|H|² + λ|L|²)``, résidu
``‖h ∗ x_λ − y‖² = (1/N) Σ |(1 − S_λ) Y|²``.

- **GCV** (Golub, Heath & Wahba 1979) : minimise
  ``GCV(λ) = (1/N)‖(I − S_λ) y‖² / (1 − tr(S_λ)/N)²``, estimation de
  l'erreur de prédiction sans connaître σ. Tend à sous-régulariser sur
  une image naturelle, dont le spectre est loin d'être plat ;
- **discrépance** (Morozov 1966) : ``λ`` tel que le résidu égale
  l'énergie attendue du bruit, ``N σ²``. Nécessite σ ; tend à
  sur-régulariser légèrement ;
- **L-curve** (Hansen 1992) : coin de courbure maximale de
  ``(log ‖résidu‖, log ‖L x_λ‖)``. Moins stable que les deux autres sur
  image : fourni pour diagnostic (:func:`l_curve`).
"""
from __future__ import annotations

import numpy as np
from scipy import fft as spfft
from scipy.optimize import minimize_scalar

__all__ = ["gcv_curve", "discrepancy_curve", "l_curve", "select_lambda", "residual_energy", "DEFAULT_LAMBDAS"]

#: Grille de recherche par défaut de λ (balayage logarithmique).
DEFAULT_LAMBDAS = np.logspace(-10, 3, 66)


def residual_energy(Y: np.ndarray, S: np.ndarray) -> float:
    """``‖(I − S) y‖²`` dans le domaine image (Parseval, FFT non normalisée)."""
    return float(np.sum(np.abs((1.0 - S) * Y) ** 2) / Y.size)


def _gcv(lam, Y, abs2, L2):
    S = abs2 / np.maximum(abs2 + lam * L2, 1e-300)
    n = Y.size
    res = residual_energy(Y, S)
    return res / max((1.0 - S.sum() / n) ** 2, 1e-300)


def gcv_values(Y: np.ndarray, abs2: np.ndarray, L2: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """``GCV(λ)`` pour chaque ``λ`` de ``lams`` (spectres sur la grille FFT)."""
    return np.array([_gcv(lam, Y, abs2, L2) for lam in lams])


def select_lambda(
    Y: np.ndarray,
    abs2: np.ndarray,
    L2: np.ndarray,
    *,
    rule: str = "gcv",
    sigma: float | None = None,
    lams: np.ndarray = DEFAULT_LAMBDAS,
) -> float:
    """
    ``λ`` choisi par ``rule`` (``"gcv"`` ou ``"discrepancy"``) à partir des
    spectres ``Y = FFT(y)``, ``|H|²`` et ``|L|²``.

    GCV : balayage de ``lams`` puis affinage par recherche bornée autour
    du minimum. Discrépance : bisection en ``log λ`` (le résidu croît
    avec λ) ; si ``N σ²`` est hors d'atteinte, la borne de ``lams`` la plus
    proche est renvoyée.
    """
    lams = np.asarray(lams, dtype=np.float64)
    loglams = np.log10(lams)
    if rule == "gcv":
        vals = gcv_values(Y, abs2, L2, lams)
        i = int(np.argmin(vals))
        lo, hi = loglams[max(i - 1, 0)], loglams[min(i + 1, len(lams) - 1)]
        if hi <= lo:
            return float(lams[i])
        res = minimize_scalar(lambda t: _gcv(10.0**t, Y, abs2, L2), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-3})
        return float(10.0 ** res.x) if res.fun <= vals[i] else float(lams[i])
    if rule == "discrepancy":
        if sigma is None:
            raise ValueError("rule='discrepancy' nécessite sigma.")
        target = Y.size * sigma**2

        def resid(t):
            S = abs2 / np.maximum(abs2 + 10.0**t * L2, 1e-300)
            return residual_energy(Y, S)

        lo, hi = loglams[0], loglams[-1]
        if resid(lo) >= target:
            return float(lams[0])
        if resid(hi) <= target:
            return float(lams[-1])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if resid(mid) < target:
                lo = mid
            else:
                hi = mid
        return float(10.0 ** (0.5 * (lo + hi)))
    raise ValueError(f"rule doit valoir 'gcv' ou 'discrepancy', reçu {rule!r}.")


def _spectra(image, psf, regularizer, boundary):
    from ..backends._common import regularizer_power, setup

    pb = setup(image, psf, boundary)
    return spfft.fft2(pb.y), pb.op.abs2, regularizer_power(regularizer, pb.y.shape)


def gcv_curve(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    regularizer: str = "laplacian",
    lams: np.ndarray = DEFAULT_LAMBDAS,
    boundary: str = "reflect",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Courbe ``GCV(λ)`` de Tikhonov sur ``image``.

    Returns
    -------
    (lams, gcv, lam_best)
    """
    Y, abs2, L2 = _spectra(image, psf, regularizer, boundary)
    return lams, gcv_values(Y, abs2, L2, lams), select_lambda(Y, abs2, L2, rule="gcv", lams=lams)


def discrepancy_curve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float,
    *,
    regularizer: str = "laplacian",
    lams: np.ndarray = DEFAULT_LAMBDAS,
    boundary: str = "reflect",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Résidu ``‖h ∗ x_λ − y‖² / N`` (par pixel) en fonction de ``λ``, à
    comparer à ``σ²``.

    Returns
    -------
    (lams, residual_per_pixel, sigma², lam_best)
    """
    Y, abs2, L2 = _spectra(image, psf, regularizer, boundary)
    res = np.array([residual_energy(Y, abs2 / np.maximum(abs2 + lam * L2, 1e-300)) for lam in lams]) / Y.size
    lam = select_lambda(Y, abs2, L2, rule="discrepancy", sigma=sigma, lams=lams)
    return lams, res, sigma**2, lam


def l_curve(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    regularizer: str = "laplacian",
    lams: np.ndarray = DEFAULT_LAMBDAS,
    boundary: str = "reflect",
) -> dict:
    """
    L-curve : ``log ‖h ∗ x_λ − y‖`` et ``log ‖L x_λ‖`` en fonction de
    ``λ``, courbure et ``λ`` au coin (courbure maximale).

    Returns
    -------
    dict
        ``lams``, ``log_residual``, ``log_seminorm``, ``curvature``,
        ``lam_corner``.
    """
    Y, abs2, L2 = _spectra(image, psf, regularizer, boundary)
    lr, ls = [], []
    for lam in lams:
        denom = np.maximum(abs2 + lam * L2, 1e-300)
        S = abs2 / denom
        lr.append(0.5 * np.log(residual_energy(Y, S) + 1e-300))
        # |X| = |H| |Y| / denom ; ‖L x‖² = (1/N) Σ |L|² |X|².
        absx2 = abs2 * np.abs(Y) ** 2 / denom**2
        ls.append(0.5 * np.log(np.sum(L2 * absx2) / Y.size + 1e-300))
    lr, ls = np.array(lr), np.array(ls)
    t = np.log(lams)
    d1x, d1y = np.gradient(lr, t), np.gradient(ls, t)
    d2x, d2y = np.gradient(d1x, t), np.gradient(d1y, t)
    kappa = (d1x * d2y - d1y * d2x) / np.maximum(d1x**2 + d1y**2, 1e-300) ** 1.5
    i = int(np.nanargmax(kappa))
    return {"lams": lams, "log_residual": lr, "log_seminorm": ls, "curvature": kappa, "lam_corner": float(lams[i])}
