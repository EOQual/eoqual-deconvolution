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
- **vraisemblance marginale** (« evidence », Wiener–Hunt bayésien) :
  avec un a priori gaussien ``x ~ N(0, (γ_x LᵀL)⁻¹)`` et un bruit blanc
  ``N(0, γ_b⁻¹)``, l'image observée est gaussienne, diagonale en
  Fourier : ``E|Y(ω)|²/N = v(ω) = σ² (1 + |H|²/(λ |L|²))`` avec
  ``σ² = 1/γ_b`` et ``λ = γ_x/γ_b``. Maximiser ``p(y | λ, σ²)`` (la scène
  intégrée) choisit λ **et** σ sans référence (MacKay 1992 ; Molina 1994 ;
  Orieux et al. 2010 pour l'échantillonnage) — voir
  :func:`wiener_hunt_evidence` ;
- **L-curve** (Hansen 1992) : coin de courbure maximale de
  ``(log ‖résidu‖, log ‖L x_λ‖)``. Moins stable que les deux autres sur
  image : fourni pour diagnostic (:func:`l_curve`).
"""
from __future__ import annotations

import numpy as np
from scipy import fft as spfft
from scipy.optimize import minimize_scalar

__all__ = [
    "gcv_curve", "discrepancy_curve", "l_curve", "select_lambda", "residual_energy", "DEFAULT_LAMBDAS",
    "wiener_hunt_evidence", "wiener_hunt_estimate", "evidence_curve",
]

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


def _evidence_terms(Y, abs2, L2):
    keep = L2 > 0
    keep[0, 0] = False                                   # moyenne : a priori impropre
    return np.abs(Y[keep]) ** 2 / Y.size, abs2[keep], L2[keep]


def wiener_hunt_evidence(
    Y: np.ndarray, abs2: np.ndarray, L2: np.ndarray, lam: float, sigma: float | None = None
) -> tuple[float, float]:
    """
    Opposé du logarithme de la vraisemblance marginale ``p(y | λ, σ²)`` du
    modèle de Wiener–Hunt (constantes omises) et σ² correspondant.

    Modèle : ``Y(ω)`` gaussien complexe centré de variance ``N v(ω)``,
    ``v = σ² s``, ``s(ω) = 1 + |H(ω)|² / (λ |L(ω)|²)``, fréquences où
    ``|L| > 0`` hors fréquence nulle. ``−log p = Σ [log v + P/v]`` avec
    ``P = |Y|²/N``. Si ``sigma`` est ``None``, σ² est profilé
    analytiquement (``σ̂² = moyenne(P/s)``) et le critère devient
    ``Σ log s + M log σ̂²`` (``M`` fréquences).

    Returns
    -------
    (critère, σ²)
    """
    P, a2, l2 = _evidence_terms(Y, abs2, L2)
    s = 1.0 + a2 / (max(lam, 1e-300) * l2)
    if sigma is None:
        s2 = float(np.mean(P / s))
        return float(np.sum(np.log(s)) + P.size * np.log(max(s2, 1e-300))), s2
    s2 = float(sigma) ** 2
    v = s2 * s
    return float(np.sum(np.log(v) + P / v)), s2


def wiener_hunt_estimate(
    Y: np.ndarray,
    abs2: np.ndarray,
    L2: np.ndarray,
    sigma: float | None = None,
    lams: np.ndarray = DEFAULT_LAMBDAS,
) -> tuple[float, float, float]:
    """
    ``λ`` (et σ si ``sigma=None``) maximisant la vraisemblance marginale de
    Wiener–Hunt : balayage de ``lams`` puis recherche bornée autour du
    minimum.

    Returns
    -------
    (λ, σ, critère)
    """
    lams = np.asarray(lams, dtype=np.float64)
    terms = _evidence_terms(Y, abs2, L2)

    def crit(t):
        P, a2, l2 = terms
        s = 1.0 + a2 / (10.0**t * l2)
        if sigma is None:
            return float(np.sum(np.log(s)) + P.size * np.log(max(np.mean(P / s), 1e-300)))
        v = sigma**2 * s
        return float(np.sum(np.log(v) + P / v))

    loglams = np.log10(lams)
    vals = np.array([crit(t) for t in loglams])
    i = int(np.argmin(vals))
    lo, hi = loglams[max(i - 1, 0)], loglams[min(i + 1, len(lams) - 1)]
    t = loglams[i]
    if hi > lo:
        res = minimize_scalar(crit, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
        if res.fun <= vals[i]:
            t = float(res.x)
    lam = float(10.0**t)
    j, s2 = wiener_hunt_evidence(Y, abs2, L2, lam, sigma)
    return lam, float(np.sqrt(s2)), j


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
    ``λ`` choisi par ``rule`` (``"gcv"``, ``"discrepancy"`` ou
    ``"evidence"``) à partir des spectres ``Y = FFT(y)``, ``|H|²`` et
    ``|L|²``.

    GCV : balayage de ``lams`` puis affinage par recherche bornée autour
    du minimum. Discrépance : bisection en ``log λ`` (le résidu croît
    avec λ) ; si ``N σ²`` est hors d'atteinte, la borne de ``lams`` la plus
    proche est renvoyée.
    """
    lams = np.asarray(lams, dtype=np.float64)
    loglams = np.log10(lams)
    if rule == "evidence":
        return wiener_hunt_estimate(Y, abs2, L2, sigma, lams)[0]
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
    raise ValueError(f"rule doit valoir 'gcv', 'discrepancy' ou 'evidence', reçu {rule!r}.")


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


def evidence_curve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    regularizer: str = "laplacian",
    lams: np.ndarray = DEFAULT_LAMBDAS,
    boundary: str = "reflect",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Critère de vraisemblance marginale de Wiener–Hunt en fonction de ``λ``.

    Returns
    -------
    (lams, critère, lam_best, sigma_best)
    """
    Y, abs2, L2 = _spectra(image, psf, regularizer, boundary)
    vals = np.array([wiener_hunt_evidence(Y, abs2, L2, lam, sigma)[0] for lam in lams])
    lam, s, _ = wiener_hunt_estimate(Y, abs2, L2, sigma, lams)
    return lams, vals, lam, s


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
