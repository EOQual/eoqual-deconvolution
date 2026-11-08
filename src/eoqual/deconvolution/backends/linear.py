#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes linéaires en une passe, résolues dans le domaine de Fourier :
filtre inverse tronqué, Wiener, Wiener–Hunt (paramètres par
vraisemblance marginale), Wiener non supervisé (scikit-image), Tikhonov.

Notations (FFT non normalisée, ``N`` pixels) : ``Y = H·X + B``, avec
``E|B(ω)|² = N σ²``.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as spfft

from ._common import regularizer_power, setup, solve_with_fill
from ..utils import resolve_sigma

__all__ = [
    "inverse_deconvolve",
    "wiener_deconvolve",
    "wiener_unsupervised_deconvolve",
    "wiener_hunt_deconvolve",
    "wiener_hunt_parameters",
    "tikhonov_deconvolve",
    "fit_power_law_signal",
]


def inverse_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    threshold: float = 0.15,
    boundary: str = "reflect",
    mask: np.ndarray | None = None,
    saturation: float | None = None,
    n_fill_iter: int = 30,
) -> np.ndarray:
    """
    Filtre inverse tronqué : ``X = Y / H`` là où ``|H| ≥ threshold``,
    0 ailleurs.

    Méthode de référence, sans modèle de bruit : elle amplifie le bruit
    d'un facteur ``1/threshold`` aux fréquences conservées. ``sigma`` est
    ignoré (présent pour la signature commune).

    Parameters
    ----------
    threshold : float, default 0.15
        Seuil sur ``|H|`` (``H(0) = 1``), dans ]0, 1] ; 0.15 maximise le
        PSNR moyen sur quatre flous et trois niveaux de bruit (BENCHMARK.md).
    boundary : str, default "reflect"
        Traitement des bords (:mod:`eoqual.deconvolution.operators`).
    mask : np.ndarray of bool, optional
        Pixels valides (``True``) ; les autres (et les NaN) sont inconnus.
    saturation : float, optional
        Seuil de saturation (unités de l'image) : attache unilatérale.
    n_fill_iter : int, default 30
        Itérations de remplissage des pixels inconnus ou saturés (et de la
        marge si ``boundary="unknown"``) ; sans effet sur des données
        complètes.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold doit être dans ]0, 1], reçu {threshold}.")
    pb = setup(image, psf, boundary, mask=mask, saturation=saturation)
    H = pb.op.otf
    keep = np.abs(H) >= threshold
    G = np.zeros_like(H)
    G[keep] = 1.0 / H[keep]
    x = solve_with_fill(pb, lambda y: pb.op.solve_quadratic(spfft.fft2(y) * G, 1.0), n_fill_iter)
    return pb.domain.restore(x)


def fit_power_law_signal(
    Y: np.ndarray, abs2: np.ndarray, sigma: float, beta: float | None = None, reliable: float = 0.1
) -> tuple[np.ndarray, float, float]:
    """
    Densité spectrale du signal ``S(ω) = A |ω|^-β``, ajustée sur l'image
    observée.

    Dans la bande où le flou est faible (``|H|² ≥ reliable``, fréquence
    nulle exclue), ``E|Y|² = |H|² S + N σ²`` donne une estimation directe
    ``Ŝ = (|Y|² − Nσ²) / |H|²``. ``log Ŝ`` moyenné par couronne est
    ajusté en ``log A − β log|ω|`` par moindres carrés (``β`` imposé s'il
    est donné). Le modèle prolonge ensuite ``S`` aux hautes fréquences, où
    le flou masque le signal.

    Returns
    -------
    (S, A, β)
    """
    n = Y.size
    rho = np.sqrt(np.fft.fftfreq(Y.shape[0])[:, None] ** 2 + np.fft.fftfreq(Y.shape[1])[None, :] ** 2)
    band = (abs2 >= reliable) & (rho > 0)
    est = (np.abs(Y) ** 2 - n * sigma**2) / np.maximum(abs2, 1e-300)
    edges = np.geomspace(max(rho[band].min(), 1e-6), rho[band].max() * (1 + 1e-9), 25) if band.any() else None
    logr, logs = [], []
    if edges is not None:
        idx = np.digitize(rho[band], edges)
        vals, radii = est[band], rho[band]
        for b in np.unique(idx):
            sel = idx == b
            m = vals[sel].mean()
            if m > 0:
                logr.append(np.log(radii[sel].mean()))
                logs.append(np.log(m))
    logr, logs = np.array(logr), np.array(logs)
    if logr.size >= 3:
        if beta is None:
            slope, intercept = np.polyfit(logr, logs, 1)
            beta_fit = float(np.clip(-slope, 0.5, 4.0))
        else:
            beta_fit = float(beta)
        amp = float(np.exp(np.mean(logs + beta_fit * logr)))
    else:  # bande fiable trop étroite : bilan d'énergie global, β = 2
        beta_fit = 2.0 if beta is None else float(beta)
        shape_s = np.where(rho > 0, np.maximum(rho, 1e-12) ** (-beta_fit), 0.0)
        energy = np.sum(np.abs(Y) ** 2) - np.abs(Y[0, 0]) ** 2 - (n - 1) * n * sigma**2
        amp = float(max(energy, 0.0) / max(np.sum(abs2 * shape_s), 1e-300))
    with np.errstate(divide="ignore"):
        S = amp * np.where(rho > 0, rho, np.inf) ** (-beta_fit)
    return S, amp, beta_fit


def wiener_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    nsr: float | None = None,
    beta: float | None = None,
    boundary: str = "unknown",
    mask: np.ndarray | None = None,
    saturation: float | None = None,
    n_fill_iter: int = 30,
) -> np.ndarray:
    """
    Filtre de Wiener ``X = H* Y / (|H|² + Sb/Sx)``.

    Deux modèles du rapport bruit/signal ``Sb/Sx`` :

    - ``nsr`` donné : constante sur tout le spectre (forme classique,
      équivalente à Tikhonov ``identity``). ``nsr`` est sans dimension ;
    - ``nsr=None`` (défaut) : signal à spectre en loi de puissance
      ``Sx(ω) = A |ω|^-β`` (β ≈ 2 pour une image naturelle, Field 1987),
      ``A`` et ``β`` ajustés sur l'image observée dans la bande où le flou
      est faible (:func:`fit_power_law_signal`) ; le rapport bruit/signal
      croît alors avec la fréquence, ce qui protège les hautes fréquences
      où le signal est faible.

    Parameters
    ----------
    sigma : float, optional
        Écart-type du bruit (unités de l'image). ``None`` : estimé
        (:func:`eoqual.deconvolution.utils.resolve_sigma`). Inutilisé si
        ``nsr`` est donné.
    nsr : float, optional
        Rapport bruit/signal constant.
    beta : float, optional
        Exposant de la loi de puissance du signal ; ``None`` : ajusté.
    boundary : str, default "unknown"
        Marge inconnue (:mod:`eoqual.deconvolution.operators`) : +0.8 à
        +2.2 dB sur un flou à OTF nulle par endroits (bougé), neutre
        ailleurs (BENCHMARK.md).
    mask : np.ndarray of bool, optional
        Pixels valides (``True``) ; les autres (et les NaN) sont inconnus.
    saturation : float, optional
        Seuil de saturation (unités de l'image) : attache unilatérale.
    n_fill_iter : int, default 30
        Itérations de remplissage des pixels inconnus ou saturés (et de la
        marge si ``boundary="unknown"``) ; sans effet sur des données
        complètes.
    """
    pb = setup(image, psf, boundary, mask=mask, saturation=saturation)
    H, abs2 = pb.op.otf, pb.op.abs2
    Y = spfft.fft2(pb.y)
    if nsr is not None:
        if nsr < 0:
            raise ValueError(f"nsr doit être positif ou nul, reçu {nsr}.")
        ratio = np.full(abs2.shape, float(nsr))
    else:
        s = resolve_sigma(pb.domain.restore(pb.y), sigma)
        sx, _, _ = fit_power_law_signal(Y, abs2, s, beta)
        n = Y.size
        with np.errstate(divide="ignore"):
            ratio = np.where(sx > 0, n * s**2 / sx, np.inf)
        ratio[0, 0] = 0.0
    W = np.conj(H) / np.maximum(abs2 + ratio, 1e-300)
    W[~np.isfinite(W)] = 0.0
    x = solve_with_fill(pb, lambda y: pb.op.solve_quadratic(spfft.fft2(y) * W, 1.0), n_fill_iter)
    return pb.domain.restore(x)


def wiener_unsupervised_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    max_num_iter: int = 200,
    seed: int | None = 0,
    boundary: str = "reflect",
    mask: np.ndarray | None = None,
    saturation: float | None = None,
) -> np.ndarray:
    """
    Wiener non supervisé (Orieux, Giovannelli & Rodet 2010), via
    ``skimage.restoration.unsupervised_wiener`` : les niveaux de bruit et
    de régularisation (laplacien) sont estimés conjointement par
    échantillonnage de Gibbs. ``sigma`` est ignoré (estimé par la méthode).

    Parameters
    ----------
    max_num_iter : int, default 200
        Nombre maximal d'itérations de l'échantillonneur.
    mask, saturation :
        Non pris en charge (``ValueError``) ; présents pour la signature
        commune.
    seed : int, optional
        Graine de l'échantillonneur (résultat reproductible).
    """
    from skimage.restoration import unsupervised_wiener

    if mask is not None or saturation is not None or boundary == "unknown":
        raise ValueError(
            "wiener_unsupervised ne gère pas les données incomplètes (mask, saturation, "
            "boundary='unknown') : utiliser wiener, tikhonov ou une méthode itérative."
        )
    pb = setup(image, psf, boundary)
    # skimage centre la PSF en (kh // 2, kw // 2), comme psf2otf.
    est, _ = unsupervised_wiener(
        pb.y, pb.psf, clip=False, user_params={"max_num_iter": max_num_iter}, rng=seed
    )
    return pb.domain.restore(np.asarray(est, dtype=np.float64))


def wiener_hunt_parameters(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    prior: str = "laplacian",
    boundary: str = "unknown",
    mask: np.ndarray | None = None,
    saturation: float | None = None,
) -> dict:
    """
    Paramètres de Wiener–Hunt estimés par maximum de vraisemblance
    marginale (:func:`eoqual.deconvolution.tuning.criteria.wiener_hunt_estimate`).

    Returns
    -------
    dict
        ``lam`` (= γ_x/γ_b), ``sigma`` (écart-type du bruit, estimé si
        ``sigma=None``, sinon recopié), ``criterion`` (−log vraisemblance
        marginale, constantes omises).
    """
    from ..tuning.criteria import wiener_hunt_estimate

    pb = setup(image, psf, boundary, mask=mask, saturation=saturation)
    L2 = regularizer_power(prior, pb.y.shape)
    lam, s, crit = wiener_hunt_estimate(spfft.fft2(pb.y), pb.op.abs2, L2, sigma)
    return {"lam": lam, "sigma": s, "criterion": crit}


def wiener_hunt_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    lam: float | None = None,
    prior: str = "laplacian",
    boundary: str = "unknown",
    mask: np.ndarray | None = None,
    saturation: float | None = None,
    n_fill_iter: int = 30,
) -> np.ndarray:
    """
    Déconvolution de Wiener–Hunt (Hunt 1973) : filtre de Wiener dont le
    spectre de la scène est remplacé par un a priori de régularité,
    ``X = H* Y / (|H|² + λ |D|²)``, ``D`` laplacien (défaut).

    Lecture bayésienne : maximum a posteriori de ``x`` pour un a priori
    gaussien ``x ~ N(0, (γ_x DᵀD)⁻¹)`` et un bruit blanc gaussien de
    précision ``γ_b`` ; ``λ = γ_x / γ_b``. Le filtre est celui de
    :func:`tikhonov_deconvolve` (``regularizer=prior``) ; ce qui distingue
    la méthode est le **choix de λ et de σ par maximum de vraisemblance
    marginale** (la scène est intégrée analytiquement, exact et instantané
    en Fourier), sans image de référence ni connaissance du bruit —
    version déterministe du Wiener–Hunt non supervisé d'Orieux et al.
    (2010), qui échantillonne la loi a posteriori (``wiener_unsupervised``).

    Parameters
    ----------
    sigma : float, optional
        Écart-type du bruit s'il est connu (alors seul λ est estimé) ;
        ``None`` : estimé avec λ.
    lam : float, optional
        λ imposé (aucune estimation).
    prior : {"laplacian", "gradient", "identity"}, default "laplacian"
        Opérateur ``D`` de l'a priori : spectre de la scène supposé en
        ``1/|D(ω)|²``, soit ``|ω|⁻⁴`` (laplacien, choix de Hunt),
        ``|ω|⁻²`` (gradient) ou plat (identité).
    boundary : str, default "unknown"
        Marge inconnue (:mod:`eoqual.deconvolution.operators`).
    mask, saturation, n_fill_iter :
        Données incomplètes (§2.2 de la référence technique) ; λ et σ sont
        alors estimés sur l'image pré-remplie.

    Examples
    --------
    >>> x = deconvolve(y, psf, "wiener_hunt")                 # λ et σ estimés
    >>> wiener_hunt_parameters(y, psf)["sigma"]               # σ estimé
    """
    pb = setup(image, psf, boundary, mask=mask, saturation=saturation)
    L2 = regularizer_power(prior, pb.y.shape)
    if lam is None:
        from ..tuning.criteria import wiener_hunt_estimate

        lam = wiener_hunt_estimate(spfft.fft2(pb.y), pb.op.abs2, L2, sigma)[0]
    if lam < 0:
        raise ValueError(f"lam doit être positif ou nul, reçu {lam}.")
    W = np.conj(pb.op.otf) / np.maximum(pb.op.abs2 + lam * L2, 1e-300)
    x = solve_with_fill(pb, lambda y: pb.op.solve_quadratic(spfft.fft2(y) * W, 1.0), n_fill_iter)
    return pb.domain.restore(x)


def tikhonov_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    lam: float | None = None,
    regularizer: str = "laplacian",
    rule: str = "gcv",
    boundary: str = "unknown",
    mask: np.ndarray | None = None,
    saturation: float | None = None,
    n_fill_iter: int = 30,
) -> np.ndarray:
    """
    Tikhonov : ``min ‖h ∗ x − y‖² + λ ‖L x‖²``, solution
    ``X = H* Y / (|H|² + λ |L|²)``.

    Parameters
    ----------
    sigma : float, optional
        Écart-type du bruit ; utilisé seulement par ``rule="discrepancy"``.
    lam : float, optional
        Poids de la régularisation (sans dimension : le critère est
        invariant d'échelle). ``None`` : choisi par ``rule``.
    regularizer : {"identity", "gradient", "laplacian"}, default "laplacian"
        Opérateur ``L``.
    rule : {"gcv", "discrepancy", "evidence"}, default "gcv"
        Choix automatique de ``λ`` quand ``lam=None`` : validation
        croisée généralisée (sans connaissance du bruit), principe de
        discrépance de Morozov (``‖h ∗ x − y‖² = N σ²``) ou vraisemblance
        marginale de Wiener–Hunt (``sigma`` utilisé s'il est donné) — voir
        :mod:`eoqual.deconvolution.tuning.criteria`. Sur données
        incomplètes, λ est choisi sur l'image pré-remplie.
    boundary : str, default "unknown"
        Marge inconnue (:mod:`eoqual.deconvolution.operators`) : +0.8 à
        +2.2 dB sur un flou à OTF nulle par endroits (bougé), neutre
        ailleurs (BENCHMARK.md).
    mask : np.ndarray of bool, optional
        Pixels valides (``True``) ; les autres (et les NaN) sont inconnus.
    saturation : float, optional
        Seuil de saturation (unités de l'image) : attache unilatérale.
    n_fill_iter : int, default 30
        Itérations de remplissage des pixels inconnus ou saturés (et de la
        marge si ``boundary="unknown"``) ; sans effet sur des données
        complètes.
    """
    pb = setup(image, psf, boundary, mask=mask, saturation=saturation)
    L2 = regularizer_power(regularizer, pb.y.shape)
    Y = spfft.fft2(pb.y)
    if lam is None:
        from ..tuning.criteria import select_lambda

        lam = select_lambda(Y, pb.op.abs2, L2, rule=rule, sigma=resolve_sigma(pb.domain.restore(pb.y), sigma) if rule == "discrepancy" else (sigma if rule == "evidence" else None))
    if lam < 0:
        raise ValueError(f"lam doit être positif ou nul, reçu {lam}.")
    W = np.conj(pb.op.otf) / np.maximum(pb.op.abs2 + lam * L2, 1e-300)
    x = solve_with_fill(pb, lambda y: pb.op.solve_quadratic(spfft.fft2(y) * W, 1.0), n_fill_iter)
    return pb.domain.restore(x)
