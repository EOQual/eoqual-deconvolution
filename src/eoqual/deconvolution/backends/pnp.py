#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plug-and-Play : l'a priori est un débruiteur, appelé à chaque itération
d'un algorithme d'éclatement.

Problème ``min ½‖Hx − y‖²/σ² + λΦ(x)``, éclaté en ``x`` (attache aux
données, résolue exactement en Fourier) et ``z`` (débruitage de niveau
``σ_k``) :

- **HQS** (*half-quadratic splitting*, Geman & Yang 1995 ; schéma de
  Zhang et al. 2017, 2021) : ``μ_k = λ/σ_k²``,
  ``x = argmin ‖Hx − y‖² + μ_k σ² ‖x − z‖²``, ``z = D(x, σ_k)`` ;
- **ADMM** (Venkatakrishnan, Bouman & Wohlberg 2013 ; Chan, Wang &
  Elgendy 2017) : même pas ``x`` appliqué à ``z − u``, ``z = D(x + u, σ_k)``,
  ``u ← u + x − z``.

Le niveau ``σ_k`` du débruiteur décroît géométriquement de
``sigma_start`` à ``max(σ, sigma_min)`` : un débruitage fort au début
stabilise, puis l'a priori s'efface devant les données.

Débruiteur : nom d'une méthode d'eoqual-denoising (extra ``pnp`` ; extra
``deep`` pour ``drunet`` et les autres réseaux), ou fonction
``denoiser(image, sigma) -> image``. Il reçoit l'image ramenée à [0, 1]
et σ dans ces mêmes unités.
"""
from __future__ import annotations

import inspect
from typing import Callable, Union

import numpy as np
from scipy import fft as spfft

from ._common import Callback, setup
from ..utils import from_unit_range, resolve_sigma, to_unit_range

__all__ = ["pnp_hqs_deconvolve", "pnp_admm_deconvolve", "make_denoiser"]

Denoiser = Union[str, Callable[[np.ndarray, float], np.ndarray]]

#: Débruiteur par défaut : NL-Ridge, réimplémentation MIT du socle
#: d'eoqual-denoising (le meilleur compromis qualité/temps sans PyTorch).
DEFAULT_DENOISER = "nl_ridge"


def make_denoiser(denoiser: Denoiser, **kwargs) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Renvoie une fonction ``f(image, sigma)``.

    ``denoiser`` est soit déjà une telle fonction (``kwargs`` lui sont
    passés), soit le nom d'une méthode d'eoqual-denoising. Dans ce cas la
    dynamique est fixée à [0, 1] (``data_range``) quand la méthode
    l'accepte : sinon elle la déduirait du min/max de l'itéré, qui varie.

    Raises
    ------
    ImportError
        Si eoqual-denoising n'est pas installé (extra ``pnp``).
    """
    if callable(denoiser):
        return lambda img, s: denoiser(img, s, **kwargs)
    try:
        from eoqual.denoising.runner import _resolve
        from eoqual.denoising.config import METHODS_CONFIGS
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            "Les méthodes Plug-and-Play nécessitent eoqual-denoising : "
            'pip install "eoqual-deconvolution[pnp]" (MIT), ou "[deep]" pour '
            "les débruiteurs à réseau de neurones (PyTorch)."
        ) from exc
    if denoiser not in METHODS_CONFIGS:
        raise ValueError(f"Débruiteur '{denoiser}' inconnu d'eoqual-denoising : {list(METHODS_CONFIGS)}")
    func = _resolve(denoiser)
    extra = dict(kwargs)
    if "data_range" in inspect.signature(func).parameters:
        extra.setdefault("data_range", (0.0, 1.0))
    return lambda img, s: np.asarray(func(img, sigma=s, **extra), dtype=np.float64)


def _schedule(sigma_start: float, sigma_end: float, n_iter: int) -> np.ndarray:
    if n_iter == 1:
        return np.array([sigma_end])
    return np.geomspace(max(sigma_start, sigma_end), sigma_end, n_iter)


def _pnp(
    image, psf, sigma, *, admm, denoiser, denoiser_kwargs, lam, n_iter, sigma_start, sigma_min,
    data_range, boundary, callback, fill_boundary=True,
):
    if n_iter < 1:
        raise ValueError("n_iter doit être ≥ 1.")
    s = resolve_sigma(image, sigma)
    pb = setup(image, psf, boundary)
    y, lo, scale = to_unit_range(pb.y, data_range)
    s_u = s / scale
    den = make_denoiser(denoiser, **(denoiser_kwargs or {}))
    sigmas = _schedule(sigma_start, max(s_u, sigma_min), n_iter)

    H, abs2 = pb.op.otf, pb.op.abs2
    # Marge (boundary="reflect") traitée comme donnée inconnue : remplacée à
    # chaque itération par H·z, le flou de l'estimée courante (les vraies
    # données ne sont imposées que sur l'image d'origine). Sans cela, la
    # marge, qui n'est pas une image floue par la PSF, est amplifiée par le
    # pas en x presque inverse des premières itérations.
    ny, nx = np.shape(image)
    inside = np.zeros(y.shape, dtype=bool)
    inside[:ny, :nx] = True
    fill = fill_boundary and boundary == "reflect" and not inside.all()
    HtY = np.conj(H) * spfft.fft2(y)
    z = y.copy()
    u = np.zeros_like(y)
    x = y
    alpha_prev = None
    for k, sk in enumerate(sigmas):
        alpha = lam * max(s_u, 1e-6) ** 2 / sk**2
        if admm and alpha_prev is not None:
            u *= alpha_prev / alpha  # variable duale réduite : u = dual / ρ, à remettre à l'échelle
        alpha_prev = alpha
        target = z - u if admm else z
        if fill and k > 0:
            y_fill = np.where(inside, y, pb.op.forward(z))
            HtY = np.conj(H) * spfft.fft2(y_fill)
        x = spfft.ifft2((HtY + alpha * spfft.fft2(target)) / (abs2 + alpha)).real
        v = x + u if admm else x
        z = den(v, float(sk))
        if admm:
            u = u + x - z
        if callback is not None:
            callback(k, from_unit_range(z, lo, scale))
    return pb.domain.restore(from_unit_range(z, lo, scale))


def pnp_hqs_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    denoiser: Denoiser = DEFAULT_DENOISER,
    denoiser_kwargs: dict | None = None,
    lam: float = 0.23,
    n_iter: int = 8,
    sigma_start: float = 0.2,
    sigma_min: float = 0.002,
    data_range: tuple[float, float] | None = None,
    fill_boundary: bool = True,
    boundary: str = "reflect",
    callback: Callback = None,
) -> np.ndarray:
    """
    Plug-and-Play HQS.

    Parameters
    ----------
    sigma : float, optional
        Écart-type du bruit (unités de l'image) ; ``None`` : estimé.
    denoiser : str or callable, default "nl_ridge"
        Méthode d'eoqual-denoising (``"nl_ridge"``, ``"nl_bayes"``,
        ``"tv_chambolle"``, ``"drunet"``...) ou fonction ``f(image, sigma)``.
    denoiser_kwargs : dict, optional
        Arguments supplémentaires du débruiteur.
    lam : float, default 0.23
        Poids de l'a priori (valeur de Zhang et al. 2021 pour le
        défloutage).
    n_iter : int, default 8
        Nombre d'itérations, donc d'appels au débruiteur.
    sigma_start : float, default 0.2
        Niveau initial du débruiteur, en fraction de la dynamique.
    sigma_min : float, default 0.002
        Plancher du niveau final (fraction de la dynamique), pour une
        image presque sans bruit.
    data_range : tuple, optional
    fill_boundary : bool, default True
        Avec ``boundary="reflect"``, remplace à chaque itération les
        données de la marge par ``H·z`` (la marge n'est pas une image
        floue par la PSF). Indispensable pour une PSF à zéros (bougé,
        +5 dB) ; coûte 0.5 à 1 dB sur un flou gaussien ou une
        défocalisation (REFERENCE_TECHNIQUE.md §7).
    boundary : str, default "reflect"
    callback : callable, optional
        ``callback(k, z)`` après chaque débruitage.
    """
    return _pnp(image, psf, sigma, admm=False, denoiser=denoiser, denoiser_kwargs=denoiser_kwargs,
                lam=lam, n_iter=n_iter, sigma_start=sigma_start, sigma_min=sigma_min,
                data_range=data_range, boundary=boundary, callback=callback,
                fill_boundary=fill_boundary)


def pnp_admm_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    denoiser: Denoiser = DEFAULT_DENOISER,
    denoiser_kwargs: dict | None = None,
    lam: float = 0.5,
    n_iter: int = 12,
    sigma_start: float = 0.2,
    sigma_min: float = 0.002,
    data_range: tuple[float, float] | None = None,
    fill_boundary: bool = True,
    boundary: str = "reflect",
    callback: Callback = None,
) -> np.ndarray:
    """
    Plug-and-Play ADMM, avec continuation du niveau du débruiteur
    (mêmes paramètres que :func:`pnp_hqs_deconvolve`, mais ``lam`` = 0.5
    par défaut, meilleur compromis mesuré sur les quatre flous du
    benchmark). La variable duale réduite ``u`` corrige l'écart entre
    ``x`` et ``z`` ; elle est remise à l'échelle quand le pas change.
    """
    return _pnp(image, psf, sigma, admm=True, denoiser=denoiser, denoiser_kwargs=denoiser_kwargs,
                lam=lam, n_iter=n_iter, sigma_start=sigma_start, sigma_min=sigma_min,
                data_range=data_range, boundary=boundary, callback=callback,
                fill_boundary=fill_boundary)
