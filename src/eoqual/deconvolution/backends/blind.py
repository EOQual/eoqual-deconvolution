#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Déconvolution aveugle : estimation conjointe de l'image et de la PSF.

Estimation de la PSF avec un a priori **L0 sur les gradients** (Xu,
Zheng & Jia 2013 ; Pan, Hu, Su & Yang 2014), du grossier au fin :

1. pas en ``x`` : ``min ‖h ∗ x − y‖² + λ‖∇x‖₀``, par éclatement
   semi-quadratique (HQS) — seuillage dur de ``∇x`` puis solution
   exacte en Fourier, pour une pénalité ``β`` croissante. L'image
   intermédiaire est volontairement « en aplats » : seuls les contours
   francs, fiables, guident la PSF ;
2. pas en ``h`` : ``min ‖h ∗ ∇x − ∇y‖² + γ‖h‖²`` dans le domaine des
   gradients (spectre presque plat, Cho & Lee 2009), solution exacte en
   Fourier, restreinte au support, positive, valeurs < 5 % du maximum
   annulées, somme 1, barycentre recentré ;
3. ``λ`` décroît au fil des itérations ; chaque échelle (facteur √2)
   initialise la suivante.

L'image finale est restaurée en non aveugle avec la PSF estimée
(variation totale, :func:`eoqual.deconvolution.backends.sparse.tv_primal_dual_deconvolve`).

La minimisation alternée directe ``min_{x,h} ‖h ∗ x − y‖² + λTV(x)``
(PALM) a été essayée et écartée : elle converge vers la solution
triviale (``h`` = Dirac, ``x = y``) ou vers une PSF décalée selon λ
(REFERENCE_TECHNIQUE.md, annexe A).
"""
from __future__ import annotations

import numpy as np
from scipy import fft as spfft
from scipy import ndimage

from ._common import Callback
from ..operators import prepare_domain
from ..psf import center_psf, gaussian_psf
from ..utils import check_image, resolve_sigma, to_unit_range
from .sparse import tv_primal_dual_deconvolve

__all__ = ["blind_deconvolve", "blind_l0_deconvolve"]


def _odd(n: float) -> int:
    n = max(int(round(n)), 3)
    return n if n % 2 else n + 1


def _otf(h: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from eoqual.core.kernel import psf2otf

    return psf2otf(h, shape)


def _diff_otfs(shape: tuple[int, int]) -> list[np.ndarray]:
    """OTF des différences avant périodiques (mêmes que :func:`operators.gradient`)."""
    dy = np.zeros(shape)
    dy[0, 0], dy[-1, 0] = -1.0, 1.0
    dx = np.zeros(shape)
    dx[0, 0], dx[0, -1] = -1.0, 1.0
    # gradient(x)[0] = x[i+1] − x[i] : noyau de corrélation → conj en convolution.
    return [np.conj(spfft.fft2(dy)), np.conj(spfft.fft2(dx))]


def _crop_kernel(full: np.ndarray, size: int) -> np.ndarray:
    """Fenêtre ``size × size`` centrée sur l'origine d'un tableau périodique."""
    rolled = np.roll(full, (size // 2, size // 2), axis=(0, 1))
    return rolled[:size, :size]


def _l0_image(Y, H, D, lam, beta_max=1e5):
    """
    ``min ‖h ∗ x − y‖² + λ‖∇x‖₀`` par HQS (Xu et al. 2011), en partant de
    l'image observée (un départ par filtre inverse amplifierait un bruit
    que le seuillage L0 conserverait ensuite comme contours).
    """
    absH2 = np.abs(H) ** 2
    absD2 = sum(np.abs(d) ** 2 for d in D)
    num0 = np.conj(H) * Y
    x = spfft.ifft2(Y).real
    beta = 2.0 * lam
    while beta < beta_max:
        X = spfft.fft2(x)
        g = [spfft.ifft2(d * X).real for d in D]
        mask = (g[0] ** 2 + g[1] ** 2) < lam / beta
        g = [np.where(mask, 0.0, gi) for gi in g]
        num = num0 + beta * sum(np.conj(d) * spfft.fft2(gi) for d, gi in zip(D, g))
        x = spfft.ifft2(num / (absH2 + beta * absD2)).real
        beta *= 2.0
    return x


def _kernel_estimate(x, y, size, gamma_rel, D):
    """``min ‖h ∗ ∇x − ∇y‖² + γ‖h‖²`` en Fourier, puis contraintes."""
    X, Y = spfft.fft2(x), spfft.fft2(y)
    gx = [d * X for d in D]
    gy = [d * Y for d in D]
    power = sum(np.abs(g) ** 2 for g in gx)
    cross = sum(np.conj(a) * b for a, b in zip(gx, gy))
    gamma = gamma_rel * float(power.max())
    h = _crop_kernel(spfft.ifft2(cross / (power + gamma)).real, size)
    h = np.clip(h, 0.0, None)
    if h.sum() <= 0:
        return None
    h[h < 0.05 * h.max()] = 0.0
    return center_psf(h, size)


def blind_deconvolve(
    image: np.ndarray,
    psf_size: int = 15,
    sigma: float | None = None,
    *,
    psf_init: np.ndarray | None = None,
    lam: float = 0.4,
    lam_l0: float = 4e-3,
    lam_min: float = 1e-4,
    gamma: float = 1e-4,
    n_scales: int | None = None,
    n_outer: int = 6,
    final_n_iter: int = 300,
    data_range: tuple[float, float] | None = None,
    boundary: str = "reflect",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estime l'image et la PSF.

    Parameters
    ----------
    image : np.ndarray
        Image floue 2D.
    psf_size : int, default 15
        Côté (impair) du support de la PSF à estimer : le prendre proche de
        l'étendue attendue du flou. Un support **trop grand** dégrade
        fortement le résultat (le surplus capte du bruit) ; un support un
        peu trop petit est bien mieux toléré (BENCHMARK.md).
    sigma : float, optional
        Écart-type du bruit (unités de l'image) pour la restauration
        finale ; ``None`` : estimé.
    psf_init : np.ndarray, optional
        PSF initiale à l'échelle fine (impose ``psf_size``) ; défaut :
        gaussienne étroite à l'échelle la plus grossière.
    lam : float, default 0.4
        Poids TV de la restauration finale, en multiples de σ.
    lam_l0 : float, default 4e-3
        Poids initial de l'a priori L0 (image ramenée à [0, 1]), valeur de
        Pan et al. (2014). Plus grand : image intermédiaire plus « en
        aplats », plus robuste au bruit, moins de détails pour la PSF.
    lam_min : float, default 1e-4
        Plancher de ``λ``, qui décroît d'un facteur 0.9 à chaque
        itération.
    gamma : float, default 1e-4
        Régularisation ℓ2 de la PSF, relative au maximum de ``|∇X|²``.
    n_scales : int, optional
        Nombre d'échelles ; défaut : jusqu'à un support de 3 à 5 pixels.
    n_outer : int, default 6
        Alternances ``x``/``h`` par échelle.
    final_n_iter : int, default 300
        Itérations de la restauration finale non aveugle.
    data_range : tuple, optional
    boundary : str, default "reflect"

    Returns
    -------
    (np.ndarray, np.ndarray)
        Image restaurée (géométrie et unités de ``image``) et PSF estimée
        (somme 1, ``psf_size × psf_size``).
    """
    y0 = check_image(image)
    if psf_init is not None:
        psf_size = np.asarray(psf_init).shape[0]
    psf_size = _odd(psf_size)
    s = resolve_sigma(y0, sigma)
    yu, _, _ = to_unit_range(y0, data_range)
    if n_scales is None:
        n_scales = max(1, int(np.floor(np.log(psf_size / 4.0) / np.log(np.sqrt(2.0)))) + 1)

    h = None
    lam_cur = float(lam_l0)
    for level in range(n_scales - 1, -1, -1):
        factor = np.sqrt(2.0) ** level
        size = _odd(psf_size / factor)
        y_s = ndimage.zoom(yu, 1.0 / factor, order=1) if level else yu
        if h is None:
            if psf_init is not None and level == 0:
                h = center_psf(np.asarray(psf_init, dtype=np.float64), size)
            else:
                h = gaussian_psf(0.6, size)
        else:
            h = center_psf(ndimage.zoom(h, size / h.shape[0], order=1), size)
        dom = prepare_domain(y_s, (size, size), boundary)
        y_w = dom.image
        D = _diff_otfs(y_w.shape)
        Y = spfft.fft2(y_w)
        for _ in range(n_outer):
            x = _l0_image(Y, _otf(h, y_w.shape), D, lam_cur)
            h_new = _kernel_estimate(x, y_w, size, gamma, D)
            if h_new is not None:
                h = h_new
            lam_cur = max(lam_cur * 0.9, lam_min)

    restored = tv_primal_dual_deconvolve(y0, h, sigma=s, lam=lam, n_iter=final_n_iter,
                                         data_range=data_range, boundary=boundary)
    return restored, h


def blind_l0_deconvolve(
    image: np.ndarray,
    psf: np.ndarray | None = None,
    sigma: float | None = None,
    *,
    psf_size: int = 15,
    callback: Callback = None,
    **kwargs,
) -> np.ndarray:
    """
    Signature commune (:func:`eoqual.deconvolution.runner.deconvolve`) de
    :func:`blind_deconvolve` : ``psf``, s'il est donné, sert de PSF
    initiale et fixe la taille du support. Ne renvoie que l'image ; pour
    obtenir aussi la PSF estimée, appeler :func:`blind_deconvolve`.

    ``callback`` est accepté pour la signature commune mais n'est pas
    appelé.
    """
    del callback
    restored, _ = blind_deconvolve(image, psf_size, sigma, psf_init=psf, **kwargs)
    return restored
