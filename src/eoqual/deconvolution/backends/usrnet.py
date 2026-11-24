#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthode ``"usrnet"`` : déconvolution par le réseau déplié USRNet, fourni par
**eoqual-neural** (extra ``deep``).

Le réseau (architecture, poids, inférence sur un domaine périodique dans
[0, 1]) est dans ``eoqual.neural.unrolled.usrnet``. Ce module n'en garde que
ce qui relève de la déconvolution : traitement des bords, pixels masqués ou
saturés, mise à l'échelle depuis les unités du capteur, estimation de σ
(docs/technique/reference.md §7.2).
"""
from __future__ import annotations

import numpy as np

from ..utils import check_image, from_unit_range, resolve_sigma, to_unit_range

__all__ = ["usrnet_deconvolve", "consistent_extension"]


def _neural():
    try:
        from eoqual.neural.unrolled import usrnet
    except ImportError as exc:
        raise ImportError(
            "La méthode 'usrnet' est un réseau de neurones fourni par eoqual-neural. "
            "Installer avec : pip install 'eoqual-deconvolution[deep]'"
        ) from exc
    return usrnet


def consistent_extension(image, psf, mask, saturation, n_fill_iter=30):
    """
    Domaine de travail prolongé dont la marge (et les pixels inconnus ou
    saturés) est remplie par le flou d'une restauration de Wiener–Hunt à
    marge inconnue (§2.2, §3.5) : l'image complète est alors une image
    floue cohérente avec la PSF sur tout le domaine périodique, ce que
    suppose le module de données d'USRNet (entraîné en flou circulaire).

    Returns
    -------
    (pb, y_dom)
        Problème préparé (domaine, opérateur) et image du domaine complété.
    """
    from scipy import fft as spfft

    from ..tuning.criteria import wiener_hunt_estimate
    from ._common import regularizer_power, setup, solve_with_fill

    pb = setup(image, psf, "unknown", mask=mask, saturation=saturation)
    L2 = regularizer_power("laplacian", pb.y.shape)
    lam = wiener_hunt_estimate(spfft.fft2(pb.y), pb.op.abs2, L2)[0]
    W = np.conj(pb.op.otf) / np.maximum(pb.op.abs2 + lam * L2, 1e-300)
    x = solve_with_fill(pb, lambda y: pb.op.solve_quadratic(spfft.fft2(y) * W, 1.0), n_fill_iter)
    return pb, pb.fill(pb.op.forward(x), pb.y, pb.level)


def usrnet_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    weights="usrnet",
    boundary: str = "unknown",
    mask=None,
    saturation: float | None = None,
    data_range: tuple[float, float] | None = None,
    device=None,
    callback=None,
) -> np.ndarray:
    """
    Déconvolution par USRNet (extra ``deep``, eoqual-neural), PSF et σ connus.

    L'image est ramenée à [0, 1] (``data_range``, défaut min/max) et
    prolongée ; le réseau reçoit l'image répliquée sur trois canaux et la
    sortie est leur moyenne, recadrée et remise à l'échelle.

    **Bords.** Le réseau suppose une convolution circulaire (son
    entraînement). ``boundary="unknown"`` (défaut) remplit la marge — et
    les pixels masqués ou saturés — par le flou d'une restauration de
    Wiener–Hunt à marge inconnue, calculée une fois avant le réseau : le
    domaine est alors une image floue cohérente avec la PSF. Un
    prolongement miroir (``"reflect"``) ne l'est pas pour une PSF
    asymétrique (bougé) et ses défauts sont amplifiés aux zéros de la FTM ;
    compléter la marge à chaque itération, comme le Plug-and-Play (§7.1),
    diverge, les estimées intermédiaires d'un réseau déplié n'étant pas des
    images (docs/technique/reference.md §7.2).

    Parameters
    ----------
    sigma : float, optional
        Écart-type du bruit (unités de l'image) ; ``None`` : estimé. Les
        poids publiés couvrent σ de 0 à 25/255 de la dynamique.
    weights : str or path, default "usrnet"
        ``"usrnet"``, ``"usrnet_tiny"`` ou chemin de poids réentraînés
        (:mod:`eoqual.deconvolution.training`).
    boundary : {"unknown", "reflect", "periodic"}, default "unknown"
    mask, saturation :
        Pixels valides et seuil de saturation (§2.2) ; avec
        ``boundary="unknown"`` seulement (remplis avant le réseau).
    device : str, optional
        ``"cuda"``, ``"mps"``, ``"cpu"`` ; défaut : le plus rapide disponible.
    callback : callable, optional
        ``callback(k, x)`` après chaque itération (domaine de travail,
        unités d'origine).
    """
    from ._common import setup

    if boundary not in ("unknown", "reflect", "periodic"):
        raise ValueError("usrnet : boundary doit valoir 'unknown', 'reflect' ou 'periodic'.")
    if boundary != "unknown" and (mask is not None or saturation is not None):
        raise ValueError("usrnet : mask/saturation demandent boundary='unknown'.")
    net = _neural()
    if boundary == "unknown":
        pb, y_dom = consistent_extension(image, psf, mask, saturation)
    else:
        pb = setup(check_image(image), psf, boundary)
        y_dom = pb.y
    observed = pb.domain.restore(pb.y)
    s = resolve_sigma(observed, sigma)
    y, lo, scale = to_unit_range(y_dom, data_range if data_range is not None
                                 else (float(np.min(observed)), float(np.max(observed))))
    cb = None if callback is None else (lambda k, x: callback(k, from_unit_range(x, lo, scale)))
    out = net.usrnet_restore(y, pb.psf, s / scale, weights=weights, device=device, callback=cb)
    return pb.domain.restore(from_unit_range(out, lo, scale))
