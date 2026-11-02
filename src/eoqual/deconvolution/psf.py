#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSF de synthèse (somme 1, taille impaire, centrées) et recentrage d'une
PSF mesurée.

- :func:`gaussian_psf` : flou optique et détecteur approché, isotrope ou
  anisotrope ;
- :func:`airy_psf` : diffraction d'une pupille circulaire ;
- :func:`moffat_psf` : PSF à ailes larges (turbulence, diffusion) ;
- :func:`disk_psf` : défocalisation en optique géométrique (disque
  uniforme) ;
- :func:`motion_psf` : bougé linéaire uniforme ;
- :func:`center_psf` : ramène le barycentre d'une PSF au centre d'un
  support de taille impaire ;
- :func:`system_psf` : PSF d'un instrument à partir de ses paramètres
  physiques (diffraction d'une pupille annulaire, défocalisation, filé,
  jitter, intégration sur le pixel) ;
- :func:`combine_psfs` : convolution de plusieurs PSF (chaîne
  d'acquisition).

Gaussienne, Airy et Moffat sont **intégrées sur la surface du pixel**
(:func:`eoqual.core.psf.integrated_profile`), comme le ferait un
détecteur ; les bords des disques et segments sont anti-crénelés par
sur-échantillonnage (moyenne de ``supersample²`` sous-pixels).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from eoqual.core.kernel import convolve_circular
from eoqual.core.psf import airy_radius_px, integrated_profile

__all__ = [
    "gaussian_psf", "airy_psf", "moffat_psf", "disk_psf", "motion_psf", "center_psf",
    "system_psf", "combine_psfs", "annular_airy",
]


def _odd(n: float) -> int:
    n = int(np.ceil(n))
    return n if n % 2 else n + 1


def _profile(n: int, profile: str, params: dict, oversample: int = 7) -> np.ndarray:
    c = (n - 1) / 2.0
    k = integrated_profile((n, n), (c, c), profile, params, oversample)
    return k / k.sum()


def gaussian_psf(
    sigma: float,
    size: int | None = None,
    *,
    sigma_x: float | None = None,
    theta: float = 0.0,
) -> np.ndarray:
    """
    PSF gaussienne de somme 1.

    Parameters
    ----------
    sigma : float
        Écart-type [pixels] (selon ``y`` si ``sigma_x`` est donné).
        FWHM = 2.355 σ.
    size : int, optional
        Côté du support (impair). Défaut : ``2·ceil(4σ) + 1``.
    sigma_x : float, optional
        Écart-type selon ``x`` (PSF anisotrope).
    theta : float, default 0
        Rotation [rad].
    """
    sy = float(sigma)
    sx = sy if sigma_x is None else float(sigma_x)
    if sy <= 0 or sx <= 0:
        raise ValueError("Les écarts-types doivent être strictement positifs.")
    n = _odd(size if size is not None else 2 * np.ceil(4 * max(sy, sx)) + 1)
    return _profile(n, "gaussian", {"sigma": (sy, sx), "theta": theta})


def airy_psf(first_zero_radius: float, size: int | None = None) -> np.ndarray:
    """
    Tache d'Airy de somme 1.

    Parameters
    ----------
    first_zero_radius : float
        Rayon du premier anneau noir [pixels] : ``1.22 λ/D`` multiplié par
        la focale et divisé par le pas pixel. Un système échantillonné à
        Nyquist (``λ/D`` = 2 pixels) a un rayon de 2.44 pixels.
    size : int, optional
        Côté du support (impair). Défaut : ``2·ceil(4·first_zero_radius) + 1``
        (quatre anneaux noirs).
    """
    if first_zero_radius <= 0:
        raise ValueError("first_zero_radius doit être strictement positif.")
    n = _odd(size if size is not None else 2 * np.ceil(4 * first_zero_radius) + 1)
    return _profile(n, "airy", {"radius_zero": float(first_zero_radius)})


def moffat_psf(alpha: float, beta: float = 4.765, size: int | None = None) -> np.ndarray:
    """
    PSF de Moffat ``(1 + r²/α²)^-β`` de somme 1 : ailes plus larges qu'une
    gaussienne (β → ∞ : gaussienne ; β = 4.765 : turbulence
    atmosphérique, Trujillo et al. 2001).

    Parameters
    ----------
    alpha : float
        Échelle [pixels].
    beta : float
        Forme (> 1).
    size : int, optional
        Côté du support (impair). Défaut : rayon où le profil tombe à
        1e-4 du pic.
    """
    if alpha <= 0 or beta <= 1:
        raise ValueError("alpha doit être > 0 et beta > 1.")
    if size is None:
        radius = alpha * np.sqrt(1e-4 ** (-1.0 / beta) - 1.0)
        size = 2 * np.ceil(radius) + 1
    return _profile(_odd(size), "moffat", {"alpha": float(alpha), "beta": float(beta)})


def _supersampled(mask_func, n: int, supersample: int) -> np.ndarray:
    c = (n - 1) / 2.0
    s = np.arange(supersample) / supersample + 0.5 / supersample - 0.5
    grid = np.arange(n) - c
    yy = (grid[:, None, None, None] + s[None, None, :, None])
    xx = (grid[None, :, None, None] + s[None, None, None, :])
    out = mask_func(yy, xx).mean(axis=(2, 3))
    return out / out.sum()


def disk_psf(radius: float, size: int | None = None, supersample: int = 8) -> np.ndarray:
    """
    Disque uniforme de rayon ``radius`` [pixels] (défocalisation,
    optique géométrique), somme 1.
    """
    if radius <= 0:
        raise ValueError("radius doit être strictement positif.")
    n = _odd(size if size is not None else 2 * np.ceil(radius) + 3)
    return _supersampled(lambda y, x: (x**2 + y**2 <= radius**2).astype(float), n, supersample)


def motion_psf(
    length: float,
    angle: float = 0.0,
    size: int | None = None,
    supersample: int = 8,
    width: float = 1.0,
) -> np.ndarray:
    """
    Bougé linéaire uniforme, somme 1.

    Parameters
    ----------
    length : float
        Longueur du déplacement [pixels].
    angle : float, default 0
        Direction [rad] par rapport à l'axe ``x``, sens trigonométrique
        (``y`` vers le haut de l'image).
    width : float, default 1
        Épaisseur du segment [pixels].
    """
    if length <= 0:
        raise ValueError("length doit être strictement positif.")
    n = _odd(size if size is not None else np.ceil(length) + 3)
    c, s = np.cos(angle), np.sin(angle)

    def mask(y, x):
        along = x * c - y * s          # ligne 0 en haut : y image vers le bas
        across = x * s + y * c
        return ((np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)).astype(float)

    return _supersampled(mask, n, supersample)


def center_psf(psf: np.ndarray, size: int | None = None) -> np.ndarray:
    """
    Recentre une PSF mesurée : fond négatif mis à zéro, barycentre ramené
    au centre d'un support carré de taille impaire (interpolation
    spline d'ordre 3), somme 1.

    Parameters
    ----------
    psf : np.ndarray
        PSF 2D, pic n'importe où dans le tableau.
    size : int, optional
        Côté du support de sortie (impair). Défaut : plus grande dimension
        de ``psf``, rendue impaire.
    """
    h = np.clip(np.asarray(psf, dtype=np.float64), 0, None)
    if h.ndim != 2 or h.sum() <= 0:
        raise ValueError("PSF 2D de somme positive attendue.")
    n = _odd(size if size is not None else max(h.shape))
    canvas = np.zeros((max(n, h.shape[0]), max(n, h.shape[1])))
    canvas[: h.shape[0], : h.shape[1]] = h
    cy, cx = ndimage.center_of_mass(canvas)
    ty, tx = (canvas.shape[0] - 1) / 2.0, (canvas.shape[1] - 1) / 2.0
    shifted = ndimage.shift(canvas, (ty - cy, tx - cx), order=3, mode="constant")
    oy, ox = (canvas.shape[0] - n) // 2, (canvas.shape[1] - n) // 2
    out = np.clip(shifted[oy : oy + n, ox : ox + n], 0, None)
    return out / out.sum()


def annular_airy(r: np.ndarray, first_zero_radius: float, obscuration: float = 0.0) -> np.ndarray:
    """
    Intensité de diffraction d'une pupille annulaire, normalisée à 1 au
    centre (Born & Wolf 1999 ; Mahajan 2001, pupilles annulaires) :

    ``I(v) = [2J1(v)/v − ε² · 2J1(εv)/(εv)]² / (1 − ε²)²``,
    ``v = 3.8317 r / r₀``, ``ε`` obturation centrale (rapport des diamètres).

    Parameters
    ----------
    r : np.ndarray
        Distance au centre [pixels].
    first_zero_radius : float
        Rayon ``r₀`` du premier anneau noir de la pupille **pleine**
        (``1.22 λ f / (D p)``, :func:`eoqual.core.psf.airy_radius_px`).
    obscuration : float, default 0
        Obturation centrale ``ε`` dans [0, 1[ (miroir secondaire d'un
        télescope de Cassegrain, typiquement 0.2 à 0.4).
    """
    from scipy.special import j1

    if not 0.0 <= obscuration < 1.0:
        raise ValueError("obscuration doit être dans [0, 1[.")
    v = 3.8317059702075125 * np.asarray(r, dtype=np.float64) / first_zero_radius
    v = np.where(v == 0, 1e-12, v)
    amp = 2 * j1(v) / v
    if obscuration > 0:
        ev = obscuration * v
        amp = amp - obscuration**2 * 2 * j1(ev) / ev
    return amp**2 / (1.0 - obscuration**2) ** 2


def combine_psfs(*psfs: np.ndarray) -> np.ndarray:
    """
    Convolution de plusieurs PSF centrées (optique ⊛ détecteur ⊛ bougé...),
    somme 1, support impair égal à la somme des supports.
    """
    if not psfs:
        raise ValueError("Au moins une PSF attendue.")
    n = _odd(sum(np.shape(h)[0] for h in psfs))
    out = np.zeros((n, n))
    out[n // 2, n // 2] = 1.0
    for h in psfs:
        h = np.asarray(h, dtype=np.float64)
        out = convolve_circular(out, h / h.sum())
    out = np.clip(out, 0, None)
    return out / out.sum()


def system_psf(
    wavelength: float,
    diameter: float,
    focal_length: float,
    pixel_pitch: float,
    *,
    obscuration: float = 0.0,
    defocus_radius: float = 0.0,
    motion_length: float = 0.0,
    motion_angle: float = 0.0,
    jitter_sigma: float = 0.0,
    size: int | None = None,
    oversample: int = 7,
) -> np.ndarray:
    """
    PSF d'un instrument d'observation, sur la grille du détecteur, somme 1.

    Chaîne modélisée (convolutions, calculées sur une grille ``oversample``
    fois plus fine puis intégrées sur la surface de chaque pixel) :

    1. **diffraction** d'une pupille circulaire de diamètre ``D``, avec
       obturation centrale ``ε`` (:func:`annular_airy`) ;
    2. **défocalisation** géométrique (disque de rayon ``defocus_radius``) ;
    3. **filé** linéaire uniforme pendant l'intégration (bougé, défilement
       mal compensé d'un capteur TDI) ;
    4. **jitter** (micro-vibrations) gaussien ;
    5. **intégration sur le pixel** (pixel carré de facteur de remplissage
       1) — c'est elle qui donne la FTM ``sinc(πf)`` du détecteur, 0.64 à
       Nyquist.

    Parameters
    ----------
    wavelength, diameter, focal_length, pixel_pitch : float
        Longueur d'onde, diamètre de pupille, focale, pas pixel (même unité).
    obscuration : float, default 0
        Obturation centrale (rapport des diamètres).
    defocus_radius : float, default 0
        Rayon du disque de défocalisation [pixels].
    motion_length : float, default 0
        Longueur du filé [pixels] ; ``motion_angle`` sa direction [rad].
    jitter_sigma : float, default 0
        Écart-type du jitter [pixels].
    size : int, optional
        Côté du support (impair) ; défaut : quatre anneaux noirs plus les
        étendues du filé, de la défocalisation et du jitter.
    oversample : int, default 7
        Sur-échantillonnage (impair) de la grille de calcul.

    Examples
    --------
    Instrument échantillonné à Nyquist (``λ f / (D p) = 2``), obturation 0.3,
    filé d'un demi-pixel :

    >>> h = system_psf(0.55e-6, 0.3, 3.0, 11e-6, obscuration=0.3, motion_length=0.5)
    """
    if oversample < 1 or oversample % 2 == 0:
        raise ValueError("oversample doit être un entier impair ≥ 1.")
    for name, v in (("defocus_radius", defocus_radius), ("motion_length", motion_length),
                    ("jitter_sigma", jitter_sigma)):
        if v < 0:
            raise ValueError(f"{name} doit être positif ou nul.")
    r0 = airy_radius_px(wavelength, diameter, focal_length, pixel_pitch)
    if size is None:
        size = 2 * np.ceil(4 * r0 + motion_length / 2 + defocus_radius + 3 * jitter_sigma) + 3
    n = _odd(size)
    m = n * oversample
    c = (m - 1) / 2.0
    yy, xx = np.indices((m, m), dtype=np.float64)
    r = np.hypot(yy - c, xx - c) / oversample
    fine = annular_airy(r, r0, obscuration)
    kernels = []
    if defocus_radius > 0:
        kernels.append(disk_psf(defocus_radius * oversample))
    if motion_length > 0:
        kernels.append(motion_psf(motion_length * oversample, motion_angle))
    if jitter_sigma > 0:
        kernels.append(gaussian_psf(jitter_sigma * oversample))
    for k in kernels:
        if k.shape[0] > m:
            raise ValueError("size trop petit pour le filé, la défocalisation ou le jitter demandés.")
        fine = convolve_circular(fine, k)
    fine = np.clip(fine, 0, None)
    det = fine.reshape(n, oversample, n, oversample).sum(axis=(1, 3))
    return det / det.sum()
