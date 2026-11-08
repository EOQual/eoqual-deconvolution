#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Déconvolution aveugle **contrainte par l'optique** (« myope ») : la PSF
n'est pas un tableau libre de ``k²`` valeurs mais la PSF d'un modèle
physique de l'instrument (:func:`eoqual.deconvolution.psf.optical_psf`) —
diffraction d'une pupille connue, quelques aberrations de Zernike
(défocalisation, astigmatisme, coma…) et un jitter, **bornés à des plages
plausibles** — dont on estime seulement les paramètres libres, puis
l'image est restaurée avec la PSF estimée.

Estimation des paramètres ``θ`` (REFERENCE_TECHNIQUE.md §8.2) :

- ``"evidence"`` (défaut) : maximum de vraisemblance **marginale**
  ``p(y | θ)``, la scène étant intégrée sous un a priori gaussien
  stationnaire de spectre ``∝ |ω|^-β`` (Wiener–Hunt généralisé, β = 3
  par défaut ; niveau de l'a priori estimé avec θ, σ profilé).
  Contrairement au maximum joint en ``(x, h)``, ce critère ne dégénère pas
  vers la PSF de Dirac (Levin et al. 2011 ; Blanco & Mugnier 2011) ;
- ``"l0"`` : alternance « image à gradients parcimonieux (L0) / paramètres
  de la PSF » (cadre de ``blind_l0``, Xu et al. 2013, avec ici une PSF
  paramétrique), qui exploite les contours de la scène.

**Variation dans le champ** : ``tile_size`` découpe l'image en tuiles
chevauchantes, ``θ`` est estimé par tuile puis lissé par un modèle de
champ (constant, plan ou quadratique — la courbure de champ donne une
défocalisation quadratique en position), et chaque tuile est restaurée
avec sa PSF, les tuiles étant recollées par pondération de Hann.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import fft as spfft
from scipy import optimize

from eoqual.core.fourier import perdecomp
from eoqual.core.kernel import psf2otf
from eoqual.core.tiling import tile_slices

from ..psf import ZERNIKE_NAMES, optical_psf
from ..utils import check_image, resolve_sigma, to_unit_range

__all__ = ["OpticalModel", "OpticsBlindResult", "optics_blind_deconvolve", "blind_optics_deconvolve",
           "estimate_optical_parameters"]

_NAME_TO_J = {v: k for k, v in ZERNIKE_NAMES.items()}
ESTIMATORS = ("evidence", "l0")
#: Exposant par défaut du spectre a priori de la scène (|ω|^-β) du critère « evidence » (§8.2).
DEFAULT_PRIOR_EXPONENT = 3.0
FIELD_MODELS = ("independent", "constant", "plane", "quadratic")


def _key(name):
    """Nom de paramètre normalisé : ``"jitter"`` ou indice de Noll."""
    if name == "jitter":
        return "jitter"
    if isinstance(name, str):
        if name not in _NAME_TO_J:
            raise ValueError(f"Paramètre inconnu {name!r} : 'jitter', un indice de Noll ou {sorted(_NAME_TO_J)}.")
        return _NAME_TO_J[name]
    j = int(name)
    if j < 2:
        raise ValueError("Piston (j = 1) sans effet sur la PSF : indices de Noll ≥ 2.")
    return j


@dataclass
class OpticalModel:
    """
    Modèle paramétrique de la PSF d'un instrument.

    Attributes
    ----------
    sampling : float or sequence
        ``Q = λf/(Dp)`` (voir :func:`eoqual.deconvolution.psf.optical_psf`).
    free : dict
        Paramètres estimés et leurs bornes plausibles, ``{nom: (bas,
        haut)}`` : ``"jitter"`` [pixels] ou un mode de Zernike (indice de
        Noll ou nom, ``"defocus"``, ``"astig_0"``…) [ondes RMS].
    fixed : dict
        Valeurs connues (mêmes noms), par défaut nulles.
    obscuration, weights, size, oversample :
        Transmis à :func:`optical_psf`.
    """

    sampling: object
    free: dict = field(default_factory=lambda: {"defocus": (0.0, 0.25)})
    fixed: dict = field(default_factory=dict)
    obscuration: float = 0.0
    weights: object = None
    size: int | None = None
    oversample: int = 5

    def __post_init__(self):
        if not self.free:
            raise ValueError("free doit contenir au moins un paramètre à estimer.")
        self._free = [(_key(k), (float(lo), float(hi))) for k, (lo, hi) in self.free.items()]
        for k, (lo, hi) in self._free:
            if not lo <= hi:
                raise ValueError(f"Bornes invalides pour {k!r} : ({lo}, {hi}).")
            if k == "jitter" and lo < 0:
                raise ValueError("Le jitter est positif : borne basse ≥ 0.")
        self._fixed = {_key(k): float(v) for k, v in self.fixed.items()}
        both = set(self._fixed) & {k for k, _ in self._free}
        if both:
            raise ValueError(f"Paramètres à la fois libres et fixés : {sorted(map(str, both))}.")

    @property
    def names(self) -> list:
        """Paramètres libres, dans l'ordre du vecteur ``θ``."""
        return [k for k, _ in self._free]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [b for _, b in self._free]

    def as_dict(self, theta) -> dict:
        """Paramètres complets (fixés et estimés) sous forme lisible."""
        out = {}
        for k, v in list(self._fixed.items()) + list(zip(self.names, np.asarray(theta, float))):
            out["jitter" if k == "jitter" else ZERNIKE_NAMES.get(k, f"Z{k}")] = float(v)
        return out

    def psf(self, theta) -> np.ndarray:
        """PSF pour le vecteur de paramètres libres ``θ``."""
        values = dict(self._fixed)
        values.update(zip(self.names, np.asarray(theta, dtype=np.float64)))
        jitter = values.pop("jitter", 0.0)
        return optical_psf(self.sampling, zernike=values, obscuration=self.obscuration,
                           jitter_sigma=max(jitter, 0.0), weights=self.weights, size=self.size,
                           oversample=self.oversample)

    def starts(self, n: int) -> list[np.ndarray]:
        """Points de départ : grille sur le premier paramètre, milieu des autres intervalles (0 s'il y est)."""
        lo, hi = np.array(self.bounds).T
        mid = np.where((lo <= 0) & (hi >= 0), 0.0, (lo + hi) / 2)
        if "jitter" in self.names:
            i = self.names.index("jitter")
            mid[i] = lo[i] + 0.3 * (hi[i] - lo[i])
        out = []
        for t in np.linspace(0.15, 0.85, max(n, 1)) if n > 1 else [0.5]:
            s = mid.copy()
            s[0] = lo[0] + t * (hi[0] - lo[0])
            out.append(s)
        return out


@dataclass
class OpticsBlindResult:
    """
    Résultat de :func:`optics_blind_deconvolve`.

    Attributes
    ----------
    image : np.ndarray
        Image restaurée.
    psf : np.ndarray or list
        PSF estimée (une par tuile si l'image est découpée).
    params : dict or list of dict
        Paramètres optiques (fixés et estimés), par tuile le cas échéant.
    sigma : float
        Écart-type du bruit utilisé pour la restauration.
    tiles : list of (slice, slice) or None
    raw_params : list of dict or None
        Paramètres estimés par tuile **avant** lissage par le modèle de champ.
    """

    image: np.ndarray
    psf: object
    params: object
    sigma: float
    tiles: list | None = None
    raw_params: list | None = None


# ---------------------------------------------------------------------------
# Critère de vraisemblance marginale (Wiener–Hunt généralisé)
# ---------------------------------------------------------------------------

class _Evidence:
    """
    ``J(θ) = min_{β, μ} Σ_ω log s + M log(moyenne(P/s))``,
    ``s = 1 + |H_θ|² / (μ G^{β/2})``, ``G = 4sin²(πf_y) + 4sin²(πf_x)``,
    ``P = |Y|²/N`` (composante périodique de l'image, Moisan 2011).
    β = 4 : a priori laplacien de Hunt ; β = 2 : gradient. ``beta`` réel :
    exposant fixé ; ``None`` : estimé dans ``beta_bounds``.
    """

    def __init__(self, y: np.ndarray, beta: float | None = DEFAULT_PRIOR_EXPONENT, beta_bounds=(1.5, 4.5)):
        p, _ = perdecomp(y)
        Y = spfft.fft2(p - p.mean())
        fy = np.fft.fftfreq(y.shape[0])[:, None]
        fx = np.fft.fftfreq(y.shape[1])[None, :]
        G = 4 * np.sin(np.pi * fy) ** 2 + 4 * np.sin(np.pi * fx) ** 2
        self.keep = G > 0
        self.P = (np.abs(Y) ** 2 / y.size)[self.keep]
        self.logG = np.log(G[self.keep])
        self.shape = y.shape
        self.beta_bounds = (float(beta), float(beta)) if beta is not None else beta_bounds
        b0 = float(beta) if beta is not None else 3.0
        self.state = np.array([b0, np.log(np.mean(self.P)) - 5.0])   # (β, log μ) chaud

    def _inner(self, a2):
        P, logG, M = self.P, self.logG, self.P.size

        def f(z):
            beta, logmu = z
            u = a2 * np.exp(-logmu - 0.5 * beta * logG)
            s = 1.0 + u
            q = P / s
            mq = q.mean()
            val = np.sum(np.log(s)) + M * np.log(mq)
            dJds = 1.0 / s - (q / s) / mq            # ∂J/∂s_i
            g_logmu = np.sum(dJds * (-u))
            g_beta = np.sum(dJds * (-u) * 0.5 * logG)
            return val, np.array([g_beta, g_logmu])

        best = None
        lo_b, hi_b = self.beta_bounds
        for z0 in (self.state, np.array([np.clip(3.0, lo_b, hi_b), self.state[1]]),
                   np.array([lo_b, self.state[1] + 3.0])):
            r = optimize.minimize(f, z0, jac=True, method="L-BFGS-B",
                                  bounds=[self.beta_bounds, (-60.0, 30.0)])
            if best is None or r.fun < best.fun:
                best = r
        self.state = best.x
        return float(best.fun), best.x

    def __call__(self, psf: np.ndarray) -> float:
        a2 = np.abs(psf2otf(psf, self.shape)) ** 2
        return self._inner(a2[self.keep])[0]

    def sigma(self, psf: np.ndarray) -> float:
        """σ du bruit au maximum de vraisemblance marginale pour cette PSF."""
        a2 = (np.abs(psf2otf(psf, self.shape)) ** 2)[self.keep]
        _, (beta, logmu) = self._inner(a2)
        s = 1.0 + a2 * np.exp(-logmu - 0.5 * beta * self.logG)
        return float(np.sqrt(np.mean(self.P / s)))


def _fit_evidence(y, model: OpticalModel, n_starts: int, prior_std: dict | None, beta):
    crit = _Evidence(y, beta)
    penalty = _prior_penalty(model, prior_std)

    def J(theta):
        return crit(model.psf(theta)) + penalty(theta)

    best = None
    for s in model.starts(n_starts):
        r = optimize.minimize(J, s, method="L-BFGS-B", bounds=model.bounds,
                              options={"eps": 2e-3, "maxiter": 60})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, crit


def _prior_penalty(model: OpticalModel, prior_std: dict | None):
    """A priori gaussien optionnel sur θ, centré au milieu des bornes : ``½ Σ ((θ − θ₀)/τ)²``."""
    if not prior_std:
        return lambda theta: 0.0
    tau = {(_key(k)): float(v) for k, v in prior_std.items()}
    idx = [(i, tau[k], 0.5 * (lo + hi)) for i, (k, (lo, hi)) in enumerate(model._free) if k in tau]

    def pen(theta):
        return 0.5 * sum(((theta[i] - c) / t) ** 2 for i, t, c in idx)

    return pen


# ---------------------------------------------------------------------------
# Raffinement « L0 » (contours de la scène)
# ---------------------------------------------------------------------------

def _refine_l0(y_unit, model: OpticalModel, theta0, n_outer=6, lam_l0=4e-3, lam_min=1e-4):
    from .blind import _diff_otfs, _l0_image
    from ..operators import prepare_domain

    size = model.psf(theta0).shape[0]
    yw = prepare_domain(y_unit, (size, size), "reflect").image
    D = _diff_otfs(yw.shape)
    Yw = spfft.fft2(yw)
    gy = [d * Yw for d in D]
    theta, lam = np.asarray(theta0, float), lam_l0
    for _ in range(n_outer):
        x = _l0_image(Yw, psf2otf(model.psf(theta), yw.shape), D, lam)
        X = spfft.fft2(x)
        gx = [d * X for d in D]

        def J(t):
            H = psf2otf(model.psf(t), yw.shape)
            return float(sum(np.sum(np.abs(H * a - b) ** 2) for a, b in zip(gx, gy)) / yw.size)

        theta = optimize.minimize(J, theta, method="L-BFGS-B", bounds=model.bounds,
                                  options={"eps": 2e-3, "maxiter": 40}).x
        lam = max(lam * 0.9, lam_min)
    return theta


def estimate_optical_parameters(
    image: np.ndarray,
    model: OpticalModel,
    *,
    estimator: str = "evidence",
    n_starts: int = 3,
    prior_std: dict | None = None,
    prior_exponent: float | None = DEFAULT_PRIOR_EXPONENT,
    data_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """
    Paramètres libres ``θ`` du modèle optique pour une image (ou une
    tuile), et σ du bruit estimé au passage.

    Returns
    -------
    (θ, σ)
        ``θ`` dans l'ordre de ``model.names`` ; σ en unités de ``image``.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator doit valoir {ESTIMATORS}, reçu {estimator!r}.")
    y = check_image(image)
    yu, _, scale = to_unit_range(y, data_range)
    if estimator == "evidence":
        theta, crit = _fit_evidence(yu, model, n_starts, prior_std, prior_exponent)
    else:
        theta = _refine_l0(yu, model, model.starts(1)[0])
        crit = _Evidence(yu, prior_exponent)
    return theta, crit.sigma(model.psf(theta)) * scale


# ---------------------------------------------------------------------------
# Modèle de champ et recollement des tuiles
# ---------------------------------------------------------------------------

def _field_design(centres: np.ndarray, field_model: str) -> np.ndarray:
    u, v = centres.T
    cols = [np.ones_like(u)]
    if field_model in ("plane", "quadratic"):
        cols += [u, v]
    if field_model == "quadratic":
        cols += [u * u, u * v, v * v]
    return np.stack(cols, axis=1)


def _smooth_field(thetas: np.ndarray, centres: np.ndarray, field_model: str, bounds) -> np.ndarray:
    if field_model == "independent":
        return thetas
    A = _field_design(centres, field_model)
    if A.shape[1] > len(thetas):
        raise ValueError(f"Trop peu de tuiles ({len(thetas)}) pour un champ '{field_model}'.")
    coef, *_ = np.linalg.lstsq(A, thetas, rcond=None)
    lo, hi = np.array(bounds).T
    return np.clip(A @ coef, lo, hi)


def _hann_weight(shape) -> np.ndarray:
    wy = np.hanning(shape[0] + 2)[1:-1]
    wx = np.hanning(shape[1] + 2)[1:-1]
    return np.outer(wy, wx) + 1e-6


def optics_blind_deconvolve(
    image: np.ndarray,
    model: OpticalModel,
    sigma: float | None = None,
    *,
    estimator: str = "evidence",
    n_starts: int = 3,
    prior_std: dict | None = None,
    prior_exponent: float | None = DEFAULT_PRIOR_EXPONENT,
    tile_size: int | None = None,
    overlap: float = 0.5,
    field_model: str = "plane",
    restore: str = "tv_primal_dual",
    restore_kwargs: dict | None = None,
    data_range: tuple[float, float] | None = None,
) -> OpticsBlindResult:
    """
    Déconvolution aveugle contrainte par l'optique : estime les paramètres
    libres de ``model`` (globalement ou par tuile), puis restaure l'image.

    Parameters
    ----------
    image : np.ndarray
        Image floue 2D.
    model : OpticalModel
        Modèle de l'instrument : échantillonnage ``Q``, obturation,
        paramètres connus (``fixed``) et plages plausibles des paramètres
        estimés (``free``).
    sigma : float, optional
        Écart-type du bruit pour la restauration ; ``None`` : celui estimé
        par vraisemblance marginale (médiane des tuiles).
    estimator : {"evidence", "l0"}, default "evidence"
        Critère d'estimation de θ (voir le module).
    prior_exponent : float or None, default 3.0
        Exposant β du spectre a priori de la scène ``∝ |ω|^-β`` (critère
        « evidence ») ; ``None`` : estimé dans [1.5, 4.5]. β = 3 donne le
        plus petit écart moyen à la vraie PSF sur les essais du §8.2
        (β libre : petites défocalisations confondues avec la pente du
        spectre ; β = 4, laplacien de Hunt : défocalisations sous-estimées).
    n_starts : int, default 3
        Départs de l'optimisation (grille sur le premier paramètre libre) :
        le critère n'est pas convexe en θ.
    prior_std : dict, optional
        Écart-type d'un a priori gaussien sur certains paramètres, centré
        au milieu de leurs bornes (``estimator`` contenant ``"evidence"``).
    tile_size : int, optional
        Côté des tuiles (variation de la PSF dans le champ) ; ``None`` :
        une PSF pour toute l'image. Une tuile doit contenir assez de
        structures : 128 px au moins.
    overlap : float, default 0.5
        Recouvrement des tuiles.
    field_model : {"plane", "quadratic", "constant", "independent"}, default "plane"
        Lissage de θ entre tuiles : ajustement par moindres carrés d'un
        polynôme de la position (``"quadratic"`` pour une courbure de
        champ), moyenne (``"constant"``) ou aucun (``"independent"``).
    restore : str, default "tv_primal_dual"
        Méthode non aveugle de restauration finale.
    restore_kwargs : dict, optional
        Paramètres de ``restore`` (``sigma`` est fourni).
    data_range : tuple, optional
        Plage de l'image pour la normalisation (défaut : min/max).

    Returns
    -------
    OpticsBlindResult
    """
    from ..runner import deconvolve

    if field_model not in FIELD_MODELS:
        raise ValueError(f"field_model doit valoir {FIELD_MODELS}, reçu {field_model!r}.")
    y = check_image(image)
    kw = dict(restore_kwargs or {})
    if tile_size is None:
        theta, s_est = estimate_optical_parameters(y, model, estimator=estimator, n_starts=n_starts,
                                                   prior_std=prior_std, prior_exponent=prior_exponent,
                                                   data_range=data_range)
        s = resolve_sigma(y, s_est if sigma is None else sigma)
        h = model.psf(theta)
        restored = deconvolve(y, h, restore, sigma=s, **kw)
        return OpticsBlindResult(restored, h, model.as_dict(theta), s)

    tiles = tile_slices(y.shape, int(tile_size), overlap)
    rng = data_range if data_range is not None else (float(y.min()), float(y.max()))
    thetas, sigmas, centres = [], [], []
    for sl in tiles:
        t, st = estimate_optical_parameters(y[sl], model, estimator=estimator, n_starts=n_starts,
                                            prior_std=prior_std, prior_exponent=prior_exponent,
                                            data_range=rng)
        thetas.append(t)
        sigmas.append(st)
        centres.append([(sl[0].start + sl[0].stop) / 2 / y.shape[0] - 0.5,
                        (sl[1].start + sl[1].stop) / 2 / y.shape[1] - 0.5])
    thetas, centres = np.array(thetas), np.array(centres)
    smooth = _smooth_field(thetas, centres, field_model, model.bounds)
    s = resolve_sigma(y, float(np.median(sigmas)) if sigma is None else sigma)
    acc, wsum, psfs = np.zeros_like(y), np.zeros_like(y), []
    for sl, t in zip(tiles, smooth):
        h = model.psf(t)
        psfs.append(h)
        w = _hann_weight(y[sl].shape)
        acc[sl] += w * deconvolve(y[sl], h, restore, sigma=s, **kw)
        wsum[sl] += w
    return OpticsBlindResult(acc / wsum, psfs, [model.as_dict(t) for t in smooth], s, tiles,
                             [model.as_dict(t) for t in thetas])


def blind_optics_deconvolve(
    image: np.ndarray,
    psf: np.ndarray | None = None,
    sigma: float | None = None,
    *,
    sampling=None,
    free: dict | None = None,
    fixed: dict | None = None,
    obscuration: float = 0.0,
    weights=None,
    psf_size: int | None = None,
    callback=None,
    **kwargs,
) -> np.ndarray:
    """
    Signature commune (:func:`eoqual.deconvolution.runner.deconvolve`) de
    :func:`optics_blind_deconvolve` : le modèle optique se donne par
    mots-clés (``sampling`` obligatoire, ``free``, ``fixed``,
    ``obscuration``…), ``psf`` doit valoir ``None``. Ne renvoie que l'image ;
    pour la PSF et les paramètres estimés, appeler
    :func:`optics_blind_deconvolve`.

    >>> x = deconvolve(y, None, "blind_optics", sampling=1.0, obscuration=0.3,
    ...                free={"defocus": (0, 0.25), "astig_0": (-0.1, 0.1)},
    ...                fixed={"jitter": 0.3})
    """
    del callback
    if psf is not None:
        raise ValueError("blind_optics estime la PSF : passer psf=None et décrire l'instrument "
                         "(sampling, obscuration, free, fixed).")
    if sampling is None:
        raise ValueError("blind_optics nécessite sampling = λf/(Dp) de l'instrument.")
    model = OpticalModel(sampling, free if free is not None else {"defocus": (0.0, 0.25)},
                         fixed or {}, obscuration, weights, psf_size)
    return optics_blind_deconvolve(image, model, sigma, **kwargs).image
