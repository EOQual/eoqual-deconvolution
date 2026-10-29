#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimisation bayésienne des paramètres d'une méthode de déconvolution
(extra ``optim`` : scikit-optimize, BSD-3-Clause).

Deux usages :

- **avec image de référence** (simulation, mire) : ``metric="psnr"`` ou
  ``"ssim"`` ; les réglages obtenus se transposent à des images de même
  nature (même capteur, même PSF, même bruit) ;
- **sans référence** (image réelle) : ``metric="sharpness_index"``, indice
  de netteté SI de Blanchet & Moisan (2012), qui croît avec la netteté et
  décroît avec le bruit et les artefacts (Leclaire & Moisan 2015 l'emploient
  pour régler une déconvolution). Indicatif : vérifier visuellement.

Les métriques viennent d'eoqual-metrics. Processus gaussien
(``skopt.gp_minimize``), point de départ = réglages par défaut, donc
l'optimisation ne fait jamais pire que les défauts sur l'image de réglage.

>>> res = optimize_parameters(blurred, psf, "tv_primal_dual", reference=scene, sigma=0.01)
>>> res.best_kwargs, res.best_score, res.gain
>>> restored = deconvolve(other, psf, "tv_primal_dual", **res.best_kwargs)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..runner import get_method
from ..utils import resolve_sigma
from .search_spaces import get_search_space, validate_search_space

__all__ = ["METRICS", "OptimizationResult", "optimize_parameters", "score_image"]

#: Score renvoyé (à minimiser, donc pire que tout score réel) quand un
#: essai échoue ou produit une sortie invalide.
PENALTY = 1e6


def _psnr(ref, img):
    from eoqual.metrics.fr.signal import psnr

    return psnr(ref, img)


def _ssim(ref, img):
    from eoqual.metrics.fr.ssim import ssim

    return float(ssim(ref, img, algo="skimage"))


def _si(_ref, img):
    from eoqual.metrics.nr.sharpness.backends.phase.sharpness_index import sharpness_index

    return float(sharpness_index(img))


#: Métriques disponibles : nom → (fonction(ref, image), référence requise).
METRICS: dict[str, tuple[Callable[[Any, np.ndarray], float], bool]] = {
    "psnr": (_psnr, True),
    "ssim": (_ssim, True),
    "sharpness_index": (_si, False),
}


def score_image(metric: str, image: np.ndarray, reference: np.ndarray | None = None) -> float:
    """Score ``metric`` de ``image`` (plus grand = meilleur)."""
    if metric not in METRICS:
        raise ValueError(f"metric doit valoir {list(METRICS)}, reçu {metric!r}.")
    func, needs_ref = METRICS[metric]
    if needs_ref and reference is None:
        raise ValueError(f"metric={metric!r} nécessite une image de référence.")
    return float(func(reference, image))


@dataclass
class OptimizationResult:
    """
    Résultat de :func:`optimize_parameters`.

    Attributes
    ----------
    method, metric : str
    best_params : dict
        Meilleur point de l'espace de recherche.
    best_kwargs : dict
        Arguments complets à repasser à ``deconvolve`` (``sigma`` inclus).
    best_score, initial_score : float
        Score du meilleur point et du point de départ (réglages par défaut).
    history : list of dict
        Un ``{"params", "score", "status", "seconds"}`` par évaluation.
    """

    method: str
    metric: str
    best_params: dict
    best_kwargs: dict
    best_score: float
    initial_score: float | None
    history: list = field(default_factory=list)

    @property
    def gain(self) -> float | None:
        """Amélioration du score par rapport aux réglages par défaut."""
        return None if self.initial_score is None else self.best_score - self.initial_score


def _build_dimensions(dims):
    from skopt.space import Categorical, Integer, Real

    out = []
    for d in dims:
        prior = d.get("prior", "uniform")
        if d["type"] == "int":
            out.append(Integer(d["low"], d["high"], prior=prior, name=d["name"]))
        elif d["type"] == "float":
            out.append(Real(d["low"], d["high"], prior=prior, name=d["name"]))
        else:
            out.append(Categorical(d["categories"], name=d["name"]))
    return out


def _to_kwargs(params: dict, fixed: dict, sigma: float) -> dict:
    p = dict(params)
    scale = p.pop("sigma_scale", 1.0)
    return {**fixed, **p, "sigma": sigma * scale}


def optimize_parameters(
    blurred: np.ndarray,
    psf: np.ndarray,
    method: str,
    *,
    reference: np.ndarray | None = None,
    metric: str | None = None,
    sigma: float | None = None,
    n_calls: int = 25,
    space: dict | None = None,
    fixed: dict | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Cherche les paramètres de ``method`` qui maximisent ``metric``.

    Parameters
    ----------
    blurred : np.ndarray
        Image floue de réglage (2D).
    psf : np.ndarray
        PSF (PSF initiale pour ``blind_l0``).
    method : str
        Méthode (voir ``list_methods()``).
    reference : np.ndarray, optional
        Image nette de référence, requise par ``"psnr"`` et ``"ssim"``.
    metric : str, optional
        ``"psnr"``, ``"ssim"`` ou ``"sharpness_index"``. Défaut : ``"psnr"``
        si ``reference`` est donnée, ``"sharpness_index"`` sinon.
    sigma : float, optional
        Écart-type du bruit ; ``None`` : estimé une fois sur ``blurred``.
    n_calls : int, default 25
        Nombre d'évaluations (une déconvolution chacune). Compter 10 par
        dimension de l'espace de recherche.
    space : dict, optional
        Espace de recherche (format de :mod:`.search_spaces`) ; défaut :
        ``spaces/<method>.jsonc``.
    fixed : dict, optional
        Paramètres fixes supplémentaires (prioritaires sur ceux de l'espace).
    seed : int, default 0
    verbose : bool, default False

    Raises
    ------
    ImportError
        Si scikit-optimize n'est pas installé (extra ``optim``).
    """
    try:
        from skopt import gp_minimize
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            'optimize_parameters nécessite scikit-optimize : pip install "eoqual-deconvolution[optim]".'
        ) from exc

    metric = metric or ("psnr" if reference is not None else "sharpness_index")
    if metric not in METRICS:
        raise ValueError(f"metric doit valoir {list(METRICS)}, reçu {metric!r}.")
    if METRICS[metric][1] and reference is None:
        raise ValueError(f"metric={metric!r} nécessite une image de référence.")
    func = get_method(method)
    spc = validate_search_space(space, method) if space is not None else get_search_space(method)
    dims = spc["search_space"]
    fixed_all = {**spc.get("fixed", {}), **(fixed or {})}
    s = resolve_sigma(blurred, sigma)
    names = [d["name"] for d in dims]
    history: list[dict] = []

    def evaluate(point) -> float:
        params = dict(zip(names, point))
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
        t0 = time.perf_counter()
        try:
            out = func(blurred, psf, **_to_kwargs(params, fixed_all, s))
            if out.shape != np.shape(blurred) or not np.all(np.isfinite(out)):
                raise ValueError("sortie invalide (forme ou valeurs non finies)")
            score = score_image(metric, out, reference)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - un essai raté ne doit pas arrêter la recherche
            score, status = None, f"échec : {type(exc).__name__}: {str(exc)[:120]}"
        history.append({"params": params, "score": score, "status": status,
                        "seconds": time.perf_counter() - t0})
        if verbose:
            print(f"[{len(history):3d}/{n_calls}] {params} → {score} ({status})")
        return PENALTY if score is None else -score

    x0 = [spc["initial_guess"][n] for n in names] if "initial_guess" in spc else None
    gp_minimize(evaluate, _build_dimensions(dims), n_calls=n_calls, x0=x0,
                n_initial_points=min(max(n_calls // 3, 1), 10), random_state=seed)
    ok = [h for h in history if h["score"] is not None]
    if not ok:
        raise RuntimeError(f"Toutes les évaluations ont échoué : {history[-1]['status']}")
    best = max(ok, key=lambda h: h["score"])
    initial = history[0]["score"] if x0 is not None else None
    return OptimizationResult(method, metric, best["params"], _to_kwargs(best["params"], fixed_all, s),
                              best["score"], initial, history)
