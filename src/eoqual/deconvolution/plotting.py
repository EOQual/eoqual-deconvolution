#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures propres à la déconvolution (Plotly, via eoqual-viz) : comparaison
d'images, PSF et FTM, convergence, courbe de réglage, historique
d'optimisation.

Chaque fonction renvoie une ``plotly.graph_objects.Figure`` sans
l'afficher (``fig.show()`` ou ``fig.write_html(...)`` à l'appelant).
Les tracés génériques (galerie, courbes) sont dans eoqual-viz ; ce module
ne fait que préparer les données.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from eoqual.core.kernel import psf2otf

__all__ = [
    "plot_restoration",
    "plot_psf",
    "plot_mtf",
    "plot_convergence",
    "plot_gcv",
    "plot_optimization",
    "ConvergenceRecorder",
]


def plot_restoration(
    observed: np.ndarray,
    restored: np.ndarray | Mapping[str, np.ndarray],
    reference: np.ndarray | None = None,
    *,
    ncols: int | None = None,
    title: str = "",
    show_scores: bool = True,
):
    """
    Galerie : image observée, une ou plusieurs restaurations, référence.

    Avec ``reference`` et ``show_scores``, le PSNR (eoqual-metrics) est
    ajouté au titre de chaque image.
    """
    from eoqual.viz import plot_image_grid

    items = dict(restored) if isinstance(restored, Mapping) else {"Restaurée": restored}
    panels = [("Observée", observed)] + list(items.items())
    if reference is not None:
        if show_scores:
            from eoqual.metrics.fr.signal import psnr

            panels = [(f"{name} — {psnr(reference, img):.2f} dB", img) for name, img in panels]
        panels.append(("Référence", reference))
    return plot_image_grid(panels, ncols=ncols, title=title)


def plot_psf(psfs: np.ndarray | Mapping[str, np.ndarray], *, log: bool = False, title: str = "PSF"):
    """PSF côte à côte (vraie, estimée...), chacune à sa propre échelle."""
    from eoqual.viz import plot_image_grid

    items = dict(psfs) if isinstance(psfs, Mapping) else {"PSF": psfs}
    if log:
        items = {k: np.log10(np.maximum(v, 1e-8 * np.max(v))) for k, v in items.items()}
    return plot_image_grid(list(items.items()), shared_scale=False, cmap="viridis", title=title)


def plot_mtf(
    psfs: np.ndarray | Mapping[str, np.ndarray],
    *,
    n: int = 256,
    axis: str = "x",
    title: str = "FTM",
):
    """
    Coupe de la FTM ``|H|`` de chaque PSF, de 0 à 0.5 cycle/pixel
    (Nyquist), selon ``axis`` (``"x"`` : horizontale, ``"y"`` : verticale).
    """
    from eoqual.viz import plot_curves

    items = dict(psfs) if isinstance(psfs, Mapping) else {"PSF": psfs}
    f = np.arange(n // 2 + 1) / n
    curves = []
    for name, h in items.items():
        mtf = np.abs(psf2otf(np.asarray(h) / np.sum(h), (n, n)))
        cut = mtf[0, : n // 2 + 1] if axis == "x" else mtf[: n // 2 + 1, 0]
        curves.append((name, f, cut))
    return plot_curves(curves, xlabel="Fréquence (cycle/pixel)", ylabel="FTM", title=title,
                       vlines=[(0.5, "Nyquist", "gray")])


class ConvergenceRecorder:
    """
    Rappel (``callback``) qui enregistre un score à chaque itération.

    L'itéré reçu est sur le domaine de travail : avec ``boundary="reflect"``
    ou ``"periodic"``, l'image d'origine en occupe le coin ``[:H, :W]``.

    >>> rec = ConvergenceRecorder(reference)          # PSNR
    >>> deconvolve(y, psf, "richardson_lucy", n_iter=100, callback=rec)
    >>> plot_convergence({"RL": rec.values}).show()
    """

    def __init__(self, reference: np.ndarray, metric: str = "psnr"):
        from .tuning.optimizer import score_image

        self.reference = np.asarray(reference)
        self.metric = metric
        self._score = score_image
        self.values: list[float] = []

    def __call__(self, k: int, x: np.ndarray) -> None:
        h, w = self.reference.shape
        self.values.append(self._score(self.metric, x[:h, :w], self.reference))


def plot_convergence(
    histories: Mapping[str, Sequence[float]],
    *,
    ylabel: str = "PSNR (dB)",
    log_x: bool = False,
    title: str = "Convergence",
):
    """Une courbe par méthode : score en fonction de l'itération."""
    from eoqual.viz import plot_curves

    curves = [(name, np.arange(1, len(v) + 1), np.asarray(v)) for name, v in histories.items()]
    return plot_curves(curves, log_x=log_x, xlabel="Itération", ylabel=ylabel, title=title)


def plot_gcv(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    regularizer: str = "laplacian",
    sigma: float | None = None,
    boundary: str = "reflect",
    title: str = "Choix de λ (Tikhonov)",
):
    """
    Courbe ``GCV(λ)`` normalisée, et résidu par pixel rapporté à ``σ²``
    si ``sigma`` est donné ; repères aux λ choisis.
    """
    from eoqual.viz import plot_curves

    from .tuning.criteria import discrepancy_curve, gcv_curve

    lams, g, lam_gcv = gcv_curve(image, psf, regularizer=regularizer, boundary=boundary)
    curves = [("GCV / min", lams, g / g.min())]
    vlines = [(lam_gcv, "GCV", "crimson")]
    if sigma is not None:
        _, res, s2, lam_d = discrepancy_curve(image, psf, sigma, regularizer=regularizer, boundary=boundary)
        curves.append(("résidu / σ²", lams, res / s2))
        vlines.append((lam_d, "discrépance", "royalblue"))
    return plot_curves(curves, log_x=True, log_y=True, xlabel="λ", ylabel="(sans dimension)",
                       title=title, vlines=vlines)


def plot_optimization(result, *, title: str | None = None):
    """Score de chaque évaluation et meilleur score cumulé d'une optimisation."""
    from eoqual.viz import plot_curves

    scores = np.array([h["score"] if h["score"] is not None else np.nan for h in result.history])
    k = np.arange(1, len(scores) + 1)
    best = np.fmax.accumulate(np.where(np.isnan(scores), -np.inf, scores))
    return plot_curves([("évaluation", k, scores), ("meilleur", k, best)], markers=True,
                       xlabel="Évaluation", ylabel=result.metric,
                       title=title or f"Optimisation de {result.method}")
