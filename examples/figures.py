#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schémas de la documentation, régénérables :

    python examples/figures.py          # écrit resources/*.png

- ``boundary_modes.png`` (REFERENCE_TECHNIQUE.md §2.1) : domaine de
  travail périodique des modes ``boundary="reflect"`` et
  ``boundary="unknown"`` — image observée (données connues) et marge
  (données inventées par miroir, ou inconnues).
- ``optical_psfs.png`` (REFERENCE_TECHNIQUE.md §8.2) : PSF du modèle
  optique de ``blind_optics`` (diffraction, défocalisation, astigmatisme,
  coma, jitter) et coupes de leur FTM sous la borne de diffraction.

Dépendance : matplotlib (installé avec eoqual-metrics).
"""
from __future__ import annotations

from pathlib import Path

RESOURCES = Path(__file__).resolve().parents[1] / "resources"


def boundary_modes(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    H, W, M = 6.0, 8.0, 2.4                                   # image H×W, marge totale M
    known, invented, unknown = "#cfe3f5", "#f6d7b0", "#e6e6e6"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=150)
    panels = [
        ("boundary = \"reflect\"", invented, "marge : miroir\nde l'image\n(données\ninventées)",
         "‖h ∗ x − y‖² imposé\nsur tout le domaine", "marge : miroir de l'image — attache imposée"),
        ("boundary = \"unknown\"", unknown, "marge :\ninconnue,\ncomplétée\npar h ∗ x",
         "‖h ∗ x − y‖² imposé\nsur l'image seulement", "marge : inconnue — aucune attache"),
    ]
    for ax, (title, margin_color, margin_text, rule, margin_rule) in zip(axes, panels):
        # domaine de travail (H + M) × (W + M), image en haut à gauche
        ax.add_patch(Rectangle((0, 0), W + M, H + M, facecolor=margin_color, edgecolor="black", lw=1.4,
                               hatch="//" if margin_color == invented else None))
        ax.add_patch(Rectangle((0, M), W, H, facecolor=known, edgecolor="black", lw=1.4))
        ax.text(W / 2, M + H / 2 + 0.55, "image observée y", ha="center", va="center", fontsize=12, weight="bold")
        ax.text(W / 2, M + H / 2 - 0.35, "H × W pixels : données connues", ha="center", va="center", fontsize=10)
        ax.text(W / 2, M + H / 2 - 1.25, rule, ha="center", va="center", fontsize=9, color="#1f4e79")
        box = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.9)
        ax.text(W + M / 2, M + H / 2, margin_text, ha="center", va="center", fontsize=8, bbox=box)
        ax.text((W + M) / 2, M / 2, margin_rule, ha="center", va="center", fontsize=8.5, bbox=box)
        # périodicité : le bord droit rejoint le bord gauche
        ax.add_patch(FancyArrowPatch((0.1, H + M + 0.2), (W + M - 0.1, H + M + 0.2), arrowstyle="<->",
                                     mutation_scale=12, color="dimgray", connectionstyle="arc3,rad=-0.15"))
        ax.text((W + M) / 2, H + M + 1.35, "domaine périodique (FFT) : les bords opposés se raccordent",
                ha="center", fontsize=8.5, color="dimgray")
        ax.text(W + M / 2, -0.35, "raccord périodique\ndans la marge", ha="center", va="top", fontsize=7.5,
                color="dimgray")
        ax.set_title(title, fontsize=12, family="monospace")
        ax.set_xlim(-0.4, W + M + 0.4)
        ax.set_ylim(-1.3, H + M + 1.8)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Traitement des bords : la scène est cherchée sur un domaine plus grand que l'image ; "
                 "seule la marge diffère", fontsize=10.5)
    fig.text(0.5, 0.02, "La solution x est inconnue partout ; elle est recadrée sur [:H, :W] à la fin. "
             "« unknown » n'impose rien dans la marge : elle n'est contrainte que par la régularisation.",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def optical_psfs(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from eoqual.core.kernel import psf2otf
    from eoqual.deconvolution.psf import optical_psf

    q, eps, size, n = 2.0, 0.3, 21, 256           # Q = 2 (Nyquist) pour bien voir la forme
    cases = [
        ("diffraction seule", {}, 0.0),
        ("défocalisation 0.15 onde RMS", {"defocus": 0.15}, 0.0),
        ("astigmatisme 0.1", {"astig_0": 0.1}, 0.0),
        ("coma 0.1", {"coma_x": 0.1}, 0.0),
        ("défoc. 0.1 + jitter 0.5 px", {"defocus": 0.1}, 0.5),
    ]
    fig, axes = plt.subplots(2, len(cases), figsize=(14, 5.8), dpi=130,
                             gridspec_kw={"height_ratios": [1, 0.9]})
    f = np.fft.fftfreq(n)[: n // 2]
    ref = np.abs(psf2otf(optical_psf(q, obscuration=eps, size=size), (n, n)))
    for k, (title, z, jit) in enumerate(cases):
        h = optical_psf(q, zernike=z, obscuration=eps, jitter_sigma=jit, size=size)
        ax = axes[0, k]
        ax.imshow(np.sqrt(h), cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        H = np.abs(psf2otf(h, (n, n)))
        ax = axes[1, k]
        ax.plot(f, ref[0, : n // 2], "k:", lw=1.2, label="diffraction (borne)")
        ax.plot(f, H[0, : n // 2], lw=1.6, label="|H| ligne")
        ax.plot(f, H[: n // 2, 0], lw=1.2, ls="--", label="|H| colonne")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("fréquence (cycles/pixel)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("module de la FTM", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Modèle optique de blind_optics : pupille annulaire (ε = 0.3), Q = λf/(Dp) = 2, pixel intégré "
                 "— PSF en √intensité, FTM jamais au-dessus de la diffraction", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    for name, func in (("boundary_modes.png", boundary_modes), ("optical_psfs.png", optical_psfs)):
        func(RESOURCES / name)
        print(f"Écrit : {RESOURCES / name}")


if __name__ == "__main__":
    main()
