#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schémas de la documentation, régénérables :

    python examples/figures.py          # écrit resources/boundary_modes.png

- ``boundary_modes.png`` (REFERENCE_TECHNIQUE.md §2.1) : domaine de
  travail périodique des modes ``boundary="reflect"`` et
  ``boundary="unknown"`` — image observée (données connues) et marge
  (données inventées par miroir, ou inconnues).

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


def main() -> None:
    boundary_modes(RESOURCES / "boundary_modes.png")
    print(f"Écrit : {RESOURCES / 'boundary_modes.png'}")


if __name__ == "__main__":
    main()
