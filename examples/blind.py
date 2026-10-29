#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Déconvolution aveugle : estime la PSF d'un flou de bougé et restaure
l'image, comparée à la restauration avec la vraie PSF.

    python examples/blind.py

Image : « astronaut » (NASA, domaine public, fournie par scikit-image).
"""
import os
from pathlib import Path

from skimage import color, data

from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.backends.blind import blind_deconvolve
from eoqual.deconvolution.plotting import plot_psf, plot_restoration
from eoqual.deconvolution.psf import motion_psf
from eoqual.deconvolution.simulate import simulate_observation

OUT = Path(os.environ.get("EOQUAL_EXAMPLES_OUT", Path(__file__).with_name("output")))
SIZE = int(os.environ.get("EOQUAL_EXAMPLES_SIZE", "256"))


def main() -> None:
    scene = color.rgb2gray(data.astronaut())[100 : 100 + SIZE, 150 : 150 + SIZE]
    truth, sigma = motion_psf(9, 0.5, 15), 0.01
    blurred = simulate_observation(scene, truth, sigma, boundary="reflect", seed=0)
    restored, psf_est = blind_deconvolve(blurred, psf_size=15, sigma=sigma)
    known = deconvolve(blurred, truth, "tv_primal_dual", sigma=sigma)
    OUT.mkdir(parents=True, exist_ok=True)
    plot_restoration(blurred, {"Aveugle (L0)": restored, "PSF connue (TV)": known}, scene).write_html(
        OUT / "blind.html")
    plot_psf({"Vraie PSF": truth, "PSF estimée": psf_est}).write_html(OUT / "blind_psf.html")
    print(f"Erreur L1 de la PSF : {abs(psf_est - truth).sum():.3f} (Dirac : 1.79)")
    print(f"Figures : {OUT}")


if __name__ == "__main__":
    main()
