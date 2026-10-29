#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démarrage rapide : simule une observation floue et bruitée, la restaure
avec plusieurs méthodes et enregistre une figure HTML interactive
(``examples/output/quickstart.html``).

    python examples/quickstart.py

Image : « astronaut » (NASA, domaine public, fournie par scikit-image).
"""
import os
from pathlib import Path

from skimage import color, data

from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.plotting import plot_mtf, plot_restoration
from eoqual.deconvolution.psf import gaussian_psf
from eoqual.deconvolution.simulate import simulate_observation
from eoqual.metrics.fr.signal import psnr

OUT = Path(os.environ.get("EOQUAL_EXAMPLES_OUT", Path(__file__).with_name("output")))
SIZE = int(os.environ.get("EOQUAL_EXAMPLES_SIZE", "256"))


def main() -> None:
    scene = color.rgb2gray(data.astronaut())[100 : 100 + SIZE, 150 : 150 + SIZE]
    psf = gaussian_psf(2.0)
    sigma = 0.01
    blurred = simulate_observation(scene, psf, sigma, boundary="reflect", seed=0)

    results = {m: deconvolve(blurred, psf, m, sigma=sigma)
               for m in ("wiener", "tikhonov", "richardson_lucy", "tv_primal_dual")}
    print(f"observée        : {psnr(scene, blurred):6.2f} dB")
    for name, img in results.items():
        print(f"{name:16s}: {psnr(scene, img):6.2f} dB")

    OUT.mkdir(parents=True, exist_ok=True)
    plot_restoration(blurred, results, scene, ncols=3, title="Déconvolution, PSF gaussienne σ = 2 px").write_html(
        OUT / "quickstart.html")
    plot_mtf({"PSF gaussienne σ = 2 px": psf}).write_html(OUT / "quickstart_mtf.html")
    print(f"Figures : {OUT}")


if __name__ == "__main__":
    main()
