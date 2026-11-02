#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutoriel pas à pas (TUTORIEL.md) : chaque étape affiche les chiffres
cités dans le texte et enregistre ses figures HTML dans
``examples/output/tutorial_*.html``.

    python examples/tutorial.py            # extra "pnp" pour l'étape 6

Image : « astronaut » (NASA, domaine public, fournie par scikit-image).
"""
import os
from pathlib import Path

import numpy as np
from skimage import color, data

from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.plotting import (
    ConvergenceRecorder, plot_convergence, plot_gcv, plot_mtf, plot_psf, plot_restoration,
)
from eoqual.deconvolution.psf import gaussian_psf, motion_psf, system_psf
from eoqual.deconvolution.simulate import add_noise, blur, simulate_observation
from eoqual.metrics.fr.signal import psnr

OUT = Path(os.environ.get("EOQUAL_EXAMPLES_OUT", Path(__file__).with_name("output")))
SIZE = int(os.environ.get("EOQUAL_EXAMPLES_SIZE", "256"))
SIGMA = 0.01


def _p(scene, img):
    return f"{psnr(scene, img):5.2f} dB"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = color.rgb2gray(data.astronaut())[100 : 100 + SIZE, 150 : 150 + SIZE]

    print("1. Modèle direct")
    psf = gaussian_psf(2.0)
    blurred = simulate_observation(scene, psf, SIGMA, boundary="reflect", seed=0)
    print(f"   observée : {_p(scene, blurred)}")
    plot_mtf({"gaussienne σ = 2 px": psf, "bougé 9 px": motion_psf(9, np.pi / 6)}).write_html(
        OUT / "tutorial_1_mtf.html")

    print("2. Inversion directe")
    for t in (0.001, 0.01, 0.05, 0.2):
        print(f"   seuil {t:5.3f} : {_p(scene, deconvolve(blurred, psf, 'inverse', threshold=t))}")

    print("3. Régularisation linéaire")
    lin = {
        "Wiener": deconvolve(blurred, psf, "wiener", sigma=SIGMA),
        "Tikhonov (GCV)": deconvolve(blurred, psf, "tikhonov"),
        "Tikhonov (discrépance)": deconvolve(blurred, psf, "tikhonov", sigma=SIGMA, rule="discrepancy"),
    }
    for k, v in lin.items():
        print(f"   {k:22s} : {_p(scene, v)}")
    plot_gcv(blurred, psf, sigma=SIGMA).write_html(OUT / "tutorial_3_gcv.html")

    print("4. Itérations et semi-convergence")
    curves = {}
    for name, kw in (("Richardson-Lucy", {"acceleration": False, "tv_weight": 0.0}),
                     ("RL accéléré", {"tv_weight": 0.0}),
                     ("RL accéléré + TV (défaut)", {})):
        rec = ConvergenceRecorder(scene)
        deconvolve(blurred, psf, "richardson_lucy", n_iter=80, callback=rec, **kw)
        curves[name] = rec.values
        best = int(np.argmax(rec.values))
        print(f"   {name:16s} : meilleur {rec.values[best]:5.2f} dB à l'itération {best + 1},"
              f" {rec.values[-1]:5.2f} dB à la 80ᵉ")
    plot_convergence(curves).write_html(OUT / "tutorial_4_convergence.html")

    print("5. A priori parcimonieux")
    sparse = {
        "FISTA ondelettes": deconvolve(blurred, psf, "fista_wavelet", sigma=SIGMA),
        "Variation totale": deconvolve(blurred, psf, "tv_primal_dual", sigma=SIGMA),
    }
    for k, v in sparse.items():
        print(f"   {k:22s} : {_p(scene, v)}")
    plot_restoration(blurred, {**lin, **sparse}, scene, ncols=3).write_html(OUT / "tutorial_5_gallery.html")

    print("6. Plug-and-Play")
    try:
        for den in ("tv_chambolle", "nl_ridge"):
            out = deconvolve(blurred, psf, "pnp_hqs", sigma=SIGMA, denoiser=den)
            print(f"   débruiteur {den:12s} : {_p(scene, out)}")
    except ImportError as exc:
        print(f"   ignoré : {exc}")

    print("7. Données réelles")
    mpsf = motion_psf(9, np.pi / 6)
    y = simulate_observation(scene, mpsf, SIGMA, boundary="reflect", seed=0)
    for b in ("periodic", "reflect", "unknown"):
        print(f"   bougé, TV, boundary={b:8s} : {_p(scene, deconvolve(y, mpsf, 'tv_primal_dual', sigma=SIGMA, boundary=b))}")
    yb = blur(scene, psf, "reflect")
    level = 0.85 * yb.max()
    ysat = np.minimum(add_noise(yb, SIGMA, seed=0), level)
    print(f"   saturation ({(ysat >= level).mean() * 100:.1f} % des pixels) : TV "
          f"{_p(scene, deconvolve(ysat, psf, 'tv_primal_dual', sigma=SIGMA))} → "
          f"{_p(scene, deconvolve(ysat, psf, 'tv_primal_dual', sigma=SIGMA, saturation=level))} avec saturation=")
    mask = np.ones_like(blurred, dtype=bool)
    mask[::20, :] = False
    dead = np.where(mask, blurred, 0.0)
    print(f"   lignes mortes : TV {_p(scene, deconvolve(dead, psf, 'tv_primal_dual', sigma=SIGMA))} → "
          f"{_p(scene, deconvolve(dead, psf, 'tv_primal_dual', sigma=SIGMA, mask=mask))} avec mask=")
    focal = 2 * 0.3 * 11e-6 / 0.55e-6  # Nyquist : λ f / (D p) = 2
    sys_psf = system_psf(0.55e-6, 0.3, focal, 11e-6, obscuration=0.3, motion_length=1.0)
    plot_psf({"PSF système": sys_psf}, log=True).write_html(OUT / "tutorial_7_system_psf.html")
    plot_mtf({"PSF système": sys_psf, "gaussienne σ = 2 px": psf}).write_html(OUT / "tutorial_7_mtf.html")
    ys = simulate_observation(scene, sys_psf, SIGMA, boundary="reflect", seed=0)
    print(f"   PSF système ({sys_psf.shape[0]} px) : observée {_p(scene, ys)}, TV "
          f"{_p(scene, deconvolve(ys, sys_psf, 'tv_primal_dual', sigma=SIGMA))}")
    print(f"Figures : {OUT}")


if __name__ == "__main__":
    main()
