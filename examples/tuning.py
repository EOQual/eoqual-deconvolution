#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Réglage des paramètres :

1. Tikhonov : courbe GCV et principe de discrépance (sans référence) ;
2. variation totale : optimisation bayésienne du poids sur une image de
   réglage (moitié gauche) avec référence, évaluée sur l'autre moitié ;
3. même optimisation sans référence (indice de netteté SI).

    python examples/tuning.py      # extra "optim" requis pour 2 et 3

Image : « astronaut » (NASA, domaine public, fournie par scikit-image).
"""
import os
from pathlib import Path

from skimage import color, data

from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.plotting import plot_gcv, plot_optimization
from eoqual.deconvolution.psf import motion_psf
from eoqual.deconvolution.simulate import simulate_observation
from eoqual.metrics.fr.signal import psnr

OUT = Path(os.environ.get("EOQUAL_EXAMPLES_OUT", Path(__file__).with_name("output")))
SIZE = int(os.environ.get("EOQUAL_EXAMPLES_SIZE", "256"))
N_CALLS = int(os.environ.get("EOQUAL_EXAMPLES_N_CALLS", "15"))


def main() -> None:
    gray = color.rgb2gray(data.astronaut())
    tune, test = gray[100 : 100 + SIZE, 0:SIZE], gray[100 : 100 + SIZE, 256 : 256 + SIZE]
    psf, sigma = motion_psf(9, 0.5), 0.01
    y_tune = simulate_observation(tune, psf, sigma, boundary="reflect", seed=0)
    y_test = simulate_observation(test, psf, sigma, boundary="reflect", seed=1)
    OUT.mkdir(parents=True, exist_ok=True)

    plot_gcv(y_test, psf, sigma=sigma).write_html(OUT / "tuning_gcv.html")
    for rule in ("gcv", "discrepancy"):
        out = deconvolve(y_test, psf, "tikhonov", sigma=sigma, rule=rule)
        print(f"tikhonov ({rule:11s}) : {psnr(test, out):6.2f} dB")

    try:
        from eoqual.deconvolution.tuning import optimize_parameters

        for ref, label in ((tune, "avec référence"), (None, "sans référence (SI)")):
            res = optimize_parameters(y_tune, psf, "tv_primal_dual", reference=ref, sigma=sigma,
                                      n_calls=N_CALLS, fixed={"n_iter": 150})
            out = deconvolve(y_test, psf, "tv_primal_dual", **res.best_kwargs)
            default = deconvolve(y_test, psf, "tv_primal_dual", sigma=sigma)
            print(f"tv_primal_dual {label:20s} : lam = {res.best_params['lam']:.3f}, "
                  f"évaluation {psnr(test, out):6.2f} dB (défaut {psnr(test, default):6.2f} dB)")
            plot_optimization(res).write_html(OUT / f"tuning_{res.metric}.html")
    except ImportError as exc:
        print(f"Optimisation ignorée : {exc}")
    print(f"Figures : {OUT}")


if __name__ == "__main__":
    main()
