#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark des méthodes (tableaux de BENCHMARK.md).

- image : « astronaut » 512×512 en niveaux de gris, dynamique [0, 1]
  (NASA, domaine public, fournie par scikit-image) ;
- **moitié gauche** (512×256) : image de **réglage** (optimisation) ;
  **moitié droite** (512×256) : image d'**évaluation** ;
- flous : gaussien σ = 2 px, Airy échantillonné à Nyquist (rayon 2.44 px),
  défocalisation (disque de rayon 3 px), bougé (9 px à 30°) ;
  convolution avec prolongement miroir (bords réalistes) ;
- bruit gaussien σ = 0.01 (connu des méthodes) ;
- scores PSNR et SSIM d'eoqual-metrics sur l'image d'évaluation.

    python examples/benchmark.py [--optimize] [--out BENCHMARK_tables.md]

``--optimize`` (extra ``optim``) : optimisation bayésienne du PSNR sur
l'image de réglage (``N_CALLS`` évaluations), puis évaluation des
paramètres trouvés sur l'autre moitié.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from skimage import color, data

from eoqual.deconvolution import METHODS_CONFIGS, deconvolve
from eoqual.deconvolution.psf import airy_psf, disk_psf, gaussian_psf, motion_psf
from eoqual.deconvolution.simulate import simulate_observation
from eoqual.metrics.fr.signal import psnr
from eoqual.metrics.fr.ssim import ssim

SIGMA = 0.01
N_CALLS = 15
PSFS = {
    "gaussienne σ=2": gaussian_psf(2.0),
    "Airy (Nyquist)": airy_psf(2.44),
    "défocus r=3": disk_psf(3.0),
    "bougé 9 px": motion_psf(9, np.pi / 6),
}
OPTIMIZED = ["inverse", "wiener", "tikhonov", "richardson_lucy", "landweber", "fista_wavelet",
             "tv_primal_dual", "pnp_hqs"]


def _scores(ref, img):
    return psnr(ref, img), float(ssim(ref, img, algo="skimage"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    gray = color.rgb2gray(data.astronaut()).astype(np.float64)
    tune, test = gray[:, :256], gray[:, 256:]
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    obs = {}
    rows = {m: {} for m in METHODS_CONFIGS}
    rows["tikhonov (discrépance)"] = {}
    times = {m: [] for m in rows}
    for pname, psf in PSFS.items():
        y_test = simulate_observation(test, psf, SIGMA, boundary="reflect", seed=1)
        obs[pname] = _scores(test, y_test)
        for m in rows:
            method, kw = (m, {}) if m in METHODS_CONFIGS else ("tikhonov", {"rule": "discrepancy"})
            k = psf if method != "blind_l0" else None
            if method == "blind_l0":
                kw["psf_size"] = psf.shape[0]  # support connu, PSF inconnue
            t0 = time.perf_counter()
            out = deconvolve(y_test, k, method, sigma=SIGMA, **kw)
            times[m].append(time.perf_counter() - t0)
            rows[m][pname] = _scores(test, out)

    head = "| Méthode | " + " | ".join(PSFS) + " | Temps (s) |"
    emit("### Réglages par défaut — PSNR (dB) / SSIM")
    emit()
    emit(head)
    emit("|" + "---|" * (len(PSFS) + 2))
    emit("| *(observée)* | " + " | ".join(f"{p:.2f} / {s:.3f}" for p, s in obs.values()) + " | — |")
    for m, r in rows.items():
        emit(f"| `{m}` | " + " | ".join(f"{p:.2f} / {s:.3f}" for p, s in r.values())
             + f" | {np.mean(times[m]):.1f} |")

    if args.optimize:
        from eoqual.deconvolution.tuning import optimize_parameters

        emit()
        emit(f"### Réglages optimisés ({N_CALLS} évaluations sur l'image de réglage) — PSNR (dB) sur l'image d'évaluation")
        emit()
        emit("| Méthode | " + " | ".join(PSFS) + " |")
        emit("|" + "---|" * (len(PSFS) + 1))
        best_params = {}
        for m in OPTIMIZED:
            cells = []
            for pname, psf in PSFS.items():
                y_tune = simulate_observation(tune, psf, SIGMA, boundary="reflect", seed=0)
                y_test = simulate_observation(test, psf, SIGMA, boundary="reflect", seed=1)
                n_calls = 10 if m == "pnp_hqs" else N_CALLS
                res = optimize_parameters(y_tune, psf, m, reference=tune, sigma=SIGMA, n_calls=n_calls)
                p = psnr(test, deconvolve(y_test, psf, m, **res.best_kwargs))
                cells.append(f"{p:.2f} ({p - rows[m][pname][0]:+.2f})")
                best_params[(m, pname)] = res.best_params
            emit(f"| `{m}` | " + " | ".join(cells) + " |")
        emit()
        emit("Paramètres trouvés :")
        emit()
        for (m, pname), prm in best_params.items():
            txt = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in prm.items())
            emit(f"- `{m}`, {pname} : {txt}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
