#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Banc d'essai d'USRNet (REFERENCE_TECHNIQUE.md §7.2, BENCHMARK.md §8) :
poids publiés (KAIR, appris sur photographies) face aux méthodes du
paquet, PSF connue, puis sensibilité à une PSF fausse.

    python examples/usrnet_benchmark.py [IMAGE.tif] [--weights usrnet|usrnet_tiny|CHEMIN]

Scènes 256×256 : « astronaut » (scikit-image) et, si IMAGE.tif est donnée
(orthophotographie RVB 8 bits), un extrait réduit d'un facteur 2.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from skimage import color, data

from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.psf import airy_psf, gaussian_psf, motion_psf, optical_psf
from eoqual.deconvolution.simulate import simulate_observation

PSFS = {"gauss1.6": gaussian_psf(1.6), "airy": airy_psf(2.5), "bougé": motion_psf(9, 0.4),
        "optique": optical_psf(1.0, zernike={4: 0.15}, obscuration=0.3, jitter_sigma=0.3)}
#: PSF fausses : (vraie, utilisée pour la restauration)
ERREURS = {"gauss 1.6 → 1.8": (gaussian_psf(1.6), gaussian_psf(1.8)),
           "défoc. 0.15 → 0.10": (optical_psf(1.0, zernike={4: 0.15}, obscuration=0.3, jitter_sigma=0.3),
                                  optical_psf(1.0, zernike={4: 0.10}, obscuration=0.3, jitter_sigma=0.3))}


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))


def scenes(path=None):
    out = {"astronaut": color.rgb2gray(data.astronaut())[100:356, 150:406].astype(np.float64)}
    if path:
        import tifffile

        rgb = tifffile.imread(path)[..., :3].astype(np.float64) / 255.0
        gray = (rgb**2.2) @ np.array([0.2126, 0.7152, 0.0722])
        n = min(gray.shape) // 2 * 2
        out["toulouse"] = gray[:n, :n].reshape(n // 2, 2, n // 2, 2).mean(axis=(1, 3))[200:456, 300:556]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--weights", default="usrnet")
    args = ap.parse_args()
    methods = [("wiener_hunt", {}), ("tv_primal_dual", {}), ("pnp_hqs", {}),
               ("usrnet", {"weights": args.weights}), ("usrnet", {"weights": "usrnet_tiny"})]
    names = ["wiener_hunt", "tv", "pnp_hqs", f"usrnet[{args.weights}]", "usrnet_tiny"]
    print("| scène | flou | σ | observée | " + " | ".join(names) + " | temps usrnet |")
    print("|---" * (len(names) + 5) + "|")
    for sn, x in scenes(args.image).items():
        for pn, h in PSFS.items():
            for sig in (0.003, 0.01, 0.03):
                y = simulate_observation(x, h, sig, boundary="reflect", seed=0)
                row, dt = [], 0.0
                for m, kw in methods:
                    t = time.time()
                    row.append(f"{psnr(x, deconvolve(y, h, m, sigma=sig, **kw)):.2f}")
                    if m == "usrnet" and kw["weights"] == args.weights:
                        dt = time.time() - t
                print(f"| {sn} | {pn} | {sig} | {psnr(x, y):.2f} | " + " | ".join(row) + f" | {dt:.1f} s |", flush=True)
    print("\n| scène | PSF (vraie → utilisée) | σ | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 3) + "|")
    for sn, x in scenes(args.image).items():
        for en, (hv, hu) in ERREURS.items():
            y = simulate_observation(x, hv, 0.01, boundary="reflect", seed=0)
            row = [f"{psnr(x, deconvolve(y, hu, m, sigma=0.01, **kw)):.2f}" for m, kw in methods]
            print(f"| {sn} | {en} | 0.01 | " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
