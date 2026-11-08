#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Banc d'essai de ``wiener_hunt`` (REFERENCE_TECHNIQUE.md §3.5,
BENCHMARK.md §6) face aux autres méthodes linéaires, PSF connue.

    python examples/wiener_hunt_benchmark.py [IMAGE.tif]

Scènes 256×256 : « astronaut » (scikit-image) et, si ``IMAGE.tif`` est
donnée (orthophotographie RVB 8 bits), un extrait réduit d'un facteur 2.
"""
from __future__ import annotations

import sys

import numpy as np
from skimage import color, data

from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.backends.linear import wiener_hunt_parameters
from eoqual.deconvolution.psf import airy_psf, gaussian_psf, motion_psf, optical_psf
from eoqual.deconvolution.simulate import simulate_observation

PSFS = {"gauss1.6": gaussian_psf(1.6), "airy": airy_psf(2.5), "bougé": motion_psf(9, 0.4),
        "optique": optical_psf(1.0, zernike={4: 0.15}, obscuration=0.3, jitter_sigma=0.3)}
CONFIGS = [("tikhonov", {}), ("wiener", {}), ("wiener_unsupervised", {}), ("wiener_hunt", {}),
           ("wiener_hunt", {"sigma": "known"}), ("wiener_hunt", {"prior": "gradient"})]


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))


def main() -> None:
    scenes = {"astronaut": color.rgb2gray(data.astronaut())[100:356, 150:406].astype(np.float64)}
    if len(sys.argv) > 1:
        import tifffile

        rgb = tifffile.imread(sys.argv[1])[..., :3].astype(np.float64) / 255.0
        gray = (rgb**2.2) @ np.array([0.2126, 0.7152, 0.0722])
        n = min(gray.shape) // 2 * 2
        scenes["toulouse"] = gray[:n, :n].reshape(n // 2, 2, n // 2, 2).mean(axis=(1, 3))[200:456, 300:556]
    print("scène psf σ | tikhonov gcv | wiener | wiener_unsup | WH lap | WH lap σ connu | WH grad | σ estimé")
    for sn, x in scenes.items():
        for pn, h in PSFS.items():
            for sig in (0.003, 0.01, 0.03):
                y = simulate_observation(x, h, sig, boundary="reflect", seed=0)
                row = []
                for method, kw in CONFIGS:
                    kw = {k: (sig if v == "known" else v) for k, v in kw.items()}
                    row.append(f"{psnr(x, deconvolve(y, h, method, **kw)):.2f}")
                est = wiener_hunt_parameters(y, h)["sigma"]
                print(sn, pn, sig, "|", " | ".join(row), f"| {est:.4f}", flush=True)


if __name__ == "__main__":
    main()
