#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Banc d'essai de ``blind_optics`` (REFERENCE_TECHNIQUE.md §8.2,
BENCHMARK.md) : PSNR après restauration TV avec la PSF estimée, comparé à
la vraie PSF, à la PSF nominale (sans aberration) et à ``blind_l0``.

    python examples/blind_optics_benchmark.py [IMAGE.tif]

Scènes : « astronaut » (scikit-image) et, si ``IMAGE.tif`` est donnée
(orthophotographie RVB 8 bits, 2048 px au moins), deux extraits 256×256
réduits d'un facteur 4 (moyenne 2×2 puis un pixel sur deux : scène nette
à l'échelle du pixel, sans flou propre marqué).

Instrument : Q = 1, obturation 0.3, jitter 0.35 px connu (fixé), cinq
cas d'aberrations inconnues, bruit σ = 0.003 et 0.01.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from skimage import color, data

from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.backends.optics_blind import OpticalModel, optics_blind_deconvolve
from eoqual.deconvolution.psf import optical_psf
from eoqual.deconvolution.simulate import simulate_observation

Q, EPS, JIT = 1.0, 0.3, 0.35
CASES = {
    "défoc. 0.12": ({"defocus": 0.12}, {"defocus": (0, 0.4)}, {"jitter": JIT}),
    "défoc. 0.25": ({"defocus": 0.25}, {"defocus": (0, 0.4)}, {"jitter": JIT}),
    "défoc. + astig.": ({"defocus": 0.12, "astig_0": 0.06},
                        {"defocus": (0, 0.4), "astig_0": (-0.1, 0.1), "astig_45": (-0.1, 0.1)}, {"jitter": JIT}),
    "défoc. + jitter libre": ({"defocus": 0.12}, {"defocus": (0, 0.4), "jitter": (0, 0.8)}, {}),
    "défoc. + coma": ({"defocus": 0.2, "coma_x": 0.06},
                      {"defocus": (0, 0.4), "coma_x": (-0.1, 0.1), "coma_y": (-0.1, 0.1)}, {"jitter": JIT}),
}


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))


def scenes(path=None):
    out = {"astronaut": color.rgb2gray(data.astronaut())[100:356, 150:406].astype(np.float64)}
    if path:
        import tifffile

        rgb = tifffile.imread(path)[..., :3].astype(np.float64) / 255.0
        gray = (rgb**2.2) @ np.array([0.2126, 0.7152, 0.0722])
        n = min(gray.shape) // 2 * 2
        small = gray[:n, :n].reshape(n // 2, 2, n // 2, 2).mean(axis=(1, 3))[::2, ::2]
        out["urbain"] = small[100:356, 150:406]
        out["parc"] = small[0:256, 256:512]
    return out


def main() -> None:
    sc = scenes(sys.argv[1] if len(sys.argv) > 1 else None)
    print("| cas | scène | σ | vraie PSF | nominale | blind_l0 | evidence | l0 | θ estimé (evidence) | temps |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cname, (true, free, fixed) in CASES.items():
        h_true = optical_psf(Q, zernike=true, obscuration=EPS, jitter_sigma=JIT, size=15)
        model = OpticalModel(Q, free, fixed, EPS, size=15)
        nominal_theta = [JIT if k == "jitter" else 0.0 for k in model.names]
        for sn, x in sc.items():
            for sig in (0.003, 0.01):
                y = simulate_observation(x, h_true, sig, boundary="reflect", seed=1)
                r = lambda h: psnr(x, deconvolve(y, h, "tv_primal_dual", sigma=sig))
                t = time.time()
                ev = optics_blind_deconvolve(y, model, sig)
                dt = time.time() - t
                l0 = optics_blind_deconvolve(y, model, sig, estimator="l0")
                bl = deconvolve(y, None, "blind_l0", sigma=sig, psf_size=15)
                est = ", ".join(f"{k} {v:.2f}" for k, v in ev.params.items() if k != "jitter" or "jitter" in free)
                print(f"| {cname} | {sn} | {sig} | {r(h_true):.2f} | {r(model.psf(nominal_theta)):.2f} | "
                      f"{psnr(x, bl):.2f} | {psnr(x, ev.image):.2f} | {psnr(x, l0.image):.2f} | {est} | {dt:.0f} s |",
                      flush=True)


if __name__ == "__main__":
    main()
