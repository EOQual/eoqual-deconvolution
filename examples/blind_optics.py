#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Déconvolution aveugle contrainte par l'optique (REFERENCE_TECHNIQUE.md
§8.2) sur un instrument simulé dont la PSF varie dans le champ.

Instrument : pupille annulaire (obturation 0.3), Q = λf/(Dp) = 1 (cas
courant en observation de la Terre), jitter connu (0.35 px, contrôle
d'attitude). Inconnus : une défocalisation qui croît du centre vers les
bords (courbure de champ, quadratique en position) et un astigmatisme
constant. On compare :

- la PSF nominale (instrument parfaitement réglé) ;
- ``blind_l0`` (PSF libre, a priori L0) ;
- ``blind_optics`` avec une seule PSF pour toute l'image ;
- ``blind_optics`` par tuiles, champ quadratique ;
- la vraie PSF (variable), restaurée par tuiles — la référence.

    python examples/blind_optics.py

Image : « astronaut » (NASA, domaine public, fournie par scikit-image).
"""
import os
from pathlib import Path

import numpy as np
from skimage import color, data

from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.backends.optics_blind import OpticalModel, optics_blind_deconvolve
from eoqual.deconvolution.plotting import plot_psf, plot_restoration
from eoqual.deconvolution.simulate import add_noise, blur

OUT = Path(os.environ.get("EOQUAL_EXAMPLES_OUT", Path(__file__).with_name("output")))
SIZE = int(os.environ.get("EOQUAL_EXAMPLES_SIZE", "512"))
SIGMA = 0.005
ASTIG = 0.05


def defocus_map(shape, centre=0.08, edge=0.28):
    """Défocalisation [ondes RMS] quadratique en distance au centre (courbure de champ)."""
    u = np.linspace(-0.5, 0.5, shape[0])[:, None]
    v = np.linspace(-0.5, 0.5, shape[1])[None, :]
    return centre + (edge - centre) * (u**2 + v**2) / 0.5


def simulate(scene, model):
    """Flou variable : images floues à quelques défocalisations, interpolées pixel par pixel."""
    dmap = defocus_map(scene.shape)
    levels = np.linspace(dmap.min(), dmap.max(), 6)
    stack = np.stack([blur(scene, model.psf([d, ASTIG]), boundary="reflect") for d in levels])
    pos = np.interp(dmap, levels, np.arange(levels.size))
    i0 = np.clip(np.floor(pos).astype(int), 0, levels.size - 2)
    w = pos - i0
    rows, cols = np.indices(scene.shape)
    clean = (1 - w) * stack[i0, rows, cols] + w * stack[i0 + 1, rows, cols]
    return add_noise(clean, SIGMA, seed=0)


def psnr(ref, img):
    return 10 * np.log10(1.0 / np.mean((ref - img) ** 2))


def main() -> None:
    scene = color.rgb2gray(data.astronaut()).astype(np.float64)
    scene = scene[: SIZE, : SIZE] if SIZE <= 512 else scene
    model = OpticalModel(1.0, free={"defocus": (0.0, 0.4), "astig_0": (-0.1, 0.1)},
                         fixed={"jitter": 0.35}, obscuration=0.3, size=15)
    blurred = simulate(scene, model)
    tile = max(64, SIZE // 4)

    nominal = deconvolve(blurred, model.psf([0.0, 0.0]), "tv_primal_dual", sigma=SIGMA)
    free_l0 = deconvolve(blurred, None, "blind_l0", sigma=SIGMA, psf_size=15)
    glob = optics_blind_deconvolve(blurred, model, SIGMA)
    tiled = optics_blind_deconvolve(blurred, model, SIGMA, tile_size=tile, overlap=0.5,
                                    field_model="quadratic")
    # Référence : vraie PSF de chaque tuile (défocalisation au centre de la tuile)
    acc, wsum = np.zeros_like(blurred), np.zeros_like(blurred)
    dmap = defocus_map(blurred.shape)
    for sl in tiled.tiles:
        d = dmap[(sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2]
        wy, wx = np.hanning(sl[0].stop - sl[0].start + 2)[1:-1], np.hanning(sl[1].stop - sl[1].start + 2)[1:-1]
        w = np.outer(wy, wx) + 1e-6
        acc[sl] += w * deconvolve(blurred[sl], model.psf([d, ASTIG]), "tv_primal_dual", sigma=SIGMA)
        wsum[sl] += w
    truth = acc / wsum

    results = {"PSF nominale": nominal, "blind_l0 (PSF libre)": free_l0,
               "blind_optics, PSF unique": glob.image, "blind_optics, tuiles": tiled.image,
               "vraie PSF (tuiles)": truth}
    print(f"Observation : {psnr(scene, blurred):.2f} dB")
    for name, img in results.items():
        print(f"  {name:26s} {psnr(scene, img):.2f} dB")
    print(f"PSF unique estimée : {glob.params}")
    centre = len(tiled.tiles) // 2
    print(f"Tuile centrale : estimé {tiled.params[centre]['defocus']:.3f}, vrai "
          f"{dmap[(tiled.tiles[centre][0].start + tiled.tiles[centre][0].stop) // 2, (tiled.tiles[centre][1].start + tiled.tiles[centre][1].stop) // 2]:.3f} onde RMS")
    print(f"Tuile du coin   : estimé {tiled.params[0]['defocus']:.3f}, vrai "
          f"{dmap[(tiled.tiles[0][0].start + tiled.tiles[0][0].stop) // 2, (tiled.tiles[0][1].start + tiled.tiles[0][1].stop) // 2]:.3f} onde RMS")
    OUT.mkdir(parents=True, exist_ok=True)
    plot_restoration(blurred, results, scene).write_html(OUT / "blind_optics.html")
    plot_psf({"Vraie PSF (centre)": model.psf([dmap[SIZE // 2, SIZE // 2], ASTIG]),
              "Estimée (tuile centrale)": tiled.psf[centre],
              "Vraie PSF (coin)": model.psf([dmap[tile // 2, tile // 2], ASTIG]),
              "Estimée (coin)": tiled.psf[0]}).write_html(OUT / "blind_optics_psf.html")
    print(f"Figures : {OUT}")


if __name__ == "__main__":
    main()
