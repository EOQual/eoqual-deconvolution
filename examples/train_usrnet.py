#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Réentraînement (réglage fin) d'USRNet sur des PSF d'instrument
(REFERENCE_TECHNIQUE.md §7.3), puis évaluation sur une zone disjointe.

    python examples/train_usrnet.py --config examples/train_usrnet.jsonc \\
        --train ortho1.tif ortho2.tif --val ortho_val.tif [--out DIR]
    python examples/train_usrnet.py --config ... --image IMAGE.tif --split 0.75

Avec ``--image`` et ``--split``, une seule grande image est coupée en deux
bandes horizontales disjointes : entraînement en haut, validation en bas.

Le fichier de configuration (JSONC) décrit l'instrument (lois des PSF et du
bruit) et l'entraînement ; voir ``examples/train_usrnet.jsonc``. À la fin,
le script compare, sur la zone de validation, les poids de départ et les
poids réentraînés pour quelques PSF tirées de la même loi, et écrit
``evaluation.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.deep.training import (
    NoiseSampler, PSFSampler, TrainingConfig, load_training_images, train_usrnet,
)
from eoqual.deconvolution.simulate import add_noise, blur
from eoqual.deconvolution.tuning.jsonc import load_jsonc


def make_config(conf: dict, out: str | None) -> TrainingConfig:
    psf = PSFSampler(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in conf.get("psf", {}).items()})
    if "zernike" in conf.get("psf", {}):
        psf.zernike = {k: tuple(v) for k, v in conf["psf"]["zernike"].items()}
    noise = NoiseSampler(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in conf.get("noise", {}).items()})
    train = dict(conf.get("training", {}))
    if out:
        train["out_dir"] = out
    if "milestones" in train:
        train["milestones"] = tuple(train["milestones"])
    return TrainingConfig(psf=psf, noise=noise, **train)


def evaluate(images, cfg: TrainingConfig, checkpoint, n_cases=24, seed=7):
    """PSNR (dB) sur des vignettes de validation, poids de départ contre poids réentraînés."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_cases):
        im = images[rng.integers(len(images))]
        P = 256
        r, c = rng.integers(0, im.shape[0] - P + 1), rng.integers(0, im.shape[1] - P + 1)
        x = im[r : r + P, c : c + P].astype(np.float64)
        k = cfg.psf(rng).astype(np.float64)
        s = float(rng.uniform(0.002, max(cfg.noise.sigma[1], 0.003)))
        y = add_noise(blur(x, k, boundary="reflect"), s, seed=int(rng.integers(1 << 30)))
        ps = lambda a: float(10 * np.log10(1.0 / np.mean((a - x) ** 2)))
        rows.append({"sigma": s, "observée": ps(y), "wiener_hunt": ps(deconvolve(y, k, "wiener_hunt", sigma=s)),
                     "départ": ps(deconvolve(y, k, "usrnet", sigma=s, weights=cfg.init_weights or "usrnet")),
                     "réentraîné": ps(deconvolve(y, k, "usrnet", sigma=s, weights=str(checkpoint)))})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "sigma"}
    return rows, mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", nargs="*")
    ap.add_argument("--val", nargs="*")
    ap.add_argument("--image")
    ap.add_argument("--split", type=float, default=0.75)
    ap.add_argument("--out")
    args = ap.parse_args()
    conf = load_jsonc(args.config)
    cfg = make_config(conf, args.out)
    prep = conf.get("images", {})
    if args.image:
        full = load_training_images([args.image], gamma=prep.get("gamma"), reduce=prep.get("reduce", 1))[0]
        cut = int(args.split * full.shape[0])
        tr, va = [full[:cut]], [full[cut:]]
    else:
        tr = load_training_images(args.train, gamma=prep.get("gamma"), reduce=prep.get("reduce", 1))
        va = load_training_images(args.val, gamma=prep.get("gamma"), reduce=prep.get("reduce", 1))
    ckpt = train_usrnet(tr, va, cfg)
    rows, mean = evaluate(va, cfg, ckpt)
    out = Path(cfg.out_dir)
    (out / "evaluation.json").write_text(json.dumps({"mean_psnr_db": mean, "cases": rows}, indent=1))
    print("PSNR moyen sur la validation (dB) :", {k: round(v, 2) for k, v in mean.items()})
    print(f"Poids : {ckpt}  —  empreinte : {(out / 'best.sha256').read_text().split()[0]}")


if __name__ == "__main__":
    main()
