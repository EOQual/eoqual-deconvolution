#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Réentraînement et réglage fin d'USRNet sur des paires simulées
(REFERENCE_TECHNIQUE.md §7.3).

Les poids publiés (KAIR) sont appris sur des photographies, avec des
noyaux gaussiens et de bougé. Pour un instrument d'observation de la
Terre, on les adapte (réglage fin) ou on les réapprend sur :

- des **vignettes nettes** tirées d'images de licence claire (par
  exemple des orthophotographies sous Licence Ouverte 2.0), lues sur
  disque, découpées au hasard, retournées et tournées ;
- des **PSF d'instrument** tirées à chaque exemple : modèle optique
  (:func:`eoqual.deconvolution.psf.optical_psf`, échantillonnage ``Q``,
  obturation, défocalisation et aberrations de Zernike, jitter dans des
  plages), complété au besoin de gaussiennes et de filés (§10.1) ;
- un **bruit** gaussien ou poisson-gaussien de niveau tiré au hasard.

Le flou est **circulaire** (comme le module de données du réseau) ;
l'inférence prolonge l'image par miroir (``boundary="reflect"``).

Usage minimal ::

    from eoqual.deconvolution.deep.training import TrainingConfig, train_usrnet
    cfg = TrainingConfig(iterations=20000, init_weights="usrnet")
    ckpt = train_usrnet(["ortho_a.tif", "ortho_b.tif"], ["ortho_val.tif"], cfg)
    restored = deconvolve(y, psf, "usrnet", sigma=s, weights=ckpt)
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .._common_torch import require_torch
from ..psf import gaussian_psf, motion_psf, optical_psf

torch = require_torch()

__all__ = ["PSFSampler", "NoiseSampler", "TrainingConfig", "PatchSampler", "load_training_images",
           "train_usrnet", "evaluate_usrnet"]


# ---------------------------------------------------------------------------
# Tirage des dégradations
# ---------------------------------------------------------------------------

@dataclass
class PSFSampler:
    """
    Loi des PSF d'entraînement. Chaque famille est tirée avec sa
    probabilité ``weights`` ; les plages sont des intervalles uniformes.

    - ``optical`` : :func:`optical_psf` avec ``sampling`` (Q), ``obscuration``,
      coefficients de Zernike ``zernike`` (``{nom ou indice de Noll: (bas,
      haut)}``, ondes RMS) et ``jitter`` (pixels) ;
    - ``gaussian`` : écart-type ``gaussian_sigma`` (pixels), anisotropie
      jusqu'à ``gaussian_anisotropy``, orientation quelconque ;
    - ``motion`` : filé de longueur ``motion_length`` (pixels), direction
      quelconque.
    """

    size: int = 25
    weights: dict = field(default_factory=lambda: {"optical": 0.7, "gaussian": 0.15, "motion": 0.15})
    sampling: tuple = (0.8, 2.0)
    obscuration: tuple = (0.0, 0.35)
    zernike: dict = field(default_factory=lambda: {"defocus": (0.0, 0.3), "astig_0": (-0.08, 0.08),
                                                   "astig_45": (-0.08, 0.08), "coma_x": (-0.05, 0.05),
                                                   "coma_y": (-0.05, 0.05), "spherical": (-0.05, 0.05)})
    jitter: tuple = (0.0, 0.6)
    gaussian_sigma: tuple = (0.5, 3.0)
    gaussian_anisotropy: float = 0.5
    motion_length: tuple = (1.0, 7.0)

    def __call__(self, rng: np.random.Generator) -> np.ndarray:
        fams = list(self.weights)
        p = np.array([self.weights[f] for f in fams], float)
        fam = fams[rng.choice(len(fams), p=p / p.sum())]
        u = lambda r: float(rng.uniform(*r))
        if fam == "optical":
            z = {k: u(r) for k, r in self.zernike.items()}
            k = optical_psf(u(self.sampling), zernike=z, obscuration=u(self.obscuration),
                            jitter_sigma=u(self.jitter), size=self.size)
        elif fam == "gaussian":
            s = u(self.gaussian_sigma)
            k = gaussian_psf(s, self.size, sigma_x=s * (1 - u((0.0, self.gaussian_anisotropy))),
                             theta=u((0.0, np.pi)))
        elif fam == "motion":
            k = motion_psf(u(self.motion_length), u((0.0, np.pi)), self.size)
        else:
            raise ValueError(f"Famille de PSF inconnue {fam!r}.")
        return (k / k.sum()).astype(np.float32)


@dataclass
class NoiseSampler:
    """
    Bruit : écart-type ``sigma`` (fraction de la dynamique, image dans
    [0, 1]) tiré uniformément ; avec ``poisson_fraction > 0``, une part des
    exemples reçoit un bruit poisson-gaussien de même écart-type au niveau
    moyen (``σ² = a + b·L``). ``zero_fraction`` : part sans bruit.
    """

    sigma: tuple = (0.0, 0.05)
    zero_fraction: float = 0.1
    poisson_fraction: float = 0.5

    def __call__(self, clean: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        if rng.random() < self.zero_fraction:
            return clean, 0.0
        s = float(rng.uniform(*self.sigma))
        if rng.random() < self.poisson_fraction:
            mean = max(float(clean.mean()), 1e-3)
            a = rng.uniform(0.0, 0.5) * s**2                     # part du bruit de lecture
            b = (s**2 - a) / mean
            noise = rng.standard_normal(clean.shape) * np.sqrt(a + b * np.clip(clean, 0, None))
        else:
            noise = rng.normal(0.0, s, clean.shape)
        return clean + noise.astype(clean.dtype), s


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def load_training_images(paths, *, gamma: float | None = None, reduce: int = 1) -> list[np.ndarray]:
    """
    Images d'entraînement en niveaux de gris, flottantes dans [0, 1].

    Parameters
    ----------
    paths : sequence
        Fichiers (TIFF, PNG, JPEG) ou tableaux déjà chargés.
    gamma : float, optional
        Linéarisation ``x**gamma`` d'une image encodée (sRGB ≈ 2.2) : le
        flou d'un instrument s'applique à la luminance **linéaire**.
    reduce : int, default 1
        Réduction par moyenne ``reduce × reduce`` (une orthophotographie
        plus fine que le pixel visé donne des vignettes nettes à l'échelle
        du pixel, sans flou propre marqué).
    """
    out = []
    for p in paths:
        if isinstance(p, np.ndarray):
            img = p.astype(np.float64)
        else:
            path = Path(p)
            if path.suffix.lower() in (".tif", ".tiff"):
                import tifffile

                img = tifffile.imread(path).astype(np.float64)
            else:
                import imageio.v3 as iio

                img = iio.imread(path).astype(np.float64)
        if img.ndim == 3:
            img = img[..., :3] @ np.array([0.2126, 0.7152, 0.0722])[: img.shape[-1]] if img.shape[-1] >= 3 else img[..., 0]
        img = img / (255.0 if img.max() > 1.5 and img.max() <= 255 else (img.max() if img.max() > 1.5 else 1.0))
        if gamma:
            img = np.clip(img, 0, 1) ** gamma
        if reduce > 1:
            h, w = (img.shape[0] // reduce) * reduce, (img.shape[1] // reduce) * reduce
            img = img[:h, :w].reshape(h // reduce, reduce, w // reduce, reduce).mean(axis=(1, 3))
        out.append(img.astype(np.float32))
    return out


class PatchSampler:
    """
    Paires (dégradée, nette, PSF, σ) tirées à la volée : vignette au hasard
    dans une image au hasard, retournement et rotation, flou par la PSF
    tirée, bruit.

    ``borders="real"`` (défaut) reproduit l'usage (§7.2) : la vignette est
    découpée au centre d'une zone plus grande floutée, si bien que ses bords
    contiennent la contribution de la scène extérieure, comme une vraie
    image ; puis sa marge est complétée par le flou d'une restauration de
    Wiener–Hunt à marge inconnue, exactement comme à l'inférence, et
    l'image est ramenée à [0, 1] par son min/max, σ en proportion. La perte
    porte sur la vignette seule (``[:P, :P]`` du domaine de travail).
    ``borders="circular"`` : flou circulaire de la vignette (cadre exact du
    module de données, comme l'entraînement d'origine de KAIR).
    """

    def __init__(self, images, patch: int, psf: PSFSampler, noise: NoiseSampler, seed: int = 0,
                 borders: str = "real"):
        if borders not in ("real", "circular"):
            raise ValueError("borders doit valoir 'real' ou 'circular'.")
        margin = psf.size if borders == "real" else 0
        self.images = [im for im in images if min(im.shape) >= patch + 2 * margin]
        if not self.images:
            raise ValueError(f"Aucune image d'au moins {patch + 2 * margin} pixels de côté.")
        self.patch, self.psf, self.noise, self.borders, self.margin = patch, psf, noise, borders, margin
        self.rng = np.random.default_rng(seed)

    def one(self):
        from eoqual.core.kernel import psf2otf

        rng, P, m = self.rng, self.patch, self.margin
        im = self.images[rng.integers(len(self.images))]
        Q = P + 2 * m
        r, c = rng.integers(0, im.shape[0] - Q + 1), rng.integers(0, im.shape[1] - Q + 1)
        big = np.rot90(im[r : r + Q, c : c + Q], rng.integers(4))
        if rng.random() < 0.5:
            big = big[:, ::-1]
        big = np.ascontiguousarray(big, dtype=np.float64)
        k = self.psf(rng)
        blurred = np.fft.ifft2(np.fft.fft2(big) * psf2otf(k, big.shape)).real
        x = big[m : m + P, m : m + P]
        y, s = self.noise(blurred[m : m + P, m : m + P], rng)
        if self.borders == "circular":
            return y.astype(np.float32), x.astype(np.float32), k, s
        from .usrnet import _consistent_extension

        _, y_dom = _consistent_extension(y, k, None, None)
        lo, scale = float(y.min()), max(float(y.max() - y.min()), 1e-6)
        return (((y_dom - lo) / scale).astype(np.float32), ((x - lo) / scale).astype(np.float32), k, s / scale)

    def batch(self, n: int, device):
        ys, xs, ks, ss = zip(*(self.one() for _ in range(n)))
        t = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32, device=device)
        y = t(ys)[:, None].repeat(1, 3, 1, 1)                  # le réseau est à 3 canaux
        x = t(xs)[:, None].repeat(1, 3, 1, 1)
        k = t(ks)[:, None]
        s = torch.as_tensor(ss, dtype=torch.float32, device=device).view(-1, 1, 1, 1)
        return y, x, k, s


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Réglages d'entraînement (valeurs par défaut : réglage fin depuis les
    poids publiés ; pour un entraînement complet, ``init_weights=None``,
    ``iterations`` ~ 4·10⁵ et ``lr`` 1e-4 comme KAIR).
    """

    init_weights: str | None = "usrnet"
    arch: dict | None = None                  # si init_weights=None : {"n_iter", "h_nc", "nc", "nb"}
    iterations: int = 20000
    batch_size: int = 8
    patch: int = 96
    lr: float = 5e-5
    milestones: tuple = (0.5, 0.75)           # fractions des itérations où lr est divisé par 2
    loss: str = "l1"
    seed: int = 0
    device: str | None = None
    val_every: int = 1000
    val_samples: int = 32
    borders: str = "real"                 # "real" : comme à l'usage (§7.3) ; "circular" : comme KAIR
    out_dir: str = "usrnet_training"
    psf: PSFSampler = field(default_factory=PSFSampler)
    noise: NoiseSampler = field(default_factory=NoiseSampler)


def _model_for(cfg: TrainingConfig, device):
    from .usrnet import USRNet, select_device
    from .weights import KNOWN_WEIGHTS, resolve_weights

    if cfg.init_weights is not None:
        state = torch.load(resolve_weights(cfg.init_weights), map_location="cpu", weights_only=True)
        arch = cfg.arch
        if isinstance(state, dict) and "state_dict" in state:
            arch, state = arch or state.get("arch"), state["state_dict"]
        arch = arch or KNOWN_WEIGHTS.get(str(cfg.init_weights), {}).get("arch") or KNOWN_WEIGHTS["usrnet"]["arch"]
    else:
        arch, state = cfg.arch or KNOWN_WEIGHTS["usrnet"]["arch"], None
    model = USRNet(n_iter=arch["n_iter"], h_nc=arch["h_nc"], nc=tuple(arch["nc"]), nb=arch["nb"])
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model.to(select_device(device)), {k: (list(v) if isinstance(v, tuple) else v) for k, v in arch.items()}


def evaluate_usrnet(model, sampler: PatchSampler, n: int, device) -> float:
    """PSNR moyen (dB, dynamique 1) sur ``n`` paires du jeu de validation (tirage fixe)."""
    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(n):
            y, x, k, s = sampler.batch(1, device)
            out = model(y, k, 1, s)[..., : x.shape[-2], : x.shape[-1]].mean(1)
            vals.append(float(-10 * torch.log10(torch.mean((out - x[:, 0]) ** 2))))
    model.train()
    return float(np.mean(vals))


def train_usrnet(train_images, val_images, cfg: TrainingConfig | None = None, log=print) -> Path:
    """
    Entraîne (ou règle finement) USRNet et renvoie le chemin du meilleur
    point de contrôle (PSNR de validation maximal).

    Produits dans ``cfg.out_dir`` : ``best.pth`` et ``last.pth`` (poids,
    architecture, réglages, historique), ``history.csv`` (itération, perte,
    PSNR de validation), ``config.json``, ``best.sha256``.

    Parameters
    ----------
    train_images, val_images :
        Images (chemins ou tableaux), **disjointes** : la validation doit
        porter sur des scènes jamais vues.
    """
    from .usrnet import select_device

    cfg = cfg or TrainingConfig()
    torch.manual_seed(cfg.seed)
    device = select_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = load_training_images(train_images) if not all(isinstance(i, np.ndarray) and i.dtype == np.float32 for i in train_images) else list(train_images)
    va = load_training_images(val_images) if not all(isinstance(i, np.ndarray) and i.dtype == np.float32 for i in val_images) else list(val_images)
    train_s = PatchSampler(tr, cfg.patch, cfg.psf, cfg.noise, cfg.seed, cfg.borders)
    model, arch = _model_for(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(m * cfg.iterations) for m in cfg.milestones], 0.5)
    lossf = torch.nn.L1Loss() if cfg.loss == "l1" else torch.nn.MSELoss()
    conf = asdict(cfg)
    (out / "config.json").write_text(json.dumps(conf, indent=1, default=str))

    def val_psnr():
        return evaluate_usrnet(model, PatchSampler(va, cfg.patch, cfg.psf, cfg.noise, seed=12345, borders=cfg.borders), cfg.val_samples, device)

    best = val_psnr()
    history = [(0, float("nan"), best)]
    log(f"itération 0 : PSNR de validation {best:.2f} dB (poids de départ)")
    save = lambda path: torch.save({"state_dict": model.state_dict(), "arch": arch, "config": conf,
                                    "history": history}, path)
    save(out / "best.pth")
    running, t0 = [], time.time()
    model.train()
    for it in range(1, cfg.iterations + 1):
        y, x, k, s = train_s.batch(cfg.batch_size, device)
        P = x.shape[-1]
        loss = lossf(model(y, k, 1, s)[..., :P, :P], x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running.append(loss.item())
        if it % cfg.val_every == 0 or it == cfg.iterations:
            v = val_psnr()
            history.append((it, float(np.mean(running)), v))
            log(f"itération {it} : perte {np.mean(running):.4f}, PSNR de validation {v:.2f} dB "
                f"({time.time() - t0:.0f} s)")
            running = []
            if v > best:
                best = v
                save(out / "best.pth")
    save(out / "last.pth")
    with open(out / "history.csv", "w", newline="") as f:
        csv.writer(f).writerows([("iteration", "loss", "val_psnr_db"), *history])
    from .weights import sha256

    (out / "best.sha256").write_text(sha256(out / "best.pth") + "  best.pth\n")
    return out / "best.pth"
