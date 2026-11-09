#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USRNet — réseau déplié pour la déconvolution (et la super-résolution) non
aveugle : la PSF et le niveau de bruit sont des **entrées** du réseau.

K. Zhang, L. Van Gool, R. Timofte, « Deep unfolding network for image
super-resolution », CVPR 2020.

Modèle : ``y = (k ∗ x)↓s + n``. Le problème MAP
``min ½‖y − (k ∗ x)↓s‖² + λΦ(x)`` est résolu par éclatement
semi-quadratique (HQS) déroulé sur ``n_iter`` itérations :

- **module de données** (sans paramètre appris) : solution exacte, en
  Fourier, de ``min ‖y − (k ∗ z)↓s‖² + α_k‖z − x_{k−1}‖²``. Pour ``s = 1``
  (déconvolution, cas traité ici en priorité) :
  ``Z = (H* Y + α_k X) / (|H|² + α_k)`` — un filtre de Wiener–Tikhonov
  rappelé vers l'estimée courante ;
- **module d'a priori** : un ResUNet débruiteur conditionné par β_k (carte
  constante concaténée à l'image) ;
- **module d'hyperparamètres** : un petit MLP qui donne ``(α_k, β_k)``
  pour chaque itération à partir de ``(σ, s)``.

Licence : architecture et poids publiés de KAIR (© 2019 Kai Zhang, MIT,
https://github.com/cszn/KAIR) ; le code ci-dessous en reprend la structure
et les noms de paramètres (pour charger ces poids), réécrit pour l'API FFT
complexe actuelle de PyTorch et branché sur le paquet (bords, échelle,
registre). Voir THIRD_PARTY_LICENSES.md §2.2 et §3.
"""
from __future__ import annotations

import numpy as np

from .._common_torch import require_torch
from ..utils import check_image, from_unit_range, resolve_sigma, to_unit_range

torch = require_torch()
nn = torch.nn

__all__ = ["USRNet", "load_usrnet", "usrnet_deconvolve", "p2o", "data_step", "select_device"]


# ---------------------------------------------------------------------------
# Architecture (noms de paramètres identiques à KAIR)
# ---------------------------------------------------------------------------

def _seq(*mods):
    """``nn.Sequential`` aplati ; un seul module est renvoyé tel quel (comme KAIR)."""
    flat = []
    for m in mods:
        flat.extend(m.children() if isinstance(m, nn.Sequential) else [m])
    return flat[0] if len(flat) == 1 else nn.Sequential(*flat)


class _ResBlock(nn.Module):
    def __init__(self, nc: int):
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(nc, nc, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(nc, nc, 3, 1, 1, bias=False))

    def forward(self, x):
        return x + self.res(x)


class ResUNet(nn.Module):
    """Débruiteur U-Net résiduel (module d'a priori)."""

    def __init__(self, in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=2):
        super().__init__()
        rb = lambda c: [_ResBlock(c) for _ in range(nb)]
        down = lambda a, b: nn.Conv2d(a, b, 2, 2, 0, bias=False)
        up = lambda a, b: nn.ConvTranspose2d(a, b, 2, 2, 0, bias=False)
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=False)
        self.m_down1 = _seq(*rb(nc[0]), down(nc[0], nc[1]))
        self.m_down2 = _seq(*rb(nc[1]), down(nc[1], nc[2]))
        self.m_down3 = _seq(*rb(nc[2]), down(nc[2], nc[3]))
        self.m_body = _seq(*rb(nc[3]))
        self.m_up3 = _seq(up(nc[3], nc[2]), *rb(nc[2]))
        self.m_up2 = _seq(up(nc[2], nc[1]), *rb(nc[1]))
        self.m_up1 = _seq(up(nc[1], nc[0]), *rb(nc[0]))
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = nn.functional.pad(x, (0, int(np.ceil(w / 8) * 8 - w), 0, int(np.ceil(h / 8) * 8 - h)), mode="replicate")
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x[..., :h, :w]


class HyPaNet(nn.Module):
    """``(σ, s) ↦ (α_1..α_K, β_1..β_K)``, positifs (Softplus)."""

    def __init__(self, in_nc=2, out_nc=16, channel=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(in_nc, channel, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, channel, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, out_nc, 1), nn.Softplus())

    def forward(self, x):
        return self.mlp(x) + 1e-6


def p2o(psf, shape):
    """OTF complexe ``N×C×H×W`` d'une PSF ``N×C×h×w`` centrée en ``(h//2, w//2)`` (convention du paquet)."""
    otf = torch.zeros(psf.shape[:-2] + tuple(shape), dtype=psf.dtype, device=psf.device)
    otf[..., : psf.shape[-2], : psf.shape[-1]] = psf
    otf = torch.roll(otf, shifts=(-(psf.shape[-2] // 2), -(psf.shape[-1] // 2)), dims=(-2, -1))
    return torch.fft.fft2(otf)


def _splits(a, sf):
    """Blocs ``sf × sf`` distincts du plan de Fourier, empilés sur un dernier axe."""
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=-1)
    return torch.cat(torch.chunk(b, sf, dim=3), dim=-1)


def data_step(x, FB, FBC, F2B, FBFy, alpha, sf):
    """
    Module de données : ``argmin_z ‖y − (k ∗ z)↓s‖² + α‖z − x‖²`` en Fourier
    (Zhao et al. 2016, formule fermée). Pour ``s = 1`` :
    ``Z = (H* Y + α X) / (|H|² + α)``.
    """
    FR = FBFy + torch.fft.fft2(alpha * x)
    x1 = FB * FR
    FBR = torch.mean(_splits(x1, sf), dim=-1)
    invW = torch.mean(_splits(F2B, sf), dim=-1)
    invWBR = FBR / (invW + alpha)
    FCBinvWBR = FBC * invWBR.repeat(1, 1, sf, sf)
    FX = (FR - FCBinvWBR) / alpha
    return torch.fft.ifft2(FX).real


class USRNet(nn.Module):
    """
    Réseau déplié : ``forward(y, k, sf, sigma)`` avec ``y`` ``N×3×H×W`` dans
    [0, 1], ``k`` ``N×1×h×w`` (somme 1), ``sigma`` ``N×1×1×1`` (écart-type du
    bruit dans les unités de ``y``), ``sf`` facteur de super-résolution
    (1 : déconvolution).
    """

    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=2):
        super().__init__()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb)
        self.h = HyPaNet(in_nc=2, out_nc=n_iter * 2, channel=h_nc)
        self.n = n_iter

    def forward(self, y, k, sf, sigma):
        H, W = y.shape[-2:]
        FB = p2o(k, (H * sf, W * sf))
        FBC = torch.conj(FB)
        F2B = (FB.real**2 + FB.imag**2).to(FB.dtype)
        STy = torch.zeros(y.shape[0], y.shape[1], H * sf, W * sf, dtype=y.dtype, device=y.device)
        STy[..., ::sf, ::sf] = y
        FBFy = FBC * torch.fft.fft2(STy)
        x = nn.functional.interpolate(y, scale_factor=sf, mode="nearest") if sf > 1 else y
        ab = self.h(torch.cat((sigma, torch.full_like(sigma, float(sf))), dim=1))
        for i in range(self.n):
            x = data_step(x, FB, FBC, F2B, FBFy, ab[:, i : i + 1], sf)
            beta = ab[:, i + self.n : i + self.n + 1].expand(-1, -1, x.shape[2], x.shape[3])
            x = self.p(torch.cat((x, beta), dim=1))
        return x


# ---------------------------------------------------------------------------
# Chargement et inférence
# ---------------------------------------------------------------------------

_MODELS: dict = {}


def select_device(device=None):
    """``device`` explicite, sinon CUDA, puis MPS (GPU Apple), puis CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_usrnet(weights="usrnet", device=None, arch: dict | None = None) -> USRNet:
    """
    USRNet en mode évaluation, poids chargés strictement.

    Parameters
    ----------
    weights : str or path
        ``"usrnet"`` (défaut, 17 M de paramètres), ``"usrnet_tiny"``
        (0.6 M), ou chemin d'un fichier de poids (réentraînement,
        :mod:`eoqual.deconvolution.deep.training`) ; un point de contrôle
        produit par l'entraînement contient aussi son architecture.
    arch : dict, optional
        Architecture (``n_iter``, ``h_nc``, ``nc``, ``nb``) d'un fichier de
        poids qui ne la contient pas.
    """
    from .weights import KNOWN_WEIGHTS, resolve_weights

    dev = select_device(device)
    key = (str(weights), str(dev))
    if key in _MODELS:
        return _MODELS[key]
    path = resolve_weights(weights)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        arch = arch or state.get("arch")
        state = state["state_dict"]
    arch = arch or KNOWN_WEIGHTS.get(str(weights), {}).get("arch") or KNOWN_WEIGHTS["usrnet"]["arch"]
    model = USRNet(n_iter=arch["n_iter"], h_nc=arch["h_nc"], nc=tuple(arch["nc"]), nb=arch["nb"])
    model.load_state_dict(state, strict=True)
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    _MODELS[key] = model
    return model


def _consistent_extension(image, psf, mask, saturation, n_fill_iter=30):
    """
    Domaine de travail prolongé dont la marge (et les pixels inconnus ou
    saturés) est remplie par le flou d'une restauration de Wiener–Hunt à
    marge inconnue (§2.2, §3.5) : l'image complète est alors une image
    floue cohérente avec la PSF sur tout le domaine périodique, ce que
    suppose le module de données d'USRNet (entraîné en flou circulaire).
    """
    from scipy import fft as spfft

    from ..backends._common import regularizer_power, setup, solve_with_fill
    from ..tuning.criteria import wiener_hunt_estimate

    pb = setup(image, psf, "unknown", mask=mask, saturation=saturation)
    L2 = regularizer_power("laplacian", pb.y.shape)
    lam = wiener_hunt_estimate(spfft.fft2(pb.y), pb.op.abs2, L2)[0]
    W = np.conj(pb.op.otf) / np.maximum(pb.op.abs2 + lam * L2, 1e-300)
    x = solve_with_fill(pb, lambda y: pb.op.solve_quadratic(spfft.fft2(y) * W, 1.0), n_fill_iter)
    return pb, pb.fill(pb.op.forward(x), pb.y, pb.level)


def usrnet_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    sigma: float | None = None,
    *,
    weights="usrnet",
    boundary: str = "unknown",
    mask=None,
    saturation: float | None = None,
    data_range: tuple[float, float] | None = None,
    device=None,
    callback=None,
) -> np.ndarray:
    """
    Déconvolution par USRNet (extra ``deep``), PSF et σ connus.

    L'image est ramenée à [0, 1] (``data_range``, défaut min/max) et
    prolongée ; le réseau, en couleur, reçoit l'image répliquée sur trois
    canaux et la sortie est leur moyenne, recadrée et remise à l'échelle.

    **Bords.** Le réseau suppose une convolution circulaire (son
    entraînement). ``boundary="unknown"`` (défaut) remplit la marge — et
    les pixels masqués ou saturés — par le flou d'une restauration de
    Wiener–Hunt à marge inconnue, calculée une fois avant le réseau : le
    domaine est alors une image floue cohérente avec la PSF. Un
    prolongement miroir (``"reflect"``) ne l'est pas pour une PSF
    asymétrique (bougé) et ses défauts sont amplifiés aux zéros de la FTM ;
    compléter la marge à chaque itération, comme le Plug-and-Play (§7.1),
    diverge, les estimées intermédiaires d'un réseau déplié n'étant pas des
    images (REFERENCE_TECHNIQUE.md §7.2).

    Parameters
    ----------
    sigma : float, optional
        Écart-type du bruit (unités de l'image) ; ``None`` : estimé. Les
        poids publiés couvrent σ de 0 à 25/255 de la dynamique.
    weights : str or path, default "usrnet"
        ``"usrnet"``, ``"usrnet_tiny"`` ou chemin de poids réentraînés
        (:mod:`eoqual.deconvolution.deep.training`).
    boundary : {"unknown", "reflect", "periodic"}, default "unknown"
    mask, saturation :
        Pixels valides et seuil de saturation (§2.2) ; avec
        ``boundary="unknown"`` seulement (remplis avant le réseau).
    device : str, optional
        ``"cuda"``, ``"mps"``, ``"cpu"`` ; défaut : le plus rapide disponible.
    callback : callable, optional
        ``callback(k, x)`` après chaque itération (domaine de travail,
        unités d'origine).
    """
    from .._common_torch import no_grad
    from ..backends._common import setup

    if boundary not in ("unknown", "reflect", "periodic"):
        raise ValueError("usrnet : boundary doit valoir 'unknown', 'reflect' ou 'periodic'.")
    if boundary != "unknown" and (mask is not None or saturation is not None):
        raise ValueError("usrnet : mask/saturation demandent boundary='unknown'.")
    if boundary == "unknown":
        pb, y_dom = _consistent_extension(image, psf, mask, saturation)
    else:
        pb = setup(check_image(image), psf, boundary)
        y_dom = pb.y
    s = resolve_sigma(pb.domain.restore(pb.y), sigma)
    y, lo, scale = to_unit_range(y_dom, data_range if data_range is not None
                                 else (float(np.min(pb.domain.restore(pb.y))), float(np.max(pb.domain.restore(pb.y)))))
    model = load_usrnet(weights, device)
    dev = next(model.parameters()).device
    f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    with no_grad():
        y_t = f32(y)[None, None].repeat(1, 3, 1, 1)
        FB = p2o(f32(pb.psf)[None, None], y.shape)
        FBC = torch.conj(FB)
        F2B = (FB.real**2 + FB.imag**2).to(FB.dtype)
        FBFy = FBC * torch.fft.fft2(y_t)
        sig_t = torch.full((1, 1, 1, 1), s / scale, dtype=torch.float32, device=dev)
        ab = model.h(torch.cat((sig_t, torch.ones_like(sig_t)), dim=1))
        x = y_t
        for i in range(model.n):
            x = data_step(x, FB, FBC, F2B, FBFy, ab[:, i : i + 1], 1)
            beta = ab[:, i + model.n : i + model.n + 1].expand(-1, -1, x.shape[2], x.shape[3])
            x = model.p(torch.cat((x, beta), dim=1))
            if callback is not None:
                callback(i, from_unit_range(x.mean(dim=1)[0].float().cpu().numpy().astype(np.float64), lo, scale))
        out = x.mean(dim=1)[0].float().cpu().numpy().astype(np.float64)
    return pb.domain.restore(from_unit_range(out, lo, scale))
