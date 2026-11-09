"""USRNet (extra deep) : module de données, poids publiés, bords, réentraînement."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from _helpers import gain_db  # noqa: E402
from conftest import SIGMA  # noqa: E402
from eoqual.core.kernel import psf2otf  # noqa: E402
from eoqual.deconvolution import deconvolve  # noqa: E402
from eoqual.deconvolution.deep.training import (  # noqa: E402
    NoiseSampler, PSFSampler, TrainingConfig, train_usrnet,
)
from eoqual.deconvolution.deep.usrnet import USRNet, data_step, load_usrnet, p2o  # noqa: E402
from eoqual.deconvolution.psf import gaussian_psf, motion_psf  # noqa: E402
from eoqual.deconvolution.simulate import simulate_observation  # noqa: E402

TINY = {"n_iter": 2, "h_nc": 8, "nc": [8, 8, 8, 8], "nb": 1}


def _t(a):
    return torch.as_tensor(a, dtype=torch.float64)[None, None]


def test_p2o_matches_core_psf2otf():
    k = motion_psf(7, 0.5, 9)
    np.testing.assert_allclose(p2o(_t(k), (40, 36))[0, 0].numpy(), psf2otf(k, (40, 36)), atol=1e-12)


def test_data_step_is_wiener_tikhonov():
    # s = 1 : Z = (H* Y + α X) / (|H|² + α)
    rng = np.random.default_rng(0)
    y, x, k, a = rng.random((32, 40)), rng.random((32, 40)), gaussian_psf(1.5, 9), 0.03
    H = psf2otf(k, y.shape)
    ref = np.fft.ifft2((np.conj(H) * np.fft.fft2(y) + a * np.fft.fft2(x)) / (np.abs(H) ** 2 + a)).real
    FB = p2o(_t(k), y.shape)
    FBC = torch.conj(FB)
    z = data_step(_t(x), FB, FBC, (FB.abs() ** 2).to(FB.dtype), FBC * torch.fft.fft2(_t(y)),
                  torch.tensor(a, dtype=torch.float64), 1)
    np.testing.assert_allclose(z[0, 0].numpy(), ref, atol=1e-12)


def test_samplers():
    rng = np.random.default_rng(1)
    ps = PSFSampler()
    for _ in range(15):
        k = ps(rng)
        assert k.shape == (25, 25) and k.sum() == pytest.approx(1.0, abs=1e-5) and k.min() >= 0
    y, s = NoiseSampler(sigma=(0.01, 0.02), zero_fraction=0.0)(np.full((64, 64), 0.5, np.float32), rng)
    assert 0.01 <= s <= 0.02 and np.std(y - 0.5) == pytest.approx(s, rel=0.2)


def test_training_from_scratch_and_reload(tmp_path, scene):
    cfg = TrainingConfig(init_weights=None, arch=TINY, iterations=4, batch_size=2, patch=32,
                         val_every=2, val_samples=2, out_dir=str(tmp_path), device="cpu", seed=0)
    ckpt = train_usrnet([scene.astype(np.float32)], [scene.astype(np.float32)], cfg, log=lambda m: None)
    assert ckpt.exists() and (tmp_path / "history.csv").exists() and (tmp_path / "best.sha256").exists()
    model = load_usrnet(str(ckpt), "cpu")
    assert isinstance(model, USRNet) and model.n == TINY["n_iter"]
    out = deconvolve(scene[:64, :64], gaussian_psf(1.2, 9), "usrnet", sigma=0.01, weights=str(ckpt), device="cpu")
    assert out.shape == (64, 64) and np.isfinite(out).all()


# --- poids publiés (téléchargement au premier appel) -----------------------

deep = pytest.mark.deep


@deep
def test_published_weights_and_gain(scene, blurred, psf):
    model = load_usrnet("usrnet_tiny", "cpu")
    assert sum(p.numel() for p in model.parameters()) == 590332
    out = deconvolve(blurred, psf, "usrnet", sigma=SIGMA, weights="usrnet_tiny", device="cpu")
    assert gain_db(scene, blurred, out) > gain_db(scene, blurred, deconvolve(blurred, psf, "tikhonov"))


@deep
@pytest.mark.slow
def test_unknown_boundary_on_motion_blur():
    from skimage import color, data

    sc = color.rgb2gray(data.astronaut())[100:356, 150:406]
    h = motion_psf(9, 0.4)
    y = simulate_observation(sc, h, 0.003, boundary="reflect", seed=0)
    psnr = lambda a: 10 * np.log10(1 / np.mean((a - sc) ** 2))
    unknown = psnr(deconvolve(y, h, "usrnet", sigma=0.003))
    reflect = psnr(deconvolve(y, h, "usrnet", sigma=0.003, boundary="reflect"))
    assert unknown > reflect + 10                       # marge cohérente avec la PSF (REFERENCE §7.2)
    assert unknown > psnr(deconvolve(y, h, "tv_primal_dual", sigma=0.003))


def test_errors(blurred, psf):
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "usrnet", boundary="perdecomp")
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "usrnet", boundary="reflect", mask=np.ones_like(blurred, bool))
    with pytest.raises(ValueError):
        load_usrnet("inconnu")


@pytest.fixture(autouse=True)
def _weights_cache(tmp_path_factory, monkeypatch):
    # cache des poids partagé entre tests (variable d'environnement respectée si définie)
    if "EOQUAL_WEIGHTS_DIR" not in os.environ:
        monkeypatch.setenv("EOQUAL_WEIGHTS_DIR", str(tmp_path_factory.getbasetemp() / "weights"))


@deep
def test_mask_dead_lines(scene, blurred, psf):
    mask = np.ones_like(blurred, dtype=bool)
    mask[::16, :] = False
    damaged = np.where(mask, blurred, 0.0)
    psnr = lambda a: 10 * np.log10(1 / np.mean((a - scene) ** 2))
    kw = dict(sigma=SIGMA, weights="usrnet_tiny", device="cpu")
    with_mask = psnr(deconvolve(damaged, psf, "usrnet", mask=mask, **kw))
    without = psnr(deconvolve(damaged, psf, "usrnet", **kw))
    assert with_mask > without + 3.0
    assert with_mask > psnr(deconvolve(blurred, psf, "usrnet", **kw)) - 1.0
