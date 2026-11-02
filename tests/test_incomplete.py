"""Données incomplètes : pixels masqués, NaN, saturation, bords inconnus ; pas de Barzilai-Borwein."""
import numpy as np
import pytest

from conftest import SIGMA
from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.backends._common import setup
from eoqual.deconvolution.simulate import add_noise, blur, simulate_observation
from eoqual.metrics.fr.signal import psnr

ITERATIVE = ["richardson_lucy", "landweber", "fista_wavelet", "tv_primal_dual"]
LINEAR = ["inverse", "wiener", "tikhonov"]


@pytest.fixture(scope="module")
def dead_lines(blurred):
    mask = np.ones_like(blurred, dtype=bool)
    mask[::16, :] = False
    return np.where(mask, blurred, 0.0), mask


@pytest.mark.parametrize("method", LINEAR + ITERATIVE)
def test_mask_recovers_dead_lines(method, scene, blurred, psf, dead_lines):
    damaged, mask = dead_lines
    reference = psnr(scene, deconvolve(blurred, psf, method, sigma=SIGMA))
    with_mask = psnr(scene, deconvolve(damaged, psf, method, sigma=SIGMA, mask=mask))
    without = psnr(scene, deconvolve(damaged, psf, method, sigma=SIGMA))
    assert with_mask > reference - 0.6
    assert with_mask > without + 3.0


def test_nan_needs_mask(blurred, psf, dead_lines):
    _, mask = dead_lines
    with_nan = np.where(mask, blurred, np.nan)
    with pytest.raises(ValueError):
        deconvolve(with_nan, psf, "wiener", sigma=SIGMA)
    out = deconvolve(with_nan, psf, "wiener", sigma=SIGMA, mask=np.isfinite(with_nan))
    assert np.isfinite(out).all()


def test_mask_errors(blurred, psf):
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "wiener", mask=np.ones((3, 3), bool))
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "wiener", mask=np.zeros_like(blurred, bool))
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "tikhonov", mask=np.ones_like(blurred, bool), boundary="perdecomp")
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "wiener_unsupervised", saturation=0.5)


@pytest.fixture(scope="module")
def saturated(scene, psf):
    yb = blur(scene, psf, "reflect")
    level = 0.85 * yb.max()
    return np.minimum(add_noise(yb, SIGMA, seed=2), level), level


def test_setup_states(saturated, psf):
    y, level = saturated
    pb = setup(y, psf, "unknown", saturation=level)
    ny, nx = y.shape
    assert pb.incomplete and pb.level == level
    assert not pb.known[ny:, :].any() and not pb.known[:, nx:].any()  # marge inconnue
    assert pb.saturated[:ny, :nx].sum() == (y >= level).sum()
    assert not (pb.known & pb.saturated).any()
    hx = np.full(pb.y.shape, level + 1.0)
    filled = pb.fill(hx, pb.y, level)
    np.testing.assert_allclose(filled[pb.saturated], level + 1.0)  # Hx ≥ L : pas de pénalité
    np.testing.assert_allclose(pb.residual(np.zeros_like(hx), pb.y, level)[pb.saturated], -level)
    assert not setup(y, psf, "reflect").incomplete


@pytest.mark.parametrize("method", ["wiener", "tikhonov", "richardson_lucy", "landweber", "tv_primal_dual"])
def test_saturation_helps(method, scene, psf, saturated):
    y, level = saturated
    plain = psnr(scene, deconvolve(y, psf, method, sigma=SIGMA))
    aware = psnr(scene, deconvolve(y, psf, method, sigma=SIGMA, saturation=level))
    assert aware > plain


def test_unknown_boundary_on_motion_blur(scene):
    from eoqual.deconvolution.psf import motion_psf

    psf = motion_psf(9, np.pi / 6)
    y = simulate_observation(scene, psf, SIGMA, boundary="reflect", seed=0)
    for method in ("tikhonov", "tv_primal_dual"):
        r = psnr(scene, deconvolve(y, psf, method, sigma=SIGMA, boundary="reflect"))
        u = psnr(scene, deconvolve(y, psf, method, sigma=SIGMA, boundary="unknown"))
        assert u > r


def test_landweber_barzilai_borwein_fewer_iterations(scene, psf):
    y = simulate_observation(scene, psf, 0.001, boundary="periodic", seed=0)
    counts, scores = {}, {}
    for acc in (False, True):
        it = []
        out = deconvolve(y, psf, "landweber", sigma=0.001, acceleration=acc, n_iter=3000,
                         boundary="periodic", callback=lambda k, x: it.append(k))
        counts[acc], scores[acc] = len(it), psnr(scene, out)
    assert counts[True] < 0.6 * counts[False]
    assert scores[True] > scores[False] - 0.3


@pytest.mark.slow
def test_pnp_with_mask(scene, blurred, psf, dead_lines):
    pytest.importorskip("eoqual.denoising")
    damaged, mask = dead_lines
    out = deconvolve(damaged, psf, "pnp_hqs", sigma=SIGMA, mask=mask, denoiser="tv_chambolle")
    assert psnr(scene, out) > psnr(scene, blurred) + 1.5
