"""Déconvolution aveugle (a priori L0) : PSF estimée et gain."""
import numpy as np
import pytest

from _helpers import gain_db
from conftest import SIGMA
from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.backends.blind import blind_deconvolve
from eoqual.deconvolution.psf import gaussian_psf, motion_psf
from eoqual.deconvolution.simulate import simulate_observation


@pytest.fixture(scope="module")
def big_scene():
    from skimage import color, data

    return color.rgb2gray(data.astronaut())[100:356, 150:406].astype(np.float64)


@pytest.mark.slow
@pytest.mark.parametrize("truth", [gaussian_psf(2.0, 15), motion_psf(9, 0.5, 15)])
def test_blind_recovers_psf(big_scene, truth):
    y = simulate_observation(big_scene, truth, SIGMA, boundary="reflect", seed=0)
    x, h = blind_deconvolve(y, 15, SIGMA)
    assert h.shape == (15, 15) and h.sum() == pytest.approx(1.0) and h.min() >= 0
    delta = np.zeros((15, 15))
    delta[7, 7] = 1.0
    assert np.abs(h - truth).sum() < 0.7 * np.abs(delta - truth).sum()
    assert gain_db(big_scene, y, x) > 0.3  # gain modeste mais positif (BENCHMARK.md)


def test_blind_runner_signature(blurred, psf):
    out = deconvolve(blurred, None, "blind_l0", sigma=SIGMA, psf_size=9, n_outer=2, final_n_iter=30)
    assert out.shape == blurred.shape and np.isfinite(out).all()
    out = deconvolve(blurred, psf, "blind_l0", sigma=SIGMA, n_outer=2, final_n_iter=30)
    assert out.shape == blurred.shape
