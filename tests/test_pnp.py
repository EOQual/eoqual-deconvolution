"""Plug-and-Play : débruiteur fourni, débruiteurs d'eoqual-denoising (extras pnp, deep)."""
import pytest

from _helpers import gain_db
from conftest import SIGMA
from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.backends.pnp import make_denoiser


def _gauss_denoiser(img, sigma):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(img, 0.5 + 10 * sigma)


@pytest.mark.parametrize("method", ["pnp_hqs", "pnp_admm"])
def test_callable_denoiser(method, scene, blurred, psf):
    calls = []
    out = deconvolve(blurred, psf, method, sigma=SIGMA, denoiser=_gauss_denoiser,
                     callback=lambda k, z: calls.append(k))
    assert len(calls) == (8 if method == "pnp_hqs" else 12)
    assert gain_db(scene, blurred, out) > 0.5


def test_denoising_tv_chambolle(scene, blurred, psf):
    pytest.importorskip("eoqual.denoising")
    out = deconvolve(blurred, psf, "pnp_hqs", sigma=SIGMA, denoiser="tv_chambolle")
    assert gain_db(scene, blurred, out) > 1.5


def test_unknown_denoiser():
    pytest.importorskip("eoqual.denoising")
    with pytest.raises(ValueError):
        make_denoiser("nope")


@pytest.mark.slow
def test_default_nl_ridge(scene, blurred, psf):
    pytest.importorskip("eoqual.denoising")
    out = deconvolve(blurred, psf, "pnp_hqs", sigma=SIGMA)
    assert gain_db(scene, blurred, out) > 2.5


@pytest.mark.slow
@pytest.mark.deep
def test_drunet_extra_deep(scene, blurred, psf):
    pytest.importorskip("eoqual.denoising")
    pytest.importorskip("deepinv")
    out = deconvolve(blurred, psf, "pnp_hqs", sigma=SIGMA, denoiser="drunet")
    assert gain_db(scene, blurred, out) > 2.5
