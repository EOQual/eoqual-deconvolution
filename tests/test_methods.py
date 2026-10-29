"""Méthodes non aveugles : contrat commun, gains, cas limites, invariance d'échelle."""
import inspect

import numpy as np
import pytest

from _helpers import gain_db
from conftest import SIGMA
from eoqual.deconvolution import METHODS_CONFIGS, deconvolve, get_method, list_methods
from eoqual.deconvolution.runner import _DISPATCH

#: Gain de PSNR minimal (dB) de chaque méthode, réglages par défaut, flou gaussien σ = 1.6 px, bruit 0.01.
MIN_GAIN = {
    "inverse": 0.5,
    "wiener": 1.5,
    "wiener_unsupervised": 1.0,
    "tikhonov": 1.5,
    "richardson_lucy": 1.5,
    "landweber": 1.5,
    "fista_wavelet": 1.5,
    "tv_primal_dual": 1.5,
}
FAST = list(MIN_GAIN)


def test_registry_consistent():
    assert list(_DISPATCH) == list(METHODS_CONFIGS)
    for cfg in METHODS_CONFIGS.values():
        assert {"family", "extra", "license", "restriction", "reference"} <= set(cfg)
    list_methods()
    with pytest.raises(ValueError):
        deconvolve(np.zeros((8, 8)), np.ones((3, 3)), "unknown")


@pytest.mark.parametrize("method", list(METHODS_CONFIGS))
def test_unified_signature(method):
    params = inspect.signature(get_method(method)).parameters
    names = list(params)
    assert names[:3] == ["image", "psf", "sigma"]
    assert params["sigma"].default is None
    assert "method" not in names
    assert "boundary" in names or "kwargs" in names


@pytest.mark.parametrize("method", FAST)
def test_gain(method, scene, blurred, psf):
    out = deconvolve(blurred, psf, method, sigma=SIGMA)
    assert out.dtype == np.float64
    assert gain_db(scene, blurred, out) > MIN_GAIN[method]


@pytest.mark.parametrize("method", ["inverse", "wiener", "tikhonov", "landweber", "tv_primal_dual"])
def test_delta_psf_keeps_noiseless_image(method, scene):
    delta = np.zeros((5, 5))
    delta[2, 2] = 1.0
    out = deconvolve(scene, delta, method, sigma=1e-4, **({"lam": 1e-6} if method == "tikhonov" else {}))
    np.testing.assert_allclose(out, scene, atol=2e-2)


@pytest.mark.parametrize("method", ["wiener", "tikhonov", "richardson_lucy", "fista_wavelet", "tv_primal_dual"])
def test_scale_invariance(method, blurred, psf):
    # Richardson-Lucy (modèle de Poisson) est invariant par changement d'échelle, pas par décalage.
    offset = 0.0 if method == "richardson_lucy" else 50.0
    a = deconvolve(blurred, psf, method, sigma=SIGMA)
    b = deconvolve(blurred * 1000.0 + offset, psf, method, sigma=SIGMA * 1000.0)
    np.testing.assert_allclose((b - offset) / 1000.0, a, atol=1e-3)


@pytest.mark.parametrize("boundary", ["reflect", "perdecomp", "periodic"])
def test_boundaries_run(boundary, scene, blurred, psf):
    out = deconvolve(blurred, psf, "wiener", sigma=SIGMA, boundary=boundary)
    assert out.shape == blurred.shape
    if boundary != "periodic":
        assert gain_db(scene, blurred, out) > 1.0


def test_periodic_exact_on_circular_blur(scene, psf):
    from eoqual.deconvolution.simulate import blur

    y = blur(scene, psf, "periodic")
    out = deconvolve(y, psf, "tikhonov", lam=1e-14, regularizer="identity", boundary="periodic")
    assert np.abs(out - scene).max() < 0.05


def test_reflect_beats_periodic_on_real_borders(scene, blurred, psf):
    from eoqual.metrics.fr.signal import psnr

    r = deconvolve(blurred, psf, "tikhonov", boundary="reflect")
    p = deconvolve(blurred, psf, "tikhonov", boundary="periodic")
    assert psnr(scene, r) > psnr(scene, p)


def test_channel_axis(scene, psf):
    from eoqual.deconvolution.simulate import simulate_observation

    rgb = np.stack([simulate_observation(scene * k, psf, SIGMA, seed=i) for i, k in enumerate((1.0, 0.5))], axis=-1)
    out = deconvolve(rgb, psf, "wiener", channel_axis=-1, sigma=[SIGMA, SIGMA])
    assert out.shape == rgb.shape
    np.testing.assert_allclose(out[..., 0], deconvolve(rgb[..., 0], psf, "wiener", sigma=SIGMA))


def test_parameter_errors(blurred, psf):
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "inverse", threshold=0.0)
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "wiener", nsr=-1.0)
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "tikhonov", regularizer="tv")
    for m in ("richardson_lucy", "landweber", "fista_wavelet", "tv_primal_dual"):
        with pytest.raises(ValueError):
            deconvolve(blurred, psf, m, n_iter=0)


def test_wiener_constant_nsr_equals_tikhonov_identity(blurred, psf):
    a = deconvolve(blurred, psf, "wiener", nsr=1e-2)
    b = deconvolve(blurred, psf, "tikhonov", lam=1e-2, regularizer="identity")
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_richardson_lucy_positive_and_acceleration(scene, blurred, psf):
    from eoqual.metrics.fr.signal import psnr

    shifted = blurred - 0.2  # valeurs négatives : décalage interne puis recalage
    out = deconvolve(shifted, psf, "richardson_lucy", n_iter=10)
    assert out.min() >= shifted.min() - 1e-9
    plain = psnr(scene, deconvolve(blurred, psf, "richardson_lucy", n_iter=10, acceleration=False))
    fast = psnr(scene, deconvolve(blurred, psf, "richardson_lucy", n_iter=10, acceleration=True))
    assert fast > plain
    tv = deconvolve(blurred, psf, "richardson_lucy", n_iter=60, tv_weight=0.005)
    assert np.isfinite(tv).all()


def test_landweber_discrepancy_stops_early(blurred, psf):
    iters = []
    deconvolve(blurred, psf, "landweber", sigma=SIGMA, n_iter=500, stop_at_discrepancy=True,
               callback=lambda k, x: iters.append(k))
    assert 0 < len(iters) < 500


def test_tv_objective_decreases(blurred, psf):
    from eoqual.deconvolution.backends._common import setup
    from eoqual.deconvolution.backends.sparse import tv_objective
    from eoqual.deconvolution.utils import to_unit_range

    pb = setup(blurred, psf, "reflect")
    yu, lo, sc = to_unit_range(pb.y)
    vals = []
    deconvolve(blurred, psf, "tv_primal_dual", sigma=SIGMA, n_iter=150, tol=0,
               callback=lambda k, x: vals.append(tv_objective((x - lo) / sc, yu, pb.op, 0.1 * SIGMA / sc)))
    assert vals[-1] < vals[0]
    assert vals[-1] <= min(vals[:10])
