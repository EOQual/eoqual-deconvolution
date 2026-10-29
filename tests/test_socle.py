"""Socle : contrôles, PSF de synthèse, simulation, opérateurs et bords."""
import numpy as np
import pytest

from eoqual.core.kernel import convolve_circular
from eoqual.deconvolution import operators as ops
from eoqual.deconvolution import psf as P
from eoqual.deconvolution.simulate import add_noise, blur, simulate_observation
from eoqual.deconvolution.utils import (
    apply_per_channel, check_image, check_psf, from_unit_range, resolve_sigma, to_unit_range,
)


def _centroid(h):
    y, x = np.indices(h.shape)
    return (h * y).sum() / h.sum(), (h * x).sum() / h.sum()


@pytest.mark.parametrize("h", [P.gaussian_psf(1.5), P.gaussian_psf(1.0, sigma_x=2.5, theta=0.3),
                               P.airy_psf(2.44), P.moffat_psf(2.0), P.disk_psf(3.0), P.motion_psf(7, 0.4)])
def test_psf_normalised_odd_centred(h):
    assert h.sum() == pytest.approx(1.0)
    assert h.shape[0] % 2 == 1 and h.shape[0] == h.shape[1]
    c = (h.shape[0] - 1) / 2
    np.testing.assert_allclose(_centroid(h), (c, c), atol=1e-6)
    assert h.min() >= 0


def test_psf_shapes_and_errors():
    assert P.gaussian_psf(2.0).shape == (17, 17)
    assert P.gaussian_psf(2.0, 10).shape == (11, 11)
    g = P.gaussian_psf(0.8, sigma_x=2.5)
    assert g[g.shape[0] // 2].std() > g[:, g.shape[1] // 2].std()  # plus large selon x
    m = P.motion_psf(9, 0.0)
    assert np.count_nonzero(m.sum(axis=1) > 1e-3) <= 3  # segment horizontal
    a = P.airy_psf(3.0)
    c = a.shape[0] // 2
    assert a[c, c + 3] < 0.02 * a[c, c]  # premier anneau noir
    for bad in (lambda: P.gaussian_psf(0), lambda: P.airy_psf(-1), lambda: P.disk_psf(0),
                lambda: P.motion_psf(0), lambda: P.moffat_psf(1, 0.5)):
        with pytest.raises(ValueError):
            bad()


def test_center_psf_recentres_offset_peak():
    h = np.zeros((21, 21))
    h[3:8, 12:17] = P.gaussian_psf(0.8, 5)
    out = P.center_psf(h, 15)
    assert out.shape == (15, 15)
    np.testing.assert_allclose(_centroid(out), (7, 7), atol=0.05)


def test_checks_and_ranges():
    with pytest.raises(ValueError):
        check_image(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        check_image(np.full((4, 4), np.nan))
    with pytest.raises(ValueError):
        check_psf(-np.ones((3, 3)))
    with pytest.raises(ValueError):
        check_psf(np.ones((9, 9)), (4, 4))
    assert check_psf(np.ones((3, 3))).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        resolve_sigma(np.zeros((8, 8)), -1.0)
    x = np.linspace(10, 20, 64).reshape(8, 8)
    u, lo, sc = to_unit_range(x)
    assert (u.min(), u.max()) == (0.0, 1.0)
    np.testing.assert_allclose(from_unit_range(u, lo, sc), x)


def test_noise_estimate_on_blurred_scene(scene, psf):
    y = simulate_observation(scene, psf, 0.02, seed=3)
    assert resolve_sigma(y, None) == pytest.approx(0.02, rel=0.15)


def test_blur_periodic_and_reflect_agree_inside(scene, psf):
    a, b = blur(scene, psf, "periodic"), blur(scene, psf, "reflect")
    m = psf.shape[0]
    np.testing.assert_allclose(a[m:-m, m:-m], b[m:-m, m:-m], atol=1e-12)
    with pytest.raises(ValueError):
        blur(scene, psf, "zero")


def test_add_noise_statistics():
    x = np.full((256, 256), 0.5)
    assert add_noise(x, 0.1, seed=0).std() == pytest.approx(0.1, rel=0.03)
    y = add_noise(x, 0.0, gain=100.0, seed=0)
    assert y.var() == pytest.approx(0.5 / 100.0, rel=0.05)
    with pytest.raises(ValueError):
        add_noise(x, gain=0.0)


def test_gradient_divergence_adjoint():
    rng = np.random.default_rng(0)
    x, p = rng.normal(size=(17, 23)), rng.normal(size=(2, 17, 23))
    assert np.sum(ops.gradient(x) * p) == pytest.approx(-np.sum(x * ops.divergence(p)))


def test_blur_operator_adjoint_and_lipschitz(psf):
    rng = np.random.default_rng(1)
    op = ops.BlurOperator(psf, (40, 36))
    x, y = rng.normal(size=(40, 36)), rng.normal(size=(40, 36))
    assert np.sum(op.forward(x) * y) == pytest.approx(np.sum(x * op.adjoint(y)))
    assert op.lipschitz == pytest.approx(1.0)
    np.testing.assert_allclose(op.forward(x), convolve_circular(x, psf), atol=1e-12)


@pytest.mark.parametrize("boundary", ops.BOUNDARIES)
def test_prepare_domain_roundtrip(scene, boundary):
    dom = ops.prepare_domain(scene, (9, 9), boundary, multiple=8)
    np.testing.assert_allclose(dom.restore(dom.image), scene, atol=1e-12)
    if boundary == "reflect":
        assert all(n % 8 == 0 for n in dom.image.shape)
        assert all(n >= s + 18 for n, s in zip(dom.image.shape, scene.shape))
    with pytest.raises(ValueError):
        ops.prepare_domain(scene, (9, 9), "zero")


def test_reflect_extension_is_continuous_and_periodic(scene):
    ext = ops.prepare_domain(scene, (9, 9), "reflect").image
    # Saut maximal entre pixels voisins, y compris au raccord périodique.
    jump = max(np.abs(np.diff(ext, axis=0, append=ext[:1])).max(),
               np.abs(np.diff(ext, axis=1, append=ext[:, :1])).max())
    inside = max(np.abs(np.diff(scene, axis=0)).max(), np.abs(np.diff(scene, axis=1)).max())
    assert jump <= inside + 1e-12


def test_apply_per_channel():
    img = np.stack([np.ones((4, 4)), 2 * np.ones((4, 4))], axis=0)
    out = apply_per_channel(lambda b, sigma: b * sigma, img, 0, sigma=[1.0, 3.0])
    np.testing.assert_allclose(out[1], 6.0)
    with pytest.raises(ValueError):
        apply_per_channel(lambda b, sigma: b, img, 0, sigma=[1.0])
