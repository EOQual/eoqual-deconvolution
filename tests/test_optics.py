"""PSF optique aberrée, Wiener–Hunt et déconvolution aveugle contrainte par l'optique."""
import numpy as np
import pytest

from _helpers import gain_db
from conftest import SIGMA
from eoqual.core.kernel import psf2otf
from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.backends.linear import wiener_hunt_parameters
from eoqual.deconvolution.backends.optics_blind import (
    OpticalModel, _smooth_field, estimate_optical_parameters, optics_blind_deconvolve,
)
from eoqual.deconvolution.psf import defocus_to_zernike, optical_psf, system_psf
from eoqual.deconvolution.simulate import simulate_observation
from eoqual.deconvolution.tuning.criteria import wiener_hunt_evidence


# --- optical_psf -----------------------------------------------------------

@pytest.mark.parametrize("q, eps", [(1.0, 0.0), (2.0, 0.3), (0.7, 0.2)])
def test_optical_psf_matches_analytic_airy(q, eps):
    # system_psf : Airy annulaire analytique ; λ = Q, D = f = p = 1 → Q = λf/(Dp)
    a = optical_psf(q, obscuration=eps, size=31, oversample=7)
    b = system_psf(q, 1.0, 1.0, 1.0, obscuration=eps, size=31)
    assert np.abs(a - b).max() < 0.01 * b.max()


def test_optical_psf_mtf_below_diffraction():
    # Inégalité de Schwarz : une aberration ne relève la FTM à aucune fréquence
    ref = np.abs(psf2otf(optical_psf(1.0, obscuration=0.3, size=21), (64, 64)))
    for z in ({4: 0.1}, {6: 0.08, 8: 0.05}, {4: 0.2, 11: 0.05}):
        ab = np.abs(psf2otf(optical_psf(1.0, zernike=z, obscuration=0.3, size=21), (64, 64)))
        assert np.all(ab <= ref + 2e-3)


def test_optical_psf_sign_symmetry_and_names():
    h_plus = optical_psf(1.5, zernike={"defocus": 0.1, "coma_x": 0.05})
    h_minus = optical_psf(1.5, zernike={4: -0.1, 8: -0.05})
    # φ → −φ : PSF tournée de 180°, même module de FTO
    np.testing.assert_allclose(h_minus, h_plus[::-1, ::-1], atol=1e-10)
    assert h_plus.sum() == pytest.approx(1.0) and h_plus.shape[0] % 2 == 1
    with pytest.raises(ValueError):
        optical_psf(0.0)
    with pytest.raises(KeyError):
        optical_psf(1.0, zernike={"inconnu": 0.1})


def test_defocus_to_zernike():
    # W020 = Δz/(8N²) ; a4 = W020/(2√3) : Δz = 8N²·2√3·λ donne une onde RMS
    n, lam = 8.0, 0.5e-6
    assert defocus_to_zernike(8 * n**2 * 2 * np.sqrt(3) * lam, n, lam) == pytest.approx(1.0)


def test_polychromatic_is_weighted_sum():
    h = optical_psf([1.0, 1.2], weights=[1.0, 0.0], size=15)
    np.testing.assert_allclose(h, optical_psf(1.0, size=15), atol=1e-12)


# --- Wiener–Hunt -----------------------------------------------------------

def test_wiener_hunt_estimates_sigma(blurred, psf):
    est = wiener_hunt_parameters(blurred, psf)
    assert est["sigma"] == pytest.approx(SIGMA, rel=0.15)
    assert est["lam"] > 0


def test_wiener_hunt_close_to_tikhonov_gcv(scene, blurred, psf):
    wh = deconvolve(blurred, psf, "wiener_hunt")
    tk = deconvolve(blurred, psf, "tikhonov")
    assert gain_db(scene, blurred, wh) > gain_db(scene, blurred, tk) - 0.5
    # λ imposé : même filtre que Tikhonov laplacien
    np.testing.assert_allclose(deconvolve(blurred, psf, "wiener_hunt", lam=1e-3),
                               deconvolve(blurred, psf, "tikhonov", lam=1e-3), atol=1e-10)


def test_evidence_known_sigma_is_proper_likelihood(blurred, psf):
    from scipy import fft as spfft
    from eoqual.deconvolution.backends._common import regularizer_power, setup

    pb = setup(blurred, psf, "reflect")
    Y, a2, L2 = spfft.fft2(pb.y), pb.op.abs2, regularizer_power("laplacian", pb.y.shape)
    j_free, s2 = wiener_hunt_evidence(Y, a2, L2, 1e-3)
    j_fixed, _ = wiener_hunt_evidence(Y, a2, L2, 1e-3, sigma=np.sqrt(s2))
    # σ profilé = σ imposé à sa valeur optimale : mêmes critères à une constante près (M)
    m = int((L2 > 0).sum() - (L2[0, 0] > 0))
    assert j_fixed - j_free == pytest.approx(m, rel=1e-6)
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "tikhonov", rule="inconnue")


# --- blind_optics ----------------------------------------------------------

@pytest.fixture(scope="module")
def big_scene():
    from skimage import color, data

    return color.rgb2gray(data.astronaut())[100:356, 150:406].astype(np.float64)


def test_optical_model_validation():
    with pytest.raises(ValueError):
        OpticalModel(1.0, free={})
    with pytest.raises(ValueError):
        OpticalModel(1.0, free={"defocus": (0.2, 0.1)})
    with pytest.raises(ValueError):
        OpticalModel(1.0, free={"defocus": (0, 0.2)}, fixed={4: 0.1})
    with pytest.raises(ValueError):
        OpticalModel(1.0, free={"jitter": (-0.1, 0.5)})
    m = OpticalModel(1.0, free={"defocus": (0, 0.3), "astig_0": (-0.1, 0.1)}, fixed={"jitter": 0.3})
    assert m.names == [4, 6]
    assert m.as_dict([0.1, 0.02]) == {"jitter": 0.3, "defocus": 0.1, "astig_0": 0.02}


@pytest.mark.slow
def test_blind_optics_recovers_defocus(big_scene):
    model = OpticalModel(1.0, free={"defocus": (0.0, 0.4)}, fixed={"jitter": 0.35}, obscuration=0.3, size=15)
    truth = model.psf([0.25])
    y = simulate_observation(big_scene, truth, SIGMA, boundary="reflect", seed=0)
    theta, s = estimate_optical_parameters(y, model)
    assert theta[0] == pytest.approx(0.25, abs=0.05)
    assert s == pytest.approx(SIGMA, rel=0.3)
    res = optics_blind_deconvolve(y, model, SIGMA)
    assert gain_db(big_scene, y, res.image) > gain_db(big_scene, y, deconvolve(y, model.psf([0.0]), "tv_primal_dual", sigma=SIGMA))


def test_blind_optics_runner_and_tiles(blurred):
    out = deconvolve(blurred, None, "blind_optics", sigma=SIGMA, sampling=1.0, free={"defocus": (0, 0.3)},
                     fixed={"jitter": 0.4}, psf_size=9, n_starts=1, restore="wiener_hunt")
    assert out.shape == blurred.shape and np.isfinite(out).all()
    with pytest.raises(ValueError):
        deconvolve(blurred, np.ones((3, 3)), "blind_optics", sampling=1.0)
    with pytest.raises(ValueError):
        deconvolve(blurred, None, "blind_optics")
    model = OpticalModel(1.0, free={"defocus": (0, 0.3)}, fixed={"jitter": 0.4}, size=9)
    res = optics_blind_deconvolve(blurred, model, SIGMA, tile_size=64, overlap=0.5, field_model="constant",
                                  n_starts=1, restore="wiener_hunt")
    assert len(res.psf) == len(res.tiles) == len(res.params) == 9
    assert len({round(p["defocus"], 9) for p in res.params}) == 1      # champ constant
    assert res.image.shape == blurred.shape and np.isfinite(res.image).all()


def test_field_models():
    c = np.array([[u, v] for u in (-0.3, 0.0, 0.3) for v in (-0.3, 0.0, 0.3)])
    plane = 0.1 + 0.2 * c[:, :1] - 0.1 * c[:, 1:]
    noisy = plane + 0.01 * np.random.default_rng(0).standard_normal(plane.shape)
    np.testing.assert_allclose(_smooth_field(noisy, c, "plane", [(-1, 1)]), plane, atol=0.015)
    quad = 0.1 + 0.5 * (c[:, :1] ** 2 + c[:, 1:] ** 2)
    np.testing.assert_allclose(_smooth_field(quad, c, "quadratic", [(-1, 1)]), quad, atol=1e-10)
    np.testing.assert_allclose(_smooth_field(noisy, c, "independent", [(-1, 1)]), noisy)
    with pytest.raises(ValueError):
        _smooth_field(noisy[:4], c[:4], "quadratic", [(-1, 1)])
