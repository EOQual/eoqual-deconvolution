"""Réglage des paramètres : GCV, discrépance, L-curve, espaces de recherche, optimisation."""
import numpy as np
import pytest

from conftest import SIGMA
from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.tuning import (
    SEARCH_SPACES, discrepancy_curve, gcv_curve, get_search_space, l_curve, load_search_space,
    optimize_parameters, score_image, validate_search_space,
)
from eoqual.metrics.fr.signal import psnr


def _oracle_lambda(scene, blurred, psf, reg):
    lams = np.logspace(-5, 0, 21)
    scores = [psnr(scene, deconvolve(blurred, psf, "tikhonov", lam=l, regularizer=reg)) for l in lams]
    return lams[int(np.argmax(scores))], max(scores)


@pytest.mark.parametrize("reg", ["gradient", "laplacian"])
def test_gcv_close_to_oracle(reg, scene, blurred, psf):
    lams, g, best = gcv_curve(blurred, psf, regularizer=reg)
    assert g.shape == lams.shape and np.isclose(g.min(), g).any()
    lam_o, score_o = _oracle_lambda(scene, blurred, psf, reg)
    assert 0.1 < best / lam_o < 10
    assert psnr(scene, deconvolve(blurred, psf, "tikhonov", lam=best, regularizer=reg)) > score_o - 0.3


def test_identity_gcv_underregularizes_discrepancy_fits(scene, blurred, psf):
    # Constat documenté (REFERENCE_TECHNIQUE.md §5.1) : avec L = I, GCV sous-régularise
    # (spectre de l'image loin d'être plat) ; la discrépance est alors proche de l'optimum.
    _, score_o = _oracle_lambda(scene, blurred, psf, "identity")
    gcv = psnr(scene, deconvolve(blurred, psf, "tikhonov", regularizer="identity"))
    disc = psnr(scene, deconvolve(blurred, psf, "tikhonov", regularizer="identity", rule="discrepancy",
                                  sigma=SIGMA))
    assert gcv > psnr(scene, blurred)
    assert disc > score_o - 0.3 and disc > gcv


def test_discrepancy_and_default_rule(scene, blurred, psf):
    lams, res, s2, lam = discrepancy_curve(blurred, psf, SIGMA)
    assert np.all(np.diff(res) >= -1e-12)  # résidu croissant avec λ
    i = np.searchsorted(lams, lam)
    assert res[max(i - 1, 0)] <= s2 * 1.05 and res[min(i, len(res) - 1)] >= s2 * 0.95
    out = deconvolve(blurred, psf, "tikhonov", sigma=SIGMA, rule="discrepancy")
    assert psnr(scene, out) > psnr(scene, blurred)
    with pytest.raises(ValueError):
        deconvolve(blurred, psf, "tikhonov", rule="oracle")


def test_l_curve(blurred, psf):
    out = l_curve(blurred, psf)
    assert set(out) == {"lams", "log_residual", "log_seminorm", "curvature", "lam_corner"}
    assert np.all(np.diff(out["log_residual"]) >= -1e-9)
    assert np.all(np.diff(out["log_seminorm"]) <= 1e-9)


@pytest.mark.parametrize("name", sorted(SEARCH_SPACES))
def test_default_spaces_valid(name):
    from eoqual.deconvolution import METHODS_CONFIGS

    assert name in METHODS_CONFIGS
    space = get_search_space(name)
    assert set(space["initial_guess"]) == {d["name"] for d in space["search_space"]}


def test_space_validation_errors(tmp_path):
    with pytest.raises(ValueError):
        validate_search_space({"search_space": []})
    with pytest.raises(ValueError):
        validate_search_space({"search_space": [{"name": "a", "type": "float", "low": 1, "high": 0}]})
    with pytest.raises(ValueError):
        validate_search_space({"search_space": [{"name": "a", "type": "float", "low": 0, "high": 1}],
                               "initial_guess": {"b": 1}})
    with pytest.raises(ValueError):
        get_search_space("wiener_unsupervised")
    f = tmp_path / "s.jsonc"
    f.write_text('{"search_space": [{"name": "lam", "type": "float", "low": 0.1, "high": 1,}], // c\n}')
    assert load_search_space(f)["search_space"][0]["name"] == "lam"


def test_score_image(scene, blurred):
    assert score_image("psnr", blurred, scene) == pytest.approx(psnr(scene, blurred))
    assert 0 < score_image("ssim", blurred, scene) < 1
    assert score_image("sharpness_index", scene) > score_image("sharpness_index", blurred)
    with pytest.raises(ValueError):
        score_image("psnr", blurred)
    with pytest.raises(ValueError):
        score_image("mse", blurred, scene)


@pytest.mark.slow
def test_optimize_never_worse_than_defaults(scene, blurred, psf):
    pytest.importorskip("skopt")
    res = optimize_parameters(blurred, psf, "tv_primal_dual", reference=scene, sigma=SIGMA, n_calls=8,
                              fixed={"n_iter": 80})
    assert res.metric == "psnr" and len(res.history) == 8
    assert res.gain >= 0
    out = deconvolve(blurred, psf, "tv_primal_dual", **res.best_kwargs)
    assert psnr(scene, out) == pytest.approx(res.best_score)


@pytest.mark.slow
def test_optimize_no_reference_uses_sharpness_index(blurred, psf):
    pytest.importorskip("skopt")
    res = optimize_parameters(blurred, psf, "landweber", sigma=SIGMA, n_calls=5)
    assert res.metric == "sharpness_index"
    assert res.best_kwargs["sigma"] == pytest.approx(SIGMA)
    with pytest.raises(ValueError):
        optimize_parameters(blurred, psf, "landweber", metric="psnr", n_calls=5)
