"""Figures Plotly (construction seulement)."""
import numpy as np

from conftest import SIGMA
from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.plotting import (
    ConvergenceRecorder, plot_convergence, plot_gcv, plot_mtf, plot_optimization, plot_psf, plot_restoration,
)
from eoqual.deconvolution.psf import disk_psf
from eoqual.deconvolution.tuning.optimizer import OptimizationResult


def test_figures(scene, blurred, psf):
    out = deconvolve(blurred, psf, "wiener", sigma=SIGMA)
    fig = plot_restoration(blurred, {"Wiener": out}, scene)
    assert len(fig.data) == 3 and "dB" in fig.layout.annotations[0].text
    assert len(plot_psf({"a": psf, "b": disk_psf(3)}, log=True).data) == 2
    fig = plot_mtf({"g": psf}, n=64)
    assert fig.data[0].y[0] == np.float64(1.0) or abs(fig.data[0].y[0] - 1) < 1e-9
    rec = ConvergenceRecorder(scene)
    deconvolve(blurred, psf, "richardson_lucy", n_iter=5, callback=rec)
    assert len(rec.values) == 5
    assert len(plot_convergence({"RL": rec.values}).data) == 1
    assert len(plot_gcv(blurred, psf, sigma=SIGMA).data) == 2
    res = OptimizationResult("x", "psnr", {}, {}, 2.0, 1.0,
                             [{"score": 1.0}, {"score": None}, {"score": 2.0}])
    assert len(plot_optimization(res).data) == 2
