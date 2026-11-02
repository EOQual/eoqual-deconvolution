"""
Solution de référence : la TV primal-duale atteint l'optimum du problème
convexe, calculé indépendamment par CVXPY (dépendance de développement,
test sauté sans elle).
"""
import numpy as np
import pytest

from eoqual.core.kernel import convolve_circular
from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.operators import BlurOperator
from eoqual.deconvolution.psf import gaussian_psf
from eoqual.deconvolution.backends.sparse import tv_objective

cp = pytest.importorskip("cvxpy")


@pytest.mark.slow
def test_tv_primal_dual_reaches_cvxpy_optimum(scene):
    n, sigma, lam = 24, 0.01, 2.0
    x0 = scene[40 : 40 + n, 40 : 40 + n]
    psf = gaussian_psf(1.2, 7)
    rng = np.random.default_rng(0)
    y = convolve_circular(x0, psf) + rng.normal(0, sigma, x0.shape)
    weight = lam * sigma  # data_range=(0, 1) : unités inchangées

    # Matrices du flou circulaire et des différences avant périodiques.
    eye = np.eye(n * n).reshape(n * n, n, n)
    H = np.stack([convolve_circular(e, psf).ravel() for e in eye], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    Dy = np.zeros((n * n, n * n)); Dx = np.zeros((n * n, n * n))
    Dy[idx.ravel(), np.roll(idx, -1, axis=0).ravel()] += 1; Dy[idx.ravel(), idx.ravel()] -= 1
    Dx[idx.ravel(), np.roll(idx, -1, axis=1).ravel()] += 1; Dx[idx.ravel(), idx.ravel()] -= 1

    xv = cp.Variable(n * n)
    tv = cp.sum(cp.norm(cp.vstack([Dy @ xv, Dx @ xv]), 2, axis=0))
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(H @ xv - y.ravel()) + weight * tv))
    prob.solve(solver="CLARABEL")
    x_ref = xv.value.reshape(n, n)

    x = deconvolve(y, psf, "tv_primal_dual", sigma=sigma, lam=lam, boundary="periodic",
                   data_range=(0.0, 1.0), n_iter=20000, tol=1e-12)
    op = BlurOperator(psf, (n, n))
    f_ref, f = tv_objective(x_ref, y, op, weight), tv_objective(x, y, op, weight)
    assert f == pytest.approx(prob.value, rel=1e-6)  # même fonctionnelle que CVXPY
    assert f <= f_ref * (1 + 1e-4)
    assert np.abs(x - x_ref).max() < 5e-3
