#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scènes synthétiques communes aux tests (vérité terrain connue)."""
import os

import numpy as np
import pytest

from eoqual.deconvolution.psf import gaussian_psf
from eoqual.deconvolution.simulate import simulate_observation

#: Graine des tirages aléatoires, modifiable par EOQUAL_TEST_SEED.
SEED = int(os.environ.get("EOQUAL_TEST_SEED", "0"))
SIGMA = 0.01


@pytest.fixture(scope="session")
def scene():
    """« astronaut » (NASA, domaine public, scikit-image), 128×128, niveaux de gris dans [0, 1]."""
    from skimage import color, data

    return color.rgb2gray(data.astronaut())[150:278, 200:328].astype(np.float64)


@pytest.fixture(scope="session")
def psf():
    return gaussian_psf(1.6)


@pytest.fixture(scope="session")
def blurred(scene, psf):
    return simulate_observation(scene, psf, SIGMA, boundary="reflect", seed=SEED)
