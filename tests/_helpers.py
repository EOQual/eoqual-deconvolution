"""Assistants communs aux tests."""
import numpy as np

from eoqual.metrics.fr.signal import psnr


def gain_db(reference, observed, restored):
    """Gain de PSNR (dB) de ``restored`` sur ``observed``."""
    assert restored.shape == observed.shape
    assert np.all(np.isfinite(restored))
    return psnr(reference, restored) - psnr(reference, observed)
