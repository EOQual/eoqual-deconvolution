"""Déprécié : voir :mod:`eoqual.neural.unrolled.usrnet` et :mod:`eoqual.deconvolution.backends.usrnet`."""
from eoqual.neural.unrolled.usrnet import USRNet, data_step, load_usrnet, p2o, select_device  # noqa: F401

from ..backends.usrnet import consistent_extension as _consistent_extension  # noqa: F401
from ..backends.usrnet import usrnet_deconvolve  # noqa: F401

__all__ = ["USRNet", "load_usrnet", "usrnet_deconvolve", "p2o", "data_step", "select_device"]
