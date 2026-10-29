"""
Registre central des méthodes de déconvolution disponibles dans
``eoqual.deconvolution``.

Chaque entrée décrit :

- ``family`` : famille de la méthode (linéaire, itérative, variationnelle,
  Plug-and-Play, aveugle) ;
- ``extra`` : extra pip requis (``None`` = socle) ;
- ``license`` : licence du code qui fait réellement le calcul ;
- ``restriction`` : restriction d'usage à connaître (``None`` si aucune
  au-delà de la conservation des notices) ;
- ``reference`` : référence bibliographique courte (complète dans
  ``BIBLIOGRAPHY.md``).
"""

from collections import OrderedDict

_OWN = "MIT (ce paquet)"

METHODS_CONFIGS: OrderedDict = OrderedDict(
    {
        "inverse": {
            "family": "linéaire",
            "extra": None,
            "license": _OWN,
            "restriction": None,
            "reference": "filtre inverse tronqué (Gonzalez & Woods)",
        },
        "wiener": {
            "family": "linéaire",
            "extra": None,
            "license": _OWN,
            "restriction": None,
            "reference": "Wiener (1949) ; spectre en 1/f² : Field (1987), JOSA A 4(12)",
        },
        "wiener_unsupervised": {
            "family": "linéaire",
            "extra": None,
            "license": "BSD-3-Clause (scikit-image)",
            "restriction": None,
            "reference": "Orieux, Giovannelli & Rodet (2010), JOSA A 27(7)",
        },
        "tikhonov": {
            "family": "linéaire",
            "extra": None,
            "license": _OWN,
            "restriction": None,
            "reference": "Tikhonov & Arsenin (1977) ; GCV : Golub, Heath & Wahba (1979)",
        },
        "richardson_lucy": {
            "family": "itérative",
            "extra": None,
            "license": _OWN,
            "restriction": None,
            "reference": "Richardson (1972) ; Lucy (1974) ; Biggs & Andrews (1997) ; Dey et al. (2006)",
        },
        "landweber": {
            "family": "itérative",
            "extra": None,
            "license": _OWN,
            "restriction": None,
            "reference": "Landweber (1951) ; Bertero & Boccacci (1998)",
        },
        "fista_wavelet": {
            "family": "variationnelle",
            "extra": None,
            "license": _OWN + " / MIT (PyWavelets)",
            "restriction": None,
            "reference": "Beck & Teboulle (2009), SIAM J. Imaging Sci. 2(1)",
        },
        "tv_primal_dual": {
            "family": "variationnelle",
            "extra": None,
            "license": _OWN,
            "restriction": None,
            "reference": "Condat (2013), JOTA 158 ; Vũ (2013), Adv. Comput. Math. 38",
        },
        "pnp_hqs": {
            "family": "Plug-and-Play",
            "extra": "pnp (deep pour les réseaux)",
            "license": _OWN + " + licence du débruiteur (eoqual-denoising)",
            "restriction": "celle du débruiteur choisi (voir eoqual-denoising list_methods())",
            "reference": "Zhang et al. (2017) CVPR ; Zhang et al. (2021) IEEE TPAMI 44(10)",
        },
        "pnp_admm": {
            "family": "Plug-and-Play",
            "extra": "pnp (deep pour les réseaux)",
            "license": _OWN + " + licence du débruiteur (eoqual-denoising)",
            "restriction": "celle du débruiteur choisi (voir eoqual-denoising list_methods())",
            "reference": "Venkatakrishnan et al. (2013) GlobalSIP ; Chan et al. (2017) IEEE TCI 3(1)",
        },
        "blind_l0": {
            "family": "aveugle",
            "extra": None,
            "license": _OWN,
            "restriction": "expérimentale : vérifier la PSF estimée (blind_deconvolve)",
            "reference": "Xu, Zheng & Jia (2013) CVPR ; Pan et al. (2014) CVPR",
        },
    }
)
