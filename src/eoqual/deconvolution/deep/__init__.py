"""
Ancien emplacement des méthodes apprises — **déprécié** depuis la version
0.6.0, conservé pour compatibilité.

Les réseaux de neurones d'EOQual sont regroupés dans eoqual-neural :

- réseau USRNet, poids et boucle d'apprentissage : ``eoqual.neural.unrolled``,
  ``eoqual.neural.weights`` ;
- méthode ``"usrnet"`` de ce paquet (bords, échelle) :
  :mod:`eoqual.deconvolution.backends.usrnet` ;
- paires d'entraînement de déconvolution et ``train_usrnet`` :
  :mod:`eoqual.deconvolution.training`.
"""
import warnings

warnings.warn(
    "eoqual.deconvolution.deep est déprécié : utiliser eoqual.neural (réseau, poids), "
    "eoqual.deconvolution.backends.usrnet et eoqual.deconvolution.training.",
    DeprecationWarning,
    stacklevel=2,
)
