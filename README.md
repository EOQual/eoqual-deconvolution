<p align="center">
  <img src="resources/eoqual_rounded.png" alt="EOQual" width="280">
</p>

# eoqual-deconvolution

Collection de méthodes de déconvolution d'images, pensée pour
l'observation de la Terre : PSF connue (méthodes linéaires, itératives,
variationnelles, Plug-and-Play) ou estimée (déconvolution aveugle),
traitement des bords, réglage automatique des paramètres. Voir
**[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)** pour le détail de
chaque méthode (principe, implémentation, limites),
**[BENCHMARK.md](BENCHMARK.md)** pour la comparaison chiffrée,
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** pour l'audit des
licences et **[BIBLIOGRAPHY.md](BIBLIOGRAPHY.md)** pour les publications
de référence.

## Sommaire

1. [Installation](#installation)
2. [Démarrage rapide](#démarrage-rapide)
3. [Catalogue des méthodes](#catalogue-des-méthodes)
4. [Conventions](#conventions)
5. [Modules EOQual utilisés](#modules-eoqual-utilisés)
6. [Licence](#licence)
7. [Contribuer](#contribuer)

---

## Installation

### 1. Socle

```bash
pip install git+https://github.com/EOQual/eoqual-deconvolution.git
```

Installe les 8 méthodes à PSF connue hors Plug-and-Play et la méthode
aveugle, le traitement des bords, la simulation et le réglage par GCV ou
discrépance. Dépendances : numpy, scipy, scikit-image, PyWavelets,
plotly, rich, et les modules EOQual eoqual-core, eoqual-metrics et
eoqual-viz, tous sous licence permissive.

### 2. Extras optionnels

| Extra | Commande | Débloque | Pourquoi optionnel |
|---|---|---|---|
| `pnp` | `pip install "eoqual-deconvolution[pnp]"` | `pnp_hqs`, `pnp_admm` avec les débruiteurs du socle d'eoqual-denoising (`nl_ridge`, `nl_bayes`, `tv_chambolle`...) | dépend d'un autre module |
| `deep` | `pip install "eoqual-deconvolution[deep]"` | Plug-and-Play avec les débruiteurs à réseau d'eoqual-denoising (`drunet`...) | dépendance lourde (PyTorch) ; code et poids DRUNet BSD-3-Clause |
| `optim` | `pip install "eoqual-deconvolution[optim]"` | `optimize_parameters` (optimisation bayésienne) | scikit-optimize (BSD-3-Clause), inutile pour déconvoluer |

Sans l'extra requis, l'appel de la méthode lève une erreur explicite qui
indique quoi installer. Aucune méthode ne plante à l'*import* du paquet.

```python
from eoqual.deconvolution import list_methods
list_methods()      # méthodes, famille, extra, licence, référence
```

---

## Démarrage rapide

```python
from eoqual.deconvolution import deconvolve
from eoqual.deconvolution.psf import gaussian_psf
from eoqual.deconvolution.simulate import simulate_observation

psf = gaussian_psf(2.0)                                   # σ = 2 pixels
blurred = simulate_observation(scene, psf, sigma=0.01, boundary="reflect", seed=0)

x = deconvolve(blurred, psf, "wiener", sigma=0.01)        # instantané
x = deconvolve(blurred, psf, "tikhonov")                  # λ choisi par GCV
x = deconvolve(blurred, psf, "tv_primal_dual", sigma=0.01)
x = deconvolve(blurred, psf, "pnp_hqs", sigma=0.01)       # extra "pnp" (NL-Ridge)
x = deconvolve(blurred, psf, "pnp_hqs", sigma=0.01, denoiser="drunet")   # extra "deep"
```

`sigma` est l'écart-type du bruit **dans les unités de l'image** ; s'il
est omis, il est estimé sur l'image (Donoho & Johnstone).

**Déconvolution aveugle** (PSF inconnue) :

```python
from eoqual.deconvolution.backends.blind import blind_deconvolve
x, psf_est = blind_deconvolve(blurred, psf_size=15, sigma=0.01)
```

**Réglage des paramètres** :

```python
from eoqual.deconvolution.tuning import optimize_parameters, gcv_curve
res = optimize_parameters(blurred, psf, "tv_primal_dual", reference=scene)   # extra "optim"
x = deconvolve(other, psf, "tv_primal_dual", **res.best_kwargs)
res = optimize_parameters(blurred, psf, "tv_primal_dual")   # sans référence : indice SI
```

**Figures** (Plotly, `fig.show()` ou `fig.write_html(...)`) :

```python
from eoqual.deconvolution.plotting import plot_restoration, plot_mtf, plot_gcv
plot_restoration(blurred, {"Wiener": x}, scene).show()
```

Exemples prêts à l'emploi dans `examples/` :

| Script | Usage |
|---|---|
| `quickstart.py` | simuler, déconvoluer avec quatre méthodes, figure comparative |
| `tuning.py` | GCV, discrépance, optimisation avec et sans référence |
| `blind.py` | déconvolution aveugle d'un flou de bougé |
| `benchmark.py` | tableaux de `BENCHMARK.md` |

---

## Catalogue des méthodes

| Méthode | Famille | Principe | Extra |
|---|---|---|---|
| `inverse` | linéaire | filtre inverse tronqué (référence) | — |
| `wiener` | linéaire | Wiener, spectre du signal en loi de puissance ajusté sur l'image | — |
| `wiener_unsupervised` | linéaire | Wiener bayésien non supervisé (Orieux et al. 2010, scikit-image) | — |
| `tikhonov` | linéaire | Tikhonov (identité, gradient, laplacien), λ par GCV ou discrépance | — |
| `richardson_lucy` | itérative | Richardson-Lucy, accélération Biggs-Andrews, option TV | — |
| `landweber` | itérative | gradient projeté, arrêt par discrépance | — |
| `fista_wavelet` | variationnelle | parcimonie en ondelettes, FISTA, *cycle spinning* | — |
| `tv_primal_dual` | variationnelle | variation totale, primal-dual de Condat-Vũ | — |
| `pnp_hqs` | Plug-and-Play | éclatement semi-quadratique, débruiteur eoqual-denoising | `pnp` / `deep` |
| `pnp_admm` | Plug-and-Play | ADMM, débruiteur eoqual-denoising | `pnp` / `deep` |
| `blind_l0` | aveugle | a priori L0 sur les gradients, multi-échelle (expérimental) | — |

Toutes les méthodes sont des implémentations originales sous MIT, écrites
à partir des publications citées (sauf `wiener_unsupervised`, qui appelle
scikit-image).

Résultats indicatifs (PSNR, image d'évaluation 512×256, bruit 0.01,
réglages par défaut ; détail dans [BENCHMARK.md](BENCHMARK.md)) :
`tv_primal_dual` est le meilleur compromis du socle (+3 à +6 dB sur
l'image observée, 2 s) ; `pnp_admm` gagne encore 0.4 à 2.5 dB, au prix
d'un temps de calcul 10 fois plus long ; les méthodes linéaires sont
instantanées (+2 à +4 dB) et sont le bon point de départ.

---

## Conventions

- **Appel unifié** : `deconvolve(image, psf, method, **kwargs)` ;
  chaque méthode a la signature `f(image, psf, sigma=None, ...)` et
  renvoie une image `float64` de même forme (vérifié par test).
- **Image** : 2D ; une image multi-bandes se traite avec
  `channel_axis=` (bande par bande, un σ par bande possible).
- **PSF** : 2D, normalisée en interne à une somme de 1, centre en
  `(kh // 2, kw // 2)` (préférer une taille impaire ;
  `psf.center_psf` recentre une PSF mesurée).
- **Bords** (`boundary=`) : `"reflect"` par défaut (prolongement miroir
  continu et périodique, recadré ensuite), `"perdecomp"` (décomposition
  périodique + lisse), `"periodic"` (aucun traitement, exact pour une
  simulation circulaire).
- **Paramètres sans dimension ou en multiples de σ** : les résultats ne
  dépendent pas des unités de l'image (8 bits, 12 bits, réflectance),
  vérifié par test.

---

## Modules EOQual utilisés

| Module | Usage |
|---|---|
| [eoqual-core](https://github.com/EOQual/eoqual-core) | `psf2otf`, convolution circulaire, estimation du bruit, profils de PSF intégrés sur le pixel, décomposition périodique + lisse |
| [eoqual-metrics](https://github.com/EOQual/eoqual-metrics) | PSNR, SSIM, indice de netteté SI (réglage, benchmark, figures) |
| [eoqual-viz](https://github.com/EOQual/eoqual-viz) | galerie d'images et courbes Plotly |
| [eoqual-denoising](https://github.com/EOQual/eoqual-denoising) | débruiteurs du Plug-and-Play (extras `pnp`, `deep`) |

---

## Licence

MIT — voir `license.txt`. Licences des dépendances et des extras :
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md).
