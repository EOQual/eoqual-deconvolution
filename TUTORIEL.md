# Tutoriel — la déconvolution pas à pas

Ce tutoriel part du modèle d'acquisition et arrive aux méthodes les plus
performantes du paquet, en montrant à chaque étape **pourquoi** la méthode
suivante est nécessaire. Tous les chiffres sont produits par
`python examples/tutorial.py`, qui enregistre aussi les figures
interactives correspondantes (`examples/output/tutorial_*.html`).

Scène : « astronaut » (NASA, domaine public, scikit-image), 256×256 en
niveaux de gris, dynamique [0, 1] ; bruit gaussien σ = 0.01.
Détails de chaque méthode : [REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md) ;
comparaison chiffrée complète : [BENCHMARK.md](BENCHMARK.md).

## Sommaire

1. [Le modèle direct](#1-le-modèle-direct)
2. [Inverser naïvement : pourquoi ça échoue](#2-inverser-naïvement--pourquoi-ça-échoue)
3. [Régulariser : Wiener et Tikhonov](#3-régulariser--wiener-et-tikhonov)
4. [Itérer : la semi-convergence](#4-itérer--la-semi-convergence)
5. [Un a priori parcimonieux : ondelettes et variation totale](#5-un-a-priori-parcimonieux--ondelettes-et-variation-totale)
6. [Plug-and-Play : un débruiteur comme a priori](#6-plug-and-play--un-débruiteur-comme-a-priori)
7. [Données réelles : bords, saturation, pixels manquants, PSF d'instrument](#7-données-réelles--bords-saturation-pixels-manquants-psf-dinstrument)
8. [Régler les paramètres](#8-régler-les-paramètres)
9. [Pour aller plus loin](#9-pour-aller-plus-loin)

---

## 1. Le modèle direct

Une image observée est la scène floutée par la réponse impulsionnelle de
l'instrument (PSF), plus du bruit : `y = h ∗ x + b`.

```python
from eoqual.deconvolution.psf import gaussian_psf
from eoqual.deconvolution.simulate import simulate_observation

psf = gaussian_psf(2.0)                                     # σ = 2 pixels
y = simulate_observation(scene, psf, 0.01, boundary="reflect", seed=0)
```

PSNR de l'image observée : **24.50 dB**.

Dans le domaine de Fourier, la convolution devient un produit,
`Y = H·X + B`. `H`, la fonction de transfert, a pour module la **FTM**
(`plotting.plot_mtf`) : elle vaut 1 à la fréquence nulle et décroît vers
les hautes fréquences. Toute la difficulté de la déconvolution est là :
là où `|H|` est petit, le signal est noyé dans le bruit. Un flou de bougé
est pire encore : sa FTM s'annule à certaines fréquences.

> **Pourquoi la FFT ?** Une convolution par `h` est une application
> linéaire : `y = H x`, avec `H` une matrice `N × N` (`N` pixels), trop
> grande pour être formée (10¹⁰ coefficients pour une image de 100 000
> pixels). Sur un domaine périodique, `H` est **circulante par blocs à
> blocs circulants** : la transformée de Fourier discrète la diagonalise,
> ses valeurs propres sont les coefficients de `H(ω)`. Appliquer `H`,
> `Hᵀ` ou `(HᵀH + λI)⁻¹` coûte alors une FFT, `O(N log N)`. C'est
> pourquoi toutes les méthodes du paquet travaillent sur un domaine
> périodique — et pourquoi le traitement des bords (§7) compte autant.

## 2. Inverser naïvement : pourquoi ça échoue

Diviser par `H` (`method="inverse"`) annule le flou… et multiplie le
bruit par `1/|H|`. On ne garde que les fréquences où `|H| ≥ seuil` :

| Seuil | 0.001 | 0.01 | 0.05 | 0.2 |
|---|---|---|---|---|
| PSNR | −2.69 dB | 16.89 dB | 26.59 dB | 26.20 dB |

Un seuil trop bas laisse exploser le bruit, un seuil trop haut coupe les
détails. Le filtre inverse ne sait pas faire mieux que ce compromis
grossier : il faut un **modèle** du bruit et de l'image.

## 3. Régulariser : Wiener et Tikhonov

Les méthodes linéaires pondèrent chaque fréquence selon le rapport entre
le signal attendu et le bruit, au lieu d'un tout-ou-rien.

```python
deconvolve(y, psf, "wiener", sigma=0.01)          # spectre du signal en loi de puissance, ajusté
deconvolve(y, psf, "tikhonov")                    # λ choisi par GCV, sans connaître σ
deconvolve(y, psf, "tikhonov", sigma=0.01, rule="discrepancy")
```

| Méthode | PSNR |
|---|---|
| Wiener | 27.52 dB |
| Tikhonov (GCV) | 27.60 dB |
| Tikhonov (discrépance) | 26.66 dB |

Le paramètre de régularisation se choisit sans image de référence :
validation croisée généralisée (GCV) ou principe de discrépance (le
résidu doit valoir le bruit). `plotting.plot_gcv` trace les deux
critères. GCV est le meilleur ici ; la discrépance est plus sûre sur un
flou de bougé (BENCHMARK.md).

## 4. Itérer : la semi-convergence

Richardson-Lucy (maximum de vraisemblance pour un bruit de Poisson) et
Landweber (descente de gradient) n'ont pas de terme de régularisation :
c'est le **nombre d'itérations** qui en tient lieu.

```python
from eoqual.deconvolution.plotting import ConvergenceRecorder, plot_convergence
rec = ConvergenceRecorder(scene)                  # PSNR à chaque itération
deconvolve(y, psf, "richardson_lucy", n_iter=80, tv_weight=0.0, callback=rec)
plot_convergence({"RL accéléré": rec.values}).show()
```

| | meilleur PSNR | à la 80ᵉ itération |
|---|---|---|
| Richardson-Lucy | 27.78 dB (itération 80) | 27.78 dB |
| Richardson-Lucy accéléré (Biggs-Andrews) | 27.73 dB (itération 22) | 22.66 dB |
| accéléré + TV (`tv_weight=0.005`, défaut) | 27.68 dB (itération 80) | 27.68 dB |

L'accélération atteint le même niveau en 4 fois moins d'itérations… puis
l'image se dégrade : les itérations finissent par ajuster le bruit
(**semi-convergence**). Deux parades : arrêter à temps (l'arrêt par
discrépance de Landweber), ou ajouter un peu de régularisation — c'est le
défaut de Richardson-Lucy (`tv_weight=0.005`, 50 itérations) : la
variation totale (§5) stabilise les itérations, le résultat ne dépend
presque plus de leur nombre. Le tableau utilise `tv_weight=0` pour les
deux premières lignes, pour montrer le phénomène.

## 5. Un a priori parcimonieux : ondelettes et variation totale

Une image naturelle est **parcimonieuse** : peu de coefficients
d'ondelettes significatifs, ou un gradient nul presque partout (zones
homogènes séparées de contours). Pénaliser la norme ℓ1 de ces
coefficients (et non leur énergie, comme Tikhonov) supprime le bruit sans
lisser les contours.

```python
deconvolve(y, psf, "fista_wavelet", sigma=0.01)   # ‖Wx‖₁, FISTA
deconvolve(y, psf, "tv_primal_dual", sigma=0.01)  # variation totale, Condat-Vũ
```

| Méthode | PSNR |
|---|---|
| FISTA ondelettes | 27.83 dB |
| Variation totale | **28.10 dB** |

Ces problèmes sont convexes mais non différentiables : les algorithmes
proximaux (seuillage doux pour les ondelettes, projection duale pour la
TV) les résolvent. La TV primal-duale atteint l'optimum calculé
indépendamment par CVXPY (`tests/test_reference_cvxpy.py`).

## 6. Plug-and-Play : un débruiteur comme a priori

Les algorithmes proximaux alternent « respecter les données » et
« appliquer l'a priori », et l'étape d'a priori est un débruitage. Le
Plug-and-Play remplace cette étape par **n'importe quel débruiteur**,
même sans fonctionnelle explicite (extra `pnp`, débruiteurs
d'eoqual-denoising) :

```python
deconvolve(y, psf, "pnp_hqs", sigma=0.01, denoiser="nl_ridge")
deconvolve(y, psf, "pnp_hqs", sigma=0.01, denoiser="drunet")   # extra "deep"
```

| Débruiteur | PSNR |
|---|---|
| `tv_chambolle` | 28.22 dB |
| `nl_ridge` | 28.76 dB |

Sur l'image du benchmark (512×256, quatre flous), le Plug-and-Play
gagne 0.5 à 2.4 dB sur la TV avec NL-Ridge (`pnp_admm`), et encore 1.3 à 2.9 dB avec
DRUNet (BENCHMARK.md, REFERENCE_TECHNIQUE.md §7). Tout
débruiteur d'eoqual-denoising convient (`nl_means`, `bilateral`,
`nl_bayes`...) ; vérifier sa licence (THIRD_PARTY_LICENSES.md).

## 7. Données réelles : bords, saturation, pixels manquants, PSF d'instrument

**Bords.** Une image réelle n'est pas périodique : la FFT voit une
discontinuité entre bords opposés, que la déconvolution transforme en
oscillations.

| Flou de bougé, TV | PSNR |
|---|---|
| `boundary="periodic"` (aucun traitement) | 18.14 dB |
| `boundary="reflect"` (marge miroir continue) | 27.67 dB |
| `boundary="unknown"` (marge inconnue, défaut de la TV) | **29.74 dB** |

Pourquoi `"unknown"` gagne : les pixels proches des bords mélangent, par
le flou, des points de la scène **situés hors du champ**, dont on n'a
aucune mesure. `"reflect"` invente ces points (miroir de l'image) et
oblige la solution à les respecter ; `"unknown"` cherche une scène plus
grande que l'image et n'impose l'accord avec les données que sur l'image
elle-même. La marge est complétée à chaque itération par le flou de
l'estimée courante, si bien qu'elle ne tire jamais la solution ; elle est
jetée à la fin. Schéma et détails : REFERENCE_TECHNIQUE.md §2.1.

**Saturation.** Un pixel saturé ne vaut pas le seuil : il vaut *au moins*
le seuil. Le traiter comme une mesure exacte crée des artefacts autour
des zones claires (nuages, reflets) ; `saturation=` rend l'attache aux
données unilatérale.

```python
deconvolve(y, psf, "tv_primal_dual", sigma=0.01, saturation=0.85)
```

Flou écrêté (7.5 % de pixels saturés) : 27.44 dB → **27.84 dB**.

**Pixels manquants.** Lignes mortes, no-data, NaN : `mask=` (pixels
valides à `True`) exclut ces pixels de l'attache aux données ; la
régularisation les reconstruit.

Une ligne sur 20 mise à 0 : 7.83 dB sans masque → **28.03 dB** avec `mask=`.

**PSF d'instrument.** `system_psf` construit la PSF à partir des
paramètres physiques : diffraction d'une pupille avec obturation
centrale, défocalisation, filé, jitter, intégration sur le pixel.

```python
from eoqual.deconvolution.psf import system_psf
h = system_psf(wavelength=0.55e-6, diameter=0.3, focal_length=12.0, pixel_pitch=11e-6,
               obscuration=0.3, motion_length=1.0)          # échantillonné à Nyquist
```

Avec cette PSF (25 px de support) : observée 27.44 dB, TV **33.24 dB**.
`plot_mtf` compare sa FTM à celle d'une gaussienne : la coupure optique
tombe exactement à Nyquist.

## 8. Régler les paramètres

Les réglages par défaut sont des compromis mesurés sur quatre flous
(BENCHMARK.md). Pour une chaîne donnée (même capteur, même PSF, même
bruit), l'optimisation bayésienne cherche mieux (extra `optim`) :

```python
from eoqual.deconvolution.tuning import optimize_parameters
res = optimize_parameters(y_reglage, psf, "tv_primal_dual", reference=scene_reglage)
x = deconvolve(y, psf, "tv_primal_dual", **res.best_kwargs)
res = optimize_parameters(y, psf, "tv_primal_dual")   # sans référence : indice de netteté SI
```

Régler sur une image, évaluer sur une autre : sinon le gain est
surestimé (sur-apprentissage). Exemple complet : `examples/tuning.py`.

## 9. Pour aller plus loin

Cours et tutoriels publics sur lesquels s'appuie la progression de ce
document (aucun contenu n'en est repris) :

- C. Chaux, TP « Méthodes variationnelles pour l'imagerie », École
  GEOMDATA 2018 —
  [i2m.univ-amu.fr/~caroline.chaux/enseignements.html](https://www.i2m.univ-amu.fr/~caroline.chaux/enseignements.html),
  cours [PbInv_Cours2.pdf](https://geomdata.sciencesconf.org/data/PbInv_Cours2.pdf) :
  inversion, Wiener, Tikhonov, ondelettes, ISTA/FISTA.
- G. Peyré, Numerical Tours, « Image Deconvolution using Variational
  Method » —
  [github.com/gpeyre/numerical-tours](https://github.com/gpeyre/numerical-tours/blob/master/python/inverse_2_deconvolution_variational.ipynb) :
  régularisations de Sobolev et TV.
- H. Talbot, scripts du manuel d'optimisation, déconvolution de Tikhonov —
  [github.com/hugues-talbot/optimization_textbook](https://github.com/hugues-talbot/optimization_textbook).
- M. Ponti, cours de traitement d'images (ICMC-USP), restauration et
  déconvolution —
  [github.com/maponti/imageprocessing_course_icmc](https://github.com/maponti/imageprocessing_course_icmc/blob/master/05b_restoration_deconvolution.ipynb).
- Waller Lab, tutoriel DiffuserCam (ADMM avec bords inconnus) —
  [github.com/Waller-Lab/DiffuserCam-Tutorial](https://github.com/Waller-Lab/DiffuserCam-Tutorial).
- Stanford EE367 « Computational Imaging » (Plug-and-Play ADMM, a priori de
  diffusion) — [stanford.edu/class/ee367](https://stanford.edu/class/ee367/).

Références des méthodes : [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md).
