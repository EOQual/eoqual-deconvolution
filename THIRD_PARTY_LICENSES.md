# Licences tierces — inventaire et statut

> **Avertissement.** Cet inventaire est tenu au meilleur effort, à partir
> des fichiers LICENSE et des métadonnées PyPI des paquets, vérifiés le
> 2026-10-29. Si une erreur y est identifiée, elle sera corrigée dès que
> signalée — et si nécessaire, l'intégration concernée sera retirée
> plutôt que sa licence « arrangée » a posteriori. Pour signaler un
> problème, voir `CONTRIBUTING.md`.

Ce document recense, pour chaque méthode de
`src/eoqual/deconvolution/`, le code qui fait réellement le calcul et sa
licence, au regard de la licence MIT d'`eoqual-deconvolution`
(`license.txt`).

**Principe** : **aucun code tiers n'est copié dans ce dépôt.** Toutes
les méthodes sauf `wiener_unsupervised` sont des implémentations
originales, écrites à partir des publications de
[BIBLIOGRAPHY.md](BIBLIOGRAPHY.md) ; `wiener_unsupervised` appelle
scikit-image.

## 1. Légende

- 🟢 permissif (MIT, BSD, Apache-2.0) : aucune restriction au-delà de la
  conservation des notices ;
- 🟡 restriction d'usage à lire avant utilisation.

## 2. Inventaire

### 2.1 Socle (🟢)

| Méthode / brique | Code qui calcule | Licence |
|---|---|---|
| `inverse`, `wiener`, `tikhonov`, `richardson_lucy`, `landweber`, `tv_primal_dual`, `blind_l0` | ce paquet (numpy, scipy) | MIT |
| `fista_wavelet` | ce paquet + PyWavelets (transformée) | MIT |
| `wiener_unsupervised` | scikit-image (`restoration.unsupervised_wiener`) | BSD-3-Clause |
| bords, PSF, bruit, `psf2otf` | ce paquet + eoqual-core | MIT |
| métriques (PSNR, SSIM, SI) | eoqual-metrics (SSIM : scikit-image) | MIT / BSD-3-Clause |
| figures | eoqual-viz + Plotly | MIT |

### 2.2 Extras pip optionnels

| Extra | Paquets | Licence | Remarque |
|---|---|---|---|
| `pnp` | eoqual-denoising | MIT | 🟢 avec les débruiteurs du socle d'eoqual-denoising |
| `deep` | eoqual-denoising, torch, deepinv, einops | BSD-3-Clause / MIT | 🟢 code et poids DRUNet (BSD-3-Clause) |
| `optim` | scikit-optimize | BSD-3-Clause | 🟢 |

🟡 **Le Plug-and-Play hérite de la licence du débruiteur choisi.** Tous
les débruiteurs d'eoqual-denoising sont acceptés par `denoiser=` ;
certains ont une restriction (par exemple `bm3d` : licence **non
commerciale** ; `gsdrunet`, `epll` : poids sans licence déclarée ;
`nl_ridge_inria`, `nl_bayes_ipol` : **AGPL-3.0** et **GPL-3.0+**).
Vérifier avec `eoqual.denoising.runner.list_methods()` et le
`THIRD_PARTY_LICENSES.md` d'eoqual-denoising avant de les utiliser dans
un logiciel distribué ou un service. Le débruiteur par défaut,
`nl_ridge`, est une réimplémentation MIT.

### 2.3 Dépendances pip — licences déclarées

| Paquet | Licence |
|---|---|
| numpy, scipy, scikit-image | BSD-3-Clause |
| PyWavelets | MIT |
| plotly | MIT |
| rich | MIT |
| eoqual-core, eoqual-metrics, eoqual-viz, eoqual-denoising | MIT |
| scikit-optimize | BSD-3-Clause |
| torch, deepinv | BSD-3-Clause |
| einops | MIT |

eoqual-metrics installe lui-même d'autres dépendances (OpenCV, sewar,
cpbd, pyrtools, pyfftw...) : voir son propre `THIRD_PARTY_LICENSES.md`.

## 3. Principe appliqué : publication vs code

Une méthode publiée peut être réimplémentée librement à partir de
l'article ; un code publié ne peut être repris que selon sa licence.
Les méthodes de ce paquet ont été écrites à partir des seules
publications citées dans l'en-tête de chaque module et dans
[BIBLIOGRAPHY.md](BIBLIOGRAPHY.md), sans consulter ni reprendre de code
tiers. Le filtre de Wiener-Tikhonov et ses critères GCV et L-curve
reprennent la formulation d'un prototype personnel de l'auteur, publié
ici sous MIT.

## 4. Images

Les tests, exemples et benchmarks utilisent l'image « astronaut »
fournie par scikit-image (NASA, domaine public).
