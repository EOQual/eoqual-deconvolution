# Benchmark — eoqual-deconvolution 0.3.0

Comparaison chiffrée des méthodes : réglages par défaut, données
incomplètes, réglages optimisés. Reproduction :
`python examples/benchmark.py [--optimize]`. Le choix des réglages par
défaut eux-mêmes est décrit dans REFERENCE_TECHNIQUE.md §9.1.

## 1. Protocole

- **Image** : « astronaut » 512×512 en niveaux de gris, dynamique [0, 1]
  (NASA, domaine public, fournie par scikit-image). **Moitié gauche**
  (512×256) : image de **réglage** (optimisation seulement) ; **moitié
  droite** (512×256) : image d'**évaluation** (tous les scores).
- **Flous** : gaussien σ = 2 px ; Airy échantillonné à Nyquist (premier
  anneau noir à 2.44 px) ; défocalisation (disque de rayon 3 px) ; bougé
  linéaire de 9 px à 30°. Convolution avec **prolongement miroir** : les
  bords sont ceux d'une vraie scène, pas d'une simulation périodique.
- **Bruit** gaussien σ = 0.01, transmis aux méthodes (`sigma=0.01`).
- **Scores** : PSNR (dB) et SSIM d'eoqual-metrics.
- `blind_l0` ne reçoit pas la PSF, seulement la taille de son support.
- Machine : Apple Silicon, CPU (débruiteur par défaut `nl_ridge`, pas de
  PyTorch) ; temps moyen par image.
- **Bords** : chaque méthode avec son défaut — `boundary="unknown"` pour
  `wiener`, `tikhonov`, `fista_wavelet`, `tv_primal_dual` ; `"reflect"`
  pour les autres (REFERENCE_TECHNIQUE.md §2.1).

## 2. Réglages par défaut — PSNR (dB) / SSIM

| Méthode | gaussienne σ=2 | Airy (Nyquist) | défocus r=3 | bougé 9 px | Temps (s) |
|---|---|---|---|---|---|
| *(observée)* | 25.38 / 0.819 | 29.94 / 0.897 | 26.32 / 0.844 | 24.53 / 0.805 | — |
| `inverse` | 27.85 / 0.879 | 31.60 / 0.826 | 28.65 / 0.890 | 26.86 / 0.761 | 0.0 |
| `wiener` | 28.55 / 0.829 | 33.22 / 0.881 | 28.81 / 0.770 | 28.94 / 0.848 | 0.1 |
| `wiener_unsupervised` | 27.35 / 0.723 | 33.30 / 0.898 | 28.89 / 0.795 | 27.09 / 0.813 | 0.1 |
| `tikhonov` (GCV) | 28.73 / 0.871 | 33.62 / 0.909 | 29.44 / 0.863 | 29.09 / 0.804 | 0.2 |
| `richardson_lucy` | 29.17 / 0.903 | 34.41 / 0.905 | 30.52 / 0.897 | 29.63 / 0.860 | 0.3 |
| `landweber` | 28.06 / 0.882 | 33.24 / 0.910 | 28.79 / 0.894 | 27.91 / 0.823 | 0.0 |
| `fista_wavelet` | 29.24 / 0.927 | 34.78 / 0.934 | 30.71 / 0.938 | 30.57 / 0.909 | 1.0 |
| `tv_primal_dual` | 29.90 / 0.925 | 34.93 / 0.946 | 31.32 / 0.935 | 31.41 / 0.926 | 2.0 |
| `pnp_hqs` (`nl_ridge`) | 30.25 / 0.940 | 37.30 / 0.975 | 32.97 / 0.955 | 32.21 / 0.953 | 12.4 |
| `pnp_admm` (`nl_ridge`) | 30.42 / 0.942 | 37.33 / 0.975 | 33.40 / 0.957 | 33.06 / 0.958 | 21.6 |
| `blind_l0` (PSF inconnue) | 28.21 / 0.906 | 34.09 / 0.955 | 29.99 / 0.929 | 28.52 / 0.933 | 3.9 |
| `tikhonov` (discrépance) | 27.73 / 0.902 | 32.60 / 0.946 | 28.61 / 0.916 | 28.65 / 0.862 | 0.2 |

## 3. Réglages optimisés — PSNR (dB) sur l'image d'évaluation

Optimisation bayésienne du PSNR sur l'image de réglage (15 évaluations,
10 pour `pnp_hqs`), puis paramètres appliqués à l'image d'évaluation.
Entre parenthèses : écart aux réglages par défaut du §2.

| Méthode | gaussienne σ=2 | Airy (Nyquist) | défocus r=3 | bougé 9 px |
|---|---|---|---|---|
| `inverse` | 28.24 (+0.39) | 32.52 (+0.92) | 28.71 (+0.06) | 26.86 (+0.00) |
| `wiener` | 28.65 (+0.10) | 33.38 (+0.16) | 29.26 (+0.45) | 29.08 (+0.14) |
| `tikhonov` | 28.70 (-0.02) | 33.45 (-0.17) | 29.40 (-0.04) | 29.11 (+0.01) |
| `richardson_lucy` | 29.32 (+0.14) | 34.61 (+0.19) | 30.58 (+0.07) | 29.48 (-0.15) |
| `landweber` | 28.89 (+0.84) | 33.46 (+0.22) | 29.37 (+0.58) | 28.08 (+0.16) |
| `fista_wavelet` | 29.54 (+0.30) | 35.36 (+0.58) | 31.01 (+0.30) | 30.64 (+0.07) |
| `tv_primal_dual` | 29.91 (+0.01) | 35.13 (+0.20) | 31.29 (-0.02) | 31.33 (-0.08) |
| `pnp_hqs` | 30.59 (+0.34) | 37.26 (-0.05) | 32.97 (+0.00) | 31.63 (-0.59) |

Les réglages par défaut de la 0.3.0 sont proches de l'optimum de chaque
cas : l'optimisation n'apporte plus que 0 à +0.9 dB (`landweber`,
régularisé par le seul nombre d'itérations, et `inverse` restent les plus
sensibles). Les écarts négatifs (jusqu'à −0.6 dB) sont du
sur-apprentissage : le réglage optimal de la moitié gauche est moins bon
que le défaut sur la moitié droite. Optimiser reste utile pour une chaîne
d'acquisition précise (capteur, PSF et bruit fixés), sur une image de
réglage distincte.

## 4. Données incomplètes — PSNR (dB) sur l'image d'évaluation

Flou gaussien σ = 2 px pour les deux premières colonnes, bougé pour la
troisième.

- **Saturation** : flou écrêté à 85 % de son maximum (6.3 % de pixels
  saturés), traité comme une mesure exacte, puis avec `saturation=` ;
- **lignes mortes** : une ligne sur 25 mise à 0, ignorée, puis signalée
  par `mask=` ;
- **bords** : `boundary="reflect"` puis `"unknown"` (marge inconnue).

| Méthode | saturation : ignorée → `saturation=` | lignes mortes : ignorées → `mask=` | bougé : `reflect` → `unknown` |
|---|---|---|---|
| `wiener` | 27.28 → 27.62 | 8.86 → 28.51 | 28.16 → 28.94 |
| `tikhonov` | 27.25 → 27.55 | -56.17 → 28.69 | 26.92 → 29.09 |
| `richardson_lucy` | 28.03 → 28.86 | 14.72 → 29.11 | 29.63 → 29.70 |
| `landweber` | 27.16 → 27.63 | 6.21 → 27.96 | 27.91 → 27.42 |
| `fista_wavelet` | 27.77 → 28.77 | 3.35 → 29.19 | 29.24 → 30.57 |
| `tv_primal_dual` | 28.19 → 29.59 | 9.02 → 29.83 | 30.12 → 31.41 |
| `pnp_hqs` | 28.11 → 28.48 | 7.19 → 30.12 | 32.21 → 32.21 |

Avec `mask=`, toutes les méthodes reviennent à 0.15 dB au plus de leur
résultat sans défaut, alors que les lignes mortes ignorées détruisent
l'image. `saturation=` gagne 0.3 à 1.4 dB ; le gain se concentre autour
des zones saturées, le reste de l'image est inchangé. La marge inconnue
gagne 0.8 à 2.2 dB sur le bougé pour les quatre méthodes qui l'ont par
défaut ; elle est neutre pour Richardson-Lucy (+0.07 dB) et le
Plug-and-Play (qui complète déjà la marge), défavorable pour Landweber
(−0.5 dB) : ces trois méthodes gardent `"reflect"`.

## 5. Lecture

- **Meilleur compromis du socle : `tv_primal_dual`** — +4.5 à +6.9 dB sur
  l'observée, 2 s. `fista_wavelet` et `richardson_lucy` (avec sa TV par
  défaut) suivent à 0.2 à 1.8 dB, Richardson-Lucy en 0.3 s.
- **Plug-and-Play** : +0.5 à +2.4 dB au-delà de la TV (`pnp_admm`), au
  prix d'un temps 6 à 10 fois plus long ; `pnp_hqs` est à 0.1 à 0.9 dB
  derrière, 1.7 fois plus rapide. Avec DRUNet (extra `deep`) au lieu de
  NL-Ridge, `pnp_hqs` gagne encore +1.3 à +2.9 dB sur une partie de la
  même image (REFERENCE_TECHNIQUE.md §7).
- **Méthodes linéaires** : instantanées, +2.5 à +4.6 dB (`wiener`,
  `tikhonov`) ; le bon point de départ. `inverse` n'est qu'une référence.
- **`blind_l0`** : sans connaître la PSF, résultat proche des méthodes
  linéaires à PSF connue (+2.8 à +4.2 dB sur l'observée), à condition de
  donner une taille de support proche de l'étendue du flou
  (REFERENCE_TECHNIQUE.md §8).
- **`wiener_hunt`** : au niveau de `tikhonov` (GCV) en moyenne
  (+0.02 dB sur 24 cas), sans réglage, en estimant σ au passage (3 %
  d'erreur médiane) ; bien plus sûr que `wiener_unsupervised` (§6).
- **`blind_optics`** : quand l'instrument est connu, contraindre la PSF
  au modèle optique ramène l'écart à la vraie PSF de 3.1 dB
  (`blind_l0`) à 1.2 dB en moyenne (§7).
- **Données réelles** : signaler les pixels manquants (`mask=`) est
  indispensable ; la saturation et la marge inconnue apportent un gain
  plus modeste mais gratuit.
- Le SSIM classe parfois autrement que le PSNR : les méthodes linéaires
  laissent un bruit fin (PSNR correct, SSIM bas), la TV et le PnP le
  suppriment.

Ces chiffres valent pour une scène, un niveau de bruit et des flous
synthétiques : ils classent les méthodes, ils ne prédisent pas le gain
sur une image réelle.

## 6. Wiener–Hunt (0.4.0) — PSNR (dB), PSF connue

`wiener_hunt` (λ et σ par vraisemblance marginale, REFERENCE_TECHNIQUE.md
§3.5) face aux autres méthodes linéaires, réglages par défaut
(`python examples/wiener_hunt_benchmark.py IMAGE.tif`). Scènes
256 × 256 : « astronaut » et un extrait d'orthophotographie (Toulouse
Métropole, Licence Ouverte 2.0) réduit d'un facteur 2 ; flous gaussien
(σ = 1.6 px), Airy (premier anneau à 2.5 px), bougé (9 px), PSF optique
(Q = 1, obturation 0.3, défocalisation 0.15 onde RMS, jitter 0.3 px) ;
bruit blanc gaussien, graine 0.

| scène | flou | σ | `tikhonov` (GCV) | `wiener` | `wiener_unsupervised` | `wiener_hunt` | `wiener_hunt`, σ connu | `wiener_hunt`, `prior="gradient"` | σ estimé |
|---|---|---|---|---|---|---|---|---|---|
| astronaut | gauss1.6 | 0.003 | 30.43 | 30.30 | 30.04 | 30.43 | 30.42 | 30.06 | 0.0029 |
| astronaut | gauss1.6 | 0.01 | 28.83 | 28.72 | 28.21 | 28.82 | 28.81 | 28.47 | 0.0097 |
| astronaut | gauss1.6 | 0.03 | 27.04 | 26.98 | 25.90 | 27.02 | 27.00 | 26.71 | 0.0292 |
| astronaut | airy | 0.003 | 35.37 | 35.22 | 35.00 | 35.36 | 35.38 | 34.35 | 0.0029 |
| astronaut | airy | 0.01 | 32.39 | 32.21 | 32.09 | 32.40 | 32.40 | 31.41 | 0.0096 |
| astronaut | airy | 0.03 | 29.23 | 29.13 | 28.93 | 29.23 | 29.21 | 28.42 | 0.0290 |
| astronaut | bougé | 0.003 | 30.79 | 31.16 | 22.36 | 31.14 | 29.89 | 30.95 | 0.0045 |
| astronaut | bougé | 0.01 | 28.16 | 28.31 | 25.76 | 28.38 | 28.36 | 27.78 | 0.0107 |
| astronaut | bougé | 0.03 | 26.12 | 26.16 | 25.19 | 26.17 | 26.17 | 25.73 | 0.0299 |
| astronaut | optique | 0.003 | 36.06 | 35.70 | 35.80 | 35.95 | 36.14 | 31.82 | 0.0026 |
| astronaut | optique | 0.01 | 31.30 | 30.99 | 31.04 | 31.34 | 31.34 | 28.25 | 0.0097 |
| astronaut | optique | 0.03 | 28.29 | 28.18 | 27.84 | 28.28 | 28.27 | 26.76 | 0.0293 |
| toulouse | gauss1.6 | 0.003 | 30.75 | 30.72 | 30.47 | 30.70 | 30.69 | 30.64 | 0.0029 |
| toulouse | gauss1.6 | 0.01 | 29.48 | 29.47 | 28.86 | 29.47 | 29.47 | 29.33 | 0.0097 |
| toulouse | gauss1.6 | 0.03 | 28.18 | 28.18 | 26.64 | 28.19 | 28.18 | 27.98 | 0.0292 |
| toulouse | airy | 0.003 | 35.59 | 34.89 | 34.75 | 34.91 | 35.00 | 35.43 | 0.0031 |
| toulouse | airy | 0.01 | 31.97 | 31.91 | 31.81 | 31.93 | 31.92 | 31.79 | 0.0098 |
| toulouse | airy | 0.03 | 29.58 | 29.56 | 29.32 | 29.54 | 29.52 | 29.27 | 0.0291 |
| toulouse | bougé | 0.003 | 32.04 | 32.22 | 27.20 | 32.26 | 31.97 | 32.07 | 0.0038 |
| toulouse | bougé | 0.01 | 29.67 | 30.09 | 28.79 | 29.92 | 29.89 | 29.48 | 0.0103 |
| toulouse | bougé | 0.03 | 28.17 | 28.56 | 26.96 | 28.36 | 28.39 | 27.98 | 0.0295 |
| toulouse | optique | 0.003 | 36.22 | 36.32 | 36.35 | 36.60 | 36.64 | 33.28 | 0.0028 |
| toulouse | optique | 0.01 | 31.55 | 31.43 | 31.33 | 31.36 | 31.38 | 30.38 | 0.0101 |
| toulouse | optique | 0.03 | 28.97 | 28.96 | 28.50 | 28.94 | 28.93 | 28.49 | 0.0294 |

Moyennes : `wiener_hunt` 30.70, `tikhonov` 30.67, `wiener` 30.64,
`wiener_hunt` σ connu 30.64, `prior="gradient"` 29.87,
`wiener_unsupervised` 29.55.

## 7. Déconvolution aveugle contrainte par l'optique (0.4.0) — PSNR (dB)

`python examples/blind_optics_benchmark.py IMAGE.tif` (REFERENCE_TECHNIQUE.md
§8.2.7) : Q = 1, obturation 0.3, jitter vrai 0.35 px (fixé sauf dans le
cas « jitter libre »), restauration TV (`tv_primal_dual`) avec la PSF
indiquée ; « urbain » et « parc » : orthophotographie réduite d'un
facteur 4 (moyenne 2 × 2 puis un pixel sur deux). Colonnes `evidence` et
`l0` : `blind_optics` avec chacun des deux critères ; θ et temps :
critère `evidence`.

| cas | scène | σ | vraie PSF | nominale | blind_l0 | evidence | l0 | θ estimé (evidence) | temps |
|---|---|---|---|---|---|---|---|---|---|
| défoc. 0.12 | astronaut | 0.003 | 39.29 | 33.65 | 34.60 | 36.49 | 38.89 | defocus 0.14 | 3 s |
| défoc. 0.12 | astronaut | 0.01 | 33.38 | 31.39 | 32.02 | 33.24 | 33.39 | defocus 0.13 | 3 s |
| défoc. 0.12 | urbain | 0.003 | 36.58 | 30.83 | 30.26 | 33.41 | 33.91 | defocus 0.08 | 3 s |
| défoc. 0.12 | urbain | 0.01 | 31.04 | 28.99 | 27.92 | 29.42 | 30.08 | defocus 0.05 | 2 s |
| défoc. 0.12 | parc | 0.003 | 36.21 | 31.74 | 30.62 | 34.15 | 34.48 | defocus 0.09 | 3 s |
| défoc. 0.12 | parc | 0.01 | 31.07 | 29.63 | 28.41 | 30.07 | 30.47 | defocus 0.06 | 3 s |
| défoc. 0.25 | astronaut | 0.003 | 28.76 | 25.25 | 27.85 | 28.70 | 28.15 | defocus 0.25 | 4 s |
| défoc. 0.25 | astronaut | 0.01 | 27.99 | 24.94 | 25.15 | 27.88 | 27.50 | defocus 0.24 | 3 s |
| défoc. 0.25 | urbain | 0.003 | 25.11 | 23.21 | 24.57 | 25.11 | 24.55 | defocus 0.25 | 5 s |
| défoc. 0.25 | urbain | 0.01 | 24.71 | 23.05 | 23.81 | 24.62 | 24.35 | defocus 0.24 | 5 s |
| défoc. 0.25 | parc | 0.003 | 26.27 | 24.47 | 25.71 | 26.25 | 25.36 | defocus 0.25 | 4 s |
| défoc. 0.25 | parc | 0.01 | 25.81 | 24.25 | 23.91 | 25.62 | 25.13 | defocus 0.23 | 3 s |
| défoc. + astig. | astronaut | 0.003 | 38.04 | 32.55 | 33.51 | 36.67 | 35.60 | defocus 0.09, astig_0 0.09, astig_45 0.05 | 4 s |
| défoc. + astig. | astronaut | 0.01 | 32.93 | 30.75 | 31.68 | 32.69 | 32.47 | defocus 0.13, astig_0 0.06, astig_45 0.03 | 7 s |
| défoc. + astig. | urbain | 0.003 | 35.23 | 30.46 | 30.12 | 34.37 | 33.48 | defocus 0.10, astig_0 0.10, astig_45 0.01 | 6 s |
| défoc. + astig. | urbain | 0.01 | 30.70 | 28.82 | 28.04 | 30.49 | 30.03 | defocus 0.10, astig_0 0.10, astig_45 0.01 | 6 s |
| défoc. + astig. | parc | 0.003 | 34.84 | 30.69 | 30.10 | 34.01 | 32.82 | defocus 0.09, astig_0 0.08, astig_45 0.02 | 3 s |
| défoc. + astig. | parc | 0.01 | 30.60 | 29.02 | 28.24 | 30.30 | 29.78 | defocus 0.08, astig_0 0.09, astig_45 0.04 | 5 s |
| défoc. + jitter libre | astronaut | 0.003 | 39.29 | 33.65 | 34.60 | 38.03 | 37.52 | defocus 0.11, jitter 0.51 | 2 s |
| défoc. + jitter libre | astronaut | 0.01 | 33.38 | 31.39 | 32.02 | 33.47 | 33.42 | defocus 0.11, jitter 0.50 | 9 s |
| défoc. + jitter libre | urbain | 0.003 | 36.58 | 30.83 | 30.26 | 32.33 | 33.35 | defocus 0.03, jitter 0.43 | 4 s |
| défoc. + jitter libre | urbain | 0.01 | 31.04 | 28.99 | 27.92 | 29.32 | 29.92 | defocus 0.00, jitter 0.41 | 5 s |
| défoc. + jitter libre | parc | 0.003 | 36.21 | 31.74 | 30.62 | 33.25 | 33.77 | defocus 0.04, jitter 0.43 | 3 s |
| défoc. + jitter libre | parc | 0.01 | 31.07 | 29.63 | 28.41 | 30.08 | 30.30 | defocus 0.00, jitter 0.45 | 6 s |
| défoc. + coma | astronaut | 0.003 | 34.65 | 27.30 | 30.51 | 33.76 | 32.50 | defocus 0.20, coma_x 0.04, coma_y -0.02 | 7 s |
| défoc. + coma | astronaut | 0.01 | 31.06 | 26.74 | 29.07 | 30.73 | 30.07 | defocus 0.20, coma_x 0.05, coma_y -0.02 | 7 s |
| défoc. + coma | urbain | 0.003 | 30.17 | 25.08 | 26.28 | 25.90 | 28.22 | defocus 0.00, coma_x 0.10, coma_y 0.08 | 4 s |
| défoc. + coma | urbain | 0.01 | 27.11 | 24.67 | 25.56 | 25.29 | 26.37 | defocus 0.00, coma_x 0.10, coma_y 0.10 | 4 s |
| défoc. + coma | parc | 0.003 | 30.61 | 26.23 | 27.28 | 29.91 | 28.38 | defocus 0.19, coma_x 0.08, coma_y -0.04 | 6 s |
| défoc. + coma | parc | 0.01 | 27.86 | 25.74 | 26.44 | 26.24 | 26.84 | defocus 0.00, coma_x 0.09, coma_y 0.10 | 5 s |
