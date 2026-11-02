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
- **Données réelles** : signaler les pixels manquants (`mask=`) est
  indispensable ; la saturation et la marge inconnue apportent un gain
  plus modeste mais gratuit.
- Le SSIM classe parfois autrement que le PSNR : les méthodes linéaires
  laissent un bruit fin (PSNR correct, SSIM bas), la TV et le PnP le
  suppriment.

Ces chiffres valent pour une scène, un niveau de bruit et des flous
synthétiques : ils classent les méthodes, ils ne prédisent pas le gain
sur une image réelle.
