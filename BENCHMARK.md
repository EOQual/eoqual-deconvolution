# Benchmark — eoqual-deconvolution 0.1.0

Comparaison chiffrée des méthodes, réglages par défaut puis optimisés.
Reproduction : `python examples/benchmark.py [--optimize]`.

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
- Machine : Apple Silicon, CPU (DRUNet non utilisé : débruiteur par
  défaut `nl_ridge`) ; temps moyen par image.

## 2. Réglages par défaut — PSNR (dB) / SSIM

| Méthode | gaussienne σ=2 | Airy (Nyquist) | défocus r=3 | bougé 9 px | Temps (s) |
|---|---|---|---|---|---|
| *(observée)* | 25.38 / 0.819 | 29.94 / 0.897 | 26.32 / 0.844 | 24.53 / 0.805 | — |
| `inverse` | 27.50 / 0.746 | 24.92 / 0.568 | 23.17 / 0.507 | 21.29 / 0.485 | 0.0 |
| `wiener` | 28.56 / 0.826 | 33.22 / 0.881 | 28.82 / 0.768 | 28.16 / 0.861 | 0.0 |
| `wiener_unsupervised` | 27.35 / 0.723 | 33.30 / 0.898 | 28.89 / 0.795 | 27.09 / 0.813 | 0.1 |
| `tikhonov` (GCV) | 28.74 / 0.869 | 33.62 / 0.909 | 29.47 / 0.862 | 26.92 / 0.815 | 0.1 |
| `tikhonov` (discrépance) | 27.74 / 0.903 | 32.61 / 0.945 | 28.63 / 0.917 | 28.09 / 0.874 | 0.1 |
| `richardson_lucy` | 28.51 / 0.898 | 32.43 / 0.847 | 29.68 / 0.868 | 28.29 / 0.787 | 0.1 |
| `landweber` | 28.06 / 0.882 | 33.24 / 0.910 | 28.79 / 0.894 | 27.91 / 0.823 | 0.0 |
| `fista_wavelet` | 29.08 / 0.925 | 34.87 / 0.941 | 30.34 / 0.936 | 29.23 / 0.927 | 1.0 |
| `tv_primal_dual` | 29.92 / 0.925 | 34.94 / 0.946 | 31.38 / 0.935 | 30.12 / 0.940 | 2.0 |
| `pnp_hqs` (`nl_ridge`) | 30.05 / 0.951 | 36.45 / 0.981 | 32.71 / 0.958 | 29.74 / 0.938 | 16.4 |
| `pnp_admm` (`nl_ridge`) | **30.31** / 0.948 | **36.97** / 0.979 | **33.46** / 0.957 | **32.63** / 0.956 | 25.4 |
| `blind_l0` (PSF inconnue) | 28.21 / 0.906 | 34.09 / 0.955 | 29.99 / 0.929 | 28.52 / 0.933 | 4.3 |

## 3. Réglages optimisés — PSNR (dB) sur l'image d'évaluation

Optimisation bayésienne du PSNR sur l'image de réglage (15 évaluations,
10 pour `pnp_hqs`), puis paramètres appliqués à l'image d'évaluation.
Entre parenthèses : écart aux réglages par défaut du §2.

| Méthode | gaussienne σ=2 | Airy (Nyquist) | défocus r=3 | bougé 9 px |
|---|---|---|---|---|
| `inverse` | 28.17 (+0.67) | 32.52 (+7.60) | 28.70 (+5.53) | 26.87 (+5.58) |
| `wiener` | 28.66 (+0.10) | 33.37 (+0.15) | 29.29 (+0.47) | 28.29 (+0.13) |
| `tikhonov` | 28.72 (−0.02) | 33.48 (−0.14) | 29.42 (−0.05) | 28.28 (+1.36) |
| `richardson_lucy` | 29.54 (+1.03) | 34.48 (+2.05) | 30.42 (+0.74) | 29.61 (+1.32) |
| `landweber` | 28.89 (+0.83) | 33.46 (+0.22) | 29.37 (+0.58) | 28.08 (+0.17) |
| `fista_wavelet` | 29.57 (+0.49) | 35.41 (+0.54) | 31.21 (+0.87) | 29.24 (+0.01) |
| `tv_primal_dual` | 29.93 (+0.01) | 35.18 (+0.24) | 31.36 (−0.02) | 30.37 (+0.25) |
| `pnp_hqs` | 30.33 (+0.28) | 36.60 (+0.15) | 33.20 (+0.49) | 32.33 (+2.59) |

Paramètres trouvés (extraits) : `tv_primal_dual` λ = 0.04 à 0.17 (× σ) ;
`fista_wavelet` λ = 0.05 à 0.27, 4 ou 5 niveaux ; `richardson_lucy` 10 à
150 itérations avec une petite TV (5e-4 à 7e-3) ; `tikhonov` régularisation
`gradient` sur trois flous sur quatre ; `pnp_hqs` λ = 0.08 à 0.41. Liste
complète : sortie de `examples/benchmark.py --optimize`.

Les réglages de `fista_wavelet` et `tv_primal_dual` de la 0.1.0 (λ = 0.2
et 0.1) ont été fixés d'après ces optimums ; ceux de la première version
(0.6 et 0.4) étaient trop forts de 0.5 à 1 dB.

## 4. Lecture

- **Meilleur compromis du socle : `tv_primal_dual`** — +3 à +6 dB sur
  l'observée, à 0.25 dB au plus de son réglage optimisé, 2 s.
- **Plug-and-Play** : +0.4 à +2.5 dB au-delà de la TV, au prix d'un temps
  10 fois plus long. `pnp_admm` est le plus régulier ; `pnp_hqs` est en
  retrait sur le bougé avec ses réglages par défaut (rattrapé par
  l'optimisation). Avec DRUNet (extra `deep`) au lieu de NL-Ridge, `pnp_hqs`
  gagne +1.3 à +2.9 dB sur une partie de la même image (REFERENCE_TECHNIQUE.md §7).
- **Méthodes linéaires** : instantanées, +2 à +4 dB ; le bon point de
  départ. `wiener` (spectre ajusté) est le plus régulier ; `tikhonov` par
  GCV est le meilleur sur trois flous mais sous-régularise le bougé (OTF à
  zéros) — utiliser alors `rule="discrepancy"`, qui a aussi le meilleur
  SSIM des méthodes linéaires.
- **Itératives** : le nombre d'itérations par défaut est un compromis ; à
  régler dès que le flou est connu (Richardson-Lucy : +0.7 à +2 dB).
  `inverse` n'est qu'une référence.
- **`blind_l0`** : sans connaître la PSF, résultat proche des méthodes
  linéaires à PSF connue (+2.8 à +4.2 dB sur l'observée), à condition de
  donner une taille de support proche de l'étendue du flou (un support de
  21 px au lieu de 9 à 17 coûte jusqu'à 5 dB, REFERENCE_TECHNIQUE.md §8).
- Le SSIM classe parfois autrement que le PSNR : les méthodes linéaires
  laissent un bruit fin (PSNR correct, SSIM bas), la TV et le PnP le
  suppriment.

Ces chiffres valent pour une scène, un niveau de bruit et des flous
synthétiques : ils classent les méthodes, ils ne prédisent pas le gain
sur une image réelle.
