# Optimisation des paramètres

## 9. Optimisation des paramètres

``tuning/optimizer.py`` (extra ``optim``) :
``optimize_parameters(blurred, psf, method, reference=None, metric=None,
sigma=None, n_calls=25)``. Processus gaussien (``skopt.gp_minimize``),
point de départ = réglages par défaut (l'optimisation ne fait donc
jamais pire sur l'image de réglage, vérifié par test). Résultat :
``best_params``, ``best_kwargs`` (à repasser à ``deconvolve``),
``best_score``, ``gain``, ``history``.

- ``metric="psnr"`` ou ``"ssim"`` (eoqual-metrics) avec ``reference`` ;
- ``metric="sharpness_index"`` sans référence : indice SI de Blanchet &
  Moisan, qui croît avec la netteté et décroît avec le bruit et les
  artefacts (Leclaire & Moisan 2015). Indicatif : vérifier visuellement.

Espaces de recherche : ``tuning/spaces/<méthode>.jsonc``, commentés,
``prior: "log-uniform"`` pour les paramètres qui varient sur plusieurs
décades ; ``sigma_scale`` multiplie σ (σ mal connu). Compter 10
évaluations par dimension. Le motif (registre, JSONC, résultat) est le
même que dans eoqual-denoising.

**Sur-apprentissage** : régler sur une image, évaluer sur une autre
(``examples/benchmark.py --optimize`` : moitiés gauche et droite).

### 9.1 Choix des réglages par défaut

Un défaut doit valoir pour tous les cas, pas pour un seul : il est choisi
par **PSNR moyen sur 12 cas** — quatre flous (gaussien σ = 2 px, Airy à
Nyquist, défocalisation r = 3 px, bougé 9 px) × trois bruits
(σ = 0.005, 0.01, 0.02) — sur l'image de **réglage** (moitié gauche
d'« astronaut » ; 256×256 pour le Plug-and-Play et l'aveugle), par une
grille sur les paramètres principaux. Le choix est ensuite **vérifié
sur l'image d'évaluation** (moitié droite, autres tirages de bruit) ;
un réglage qui ne s'y confirme pas est rejeté.

| Méthode | Réglage de départ | Défaut retenu | Gain moyen (évaluation) | Pire cas |
|---|---|---|---|---|
| ``inverse`` | ``threshold = 0.05`` | ``0.15`` | +4.55 dB | −1.14 dB |
| ``richardson_lucy`` | 12 it., ``tv_weight = 0`` | 50 it., ``tv_weight = 0.005`` | +1.38 dB | +0.48 dB |
| ``fista_wavelet`` | ``lam = 0.2``, ``levels = 3`` | ``0.15``, ``5`` | +0.24 dB | −0.40 dB |
| ``pnp_hqs`` | ``sigma_start = 0.2`` | ``0.05`` | +1.18 dB | −0.73 dB |
| ``pnp_admm`` | ``sigma_start = 0.2`` | ``0.1`` | +0.37 dB | −0.30 dB |

Inchangés, déjà à l'optimum de la grille : ``tikhonov`` (laplacien,
GCV), ``landweber`` (discrépance, sans Barzilai-Borwein),
``tv_primal_dual`` (``lam = 0.1``), ``blind_l0`` (``lam_l0 = 4e-3``,
``lam = 0.4``). **Rejeté** : ``wiener`` avec β = 2.5 fixe (+0.32 dB sur
l'image de réglage, mais −0.16 dB sur l'image d'évaluation) — β reste
ajusté sur chaque image.
