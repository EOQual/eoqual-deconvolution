# Plug-and-Play et réseaux dépliés

## 7. Plug-and-Play et réseaux dépliés

### 7.1 Plug-and-Play (``pnp_hqs``, ``pnp_admm``)

``backends/pnp.py``. Problème ``½‖Hx − y‖²/σ² + λΦ(x)``, l'a priori
``Φ`` n'étant connu qu'à travers un débruiteur ``D(·, σ_k)``.

- **HQS** : ``μ_k = λ/σ_k²`` ; pas en ``x`` exact en Fourier,
  ``X = (H*Y + α_k Z)/(|H|² + α_k)`` avec ``α_k = λσ²/σ_k²`` ; puis
  ``z = D(x, σ_k)``.
- **ADMM** : même pas appliqué à ``z − u``, ``z = D(x + u, σ_k)``,
  ``u ← u + x − z``. La variable duale réduite ``u`` est remise à
  l'échelle (``u ← u·α_{k−1}/α_k``) quand le pas change.
- **Calendrier** : ``σ_k`` décroît géométriquement de ``sigma_start`` à
  ``max(σ, sigma_min)``, en ``n_iter`` pas (8 pour HQS, 12 pour ADMM).
  ``sigma_start`` = 0.05 (HQS) et 0.1 (ADMM) de la dynamique : un départ
  plus doux que le 0.2 de Zhang et al. (2021), conçu pour des niveaux de
  bruit plus forts, gagne +1.2 dB (HQS) et +0.4 dB (ADMM) en moyenne, et
  jusqu'à +6 dB sur flou de bougé ([§9.1](optimisation.md#91-choix-des-réglages-par-défaut)).
- **λ** : 0.23 pour HQS (Zhang et al. 2021) ; 0.5 pour ADMM, meilleur
  compromis mesuré sur les quatre flous du benchmark (0.23 y donnait
  26-27 dB contre 35 dB sur bougé et défocus).
- **Débruiteur** : nom d'une méthode d'eoqual-denoising (``nl_ridge``
  par défaut, réimplémentation MIT, sans PyTorch ; ``drunet`` avec l'extra
  ``deep``, +1.3 à +2.9 dB sur NL-Ridge en HQS, deux fois plus rapide sur
  Apple Silicon) ou fonction ``f(image, sigma)``. Il reçoit l'image ramenée à
  [0, 1] et σ dans ces unités ; ``data_range=(0, 1)`` lui est imposé s'il
  l'accepte (sinon il le déduirait du min/max de l'itéré, qui varie).
- **Tout débruiteur convient** : non local (``nl_means``, ``nl_bayes``),
  ``bilateral``, réseaux (``drunet``, ``swinir``...). Comparer plusieurs a
  priori dans le même algorithme se fait donc en changeant un argument.
- **Données incomplètes et marge** : à partir de la 2ᵉ itération, les
  données sont complétées par le flou de l'estimée courante ``z`` ([§2.2](modele-bords.md#22-données-incomplètes--pixels-masqués-saturés-marge-inconnue)) :
  pixels masqués, saturés et, avec ``boundary="reflect"``, la marge ; les
  vraies données ne sont imposées que sur les pixels connus de l'image
  d'origine. Sans ce remplacement, le pas en ``x`` presque inverse des
  premières itérations amplifie la marge, qui n'est pas une image floue :
  sur flou de bougé, HQS-DRUNet passait de 36.0 dB (simulation
  périodique) à 26.9 dB (bords réels) ; avec le remplacement, 32.3 dB.
  Contrepartie : −0.5 à −1 dB sur flou gaussien et défocalisation avec
  NL-Ridge (OTF sans zéro, marge déjà peu gênante). Un remplacement à
  partir de la moitié des itérations ne sauve pas le bougé (25 dB) : il
  est donc fait dès la 2ᵉ itération ; ``fill_boundary=False`` le
  désactive.

### 7.2 `usrnet` — réseau déplié (extra ``deep``)

Réseau dans eoqual-neural (``eoqual.neural.unrolled.usrnet``) depuis la
version 0.6.0 ; ce paquet garde la méthode ``"usrnet"`` et ce qui relève
de la déconvolution (bords, pixels masqués ou saturés, échelle) dans
``backends/usrnet.py``. USRNet (Zhang, Van Gool & Timofte, CVPR 2020) **déplie**
l'algorithme HQS du §7.1 sur un nombre fixe d'itérations (8, ou 6 pour
le modèle réduit) et en **apprend** les ingrédients de bout en bout, au
lieu de les régler :

- **module de données**, sans paramètre appris : la solution exacte du
  pas en ``x`` de HQS, ``Z = (H* Y + α_k X)/(|H|² + α_k)`` pour un facteur
  d'échelle 1 (Zhao et al. 2016 pour le cas général de super-résolution)
  — le même que le pas en ``x`` de ``pnp_hqs`` ;
- **module d'a priori** : un débruiteur ResUNet conditionné par β_k (carte
  constante concaténée à l'image), qui joue le rôle du débruiteur du
  Plug-and-Play ;
- **module d'hyperparamètres** : un petit réseau qui donne ``(α_k, β_k)``
  pour chaque itération à partir de ``(σ, s)`` — l'équivalent appris du
  calendrier et du λ de ``pnp_hqs``.

La **PSF et σ sont des entrées** : le réseau n'apprend pas un flou, il
apprend à inverser n'importe quel flou fourni. C'est ce qui le rend
adaptable à un instrument (PSF mesurée ou estimée par ``blind_optics``)
et réentraînable (§7.3).

**Implémentation.** Structure et noms de paramètres de KAIR (MIT, © 2019
Kai Zhang) pour charger les poids publiés, réécrits pour l'API FFT
complexe actuelle de PyTorch (``torch.rfft`` n'existe plus) ; exception
documentée dans THIRD_PARTY_LICENSES.md [§3.2](methodes-classiques.md#32-wiener--wiener-spectre-du-signal-ajusté). Poids ``"usrnet"`` (17.0 M
de paramètres) et ``"usrnet_tiny"`` (0.59 M), téléchargés à la demande
depuis la publication v1.0 de KAIR et vérifiés par SHA-256
(``eoqual.neural.weights`` ; cache ``$EOQUAL_WEIGHTS_DIR`` ou
``~/.cache/eoqual/neural`` ; un cache ``~/.cache/eoqual/deconvolution``
existant est réutilisé). Vérifications : chargement strict des
poids, module de données égal à la formule fermée à 2·10⁻¹⁵ près, OTF
égale à ``eoqual.core.kernel.psf2otf``. Le réseau, en couleur, reçoit
l'image répliquée sur trois canaux ; la sortie est leur moyenne. L'image
est ramenée à [0, 1] ; σ est exprimé dans ces unités (poids publiés : σ
de 0 à 25/255). Calcul sur CUDA, GPU Apple (MPS) ou CPU : ~0.4 s pour
256 × 256 avec le modèle complet.

**Bords : la leçon des réseaux dépliés.** Le réseau a été entraîné en flou
**circulaire** : son module de données suppose que l'image de travail
entière est une image floue par la PSF. Trois traitements ont été essayés
(scène « astronaut », bougé de 9 px, σ = 0.003) :

| Traitement de la marge | PSNR | Remarque |
|---|---|---|
| prolongement miroir (``boundary="reflect"``) | 16.8 dB | le miroir d'une image floue n'est pas l'image floue du miroir pour une PSF asymétrique : défaut amplifié aux zéros de la FTM |
| marge complétée **à chaque itération** par le flou de l'estimée (comme ``pnp_hqs``, §7.1) | diverge | les estimées intermédiaires d'un réseau déplié ne sont pas des images (12 à 18 dB, valeurs jusqu'à 119) : la marge se nourrit d'elles et explose |
| marge complétée **une fois, avant le réseau**, par le flou d'une restauration de Wiener–Hunt à marge inconnue (``boundary="unknown"``, défaut) | **36.7 dB** | domaine entier cohérent avec la PSF ; inchangé (±0.1 dB) sur les flous symétriques |
| (référence : flou simulé périodique) | 37.4 dB | |

La même complétion préalable traite les **pixels masqués et saturés**
([§2.2](modele-bords.md#22-données-incomplètes--pixels-masqués-saturés-marge-inconnue)) : une ligne sur 16 masquée est reconstruite (test). Le Plug-and-Play
supporte le remplissage itératif parce que son débruiteur renvoie à chaque
pas une image plausible ; un réseau déplié n'offre pas cette garantie.

**Résultats** (``examples/usrnet_benchmark.py``, [benchmark §8](benchmark.md) ;
poids publiés, PSF connue) :

Moyenne du PSNR (dB) sur deux scènes 256 × 256 (« astronaut » et un
extrait d'orthophotographie) et trois bruits (σ = 0.003, 0.01, 0.03) ;
``pnp_hqs`` avec son débruiteur par défaut (NL-Ridge) :

| Flou | ``wiener_hunt`` | ``tv_primal_dual`` | ``pnp_hqs`` | **``usrnet``** | ``usrnet_tiny`` |
|---|---|---|---|---|---|
| gaussien σ = 1.6 px | 29.10 | 29.34 | 30.17 | **31.00** | 29.48 |
| Airy (anneau à 2.5 px) | 32.24 | 32.32 | 34.70 | **35.07** | 30.66 |
| bougé 9 px | 29.11 | 29.92 | 29.47 | **32.44** | 24.39 |
| optique (Q = 1, défoc. 0.15, jitter 0.3) | 32.12 | 32.53 | 34.13 | **35.21** | 27.68 |
| **moyenne des 24 cas** | 30.64 | 31.03 | 32.12 | **33.43** | 28.05 |

USRNet dépasse la TV dans les 24 cas (+1.0 à +4.1 dB) et ``pnp_hqs`` dans
21 (de −1.0 dB, Airy sur l'orthophotographie, à +10 dB, bougé à faible
bruit), en ~0.5 s. Avec une PSF fausse (σ gaussien 1.6 → 1.8 ;
défocalisation 0.15 → 0.10 onde RMS), il reste au niveau de ``pnp_hqs``
(−0.3 à +0.3 dB) et au-dessus de la TV.

**Limites.**

- Poids publiés appris sur des **photographies**, avec des noyaux
  gaussiens et de bougé (25 × 25) et σ ≤ 25/255 : rien ne garantit leur
  comportement sur d'autres PSF, d'autres statistiques de scène ou un
  bruit poisson-gaussien — d'où le réentraînement (§7.3).
- Le modèle réduit ``usrnet_tiny`` s'effondre à fort bruit sur les PSF à
  FTM étalée (Airy, optique : 19 dB à σ = 0.03, sous l'image observée) :
  le réserver aux bruits faibles, ou le réentraîner.
- Comme toute méthode apprise, le réseau peut **inventer** des détails
  plausibles : vérifier l'écart à la vérité par bande de fréquence
  au-delà de la coupure de la MTF avant tout usage quantitatif.
- La PSF doit être juste : sensibilité à une PSF fausse au
  [benchmark §8](benchmark.md) ; avec une PSF estimée, enchaîner ``blind_optics``
  (estimation) puis ``usrnet`` (restauration, ``restore="usrnet"``).
- Super-résolution (s > 1) non exposée : la priorité est la
  déconvolution ; le module de données la gère déjà (Zhao et al. 2016).

### 7.3 Réentraînement : protocole et outils

Les poids publiés ne connaissent ni les PSF d'un instrument d'observation
de la Terre (diffraction d'une pupille obturée à Q ≈ 1, aberrations,
jitter, filé) ni ses scènes ni son bruit. Comme la PSF est une entrée du
réseau, il suffit de lui **montrer la bonne famille de flous** : réglage
fin depuis les poids publiés, ou entraînement complet.

**Outils** (``training.py``, extra ``deep`` ; boucle d'apprentissage dans
``eoqual.neural.unrolled.training``, paires de déconvolution ici) :

| Outil | Rôle |
|---|---|
| ``PSFSampler`` | loi des PSF : modèle optique (``optical_psf`` : Q, obturation, aberrations de Zernike, jitter, dans des plages), gaussiennes, filés ; probabilités par famille |
| ``NoiseSampler`` | bruit gaussien ou poisson-gaussien (``σ² = a + b·L``, a plancher, b pente), niveau tiré dans une plage, part sans bruit |
| ``load_training_images`` | images nettes en niveaux de gris dans [0, 1] ; linéarisation (``gamma``) et réduction (``reduce``) |
| ``PatchSampler`` | paires à la volée : vignette au hasard, retournement et rotation, PSF tirée, bruit ; ``borders="real"`` (défaut) : zone plus grande floutée puis recadrée, bruit, marge complétée par Wiener–Hunt **comme à l'inférence** (§7.2) ; ``borders="circular"`` : flou circulaire (cadre exact du module de données, plus rapide) |
| ``TrainingConfig`` | poids de départ (``"usrnet"``, ``"usrnet_tiny"``, un fichier, ou aucun avec ``arch``), itérations, lot, vignette, taux et paliers, perte L1, graine, appareil, validation |
| ``train_usrnet`` | boucle Adam, validation périodique sur des images **disjointes**, sauvegarde du meilleur point de contrôle |
| ``evaluate_usrnet`` | PSNR moyen sur des paires de validation à tirage fixe |
| ``examples/train_usrnet.py`` + ``train_usrnet.jsonc`` | tout le protocole en ligne de commande, configuration commentée ; évaluation finale contre ``wiener_hunt`` et les poids de départ |

Produits : ``best.pth`` et ``last.pth`` (poids, architecture, réglages,
historique ; ``deconvolve(..., "usrnet", weights="…/best.pth")`` les
charge directement), ``history.csv``, ``config.json``, ``best.sha256``,
``evaluation.json``.

**Protocole.**

1. **Décrire l'instrument** : plages de Q (longueurs d'onde extrêmes de
   la bande), obturation et pupille, plages plausibles des aberrations
   (budget de front d'onde, dérives thermiques), jitter (spécification du
   contrôle d'attitude), filé résiduel (TDI), modèle de bruit (PTC,
   eoqual-measure : ``a`` et ``b``). Ce sont les plages de
   ``PSFSampler`` et ``NoiseSampler`` ; les élargir un peu au-delà du
   nominal rend le réseau robuste à une PSF mal connue.
2. **Choisir les images nettes** : licence claire (Licence Ouverte 2.0,
   CC BY…), registre des sources ; linéariser (le flou s'applique à la
   luminance linéaire) ; réduire (moyenne 2 × 2 ou plus) pour des
   vignettes nettes à l'échelle du pixel, sans flou propre ([§8.2.5](aveugle.md#825-ce-qui-est-identifiable--et-ce-qui-ne-lest-pas)) ;
   varier les paysages (urbain, agricole, forêt, eau, relief).
3. **Séparer** entraînement et validation par **zones géographiques
   disjointes** (pas seulement des vignettes différentes) ; garder une
   troisième zone, jamais vue, pour l'évaluation finale.
4. **Point de départ** : réglage fin depuis ``"usrnet"`` (taux 5·10⁻⁵,
   quelques 10³ à 10⁴ itérations ; sur GPU Apple, lot de 8 : ~2 s par
   itération à bords réels, ~1 s en flou circulaire) ; entraînement complet (``init_weights=None``,
   taux 10⁻⁴, ~4·10⁵ itérations comme KAIR, GPU dédié) seulement si le
   domaine est très éloigné.
5. **Suivre** le PSNR de validation (``history.csv``) ; garder le meilleur
   point de contrôle ; arrêter quand il plafonne.
6. **Évaluer** sur la zone réservée, à PSF tirées de la même loi **et**
   hors de ses plages : face à ``wiener_hunt``, ``tv_primal_dual``, aux
   poids de départ ; avec une PSF fausse (défocalisation mal estimée) ;
   écart à la vérité par bande de fréquence (détails inventés).
7. **Publier** les poids sous une licence explicite (MIT ou CC BY 4.0),
   avec leur empreinte SHA-256, la configuration, les versions (paquet,
   PyTorch) et le registre des images ; ne jamais les inclure dans le
   paquet pip.
8. **Reproductibilité** : graine fixée, configuration versionnée ; le
   tirage des paires est déterministe pour une graine donnée, au
   non-déterminisme près des noyaux GPU.

**Démonstration** (orthophotographie de Toulouse, Licence Ouverte 2.0,
réduite d'un facteur 2 ; entraînement sur les trois quarts nord,
validation sur le quart sud ; configuration ``train_usrnet.jsonc``,
lot de 8 vignettes 96 × 96). Deux réglages fins depuis les poids
publiés, évalués sur les **mêmes** 24 vignettes 256 × 256 du quart sud,
à bords réels (``boundary="unknown"``), PSF tirées de la loi, σ de
0.004 à 0.039 :

| Mesure | Poids publiés | Flou circulaire, 2 000 itérations | Bords réels, 6 000 itérations |
|---|---|---|---|
| PSNR moyen (évaluation finale) | 31.89 dB | 32.33 dB (+0.44) | **32.33 dB (+0.44)** |
| gain médian | | +0.23 dB | +0.24 dB |
| gain, plage selon le cas | | −0.39 à +3.32 dB | **−0.11** à +3.43 dB |
| cas améliorés | | 21 / 24 | **23 / 24** |
| PSNR de validation pendant l'entraînement (paires du réglage, non comparables entre colonnes) | 34.53 / 29.74 dB | 36.60 dB | 30.76 dB |

Référence sur les mêmes vignettes : observée 27.09 dB, ``wiener_hunt``
29.78 dB.

Lecture :

- **le réglage fin apporte +0.4 dB en moyenne**, concentré là où les
  poids publiés étaient faibles : cas où le départ ne dépassait
  ``wiener_hunt`` que d'un décibel au plus, voire lui était inférieur,
  surtout à fort bruit (σ > 0.03) — gains de +0.3 à +3.4 dB, +1.4 en
  moyenne ; ailleurs, de −0.1 à +0.6 dB. Les poids
  publiés étaient déjà bons sur ces PSF (31.9 dB contre 29.8 pour
  ``wiener_hunt``) ;
- **les bords réels ne relèvent pas la moyenne mais suppriment les
  pertes** : le réseau voit à l'entraînement la marge complétée par
  Wiener–Hunt qu'il rencontre à l'usage (§7.2), si bien qu'aucun cas ne
  se dégrade de plus de 0.11 dB (contre 0.39 dB en flou circulaire).
  C'est le réglage par défaut (``borders="real"``) ;
- **le PSNR de validation de** ``history.csv`` **ne se compare pas d'un
  réglage à l'autre** : en flou circulaire, il est mesuré dans le cadre
  exact du module de données et surestime le gain (+2.1 dB contre +0.44
  à l'usage) ; à bords réels, il suit fidèlement l'évaluation finale
  (+1.0 dB sur des vignettes plus petites, où la marge pèse plus).
  Toujours juger un réentraînement sur l'évaluation finale
  (``evaluation.json``) ;
- **plateau** : le PSNR de validation progresse surtout pendant les
  1 000 premières itérations (+0.8 dB), puis de 0.2 dB jusqu'à 3 500 et
  plus du tout ensuite (30.73 à 30.76 dB) ; les deux divisions du taux
  par 2 (paliers à 50 et 75 %) restent sans effet. Au-delà, le gain est à chercher
  dans des **données** plus variées (plusieurs scènes et paysages,
  §7.3 étape 2), pas dans des itérations.

Coût : la complétion de marge double le temps d'une itération (~2 s
contre ~1 s sur GPU Apple, lot de 8) ; 6 000 itérations font environ
3 h 30 de calcul.

*Suite* : un apprentissage **auto-supervisé** (pertes de DeepInv :
imagerie équivariante, SURE, R2R) permettrait d'adapter le réseau à des
images réelles d'un capteur sans images nettes de référence ([§12.2.1](pistes.md#1221-panorama-des-solutions-neuronales-vérifié-le-2026-11-08)).
