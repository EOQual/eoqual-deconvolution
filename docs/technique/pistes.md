# Pistes et variantes écartées

## 12. Pistes

Séparées en méthodes classiques (sans apprentissage, socle numpy/scipy)
et méthodes apprises (extra ``deep``, PyTorch), qui posent en plus des
questions de licence des poids, de données d'entraînement et de
validation.

### 12.1 Méthodes classiques

- **Bruit de Poisson** : déconvolution Poisson + TV par ADMM (PIDAL,
  Figueiredo & Bioucas-Dias 2010) pour les images à faible flux ;
  stabilisation de variance avant les méthodes à bruit gaussien (comme
  ``vst_denoise`` d'eoqual-denoising). *Pourquoi pas encore* : le calcul
  est léger, mais l'interface ne l'est pas — l'attache de Poisson exige
  des données en photo-électrons (gain, offset, bruit de lecture du
  capteur, comme la PTC d'eoqual-measure), un modèle poisson-gaussien
  pour être réaliste, et un benchmark à faible flux dédié ;
  ``richardson_lucy(tv_weight=...)`` couvre déjà le cas Poisson + TV.
- **SPITFIR(e)** (Prigent et al. 2023) : a priori mêlant hessienne et
  parcimonie, rapide, conçu pour la microscopie de fluorescence 3D.
  *Pourquoi pas encore* : à réimplémenter depuis l'article (le code de
  référence, sdeconv, n'est pas repris), et son intérêt sur des scènes
  d'observation de la Terre — texturées, sans fond noir — reste à
  démontrer face à la TV et au Plug-and-Play.
- **Richardson-Lucy à pas adaptatif** (gradient projeté mis à l'échelle,
  SGP, Bonettini, Zanella & Zanni 2009), équivalent pour RL du pas de
  Barzilai-Borwein de Landweber.
- Restauration vers une **FTM cible** (filtre qui ramène la FTM du
  système à une FTM de référence plutôt qu'à 1) : à étudier.
- PSF variable dans le champ **connue** (carte de PSF fournie par
  l'utilisateur, restauration par tuiles) : le mécanisme existe dans
  ``blind_optics`` ([§8.2.6](aveugle.md#826-variation-dans-le-champ)), à exposer pour des PSF données.
- ``blind_optics`` : filé (longueur, direction) comme paramètre du
  modèle optique, pour les imageurs à balayage ou à TDI mal synchronisé
  ([§10.1](psf-simulation.md#101-jitter-et-filé--deux-mouvements-différents)) ; **diversité** (deux images à défocalisations
  connues, comme la diversité de phase d'eoqual-phaseretrieval) pour
  lever l'ambiguïté de signe et rendre les aberrations impaires
  identifiables ; a priori sur θ issu d'une mesure de FTM sur bord
  (eoqual-measure) ; gradient analytique de ``J(θ)`` ; modèle
  d'échantillonnage avec repliement.
- **Terme d'erreur parcimonieux** pour une PSF imparfaite et **base de
  PSF à poids variables** dans le champ : idées issues d'ErA, transposables
  sans apprentissage (protocole au §12.2.6).
- Aveugle : sélection des contours saillants (Cho & Lee 2009), a priori
  sur les intensités (Pan et al. 2014), pixels saturés (étoiles, reflets).

### 12.2 Méthodes apprises (extra ``deep``)

#### 12.2.1 Panorama des solutions neuronales (vérifié le 2026-11-08)

Critères : **licence** du code et des poids compatible avec un module MIT,
**publication** décrivant la méthode, **code d'entraînement** publié (le
réentraînement sur des PSF d'instrument et des scènes d'observation de la
Terre est indispensable : les poids publiés sont appris sur des
photographies), **poids** disponibles. Licences lues sur les dépôts
(fichier LICENSE ou métadonnées GitHub / Hugging Face).

*Recommandées — code permissif, entraînement publié, poids disponibles :*

| Solution | Principe | Code | Entraînement | Poids | Intérêt pour l'imagerie spatiale |
|---|---|---|---|---|---|
| **USRNet** (Zhang, Van Gool & Timofte, CVPR 2020), KAIR | HQS déplié, **PSF, σ et facteur d'échelle en entrée** | MIT | ✅ ``main_train_usrnet.py`` (KAIR) | MIT (``usrnet.pth``, ``usrnet_tiny.pth``) | **intégré** : ``usrnet`` ([§7.2](plug-and-play.md#72-usrnet--réseau-déplié-extra-deep)), réentraînement [§7.3](plug-and-play.md#73-réentraînement--protocole-et-outils) |
| **DeepInv** (déjà dans l'extra ``deep``) | bibliothèque : algorithmes déroulés, DRUNet, RAM, générateurs de PSF, **pertes auto-supervisées** (imagerie équivariante, SURE, R2R) | BSD-3-Clause | ✅ entraîneur générique | DRUNet : BSD-3-Clause ; RAM : licence **non déclarée** sur Hugging Face (``mterris/ram``) | **entraîner sans vérité terrain**, sur des images réelles d'un capteur |
| **Galaxy-Deconv** (Li & Alexander, MNRAS) | ADMM déroulé, PSF en entrée, bruit de Poisson (galaxies, lentillage faible) | MIT | ✅ ``train.py`` + simulation GalSim | ✅ ``saved_models`` | imagerie spatiale à PSF connue et faible flux ; référence compacte de déroulement physique |

*Utilisables, mais génériques ou orientées produit :*

- **NAFNet**, **Restormer** (MIT, entraînement et poids) : réseaux de
  restauration génériques, de bout en bout, sans PSF en entrée ;
  réentraînables sur paires simulées, mais rien n'empêche d'inventer des
  détails — pour l'aspect, pas pour la mesure.
- **Satlas Super-Resolution** (Allen AI, Apache-2.0, entraînement et
  poids ; Sentinel-2 → NAIP) : super-résolution de type GAN, qui invente
  des détails ; licence des poids et des données à vérifier.
- **SEN2SR** (ESA OpenSR, code CC0, poids Hugging Face) : super-résolution
  Sentinel-2 à cohérence radiométrique ; licence des poids non vérifiée,
  entraînement absent du dépôt.
- **HighRes-net** (super-résolution multi-images, PROBA-V de l'ESA,
  entraînement publié) : licence cumulant Apache-2.0 et « Do No Harm
  License » (restrictions d'usage) — à éviter dans un module MIT.

*Écartées pour la licence :* DSen2 (GPL-3.0, intégrable seulement en
dépendance externe), Z-PNN (GRIP-UNINA, usage non lucratif), KernelGAN
(Weizmann, académique non commercial), SelfDeblur et le dépôt d'origine de
l'imagerie équivariante (aucune licence ; l'imagerie équivariante est
réimplémentée sous BSD-3-Clause dans DeepInv), DWDN (aucune licence,
§12.2.4), poids de Deep-Blur (non commerciaux, §12.2.5).

*Suites retenues* : (1) **USRNet réentraîné sur nos PSF** — intégré
([§7.2](plug-and-play.md#72-usrnet--réseau-déplié-extra-deep), [§7.3](plug-and-play.md#73-réentraînement--protocole-et-outils)) ; (2) **apprentissage auto-supervisé de DeepInv** pour
adapter un réseau (USRNet, DRUNet) à un capteur réel sans images de
référence ; (3) **Galaxy-Deconv** comme référence de déroulement ADMM
pour PSF spatiale et bruit de Poisson. Restent à confirmer : licence des
poids de RAM, SEN2SR et Satlas.

#### 12.2.2 A priori de diffusion

- **A priori de diffusion** dans le Plug-and-Play : DiffPIR (Zhu et al.
  2023), échantillonnage a posteriori DPS (Chung et al. 2023), via deepinv
  (BSD-3-Clause, extra ``deep``) — comme les projets « ADMM deconvolution
  with diffusion prior » d'EE367 (Stanford). *Pourquoi pas encore* : poids
  pré-entraînés de plusieurs centaines de Mo (licences à vérifier une à
  une), 100 à 1000 passes de réseau par image (minutes sur CPU, GPU
  quasi indispensable), réseaux appris sur des photographies, pas sur
  des images de télédétection, et résultat **stochastique** pouvant
  « inventer » des détails plausibles — inacceptable sans protocole de
  validation pour une mesure quantitative.
#### 12.2.3 DPIR et calcul sur GPU

- **DPIR** (débruiteur DRUNet et calendrier dédiés) et **calcul sur GPU**
  (PyTorch) des méthodes itératives.
#### 12.2.4 DWDN

- **DWDN — Wiener profond** (Dong, Roth & Schiele, NeurIPS 2020 ;
  extension DWDN+, arXiv 2103.09962) : déconvolution **non aveugle** qui
  applique un filtre de Wiener dans un espace de caractéristiques appris
  (une convolution et trois blocs résiduels), avec rapport bruit/signal
  estimé par caractéristique, puis un raffinement multi-échelle
  encodeur-décodeur du grossier au fin ; entraîné de bout en bout, un seul
  modèle pour un bruit de 0 à 5 %. Annoncé +2 à +3 dB sur IRCNN et FNBD
  (jeux de Levin et de Sun, bruit 1 %). Prolongement naturel de
  ``wiener`` / ``wiener_hunt`` : la PSF reste une entrée, et
  ``blind_optics`` peut la fournir.

  *Licence* (vérifiée le 2026-11-08) : ni le dépôt GitHub
  (``dongjxjx/dwdn``) ni le projet d'origine (GitLab du MPI) ne portent de
  licence ; sans licence, tous droits réservés — **ni le code ni les poids
  publiés (``model_DWDN.pt``) ne peuvent être repris**, même adaptés,
  dans ce paquet MIT. L'article (licence arXiv, non exclusive) ne donne
  aucun droit sur le code ; les idées et les équations restent libres.

  *Le nécessaire pour l'ajouter* :

  1. **Réimplémentation propre** depuis l'article seul (comme le reste du
     paquet, sans consulter le dépôt) : module ``backends/deep_wiener.py``
     (PyTorch, extra ``deep``), signature commune
     ``deep_wiener_deconvolve(image, psf, sigma=None, weights=...)`` ;
     Wiener en caractéristiques calculé en Fourier avec l'OTF de la PSF
     (``eoqual.core.kernel.psf2otf``) et les bords de ``operators``
     (marge inconnue) ; architecture et pertes décrites dans un fichier
     de conception cité dans THIRD_PARTY_LICENSES.md [§3](methodes-classiques.md#3-méthodes-linéaires).
  2. **Données d'entraînement de licence claire**, adaptées à
     l'observation de la Terre : orthophotographies ouvertes (Licence
     Ouverte 2.0, Etalab, réutilisation commerciale permise), réduites et
     découpées en vignettes 256 × 256 ; **pas** les jeux de l'article
     (BSD, Waterloo Exploration : photographies, conditions d'usage à
     vérifier une à une). Registre des sources d'images et de leurs
     licences versionné avec le script d'entraînement.
  3. **Simulation des paires** (vignette nette, vignette floue bruitée)
     avec les briques du paquet : ``psf.optical_psf`` (Q de 0.7 à 2,
     obturation, défocalisation et astigmatisme tirés dans des plages
     plausibles, jitter), ``motion_psf`` pour le filé ([§10.1](psf-simulation.md#101-jitter-et-filé--deux-mouvements-différents)), bruit
     gaussien et poisson-gaussien (``simulate.add_noise``) — le réseau
     apprend sur les flous d'instruments, pas sur des noyaux de bougé
     de photographie.
  4. **Entraînement** : script ``examples/train_deep_wiener.py``
     (reproductible, graines fixées), GPU indispensable, journal des
     hyperparamètres ; poids publiés sous une licence explicite (MIT ou
     CC-BY 4.0) dans une *release* du dépôt, téléchargés à la demande et
     vérifiés par empreinte SHA-256 (jamais dans le paquet pip).
  5. **Validation** avant toute annonce : banc du [benchmark](benchmark.md) (mêmes
     scènes, flous et bruits) et banc optique ([§8.2.7](aveugle.md#827-résultats)) face à
     ``wiener_hunt``, ``tv_primal_dual`` et ``pnp_hqs`` + DRUNet ;
     robustesse à une PSF fausse (défocalisation mal estimée), à des
     tailles et Q hors de l'entraînement, et vérification qu'aucun détail
     n'est inventé (écart à la vérité par bande de fréquence, au-delà de
     la coupure de la MTF).
  6. **Tests** : contrat d'appel et petit modèle aléatoire (sans poids)
     marqués ``deep`` ; test de non-régression sur les poids publiés
     (empreinte et PSNR sur une scène fixe).

  *Pourquoi pas encore* : pas de code ni de poids réutilisables, coût de
  l'entraînement, et gain à démontrer sur des flous d'instrument —
  réguliers, sans les noyaux de bougé complexes où l'article mesure son
  avance.
#### 12.2.5 Deep-Blur

- **Deep-Blur — identification apprise d'une PSF paramétrique**
  (Debarnot & Weiss, *Biological Imaging* 4, e13, 2024 ; article CC BY
  4.0) : la méthode publiée la plus proche de ``blind_optics`` ([§8.2](aveugle.md#82-blind_optics--aveugle-contrainte-par-loptique)).
  Même hypothèse — une PSF physique décrite par quelques coefficients de
  Zernike (Z4 à Z11, tirés dans ±0.15, pupille 2D ; variante à flou
  variable dans le champ par base de PSF et poids) —, mais l'estimateur
  est **appris** : un ResNet (11 M de paramètres) lit l'image floue et
  renvoie directement les coefficients ; un Douglas–Rachford déroulé
  (4 itérations, DRUNet comme proximal, départ par gradient conjugué)
  restaure ensuite l'image avec la PSF **estimée**, ce qui lui apprend à
  absorber l'erreur d'identification. Entraînement : MS COCO, bruit
  poisson-gaussien. Annoncé : coefficients à 5 % près en moyenne sans
  bruit, PSF estimée à 13 dB (8 dB à fort bruit) quand les méthodes
  classiques comparées sont négatives, 0.3 s par image 400 × 400 sur GPU.

  *Intérêt* : un réseau exploite les statistiques de la scène au-delà de
  l'ordre deux — potentiellement ce qui manque à ``blind_optics`` pour la
  petite défocalisation confondue avec le jitter et pour les aberrations
  impaires ([§8.2.5](aveugle.md#825-ce-qui-est-identifiable--et-ce-qui-ne-lest-pas), [§8.2.7](aveugle.md#827-résultats)). Forme envisagée : un troisième estimateur,
  ``estimator="learned"`` (extra ``deep``), qui garde le modèle optique
  (``optical_psf``), les tuiles et la restauration du paquet.

  *Licences* (vérifiées le 2026-11-08, détail dans
  THIRD_PARTY_LICENSES.md [§3.1](methodes-classiques.md#31-inverse--filtre-inverse-tronqué)) : code des auteurs sous **CC0 1.0**
  (dépôt ``pierre-weiss/deepblur``, publié en janvier 2025), donc
  réutilisable ; fichiers tiers embarqués sous leurs licences d'origine,
  permissives (KAIR et sigmanet : MIT ; BasicSR et Deep Image Prior :
  Apache-2.0, notices à conserver) ; **poids** (Hugging Face
  ``pweiss/DeepBlur``) sous **CC BY-NC 4.0**, non commerciale : non
  reprenables ; code d'entraînement non publié (version mise à jour
  annoncée dans DeepInv, BSD-3-Clause).

  *État dans DeepInv* (vérifié le 2026-11-08 sur la branche principale,
  commit ``e34c06f`` du 2026-11-06, ses tickets et demandes de fusion, et la
  copie personnelle ``pierre-weiss/deepinv``) : Deep-Blur **n'y est pas
  recodé**. On y trouve seulement la génération de PSF —
  ``DiffractionBlurGenerator`` (Zernike, 2D/3D) et
  ``ProductConvolutionBlurGenerator`` / ``SpaceVaryingBlur`` (flou
  variable par base de PSF et poids) ; des branches non fusionnées de la
  copie personnelle (juin 2026) portent sur un générateur de PSF
  polychromatique. **Pas de poids Deep-Blur** : seuls existent ceux de
  ``pweiss/DeepBlur`` (CC BY-NC 4.0). **Pas de code d'entraînement** :
  ni dans le dépôt d'origine (le README annonce qu'il ne sera pas publié),
  ni dans DeepInv, qui fournit seulement les briques pour l'écrire
  (générateurs de PSF à tirage aléatoire, entraîneur générique). À ne
  pas confondre : ``deepinv.models.KernelIdentificationNetwork`` est une
  autre méthode (Carbajal et al. 2023, J-MKPD), qui estime des noyaux
  **libres** (33 × 33, bougé) et leurs poids dans l'image, pas des
  coefficients de Zernike (poids ``deepinv/kernel_identification`` sous
  BSD-3-Clause ; dépôt d'origine sans licence, code « adapté avec la
  permission de l'auteur »).

  *Le nécessaire* :

  1. **Choisir la voie du code** : réutiliser l'architecture CC0 (avec
     notices des fichiers tiers et une exception documentée à la règle
     « écrit depuis les seules publications », THIRD_PARTY_LICENSES.md [§3](methodes-classiques.md#3-méthodes-linéaires))
     — la seule voie aujourd'hui, la version DeepInv n'existant pas
     encore ; DeepInv peut toutefois servir à générer les PSF
     d'entraînement, rôle que tient déjà ``optical_psf``.
  2. **Écrire l'entraînement** (non publié) : script reproductible
     ``examples/train_optics_identification.py``, graines fixées, GPU.
  3. **Réentraîner pour l'observation de la Terre** : vignettes
     d'orthophotographies ouvertes (Licence Ouverte 2.0) floutées par
     ``optical_psf`` à l'échantillonnage ``Q``, l'obturation et la pupille
     de l'instrument visé, coefficients tirés dans les plages de
     ``OpticalModel.free``, jitter et filé ([§10.1](psf-simulation.md#101-jitter-et-filé--deux-mouvements-différents)), bruit
     poisson-gaussien — un réseau par famille d'instruments.
  4. **Traiter le signe** : vérifier si le réseau lève l'ambiguïté
     ``φ → −φ`` (l'article ne le dit pas) ; sinon, contraindre les
     plages comme ``blind_optics``.
  5. **Valider** sur le banc [§8.2.7](aveugle.md#827-résultats) (mêmes cas, mêmes scènes) face aux
     estimateurs ``evidence`` et ``l0`` : erreur sur θ, PSNR après
     restauration, robustesse hors des plages d'entraînement ; publier
     les poids sous licence explicite (MIT ou CC BY 4.0), vérifiés par
     empreinte.

  *Limites relevées* :

  - domaine d'origine : microscopie à fluorescence (objectif ×100,
    ouverture 1.49), images 400 × 400 en niveaux de gris ; poids publiés
    inadaptés à un instrument d'observation de la Terre, et non
    commerciaux ;
  - un réseau d'identification est propre à une famille de PSF (Q,
    pupille, modes, plages) : tout changement impose un réentraînement ;
  - validation sur données réelles seulement visuelle dans l'article (pas
    de vérité) ; les auteurs notent que la méthode s'étend mal aux
    variations rapides de la PSF ;
  - comme toute méthode apprise : GPU, généralisation hors des plages
    d'entraînement non garantie — d'où le maintien des estimateurs
    ``evidence`` et ``l0``, interprétables, à côté.
#### 12.2.6 ErA

- **ErA — déroulement d'ALM « conscient de l'erreur »** (Vo & Park,
  atelier NeurIPS 2025 COML ; arXiv 2606.06540, licence CC BY 4.0) :
  déconvolution **aveugle** d'un flou de défocalisation **variable dans
  l'image**. Modèle : le flou en chaque pixel est une combinaison d'une
  petite base de noyaux appris (61 × 61) par des poids propres au pixel,
  prédits par un réseau d'estimation de noyau ; un terme d'erreur
  parcimonieux ``E`` absorbe les défauts du noyau estimé :

      min_{x,h,E}  ½‖h ⊗ x − y + E‖² + ‖E‖₁ + φ(x) + f(E)

  résolu par lagrangien augmenté (ALM) avec éclatement ``U = h ⊗ x``,
  ``Z = x``, ``P = E`` : mises à jour fermées pour ``U`` et ``x`` (Fourier),
  seuillage doux pour ``E``, deux ResUNet pour les proximaux de ``φ`` et
  ``f`` ; ``K = 10`` itérations déroulées et entraînées de bout en bout
  (perte ``ω‖x − x_vrai‖₁ + (1 − ω)‖h ⊗ x − y‖₁``).

  *Licence et disponibilité* (vérifiées le 2026-11-08) : article sous CC BY
  4.0 (texte et figures réutilisables avec attribution) ; **aucun code ni
  poids publiés** (ni sur arXiv, ni sur la page ou les dépôts GitHub de
  l'auteur). Une réimplémentation propre depuis l'article est donc
  possible sans obstacle de licence — et c'est le seul chemin.

  *Limites relevées* :

  - article court (4 pages + annexe) : l'architecture du réseau
    d'estimation de noyau, la taille de la base, les ResUNet et les
    hyperparamètres (``λ₁₋₃``, ``ω``) ne sont pas décrits — la méthode
    n'est pas reproductible telle quelle, il faudrait les reconstruire ;
  - domaine : photographies d'appareils grand public, flou de
    défocalisation de profondeur de champ (noyaux jusqu'à 61 px), appris
    sur 350 paires du jeu DPDD (*dual-pixel*, conditions d'usage à
    vérifier) ; sans rapport avec les flous réguliers, de quelques pixels,
    d'un instrument d'observation de la Terre ;
  - gain annoncé faible : +0.4 dB sur DPDD, +0.1 dB sur RealDOF, en
    dessous de concurrents sur RTF, LPIPS moins bon que certains ;
  - le terme ``E`` est appris par un réseau : rien ne garantit qu'il
    n'absorbe pas une partie de la scène, ce qui est inacceptable pour une
    mesure quantitative sans protocole de validation ;
  - comme toute méthode apprise : GPU pour l'entraînement, généralisation
    hors domaine non garantie.

  *Ce qui est transposable sans apprentissage* — à évaluer d'abord, car
  peu coûteux et compatible avec les méthodes du socle :

  1. **Terme d'erreur parcimonieux** pour une PSF imparfaite :
     ``min_{x,E} ½‖h ∗ x − y + E‖² + μ‖E‖₁ + λ TV(x)`` dans
     ``tv_primal_dual`` (un proximal de plus : seuillage doux de ``E``).
     Cible : les PSF estimées par ``blind_optics`` ou mesurées avec une
     erreur. Validation : PSNR en fonction de l'erreur de PSF
     (défocalisation fausse de ±0.05 onde RMS), et vérification que ``E``
     reste concentré sur les contours (là où une erreur de PSF se voit),
     pas sur les textures. Lien : Ji & Wang (2012), déconvolution robuste à
     un noyau inexact.
  2. **Base de PSF et poids par pixel** pour la variation dans le champ :
     au lieu du recollement par tuiles de ``blind_optics`` ([§8.2.6](aveugle.md#826-variation-dans-le-champ)), une
     petite base (PSF du modèle optique en quelques points du champ, ou
     ses composantes principales) et des poids qui varient lentement
     (polynôme de la position) ; l'opérateur ``x ↦ Σ_k w_k ⊙ (h_k ∗ x)``
     reste rapide (K FFT) et s'intègre aux méthodes itératives (Nagy &
     O'Leary 1998 ; Denis et al. 2015). Validation : banc
     ``examples/blind_optics.py`` (courbure de champ), contre les tuiles.

## 13. Variantes écartées

Mesures sur scènes synthétiques.

- **Marge miroir non périodique** (``np.pad(..., "symmetric")`` de chaque
  côté) : le raccord périodique des deux marges est discontinu ; GCV et
  discrépance choisissent alors λ ≈ 1e-10 (Tikhonov à −33 dB). La marge
  continue et périodique du [§2](modele-bords.md#2-bords-et-données-incomplètes) donne Tikhonov à 27.6 dB (optimum 27.6 dB)
  et le filtre inverse à +3.4 dB.
- **Wiener à β = 2 imposé**, amplitude par bilan d'énergie global : jusqu'à
  3.4 dB sous l'oracle, contre 0.3 à 0.4 dB en ajustant ``A`` et ``β`` dans
  la bande fiable ([§3.2](methodes-classiques.md#32-wiener--wiener-spectre-du-signal-ajusté)).
- **Plug-and-Play ADMM sans remise à l'échelle de la variable duale
  réduite** quand ρ change : divergence (18 dB).
- **Déconvolution aveugle par PALM** sur ``min ‖h ∗ x − y‖² + λTV(x)``
  (gradient projeté sur le simplexe pour ``h``) : le pas en ``h`` est figé
  par la constante de Lipschitz des basses fréquences (spectre en 1/f²) ;
  en domaine des gradients avec FISTA, la PSF converge vers le Dirac
  (λ faible) ou vers une PSF décalée (λ fort).
- **A priori L0 partant d'un filtre inverse**, avec ``γ = 1e-3`` : le
  départ est à −8 dB du pas en x et γ est trop fort ; partir de l'image
  observée avec ``γ = 1e-4`` gagne +0.6 à +3.2 dB sur six cas.
- **Préfiltre anti-repliement de la pyramide** (aveugle) : le flou du
  préfiltre passe dans la PSF estimée.
- **Wiener–Hunt à a priori de gradient** : 0.8 dB de moins en moyenne que
  l'a priori laplacien de Hunt ([§3.4](methodes-classiques.md#34-tikhonov)).
- **``blind_optics``** ([§8.2](aveugle.md#82-blind_optics--aveugle-contrainte-par-loptique)) :
  - un a priori de scène à exposant **libre** laisse le critère attribuer
    une petite défocalisation à la pente du spectre ; le laplacien (β = 4)
    sous-estime les défocalisations ; β = 3 fixé donne le plus petit écart
    moyen ([§8.2.3](aveugle.md#823-estimation-par-vraisemblance-marginale-estimatorevidence-défaut)) ;
  - sur une scène réduite par moyenne de blocs, le jitter estimé inclut le
    flou de cette moyenne ([§8.2.5](aveugle.md#825-ce-qui-est-identifiable--et-ce-qui-ne-lest-pas)) : les scènes du banc sont réduites par
    moyenne 2 × 2 puis sous-échantillonnage, plus nettes à l'échelle du
    pixel ;
  - avec cinq paramètres libres (défocalisation, deux astigmatismes, deux
    comas), la coma et l'astigmatisme absorbent l'anisotropie de la scène :
    ne libérer que les défauts attendus ;
  - une variante « evidence puis l0 » (le premier critère initialise le
    second) n'apporte rien sur l0 seul.
- **Trois mécanismes séparés** pour le masque, la saturation et les bords
  inconnus : une seule opération de remplissage ([§2.2](modele-bords.md#22-données-incomplètes--pixels-masqués-saturés-marge-inconnue)) les traite tous et
  rend l'attache unilatérale de la saturation compatible avec toutes les
  méthodes, y compris linéaires (EM). Un masque des pixels saturés seul
  (sans contrainte ``≥ L``) reste possible via ``mask=``, mais la
  contrainte unilatérale récupère davantage d'information.
