# PSF, simulation et tests

## 10. PSF, simulation, figures

- ``psf.py`` : ``gaussian_psf`` (anisotrope, rotation), ``airy_psf``
  (rayon du premier anneau noir en pixels), ``moffat_psf``, intégrées sur
  la surface du pixel (``eoqual.core.psf.integrated_profile``) ;
  ``disk_psf`` (défocalisation) et ``motion_psf`` (bougé) anti-crénelées
  par sur-échantillonnage ; ``center_psf`` (barycentre au centre d'un
  support impair, interpolation spline).
- **PSF d'instrument** : ``system_psf(wavelength, diameter, focal_length,
  pixel_pitch, obscuration=, defocus_radius=, motion_length=,
  motion_angle=, jitter_sigma=)``. Sur une grille ``oversample`` fois plus
  fine (7 par défaut) : diffraction d'une pupille annulaire
  (``annular_airy`` : ``[2J1(v)/v − ε²·2J1(εv)/(εv)]²/(1 − ε²)²``, Born &
  Wolf ; Mahajan), convolée par la défocalisation (disque), le filé
  (segment) et le jitter (gaussienne), puis **intégrée sur la surface de
  chaque pixel** (sommation par blocs) — d'où la FTM ``sinc(πf)`` du
  détecteur. Rayon d'Airy en pixels : ``eoqual.core.psf.airy_radius_px``
  (``1.22 λf/(Dp)``). Validation : FTM = FTM de diffraction × sinc à 1 %
  près (``tests/test_socle.py``). Les ailes de la tache d'Airy décroissent
  en 1/r³ : avec le support par défaut (quatre anneaux), la FTM aux
  fréquences moyennes est surestimée d'environ 2 % ; 121 pixels la
  ramènent à 0.8 %.
- **PSF aberrée** : ``optical_psf(sampling, zernike=, obscuration=,
  jitter_sigma=, weights=)`` — diffraction en optique de Fourier d'une
  pupille annulaire avec front d'onde en polynômes de Zernike, jitter et
  intégration sur le pixel ; modèle de ``blind_optics`` ([§8.2.2](aveugle.md#822-modèle-optique-psfoptical_psf)) ;
  ``defocus_to_zernike`` convertit un déplacement du plan focal.
- ``combine_psfs`` : convolution de PSF (chaîne optique ⊛ détecteur ⊛
  bougé) ; les variances s'ajoutent.
- ``simulate.py`` : ``blur`` (circulaire ou miroir), ``add_noise``
  (gaussien, Poisson par ``gain``), ``simulate_observation``.
- ``plotting.py`` (Plotly via eoqual-viz) : ``plot_restoration``
  (galerie avec PSNR), ``plot_psf``, ``plot_mtf`` (coupe de ``|H|``
  jusqu'à Nyquist), ``plot_convergence`` avec ``ConvergenceRecorder``
  (rappel ``callback`` des méthodes itératives), ``plot_gcv``,
  ``plot_optimization``.

### 10.1 Jitter et filé : deux mouvements différents

Les deux termes désignent un déplacement de la ligne de visée pendant le
temps d'intégration ``T``, et la littérature les emploie parfois l'un
pour l'autre (« image motion », « smear », « jitter »). Ils décrivent
pourtant deux mouvements différents, à la FTM différente. Ce qui les
sépare est la **durée caractéristique du mouvement comparée à ``T``**
(Wulich & Kopeika 1987 ; Fiete 2010) :

| | Filé (*smear*, *linear motion blur*) | Jitter |
|---|---|---|
| Mouvement | dérive **déterministe**, à vitesse constante pendant ``T`` | vibrations **aléatoires**, nombreuses oscillations pendant ``T`` |
| Origines typiques | défilement mal compensé (vitesse au sol, synchronisation d'un TDI, dépointage lent), bougé | micro-vibrations : roues à réaction, cryoréfrigérateurs, mécanismes, structure |
| PSF | segment de longueur ``L`` dans la direction du mouvement | gaussienne d'écart-type ``σ`` (somme de nombreux petits déplacements indépendants) |
| FTM | ``|sinc(π L f∥)| = |sin(π L f∥)/(π L f∥)|``, **directionnelle**, nulle en ``f∥ = k/L`` | ``exp(−2π² σ² f²)``, sans zéro, isotrope si les vibrations le sont |
| Paramètre dans ce paquet | ``system_psf(motion_length=, motion_angle=)``, ``motion_psf`` | ``system_psf(jitter_sigma=)``, ``optical_psf(jitter_sigma=)``, paramètre ``"jitter"`` de ``blind_optics`` |

Cas intermédiaire : une vibration **sinusoïdale** d'amplitude ``a`` et de
période petite devant ``T`` donne la FTM ``|J₀(2π a f)|`` (Bessel) ; une
vibration de période **longue** devant ``T`` ne floute presque pas
l'image mais la déforme (déplacement variable d'une ligne ou d'une image
à l'autre) — un problème de géométrie, pas de déconvolution.

**Pourquoi on les confond, et quand cela compte.** Aux basses
fréquences, les deux FTM coïncident à variance de déplacement égale :
``sinc(πLf) ≈ 1 − (πLf)²/6 ≈ exp(−2π² (L²/12) f²)``, soit un jitter
d'écart-type ``σ = L/√12``. Elles divergent aux hautes fréquences, d'autant
plus que le filé est long :

| Filé ``L`` (px) | jitter équivalent ``σ = L/√12`` | FTM à 0.25 cycle/px (filé / jitter) | FTM à Nyquist 0.5 (filé / jitter) |
|---|---|---|---|
| 1 | 0.29 | 0.90 / 0.90 | 0.64 / 0.66 |
| 2 | 0.58 | 0.64 / 0.66 | **0.00** / 0.19 |
| 3 | 0.87 | 0.30 / 0.40 | 0.21 / 0.03 |

Conséquences :

- un **filé d'un pixel au plus** se modélise indifféremment par un
  jitter ``σ = L/√12`` (écart de FTM < 0.03 jusqu'à Nyquist), mais
  seulement dans la direction du filé : le jitter de ce paquet est
  isotrope ;
- au-delà, le filé a des **zéros** que le jitter n'a pas : déconvoluer
  un filé avec un jitter équivalent ne restaure pas les bonnes fréquences
  (et un filé de 3 px garde plus de contraste à Nyquist qu'un jitter
  « équivalent », 0.21 contre 0.03, à cause des lobes secondaires du
  sinus cardinal) ;
- dans ``blind_optics``, le paramètre ``"jitter"`` peut absorber un
  petit filé, au même titre qu'une petite défocalisation ([§8.2.5](aveugle.md#825-ce-qui-est-identifiable--et-ce-qui-ne-lest-pas)) ; un
  jitter estimé nettement plus grand que celui attendu de la plateforme
  doit faire soupçonner un filé (ou un flou de la scène). Le filé n'est
  pas un paramètre de ``blind_optics`` ([§12.1](pistes.md#121-méthodes-classiques)) ;
- **imageur à balayage (push-broom)** : pendant le temps de ligne, la
  scène défile d'un pixel dans le sens de la trace ; ce défilement
  nominal est un filé de longueur ``L = 1`` px (FTM ``sinc(π f)`` le long
  de la trace, en plus de celle du pixel) — sauf en TDI, où il est
  compensé, et où seul le défaut de synchronisation laisse un filé
  résiduel.

## 11. Tests

``pytest`` (``pytest -m "not slow"`` : passe rapide). Données
synthétiques seulement (« astronaut » de scikit-image, flou et bruit
connus).

| Fichier | Contenu |
|---|---|
| ``test_socle.py`` | PSF (somme, centrage, formes), PSF système (FTM de diffraction × pixel, obturation, filé), contrôles, simulation, adjoints ``∇``/``div`` et ``H``/``Hᵀ``, bords (aller-retour, continuité périodique) |
| ``test_incomplete.py`` | lignes mortes masquées, NaN, saturation, marge inconnue sur flou de bougé, états des pixels, pas de Barzilai-Borwein |
| ``test_reference_cvxpy.py`` | TV primal-duale = optimum calculé par CVXPY (sauté sans CVXPY) |
| ``test_methods.py`` | contrat d'appel, gain minimal par méthode, PSF Dirac, invariance d'échelle, bords, bandes, cas d'erreur, propriétés (Wiener = Tikhonov identité, accélération RL, arrêt de Landweber, décroissance TV) |
| ``test_tuning.py`` | GCV proche de l'oracle, discrépance, L-curve, espaces de recherche, optimisation (avec et sans référence) |
| ``test_usrnet.py`` | OTF = ``psf2otf``, module de données = formule fermée de Wiener–Tikhonov, lois de tirage, entraînement depuis zéro et rechargement ; avec les poids publiés (marqueur ``deep``) : nombre de paramètres, gain sur Tikhonov, marge inconnue sur bougé (> miroir + 10 dB, > TV), lignes masquées |
| ``test_pnp.py`` | débruiteur fourni, ``tv_chambolle``, ``nl_ridge`` ; ``drunet`` (marqueur ``deep``) |
| ``test_blind.py`` | PSF estimée plus proche de la vraie que le Dirac, gain positif |
| ``test_optics.py`` | ``optical_psf`` = Airy analytique, FTM sous la diffraction, symétrie ``φ → −φ``, défocalisation physique, bande spectrale ; Wiener–Hunt (σ estimé, égal à Tikhonov à λ imposé, critère profilé) ; ``blind_optics`` (validation du modèle, défocalisation retrouvée, tuiles, modèles de champ) |
| ``test_plotting.py``, ``test_examples.py`` | figures, exemples (dont le tutoriel) |
