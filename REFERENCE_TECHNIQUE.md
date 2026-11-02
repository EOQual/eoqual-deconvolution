# Référence technique — eoqual-deconvolution

Principe, implémentation, paramètres et limites de chaque méthode. Les
résultats chiffrés sont dans [BENCHMARK.md](BENCHMARK.md), les
références complètes dans [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md).

## Sommaire

1. [Modèle et notations](#1-modèle-et-notations)
2. [Bords et données incomplètes](#2-bords-et-données-incomplètes)
3. [Méthodes linéaires](#3-méthodes-linéaires)
4. [Méthodes itératives](#4-méthodes-itératives)
5. [Choix du paramètre des méthodes linéaires](#5-choix-du-paramètre-des-méthodes-linéaires)
6. [Méthodes variationnelles](#6-méthodes-variationnelles)
7. [Plug-and-Play](#7-plug-and-play)
8. [Déconvolution aveugle](#8-déconvolution-aveugle)
9. [Optimisation des paramètres](#9-optimisation-des-paramètres)
10. [PSF, simulation, figures](#10-psf-simulation-figures)
11. [Tests](#11-tests)
12. [Évolutions envisagées](#12-évolutions-envisagées)
- [Annexe A — Historique des choix et corrections](#annexe-a--historique-des-choix-et-corrections)

---

## 1. Modèle et notations

Observation ``y = h ∗ x + b`` : scène ``x``, PSF ``h`` (positive, somme
1), bruit ``b`` blanc d'écart-type σ. Sur un domaine périodique de ``N``
pixels, avec la FFT non normalisée : ``Y = H·X + B``, ``E|B(ω)|² = Nσ²``,
``H = psf2otf(h)`` (eoqual-core : centre de la PSF ``(kh // 2, kw // 2)``
ramené à l'origine, donc ``H`` réelle pour une PSF symétrique).

``BlurOperator`` (``operators.py``) porte ``H``, les opérations
``Hx``, ``Hᵀx``, le gradient ``Hᵀ(Hx − y)`` de ``½‖Hx − y‖²`` et sa
constante de Lipschitz ``max|H|²`` (1 pour une PSF positive de somme 1).

**Pourquoi la FFT.** La convolution est une application linéaire
``y = Hx`` dont la matrice ``N × N`` est inutilisable telle quelle
(10¹⁰ coefficients pour 10⁵ pixels). Sur un domaine périodique, ``H`` est
**circulante par blocs à blocs circulants** (BCCB) : la transformée de
Fourier discrète la diagonalise, et ses valeurs propres sont les
coefficients ``H(ω)``. ``Hx``, ``Hᵀx`` (multiplication par ``H*``) et
``(HᵀH + λLᵀL)⁻¹`` (division par ``|H|² + λ|L|²`` si ``L`` est aussi une
convolution) coûtent une FFT, ``O(N log N)``. Tout l'enjeu du traitement
des bords (§2) est de rendre ce cadre périodique légitime pour une image
qui ne l'est pas. Illustration pas à pas : [TUTORIEL.md](TUTORIEL.md) §1.

**Unités.** σ est toujours dans les unités de l'image. Les méthodes non
linéaires travaillent sur l'image ramenée à [0, 1] (``data_range``,
défaut : min/max de l'image) et expriment leur poids de régularisation en
**multiples de σ** ; les méthodes linéaires ont des paramètres sans
dimension. Toutes sont donc invariantes par changement d'échelle et de
décalage (sauf Richardson-Lucy : échelle seulement, modèle de Poisson),
ce que vérifie ``tests/test_methods.py::test_scale_invariance``.

**σ inconnu.** ``sigma=None`` : estimation de Donoho & Johnstone sur la
sous-bande d'ondelettes diagonale la plus fine
(``eoqual.core.noise.estimate_noise_sigma``, identique à
``skimage.restoration.estimate_sigma``). Sur une image floue, cette
sous-bande contient peu de signal : l'estimation est fiable (±15 % dans
les tests) ; elle sous-estime un bruit corrélé.

## 2. Bords et données incomplètes

### 2.1 Bords

Toutes les méthodes résolvent le problème sur un domaine périodique. Une
image réelle ne l'est pas : la discontinuité entre bords opposés produit
des oscillations sur toute l'image restaurée. ``boundary`` :

- **``"reflect"``** : l'image est prolongée d'une marge d'au moins deux
  tailles de PSF au total (taille rapide pour la FFT), placée après
  l'image. Dans la marge, le miroir du bord de fin se fond, par une
  pondération en cosinus surélevé, dans le miroir du bord de début : le
  domaine de travail est **continu et périodique**. La marge est traitée
  comme mesurée. Le résultat est recadré sur ``[:H, :W]`` ;
- **``"unknown"``** : même marge, mais traitée comme une **donnée
  inconnue** (§2.2) : seule l'image d'origine contraint la solution
  (Almeida & Figueiredo 2013). Le plus exact ;
- **``"perdecomp"``** : décomposition périodique + lisse de Moisan
  (eoqual-core) ; seule la composante périodique est déconvoluée, la
  composante lisse (basses fréquences, quasi insensible au flou) est
  rajoutée. Pas de marge, approximatif près des bords ; incompatible avec
  ``mask`` et ``saturation`` ;
- **``"periodic"``** : aucun traitement, exact pour une simulation par
  convolution circulaire.

#### Le mode ``"unknown"`` en détail

**Le problème.** Près d'un bord, chaque pixel observé mélange, par le
flou, des points de la scène situés *hors du champ* : la scène réelle se
poursuit au-delà de l'image, mais on n'en a aucune mesure. La FFT, elle,
suppose que l'image se répète périodiquement : le bord droit « touche »
le bord gauche. Le modèle ``y = h ∗ x`` est donc faux sur une bande de la
largeur de la PSF autour de l'image.

**L'idée.** On cherche une scène ``x`` **plus grande que l'image** —
l'image plus une marge — et on n'impose l'accord avec les données que là
où il y a des données (Almeida & Figueiredo 2013) :

```
            domaine de travail (périodique pour la FFT)
  ┌──────────────────────────────┬──────────┐
  │                              │          │
  │      image observée y        │  marge   │   x est inconnue partout ;
  │   H × W pixels : CONNUS      │ INCONNUE │   h ∗ x doit égaler y
  │  (attache ‖h ∗ x − y‖²)      │ (aucune  │   seulement sur l'image.
  │                              │ attache) │
  ├──────────────────────────────┘          │
  │              marge INCONNUE             │
  └─────────────────────────────────────────┘
```

``min_x ‖C(h ∗ x) − y‖² + R(x)``, où ``C`` ne garde que les pixels de
l'image. La marge de ``x`` n'est contrainte que par la régularisation
``R`` et par la continuité avec l'image : c'est une **extrapolation** de
la scène hors du champ, que l'on jette à la fin (recadrage sur
``[:H, :W]``). La FFT reste utilisable, puisque le domaine de travail
complet, lui, est traité comme périodique : la discontinuité au raccord
tombe dans une zone où rien n'est imposé.

**Mise en œuvre.** Plutôt qu'un opérateur de recadrage explicite, les
données de la marge sont **complétées à chaque itération par le flou de
l'estimée courante**, ``h ∗ x`` (§2.2) : le résidu y est nul, donc la
marge ne tire pas la solution. Pour les méthodes à gradient (Landweber,
FISTA, TV, Plug-and-Play), c'est exactement le gradient du problème
ci-dessus ; pour les méthodes linéaires, la boucle « compléter puis
résoudre » (``n_fill_iter = 30``) converge vers sa solution (EM).

**Différence avec ``"reflect"``.** ``"reflect"`` remplit la marge de
*données inventées* (le miroir de l'image) et demande à la solution de
les expliquer. Or un miroir n'est pas une image floue par la PSF — le
flou d'un bougé, par exemple, n'est pas symétrique : l'incohérence est
amplifiée par l'inversion, d'autant plus que la FTM a des zéros.
``"unknown"`` ne fait aucune hypothèse sur la scène hors champ.

| Flou de bougé, TV (TUTORIEL.md §7) | PSNR |
|---|---|
| ``"periodic"`` (rien) | 18.14 dB |
| ``"reflect"`` (marge inventée) | 27.67 dB |
| ``"unknown"`` (marge inconnue) | 29.74 dB |

**Coût.** Aucun pour les méthodes itératives ; ``n_fill_iter`` FFT
supplémentaires pour les méthodes linéaires (0.1 à 0.3 s pour 512×256).

**Quand le préférer.** Toujours acceptable ; indispensable dès que la
FTM s'annule (bougé, défocalisation forte). Exception mesurée :
Richardson-Lucy et Landweber, régularisés par le nombre d'itérations,
gardent ``"reflect"`` — les données de la marge y freinent utilement
l'amplification du bruit près des bords.

**Défaut par méthode**, d'après les mesures (BENCHMARK.md §4) :

| Méthodes | Défaut | Raison |
|---|---|---|
| ``wiener``, ``tikhonov``, ``fista_wavelet``, ``tv_primal_dual`` | ``"unknown"`` | +0.8 à +2.2 dB sur flou de bougé (OTF à zéros), au plus −0.06 dB ailleurs |
| ``inverse``, ``richardson_lucy``, ``landweber`` | ``"reflect"`` | ``"unknown"`` y est neutre (Richardson-Lucy : +0.07 dB) ou défavorable (inverse, Landweber : jusqu'à −0.5 dB) : régularisées par un seuil ou le nombre d'itérations, elles profitent des données synthétiques de la marge |
| ``pnp_hqs``, ``pnp_admm`` | ``"reflect"`` | marge déjà complétée (``fill_boundary``, §7) : identique à ``"unknown"`` |

Un miroir simple de chaque côté (``np.pad(..., "symmetric")``) laisse une
discontinuité au raccord périodique des deux marges. Son énergie haute
fréquence, inexpliquée par le flou, faussait les critères de réglage :
GCV et discrépance choisissaient λ ≈ 1e-10 (Tikhonov à −33 dB). Voir
l'annexe A.

### 2.2 Données incomplètes : pixels masqués, saturés, marge inconnue

``backends/_common.py``. Chaque pixel du domaine de travail est :

- **connu** : attache aux données ``(h ∗ x − y)²`` ;
- **inconnu** (``mask=False``, NaN, marge de ``"unknown"``) : aucune
  attache ; la valeur initiale est celle du plus proche pixel connu ;
- **saturé** (``saturation=L``, pixels valides ``≥ L``) : la vraie valeur
  floue est seulement connue pour être ``≥ L`` ; attache unilatérale
  ``min(h ∗ x − L, 0)²``.

Toutes les méthodes (sauf ``wiener_unsupervised`` et ``blind_l0``, qui
refusent ces options) passent par la même opération ``Problem.fill`` :
les données sont **complétées par l'estimée courante**, ``h ∗ x`` aux
pixels inconnus et ``max(h ∗ x, L)`` aux pixels saturés.

- **Méthodes à gradient** (Landweber, FISTA, TV) : ``Hᵀ(Hx − fill(Hx))``
  est exactement le gradient de l'attache masquée et unilatérale.
- **Méthodes linéaires** (inverse, Wiener, Tikhonov) : la boucle
  « compléter puis résoudre » (``n_fill_iter = 30``) est l'algorithme EM
  des données manquantes (Dempster, Laird & Rubin 1977) ; son point fixe
  vérifie ``Hᵀ M (Hx − y) + λLᵀL x = 0``, l'équation normale du problème
  masqué. Coût : ``n_fill_iter`` FFT supplémentaires (0.1 à 0.3 s pour
  512×256).
- **Richardson-Lucy** : ``x ← x · Hᵀ(fill(Hx) / Hx)``, l'EM du modèle de
  Poisson à données manquantes ; pour des bords inconnus, c'est la
  méthode de Bertero & Boccacci (2005).
- **Plug-and-Play** : remplissage à chaque itération (§7).
- Le principe de discrépance (Landweber, Tikhonov) compte les seuls
  pixels connus.

Mesures (flou gaussien σ = 2 px, bruit 0.01, image d'évaluation du
benchmark, BENCHMARK.md §4) :

- **une ligne sur 25 à 0** : sans masque, l'image est détruite (−56 à
  +15 dB selon la méthode) ; avec ``mask=``, à 0.15 dB au plus de la
  restauration sans défaut ;
- **flou écrêté à 85 % de son maximum** (6 % de pixels saturés) :
  +0.3 à +1.4 dB avec ``saturation=`` (TV : 28.19 → 29.59 dB), sans
  changement hors des zones saturées ;
- **flou de bougé, marge inconnue** : voir le tableau du §2.1.

## 3. Méthodes linéaires

Une FFT, aucune itération — sauf sur données incomplètes, où la boucle
de remplissage du §2.2 en enchaîne ``n_fill_iter``.

### 3.1 `inverse` — filtre inverse tronqué

``X = Y/H`` là où ``|H| ≥ threshold`` (défaut 0.15), 0 ailleurs. Sans
modèle de bruit ; amplifie le bruit d'un facteur ``1/threshold``.
Méthode de référence.

### 3.2 `wiener` — Wiener, spectre du signal ajusté

``X = H* Y / (|H|² + Sb/Sx)``.

- ``nsr`` donné : rapport constant (équivaut à Tikhonov ``identity``,
  vérifié par test) ;
- ``nsr=None`` (défaut) : ``Sx(ω) = A|ω|^-β``. Dans la bande où le flou
  est faible (``|H|² ≥ 0.1``), ``Ŝ = (|Y|² − Nσ²)/|H|²`` estime
  directement le spectre du signal ; sa moyenne par couronnes
  logarithmiques est ajustée en ``log A − β log|ω|`` (``β`` borné à
  [0.5, 4]). Le modèle prolonge ``Sx`` aux hautes fréquences, masquées
  par le flou. Le rapport bruit/signal croît donc avec la fréquence.

Résultat : à 0.3-0.4 dB du Wiener « oracle » (vrai spectre de la scène)
sur trois flous gaussiens. Avec β = 2 imposé (première version), l'écart
atteignait 3.4 dB.

### 3.3 `wiener_unsupervised` — Wiener non supervisé

Appel de ``skimage.restoration.unsupervised_wiener`` (Orieux et al.
2010) : niveaux de bruit et de régularisation laplacienne estimés par
échantillonnage de Gibbs ; ``sigma`` ignoré, graine fixée (``seed=0``).

### 3.4 `tikhonov`

``min ‖h ∗ x − y‖² + λ‖Lx‖²``, ``X = H*Y/(|H|² + λ|L|²)``, avec
``|L|²`` = 1 (``identity``), ``4sin²(πf_y) + 4sin²(πf_x)``
(``gradient``, différences avant) ou son carré (``laplacian``, défaut).
``lam=None`` : λ choisi par ``rule`` (§5).

## 4. Méthodes itératives

Le nombre d'itérations régularise (semi-convergence) : trop peu, l'image
reste floue ; trop, le bruit est amplifié.

### 4.1 `richardson_lucy`

``x ← x · Hᵀ(y / Hx)`` : maximum de vraisemblance pour un bruit de
Poisson, positivité préservée. Départ : image constante (moyenne).

- **Accélération** (défaut) : extrapolation vectorielle de Biggs &
  Andrews (1997), ordre 1 : ``z = x_k + α(x_k − x_{k−1})``,
  ``α = ⟨g_{k−1}, g_{k−2}⟩ / ‖g_{k−2}‖²`` borné à [0, 1], ``g`` = pas
  RL. Résultat comparable en 3 à 5 fois moins d'itérations.
- **TV** (``tv_weight``, Dey et al. 2006) : division par
  ``1 − λ div(∇x/|∇x|)`` (bornée à 0.1).
- Image négative (fond soustrait) : décalée de son minimum, puis recalée.
- Défaut ``n_iter = 50`` (accélérées) **avec** ``tv_weight = 0.005`` :
  la TV stabilise les itérations, si bien que le résultat dépend peu de
  leur nombre — meilleur PSNR moyen sur quatre flous et trois niveaux de
  bruit, +1.4 dB sur l'ancien défaut (12 itérations sans TV) et jamais
  pire (§9.1). Sans TV (``tv_weight=0``), RL pur exige d'arrêter tôt :
  12 à 20 itérations pour un flou gaussien σ = 2 px, 5 à 8 pour une tache
  d'Airy à Nyquist.

### 4.2 `landweber`

``x ← P₊(x − τHᵀ(Hx − y))``, ``τ = 1.9/max|H|²``. Arrêt par défaut au
**principe de discrépance** (``‖Hx − y‖² ≤ Nσ²``, plafond 500
itérations) : le nombre d'itérations s'adapte au flou, à 0.2-0.8 dB du
meilleur nombre fixe sur trois des quatre flous et au-dessus sur le
quatrième (Airy), alors qu'aucun nombre fixe ne convient aux quatre.

**Pas de Barzilai-Borwein** (``acceleration=True``, Barzilai & Borwein
1988) : ``τ_k = ‖s‖²/‖Hs‖²`` avec ``s = x_k − x_{k−1}`` — l'inverse du
quotient de Rayleigh de ``HᵀH`` dans la direction du dernier pas, exact
pour une attache quadratique — borné à ``[0.5, 50]/max|H|²``. Quand la
discrépance est atteignable, elle l'est en 3 à 4 fois moins d'itérations
pour le même PSNR (flou gaussien, σ = 0.002 : 63 → 23 itérations ;
σ = 0.0005 : 562 → 131). Sans projection ni recherche linéaire
non monotone, il n'est pas robuste quand le résidu plafonne au-dessus de
``Nσ²`` (bougé à très faible bruit : −2.6 dB) : désactivé par défaut.

## 5. Choix du paramètre des méthodes linéaires

``tuning/criteria.py``. Matrice d'influence diagonale
``S_λ = |H|²/(|H|² + λ|L|²)``, résidu ``(1/N)Σ|(1 − S_λ)Y|²``.

| Règle | Critère | Requiert σ | Comportement mesuré |
|---|---|---|---|
| ``gcv`` (défaut) | ``(1/N)‖(I − S)y‖² / (1 − tr S/N)²`` minimal | non | à 0.05 dB de l'optimum avec ``gradient``/``laplacian`` sur flou gaussien ; **sous-régularise de 2 dB avec ``identity``** ; sous-régularise sur flou de bougé (OTF à zéros) |
| ``discrepancy`` | résidu = ``Nσ²`` (bisection en log λ) | oui (estimé sinon) | optimal avec ``identity`` ; sur-régularise de 1 à 1.3 dB avec ``gradient``/``laplacian`` sur flou gaussien ; meilleur SSIM |
| L-curve | coin de ``(log‖résidu‖, log‖Lx‖)`` | non | diagnostic seulement (``l_curve``), moins stable |

GCV estime l'erreur de prédiction en supposant implicitement un signal
« blanc » ; une image naturelle concentre son énergie aux basses
fréquences, d'où la sous-régularisation avec ``L = I``. Les opérateurs
dérivatifs compensent en partie ce défaut. Recherche : balayage
logarithmique de 1e-10 à 1e3 (66 points) puis recherche bornée autour du
minimum. ``plotting.plot_gcv`` trace les deux critères.

## 6. Méthodes variationnelles

### 6.1 `fista_wavelet`

``min ½‖Hx − y‖² + λ‖Wx‖₁``, ``W`` ondelettes orthogonales
(PyWavelets, périodisation, coefficients de détail seulement), FISTA
(Beck & Teboulle 2009) : seuillage doux à ``λσ/max|H|²``. Les dimensions
de travail sont des multiples de ``2^levels`` (marge ``reflect``).
Défauts ``lam = 0.15`` (× σ), ``levels = 5`` (§9.1).

**Cycle spinning** (défaut) : la grille d'ondelettes est décalée
aléatoirement à chaque itération (Figueiredo & Nowak 2003), ce qui
supprime les artefacts en blocs d'une base décimée. L'objectif n'est
alors plus strictement décroissant (problème invariant par translation
en moyenne).

### 6.2 `tv_primal_dual`

``min ½‖Hx − y‖² + λTV(x) [+ ι(x ≥ 0)]``, TV isotrope (différences
avant périodiques), algorithme de Condat-Vũ avec le terme d'attache aux
données traité par son gradient :

```
x ← P(x − τ(Hᵀ(Hx − y) − div p))
p ← Proj_{‖·‖₂ ≤ λ}(p + σ_d ∇(2x_new − x))
```

Pas ``τ = 0.99/‖H‖²``, ``σ_d = ‖H‖²/16`` : la condition
``1/τ − 8σ_d ≥ ‖H‖²/2`` est satisfaite (``‖∇‖² ≤ 8``). Arrêt à
``‖Δx‖/‖x‖ < 1e-5`` ou ``n_iter = 300``. ``lam = 0.1`` (× σ) par défaut : médiane des optimums du benchmark (0.04 à 0.17).
La décroissance de l'objectif est vérifiée par test, et l'optimum par une
solution de référence indépendante : sur une image 24×24, la valeur de la
fonctionnelle atteinte égale celle de CVXPY (solveur Clarabel) à 1e-6
près, et la solution en diffère de moins de 5e-3
(``tests/test_reference_cvxpy.py``, Diamond & Boyd 2016).

## 7. Plug-and-Play

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
  jusqu'à +6 dB sur flou de bougé (§9.1).
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
  données sont complétées par le flou de l'estimée courante ``z`` (§2.2) :
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

## 8. Déconvolution aveugle

``backends/blind.py`` : ``blind_deconvolve(image, psf_size, sigma)``
renvoie ``(image, psf)`` ; ``blind_l0`` en est la forme à signature
commune (image seulement). **Expérimentale** : toujours examiner la PSF
estimée (``plotting.plot_psf``).

**Estimation de la PSF** (a priori L0 sur les gradients, Xu et al. 2013 ;
Pan et al. 2014), du grossier au fin (facteur √2, jusqu'à un support de
3 à 5 pixels) :

1. **pas en x** : ``min ‖h ∗ x − y‖² + λ‖∇x‖₀`` par éclatement
   semi-quadratique (Xu et al. 2011) : seuillage dur de ``∇x`` à
   ``√(λ/β)`` puis solution exacte en Fourier, ``β`` de 2λ à 1e5 par
   doublements. **Départ : l'image observée** — un départ par filtre
   inverse amplifie un bruit que le seuillage conserve ensuite comme
   contours ;
2. **pas en h** : ``min ‖h ∗ ∇x − ∇y‖² + γ‖h‖²`` (domaine des gradients,
   spectre presque plat), solution exacte en Fourier, ``γ = 1e-4·max|∇X|²``,
   restreinte au support, valeurs négatives et < 5 % du maximum annulées,
   somme 1, barycentre recentré ;
3. ``λ`` part de ``lam_l0 = 4e-3`` (image en [0, 1], valeur de Pan et
   al.) et décroît d'un facteur 0.9 par itération (plancher 1e-4) ;
   6 alternances par échelle.

**Restauration finale** : ``tv_primal_dual`` avec la PSF estimée
(``lam = 0.4``).

**Limites.** Gain de +0.5 à +4.2 dB selon la scène et le flou (contre
+4.5 à +6.9 dB pour la TV à PSF connue), variable d'une scène et d'un tirage à l'autre. La PSF n'est
définie qu'à une translation près (recentrée sur son barycentre) : le
résultat peut être décalé d'une fraction de pixel. Choisir ``psf_size``
proche de l'étendue attendue du flou : un support trop grand dégrade
fortement le résultat (sur l'image du benchmark, 21 px au lieu de 9 à
17 : −1 à −2 dB au lieu de +3 à +4 dB), un support un peu trop petit est
bien toléré.

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

### 9.1 Choix des réglages par défaut (0.3.0)

Un défaut doit valoir pour tous les cas, pas pour un seul : il est choisi
par **PSNR moyen sur 12 cas** — quatre flous (gaussien σ = 2 px, Airy à
Nyquist, défocalisation r = 3 px, bougé 9 px) × trois bruits
(σ = 0.005, 0.01, 0.02) — sur l'image de **réglage** (moitié gauche
d'« astronaut » ; 256×256 pour le Plug-and-Play et l'aveugle), par une
grille sur les paramètres principaux. Le choix est ensuite **vérifié
sur l'image d'évaluation** (moitié droite, autres tirages de bruit) ;
un réglage qui ne s'y confirme pas est rejeté.

| Méthode | Ancien défaut | Nouveau défaut | Gain moyen (évaluation) | Pire cas |
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
- ``combine_psfs`` : convolution de PSF (chaîne optique ⊛ détecteur ⊛
  bougé) ; les variances s'ajoutent.
- ``simulate.py`` : ``blur`` (circulaire ou miroir), ``add_noise``
  (gaussien, Poisson par ``gain``), ``simulate_observation``.
- ``plotting.py`` (Plotly via eoqual-viz) : ``plot_restoration``
  (galerie avec PSNR), ``plot_psf``, ``plot_mtf`` (coupe de ``|H|``
  jusqu'à Nyquist), ``plot_convergence`` avec ``ConvergenceRecorder``
  (rappel ``callback`` des méthodes itératives), ``plot_gcv``,
  ``plot_optimization``.

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
| ``test_pnp.py`` | débruiteur fourni, ``tv_chambolle``, ``nl_ridge`` ; ``drunet`` (marqueur ``deep``) |
| ``test_blind.py`` | PSF estimée plus proche de la vraie que le Dirac, gain positif |
| ``test_plotting.py``, ``test_examples.py`` | figures, exemples (dont le tutoriel) |

## 12. Évolutions envisagées

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
- **DPIR** (débruiteur DRUNet et calendrier dédiés) et **calcul sur GPU**
  (PyTorch) des méthodes itératives.
- **Bruit de Poisson** : déconvolution Poisson + TV par ADMM (PIDAL,
  Figueiredo & Bioucas-Dias 2010) pour les images à faible flux ;
  stabilisation de variance avant les méthodes à bruit gaussien (comme
  ``vst_denoise`` d'eoqual-denoising). *Pourquoi pas encore* : le calcul
  est léger, mais l'interface ne l'est pas — l'attache de Poisson exige
  des données en photo-électrons (gain, offset, bruit de lecture du
  capteur, comme la PTC d'eoqual-measure), un modèle poisson-gaussien
  pour être réaliste, et un benchmark à faible flux dédié ;
  ``richardson_lucy(tv_weight=...)`` couvre déjà le cas Poisson + TV.
  Candidat naturel pour la prochaine version.
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
- PSF variable dans le champ (déconvolution par tuiles, eoqual-core
  ``tiling``).
- Aveugle : sélection des contours saillants (Cho & Lee 2009), a priori
  sur les intensités (Pan et al. 2014), pixels saturés (étoiles, reflets).

---

## Annexe A — Historique des choix et corrections

Faits pendant le développement de la 0.1.0 (2026-10-29), mesures sur
scènes synthétiques.

1. **Marge miroir non périodique.** Version initiale :
   ``np.pad(..., "symmetric")`` de chaque côté. Le raccord périodique des
   deux marges est discontinu ; GCV et discrépance choisissaient
   λ ≈ 1e-10 (Tikhonov −33 dB). Remplacée par la marge continue et
   périodique du §2 : Tikhonov 27.6 dB (optimum 27.6 dB), filtre inverse
   +3.4 dB.
2. **Wiener : β ajusté.** β = 2 imposé et amplitude par bilan d'énergie
   global : jusqu'à 3.4 dB sous l'oracle. Ajustement de ``A`` et ``β`` dans
   la bande fiable : 0.3-0.4 dB sous l'oracle.
3. **Défauts.** RL : 30 → 12 itérations accélérées (30 dégradait les
   flous étroits de 4 à 7 dB). Landweber : arrêt par discrépance par
   défaut. `tv_primal_dual` λ : 0.4 → 0.1 et `fista_wavelet` λ : 0.6 →
   0.2 (× σ), d'après les optimums du benchmark (+0.5 à +1.1 dB).
4. **Plug-and-Play.** (a) ADMM : remise à l'échelle de la variable duale
   réduite quand ρ change (sans elle, divergence : 18 dB) ; λ = 0.5.
   (b) Marge traitée comme donnée inconnue (§7) : +5 dB sur flou de
   bougé.
5. **Déconvolution aveugle.**
   - PALM sur ``min ‖h ∗ x − y‖² + λTV(x)`` (gradient projeté sur le
     simplexe pour ``h``) : pas en ``h`` figé par la constante de
     Lipschitz des basses fréquences (spectre en 1/f²), puis, en domaine
     des gradients avec FISTA, convergence vers le Dirac (λ faible) ou
     vers une PSF décalée (λ fort). **Écarté.**
   - A priori L0 (Xu et al., Pan et al.) : première version partant d'un
     filtre inverse (PSNR −8 dB du pas en x) et ``γ = 1e-3`` (trop fort) ;
     corrigé (départ = image observée, ``γ = 1e-4``) : +0.6 à +3.2 dB sur
     six cas.
   - Préfiltre anti-repliement de la pyramide : essayé, écarté (le flou du
     préfiltre passe dans la PSF estimée).
6. **Mises en commun.** ``psf2otf`` et convolution circulaire (repris
   d'eoqual-measure), estimation du bruit (d'eoqual-denoising, en numpy
   seul) et profils de PSF intégrés sur le pixel (d'eoqual-phaseretrieval)
   déplacés dans eoqual-core 0.2.0 ; galerie d'images et courbes
   multiples ajoutées à eoqual-viz 0.2.0. ``gaussian_kernel``
   (eoqual-measure) laissé sur place : un seul utilisateur.
7. **Données incomplètes (0.3.0).** Masque de pixels valides, saturation
   et bords inconnus traités par une seule opération de remplissage
   (§2.2), plutôt que trois mécanismes. L'attache unilatérale pour la
   saturation vient d'un prototype personnel qui l'optimisait par L-BFGS ;
   le remplissage la rend compatible avec toutes les méthodes, y compris
   les linéaires (EM). Un masque des pixels saturés seul (sans contrainte
   ``≥ L``) est aussi possible via ``mask=`` ; la contrainte unilatérale
   récupère davantage d'information. Défaut ``boundary="unknown"`` pour
   les quatre méthodes qui en profitent (§2.1).
8. **Pas de Barzilai-Borwein pour Landweber (0.3.0)**, d'après l'idée de
   BBii-Decon (gradient projeté de Barzilai-Borwein) : implémenté depuis
   Barzilai & Borwein (1988), optionnel (§4.2).
9. **PSF d'instrument (0.3.0)** : ``system_psf`` d'après un prototype
   personnel de classes optiques (WFE → PSF → FTM) ; rayon d'Airy en
   pixels déplacé dans eoqual-core 0.3.0 (``airy_radius_px``, aussi
   présent dans eoqual-phaseretrieval).
