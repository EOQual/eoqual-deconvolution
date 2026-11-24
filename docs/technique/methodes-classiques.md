# Méthodes linéaires, itératives et variationnelles

## 3. Méthodes linéaires

Une FFT, aucune itération — sauf sur données incomplètes, où la boucle
de remplissage du [§2.2](modele-bords.md#22-données-incomplètes--pixels-masqués-saturés-marge-inconnue) en enchaîne ``n_fill_iter``.

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
sur trois flous gaussiens. Avec β = 2 imposé, l'écart atteint 3.4 dB
([§13](pistes.md#13-variantes-écartées)).

### 3.3 `wiener_unsupervised` — Wiener non supervisé

Appel de ``skimage.restoration.unsupervised_wiener`` (Orieux et al.
2010) : niveaux de bruit et de régularisation laplacienne estimés par
échantillonnage de Gibbs ; ``sigma`` ignoré, graine fixée (``seed=0``).

### 3.4 `tikhonov`

``min ‖h ∗ x − y‖² + λ‖Lx‖²``, ``X = H*Y/(|H|² + λ|L|²)``, avec
``|L|²`` = 1 (``identity``), ``4sin²(πf_y) + 4sin²(πf_x)``
(``gradient``, différences avant) ou son carré (``laplacian``, défaut).
``lam=None`` : λ choisi par ``rule`` (§5).

### 3.5 `wiener_hunt` — Wiener–Hunt, λ et σ par vraisemblance marginale

**Origine.** Le filtre de Wiener (1949) suppose connus les spectres de
la scène ``Sx`` et du bruit ``Sb`` : ``X = H* Y / (|H|² + Sb/Sx)``. Ni
l'un ni l'autre ne l'est en pratique. Hunt (1973) remplace le spectre de
la scène par une **contrainte de régularité** : parmi les images
compatibles avec les données au niveau du bruit
(``‖y − h ∗ x‖² = Nσ²``), prendre la plus lisse au sens de ``‖Dx‖²``,
``D`` laplacien (moindres carrés contraints). Par multiplicateur de
Lagrange :

    X = H* Y / (|H|² + λ |D|²),   |D(ω)|² = (4sin²(πf_y) + 4sin²(πf_x))²

C'est un filtre de Wiener dont le rapport signal/bruit est imposé :
``Sx/Sb = 1/(λ|D|²)``, soit un spectre de scène en ``|ω|⁻⁴`` aux basses
fréquences.

**Lecture bayésienne** (Orieux, Giovannelli & Rodet 2010). A priori
gaussien ``p(x) ∝ exp(−γ_x ‖Dx‖²/2)``, bruit blanc gaussien
``p(y | x) ∝ exp(−γ_b ‖y − h ∗ x‖²/2)`` : la loi a posteriori est
gaussienne, son maximum (et sa moyenne) est le filtre ci-dessus avec
``λ = γ_x / γ_b``, et ``σ² = 1/γ_b``. Le filtre est donc celui de
``tikhonov`` avec ``regularizer="laplacian"`` (§3.4, vérifié à 1e-10
près par test) ; **ce qui définit la méthode est le choix de λ et σ**.

**Choix de λ et σ : vraisemblance marginale** (« evidence » : MacKay
1992 ; Molina 1994). La scène s'intègre analytiquement :
``y ~ N(0, γ_x⁻¹ H (DᵀD)⁻¹ Hᵀ + γ_b⁻¹ I)``, matrice circulante,
diagonale en Fourier. Chaque coefficient ``Y(ω)`` est gaussien
complexe, indépendant, de variance

    E|Y(ω)|²/N = v(ω) = σ² · s(ω),   s(ω) = 1 + |H(ω)|² / (λ |D(ω)|²)

et, à des constantes près, avec ``P(ω) = |Y(ω)|²/N`` :

    −log p(y | λ, σ²) = Σ_ω [ log v(ω) + P(ω)/v(ω) ]

σ² se profile analytiquement (``σ̂² = moyenne_ω(P/s)``), ce qui laisse
un critère à une dimension, ``J(λ) = Σ log s + M log σ̂²(λ)`` (``M``
fréquences), minimisé par balayage logarithmique puis recherche bornée
(``tuning.criteria.wiener_hunt_estimate``). La fréquence nulle est
exclue : ``D(0) = 0``, l'a priori ne dit rien de la moyenne (impropre).
Si ``sigma`` est donné, seul λ est estimé.

Intuition : aux hautes fréquences, où ``|H|²`` est petit, ``s ≈ 1`` et
``P`` mesure directement le bruit ; aux basses fréquences, ``P`` mesure
le signal filtré, et λ règle la transition. Le critère pénalise à la
fois un modèle trop peu régularisé (variance prédite trop grande, terme
``log v``) et trop régularisé (énergie observée inexpliquée, terme
``P/v``) — sans référence ni connaissance du bruit.

**Liens avec les autres méthodes linéaires.**

| Méthode | Spectre de la scène | Niveau (λ, σ) |
|---|---|---|
| ``wiener`` | ``A|ω|^-β``, A et β ajustés par régression dans la bande peu floue | σ estimé (ondelettes) |
| ``wiener_hunt`` | ``∝ 1/|D|²`` (laplacien : ``|ω|⁻⁴``) | λ et σ par vraisemblance marginale |
| ``tikhonov`` | même filtre que ``wiener_hunt`` | λ par GCV, discrépance, ou ``rule="evidence"`` (même critère) |
| ``wiener_unsupervised`` | même modèle que ``wiener_hunt`` | λ et σ par échantillonnage de Gibbs (moyenne a posteriori), scikit-image |

``wiener_hunt`` est la version **déterministe** du Wiener–Hunt non
supervisé : maximum de la vraisemblance marginale (bayésien empirique)
au lieu de l'échantillonnage de la loi a posteriori des hyperparamètres.
Instantané (une FFT et ~80 évaluations d'une somme), et il hérite de
l'infrastructure commune : marge inconnue (``boundary="unknown"``,
défaut), pixels masqués et saturés ([§2.2](modele-bords.md#22-données-incomplètes--pixels-masqués-saturés-marge-inconnue)), ce que ``wiener_unsupervised``
ne gère pas.

**Mesures** ([benchmark](benchmark.md), 24 cas : deux scènes, quatre flous — gaussien,
Airy, bougé, PSF optique aberrée —, trois niveaux de bruit) : PSNR moyen
30.70 dB, contre 30.67 (``tikhonov``, GCV), 30.64 (``wiener``) et 29.55
(``wiener_unsupervised``) ; écart à ``tikhonov`` de −0.68 à +0.38 dB, gain
sur ``wiener_unsupervised`` de +0.03 à +8.8 dB (bougé à faible bruit :
22.4 dB pour ``wiener_unsupervised`` contre 31.1 dB). σ estimé à 3 % près (médiane), au pire 50 %
trop grand pour un bougé (FTO nulle par endroits) à σ = 0.003. L'a priori
``gradient`` (spectre en ``|ω|⁻²``) perd 0.8 dB en moyenne : Hunt avait
raison de prendre le laplacien pour fixer λ ; on verra au [§8.2](aveugle.md#82-blind_optics--aveugle-contrainte-par-loptique) qu'un
exposant intermédiaire (β = 3) est préférable pour **estimer la PSF**.

``tuning.criteria.evidence_curve`` trace le critère en fonction de λ ;
``backends.linear.wiener_hunt_parameters`` renvoie λ, σ et le critère.

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
  bruit, +1.4 dB sur 12 itérations sans TV et jamais
  pire ([§9.1](optimisation.md#91-choix-des-réglages-par-défaut)). Sans TV (``tv_weight=0``), RL pur exige d'arrêter tôt :
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
| ``evidence`` | vraisemblance marginale de Wiener–Hunt maximale (§3.5) | non (estimé avec λ ; utilisé s'il est donné) | équivalent à GCV en moyenne (+0.02 dB sur 24 cas avec ``laplacian``), donne σ en plus ; défaut de ``wiener_hunt`` |
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
Défauts ``lam = 0.15`` (× σ), ``levels = 5`` ([§9.1](optimisation.md#91-choix-des-réglages-par-défaut)).

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
