# Change Log

Versions publiées et contenu de chacune. L'historique détaillé des choix
de conception et des corrections faites pendant le développement est
dans `REFERENCE_TECHNIQUE.md`, annexe A.

## Version History

0.4.0 (2026-11-08)

:   -   `wiener_hunt` (nouvelle méthode linéaire) : déconvolution de
        Wiener–Hunt (Hunt 1973), filtre `H*Y/(|H|² + λ|D|²)`, λ et σ
        estimés par **vraisemblance marginale** (la scène intégrée
        analytiquement en Fourier), sans réglage ni connaissance du
        bruit ; version déterministe du Wiener–Hunt non supervisé
        d'Orieux et al. (2010). Marge inconnue, pixels masqués et saturés
        gérés. 24 cas : au niveau de `tikhonov` (GCV, +0.02 dB en
        moyenne), +1.15 dB sur `wiener_unsupervised` (jusqu'à +8.8 dB),
        σ à 3 % près (REFERENCE_TECHNIQUE.md §3.5, BENCHMARK.md §6).
        `wiener_hunt_parameters` renvoie λ, σ et le critère.
    -   `tikhonov(rule="evidence")` et `tuning.criteria` :
        `wiener_hunt_evidence`, `wiener_hunt_estimate`, `evidence_curve`.
    -   `blind_optics` (nouvelle méthode aveugle) et
        `backends.optics_blind` : **déconvolution aveugle contrainte par
        l'optique** — la PSF est celle d'un modèle physique de
        l'instrument (`OpticalModel` : échantillonnage Q, obturation,
        paramètres connus fixés, défocalisation / aberrations de
        Zernike / jitter estimés dans des bornes plausibles), estimée par
        vraisemblance marginale (`estimator="evidence"`, a priori de
        scène en |ω|^-3) ou par les contours (`estimator="l0"`) ;
        variation dans le champ par tuiles avec modèle de champ
        (constant, plan, quadratique) et recollement de Hann. 30 cas :
        1.2 dB sous la vraie PSF en moyenne, contre 3.1 dB pour
        `blind_l0` (REFERENCE_TECHNIQUE.md §8.2, BENCHMARK.md §7) ; limites
        d'identifiabilité documentées (signe, aberrations impaires, flou
        isotrope doux, anisotropie de la scène).
    -   `psf.optical_psf` : PSF de diffraction d'une pupille annulaire
        aberrée (Zernike de Noll, ondes RMS), jitter, intégration sur le
        pixel, bande spectrale ; identique à l'Airy analytique à 0.4 %
        près ; FTM toujours sous la diffraction. `psf.defocus_to_zernike`.
    -   Base de Zernike et PSF de phase prises dans eoqual-core 0.5.0
        (`eoqual.core.pupil`, déplacées depuis eoqual-phaseretrieval).
    -   Exemples `blind_optics.py`, `blind_optics_benchmark.py`,
        `wiener_hunt_benchmark.py` ; figure `resources/optical_psfs.png`
        (`examples/figures.py`) ; tutoriel §3 et §7.
    -   `REFERENCE_TECHNIQUE.md` §10.1 : jitter et filé, deux mouvements
        différents (FTM gaussienne contre sinus cardinal directionnel,
        équivalence aux basses fréquences σ = L/√12, cas du push-broom
        et du TDI).
    -   Dépendances : eoqual-core `v0.5.0`, eoqual-metrics `v1.2.3` (même
        core, pour que poetry résolve).

0.3.1 (2026-11-04)

:   -   `REFERENCE_TECHNIQUE.md` §2.1 : le schéma des modes de bords
        (`reflect`, `unknown`) devient une figure PNG
        (`resources/boundary_modes.png`), régénérable par
        `examples/figures.py`, au lieu d'un dessin en caractères.
    -   Dépend d'eoqual-core 0.4.0 (`psf.airy_radius_px`,
        `psf.supersampled_coverage`). `psf._supersampled` (interne,
        `disk_psf`/`motion_psf`) délègue maintenant à
        `eoqual.core.psf.supersampled_coverage` plutôt que de dupliquer
        le même calcul — refactor pur, comportement et sortie numérique
        inchangés (111 tests, aucune régression).
    -   **Correctif CI** : `eoqual-metrics` relevé à 1.2.2 — l'ancien
        pin (1.2.1) épinglait lui-même `eoqual-core` à 0.1.0, en
        conflit avec le pin direct 0.4.0 (`SolverProblemError` de
        poetry : deux tags d'un même paquet git ne peuvent pas
        coexister). eoqual-metrics 1.2.2 relève ce même pin à 0.4.0.

0.3.0 (2026-11-02)

:   Version intermédiaire 0.2.0 non publiée : son contenu est intégré ici.

    -   **Réglages par défaut optimisés** (PSNR moyen sur quatre flous ×
        trois niveaux de bruit, vérifié sur une image distincte) :
        `inverse` seuil 0.15 (+4.6 dB), `richardson_lucy` 50 itérations
        avec `tv_weight=0.005` (+1.4 dB), `fista_wavelet` λ 0.15 et
        5 niveaux (+0.2 dB), `pnp_hqs` `sigma_start=0.05` (+1.2 dB),
        `pnp_admm` `sigma_start=0.1` (+0.4 dB) ; REFERENCE_TECHNIQUE.md §9.1.
    -   **Données incomplètes** pour toutes les méthodes sauf
        `wiener_unsupervised` et `blind_l0` : `mask=` (pixels masqués,
        NaN), `saturation=` (attache unilatérale), et
        `boundary="unknown"` (marge inconnue, Almeida & Figueiredo 2013),
        par un même remplissage des données (EM pour les méthodes
        linéaires et Richardson-Lucy, gradient exact pour les autres).
    -   `boundary="unknown"` par défaut pour `wiener`, `tikhonov`,
        `fista_wavelet` et `tv_primal_dual` (+0.8 à +2.2 dB sur flou de
        bougé).
    -   `landweber(acceleration=True)` : pas de Barzilai-Borwein.
    -   `psf.system_psf` (diffraction d'une pupille annulaire,
        défocalisation, filé, jitter, intégration sur le pixel),
        `psf.annular_airy`, `psf.combine_psfs`.
    -   `TUTORIEL.md` et `examples/tutorial.py` ; bibliographie enrichie
        (« Pour aller plus loin ») ; `BENCHMARK.md` §4 (données
        incomplètes).
    -   Test de référence par CVXPY (dépendance de développement).
    -   Dépend d'eoqual-core 0.3.0 (`psf.airy_radius_px`) et
        d'eoqual-metrics 1.2.1.

0.1.0 (2026-10-29)

:   -   **11 méthodes de déconvolution**, appel unifié
        `deconvolve(image, psf, method, **kwargs)` (signature commune
        `f(image, psf, sigma=None, ...)`, garantie par test), images
        multi-bandes par `channel_axis` :
        -   linéaires : `inverse`, `wiener` (spectre du signal ajusté sur
            l'image), `wiener_unsupervised` (scikit-image), `tikhonov`
            (identité, gradient, laplacien ; λ par GCV ou discrépance) ;
        -   itératives : `richardson_lucy` (accélération Biggs-Andrews,
            option TV), `landweber` (arrêt par discrépance) ;
        -   variationnelles : `fista_wavelet`, `tv_primal_dual`
            (Condat-Vũ) ;
        -   Plug-and-Play (extras `pnp`, `deep`) : `pnp_hqs`, `pnp_admm`,
            débruiteurs d'eoqual-denoising ou fonction fournie ;
        -   aveugle (expérimentale) : `blind_l0` / `blind_deconvolve`.
    -   **Bords** : marge miroir continue et périodique, décomposition
        périodique + lisse, ou périodique.
    -   **Réglage** : GCV, discrépance, L-curve ; optimisation bayésienne
        (extra `optim`) avec référence (PSNR, SSIM) ou sans (indice SI),
        métriques d'eoqual-metrics, espaces de recherche JSONC.
    -   **PSF et simulation** : gaussienne, Airy, Moffat (intégrées sur le
        pixel), défocalisation, bougé, recentrage ; flou et bruit
        gaussien ou poissonien.
    -   **Figures** Plotly via eoqual-viz.
    -   **Benchmark** (`BENCHMARK.md`) : quatre flous, réglages par défaut
        et optimisés, séparation réglage / évaluation.
    -   Dépend d'eoqual-core 0.2.0, eoqual-metrics 1.2.1, eoqual-viz 0.2.0
        (et eoqual-denoising 0.1.0 pour le Plug-and-Play).
