# Change Log

Versions publiées et contenu de chacune. L'historique détaillé des choix
de conception et des corrections faites pendant le développement est
dans `REFERENCE_TECHNIQUE.md`, annexe A.

## Version History

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
