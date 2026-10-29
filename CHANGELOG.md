# Change Log

Versions publiées et contenu de chacune. L'historique détaillé des choix
de conception et des corrections faites pendant le développement est
dans `REFERENCE_TECHNIQUE.md`, annexe A.

## Version History

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
