# eoqual-deconvolution

13 méthodes de déconvolution d'images à PSF connue ou estimée : linéaires (Wiener à spectre ajusté, Wiener–Hunt sans réglage, Tikhonov avec choix automatique du paramètre), itératives (Richardson-Lucy accéléré, Landweber), variationnelles (variation totale, parcimonie en ondelettes), Plug-and-Play sur les débruiteurs d'eoqual-denoising, réseau déplié USRNet fourni par eoqual-neural, déconvolution aveugle libre ou contrainte par le modèle optique de l'instrument (défocalisation, aberrations, variation dans le champ) ; traitement des bords, optimisation bayésienne des paramètres et benchmark — **en cours de développement**.

- [Documents techniques](technique/index.md) : principe, mise en œuvre,
  validation et limites des méthodes.
- [Notes de version](changelog.md) : ce qui change d'une version à l'autre.
- [Référence de l'API](api/index) : générée à partir du code source.

```{toctree}
:hidden:
:maxdepth: 2

technique/index
changelog
api/index
```
