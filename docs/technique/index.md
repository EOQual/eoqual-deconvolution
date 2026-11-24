# Documents techniques

Principe, implémentation, paramètres et limites de chaque méthode. Les
chapitres gardent une numérotation unique d'un document à l'autre : un
renvoi « §n » désigne toujours le même chapitre. Ces documents décrivent
l'état présent du code ; ce qui change d'une version à l'autre est dans les
[notes de version](../changelog.md).

| Document | Chapitres | Contenu |
|---|---|---|
| [Modèle, bords et données incomplètes](modele-bords.md) | 1-2 | Modèle et notations, traitement des bords, pixels masqués ou saturés, marge inconnue |
| [Méthodes linéaires, itératives et variationnelles](methodes-classiques.md) | 3-6 | Inverse, Wiener, Wiener–Hunt, Tikhonov, Richardson-Lucy, Landweber, choix du paramètre, ondelettes, variation totale |
| [Plug-and-Play et réseaux dépliés](plug-and-play.md) | 7 | Plug-and-Play sur débruiteurs, USRNet, réentraînement |
| [Déconvolution aveugle](aveugle.md) | 8 | PSF libre (a priori L0), aveugle contrainte par le modèle optique de l'instrument |
| [Optimisation des paramètres](optimisation.md) | 9 | Optimisation bayésienne, choix des réglages par défaut |
| [PSF, simulation et tests](psf-simulation.md) | 10-11 | PSF d'instrument, jitter et filé, simulation, figures, tests |
| [Pistes et variantes écartées](pistes.md) | 12-13 | Méthodes classiques et apprises envisagées ; variantes écartées et pourquoi |
| [Tutoriel](tutoriel.md) | | Prise en main pas à pas |
| [Benchmark](benchmark.md) | | Comparaison chiffrée des méthodes sur cas simulés |
| [Bibliographie](bibliographie.md) | | Références des méthodes |

```{toctree}
:hidden:
:maxdepth: 2

modele-bords
methodes-classiques
plug-and-play
aveugle
optimisation
psf-simulation
pistes
tutoriel
benchmark
bibliographie
```
