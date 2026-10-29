# Contribuer à eoqual-deconvolution

Les contributions sont bienvenues, sous toutes leurs formes : signalement
de bug, correction, proposition de nouvelle méthode, implémentation d'une
méthode déjà proposée, amélioration de la documentation.

## Signaler un bug

Ouvrez une issue avec :
- la méthode concernée (`list_methods()` pour voir le registre) ;
- un exemple minimal reproduisant le problème (image ou scène
  synthétique, PSF, `sigma`, code) ;
- le résultat obtenu vs. attendu.

## Proposer une méthode

Ouvrez une issue décrivant la méthode (nom, référence bibliographique,
ce qu'elle apporte par rapport au catalogue — voir
`REFERENCE_TECHNIQUE.md`). Pas besoin de savoir l'implémenter pour la
proposer.

## Implémenter une méthode

Convention du projet : une méthode = une fonction
`<méthode>_deconvolve(image, psf, sigma=None, *, ..., boundary="reflect") -> np.ndarray`
dans `src/eoqual/deconvolution/backends/`, qui prépare le problème par
`backends._common.setup` (contrôles, bords, opérateur de flou), exprime
`sigma` dans les unités de l'image, donne ses paramètres sans dimension
ou en multiples de σ, et n'a **pas de paramètre nommé `method`**
(réservé au dispatcher). Elle est déclarée dans
`config.py::METHODS_CONFIGS` et `runner.py::_DISPATCH`, sa référence est
ajoutée à `BIBLIOGRAPHY.md`, et ses paramètres réglables reçoivent un
espace de recherche `tuning/spaces/<méthode>.jsonc`.

Une brique qui sert à deux modules EOQual au moins va dans eoqual-core
(numpy et scipy seulement) ; une figure générique va dans eoqual-viz
(Plotly).

**Licence — condition de fusion, pas une formalité.**

- **Implémentation depuis une publication** : bienvenue sous MIT.
  L'en-tête du module cite les publications utilisées ; aucun code tiers
  n'est consulté ni repris.
- **Reprise d'un code publié ailleurs** : seulement si sa licence
  vérifiée est permissive, avec conservation de sa notice. Un code sans
  licence n'est pas acceptable.
- **Bibliothèque sous licence restrictive ou dépendance lourde** :
  seulement en extra pip optionnel, jamais importée à l'import du
  paquet, avec un avertissement dans `THIRD_PARTY_LICENSES.md`.

## Tests

Les tests ne s'appuient que sur des données synthétiques (flou et bruit
connus) :

- `tests/test_methods.py` couvre le contrat d'appel, le gain de PSNR,
  l'invariance d'échelle et les cas limites ; ajouter la nouvelle méthode
  à `MIN_GAIN` ;
- une méthode en extra a son propre fichier, avec `pytest.importorskip` ;
- les tests longs portent le marqueur `slow` (`pytest -m "not slow"`
  pour une passe rapide), ceux de l'extra `deep` le marqueur `deep`.

Mettre à jour `BENCHMARK.md` avec `examples/benchmark.py --optimize`.

## Documentation

Toute méthode nouvelle ou modifiée doit rester reflétée dans `README.md`
(catalogue), `REFERENCE_TECHNIQUE.md` (principe, implémentation,
limites), `THIRD_PARTY_LICENSES.md` (licence) et `BIBLIOGRAPHY.md`
(référence).
