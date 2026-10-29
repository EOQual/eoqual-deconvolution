#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Espaces de recherche des paramètres de chaque méthode.

Un espace de recherche est un dictionnaire ::

    {
      "search_space": [
        {"name": "lam", "type": "float", "low": 0.02, "high": 3.0, "prior": "log-uniform"},
        {"name": "n_iter", "type": "int", "low": 5, "high": 500},
        {"name": "wavelet", "type": "categorical", "categories": ["db2", "db4"]},
      ],
      "initial_guess": {"lam": 0.4, "n_iter": 100, "wavelet": "db4"},
      "fixed": {"boundary": "reflect"},      # optionnel
    }

- ``type`` : ``"int"``, ``"float"`` ou ``"categorical"`` ;
- ``prior`` (optionnel) : ``"uniform"`` (défaut) ou ``"log-uniform"``
  (paramètre qui varie sur plusieurs décades) ;
- la dimension spéciale ``sigma_scale`` multiplie le σ transmis à la
  méthode (σ mal connu) ;
- ``initial_guess`` : point de départ, en général les réglages par défaut.

Les espaces par défaut sont les fichiers ``spaces/<méthode>.jsonc``,
commentés dimension par dimension ; :func:`load_search_space` charge un
autre fichier au même format.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .jsonc import load_jsonc

__all__ = ["SEARCH_SPACES", "SPACES_DIR", "get_search_space", "load_search_space", "validate_search_space"]

SPACES_DIR = Path(__file__).with_name("spaces")
SEARCH_SPACES: dict[str, dict] = {p.stem: load_jsonc(p) for p in sorted(SPACES_DIR.glob("*.jsonc"))}
# wiener_unsupervised : aucun paramètre réglable (niveaux estimés) → pas de fichier.

_TYPES = ("int", "float", "categorical")
_PRIORS = ("uniform", "log-uniform")


def validate_search_space(space: dict, name: str = "espace") -> dict:
    """Vérifie la structure d'un espace de recherche et le renvoie (copie)."""
    if "search_space" not in space or not space["search_space"]:
        raise ValueError(f"{name} : clé 'search_space' absente ou vide.")
    names = set()
    for d in space["search_space"]:
        if d.get("type") not in _TYPES:
            raise ValueError(f"{name} : type inconnu pour {d.get('name')!r} : {d.get('type')!r}.")
        if d["type"] == "categorical":
            if not d.get("categories"):
                raise ValueError(f"{name} : 'categories' manquant pour {d['name']!r}.")
        else:
            if not d["low"] < d["high"]:
                raise ValueError(f"{name} : bornes invalides pour {d['name']!r}.")
            if d.get("prior", "uniform") not in _PRIORS:
                raise ValueError(f"{name} : prior inconnu pour {d['name']!r}.")
        names.add(d["name"])
    unknown = set(space.get("initial_guess", {})) - names
    if unknown:
        raise ValueError(f"{name} : initial_guess contient des dimensions inconnues : {sorted(unknown)}.")
    return copy.deepcopy(space)


def get_search_space(method: str) -> dict:
    """Espace de recherche par défaut de ``method`` (copie modifiable)."""
    if method not in SEARCH_SPACES:
        raise ValueError(f"Pas d'espace de recherche pour '{method}' : {sorted(SEARCH_SPACES)}.")
    return validate_search_space(SEARCH_SPACES[method], method)


def load_search_space(path: str | Path) -> dict:
    """Charge et valide un espace de recherche depuis un fichier JSONC."""
    return validate_search_space(load_jsonc(path), str(path))
