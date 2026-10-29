#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lecture de fichiers JSONC : JSON tolérant les commentaires ``//`` et
``/* */`` et les virgules finales, pour documenter un espace de recherche
valeur par valeur. Sans dépendance (bibliothèque standard seulement).
"""
from __future__ import annotations

import json
import os
import re

__all__ = ["loads_jsonc", "load_jsonc"]

_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _strip_comments(text: str) -> str:
    """Retire les commentaires hors chaînes de caractères."""
    out = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
        elif c == '"':
            in_string = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise ValueError("Commentaire /* non terminé.")
            i = j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def loads_jsonc(text: str):
    """Désérialise une chaîne JSONC."""
    return json.loads(_TRAILING_COMMA.sub(r"\1", _strip_comments(text)))


def load_jsonc(path: str | os.PathLike):
    """Charge un fichier JSON ou JSONC."""
    with open(path, encoding="utf-8") as f:
        return loads_jsonc(f.read())
