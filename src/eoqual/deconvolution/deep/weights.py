"""
Poids pré-entraînés : téléchargement à la demande, cache local, vérification
par empreinte SHA-256. Les poids ne sont jamais inclus dans le paquet.

Cache : ``$EOQUAL_WEIGHTS_DIR`` s'il est défini, sinon
``~/.cache/eoqual/deconvolution``.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

__all__ = ["KNOWN_WEIGHTS", "weights_dir", "resolve_weights", "sha256"]

#: Poids connus : URL, empreinte SHA-256, licence, architecture.
KNOWN_WEIGHTS = {
    "usrnet": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/usrnet.pth",
        "sha256": "2b8c32777fc56bde7fe81f47e60b83c109606c8559aaaafecf306a2108a2082b",
        "license": "MIT (KAIR, Kai Zhang)",
        "arch": {"n_iter": 8, "h_nc": 64, "nc": (64, 128, 256, 512), "nb": 2},
    },
    "usrnet_tiny": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/usrnet_tiny.pth",
        "sha256": "d82929fc053d6bbe181adba9e2be3d23548eeca8b7b4232eb122f020352295cf",
        "license": "MIT (KAIR, Kai Zhang)",
        "arch": {"n_iter": 6, "h_nc": 32, "nc": (16, 32, 64, 64), "nb": 2},
    },
}


def weights_dir() -> Path:
    """Répertoire du cache des poids."""
    return Path(os.environ.get("EOQUAL_WEIGHTS_DIR", Path.home() / ".cache" / "eoqual" / "deconvolution"))


def sha256(path: str | Path) -> str:
    """Empreinte SHA-256 d'un fichier."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_weights(name_or_path: str | Path) -> Path:
    """
    Chemin local des poids : fichier existant, ou nom de
    :data:`KNOWN_WEIGHTS` (téléchargé au premier appel puis vérifié).

    Raises
    ------
    ValueError
        Nom inconnu, ou empreinte différente de celle attendue.
    """
    p = Path(name_or_path)
    if p.suffix and p.exists():
        return p
    name = str(name_or_path)
    if name not in KNOWN_WEIGHTS:
        raise ValueError(f"Poids inconnus {name!r} : chemin de fichier ou un de {sorted(KNOWN_WEIGHTS)}.")
    info = KNOWN_WEIGHTS[name]
    dest = weights_dir() / f"{name}.pth"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(info["url"], tmp)
        tmp.replace(dest)
    digest = sha256(dest)
    if digest != info["sha256"]:
        raise ValueError(f"Empreinte inattendue pour {dest} : {digest} (attendu {info['sha256']}). "
                         "Supprimer le fichier pour le retélécharger.")
    return dest
