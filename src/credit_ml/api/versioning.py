from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_model_version(pipeline_path: Path, metadata_path: Path) -> str:
    # Combined Hash (artifact + metadata)
    hp = sha256_file(pipeline_path)
    hm = sha256_file(metadata_path)
    combo = hashlib.sha256((hp + hm).encode("utf-8")).hexdigest()
    return combo[:12]  # short for logs/UI