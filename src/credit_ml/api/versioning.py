from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_metadata_without_version(metadata_path: Path) -> str:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    data.pop("model_version", None)  # <- key so the hash remains the same after saving the version in metadata

    # Stable serialization (sorted keys, no spaces) to ensure the same hash
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()

def compute_model_version(pipeline_path: Path, metadata_path: Path) -> str:
    # Combined Hash (artifact + metadata)
    hp = sha256_file(pipeline_path)
    hm = sha256_metadata_without_version(metadata_path)
    combo = hashlib.sha256((hp + hm).encode("utf-8")).hexdigest()
    return combo[:12]  # short for logs/UI