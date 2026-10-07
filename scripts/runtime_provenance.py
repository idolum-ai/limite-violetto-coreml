"""Hash runtime and native cleanup/prediction guard inputs for gate evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runtime_provenance() -> dict[str, str]:
    """Return a stable file->SHA-256 map for guards active in this checkout.

    Optional prediction guard sources/libraries are included only once present;
    required runtime and cleanup-guard sources must exist.
    """
    paths = [
        "coreml_port/runtime.py",
        "coreml_port/cleanup_guard.py",
        "artifacts/04b-reliability/cleanup_guard.dylib",
    ]
    optional = [
        "coreml_port/prediction_guard.py",
        "coreml_port/native/prediction_guard.mm",
        "artifacts/04e-runtime/prediction_guard.dylib",
    ]
    for relative in paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"Required runtime provenance file missing: {relative}")
    paths.extend(relative for relative in optional if (ROOT / relative).is_file())
    result = {}
    for relative in paths:
        digest = hashlib.sha256()
        with (ROOT / relative).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[relative] = digest.hexdigest()
    return result
