"""Deterministic identity manifests for a complete Core ML candidate set.

The caller must verify the manifest before opening any MLModel package. The
manifest binds every regular file under the 14 expected package directories.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping


EXPECTED_PACKAGES = ("embedding", *(str(i) for i in range(12)), "head")
SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class PackageIdentityError(ValueError):
    """The candidate tree or its identity manifest is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_package(root: Path, package: str) -> list[dict[str, Any]]:
    package_path = root / f"{package}.mlpackage"
    try:
        mode = package_path.lstat().st_mode
    except FileNotFoundError as exc:
        raise PackageIdentityError(f"Missing package: {package_path.name}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise PackageIdentityError(f"Package must be a real directory: {package_path}")

    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise PackageIdentityError(f"Cannot read package directory: {directory}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PackageIdentityError(f"Cannot inspect package entry: {path}: {exc}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise PackageIdentityError(f"Symlink is not allowed in candidate package: {path}")
            if stat.S_ISDIR(info.st_mode):
                visit(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise PackageIdentityError(f"Non-regular package entry is not allowed: {path}")
            relative = path.relative_to(root).as_posix()
            records.append(
                {"path": relative, "size": info.st_size, "sha256": _digest_file(path)}
            )

    visit(package_path)
    if not records:
        raise PackageIdentityError(f"Package contains no files: {package_path.name}")
    records.sort(key=lambda item: item["path"])
    return records


def inventory_candidate(model_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Inventory exactly the expected package set under an explicit root."""
    supplied_root = Path(model_root).expanduser()
    try:
        root_mode = supplied_root.lstat().st_mode
    except OSError as exc:
        raise PackageIdentityError(f"Cannot inspect model root {supplied_root}: {exc}") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise PackageIdentityError(f"Model root must be a real directory: {supplied_root}")
    root = supplied_root.resolve(strict=True)

    try:
        children = list(root.iterdir())
    except OSError as exc:
        raise PackageIdentityError(f"Cannot read model root {root}: {exc}") from exc
    expected_names = {f"{name}.mlpackage" for name in EXPECTED_PACKAGES}
    actual_names = {child.name for child in children}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise PackageIdentityError("Model root must contain exactly the expected 14 packages (" + "; ".join(details) + ")")

    packages = []
    for name in EXPECTED_PACKAGES:
        files = _walk_package(root, name)
        packages.append({"name": name, "files": files, "sha256": hashlib.sha256(_canonical_json(files)).hexdigest()})
    aggregate = hashlib.sha256(_canonical_json(packages)).hexdigest()
    return {"schema_version": SCHEMA_VERSION, "packages": packages, "aggregate_sha256": aggregate}


def _read_manifest(manifest: str | os.PathLike[str] | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(manifest, Mapping):
        return manifest
    path = Path(manifest)
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageIdentityError(f"Cannot read identity manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PackageIdentityError("Identity manifest must be a JSON object.")
    return value


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != {"schema_version", "packages", "aggregate_sha256"}:
        raise PackageIdentityError("Identity manifest has missing or unexpected top-level fields.")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise PackageIdentityError("Unsupported identity manifest schema version.")
    if not isinstance(manifest["aggregate_sha256"], str) or not _SHA256_RE.fullmatch(manifest["aggregate_sha256"]):
        raise PackageIdentityError("Malformed aggregate SHA-256.")
    packages = manifest["packages"]
    if not isinstance(packages, list) or len(packages) != len(EXPECTED_PACKAGES):
        raise PackageIdentityError("Identity manifest must list exactly 14 packages.")
    for expected, package in zip(EXPECTED_PACKAGES, packages):
        if not isinstance(package, Mapping) or set(package) != {"name", "files", "sha256"}:
            raise PackageIdentityError("Malformed package entry in identity manifest.")
        if package["name"] != expected:
            raise PackageIdentityError("Package names/order do not match the expected candidate set.")
        if not isinstance(package["sha256"], str) or not _SHA256_RE.fullmatch(package["sha256"]):
            raise PackageIdentityError(f"Malformed package digest for {expected}.")
        files = package["files"]
        if not isinstance(files, list) or not files:
            raise PackageIdentityError(f"Package {expected} must list at least one file.")
        previous = ""
        for record in files:
            if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
                raise PackageIdentityError(f"Malformed file entry in package {expected}.")
            file_path = record["path"]
            if not isinstance(file_path, str):
                raise PackageIdentityError(f"Malformed path in package {expected}.")
            parsed = PurePosixPath(file_path)
            if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts or not parsed.parts or parsed.parts[0] != f"{expected}.mlpackage":
                raise PackageIdentityError(f"Unsafe or misplaced manifest path: {file_path!r}")
            if file_path <= previous:
                raise PackageIdentityError(f"File paths must be unique and sorted in package {expected}.")
            previous = file_path
            if type(record["size"]) is not int or record["size"] < 0:
                raise PackageIdentityError(f"Malformed file size for {file_path}.")
            if not isinstance(record["sha256"], str) or not _SHA256_RE.fullmatch(record["sha256"]):
                raise PackageIdentityError(f"Malformed file digest for {file_path}.")


def verify_candidate_identity(
    model_root: str | os.PathLike[str],
    manifest: str | os.PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Re-inventory and require exact agreement with a previously saved manifest.

    Returns the freshly measured manifest on success. Raises
    :class:`PackageIdentityError` before any model loading should occur on a
    mismatch or invalid tree.
    """
    expected = _read_manifest(manifest)
    _validate_manifest_shape(expected)
    actual = inventory_candidate(model_root)
    if _canonical_json(actual) != _canonical_json(expected):
        raise PackageIdentityError(
            "Candidate package identity does not match the supplied manifest "
            f"(expected aggregate {expected['aggregate_sha256']}, "
            f"observed {actual['aggregate_sha256']})."
        )
    return actual


def write_candidate_manifest(
    model_root: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> dict[str, Any]:
    """Inventory a candidate and write its deterministic JSON manifest."""
    result = inventory_candidate(model_root)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result
