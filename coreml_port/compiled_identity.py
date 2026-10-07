"""Sealed identity for a side-by-side set of 14 compiled Core ML models.

Verify this manifest and its source candidate before constructing any native
CompiledMLModel. The manifest lives outside the compiled-only cache root.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
from typing import Any, Mapping

from .package_identity import EXPECTED_PACKAGES, PackageIdentityError, verify_candidate_identity


SCHEMA_VERSION = 1
KIND = "coreml_compiled_cache"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TOP = {"schema_version", "kind", "source_candidate_identity", "host", "toolchain",
        "packages", "aggregate_sha256"}
_PACKAGE = {"source_package_sha256", "source_description_sha256", "input_names",
            "state_names", "output_names", "directories", "files", "sha256"}
_INSTALLED = ("coremltools_init", "models_utils", "models_model", "libcoremlpython")


class CompiledIdentityError(ValueError):
    """A compiled cache, its source candidate, or its manifest is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _real_directory(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CompiledIdentityError(f"Cannot inspect {label}: {path}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CompiledIdentityError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def _real_file(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CompiledIdentityError(f"Cannot inspect {label}: {path}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise CompiledIdentityError(f"{label} must be a real regular file: {path}")
    return path.resolve(strict=True)


def _inventory_compiled(compiled_root: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    root = _real_directory(Path(compiled_root).expanduser(), "Compiled root")
    expected = {f"{name}.mlmodelc" for name in EXPECTED_PACKAGES}
    try:
        actual = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise CompiledIdentityError(f"Cannot list compiled root: {root}: {exc}") from exc
    if actual != expected:
        raise CompiledIdentityError(
            f"Compiled root must contain exactly 14 .mlmodelc directories: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")

    def visit(package: Path, directory: Path, directories: list[str], files: list[dict]) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise CompiledIdentityError(f"Cannot list compiled directory: {directory}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CompiledIdentityError(f"Cannot inspect compiled entry: {path}: {exc}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise CompiledIdentityError(f"Symlink in compiled cache: {path}")
            relative = path.relative_to(root).as_posix()
            if stat.S_ISDIR(info.st_mode):
                directories.append(relative)
                visit(package, path, directories, files)
            elif stat.S_ISREG(info.st_mode):
                try:
                    digest = _sha_file(path)
                except OSError as exc:
                    raise CompiledIdentityError(f"Cannot hash compiled file: {path}: {exc}") from exc
                files.append({"path": relative, "size": info.st_size, "sha256": digest})
            else:
                raise CompiledIdentityError(f"Non-regular compiled entry: {path}")

    inventory = {}
    for name in EXPECTED_PACKAGES:
        package = _real_directory(root / f"{name}.mlmodelc", f"Compiled {name}")
        directories: list[str] = []
        files: list[dict[str, Any]] = []
        visit(package, package, directories, files)
        if not files:
            raise CompiledIdentityError(f"Compiled package has no files: {name}")
        inventory[name] = {"directories": sorted(directories),
                           "files": sorted(files, key=lambda record: record["path"])}
    return inventory


def _environment_identity() -> tuple[dict[str, str], dict[str, Any]]:
    import coremltools as ct
    import coremltools.models.model as model_module
    import coremltools.models.utils as utils_module
    import coremltools.libcoremlpython as bridge_module

    host = {"system": platform.system(), "macos_version": platform.mac_ver()[0],
            "machine": platform.machine(), "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation()}
    if host["system"] != "Darwin" or not host["macos_version"]:
        raise CompiledIdentityError("Compiled Core ML cache requires macOS")
    paths = {"coremltools_init": Path(ct.__file__), "models_utils": Path(utils_module.__file__),
             "models_model": Path(model_module.__file__),
             "libcoremlpython": Path(bridge_module.__file__)}
    installed = {}
    for label, path in paths.items():
        real = _real_file(path, f"Installed {label}")
        installed[label] = {"path": str(real), "size": real.stat().st_size,
                            "sha256": _sha_file(real)}
    return host, {"coremltools_version": ct.__version__, "installed_files": installed}


def _source_metadata(package: Path) -> dict[str, Any]:
    import coremltools as ct

    spec = ct.models.utils.load_spec(str(package))
    description = spec.description
    result = {"source_description_sha256": hashlib.sha256(
                  description.SerializeToString(deterministic=True)).hexdigest(),
              "input_names": [feature.name for feature in description.input],
              "state_names": [feature.name for feature in description.state],
              "output_names": [feature.name for feature in description.output]}
    for label in ("input_names", "state_names", "output_names"):
        names = result[label]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise CompiledIdentityError(f"Invalid {label} in source {package.name}")
    if not result["input_names"] or not result["output_names"]:
        raise CompiledIdentityError(f"Source lacks inputs or outputs: {package.name}")
    return result


def build_manifest(
    compiled_root: str | os.PathLike[str],
    source_root: str | os.PathLike[str],
    source_identity: str | os.PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Measure a complete compiled tree and its verified sealed source set."""
    if not isinstance(source_identity, Mapping):
        _real_file(Path(source_identity).expanduser(), "Source identity manifest")
    try:
        source = verify_candidate_identity(source_root, source_identity)
    except PackageIdentityError as exc:
        raise CompiledIdentityError(f"Source candidate identity rejected: {exc}") from exc
    compiled = _inventory_compiled(compiled_root)
    host, toolchain = _environment_identity()
    source_packages = {item["name"]: item["sha256"] for item in source["packages"]}
    root = _real_directory(Path(source_root).expanduser(), "Source root")
    packages = {}
    for name in EXPECTED_PACKAGES:
        metadata = _source_metadata(root / f"{name}.mlpackage")
        entry = {"source_package_sha256": source_packages[name], **metadata, **compiled[name]}
        entry["sha256"] = _digest(entry)
        packages[name] = entry
    result = {"schema_version": SCHEMA_VERSION, "kind": KIND,
              "source_candidate_identity": source["aggregate_sha256"],
              "host": host, "toolchain": toolchain, "packages": packages}
    result["aggregate_sha256"] = _digest(result)
    return result


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA.fullmatch(value))


def _relative_path(value: Any, prefix: str) -> bool:
    if not isinstance(value, str) or "\\" in value or "//" in value:
        return False
    parsed = PurePosixPath(value)
    return (bool(parsed.parts) and parsed.parts[0] == prefix and
            len(parsed.parts) > 1 and not parsed.is_absolute() and
            not any(part in (".", "..") for part in parsed.parts) and
            parsed.as_posix() == value)


def _validate_manifest(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != _TOP:
        raise CompiledIdentityError("Malformed compiled manifest top-level fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise CompiledIdentityError("Unsupported compiled manifest schema")
    if value["kind"] != KIND or not _valid_sha(value["source_candidate_identity"]):
        raise CompiledIdentityError("Malformed compiled manifest source binding")
    if not _valid_sha(value["aggregate_sha256"]):
        raise CompiledIdentityError("Malformed compiled manifest aggregate")
    host = value["host"]
    if not isinstance(host, Mapping) or set(host) != {
            "system", "macos_version", "machine", "python_version", "python_implementation"}:
        raise CompiledIdentityError("Malformed compiled manifest host")
    if any(not isinstance(part, str) or not part for part in host.values()):
        raise CompiledIdentityError("Malformed compiled manifest host value")
    toolchain = value["toolchain"]
    if not isinstance(toolchain, Mapping) or set(toolchain) != {
            "coremltools_version", "installed_files"}:
        raise CompiledIdentityError("Malformed compiled manifest toolchain")
    if not isinstance(toolchain["coremltools_version"], str) or not toolchain["coremltools_version"]:
        raise CompiledIdentityError("Malformed coremltools version")
    installed = toolchain["installed_files"]
    if not isinstance(installed, Mapping) or set(installed) != set(_INSTALLED):
        raise CompiledIdentityError("Malformed installed toolchain file set")
    for label in _INSTALLED:
        record = installed[label]
        if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
            raise CompiledIdentityError(f"Malformed installed file: {label}")
        if not isinstance(record["path"], str) or not Path(record["path"]).is_absolute() or \
                type(record["size"]) is not int or record["size"] < 0 or not _valid_sha(record["sha256"]):
            raise CompiledIdentityError(f"Malformed installed file metadata: {label}")
    packages = value["packages"]
    if not isinstance(packages, Mapping) or set(packages) != set(EXPECTED_PACKAGES):
        raise CompiledIdentityError("Compiled manifest must bind exactly 14 packages")
    for name in EXPECTED_PACKAGES:
        entry = packages[name]
        if not isinstance(entry, Mapping) or set(entry) != _PACKAGE:
            raise CompiledIdentityError(f"Malformed compiled package entry: {name}")
        if not all(_valid_sha(entry[key]) for key in (
                "source_package_sha256", "source_description_sha256", "sha256")):
            raise CompiledIdentityError(f"Malformed package digest: {name}")
        for key in ("input_names", "state_names", "output_names"):
            names = entry[key]
            if not isinstance(names, list) or any(not isinstance(x, str) or not x for x in names) or \
                    len(names) != len(set(names)):
                raise CompiledIdentityError(f"Malformed {key}: {name}")
        if not entry["input_names"] or not entry["output_names"]:
            raise CompiledIdentityError(f"Missing input/output metadata: {name}")
        prefix = f"{name}.mlmodelc"
        directories = entry["directories"]
        if not isinstance(directories, list) or directories != sorted(set(directories)) or \
                any(not _relative_path(path, prefix) for path in directories):
            raise CompiledIdentityError(f"Malformed compiled directories: {name}")
        files = entry["files"]
        if not isinstance(files, list) or not files:
            raise CompiledIdentityError(f"Missing compiled file list: {name}")
        previous = ""
        for record in files:
            if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
                raise CompiledIdentityError(f"Malformed compiled file record: {name}")
            path = record["path"]
            if not _relative_path(path, prefix) or path <= previous or \
                    type(record["size"]) is not int or record["size"] < 0 or \
                    not _valid_sha(record["sha256"]):
                raise CompiledIdentityError(f"Malformed compiled file metadata: {name}")
            previous = path
        if _digest({key: entry[key] for key in _PACKAGE if key != "sha256"}) != entry["sha256"]:
            raise CompiledIdentityError(f"Compiled package aggregate mismatch: {name}")
    if _digest({key: value[key] for key in _TOP if key != "aggregate_sha256"}) != value["aggregate_sha256"]:
        raise CompiledIdentityError("Compiled aggregate mismatch")


def _read_manifest(compiled_root: Path, manifest: str | os.PathLike[str] | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(manifest, Mapping):
        return manifest
    path = _real_file(Path(manifest).expanduser(), "Compiled identity manifest")
    root = _real_directory(compiled_root.expanduser(), "Compiled root")
    if path.is_relative_to(root):
        raise CompiledIdentityError("Compiled identity manifest must be outside compiled root")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompiledIdentityError(f"Cannot read compiled identity manifest: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CompiledIdentityError("Compiled identity manifest must be a JSON object")
    return value


def verify_compiled_identity(
    compiled_root: str | os.PathLike[str],
    compiled_manifest: str | os.PathLike[str] | Mapping[str, Any],
    source_root: str | os.PathLike[str],
    source_identity: str | os.PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Re-measure and require exact sealed compiled/source identity agreement."""
    expected = _read_manifest(Path(compiled_root), compiled_manifest)
    _validate_manifest(expected)
    measured = build_manifest(compiled_root, source_root, source_identity)
    if _canonical(measured) != _canonical(expected):
        raise CompiledIdentityError(
            "Compiled identity differs from manifest "
            f"(expected {expected['aggregate_sha256']}, observed {measured['aggregate_sha256']})")
    return measured
