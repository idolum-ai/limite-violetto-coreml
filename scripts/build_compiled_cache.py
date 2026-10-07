"""Build a side-by-side sealed cache of candidate-6 compiled Core ML models.

This script never opens a native model for prediction. It compiles the 14
source packages sequentially into one fresh artifacts/05c-compiled attempt;
the identity manifest and durable progress live outside that root.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coreml_port.compiled_identity import build_manifest, verify_compiled_identity
from coreml_port.package_identity import EXPECTED_PACKAGES, verify_candidate_identity


CANDIDATE6_AGGREGATE = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def run(attempt: str, source_root: Path, source_identity: Path) -> int:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", attempt):
        raise ValueError("Attempt must be a lowercase path-safe slug")
    artifacts = ROOT / "artifacts/05c-compiled" / attempt
    evidence = ROOT / "evidence/05c-compiled" / attempt
    if artifacts.exists() or evidence.exists():
        raise FileExistsError("Fresh artifact and evidence attempt paths required")
    evidence.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    source_files = {
        "builder": Path(__file__),
        "compiled_identity": ROOT / "coreml_port/compiled_identity.py",
        "package_identity": ROOT / "coreml_port/package_identity.py",
    }
    status_file = evidence / "status.json"
    events_file = evidence / "events.jsonl"
    status = {"status": "created", "attempt": attempt, "started_utc": utc(),
              "compiled_root": str(artifacts.resolve()),
              "source_root": str(source_root.absolute()),
              "source_identity": str(source_identity.absolute()),
              "source_sha256": {},
              "packages": {name: {"status": "pending"} for name in EXPECTED_PACKAGES}}

    def save() -> None:
        atomic_json(status_file, status)

    def event(kind: str, **fields) -> None:
        row = {"utc": utc(), "kind": kind, **fields}
        with events_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)

    save()
    event("created", attempt=attempt)
    try:
        snapshots = evidence / "source-snapshots"
        snapshots.mkdir()
        source_hashes = {}
        for label, path in source_files.items():
            destination = snapshots / path.name
            shutil.copy2(path, destination)
            digest = sha256(path)
            if digest != sha256(destination):
                raise RuntimeError(f"Source snapshot changed while copying: {label}")
            source_hashes[str(path.relative_to(ROOT))] = digest
        status["source_sha256"] = source_hashes
        save()
        event("source_snapshotted", source_sha256=source_hashes)

        identity_mode = source_identity.lstat().st_mode
        if stat.S_ISLNK(identity_mode) or not stat.S_ISREG(identity_mode):
            raise ValueError("Source identity must be a real regular file")
        source_before = verify_candidate_identity(source_root, source_identity)
        if source_before["aggregate_sha256"] != CANDIDATE6_AGGREGATE:
            raise ValueError("Source is not the sealed candidate-6 aggregate")
        status["source_candidate_identity"] = source_before["aggregate_sha256"]
        status["status"] = "source_verified"
        save()
        event("source_verified", aggregate=source_before["aggregate_sha256"])

        import coremltools as ct
        status["coremltools_version"] = ct.__version__
        status["status"] = "compiling"
        save()
        for name in EXPECTED_PACKAGES:
            source = source_root / f"{name}.mlpackage"
            destination = artifacts / f"{name}.mlmodelc"
            if destination.exists():
                raise FileExistsError(destination)
            status["packages"][name] = {"status": "compiling", "started_utc": utc()}
            save()
            event("compile_start", package=name)
            begin = time.perf_counter()
            returned = ct.models.utils.compile_model(
                str(source), destination_path=str(destination))
            seconds = time.perf_counter() - begin
            if Path(returned).resolve() != destination.resolve():
                raise RuntimeError(f"Unexpected compile destination for {name}: {returned}")
            status["packages"][name] = {"status": "compiled", "seconds": seconds,
                                         "finished_utc": utc()}
            save()
            event("compile_complete", package=name, seconds=seconds)

        source_after = verify_candidate_identity(source_root, source_identity)
        if source_after != source_before:
            raise RuntimeError("Sealed source candidate changed during compilation")
        for label, path in source_files.items():
            if sha256(path) != source_hashes[str(path.relative_to(ROOT))]:
                raise RuntimeError(f"Builder/identity source changed during compilation: {label}")
        status["status"] = "source_reverified"
        save()
        event("source_reverified", aggregate=source_after["aggregate_sha256"])

        manifest = build_manifest(artifacts, source_root, source_identity)
        manifest_file = evidence / "compiled-identity.json"
        atomic_json(manifest_file, manifest)
        status["status"] = "manifest_written"
        status["compiled_aggregate_sha256"] = manifest["aggregate_sha256"]
        status["compiled_manifest_sha256"] = sha256(manifest_file)
        save()
        event("manifest_written", aggregate=manifest["aggregate_sha256"])
        measured = verify_compiled_identity(artifacts, manifest_file, source_root, source_identity)
        if measured != manifest:
            raise RuntimeError("Compiled manifest changed during verification")
        status["status"] = "complete"
        status["completed_utc"] = utc()
        save()
        event("complete", aggregate=manifest["aggregate_sha256"])
        return 0
    except BaseException as exc:
        status["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        status["failed_utc"] = utc()
        save()
        event("failed", error_type=type(exc).__name__, message=str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source-root", required=True,
                        help="Explicit sealed candidate-6 .mlpackage root")
    parser.add_argument("--source-identity", required=True,
                        help="Candidate-6 package-identity.json")
    args = parser.parse_args()
    def explicit_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path.cwd() / path

    return run(args.attempt, explicit_path(args.source_root),
               explicit_path(args.source_identity))


if __name__ == "__main__":
    raise SystemExit(main())
