"""Read-only export/source verification; no downloads, models, or ML imports."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> int:
    failures: list[str] = []
    counts = {"imported_files": 0, "source_identity_checks": 0, "python_files": 0, "json_files": 0}

    def check(path: str, expected: str, algorithm=sha256) -> None:
        file = ROOT / path
        if file.is_symlink() or not file.is_file():
            failures.append(f"Missing or symlinked file: {path}")
        elif algorithm(file.read_bytes()) != expected:
            failures.append(f"Identity mismatch: {path}")

    def load(path: str):
        return json.loads((ROOT / path).read_text())

    inventory = load("docs/import-manifest.json")
    for item in inventory["files"]:
        check(item["path"], item["export_sha256"])
        counts["imported_files"] += 1

    tree = load("evidence/01-reference/runtime-tree.json")
    if tree.get("truncated"):
        failures.append("Upstream runtime tree is truncated")
    for item in tree["tree"]:
        if item["type"] == "blob":
            check("sources/runtime/paradigma-inc-limite-violetto-cd68d27/" + item["path"], item["sha"], git_blob)
            counts["source_identity_checks"] += 1

    for directory, metadata, revision, excluded in [
        ("model", "model-metadata.json", "9402e422e87b220507963cd42c997459c1e87b43", {"model.safetensors", ".gitattributes"}),
        ("value-model", "hf-limite-1b-value-model-metadata.json", "ce83740e0e853a7d4e5b80fca29a65fd5b8db7fb", set()),
    ]:
        meta = load("evidence/01-reference/" + metadata)
        if meta["sha"] != revision:
            failures.append(f"Unexpected {directory} revision")
        upstream = {item["rfilename"]: item for item in meta["siblings"]}
        imported = [item for item in inventory["files"] if item["path"].startswith(f"sources/{directory}/")]
        for item in imported:
            name = item["path"].removeprefix(f"sources/{directory}/")
            source = upstream[name]
            if "lfs" in source:
                check(item["path"], source["lfs"]["sha256"])
            else:
                check(item["path"], source["blobId"], git_blob)
            counts["source_identity_checks"] += 1
        if directory == "model":
            actual = {item["path"].removeprefix("sources/model/") for item in imported}
            if actual != set(upstream) - excluded:
                failures.append("Unexpected retained model asset set")

    for index, prefix in [
        ("evidence/04c-precision/torch-2.7-linear-hashes.json", "sources/torch-2.7-linear/"),
        ("evidence/04b-reliability/coremltools-9.0-bridge-hashes.json", "sources/coremltools-9.0-bridge/"),
    ]:
        for name, digest in load(index).items():
            check(prefix + name, digest)
            counts["source_identity_checks"] += 1
    for index in ["arithmetic-source-identity", "softmax-source-identity", "bmm-source-identity", "sleef-source-identity"]:
        for item in load(f"evidence/04d-cpu/{index}.json")["files"]:
            path = item.get("path") or "sources/torch-2.7-linear/" + item["name"]
            check(path, item["sha256"])
            counts["source_identity_checks"] += 1

    for directory in ("coreml_port", "portable", "experiments", "scripts", "sources", "evidence", "docs"):
        for file in sorted((ROOT / directory).rglob("*")):
            if not file.is_file() or "__pycache__" in file.parts:
                continue
            try:
                if file.suffix == ".py":
                    ast.parse(file.read_text(), filename=str(file.relative_to(ROOT)))
                    counts["python_files"] += 1
                elif file.suffix == ".json":
                    json.loads(file.read_text())
                    counts["json_files"] += 1
            except (SyntaxError, ValueError) as exc:
                failures.append(f"Parse failure: {file.relative_to(ROOT)}: {exc}")
    print(json.dumps({"passed": not failures, "counts": counts, "failures": failures,
                      "scope": "export integrity and syntax only; no numerical qualification"}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
