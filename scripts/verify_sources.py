"""Verify extracted runtime/model assets against captured upstream identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/01-reference"
SOURCE = ROOT / "sources/runtime/paradigma-inc-limite-violetto-cd68d27"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> None:
    tree = json.loads((EVIDENCE / "runtime-tree.json").read_text())
    assert not tree.get("truncated")
    runtime = []
    for item in tree["tree"]:
        if item["type"] != "blob":
            continue
        data = (SOURCE / item["path"]).read_bytes()
        actual = git_blob(data)
        runtime.append({"path": item["path"], "expected_git_blob_sha1": item["sha"],
                        "actual_git_blob_sha1": actual, "matches": actual == item["sha"]})
    model = []
    metadata = json.loads((EVIDENCE / "model-metadata.json").read_text())
    assert metadata["sha"] == "9402e422e87b220507963cd42c997459c1e87b43"
    for item in metadata["siblings"]:
        path = ROOT / "sources/model" / item["rfilename"]
        if item["rfilename"] == "model.safetensors":
            continue  # Deliberately not downloaded in Step 1.
        data = path.read_bytes()
        if "lfs" in item:
            expected, actual, algorithm = item["lfs"]["sha256"], hashlib.sha256(data).hexdigest(), "sha256"
        else:
            expected, actual, algorithm = item["blobId"], git_blob(data), "git_blob_sha1"
        model.append({"path": item["rfilename"], "algorithm": algorithm,
                      "expected": expected, "actual": actual, "matches": expected == actual})
    additional = []
    extra_meta_path = EVIDENCE / "hf-limite-1b-value-model-metadata.json"
    if extra_meta_path.exists():
        extra_meta = json.loads(extra_meta_path.read_text())
        inventory = {item["rfilename"]: item for item in extra_meta["siblings"]}
        for path in sorted((ROOT / "sources/value-model").iterdir()):
            if not path.is_file():
                continue
            item = inventory[path.name]
            data = path.read_bytes()
            if "lfs" in item:
                expected, actual = item["lfs"]["sha256"], hashlib.sha256(data).hexdigest()
            else:
                expected, actual = item["blobId"], git_blob(data)
            additional.append({"path": str(path.relative_to(ROOT)), "expected": expected,
                               "actual": actual, "matches": expected == actual})
        for repo in ("autoresearch", "xi-log-concavity-lean"):
            inventory = json.loads((EVIDENCE / f"github-{repo}-tree.json").read_text())
            assert not inventory.get("truncated")
            blobs = {item["path"]: item["sha"] for item in inventory["tree"] if item["type"] == "blob"}
            for path in sorted((ROOT / "sources" / repo).iterdir()):
                if not path.is_file():
                    continue
                expected, actual = blobs[path.name], git_blob(path.read_bytes())
                additional.append({"path": str(path.relative_to(ROOT)), "expected": expected,
                                   "actual": actual, "matches": expected == actual})
    result = {"runtime": runtime, "model_assets": model, "additional_sources": additional,
              "passed": all(item["matches"] for item in runtime + model + additional)}
    (EVIDENCE / "source-integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(runtime)} runtime files, {len(model)} model assets, {len(additional)} additional reference files checked; pass={result['passed']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
