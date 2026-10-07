"""Acquire public source metadata and small model assets; never model weights.

Run from any directory with Python 3.11+. Requires ordinary authenticated gh
for GitHub API reads. No credential values are read or written by this script.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "01-reference"
RUNTIME_REV = "cd68d27f100ebb7c84f05603d5af0e8f934b28b5"
MODEL_REV = "9402e422e87b220507963cd42c997459c1e87b43"
MODEL = "paradigma-inc/limite-1b-violetto"
records = []


def save(relative: str, data: bytes, origin: str) -> None:
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    records.append({"path": relative, "source": origin, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()})
    print(f"saved {relative} ({len(data)} bytes)", flush=True)


def fetch(url: str, relative: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "violetto-reference-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(25_000_001)
    if len(data) > 25_000_000:
        raise ValueError(f"Small-file download limit exceeded: {relative}")
    save(relative, data, url)
    return data


def gh(endpoint: str, relative: str) -> bytes:
    result = subprocess.run(["gh", "api", endpoint], capture_output=True, check=True)
    save(relative, result.stdout, f"https://api.github.com/{endpoint}")
    return result.stdout


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    gh(f"repos/paradigma-inc/limite-violetto/commits/{RUNTIME_REV}", "evidence/01-reference/runtime-commit.json")
    gh(f"repos/paradigma-inc/limite-violetto/git/trees/{RUNTIME_REV}?recursive=1", "evidence/01-reference/runtime-tree.json")
    gist = json.loads(gh("gists/713fe165fded8bfeddf7b5b7f8c50239", "evidence/01-reference/challenge-gist.json"))
    for i, item in enumerate(gist["files"].values()):
        if item.get("truncated"):
            fetch(item["raw_url"], f"evidence/01-reference/challenge-{i}.md")
        else:
            save(f"evidence/01-reference/challenge-{i}.md", item["content"].encode(), item["raw_url"])
    meta = json.loads(fetch(f"https://huggingface.co/api/models/{MODEL}/revision/{MODEL_REV}?blobs=true", "evidence/01-reference/model-metadata.json"))
    if meta["sha"] != MODEL_REV:
        raise ValueError("Model revision mismatch")
    for item in meta["siblings"]:
        name = item["rfilename"]
        if name.endswith((".safetensors", ".bin", ".pt", ".pth", ".onnx")):
            continue
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unexpected model asset path")
        fetch(f"https://huggingface.co/{MODEL}/resolve/{MODEL_REV}/{name}", f"sources/model/{name}")
    for package in ("coremltools", "torch", "transformers", "pytest"):
        fetch(f"https://pypi.org/pypi/{package}/json", f"evidence/01-reference/pypi-{package}.json")
    for page in ("installing-coremltools", "convert-pytorch-workflow", "stateful-models"):
        fetch(f"https://apple.github.io/coremltools/docs-guides/source/{page}.html", f"evidence/01-reference/apple-{page}.html")
    save("evidence/01-reference/acquisition.json", json.dumps({
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": records,
    }, indent=2).encode() + b"\n", "scripts/acquire_reference.py")


if __name__ == "__main__":
    main()
