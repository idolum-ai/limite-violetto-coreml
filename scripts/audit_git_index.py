"""Inspect exact staged blobs for disallowed artifacts and common secret shapes.

Read-only, standard library only. This is a bounded publication check, not an
exhaustive secret detector. It prints paths/rule names, never matching content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re
import subprocess

LIMIT = 16 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".safetensors", ".npz", ".npy", ".pt", ".pth", ".ckpt", ".bin",
                      ".dylib", ".so", ".o", ".ips", ".log", ".jsonl", ".pem", ".key",
                      ".zip", ".gz", ".tar", ".part", ".bundle"}
FORBIDDEN_PARTS = {"artifacts", ".venv", ".cache", ".cleanup", ".codex", ".agents",
                   "__pycache__", "sessions", "transcripts", "source-snapshots"}
RULES = {
    "github token": rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "service token": rb"(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|xox[baprs]-[A-Za-z0-9-]{15,})",
    "AWS access key": rb"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "credential URL": rb"https?://[^\s/\"']+:[^\s/@\"']+@",
    "private home path": rb"/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/",
    "machine identifier": rb"(?:IOPlatformUUID|CrashReporter Key|incident_id)\s*[:=]",
    "host-local battery identifier": rb"InternalBattery-[0-9]+\s+\(id=[0-9]+\)",
    "private trace checksum entry": rb"(?m)^[0-9a-f]{64}[ \t]+\S*\.trace(?:/|$)",
    "trace-session identifier": rb"\.trace/[^\r\n]*\b[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\b",
}


def main() -> int:
    rows = subprocess.check_output(["git", "ls-files", "--stage", "-z"]).split(b"\0")
    entries = []
    failures = []
    for row in filter(None, rows):
        meta, path = row.split(b"\t", 1)
        mode, oid, stage = meta.split()
        name = path.decode()
        if mode != b"100644" or stage != b"0":
            failures.append({"path": name, "rule": "non-regular file or unresolved stage"})
        entries.append((name, oid))
    if not entries:
        raise SystemExit("Nothing staged/tracked to audit")
    # Batch-read immutable index objects; working-tree content is not trusted.
    output = subprocess.check_output(["git", "cat-file", "--batch"], input=b"\n".join(oid for _, oid in entries) + b"\n")
    pos = 0
    total = 0
    largest = []
    for name, oid in entries:
        end = output.index(b"\n", pos)
        actual_oid, kind, size = output[pos:end].split()
        size = int(size)
        pos = end + 1
        data = output[pos:pos + size]
        pos += size + 1
        if actual_oid != oid or kind != b"blob":
            raise RuntimeError("Unexpected Git object response")
        p = PurePosixPath(name)
        if set(p.parts) & FORBIDDEN_PARTS or any(part.endswith(".trace") for part in p.parts) or p.suffix in FORBIDDEN_SUFFIXES or p.name in {"OPERATOR-TRAIL.md", "compiled-identity.json", "crash-stack.json"} or p.name.startswith(".env"):
            failures.append({"path": name, "rule": "excluded material"})
        if size > LIMIT:
            failures.append({"path": name, "rule": "larger than 16 MiB"})
        if p.suffix != ".png":
            if b"\0" in data:
                failures.append({"path": name, "rule": "unexpected binary"})
            for label, pattern in RULES.items():
                if re.search(pattern, data):
                    failures.append({"path": name, "rule": label})
        total += size
        largest.append({"path": name, "bytes": size})
    print(json.dumps({"passed": not failures, "files": len(entries), "bytes": total,
                      "index_entries_sha256": hashlib.sha256(b"\0".join(rows)).hexdigest(),
                      "largest": sorted(largest, key=lambda x: x["bytes"], reverse=True)[:5],
                      "failures": failures,
                      "scope": "exact index; common secret patterns, privacy paths, host/session metadata, artifacts, sizes; not exhaustive"}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
