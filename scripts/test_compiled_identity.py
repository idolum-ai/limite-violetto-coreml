"""Synthetic compiled-cache identity rejection tests; no compilation or inference."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coreml_port import compiled_identity as ci
from coreml_port.package_identity import EXPECTED_PACKAGES, write_candidate_manifest


def fixture(base: Path):
    source = base / "source"
    compiled = base / "compiled"
    source.mkdir()
    compiled.mkdir()
    for name in EXPECTED_PACKAGES:
        package = source / f"{name}.mlpackage"
        package.mkdir()
        (package / "model.txt").write_text(f"source {name}\n")
        target = compiled / f"{name}.mlmodelc"
        target.mkdir()
        (target / "model.mil").write_text(f"compiled {name}\n")
        (target / "weights").mkdir()
        (target / "weights/weight.bin").write_bytes(name.encode())
    source_identity = base / "source-identity.json"
    source_identity.write_text(json.dumps(write_candidate_manifest(source, source_identity)))
    return source, compiled, source_identity


def fake_environment():
    host = {"system": "Darwin", "macos_version": "26.5.2", "machine": "arm64",
            "python_version": "3.12.9", "python_implementation": "CPython"}
    installed = {label: {"path": f"/mock/{label}", "size": 1, "sha256": "a" * 64}
                 for label in ci._INSTALLED}
    return host, {"coremltools_version": "9.0", "installed_files": installed}


def fake_metadata(package: Path):
    return {"source_description_sha256": ci._digest(package.name),
            "input_names": ["x"], "state_names": [], "output_names": ["y"]}


def expect_rejected(fn, label):
    try:
        fn()
    except (ci.CompiledIdentityError, ValueError):
        return
    raise AssertionError(f"Expected identity rejection: {label}")


def run():
    checks = []
    with tempfile.TemporaryDirectory(prefix="compiled-identity-test-") as temp:
        base = Path(temp)
        source, compiled, source_identity = fixture(base)
        with patch.object(ci, "_environment_identity", side_effect=fake_environment), \
             patch.object(ci, "_source_metadata", side_effect=fake_metadata):
            manifest = ci.build_manifest(compiled, source, source_identity)
            manifest_path = base / "compiled-identity.json"
            manifest_path.write_text(json.dumps(manifest))
            assert ci.verify_compiled_identity(compiled, manifest_path, source, source_identity) == manifest
            checks.append("roundtrip")
            assert set(manifest["packages"]) == set(EXPECTED_PACKAGES)
            assert manifest["source_candidate_identity"] == json.loads(source_identity.read_text())["aggregate_sha256"]
            checks.append("source_binding")

            file = compiled / "0.mlmodelc/model.mil"
            original = file.read_bytes()
            file.write_bytes(original + b"tamper")
            expect_rejected(lambda: ci.verify_compiled_identity(compiled, manifest_path, source, source_identity),
                            "changed compiled file")
            file.write_bytes(original)
            checks.append("file_tamper")

            extra = compiled / "unlisted.txt"
            extra.write_text("extra")
            expect_rejected(lambda: ci.verify_compiled_identity(compiled, manifest_path, source, source_identity),
                            "extra cache root entry")
            extra.unlink()
            checks.append("extra_root")

            extra_dir = compiled / "0.mlmodelc/empty"
            extra_dir.mkdir()
            expect_rejected(lambda: ci.verify_compiled_identity(compiled, manifest_path, source, source_identity),
                            "extra empty directory")
            extra_dir.rmdir()
            checks.append("extra_empty_directory")

            link = compiled / "0.mlmodelc/link"
            link.symlink_to("model.mil")
            expect_rejected(lambda: ci.verify_compiled_identity(compiled, manifest_path, source, source_identity),
                            "symlink")
            link.unlink()
            checks.append("symlink")

            fifo = compiled / "0.mlmodelc/fifo"
            os.mkfifo(fifo)
            expect_rejected(lambda: ci.verify_compiled_identity(compiled, manifest_path, source, source_identity),
                            "nonregular entry")
            fifo.unlink()
            checks.append("nonregular")

            missing = compiled / "head.mlmodelc"
            moved = base / "head-hidden"
            missing.rename(moved)
            expect_rejected(lambda: ci.verify_compiled_identity(compiled, manifest_path, source, source_identity),
                            "missing compiled package")
            moved.rename(missing)
            checks.append("missing_package")

            forged = copy.deepcopy(manifest)
            forged["host"]["machine"] = "x86_64"
            forged["aggregate_sha256"] = ci._digest({k: v for k, v in forged.items()
                                                      if k != "aggregate_sha256"})
            expect_rejected(lambda: ci.verify_compiled_identity(compiled, forged, source, source_identity),
                            "validly rehashed wrong host")
            checks.append("host_drift")

            malformed = copy.deepcopy(manifest)
            malformed["packages"]["0"]["files"][0]["path"] = "../unsafe"
            with patch.object(ci, "_source_metadata", side_effect=AssertionError("should not inspect source")):
                expect_rejected(lambda: ci.verify_compiled_identity(compiled, malformed, source, source_identity),
                                "malformed manifest before source load")
            checks.append("malformed_before_source_load")

            inside = compiled / "identity.json"
            inside.write_text(json.dumps(manifest))
            expect_rejected(lambda: ci.verify_compiled_identity(compiled, inside, source, source_identity),
                            "manifest inside cache")
            inside.unlink()
            checks.append("manifest_outside_root")

            source_file = source / "0.mlpackage/model.txt"
            source_file.write_text("changed")
            expect_rejected(lambda: ci.verify_compiled_identity(compiled, manifest_path, source, source_identity),
                            "source candidate tamper")
            checks.append("source_tamper")
    print(json.dumps({"status": "pass", "checks": checks, "count": len(checks)}))


if __name__ == "__main__":
    run()
