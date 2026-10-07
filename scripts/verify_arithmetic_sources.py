"""Verify retained arithmetic-source snapshots and their pinned revisions.

Read-only by default. --output writes one new JSON file and refuses overwrite.
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH_COMMIT = "134179474539648ba7dee1317959529fbd0e7f89"
SLEEF_COMMIT = "56e1f79cb140fb9326d612d0be06b5250565cade"
BRIDGE_SHA256 = "915f478a3f2a392077012f8dacd9ae433579cbb299019ef2df23217ce536ee67"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, help="Write JSON to an unused path; omit for stdout only.")
args = parser.parse_args()
output = (ROOT / args.output).resolve() if args.output and not args.output.is_absolute() else args.output
if output and output.exists():
    parser.error(f"Output already exists: {output}")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text())


checks = []


def check(label, actual, expected):
    checks.append({"label": label, "actual": actual, "expected": expected, "pass": actual == expected})


def file_check(label, name, expected):
    path = ROOT / name
    actual = digest(path) if path.is_file() else None
    check(label, {"path": name, "sha256": actual}, {"path": name, "sha256": expected})


torch_index = load("evidence/04c-precision/torch-2.7-linear-hashes.json")
for name, expected in sorted(torch_index.items()):
    file_check("torch_initial_snapshot", f"sources/torch-2.7-linear/{name}", expected)

for name in ("arithmetic-source-identity", "softmax-source-identity", "bmm-source-identity"):
    record = load(f"evidence/04d-cpu/{name}.json")
    check(f"{name}_torch_commit", record.get("torch_commit"), TORCH_COMMIT)
    for item in record["files"]:
        path = item.get("path") or f"sources/torch-2.7-linear/{item['name']}"
        file_check(name, path, item["sha256"])
        origin = item.get("origin", "")
        if origin.startswith("http"):
            check(f"{name}_origin_revision:{path}", TORCH_COMMIT in origin, True)

sleef = load("evidence/04d-cpu/sleef-source-identity.json")
check("sleef_source_commit", sleef.get("sleef_commit"), SLEEF_COMMIT)
for item in sleef["files"]:
    file_check("sleef_snapshot", item["path"], item["sha256"])
submodule = load("sources/torch-2.7-linear/sleef-submodule.json")
check("sleef_submodule_revision", submodule.get("sha"), SLEEF_COMMIT)
check("sleef_submodule_parent_revision", TORCH_COMMIT in submodule.get("url", ""), True)
check("sleef_submodule_upstream_revision", SLEEF_COMMIT in submodule.get("html_url", ""), True)

bridge = load("evidence/04b-reliability/coremltools-9.0-bridge-hashes.json")
for name, expected in sorted(bridge.items()):
    file_check("coremltools_9_bridge_snapshot", f"sources/coremltools-9.0-bridge/{name}", expected)
wheel_bridge = ROOT / ".venv/lib/python3.12/site-packages/coremltools/libcoremlpython.so"
file_check("installed_coremltools_9_bridge", str(wheel_bridge.relative_to(ROOT)), BRIDGE_SHA256)

guard_tree = ast.parse((ROOT / "coreml_port/cleanup_guard.py").read_text())
guard_pin = next((ast.literal_eval(node.value) for node in guard_tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "BRIDGE_SHA256"
                                                       for target in node.targets)), None)
check("cleanup_guard_bridge_pin", guard_pin, BRIDGE_SHA256)
guard_build = load("evidence/04b-reliability/guard-build.json")
file_check("cleanup_guard_c_source", "coreml_port/native/cleanup_guard.c", guard_build["source_sha256"])
file_check("cleanup_guard_binary", "artifacts/04b-reliability/cleanup_guard.dylib", guard_build["binary_sha256"])
lock = (ROOT / "requirements-investigation.lock").read_text()
check("locked_torch_version", "torch==2.7.0 --hash=" in lock, True)
check("locked_coremltools_version", "coremltools==9.0 --hash=" in lock, True)
source_exp = (ROOT / "coreml_port/source_exp.py").read_text()
notice = (ROOT / "coreml_port/NOTICE.md").read_text()
check("source_exp_revision_notice", SLEEF_COMMIT in source_exp, True)
check("torch_revision_notice", TORCH_COMMIT in notice, True)
check("sleef_revision_notice", SLEEF_COMMIT in notice, True)

result = {
    "script_sha256": digest(Path(__file__)),
    "pinned_revisions": {"torch": TORCH_COMMIT, "sleef": SLEEF_COMMIT,
                         "coremltools": "9.0", "installed_bridge_sha256": BRIDGE_SHA256},
    "checks": checks,
    "check_count": len(checks),
    "failed": [item["label"] for item in checks if not item["pass"]],
    "passed": all(item["pass"] for item in checks),
}
encoded = json.dumps(result, indent=2) + "\n"
if output:
    with output.open("x") as stream:
        stream.write(encoded)
    print(f"{result['check_count']} arithmetic-source checks; pass={result['passed']}; {output}")
else:
    sys.stdout.write(encoded)
raise SystemExit(0 if result["passed"] else 1)
