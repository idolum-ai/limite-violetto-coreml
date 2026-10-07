"""Locate first candidate-4 Core ML group mismatch on the frozen native tiny-3 prefix.

Each group sees the matching native Torch preceding boundary and a fresh zero
state. Stop at the first difference and save a stage-probe-compatible capture.
"""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--attempt", required=True, help="Unused single directory name under evidence/04d-gpu.")
p.add_argument("--model-dir", required=True, help="Explicit 14-package candidate root.")
p.add_argument("--identity-manifest", required=True, help="Identity manifest for that exact candidate.")
a = p.parse_args()
attempt = Path(a.attempt)
if attempt.is_absolute() or len(attempt.parts) != 1 or attempt.name in ("", ".", ".."):
    p.error("--attempt must be one safe directory name")
out = ROOT / "evidence/04d-gpu" / attempt
if out.exists():
    p.error("Attempt evidence directory already exists")

import coremltools as ct
import numpy as np
import torch
from coreml_port.package_identity import verify_candidate_identity
from coreml_port.prediction_guard import install as install_prediction_guard
from portable.model import load_model, rms


def path(value):
    supplied = Path(value).expanduser()
    return (supplied if supplied.is_absolute() else ROOT / supplied).resolve()


def sha256(file):
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


model_dir = path(a.model_dir)
manifest = path(a.identity_manifest)
guard = install_prediction_guard()
identity = verify_candidate_identity(model_dir, manifest)
out.mkdir(parents=True, exist_ok=False)
ids = [87555, 130664, 114854]
ids_t = torch.tensor([ids], dtype=torch.long)
torch.set_num_threads(4)
report = {
    "status": "starting", "input_ids": ids, "compute": "CPU_ONLY",
    "reference_semantics": "native full-prefix Torch backbone with layer hooks",
    "model_dir": str(model_dir), "identity_manifest": str(manifest),
    "package_identity": identity["aggregate_sha256"],
    "script_sha256": sha256(Path(__file__).resolve()),
    "reference_source_sha256": sha256(ROOT / "portable/model.py"),
    "prediction_guard": guard, "groups": [],
}


def save():
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def compare(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bad = np.argwhere(delta != 0)
    return {"different_elements": int(len(bad)), "max_abs": float(delta.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(expected[tuple(i)])} for i in bad[:6]]}


save()
native = load_model("cpu")
captures = {}
handles = []
for index, layer in enumerate(native.backbone.layers):
    if index in {*range(3, 48, 4), 22}:
        def keep(module, args, output, i=index):
            value = output[0] if isinstance(output, tuple) else output
            captures[f"layer_{i}"] = value.detach().float().cpu().numpy().copy()
        handles.append(layer.register_forward_hook(keep))
try:
    with torch.inference_mode():
        embed = rms(native.backbone.embed_tokens(ids_t)).detach().float().cpu().numpy().copy()
        ve = native.backbone._value_embeddings(ids_t).detach().float().cpu().numpy().copy()
        final = native.reference_hidden(ids_t).detach().float().cpu().numpy().copy()
finally:
    for handle in handles:
        handle.remove()
captures["embedding_norm"] = embed
captures["ve"] = ve
captures["final_hidden"] = final
required = {"embedding_norm", "ve", "layer_11", "layer_22", "final_hidden",
            *(f"layer_{i}" for i in range(3, 48, 4))}
if set(captures) != required:
    raise ValueError(f"Native capture keys mismatch: {sorted(set(captures) ^ required)}")
np.savez_compressed(out / "native-boundaries.npz", **captures)
report["native_boundary_sha256"] = sha256(out / "native-boundaries.npz")
report["native_shapes"] = {name: list(value.shape) for name, value in captures.items()}
save()
del native
gc.collect()

positions = np.arange(3, dtype=np.int32)
divisor = np.array([1280.], dtype=np.float32)
for group in range(12):
    input_key = "embedding_norm" if group == 0 else f"layer_{4 * group - 1}"
    output_key = "final_hidden" if group == 11 else f"layer_{4 * group + 3}"
    inputs = {
        "x": captures[input_key], "ve": ve, "positions": positions,
        "tap0": embed,
        "tap12": captures["layer_11"] if group >= 3 else embed,
        "tap23": captures["layer_22"] if group >= 6 else embed,
        "norm_divisor": divisor,
    }
    package = model_dir / f"{group}.mlpackage"
    cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    state = cm.make_state()
    before = {item.name: state.read_state(item.name).copy()
              for item in cm.get_spec().description.state}
    got = cm.predict(inputs, state=state)
    expected = captures[output_key]
    row = {"group": group, "input_reference": input_key,
           "output_reference": output_key, "y": compare(got["y"], expected)}
    if group == 5:
        row["tap23_out"] = compare(got["tap23_out"], captures["layer_22"])
    report["groups"].append(row)
    save()
    print(f"group={group} different={row['y']['different_elements']}", flush=True)
    if row["y"]["different_elements"] or row.get("tap23_out", {}).get("different_elements", 0):
        np.savez_compressed(out / "first-failure-inputs.npz", **inputs,
                            actual=got["y"], expected=expected)
        np.savez_compressed(out / "first-failure-state-before.npz", **before)
        report["first_failure"] = {"group": group, "start": 0, "end": 3,
                                    "positions": positions.tolist()}
        report["capture_sha256"] = {
            name: sha256(out / name) for name in
            ("first-failure-inputs.npz", "first-failure-state-before.npz")}
        report["status"] = "first group difference captured"
        save()
        break
    del cm, state, got
    gc.collect()
else:
    report["status"] = "all groups exact on native tiny-3 boundaries"
    save()
