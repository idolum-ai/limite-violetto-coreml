"""Replay a captured candidate-4 group mismatch and expose BF16 Q sites.

Requires an explicit verified 14-package candidate and a complete locator
capture. Never infers a candidate from the historical default. Instrumented
models can alter compiler fusion, so ordinary replay and Torch equivalence
must pass before any stage comparison is interpreted.
"""
import argparse
import gc
import hashlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", required=True, help="Explicit complete candidate root.")
parser.add_argument("--identity-manifest", required=True, help="Manifest binding that exact candidate root.")
parser.add_argument("--capture", required=True, help="Locator directory with first-failure inputs/state/results.")
parser.add_argument("--attempt", required=True, help="Unused single directory name under evidence/04d-gpu and artifacts/04d-gpu.")
a = parser.parse_args()
attempt = Path(a.attempt)
if attempt.is_absolute() or len(attempt.parts) != 1 or attempt.name in ("", ".", ".."):
    parser.error("--attempt must be one safe directory name")
out = ROOT / "evidence/04d-gpu" / attempt
art = ROOT / "artifacts/04d-gpu" / attempt
if out.exists() or art.exists():
    parser.error("Attempt evidence/artifact path already exists; choose a fresh name")

import coremltools as ct
import numpy as np
import torch
from torch import nn
import coreml_port.graphs as graphs
from coreml_port import dynamic_queries
from coreml_port.ordered_norm import NormWrapped
from coreml_port.package_identity import verify_candidate_identity
from coreml_port.prediction_guard import install as install_prediction_guard
from portable.model import load_model
from scripts.audit_torch_candidate import make_candidate


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
capture = path(a.capture)
source_files = [capture / name for name in
                ("results.json", "first-failure-inputs.npz", "first-failure-state-before.npz")]
for file in source_files:
    if not file.is_file():
        raise FileNotFoundError(f"Capture incomplete: {file}")
meta = json.loads(source_files[0].read_text())
if not isinstance(meta.get("first_failure"), dict) or "group" not in meta["first_failure"]:
    raise ValueError("Capture has no first_failure.group")
index = meta["first_failure"]["group"]
if type(index) is not int or index not in range(12):
    raise ValueError(f"Invalid group index: {index!r}")
saved = dict(np.load(source_files[1]))
states = dict(np.load(source_files[2]))
names = ("x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor")
for name in (*names, "actual", "expected"):
    if name not in saved:
        raise ValueError(f"Capture lacks {name}")
inputs = {name: saved[name] for name in names}
if inputs["positions"].dtype != np.int32:
    raise ValueError("Capture positions must be int32")

# The prediction guard installs the already-qualified cleanup guard first.
# Identity is verified before any Core ML package is loaded.
guard = install_prediction_guard()
identity = verify_candidate_identity(model_dir, manifest)
torch.set_num_threads(4)
out.mkdir(parents=True, exist_ok=False)
art.mkdir(parents=True, exist_ok=False)


def compare(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bad = np.argwhere(difference != 0)
    return {"different_elements": int(len(bad)), "max_abs": float(difference.max()),
            "first_differences": [{"index": i.tolist(),
                                   "actual": float(actual[tuple(i)]),
                                   "expected": float(expected[tuple(i)])}
                                  for i in bad[:5]]}


def match_state(name):
    suffix = name.replace(".", "_").split("_cache_")[-1]
    kind = "k_cache_" if "k_cache_" in name else "v_cache_"
    matches = [value for key, value in states.items() if key.endswith(kind + suffix)]
    if len(matches) != 1:
        raise ValueError(f"Missing/ambiguous saved state for {name}: {len(matches)} matches")
    value = matches[0]
    if not np.array_equal(value, value.astype(np.float16).astype(np.float32)):
        raise ValueError(f"Saved state is not FP16-exact: {name}")
    return value


def write_states(cm, state):
    for spec in cm.get_spec().description.state:
        state.write_state(spec.name, match_state(spec.name))


report = {
    "status": "starting", "compute": "CPU_ONLY", "capture": str(capture),
    "capture_files_sha256": {str(file.relative_to(ROOT)): sha256(file) for file in source_files},
    "model_dir": str(model_dir), "identity_manifest": str(manifest),
    "package_identity": identity["aggregate_sha256"], "group": index,
    "positions": inputs["positions"].tolist(), "prediction_guard": guard,
    "source_sha256": {str(file.relative_to(ROOT)): sha256(file) for file in
                      (Path(__file__).resolve(), ROOT / "coreml_port/graphs.py",
                       ROOT / "coreml_port/dynamic_queries.py", ROOT / "coreml_port/source_exp.py",
                       ROOT / "coreml_port/ordered_attention.py", ROOT / "coreml_port/ordered_linear.py",
                       ROOT / "coreml_port/ordered_norm.py", ROOT / "coreml_port/ordered_xsa.py",
                       ROOT / "coreml_port/prediction_guard.py", ROOT / "scripts/audit_torch_candidate.py")},
}


def save():
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


save()
ordinary_package = model_dir / f"{index}.mlpackage"
ordinary = ct.models.MLModel(str(ordinary_package), compute_units=ct.ComputeUnit.CPU_ONLY)
state = ordinary.make_state()
write_states(ordinary, state)
ordinary_y = ordinary.predict(inputs, state=state)["y"]
report["ordinary_replay_vs_saved_actual"] = compare(ordinary_y, saved["actual"])
save()
del ordinary, state
gc.collect()
if report["ordinary_replay_vs_saved_actual"]["different_elements"]:
    report["status"] = "ordinary replay mismatch; stage interpretation stopped"
    save()
    raise RuntimeError("Ordinary candidate replay does not reproduce saved actual output")

model = load_model("cpu")
_, groups, _ = make_candidate(model, "fp16", False, "ordered", "ordered", "ordered", "lookup", "source")
group = groups[index]
del model, groups
gc.collect()
# Match --queries dynamic-product in the candidate-4 converter exactly.
graphs.prepare_queries = dynamic_queries.prepare
graphs.attention_scores = dynamic_queries.scores
graphs.attention_product = dynamic_queries.product
base_q = graphs.Q


class Diagnostic(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = NormWrapped(group)

    def forward(self, *args):
        records = []
        sites = []

        def capture_q(value):
            result = base_q(value)
            records.append(result + 0.)
            caller = inspect.currentframe().f_back
            frame = caller
            layer = None
            while frame:
                obj = frame.f_locals.get("self")
                if hasattr(obj, "i"):
                    layer = obj.i
                    break
                frame = frame.f_back
            sites.append({"function": caller.f_code.co_name,
                          "line": caller.f_lineno, "layer": layer})
            return result

        graphs.Q = capture_q
        try:
            y, tap = self.inner(*args)
        finally:
            graphs.Q = base_q
        self.sites = sites
        return (y, *records)


graph = Diagnostic().eval()
args = tuple(torch.from_numpy(inputs[name]) for name in names)


def reset_torch_state():
    for name, value in graph.named_buffers():
        if "_cache_" in name:
            value.copy_(torch.from_numpy(match_state(name)))


with torch.inference_mode():
    reset_torch_state()
    expected = graph(*args)
labels = graph.sites
report["torch_with_actual_state_vs_native"] = compare(expected[0].numpy(), saved["expected"])
report["torch_with_actual_state_vs_coreml"] = compare(expected[0].numpy(), ordinary_y)
save()
if report["torch_with_actual_state_vs_native"]["different_elements"]:
    report["status"] = "Torch candidate mismatch to native; stage interpretation stopped"
    save()
    raise RuntimeError("Candidate-4 Torch graph does not match native output on captured state")

reset_torch_state()
traced = torch.jit.trace(graph, args, check_trace=False)
output_names = ["y"] + [f"round_{i}" for i in range(len(expected) - 1)]
converted = ct.convert(
    traced,
    inputs=[ct.TensorType(name=name, shape=inputs[name].shape,
                          dtype=np.int32 if name == "positions" else np.float32)
            for name in names],
    outputs=[ct.TensorType(name=name, dtype=np.float32) for name in output_names],
    states=[ct.StateType(name=name, wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16))
            for name, value in graph.named_buffers() if "_cache_" in name],
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
package = art / "stages.mlpackage"
converted.save(str(package))
cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
state = cm.make_state()
write_states(cm, state)
got = cm.predict(inputs, state=state)
keys = [item.name for item in cm.get_spec().description.output]
if len(keys) != len(expected):
    raise ValueError(f"Stage output count changed: {len(keys)} vs {len(expected)}")
rows = []
arrays = {}
for j, (key, want) in enumerate(zip(keys, expected)):
    actual = got[key]
    expected_array = want.numpy()
    row = {"output": key, **compare(actual, expected_array)}
    if j:
        row["site"] = labels[j - 1]
    rows.append(row)
    arrays[key + "_actual"] = actual
    arrays[key + "_expected"] = expected_array
report["stages"] = rows
report["first_differing_stage"] = next((row for row in rows[1:] if row["different_elements"]), None)
report["status"] = "complete"
save()
np.savez_compressed(out / "stages.npz", **arrays)
print("first differences", [(row["output"], row.get("site"), row["different_elements"])
                            for row in rows if row["different_elements"]][:12], flush=True)
