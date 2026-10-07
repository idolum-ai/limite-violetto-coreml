"""Check candidate-4 groups 10 and 11 on independently captured native tiny3 inputs."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
from coreml_port.package_identity import verify_candidate_identity
from coreml_port.prediction_guard import install as install_guard

install_guard()
model_dir = ROOT / "artifacts/04d-cpu/full-candidate-4"
manifest = ROOT / "evidence/04d-cpu/full-candidate-4/package-identity.json"
identity = verify_candidate_identity(model_dir, manifest)
source = ROOT / "evidence/04d-gpu/candidate4-tiny3-firstgroup/native-boundaries.npz"
captures = dict(np.load(source))
out = ROOT / "evidence/04d-gpu/candidate4-tiny3-remaining-groups"
if out.exists():
    raise FileExistsError(out)
out.mkdir()

def compare(actual, expected):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bad = np.argwhere(delta != 0)
    return {"different_elements": int(len(bad)), "max_abs": float(delta.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(expected[tuple(i)])} for i in bad[:8]]}

report = {"status": "starting", "package_identity": identity["aggregate_sha256"],
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "groups": []}
def save():
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
save()

for index in (10, 11):
    inputs = {"x": captures[f"layer_{index * 4 - 1}"], "ve": captures["ve"],
              "positions": np.arange(3, dtype=np.int32), "tap0": captures["embedding_norm"],
              "tap12": captures["layer_11"], "tap23": captures["layer_22"],
              "norm_divisor": np.array([1280.], dtype=np.float32)}
    expected = captures["final_hidden" if index == 11 else "layer_43"]
    cm = ct.models.MLModel(str(model_dir / f"{index}.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_ONLY)
    state = cm.make_state()
    before = {spec.name: state.read_state(spec.name).copy()
              for spec in cm.get_spec().description.state}
    got = cm.predict(inputs, state=state)["y"]
    row = {"group": index, "y": compare(got, expected)}
    report["groups"].append(row)
    save()
    if row["y"]["different_elements"]:
        np.savez_compressed(out / "first-failure-inputs.npz", **inputs,
                            actual=got, expected=expected)
        np.savez_compressed(out / "first-failure-state-before.npz", **before)
        report["first_failure"] = {"group": index, "start": 0, "end": 3,
                                    "positions": [0, 1, 2]}
        report["status"] = "first group difference captured"
        save()
        break
else:
    report["status"] = "both groups exact on native inputs"
    save()
