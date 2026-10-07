"""Verify actual candidate-2 embedding/shard0 interfaces and CPU numerics."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
import torch

import coreml_port.graphs as graphs
from coreml_port.cleanup_guard import install
from coreml_port.ordered_attention import softmax
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import NormWrapped
from coreml_port.ordered_xsa import ordered_xsa_dot, ordered_xsa_normalize
from portable.generate import tokenizer
from portable.model import load_model
from scripts.probe_full_model import metrics

install()
torch.set_num_threads(4)
graphs.linear = linear
graphs.softmax = softmax
graphs.xsa_normalize = ordered_xsa_normalize
graphs.xsa_dot = ordered_xsa_dot

BASE = ROOT / "artifacts/04d-cpu/full-candidate-2"
OUT = ROOT / "evidence/04d-gpu/integrated-shard"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "results.json"
if REPORT.exists():
    raise RuntimeError(f"Refusing to overwrite existing evidence: {REPORT}")
SOURCE = ROOT / "evidence/04d-cpu/full-candidate-2/source-identity.json"
DIVISOR = np.array([1280.], dtype=np.float32)


def feature(feature):
    typ = feature.type
    if typ.HasField("multiArrayType"):
        ma = typ.multiArrayType
        return {"name": feature.name, "shape": list(ma.shape), "dtype_enum": int(ma.dataType)}
    return {"name": feature.name, "type": typ.WhichOneof("Type")}


def interface(model):
    spec = model.get_spec()
    return {
        "inputs": [feature(x) for x in spec.description.input],
        "outputs": [feature(x) for x in spec.description.output],
        "states": [feature(x) for x in spec.description.state],
    }


def compare(actual, expected):
    row = metrics(torch.from_numpy(actual), expected)
    row["different_elements"] = int(np.count_nonzero(actual != expected.numpy()))
    return row


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "candidate_root": str(BASE.relative_to(ROOT)),
    "compute_units": "CPU_ONLY",
    "source_identity_sha256": sha(SOURCE),
    "source_identity": json.loads(SOURCE.read_text()),
    "sources": {str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT / "scripts/gpu_integrated_shard.py", ROOT / "coreml_port/graphs.py",
        ROOT / "coreml_port/ordered_linear.py", ROOT / "coreml_port/ordered_attention.py",
        ROOT / "coreml_port/ordered_norm.py", ROOT / "coreml_port/ordered_xsa.py")},
    "packages": {}, "prompts": [],
}
for name in ("embedding", "0"):
    path = BASE / f"{name}.mlpackage"
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    report["packages"][name] = {
        "interface": interface(model),
        "manifest_sha256": sha(path / "Manifest.json"),
        "model_sha256": sha(path / "Data/com.apple.CoreML/model.mlmodel"),
        "weights_sha256": sha(path / "Data/com.apple.CoreML/weights/weight.bin"),
    }
    if name == "embedding":
        embedding_ml = model
    else:
        group_ml = model

for name in ("embedding", "0"):
    inputs = {x["name"]: x for x in report["packages"][name]["interface"]["inputs"]}
    assert inputs["norm_divisor"]["shape"] == [1]
    assert inputs["norm_divisor"]["dtype_enum"] == 65568  # FLOAT32
states = {x["name"] for x in report["packages"]["0"]["interface"]["states"]}
assert states == {f"inner_{kind}_cache_{j}" for kind in ("k", "v") for j in range(4)}
report["interface_verified"] = True
REPORT.write_text(json.dumps(report, indent=2) + "\n")

model = load_model()
embedding_torch = NormWrapped(graphs.Embedding(model).eval())
group_torch = NormWrapped(graphs.Group(model, 0).eval())
cases = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"]
decode_token = tokenizer().encode("17 * 23 = 391.", add_special_tokens=False)[0]
report["decode_token"] = int(decode_token)

with torch.inference_mode():
    for index, case in enumerate(cases):
        for name, value in group_torch.named_buffers():
            if "_cache_" in name:
                value.zero_()
        state = group_ml.make_state()
        ids = case["input_ids"]
        spans = [(start, min(start + 32, len(ids)), ids[start:start + 32], "prefill")
                 for start in range(0, len(ids), 32)]
        spans.append((len(ids), len(ids) + 1, [int(decode_token)], "decode"))
        prompt = {"index": index, "prompt_tokens": len(ids), "steps": []}
        report["prompts"].append(prompt)
        for start, end, chunk, kind in spans:
            ids_t = torch.tensor([chunk], dtype=torch.int32)
            ref_x, ref_ve = embedding_torch(ids_t, torch.from_numpy(DIVISOR))
            emb = embedding_ml.predict({"ids": ids_t.numpy(), "norm_divisor": DIVISOR})
            positions = torch.arange(start, end, dtype=torch.int32)
            ref_y, ref_tap = group_torch(ref_x, ref_ve, positions,
                                         ref_x.clone(), ref_x.clone(), ref_x.clone(),
                                         torch.from_numpy(DIVISOR))
            payload = {"x": emb["x"], "ve": emb["ve"], "positions": positions.numpy(),
                       "tap0": emb["x"], "tap12": emb["x"], "tap23": emb["x"],
                       "norm_divisor": DIVISOR}
            got = group_ml.predict(payload, state=state)
            step = {
                "kind": kind, "start": start, "end": end,
                "embedding_x": compare(emb["x"], ref_x),
                "embedding_ve": compare(emb["ve"], ref_ve),
                "shard_y": compare(got["y"], ref_y),
                "shard_tap23": compare(got["tap23_out"], ref_tap),
            }
            prompt["steps"].append(step)
            print("prompt", index, kind, start, end,
                  {name: row["different_elements"] for name, row in step.items() if isinstance(row, dict)}, flush=True)
            REPORT.write_text(json.dumps(report, indent=2) + "\n")
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
report["all_exact"] = all(row["different_elements"] == 0 for p in report["prompts"] for s in p["steps"] for row in (s["embedding_x"], s["embedding_ve"], s["shard_y"], s["shard_tap23"]))
REPORT.write_text(json.dumps(report, indent=2) + "\n")
print("all_exact", report["all_exact"], flush=True)
