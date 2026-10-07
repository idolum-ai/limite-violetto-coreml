"""Compare one candidate-3 Core ML group with an exact Torch prefix.

Run only after the full candidate gate identifies a failing prompt/group. Each
preceding group is run in Torch, so the selected Core ML group receives matched
reference inputs and warmed state across the chosen prompt's chunks.
"""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--candidate-root", required=True, type=Path)
parser.add_argument("--group", required=True, type=int, choices=range(12))
parser.add_argument("--prompt", required=True, type=int, choices=range(3))
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

import coremltools as ct
import numpy as np
import torch
import coreml_port.graphs as graphs
import coreml_port.ordered_attention as attention
import coreml_port.source_exp as source_exp
from coreml_port.bf16_gelu import BF16Gelu
from coreml_port.cleanup_guard import install
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import NormWrapped
from coreml_port.ordered_xsa import ordered_xsa_dot, ordered_xsa_normalize
from portable.model import load_model
from scripts.probe_full_model import metrics

install()
torch.set_num_threads(4)
graphs.linear = linear
graphs.softmax = attention.softmax
attention.exponential = source_exp.exp
graphs.xsa_normalize = ordered_xsa_normalize
graphs.xsa_dot = ordered_xsa_dot
graphs.gelu_factory = BF16Gelu
DIVISOR = torch.tensor([1280.], dtype=torch.float32)
candidate_root = args.candidate_root.resolve()
package = candidate_root / f"{args.group}.mlpackage"
if not package.exists():
    raise FileNotFoundError(package)
output = args.output.resolve()
if output.exists():
    raise FileExistsError(output)
output.parent.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected):
    actual = torch.from_numpy(actual)
    row = metrics(actual, expected)
    row["different_elements"] = int(torch.count_nonzero(actual != expected))
    difference = actual - expected
    ix = torch.nonzero(difference != 0)
    row["first_differences"] = [
        {"index": i.tolist(), "actual": float(actual[tuple(i)]), "expected": float(expected[tuple(i)])}
        for i in ix[:8]
    ]
    return row


report = {
    "script_sha256": sha(Path(__file__)),
    "sources": {str(p.relative_to(ROOT)): sha(p) for p in (
        ROOT / "coreml_port/graphs.py", ROOT / "coreml_port/ordered_attention.py",
        ROOT / "coreml_port/source_exp.py", ROOT / "coreml_port/ordered_linear.py",
        ROOT / "coreml_port/ordered_norm.py", ROOT / "coreml_port/ordered_xsa.py",
        ROOT / "coreml_port/bf16_gelu.py")},
    "candidate_root": str(candidate_root), "group": args.group,
    "prompt": args.prompt, "compute_units": "CPU_ONLY", "steps": [],
    "package_files_sha256": {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*")) if p.is_file()},
}
output.write_text(json.dumps(report, indent=2) + "\n")

case = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][args.prompt]
ids = case["input_ids"]
spans = [(start, min(start + 32, len(ids)), ids[start:start + 32], "prefill")
         for start in range(0, len(ids), 32)]
spans.append((len(ids), len(ids) + 1, [16], "decode"))
model = load_model()
embedding = NormWrapped(graphs.Embedding(model).eval())
steps = []
with torch.inference_mode():
    for start, end, chunk, kind in spans:
        x, ve = embedding(torch.tensor([chunk], dtype=torch.int32), DIVISOR)
        steps.append({"start": start, "end": end, "kind": kind,
                      "x": x, "ve": ve, "tap0": x.clone(), "tap12": x.clone(),
                      "tap23": x.clone(), "positions": torch.arange(start, end, dtype=torch.int32)})
del embedding
gc.collect()

with torch.inference_mode():
    for index in range(args.group + 1):
        group = NormWrapped(graphs.Group(model, 4 * index).eval()).eval()
        for step in steps:
            step["input"] = tuple(step[name].clone() for name in
                                  ("x", "ve", "positions", "tap0", "tap12", "tap23")) + (DIVISOR,)
            y, tap = group(*step["input"])
            if index == args.group:
                step["expected_y"] = y.clone()
                step["expected_tap23"] = tap.clone()
            step["x"] = y.clone()
            step["tap23"] = tap.clone()
            if index == 2:
                step["tap12"] = y.clone()
        del group
        gc.collect()
del model
gc.collect()

cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
state = cm.make_state()
input_names = ("x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor")
for step in steps:
    payload = {name: value.numpy() for name, value in zip(input_names, step["input"])}
    got = cm.predict(payload, state=state)
    row = {"start": step["start"], "end": step["end"], "kind": step["kind"],
           "y": compare(got["y"], step["expected_y"]),
           "tap23_out": compare(got["tap23_out"], step["expected_tap23"])}
    report["steps"].append(row)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.group, args.prompt, row["start"], row["end"],
          row["y"]["different_elements"], row["tap23_out"]["different_elements"], flush=True)
report["all_exact"] = all(row[name]["different_elements"] == 0 for row in report["steps"] for name in ("y", "tap23_out"))
output.write_text(json.dumps(report, indent=2) + "\n")
