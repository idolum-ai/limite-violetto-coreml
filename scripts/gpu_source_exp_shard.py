"""Convert and validate source-exponential corrected shard 0 on canonical prompts."""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
import torch

import coreml_port.graphs as graphs
import coreml_port.ordered_attention as attention
import coreml_port.source_exp as source_exp
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
graphs.xsa_normalize = ordered_xsa_normalize
graphs.xsa_dot = ordered_xsa_dot

OUT = ROOT / "evidence/04d-gpu/source-exp/shard0"
ART = ROOT / "artifacts/04d-gpu/source-exp/shard0"
OUT.mkdir(parents=True, exist_ok=False)
ART.mkdir(parents=True, exist_ok=False)
PACKAGE = ART / "0.mlpackage"
DIVISOR = torch.tensor([1280.], dtype=torch.float32)
report = {
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        ROOT / "coreml_port/graphs.py", ROOT / "coreml_port/ordered_attention.py",
        ROOT / "coreml_port/source_exp.py", ROOT / "coreml_port/ordered_linear.py",
        ROOT / "coreml_port/ordered_norm.py", ROOT / "coreml_port/ordered_xsa.py")},
    "compute_units": "CPU_ONLY", "capacity": 2048, "prompts": [], "torch_comparison": [],
}
REPORT = OUT / "summary.json"


def save():
    REPORT.write_text(json.dumps(report, indent=2) + "\n")


def compare(actual, expected):
    if isinstance(actual, np.ndarray):
        actual = torch.from_numpy(actual)
    row = metrics(actual, expected)
    row["different_elements"] = int(torch.count_nonzero(actual != expected))
    return row


model = load_model()
embedding = NormWrapped(graphs.Embedding(model).eval())
group = NormWrapped(graphs.Group(model, 0).eval()).eval()
del model
gc.collect()
cases = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"]
decode_token = 16
report["decode_token"] = decode_token
samples = []
with torch.inference_mode():
    # Native source Torch exponential is the reference for the corrected graph.
    attention.exponential = torch.exp
    for index, case in enumerate(cases):
        for name, value in group.named_buffers():
            if "_cache_" in name:
                value.zero_()
        ids = case["input_ids"]
        spans = [(start, min(start + 32, len(ids)), ids[start:start + 32], "prefill")
                 for start in range(0, len(ids), 32)]
        spans.append((len(ids), len(ids) + 1, [decode_token], "decode"))
        prompt = {"index": index, "prompt_tokens": len(ids), "steps": []}
        report["prompts"].append(prompt)
        for start, end, chunk, kind in spans:
            x, ve = embedding(torch.tensor([chunk], dtype=torch.int32), DIVISOR)
            pos = torch.arange(start, end, dtype=torch.int32)
            args = (x, ve, pos, x.clone(), x.clone(), x.clone(), DIVISOR)
            y, tap = group(*args)
            samples.append((index, start, end, kind, tuple(v.clone() for v in args), y.clone(), tap.clone()))
            prompt["steps"].append({"kind": kind, "start": start, "end": end})
    attention.exponential = source_exp.exp
    for index, case in enumerate(cases):
        for name, value in group.named_buffers():
            if "_cache_" in name:
                value.zero_()
        for i, start, end, kind, args, expected, tap_expected in samples:
            if i != index:
                continue
            y, tap = group(*args)
            report["torch_comparison"].append({"prompt": i, "kind": kind, "start": start, "end": end,
                "y": compare(y, expected), "tap23_out": compare(tap, tap_expected)})
            print("torch", i, start, end, report["torch_comparison"][-1]["y"]["different_elements"], flush=True)
save()

# Compile the same group with the source exponential and the exact candidate-2
# norm-divisor wrapper/input contract.
for name, value in group.named_buffers():
    if "_cache_" in name:
        value.zero_()
x = torch.randn(1, 3, 1280) * .1
sample = (x, torch.zeros(1, 3, 2, 128), torch.arange(3, dtype=torch.int32),
          x.clone(), x.clone(), x.clone(), DIVISOR)
with torch.inference_mode():
    traced = torch.jit.trace(group, sample, check_trace=False)
seq = ct.RangeDim(1, 32, default=3)
def tensor(name, shape, dtype=np.float32):
    return ct.TensorType(name=name, shape=shape, dtype=dtype)
inputs = [tensor("x", (1, seq, 1280)), tensor("ve", (1, seq, 2, 128)),
          tensor("positions", (seq,), np.int32)]
inputs += [tensor(name, (1, seq, 1280)) for name in ("tap0", "tap12", "tap23")]
inputs += [tensor("norm_divisor", (1,))]
states = [ct.StateType(name=name, wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16))
          for name, value in group.named_buffers() if "_cache_" in name]
began = time.time()
converted = ct.convert(traced, inputs=inputs,
    outputs=[tensor("y", None), tensor("tap23_out", None)], states=states,
    convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
converted.save(str(PACKAGE))
report["conversion_seconds"] = time.time() - began
report["package_files_sha256"] = {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(PACKAGE.rglob("*")) if p.is_file()}
save()

cm = ct.models.MLModel(str(PACKAGE), compute_units=ct.ComputeUnit.CPU_ONLY)
for prompt in report["prompts"]:
    state = cm.make_state()
    for i, start, end, kind, args, expected, tap_expected in samples:
        if i != prompt["index"]:
            continue
        payload = {name: value.numpy() for name, value in zip(
            ("x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor"), args)}
        got = cm.predict(payload, state=state)
        step = next(s for s in prompt["steps"] if s["start"] == start and s["end"] == end)
        step["y"] = compare(got["y"], expected)
        step["tap23_out"] = compare(got["tap23_out"], tap_expected)
        print("coreml", i, start, end, step["y"]["different_elements"], flush=True)
        save()
report["torch_all_exact"] = all(r[n]["different_elements"] == 0 for r in report["torch_comparison"] for n in ("y", "tap23_out"))
report["coreml_all_exact"] = all(s[n]["different_elements"] == 0 for p in report["prompts"] for s in p["steps"] for n in ("y", "tap23_out"))
save()
print("all_exact", report["torch_all_exact"], report["coreml_all_exact"], flush=True)
