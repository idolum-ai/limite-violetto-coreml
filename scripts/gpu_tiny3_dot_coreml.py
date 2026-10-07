"""Core ML CPU q·k geometry probe using tiny3 layer-32 native operands."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coremltools as ct
import numpy as np
import torch
from torch import nn
from coreml_port.cleanup_guard import install
from portable.model import load_model

install()
torch.set_num_threads(4)
OUT = ROOT / "evidence/04d-gpu/tiny3-dot-coreml"
ART = ROOT / "artifacts/04d-gpu/tiny3-dot-coreml"
OUT.mkdir(parents=True, exist_ok=False)
ART.mkdir(parents=True, exist_ok=False)
ids = [87555, 130664, 114854]
report = {"ids": ids, "compute_units": "CPU_ONLY", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "reference_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
              ROOT / "portable/model.py", ROOT / "sources/value-model/modeling_backbone.py")}, "experiments": []}
def save():
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()

model = load_model("cpu")
attention = model.backbone.layers[32].self_attn
captured = {}
original_project = attention.project
def project(hidden, qkv_scale, value_embeds, cos, sin):
    result = original_project(hidden, qkv_scale, value_embeds, cos, sin)
    captured["q"] = result[0].detach().clone()
    captured["k"] = result[1].detach().clone()
    return result
attention.project = project
try:
    with torch.inference_mode():
        model.reference_hidden(torch.tensor([ids], dtype=torch.long))
finally:
    attention.project = original_project

q3 = captured["q"].float().permute(0, 2, 1, 3).contiguous()
k3 = captured["k"].repeat_interleave(5, dim=2).float().permute(0, 2, 3, 1).contiguous()
native_dot = torch.einsum("bthd,bshd->bhts", captured["q"].float(),
                          captured["k"].repeat_interleave(5, dim=2).float())
native_scores = native_dot * .1

class Dot(nn.Module):
    def forward(self, q, k):
        dot = q @ k
        return dot, dot * .1

def stats(actual, target):
    diff = np.abs(actual.astype(np.float64) - target.astype(np.float64))
    indexes = np.argwhere(diff != 0)
    return {"different_elements": int(len(indexes)), "max_abs": float(diff.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(target[tuple(i)])} for i in indexes[:8]]}

for rows, cols, real_position in ((3, 3, "front"), (3, 2048, "front"),
                                  (16, 3, "back"), (32, 3, "front"),
                                  (32, 2048, "front")):
    if rows == 3:
        q = q3
        selection = slice(0, 3)
    elif real_position == "back":
        q = torch.cat((torch.zeros(1, 10, rows - 3, 128), q3), dim=2)
        selection = slice(rows - 3, rows)
    else:
        q = torch.cat((q3, torch.zeros(1, 10, rows - 3, 128)), dim=2)
        selection = slice(0, 3)
    k = k3 if cols == 3 else torch.cat((k3, torch.zeros(1, 10, 128, cols - 3)), dim=-1)
    with torch.inference_mode():
        torch_dot, torch_scores = Dot().eval()(q, k)
    trace = torch.jit.trace(Dot().eval(), (q, k), check_trace=False)
    converted = ct.convert(trace,
        inputs=[ct.TensorType(name="q", shape=q.shape, dtype=np.float32),
                ct.TensorType(name="k", shape=k.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="dot", dtype=np.float32),
                 ct.TensorType(name="scores", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
    name = f"q{rows}-k{cols}-{real_position}"
    package = ART / (name + ".mlpackage")
    converted.save(str(package))
    cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    actual = cm.predict({"q": q.numpy(), "k": k.numpy()})
    row = {"name": name, "q_shape": list(q.shape), "k_shape": list(k.shape),
           "package_files_sha256": {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(package.rglob("*")) if p.is_file()}}
    for output_name, torch_value, native_value in (("dot", torch_dot, native_dot),
                                                    ("scores", torch_scores, native_scores)):
        v = actual[output_name][:, :, selection, :3]
        t = torch_value[:, :, selection, :3].numpy()
        n = native_value.numpy()
        row[output_name] = {"torch_vs_native": stats(t, n),
                            "coreml_vs_torch": stats(v, t),
                            "coreml_vs_native": stats(v, n)}
    report["experiments"].append(row)
    print(name, row["scores"]["torch_vs_native"]["different_elements"],
          row["scores"]["coreml_vs_torch"]["different_elements"],
          row["scores"]["coreml_vs_native"]["different_elements"], flush=True)
    save()
