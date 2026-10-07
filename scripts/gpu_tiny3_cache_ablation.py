"""Test whether the FP16 KV cache causes tiny_33[:3] layer-32 mismatch."""
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import coreml_port.graphs as graphs
import coreml_port.ordered_attention as attention
from coreml_port.bf16_gelu import BF16Gelu
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import raw_norm
from coreml_port.ordered_xsa import ordered_xsa_dot, ordered_xsa_normalize
from coreml_port.source_exp import exp
from portable.model import load_model
from scripts.probe_full_model import metrics

torch.set_num_threads(4)
graphs.linear = linear
graphs.softmax = attention.softmax
attention.exponential = exp
graphs.xsa_normalize = ordered_xsa_normalize
graphs.xsa_dot = ordered_xsa_dot
graphs.gelu_factory = BF16Gelu

def norm(value):
    divisor = torch.tensor([float(value.shape[-1])], dtype=value.dtype)
    return graphs.Q(raw_norm(value, divisor))

graphs.norm = norm
ids = [87555, 130664, 114854]
OUT = ROOT / "evidence/04d-gpu/tiny3-cache-ablation"
OUT.mkdir(parents=True, exist_ok=False)
report = {"ids": ids, "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    ROOT / "scripts/gpu_tiny3_cache_ablation.py", ROOT / "coreml_port/graphs.py",
    ROOT / "coreml_port/ordered_attention.py", ROOT / "coreml_port/source_exp.py",
    ROOT / "portable/model.py")}, "layers": []}

def save():
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

save()
m = load_model("cpu")
ref = {}
handles = []
for i in range(32, 36):
    def keep(module, args, output, index=i):
        value = output[0] if isinstance(output, tuple) else output
        ref[index] = value.detach().clone()
    handles.append(m.backbone.layers[i].register_forward_hook(keep))
with torch.inference_mode():
    m.reference_hidden(torch.tensor([ids], dtype=torch.long))
for h in handles:
    h.remove()

with torch.inference_mode():
    x, ve = graphs.Embedding(m).eval()(torch.tensor([ids], dtype=torch.long))
    tap0 = x.clone()
    tap12 = x.clone()
    tap23 = x.clone()
    pos = torch.arange(3, dtype=torch.int32)
    for g in range(8):
        group = graphs.Group(m, 4 * g).eval()
        x, tap23 = group(x, ve, pos, tap0, tap12, tap23)
        if g == 2:
            tap12 = x
        del group
        gc.collect()
    for cache_dtype in (torch.float16, torch.float32):
        original_finish = graphs.Layer.finish
        captures = {}
        def intercept(self, x, residual, q, keys, values, v, positions, slots):
            out = original_finish(self, x, residual, q, keys, values, v, positions, slots)
            if 32 <= self.i <= 35:
                captures[self.i] = out.detach().clone()
            return out
        graphs.Layer.finish = intercept
        try:
            group = graphs.Group(m, 32).eval()
            if cache_dtype == torch.float32:
                for name, value in list(group.named_buffers()):
                    if name.startswith(("k_cache_", "v_cache_")):
                        setattr(group, name, torch.zeros_like(value, dtype=torch.float32))
            group(x, ve, pos, tap0, tap12, tap23)
        finally:
            graphs.Layer.finish = original_finish
        for i in range(32, 36):
            got = captures[i]
            want = ref[i]
            diff = torch.nonzero(got != want)
            row = {"cache_dtype": str(cache_dtype), "layer": i, **metrics(got, want),
                   "different_elements": int(diff.shape[0]),
                   "first_differences": [{"index": ix.tolist(), "got": float(got[tuple(ix)]),
                                          "want": float(want[tuple(ix)])} for ix in diff[:8]]}
            report["layers"].append(row)
            print(cache_dtype, i, row["different_elements"], row["max_abs"], flush=True)
            save()
        del group
        gc.collect()
