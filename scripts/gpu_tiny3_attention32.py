"""Compare reference and corrected-graph layer-32 attention stages for tiny3."""
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

torch.set_num_threads(4)
graphs.linear = linear
graphs.softmax = attention.softmax
attention.exponential = exp
graphs.xsa_normalize = ordered_xsa_normalize
graphs.xsa_dot = ordered_xsa_dot
graphs.gelu_factory = BF16Gelu
def norm(value):
    return graphs.Q(raw_norm(value, torch.tensor([float(value.shape[-1])], dtype=value.dtype)))
graphs.norm = norm
OUT = ROOT / "evidence/04d-gpu/tiny3-attention32"
OUT.mkdir(parents=True, exist_ok=False)
ids = [87555, 130664, 114854]
report = {"ids": ids, "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    ROOT / "scripts/gpu_tiny3_attention32.py", ROOT / "coreml_port/graphs.py",
    ROOT / "coreml_port/ordered_attention.py", ROOT / "coreml_port/source_exp.py",
    ROOT / "portable/model.py", ROOT / "sources/value-model/modeling_backbone.py")}, "stages": {}}
def save():
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
def compare(name, got, want):
    g = got.detach().float().cpu()
    w = want.detach().float().cpu()
    d = (g - w).abs()
    ix = torch.nonzero(d != 0)
    row = {"shape": list(g.shape), "different_elements": int(ix.shape[0]),
           "max_abs": float(d.max()),
           "first_differences": [{"index": i.tolist(), "got": float(g[tuple(i)]),
                                  "want": float(w[tuple(i)])} for i in ix[:8]]}
    report["stages"][name] = row
    print(name, row["different_elements"], row["max_abs"], flush=True)
    save()

save()
model = load_model("cpu")
attn = model.backbone.layers[32].self_attn
reference = {}
original_eager = attn._eager
def eager(q, k, v, seq_ids):
    reference.update(q=q.clone(), k=k.clone(), v=v.clone())
    result = original_eager(q, k, v, seq_ids)
    reference["weighted"] = result.clone()
    return result
attn._eager = eager
def keep_input(module, args):
    reference["input"] = args[1].clone()
handle = model.backbone.layers[32].register_forward_pre_hook(keep_input)
try:
    with torch.inference_mode():
        model.reference_hidden(torch.tensor([ids], dtype=torch.long))
finally:
    attn._eager = original_eager
    handle.remove()

with torch.inference_mode():
    x, ve = graphs.Embedding(model).eval()(torch.tensor([ids], dtype=torch.long))
    tap0 = x.clone(); tap12 = x.clone(); tap23 = x.clone()
    positions = torch.arange(3, dtype=torch.int32)
    for g in range(8):
        group = graphs.Group(model, 4 * g).eval()
        x, tap23 = group(x, ve, positions, tap0, tap12, tap23)
        if g == 2: tap12 = x
        del group
        gc.collect()
    graph = {}
    original_project = graphs.Layer.project
    original_finish = graphs.Layer.finish
    def project(self, x, ve, cos, sin):
        result = original_project(self, x, ve, cos, sin)
        if self.i == 32:
            graph["input"] = x.clone()
            graph.update(q=result[0].clone(), k=result[1].clone(), v=result[2].clone())
        return result
    def finish(self, x, residual, q, keys, values, v, positions, slots):
        if self.i == 32:
            count = q.shape[1]
            padded_q = torch.cat((q, torch.zeros(1, 31, 10, 128)), dim=1)[:, :32]
            padded_pos = torch.cat((positions, torch.zeros(31, dtype=torch.int32)))[:32]
            k = keys.repeat_interleave(5, dim=2).permute(0, 2, 3, 1)
            val = values.repeat_interleave(5, dim=2).permute(0, 2, 1, 3)
            scores = torch.matmul(padded_q.permute(0, 2, 1, 3), k) * .1
            mask = slots[None, :] <= padded_pos[:, None]
            if self.window >= 0:
                mask = mask & (slots[None, :] >= padded_pos[:, None] - self.window)
            weights = graphs.softmax(scores.masked_fill(~mask[None, None], float("-inf")))
            graph["scores"] = scores[:, :, :count, :count].clone()
            graph["weights"] = weights[:, :, :count, :count].clone()
            graph["weighted_raw"] = torch.matmul(weights, val).permute(0, 2, 1, 3)[:, :count].clone()
            graph["weighted"] = graphs.Q(graph["weighted_raw"])
        return original_finish(self, x, residual, q, keys, values, v, positions, slots)
    graphs.Layer.project = project
    graphs.Layer.finish = finish
    try:
        group = graphs.Group(model, 32).eval()
        group(x, ve, positions, tap0, tap12, tap23)
    finally:
        graphs.Layer.project = original_project
        graphs.Layer.finish = original_finish
    for key in ("input", "q", "k", "v", "weighted"):
        compare(key, graph[key], reference[key])
    # Reference's unpadded attention arithmetic, computed from the captured
    # native project outputs. Verify its weighted output against _eager first.
    q = reference["q"].float().permute(0, 2, 1, 3)
    k = reference["k"].repeat_interleave(5, dim=2).float().permute(0, 2, 3, 1)
    val = reference["v"].repeat_interleave(5, dim=2).float().permute(0, 2, 1, 3)
    scores = (q @ k) * .1
    mask = torch.arange(3)[None, :] <= torch.arange(3)[:, None]
    if attn.window >= 0:
        mask = mask & (torch.arange(3)[None, :] >= torch.arange(3)[:, None] - attn.window)
    masked = scores.masked_fill(~mask[None, None], float("-inf"))
    native_weights = masked.softmax(-1)
    native_weighted = (native_weights @ val).permute(0, 2, 1, 3).to(reference["v"].dtype)
    compare("recomputed_reference_weighted", native_weighted, reference["weighted"])
    compare("scores", graph["scores"], scores)
    compare("weights", graph["weights"], native_weights)
    compare("weighted_using_native_weights", (native_weights @ val).permute(0, 2, 1, 3).to(reference["v"].dtype), reference["weighted"])
