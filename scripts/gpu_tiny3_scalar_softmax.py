"""Test whether Torch's three-key scalar softmax order fixes tiny_33[:3]."""
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
from coreml_port.source_exp import exp as source_exp
from portable.model import load_model
from scripts.audit_torch_candidate import reference_prefix, score_metrics
from scripts.probe_full_model import metrics

torch.set_num_threads(4)
OUT = ROOT / "evidence/04d-gpu/tiny3-scalar-softmax"
OUT.mkdir(parents=True, exist_ok=False)
REPORT = OUT / "summary.json"
case = next(c for c in json.loads((ROOT / "evidence/04d-audit/candidate3-partitions-host/results.json").read_text())["case_plan"]
            if c["name"] == "single_prefill_3")
ids = case["input_ids"]
assert ids == [87555, 130664, 114854]
report = {
    "fixture": case, "scope": "pure Torch full 48-layer corrected graph versus pinned CPU reference",
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        ROOT / "scripts/gpu_tiny3_scalar_softmax.py", ROOT / "coreml_port/graphs.py",
        ROOT / "coreml_port/ordered_attention.py", ROOT / "coreml_port/source_exp.py",
        ROOT / "coreml_port/ordered_linear.py", ROOT / "coreml_port/ordered_norm.py",
        ROOT / "coreml_port/ordered_xsa.py", ROOT / "coreml_port/bf16_gelu.py",
        ROOT / "portable/model.py")},
    "variants": {},
}

def save():
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

save()
graphs.linear = linear
graphs.xsa_normalize = ordered_xsa_normalize
graphs.xsa_dot = ordered_xsa_dot
graphs.gelu_factory = BF16Gelu
attention.exponential = source_exp

def ordered_graph_norm(value):
    divisor = torch.tensor([float(value.shape[-1])], dtype=value.dtype, device=value.device)
    return graphs.Q(raw_norm(value, divisor))

graphs.norm = ordered_graph_norm

def scalar_three_softmax(x):
    # The reference has a true last dimension of three. Its pinned CPU kernel
    # uses the size<4 scalar reduction: ((exp0+exp1)+exp2).
    ex = attention.exponential(x - x.max(-1, keepdim=True).values)
    total = (ex[..., 0:1] + ex[..., 1:2]) + ex[..., 2:3]
    return ex * (torch.ones_like(total) / total)


model = load_model("cpu")
with torch.inference_mode():
    reference_hidden, reference_layers = reference_prefix(model, ids)
    reference_scores = model.scores(reference_hidden[:, -1:])
    for variant, softmax in (("four_lane", attention.softmax), ("scalar_three", scalar_three_softmax)):
        graphs.softmax = softmax
        embedding = graphs.Embedding(model).eval()
        head = graphs.Head(model).eval()
        x, ve = embedding(torch.tensor([ids], dtype=torch.long))
        tap0 = x.clone()
        tap12 = x.clone()
        tap23 = x.clone()
        positions = torch.arange(3, dtype=torch.int32)
        row = {"softmax_sum_order": "((0+2)+(1+3)) with masked zeros" if variant == "four_lane" else "((0+1)+2)",
               "groups": []}
        report["variants"][variant] = row
        save()
        for group_index in range(12):
            group = graphs.Group(model, 4 * group_index).eval()
            x, tap23 = group(x, ve, positions, tap0, tap12, tap23)
            if group_index == 2:
                tap12 = x
            boundary = {"group": group_index, "last_layer": 4 * group_index + 3}
            expected = reference_layers.get(f"layer_{4 * group_index + 3}")
            if expected is not None:
                boundary["versus_reference"] = metrics(x, expected)
                boundary["different_elements"] = int(torch.count_nonzero(x != expected))
            row["groups"].append(boundary)
            print(variant, group_index, boundary.get("different_elements"),
                  boundary.get("versus_reference", {}).get("max_abs"), flush=True)
            save()
            del group
            gc.collect()
        candidate_scores = head(x[:, -1:])
        row["hidden"] = metrics(x, reference_hidden)
        row["hidden"]["different_elements"] = int(torch.count_nonzero(x != reference_hidden))
        row["scores"] = score_metrics(candidate_scores, reference_scores, model.config.tokenizer_vocab_size)
        row["scores"]["different_elements"] = int(torch.count_nonzero(candidate_scores != reference_scores))
        print(variant, "final", row["hidden"], row["scores"], flush=True)
        save()
        del embedding, head
        gc.collect()
