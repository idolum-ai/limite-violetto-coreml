"""Check pinned vector-norm reduction order on captured layer-2 values."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from coreml_port.ordered_norm import ordered_sum

torch.set_num_threads(4)
v = torch.from_numpy(np.load(ROOT / "artifacts/04d-gpu/xsa/layer2-inputs.npz")["v"]).repeat_interleave(5, dim=2)
with torch.inference_mode():
    reference_norm = v.norm(p=2, dim=-1, keepdim=True)
    reference_vn = F.normalize(v, dim=-1, eps=1e-4)
    lanes = v[..., :4] * v[..., :4]
    for j in range(4, 128, 4):
        q = v[..., j:j + 4]
        lanes = lanes + q * q
    pairwise = (lanes[..., 0:1] + lanes[..., 2:3]) + (lanes[..., 1:2] + lanes[..., 3:4])
    linear = ((lanes[..., 0:1] + lanes[..., 1:2]) + lanes[..., 2:3]) + lanes[..., 3:4]
    alternatives = {
        "vec4_pairwise": torch.sqrt(pairwise),
        "vec4_linear": torch.sqrt(linear),
        "sqrt_ordered_sum": torch.sqrt(ordered_sum(v * v)),
        "sqrt_torch_sum": torch.sqrt((v * v).sum(-1, keepdim=True)),
    }
    report = {}
    for name, norm in alternatives.items():
        vn = v / norm.clamp_min(1e-4)
        report[name] = {
            "norm_different": int(torch.count_nonzero(norm != reference_norm)),
            "norm_max_abs": float(torch.max(torch.abs(norm - reference_norm))),
            "vn_different": int(torch.count_nonzero(vn != reference_vn)),
            "vn_max_abs": float(torch.max(torch.abs(vn - reference_vn))),
        }
print(json.dumps(report, indent=2))
(ROOT / "evidence/04d-gpu/xsa/norm-orders.json").write_text(json.dumps(report, indent=2) + "\n")
