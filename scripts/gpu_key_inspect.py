"""Inspect previously generated QKV Core ML outputs at key BF16 ties."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
import torch

from coreml_port.cleanup_guard import install
from coreml_port.graphs import Q, Embedding, Layer
from coreml_port.ordered_linear import linear
from portable.model import load_model

install()
torch.set_num_threads(4)
model = load_model()
ids = torch.tensor([json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][0]["input_ids"][:32]])
with torch.inference_mode():
    x, _ = Embedding(model)(ids)
    layer = Layer(model, 0)
    raw = linear(x, layer.k).numpy()
    rounded = Q(torch.from_numpy(raw)).numpy()

report = {
    "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
    "package": "artifacts/04d-gpu/qkv.mlpackage",
    "sites": [],
}
locations = [(0, 12, 199), (0, 14, 199)]
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(ROOT / report["package"]), compute_units=unit)
    got = cm.predict({"x": x.numpy()})
    for ix in locations:
        val = float(raw[ix])
        upper = float(np.float32(0.79296875))
        lower = float(np.float32(0.7890625))
        report["sites"].append({
            "compute_unit": unit.name, "index": ix,
            "torch_raw": val, "coreml_raw": float(got["kraw"][ix]),
            "torch_round": float(rounded[ix]), "coreml_round": float(got["k"][ix]),
            "bf16_midpoint": (lower + upper) / 2,
            "raw_minus_midpoint": val - (lower + upper) / 2,
            "coreml_raw_minus_midpoint": float(got["kraw"][ix]) - (lower + upper) / 2,
            "raw_bits": format(np.asarray(raw[ix]).view(np.uint32).item(), "08x"),
        })

path = ROOT / "evidence/04d-gpu/key-ties.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
