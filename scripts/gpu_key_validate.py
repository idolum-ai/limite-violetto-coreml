"""Validate existing composed GPU projection packages at multiple lengths."""
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
report = {
    "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "runs": {},
}
for variant in ("qkv", "qkv-even"):
    report["runs"][variant] = {}
    path = ROOT / f"artifacts/04d-gpu/{variant}.mlpackage"
    for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
        cm = ct.models.MLModel(str(path), compute_units=unit)
        report["runs"][variant][unit.name] = {}
        for count in (1, 3, 32):
            inp = x[:, :count]
            with torch.inference_mode():
                q = Q(linear(inp, layer.q))
                kraw = linear(inp, layer.k)
                k = Q(kraw)
                v = Q(linear(inp, layer.v))
            got = cm.predict({"x": inp.numpy()})
            row = {}
            for name, expected in zip(("q", "kraw", "k", "v"), (q, kraw, k, v)):
                arr = expected.numpy()
                diff = np.abs(got[name].astype(np.float64) - arr.astype(np.float64))
                row[name] = {"different_elements": int(np.count_nonzero(diff)), "max_abs": float(diff.max())}
            report["runs"][variant][unit.name][str(count)] = row
            print(variant, unit.name, count, {n: r["different_elements"] for n, r in row.items()}, flush=True)
out = ROOT / "evidence/04d-gpu/key-validation.json"
if out.exists():
    raise RuntimeError(f"Refusing to overwrite existing evidence: {out}")
out.write_text(json.dumps(report, indent=2) + "\n")
