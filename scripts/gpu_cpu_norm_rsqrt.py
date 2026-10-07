"""Compare reciprocal-square-root expressions at the layer-2 tie boundary."""
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
from coreml_port.graphs import Q
from portable.model import EPS

install()
torch.set_num_threads(4)
x = torch.from_numpy(np.load(ROOT / "artifacts/04d-gpu/cpu-norm/norm-input.npz")["value"])
denom = (x * x).mean(-1, keepdim=True) + EPS
with torch.inference_mode():
    reference_inv = torch.rsqrt(denom)
    reference_round = Q(x * reference_inv)


class Alternatives(nn.Module):
    def forward(self, x, d):
        inv_rsqrt = torch.rsqrt(d)
        inv_sqrt_div = torch.ones_like(d) / torch.sqrt(d)
        inv_pow = d.pow(-0.5)
        return (inv_rsqrt, inv_sqrt_div, inv_pow,
                Q(x * inv_rsqrt), Q(x * inv_sqrt_div), Q(x * inv_pow))


names = ("rsqrt", "sqrt_div", "pow", "rsqrt_round", "sqrt_div_round", "pow_round")
graph = Alternatives().eval()
path = ROOT / "artifacts/04d-gpu/cpu-norm/rsqrt.mlpackage"
if path.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {path}")
trace = torch.jit.trace(graph, (x, denom), check_trace=False)
converted = ct.convert(
    trace, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32),
                   ct.TensorType(name="d", shape=denom.shape, dtype=np.float32)],
    outputs=[ct.TensorType(name=name, dtype=np.float32) for name in names],
    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
converted.save(str(path))
report = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "denom_sha256": hashlib.sha256(denom.numpy().tobytes()).hexdigest(),
          "row9_denom": float(denom[0, 9, 0]), "row9_reference_inv": float(reference_inv[0, 9, 0]),
          "results": {}}
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(path), compute_units=unit)
    got = cm.predict({"x": x.numpy(), "d": denom.numpy()})
    report["results"][unit.name] = {}
    for name in names:
        expected = reference_round.numpy() if name.endswith("_round") else reference_inv.numpy()
        actual = got[name]
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        ix = np.argwhere(diff != 0)
        report["results"][unit.name][name] = {
            "different_elements": len(ix), "max_abs": float(diff.max()),
            "row9": float(actual[0, 9, 0]), "row9_want": float(expected[0, 9, 0]),
            "mismatches": [{"index": row.tolist(), "got": float(actual[tuple(row)]),
                            "want": float(expected[tuple(row)])} for row in ix[:20]],
        }
    print(unit.name, {name: (r["different_elements"], r["max_abs"]) for name, r in report["results"][unit.name].items()}, flush=True)
    (ROOT / "evidence/04d-gpu/cpu-norm/rsqrt.json").write_text(json.dumps(report, indent=2) + "\n")
