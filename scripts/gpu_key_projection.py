"""Bounded Core ML key projection placement/composition experiment."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
import torch
from torch import nn

from coreml_port.cleanup_guard import install
from coreml_port.graphs import Q, Embedding, Layer
from coreml_port.ordered_linear import linear
from portable.model import load_model

install()
torch.set_num_threads(4)
OUT = ROOT / "evidence/04d-gpu"
ART = ROOT / "artifacts/04d-gpu"
OUT.mkdir(exist_ok=True, parents=True)
ART.mkdir(exist_ok=True, parents=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_package(path):
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


class Key(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.register_buffer("w", layer.k)

    def forward(self, x):
        raw = linear(x, self.w)
        return raw, Q(raw)


class QKV(nn.Module):
    def __init__(self, layer, companion):
        super().__init__()
        self.companion = companion
        for name in ("q", "k", "v"):
            self.register_buffer(name, getattr(layer, name))

    def forward(self, x):
        q = Q(linear(x, self.q)) if "q" in self.companion else None
        kraw = linear(x, self.k)
        k = Q(kraw)
        v = Q(linear(x, self.v)) if "v" in self.companion else None
        return tuple(a for a in (q, kraw, k, v) if a is not None)


def compare(got, want):
    a = got.astype(np.float64)
    b = want.astype(np.float64)
    diff = np.abs(a - b)
    ix = np.argwhere(diff != 0)
    return {
        "different_elements": len(ix),
        "max_abs": float(diff.max()),
        "mismatches": [
            {"index": row.tolist(), "got": float(got[tuple(row)]),
             "want": float(want[tuple(row)]), "abs": float(diff[tuple(row)])}
            for row in ix[:20]
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", choices=("key", "qk", "kv", "qkv"), default=["key", "qkv"])
    p.add_argument("--counts", nargs="+", type=int, default=[32])
    args = p.parse_args()
    report_path = OUT / ("key-projection.json" if args.variants == ["key", "qkv"] else "key-projection-" + "-".join(args.variants) + ".json")
    if report_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing evidence: {report_path}")
    model = load_model()
    ids = torch.tensor([json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][0]["input_ids"][:32]])
    with torch.inference_mode():
        x, _ = Embedding(model)(ids)
    layer = Layer(model, 0)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT / "scripts/gpu_key_projection.py", ROOT / "coreml_port/graphs.py",
            ROOT / "coreml_port/ordered_linear.py", ROOT / "evidence/01-reference/local-input-validation.json"]},
        "input_shape": list(x.shape), "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        "results": {},
    }
    for variant in args.variants:
        graph = (Key(layer) if variant == "key" else QKV(layer, variant)).eval()
        names = (["kraw", "k"] if variant == "key" else [n for n in ("q", "kraw", "k", "v") if n in ("kraw", "k") or n in variant])
        path = ART / f"{variant}.mlpackage"
        if path.exists():
            raise RuntimeError(f"Refusing to reuse stale package: {path}")
        traced = torch.jit.trace(graph, x[:, :3], check_trace=False)
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(1, ct.RangeDim(1, 32, default=3), 1280), dtype=np.float32)],
            outputs=[ct.TensorType(name=n, dtype=np.float32) for n in names],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True,
        )
        converted.save(str(path))
        report["results"][variant] = {"package_sha256": digest_package(path), "runs": {}}
        for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
            cm = ct.models.MLModel(str(path), compute_units=unit)
            report["results"][variant]["runs"][unit.name] = {}
            for count in args.counts:
                with torch.inference_mode():
                    inp = x[:, :count]
                    want = graph(inp)
                got = cm.predict({"x": inp.numpy()})
                item = {name: compare(got[name], expected.numpy()) for name, expected in zip(names, want)}
                report["results"][variant]["runs"][unit.name][str(count)] = item
                print(variant, unit.name, count, {n: (v["different_elements"], v["max_abs"]) for n, v in item.items()}, flush=True)
                report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
