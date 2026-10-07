"""Compare integrated candidate-2 shard0 with the prior exact-probe package."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np

from coreml_port.cleanup_guard import install

install()
BASE = ROOT / "artifacts/04d-cpu/full-candidate-2"
OLD = ROOT / "artifacts/04d-gpu/xsa/shard0.mlpackage"
OUT = ROOT / "evidence/04d-gpu/integrated-shard/compare.json"
if OUT.exists():
    raise RuntimeError(f"Refusing to overwrite existing evidence: {OUT}")
embedding = ct.models.MLModel(str(BASE / "embedding.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
current = ct.models.MLModel(str(BASE / "0.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
previous = ct.models.MLModel(str(OLD), compute_units=ct.ComputeUnit.CPU_ONLY)
cases = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"]
divisor = np.array([1280.], dtype=np.float32)
report = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "current": str((BASE / "0.mlpackage").relative_to(ROOT)),
          "previous": str(OLD.relative_to(ROOT)), "prompts": []}
for index, case in enumerate(cases):
    ids = case["input_ids"]
    state_current = current.make_state()
    state_previous = previous.make_state()
    rows = []
    report["prompts"].append(rows)
    spans = [(start, min(start + 32, len(ids)), ids[start:start + 32]) for start in range(0, len(ids), 32)]
    spans.append((len(ids), len(ids) + 1, [16]))
    for start, end, chunk in spans:
        emb = embedding.predict({"ids": np.array([chunk], dtype=np.int32), "norm_divisor": divisor})
        payload = {"x": emb["x"], "ve": emb["ve"], "positions": np.arange(start, end, dtype=np.int32),
                   "tap0": emb["x"], "tap12": emb["x"], "tap23": emb["x"]}
        got_current = current.predict({**payload, "norm_divisor": divisor}, state=state_current)
        got_previous = previous.predict({**payload, "divisor1280": divisor}, state=state_previous)
        row = {"start": start, "end": end}
        for key in ("y", "tap23_out"):
            a, b = got_current[key], got_previous[key]
            row[key] = {"different_elements": int(np.count_nonzero(a != b)),
                        "max_abs": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))}
        rows.append(row)
        print(index, start, end, row, flush=True)
        OUT.write_text(json.dumps(report, indent=2) + "\n")
OUT.write_text(json.dumps(report, indent=2) + "\n")
