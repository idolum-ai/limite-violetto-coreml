"""Record anticipated Core ML device placement for GPU projection probes."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
from coreml_port.cleanup_guard import install

install()
report = {}
for variant in ("key", "qk", "kv", "qkv", "qkv-even", "round-ties"):
    path = ROOT / f"artifacts/04d-gpu/{variant}.mlpackage"
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(model.get_compiled_model_path(), ct.ComputeUnit.CPU_AND_GPU)
    ops = plan.model_structure.program.functions["main"].block.operations
    rows = []
    for op in ops:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            preferred = "unknown"
            supported = []
        else:
            preferred = type(usage.preferred_compute_device).__name__
            supported = [type(d).__name__ for d in usage.supported_compute_devices]
        rows.append({"op": op.operator_name, "outputs": [v.name for v in op.outputs], "preferred": preferred, "supported": supported})
    report[variant] = {
        "operation_count": len(rows),
        "preferred_counts": dict(Counter(r["preferred"] for r in rows)),
        "by_op": {name: dict(Counter(r["preferred"] for r in rows if r["op"] == name)) for name in sorted(set(r["op"] for r in rows))},
        "rounding_ops": [r for r in rows if r["op"] in ("round", "floor", "gather", "div", "mul")][-30:],
    }
    print(variant, report[variant]["preferred_counts"], flush=True)
(ROOT / "evidence/04d-gpu/compute-plan.json").write_text(json.dumps(report, indent=2) + "\n")
