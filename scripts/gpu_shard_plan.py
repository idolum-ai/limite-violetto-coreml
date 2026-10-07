"""Record anticipated device use in the corrected shard and key tap."""
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
for name in ("shard0-even", "shard0-keytap"):
    model = ct.models.MLModel(str(ROOT / f"artifacts/04d-gpu/{name}.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(model.get_compiled_model_path(), ct.ComputeUnit.CPU_AND_GPU)
    ops = plan.model_structure.program.functions["main"].block.operations
    tally = Counter()
    math_tally = Counter()
    for op in ops:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        device = "unknown" if usage is None else type(usage.preferred_compute_device).__name__
        tally[device] += 1
        if op.operator_name.endswith((".round", ".floor", ".real_div", ".mul", ".add")):
            math_tally[(op.operator_name, device)] += 1
    report[name] = {"operations": len(ops), "preferred_counts": dict(tally),
                    "arithmetic_preferred_counts": {f"{op} {device}": count for (op, device), count in math_tally.items()}}
    print(name, report[name]["preferred_counts"], flush=True)
(ROOT / "evidence/04d-gpu/shard-compute-plan.json").write_text(json.dumps(report, indent=2) + "\n")
