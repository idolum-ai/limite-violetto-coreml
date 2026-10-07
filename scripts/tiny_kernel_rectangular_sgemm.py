"""Check source-derived Accelerate calls across short Q/K geometries."""
import hashlib
import json
from pathlib import Path

import torch

from tiny_kernel_direct_sgemm import direct, metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/04f-tiny/rectangular-sgemm"
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
query_lengths = (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15)
key_lengths = (1, 2, 3, 7, 15, 2048)
report = {"status": "starting", "torch_version": torch.__version__,
          "query_lengths": query_lengths, "key_lengths": key_lengths,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "direct_call_script_sha256": hashlib.sha256((ROOT / "scripts/tiny_kernel_direct_sgemm.py").read_bytes()).hexdigest(),
          "cases": []}
for q_length in query_lengths:
    for key_length in key_lengths:
        seed = 700_000 + q_length * 3000 + key_length
        torch.manual_seed(seed)
        q = torch.randn(1, q_length, 10, 128).to(torch.bfloat16).float().contiguous()
        k2 = torch.randn(1, key_length, 2, 128).to(torch.bfloat16).float().contiguous()
        k = k2.repeat_interleave(5, dim=2).contiguous()
        with torch.inference_mode():
            native = torch.einsum("bthd,bshd->bhts", q, k).numpy()
        actual = direct(q.numpy(), k.numpy())
        row = {"q": q_length, "k": key_length, "seed": seed,
               "contract_work": 128 * q_length * key_length,
               "torch_path_by_source_cutoff": "naive" if 128 * q_length * key_length < 400 else "addmm_sgemm",
               "q_sha256": hashlib.sha256(q.numpy().tobytes()).hexdigest(),
               "k_sha256": hashlib.sha256(k.numpy().tobytes()).hexdigest(),
               "raw_dot": metrics(actual, native)}
        report["cases"].append(row)
report["status"] = "complete"
report["all_addmm_sgemm_exact"] = all(c["raw_dot"]["different"] == 0 for c in report["cases"]
                                        if c["torch_path_by_source_cutoff"] == "addmm_sgemm")
(OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "all_addmm_sgemm_exact": report["all_addmm_sgemm_exact"],
                  "nonexact": [{"q": c["q"], "k": c["k"], "path": c["torch_path_by_source_cutoff"],
                                "different": c["raw_dot"]["different"]} for c in report["cases"]
                               if c["raw_dot"]["different"] != 0]}))
