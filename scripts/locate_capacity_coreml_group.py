"""Locate first Core ML group mismatch over the frozen 2048-token capacity path.

Uses native CPU group-boundary activations for each 32-token input chunk while
retaining each candidate group's Core ML cache state across the exact frozen
partition endpoints. This isolates the earliest group/chunk state divergence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
import torch

from coreml_port.cleanup_guard import install as install_cleanup_guard
from coreml_port.package_identity import verify_candidate_identity
from portable.model import load_model, rms
from scripts.audit_torch_candidate import reference_prefix
from scripts.probe_full_model import metrics


ENDPOINTS = (1023, 1024, 1025, 1026, 2048)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, help="New directory name below evidence/04d-audit/.")
    parser.add_argument("--model-dir", required=True, help="Explicit candidate package root.")
    parser.add_argument("--identity-manifest", required=True, help="Package identity manifest for that root.")
    parser.add_argument("--input-ids", required=True, help="JSON token ID array from the frozen boundary run.")
    parser.add_argument("--prediction-guard", action="store_true", help="Install the qualified process-local prediction guard before loading Core ML models.")
    args = parser.parse_args()

    attempt_name = Path(args.attempt)
    if attempt_name.is_absolute() or ".." in attempt_name.parts or len(attempt_name.parts) != 1:
        parser.error("--attempt must be one safe directory name")
    out = ROOT / "evidence/04d-audit" / attempt_name
    out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(args.model_dir).expanduser()
    manifest_path = Path(args.identity_manifest).expanduser()
    input_path = Path(args.input_ids).expanduser()
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    model_dir = model_dir.resolve()
    manifest_path = manifest_path.resolve()
    input_path = input_path.resolve()
    ids = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or len(ids) != 2048 or any(type(token) is not int for token in ids):
        raise ValueError("Frozen capacity input must contain exactly 2048 integer token IDs")

    source_paths = [
        Path(__file__), ROOT / "scripts/audit_torch_candidate.py",
        ROOT / "portable/model.py", ROOT / "coreml_port/runtime.py",
        ROOT / "coreml_port/cleanup_guard.py", ROOT / "coreml_port/native/cleanup_guard.c",
    ]
    if args.prediction_guard:
        from coreml_port.prediction_guard import install as install_prediction_guard

        source_paths.extend([
            ROOT / "coreml_port/prediction_guard.py",
            ROOT / "coreml_port/native/prediction_guard.mm",
            ROOT / "artifacts/04e-runtime/prediction_guard.dylib",
        ])
        prediction_guard = install_prediction_guard()
    else:
        prediction_guard = {"installed": False}
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}

    install_cleanup_guard()
    torch.set_num_threads(4)
    package_identity = verify_candidate_identity(model_dir, manifest_path)
    record: dict = {
        "status": "starting",
        "script_sha256": sha256(Path(__file__)),
        "model_dir": str(model_dir),
        "identity_manifest": str(manifest_path),
        "package_identity": package_identity,
        "input_ids_path": str(input_path),
        "input_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "input_length": len(ids),
        "endpoints": list(ENDPOINTS),
        "chunk_size": 32,
        "runtime_adapter": "group_by_group_native_activation_inputs",
        "prediction_guard": prediction_guard,
        "source_sha256": source_hashes,
        "groups": [],
    }

    def save() -> None:
        temporary = out / "results.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(out / "results.json")

    save()
    reference = load_model()
    tokens = torch.tensor([ids], dtype=torch.long, device=reference.device)
    taps: dict[str, torch.Tensor] = {}

    def keep_layer22(module, args, value):
        taps["layer_22"] = value[0] if isinstance(value, tuple) else value

    hook = reference.backbone.layers[22].register_forward_hook(keep_layer22)
    try:
        with torch.inference_mode():
            hidden, captures = reference_prefix(reference, ids)
            captures["layer_22"] = taps["layer_22"].detach().float().cpu().clone()
            captures["embedding_norm"] = rms(reference.backbone.embed_tokens(tokens)).detach().float().cpu().clone()
            captures["ve"] = reference.backbone._value_embeddings(tokens).detach().float().cpu().clone()
            # Preserve native BF16 through the qualified head before converting
            # any tensors for saved group-level comparisons.
            native_scores = reference.scores(hidden[:, -1:])
            captures["scores"] = native_scores.detach().float().cpu().clone()
    finally:
        hook.remove()
    arrays = {name: value.detach().float().cpu().numpy().copy() for name, value in captures.items()}
    np.savez_compressed(out / "reference-native-prefix.npz", **arrays)
    record["reference_head_input_dtype"] = str(hidden.dtype)
    record["reference_head_input_native_bf16"] = hidden.dtype == torch.bfloat16
    record["reference_score_output_dtype"] = str(native_scores.dtype)
    record["reference_scores_max_abs"] = float(np.max(np.abs(arrays["scores"])))
    del reference, captures, taps, hidden, native_scores
    gc.collect()
    record["status"] = "reference captured; comparing group states"
    save()

    spans = []
    start = 0
    for endpoint in ENDPOINTS:
        spans.append((start, endpoint))
        start = endpoint
    chunk_plan = []
    for partition_index, (partition_start, partition_end) in enumerate(spans):
        offset = partition_start
        while offset < partition_end:
            end = min(offset + 32, partition_end)
            chunk_plan.append({
                "partition_index": partition_index,
                "partition_start": partition_start,
                "partition_end": partition_end,
                "start": offset,
                "end": end,
            })
            offset = end
    record["chunk_plan"] = chunk_plan
    save()

    for group in range(12):
        package = model_dir / f"{group}.mlpackage"
        model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
        state = model.make_state()
        state_names = [item.name for item in model.get_spec().description.state]
        input_key = "embedding_norm" if group == 0 else f"layer_{4 * group - 1}"
        output_key = "final_hidden" if group == 11 else f"layer_{4 * group + 3}"
        row = {"group": group, "package": str(package), "input_reference": input_key,
               "output_reference": output_key, "steps": []}
        record["groups"].append(row)
        for chunk_index, chunk in enumerate(chunk_plan):
            left, right = chunk["start"], chunk["end"]
            inputs = {
                "x": arrays[input_key][:, left:right].copy(),
                "ve": arrays["ve"][:, left:right].copy(),
                "positions": np.arange(left, right, dtype=np.int32),
                "tap0": arrays["embedding_norm"][:, left:right].copy(),
                "tap12": arrays["layer_11"][:, left:right].copy(),
                "tap23": arrays["layer_22"][:, left:right].copy(),
                "norm_divisor": np.array([1280.0], dtype=np.float32),
            }
            before = {name: state.read_state(name).copy() for name in state_names}
            actual = model.predict(inputs, state=state)["y"]
            expected = arrays[output_key][:, left:right]
            different = int(np.count_nonzero(actual != expected))
            summary = metrics(torch.from_numpy(actual), torch.from_numpy(expected))
            step = {**chunk, "chunk_index": chunk_index, "different_elements": different, **summary}
            row["steps"].append(step)
            save()
            print(f"group={group} span={left}:{right} diff={different}", flush=True)
            if different:
                replay_inputs = dict(inputs)
                replay_inputs["actual"] = actual
                replay_inputs["expected"] = expected
                np.savez_compressed(out / "first-failure-inputs.npz", **replay_inputs)
                np.savez_compressed(out / "first-failure-state-before.npz", **before)
                record["status"] = "first group/partition/chunk difference captured"
                record["first_failure"] = {"group": group, **chunk, "chunk_index": chunk_index}
                save()
                return 0
        del model, state
        gc.collect()

    record["status"] = "all groups exact across capacity path"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
