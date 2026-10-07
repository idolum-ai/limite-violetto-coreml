# Setup and reproduction

## Review without model execution

Run `python3 scripts/verify_repository.py` from the repository root. This checks imported file identities, Python syntax, and JSON parseability without importing Torch, reading weights, loading Core ML, or accessing the network.

The historical reports are a curated export. Read [archive.md](archive.md) before using old hashes, identity files, or commands as executable inputs.

## Pinned numerical environment

The recorded host was macOS 26.5.2 on Apple M4 with 32 GiB memory and CPython 3.12.9 (arm64). Principal packages are Torch 2.7.0, coremltools 9.0, Transformers 5.6.2, NumPy 2.2.6, pytest 9.1.1, tokenizers 0.22.2, and safetensors 0.8.0. The lock selects platform-specific distributions; it is not a portable Linux/CUDA lock.

With CPython **3.12.9** already installed:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-investigation.lock
.venv/bin/python -m pip check
```

Verify the interpreter version first. Do not use the upstream runtime's `uv.lock` for the macOS environment: it records a different NVIDIA/CUDA stack. This cleanup did not install dependencies, change versions, or test a clean machine.

## Checkpoint (optional, approximately 2.07 GB)

Pinned model revision: `9402e422e87b220507963cd42c997459c1e87b43`.

Expected checkpoint: `sources/model/model.safetensors`, exactly **2,070,811,520 bytes**, SHA-256 **`9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5`**.

The tokenizer, configs, and chat template are included. Prefer copying the verified checkpoint from the existing local archive. Otherwise, explicitly download the pinned public file; this block refuses to replace an existing checkpoint or partial download:

```sh
(
  set -eu
  cd sources/model
  test ! -e model.safetensors
  test ! -e model.safetensors.part
  curl --fail --location \
    'https://huggingface.co/paradigma-inc/limite-1b-violetto/resolve/9402e422e87b220507963cd42c997459c1e87b43/model.safetensors' \
    --output model.safetensors.part
  printf '%s  %s\n' \
    '9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5' \
    'model.safetensors.part' | shasum -a 256 -c -
  test "$(wc -c < model.safetensors.part | tr -d ' ')" = 2070811520
  mv model.safetensors.part model.safetensors
)
```

No download is performed by the repository verification or bounded tests. The portable loader also verifies the checkpoint identity. Weights remain ignored by Git.

## Bounded tests without weights or Core ML models

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest scripts/test_*.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/test_compiled_identity.py
PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/sources/runtime/paradigma-inc-limite-violetto-cd68d27/src" \
  .venv/bin/python -m pytest \
  sources/runtime/paradigma-inc-limite-violetto-cd68d27/tests/test_package_contract.py \
  -q -p no:cacheprovider
```

The compiled identity script has its own executable harness and must be run separately; pytest alone does not execute its `run()` function. The upstream contract tests stub attention and establish package/API behavior, not model fidelity. CLI help probes import the numerical libraries but do not load model packages.

## Core ML execution requires the local archive

First restore the required raw artifacts and unmodified identity manifests as described in [archive.md](archive.md). Both source packages and compiled caches are required. Compiled identities bind the environment, absolute installed-file paths, and binaries: copying a cache to another checkout can invalidate it. Preserve the original working archive for recorded-cache execution; a new environment requires fresh manifests and qualification, not bypassed identity checks.

The process-local cleanup and prediction guards require the recorded coremltools bridge hash and CPython 3.12, plus Xcode command-line tools for compilation. The guard loaders reject an unexpected bridge. The historical build commands are:

```sh
.venv/bin/python scripts/build_cleanup_guard.py
.venv/bin/python scripts/build_prediction_guard.py
```

These historical scripts write fixed build/evidence paths. Run them in a separate experiment checkout after preserving its records, not over the sealed archive. The cleanup guard must be built first. Native Core ML behavior requires normal host access; a sandbox that hides Metal does not establish hardware capability.

Within the original qualified archive, the promoted candidate is selected explicitly:

```sh
.venv/bin/python -m coreml_port.compiled_generate 'What is 17 * 23?' \
  --model-dir artifacts/05g-integration/full-1/source \
  --identity-manifest evidence/05g-integration/full-1/package-identity.json \
  --compiled-dir artifacts/05g-integration/full-1/compiled \
  --compiled-manifest evidence/05g-integration/full-1/compiled-identity.json \
  --max-new-tokens 1024 --sample --seed 0 --stop-at-tokenizer-eos --stream
```

The fallback uses source `artifacts/04d-cpu/full-candidate-6`, its package manifest under `evidence/04d-cpu/full-candidate-6/`, and compiled cache `artifacts/05c-compiled/full-cache-1` with the corresponding raw compiled manifest. CPU_ONLY is the qualified policy. JSON goes to stdout, streamed text to stderr, and cancellation disposes request state.

## Repeating the research

Start from [CPU integration commands](../evidence/05g-integration/commands.md), [CPU baseline commands](../evidence/04d-cpu/commands.md), and [portable cache commands](../evidence/03b-cache/commands.md). Use unused attempt names and output directories. Some old scripts have fixed output paths; inspect them before running. Keep the original acceptance criteria and record any deliberate changes separately.

A normal process exit is not a numerical pass. Inspect the gate reports and failure lists. Do not infer NVIDIA parity, GPU correctness, ANE support, device placement, or production readiness from contract/unit tests. Rebuilding every stage is expensive and was outside this cleanup; no single-command fresh full-model qualification is promised.
