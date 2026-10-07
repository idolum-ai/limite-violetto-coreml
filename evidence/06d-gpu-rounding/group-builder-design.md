# Original group-0 signed ties-even prototype

`build_signed_even_group.py` is a separate, optional-compile protobuf builder. It uses original sealed candidate6 (`5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6`), never the conditional CPU candidate. No qualified model, runtime, validator, prior builder, or old evidence changes.

The default replaces all 131 `round` operations in the original 42,498-operation group0 graph. `--site` may instead name one or more exact original output names. The input and output descriptor types of each replaced round must match. The helper computes magnitude/floor/fraction/parity, selects the nearest even magnitude, restores negative sign with multiplication by -1, and preserves literal input zero through a final select. The original exponent, step, multiplication, and every other operation are retained byte-for-byte. The helper's finite FP32 quotient domain is a numerical qualification prerequisite; structural success does not prove numerical equivalence.

The MIL Builder fragment is converted symbolically using `skip_model_load=True`, FP32, macOS15/CoreML8, and an empty configurable optimization pipeline. The converter still runs its required six backend passes, so the exported fragment is checked against the exact expected operation counts: 17 pure nonconstant operations plus scalar constants (8 in this environment). No `round` is permitted in the fragment. On cloning, intermediate names are unique; tensor rank/dimensions specialize to the original round output, boolean result dtypes and scalar constant types stay intact, and the final original output descriptor is copied exactly.

Checks cover every unaffected operation's serialized bytes, topological references, duplicate output names, original output names/types, and the entire model protobuf with operation lists removed. Thus weights references, functions, state interfaces, and all state operations are unchanged. The package copy must differ only at `Data/com.apple.CoreML/model.mlmodel`; every package file's SHA is recorded. Complete per-site input/output/name maps, operation hashes, helper/builder snapshots, source cohort identity, and structural results are saved.

CLI for the root runner after the operator gate:

```sh
.venv/bin/python scripts/build_signed_even_group.py --group 0 --attempt group0-1 --compile
```

Without `--compile`, this performs only package/protobuf construction. With `--compile`, prediction/cleanup guards are installed first; guard prediction counters must remain unchanged. This script never loads a model for prediction. Numerical operator, CPU preservation, GPU replay, broad group, and whole-model gates are separate root-owned stages.

Preflight: the real original graph passed all131 and selected `var_1623` in-memory rewrites; the latter leaves130 rounds. An unknown site rejects. No native compilation, inference, or package artifact creation was needed for this check.
