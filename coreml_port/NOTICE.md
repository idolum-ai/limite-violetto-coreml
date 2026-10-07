# Source notices

`graphs.py` adapts the pinned Paradigma GMPT implementation under Apache-2.0; original sources and notices are retained in `sources/value-model` and `sources/runtime`.

`ordered_linear.py` re-expresses the reduction order described by PyTorch's `ReducedPrecisionFloatGemvFastPathKernel.cpp` at commit `134179474539648ba7dee1317959529fbd0e7f89`. PyTorch's license is retained at `sources/torch-2.7-linear/LICENSE`. That source credits the following reduction algorithm to GGML under MIT:

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The local cleanup guard and test reproducer interoperate with Core ML Tools' `PybindCompatibleArray`. Inspected Apple source and its BSD-3-Clause license are retained under `sources/coremltools-9.0-bridge`. The guard does not replace or redistribute a modified coremltools wheel.

`ordered_attention.py` re-expresses the four-lane sum in PyTorch CPU `SoftMaxKernel.cpp` and `functional_base.h` at the same pinned commit. That source and the installed header snapshots are retained in `sources/torch-2.7-linear` under PyTorch's retained license.

`ordered_norm.py` re-expresses the two model widths' sum order from `SumKernel.cpp`. `ordered_xsa.py` re-expresses the 128-wide norm2 order from `ReduceOpsKernel.cpp`, plus that ordered sum for the dot product. Both source snapshots are retained at the same pinned PyTorch commit and license. `bf16_gelu.py` generates a bounded lookup table from the pinned Torch GELU operator; it does not claim arbitrary floating-point or nonfinite-input equivalence.

`source_exp.py` re-expresses the nonpositive-domain `xexpf` polynomial and range reduction from SLEEF commit `56e1f79cb140fb9326d612d0be06b5250565cade` (the pinned Torch submodule). Copyright Naoki Shibata and contributors, 2010-2024. The Boost Software License and inspected sources are retained in `sources/sleef-pinned`. Compensated FP32 operations reproduce the tested source exponential domain; this is not a general FMA equivalence claim.
