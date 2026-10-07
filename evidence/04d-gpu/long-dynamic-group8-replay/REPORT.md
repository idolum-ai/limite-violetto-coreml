# Dynamic-row group 8 on captured long-context failure

The [replay](summary.json) loaded the exact 1024:1026 input and pre-step KV state captured by the root group locator. All eight state arrays are exactly representable as FP16; the local Core ML bridge accepted their float32 representation through `MLState.write_state`, and read-back matched every saved value. Candidate 3 reproduced its previously recorded `actual` output **bitwise**, establishing that the input/state replay is faithful.

The previously successful tiny3 dynamic-row/raw-einsum [group-8 package](../../../artifacts/04d-gpu/tiny3-dynamic-group8/8.mlpackage) produced an output **bitwise identical to candidate 3** on this long-context step. Both have 543 differing `y` elements versus the native reference, maximum absolute difference 32. Both pass through `tap23_out` bitwise. Complete outputs are retained in [outputs.npz](outputs.npz), and the evidence records capture/package hashes.

Thus the tiny3 correction does not resolve the first long-context group-8 failure. At absolute position 1025, its row policy saturates at 32, and this raw-layout einsum compiles to the same result as candidate 3. No shared source or full-model package was changed.
