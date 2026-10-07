# All-group conditional-score static review

Builder SHA-256: `d8e9d0162fd829f05af5c4b93f9375144240e225034962a170c587e2c54cb019` (`scripts/build_conditional_scores.py`). Source: sealed candidate-6 aggregate `5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6`.

I loaded each of the 12 qualified source group specs and ran the complete rewrite in memory without saving a package, compiling, loading a native model, or calling prediction. All 12 passed selector recognition, short-branch partitioning, parent and nested-block SSA checks, byte equality for every moved original operation and all unchanged parent operations, non-operation spec metadata equality, serialization, and unchanged matmul/read-state/write-state counts. Each group had four selectors, 10,009 short-only operations, four disjoint 2,501-operation short branches (10,004 moved), and five shared short ancestors retained in the parent. Each new true branch also has one local identity for the refined score type; each false branch has one local identity returning the ordinary score. All groups retain 68 matmuls, 16 state reads, and eight state writes.

| Group | Original operations | Original identities | Full in-memory rewrite |
| --- | ---: | ---: | --- |
| 0 | 42,498 | 2 | Pass |
| 1 | 42,707 | 2 | Pass |
| 2 | 42,498 | 2 | Pass |
| 3 | 42,498 | 2 | Pass |
| 4 | 42,707 | 2 | Pass |
| 5 | 42,499 | 3 | Pass |
| 6 | 44,003 | 0 | Pass, sealed group-0 identity template fallback |
| 7 | 42,707 | 2 | Pass |
| 8 | 42,498 | 2 | Pass |
| 9 | 42,498 | 2 | Pass |
| 10 | 42,705 | 2 | Pass |
| 11 | 44,476 | 1 | Pass |

This is a structural review, not native equivalence or proof that Core ML schedules the conditional lazily. Compilation, native exactness, full-model qualification, and performance are separate gates.
