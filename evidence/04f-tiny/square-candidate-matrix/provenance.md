The frozen `summary.json` was produced by `source-snapshot.py`; its SHA-256
matches `script_sha256` in that summary. During discovery, the working script
was rerun in this same attempt directory as candidates were added. The current
`scripts/tiny_kernel_square_candidates.py` now refuses an occupied output
directory. The independent holdout is bound to the final frozen summary SHA-256.
