# Independent read-only review

The reviewer verified all ten trials per backend, repeat0 exclusion from warm statistics, two warm prefills and sixteen warm decode samples per shape, and correct CoreML/PyTorch latency ratios. The 27 cross-backend score vectors pass the frozen limits, with all top1 selections matching and maximum score error3.814697265625e-6. Source snapshots, protocol, stream and runtime identities match. PeakRSS uses Darwin bytes divided by2^30 forGiB.

Interpretation boundaries: model initialization includes backend import and identity/checkpoint verification. Entry-to-ready is the reported startup measure. One process per backend does not establish startup variance or cold-cache behavior. Initial host load averages differ, although both runs are onAC with no recorded thermal warning. The two warm samples per shape support a descriptive pilot, not a universal hardware/framework performance claim. No material arithmetic or provenance inconsistency was found.
