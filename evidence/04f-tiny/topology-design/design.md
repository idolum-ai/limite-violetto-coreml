# Recovering a fixed FP32 sum tree

The helper `scripts/sum_topology_reconstruct.py` isolates topology recovery
from the backend. Its oracle receives FP32 arrays `[batch,128]` and must return
raw, unscaled scalar dot sums. The backend should place distinct probes into
the 10 query heads while keeping the source `[1,Q,10,128]` / `[1,K,10,128]`
layouts, key operands equal to one, and fixed Q/K geometry and query row.
Read one specified key column; separately validate all columns and tail rows.
Do not import a script with top-level execution as a backend library.

For distinct input positions a,b,c, query three vectors. Each has +2^26 and
-2^26 in one of the pairs ab,ac,bc and +1 in the remaining position; other
positions are zero. The unique pair whose output equals one is the closer
pair in the rooted binary addition tree restricted to those three leaves.
Both large terms and one are BF16-exact, so every product with key=one is
exact FP32. Adding one to either signed large value rounds away the one;
combining the large values first produces exact zero, preserving one at their
common ancestor with its leaf. Initial zero additions do not affect this
argument. Use pre-scaling raw dots so scale arithmetic cannot obscure it.

Recover a tree by inserting leaves successively. At a node with children A,B,
choose representative leaves a,b. For new leaf x, a triple with closer pair
ab inserts x above the current node; pair ax descends into A; pair bx descends
into B. At a leaf, attach x as its sibling. This is correct by induction from
the induced rooted tree over leaves inserted so far. It identifies unordered
children and bracketing, not operand orientation, explicit zero accumulators,
or a unique machine instruction sequence.

All three placements are tested. Require exactly one output one and two
outputs zero. A full FP64 reduction returns three ones, which the helper
rejects. Nonbinary dot instructions, mixed precision, data-dependent order,
or cancellation behavior outside the assumptions can also violate this
condition. Passing it only establishes compatibility of the sampled triples:
an independently checked tree and fresh dense products remain necessary.
Subnormal/overflow behavior and NaNs are outside this probe's claim.

For n=128 the reconstruction needs at most 24,003 scalar probe vectors.
Ascending sequential accumulation needs only 378; cyclic lane trees tested
here need 750–3,165. The current helper requests one triple at a time (3
vectors), allowing one native 10-head operation per tree decision. Validation
requests many triples together, so the backend can fill all 10 heads.

Self-tests recover sequential, reverse sequential, cyclic 2/4/8/16-lane,
and random binary trees. Each recovered tree matches 1,000 independently
sampled topology triplets and 500 exponent-diverse dense input sums, exactly.
FP64 is rejected. `self-test.json` retains those outcomes. These are tests
of the helper, not evidence about Accelerate's actual tree.

Backend validation should freeze Q/K, query row, key column, selected head,
thread count and library/build metadata, retaining the recovered tree and
raw oracle triples/results. Then check:

1. At least 1,000 new random triples against the topology-only induced pair.
2. Swapped signs of the cancelling terms, negative unit, and changed large
   amplitude (e.g. 2^27). The large/unit ratio must be high enough to absorb
   the unit in binary32 in both signed directions.
3. Fresh dense BF16-exact q/k products over several fixed seeds, including
   cancellation-heavy exponent-diverse distributions, compared bitwise with
   explicit FP32 tree evaluation; preserve actual operand layouts.
4. Other heads, key columns, relevant tail rows and Q/K geometries. A match at
   Q3/K3 says nothing by itself about Q3/K2048, Q7, or below-cutoff Torch calls.

If these fail, preserve the negative result; do not choose a best-fitting
fixture-specific tree or infer that a local BF16 output match validates it.

If the 2^26 oracle returns unresolved triples, repeating at 2^55 or 2^60 is a
useful precision diagnostic: these powers remain BF16-exact and now absorb
one even in binary64. Full FP64 or mixed FP64/FP32 trees can become resolvable
at those amplitudes. Therefore a recovered high-amplitude topology alone is
not evidence of FP32 node precision; compare triplets across amplitudes and
validate dense products against candidate precision assignments separately.
