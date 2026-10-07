"""Recover a fixed FP32 binary sum tree from cancellation triplets.

No backend or model code is imported. A backend supplies evaluate(vectors),
where vectors has shape [batch,width] and the returned raw dot sums have
shape [batch]. Preserve the desired SGEMM geometry inside that callback.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

Tree = Union[int, tuple['Tree', 'Tree']]


def leaves(tree: Tree) -> list[int]:
    return [tree] if isinstance(tree, int) else leaves(tree[0]) + leaves(tree[1])


def representative(tree: Tree) -> int:
    while not isinstance(tree, int):
        tree = tree[0]
    return tree


def evaluate_tree(tree: Tree, vectors: np.ndarray) -> np.ndarray:
    if isinstance(tree, int):
        return vectors[..., tree].astype(np.float32)
    return np.add(evaluate_tree(tree[0], vectors), evaluate_tree(tree[1], vectors),
                  dtype=np.float32)


def triplet_vectors(triples, width: int, large: float = 2**26,
                    unit: float = 1, reverse_signs: bool = False):
    """Three rows per triple: cancellation pairs ab, ac, bc in that order."""
    vectors = np.zeros((3 * len(triples), width), dtype=np.float32)
    for t, (a, b, c) in enumerate(triples):
        if len({a, b, c}) != 3:
            raise ValueError('Triplet leaves must be distinct')
        for p, (i, j, k) in enumerate(((a,b,c), (a,c,b), (b,c,a))):
            vectors[3*t+p, i] = -large if reverse_signs else large
            vectors[3*t+p, j] = large if reverse_signs else -large
            vectors[3*t+p, k] = unit
    return vectors


@dataclass
class Oracle:
    evaluate: Callable[[np.ndarray], np.ndarray]
    width: int = 128
    large: float = 2**26
    unit: float = 1
    reverse_signs: bool = False
    vectors_used: int = 0
    calls: int = 0

    def winners(self, triples) -> list[tuple[int, int]]:
        vectors = triplet_vectors(triples, self.width, self.large, self.unit,
                                  self.reverse_signs)
        observed = np.asarray(self.evaluate(vectors), dtype=np.float32)
        self.calls += 1
        self.vectors_used += len(vectors)
        if observed.shape != (len(vectors),):
            raise ValueError(f'Expected raw dot shape {(len(vectors),)}, got {observed.shape}')
        observed = observed.reshape(-1, 3)
        out = []
        for triple, result in zip(triples, observed):
            good = result == np.float32(self.unit)
            if not np.all((result == 0) | good) or np.count_nonzero(good) != 1:
                raise ValueError(f'Not a binary FP32 triplet: {triple}, results={result.tolist()}')
            a,b,c = triple
            out.append(((a,b),(a,c),(b,c))[int(np.argmax(good))])
        return out


def reconstruct(oracle: Oracle, order=None) -> Tree:
    """Incremental rooted-tree insertion; at most 3*(n-1)*(n-2)/2 vectors.

    Leaves are unordered children: FP32 + is commutative for finite non-NaN
    values apart from zero-sign details. This recovers bracketing, not operand
    orientation or explicit additions of zero.
    """
    order = list(range(oracle.width)) if order is None else list(order)
    if sorted(order) != list(range(oracle.width)):
        raise ValueError('order must be a permutation of every leaf')
    if len(order) == 1:
        return order[0]
    tree = (order[0], order[1])
    def insert(node: Tree, x: int) -> Tree:
        if isinstance(node, int):
            return (node, x)
        a,b = representative(node[0]), representative(node[1])
        winner = set(oracle.winners([(a,b,x)])[0])
        if winner == {a,b}:
            return (node, x)
        if winner == {a,x}:
            return (insert(node[0], x), node[1])
        return (node[0], insert(node[1], x))
    for x in order[2:]:
        tree = insert(tree, x)
    return tree


def induced_pair(tree: Tree, triple) -> tuple[int, int]:
    """Topology-only expected answer, without any floating arithmetic."""
    selected = set(triple)
    def visit(node):
        if isinstance(node, int):
            return {node} & selected, None
        la,pa = visit(node[0]); lb,pb = visit(node[1])
        present = la | lb
        pair = pa or pb
        if pair is None and len(present) == 2:
            pair = tuple(sorted(present))
        return present, pair
    pair = visit(tree)[1]
    assert pair is not None
    return pair


def validate_triplets(tree: Tree, oracle: Oracle, count=1000, seed=917):
    rng = np.random.default_rng(seed)
    triples = [tuple(int(x) for x in rng.choice(oracle.width, 3, replace=False))
               for _ in range(count)]
    actual = oracle.winners(triples)
    wrong = [{'triple': t, 'expected': induced_pair(tree,t), 'actual': a}
             for t,a in zip(triples,actual) if set(induced_pair(tree,t)) != set(a)]
    return {'count':count, 'different':len(wrong), 'first_failures':wrong[:10]}


def left_fold(items):
    it = iter(items); tree = next(it)
    for item in it:
        tree = (tree,item)
    return tree


def self_test(width=128):
    cases = {'sequential':left_fold(range(width)),
             'reverse':left_fold(reversed(range(width)))}
    for lanes in (2,4,8,16):
        lane_trees = [left_fold(range(i,width,lanes)) for i in range(lanes)]
        cases[f'cyclic{lanes}'] = left_fold(lane_trees)
    rng = np.random.default_rng(411)
    random_trees = list(range(width))
    while len(random_trees)>1:
        i,j = sorted(rng.choice(len(random_trees),2,replace=False),reverse=True)
        a,b = random_trees.pop(int(i)),random_trees.pop(int(j))
        random_trees.append((a,b))
    cases['random'] = random_trees[0]
    # Products of two small integer-scaled values are exactly representable
    # as FP32. Include exponent diversity and cancellation.
    v = (rng.integers(-127,128,size=(500,width)).astype(np.float32) *
         np.exp2(rng.integers(-15,16,size=(500,width))).astype(np.float32))
    result = []
    for name,tree in cases.items():
        oracle = Oracle(lambda x,t=tree:evaluate_tree(t,x), width)
        recovered = reconstruct(oracle)
        reconstruction_vectors = oracle.vectors_used
        held_out = validate_triplets(recovered,oracle,1000)
        unequal = np.count_nonzero(evaluate_tree(tree,v) != evaluate_tree(recovered,v))
        assert held_out['different'] == 0 and unequal == 0
        result.append({'tree':name,'reconstruction_vectors':reconstruction_vectors,
                       'held_out_triplets':held_out,'random_vectors':len(v),
                       'different_random_sums':int(unequal)})
    # A full FP64 sum must fail, instead of fabricating an FP32 tree.
    try:
        reconstruct(Oracle(lambda x:x.sum(axis=1,dtype=np.float64).astype(np.float32),width))
    except ValueError:
        fp64_rejected = True
    else:
        raise AssertionError('FP64 incorrectly accepted')
    return {'width':width,'cases':result,'fp64_rejected':fp64_rejected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('Import this module with a backend Oracle, or use --self-test')
    report = json.dumps(self_test(),indent=2)+'\n'
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(report)
    print(report)
