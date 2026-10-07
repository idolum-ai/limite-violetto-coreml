"""Host-only exact arithmetic checks for the boundary reciprocal candidate.

No Core ML import, conversion, compilation, model loading, or native inference.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRIOR = ROOT / 'evidence/06e-gpu-normalization/reciprocal-host-1/host-preflight.json'
PRIOR_SHA = 'ea705cefe8bb118632088c63afa3c4f8b850a15b4180995c34079c47be64dd5b'
ORACLE = ROOT / 'scripts/probe_reciprocal_refinement.py'
ORACLE_SHA = '8eff2de8fd978fd850c8ab71ec1939b0c6c10b25800dd6c3b254052b11ddcaa3'
HELPER = ROOT / 'scripts/boundary_reciprocal.py'
DESIGN = ROOT / 'evidence/06f-gpu-reciprocal/candidate-design.md'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='host-1')
    args = parser.parse_args()
    assert args.attempt and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.attempt)
    assert sha(PRIOR) == PRIOR_SHA and sha(ORACLE) == ORACLE_SHA
    import numpy as np
    import torch
    from scripts.probe_reciprocal_refinement import nearest_positive_fp32_bits, rational_of_bits, compare
    from scripts.boundary_reciprocal import reciprocal_diagnostics
    torch.set_num_threads(4)
    out = ROOT / 'evidence/06f-gpu-reciprocal' / args.attempt
    out.mkdir(parents=True, exist_ok=False)
    pins = {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), HELPER, DESIGN, PRIOR, ORACLE)}
    for name, path in [('source-snapshot.py', Path(__file__)), ('helper-snapshot.py', HELPER),
                       ('design-snapshot.md', DESIGN), ('prior-host-snapshot.json', PRIOR),
                       ('oracle-source-snapshot.py', ORACLE)]:
        (out / name).write_bytes(path.read_bytes())
    report = {'status': 'starting', 'passed': False, 'pins': pins, 'native_calls': 0,
              'native_building': False, 'runtime_fp64': False,
              'implementation_qualified': 'Torch host helper only; MIL expression/native backend remain unqualified',
              'test_plan': {'prior_inputs': 108, 'boundary_window_radius_ulp': 256,
                            'boundary_centers': [.5, 1., 2., 4., 8.], 'boundary_inputs': 2053,
                            'random_seed': 20260924, 'random_inputs': 8192,
                            'random_distribution': 'uniform positive FP32 bit patterns in inclusive[0.5,8]',
                            'denominator_entries': 10353, 'neighbor_seed_runs': 31059,
                            'saved_native_seed_runs': 88, 'total_corrections': 31147}, 'runs': []}
    started = time.perf_counter()
    save(out / 'results.json', report)
    try:
        prior = json.loads(PRIOR.read_text())
        archived, captured, case_meta = [], [], []
        for record in prior['case_results']:
            path = ROOT / record['archive']['path']
            assert sha(path) == record['archive']['sha256']
            with np.load(path, allow_pickle=False) as source:
                d = source['d'].copy()
                for name in source.files:
                    if name.startswith('saved_'):
                        captured.append((record['case'] + '/' + name, d.copy(), source[name].copy()))
            archived.append(d.ravel())
            case_meta.append({'name': record['case'], 'archive': record['archive'], 'shape': list(d.shape)})
        old = np.concatenate(archived)
        assert len(old) == 108 and sum(d.size for _, d, _ in captured) == 88
        lo_bits, hi_bits = 0x3f000000, 0x41000000
        boundary_bits = []
        for center in (0x3f000000, 0x3f800000, 0x40000000, 0x40800000, 0x41000000):
            boundary_bits.extend(b for b in range(center - 256, center + 257) if lo_bits <= b <= hi_bits)
        boundary = np.array(boundary_bits, dtype=np.uint32).view(np.float32)
        assert boundary.size == 2053
        rng = np.random.default_rng(20260924)
        random = rng.integers(lo_bits, hi_bits + 1, size=8192, dtype=np.uint32).view(np.float32)
        denominators = np.concatenate((old, boundary, random))
        assert len(denominators) == 10353 and np.all((denominators >= .5) & (denominators <= 8))

        @lru_cache(maxsize=None)
        def fraction(bits): return rational_of_bits(int(bits))
        @lru_cache(maxsize=None)
        def reciprocal_bits(bits): return nearest_positive_fp32_bits(1 / fraction(bits))
        def signed_fraction(x): return Fraction(*float(x).as_integer_ratio())
        def exact_signed_bits(value):
            return 0 if value == 0 else nearest_positive_fp32_bits(abs(value)) | (0x80000000 if value < 0 else 0)
        reference_bits = np.array([reciprocal_bits(int(b)) for b in denominators.view(np.uint32)], dtype=np.uint32)
        reference = reference_bits.view(np.float32)
        np.savez_compressed(out / 'inputs.npz', prior=old, boundary=boundary, random=random,
                            d=denominators, reference=reference,
                            seed_previous=(reference_bits - 1).view(np.float32),
                            seed_following=(reference_bits + 1).view(np.float32))
        report['inputs'] = {'path': str((out / 'inputs.npz').relative_to(ROOT)), 'sha256': sha(out / 'inputs.npz'),
                            'prior_cases': case_meta, 'denominator_bytes_sha256': hashlib.sha256(denominators.tobytes()).hexdigest()}
        tests = [('nearest', denominators, reference),
                 ('previous', denominators, (reference_bits - 1).view(np.float32)),
                 ('following', denominators, (reference_bits + 1).view(np.float32)),
                 *[(name, d.ravel(), seed.ravel()) for name, d, seed in captured]]
        report['exact_identity_checks'] = 0
        save(out / 'results.json', report)
        print('Frozen cohort:10353 denominators;31147 supplied-seed corrections', flush=True)
        for run_index, (name, d, seed) in enumerate(tests):
            expected_bits = np.array([reciprocal_bits(int(b)) for b in d.view(np.uint32)], dtype=np.uint32)
            assert np.all(np.abs(seed.view(np.uint32).astype(np.int64) - expected_bits.astype(np.int64)) <= 1)
            with torch.inference_mode():
                observed = {key: value.numpy().copy() for key, value in reciprocal_diagnostics(torch.from_numpy(d.copy()), torch.from_numpy(seed.copy())).items()}
            assert all(np.isfinite(v).all() for v in observed.values())
            checks = compare(observed['corrected'], expected_bits.view(np.float32))
            bit_seed = seed.view(np.uint32)
            assert np.array_equal(observed['previous'].view(np.uint32), bit_seed - 1)
            assert np.array_equal(observed['following'].view(np.uint32), bit_seed + 1)
            assert np.array_equal(observed['even'], bit_seed % 2 == 0)
            assert not np.any(observed['go_up'] & observed['go_down'])
            for i in range(d.size):
                D, R = fraction(int(d.view(np.uint32)[i])), fraction(int(bit_seed[i]))
                P, E, H = (signed_fraction(observed[n][i]) for n in ('product', 'product_error', 'residual_hi'))
                assert P + E == D * R, (name, i, 'TwoProduct exactness')
                assert H == 1 - P, (name, i, 'Sterbenz residual')
                for side, sign in [('upper', 1), ('lower', -1)]:
                    step = signed_fraction(observed['up' if side == 'upper' else 'down'][i])
                    term = signed_fraction(observed[side + '_term'][i])
                    assert term == D * step / 2, (name, i, 'Exact midpoint scaling')
                    high, low = signed_fraction(observed[side + '_hi'][i]), signed_fraction(observed[side + '_lo'][i])
                    exact = E + sign * term
                    assert high + low == exact, (name, i, 'TwoSum exactness')
                    assert int(observed[side + '_hi'].view(np.uint32)[i]) == exact_signed_bits(exact), (name, i, 'TwoSum hi nearest rounding')
                    relation = (H > high) - (H < high) if H != high else ((0 > low) - (0 < low))
                    assert relation == ((H > exact) - (H < exact)), (name, i, 'Midpoint comparison')
                report['exact_identity_checks'] += 8
            target = out / f'run-{run_index}.npz'
            np.savez_compressed(target, d=d, seed=seed, reference=expected_bits.view(np.float32), **observed)
            result = {'name': name, 'elements': int(d.size), 'comparison': checks,
                      'supplied_seed_max_ulp_from_nearest': int(np.max(np.abs(bit_seed.astype(np.int64) - expected_bits.astype(np.int64)))),
                      'neighbor_spacing_and_parity_exact': True, 'expansion_identities_exact': True,
                      'midpoint_comparisons_exact': True,
                      'archive': {'path': str(target.relative_to(ROOT)), 'sha256': sha(target)}}
            report['runs'].append(result)
            save(out / 'results.json', report)
            print(name, d.size, checks['bitwise_exact'], flush=True)
        assert sum(x['elements'] for x in report['runs']) == 31147
        assert all(sha(ROOT / path) == digest for path, digest in pins.items())
        assert all(sha(ROOT / row['archive']['path']) == row['archive']['sha256'] for row in report['runs'])
        report.update(status='completed', passed=all(x['comparison']['bitwise_exact'] for x in report['runs']),
                      corrections=31147, seconds=time.perf_counter() - started,
                      cpu_reference_preservation=report['runs'][0]['comparison']['bitwise_exact'],
                      decision='Host finite-domain qualification only; native externally supplied-seed component gate required next')
    except BaseException as exc:
        report.update(status='failed', passed=False, error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        save(out / 'results.json', report)
    print(json.dumps({'passed': report['passed'], 'helper_sha256': pins[str(HELPER.relative_to(ROOT))],
                      'corrections': report['corrections'], 'native_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
