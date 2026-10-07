"""Independently verify saved reciprocal group evidence; never runs a model."""
from __future__ import annotations
import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = 'evidence/06d-gpu-rounding/group0-1/results.json'
PRIOR = 'evidence/06d-gpu-rounding/group0-replay-1/results.json'
TAP_PRIOR = 'evidence/06e-gpu-normalization/replay-1/results.json'
WORKER = 'scripts/probe_boundary_reciprocal_group.py'
STATES = tuple(f'inner_{kind}_cache_{layer}' for layer in range(4) for kind in ('k', 'v'))
ENDPOINTS = ('y', 'tap23_out')
PRE = ('value_17', 'x_65', 'var_5669', 'mean_7', 'var_5672', 'denom_7')


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    assert isinstance(value, dict)
    return value


def save(path, value):
    t = path.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    t.replace(path)


def resolve(value):
    p = Path(value)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    assert p.is_relative_to(ROOT)
    return p


def metrics(actual, expected):
    assert actual.dtype == expected.dtype == np.float32 and actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    a = np.ascontiguousarray(actual).view(np.uint32)
    b = np.ascontiguousarray(expected).view(np.uint32)
    changed = a != b
    numeric = actual != expected
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    first = None
    indices = np.flatnonzero(changed)
    if indices.size:
        ix = np.unravel_index(int(indices[0]), actual.shape)
        first = {'index': list(map(int, ix)), 'actual': float(actual[ix]), 'expected': float(expected[ix]),
                 'actual_bits': f'0x{int(a[ix]):08x}', 'expected_bits': f'0x{int(b[ix]):08x}'}
    return {'bitwise_exact': not bool(changed.any()), 'bitwise_error_count': int(changed.sum()),
            'numeric_error_count': int(numeric.sum()), 'rmse': rmse,
            'relative_rmse': rmse / rms if rms else (0. if rmse == 0 else None),
            'max_abs': float(np.max(np.abs(diff))), 'first_difference': first,
            'shape': list(actual.shape), 'dtype': str(actual.dtype)}


def compare_maps(actual, expected):
    assert set(actual) == set(expected)
    return {key: metrics(actual[key], expected[key]) for key in actual}


def exact(comparisons): return all(x['bitwise_exact'] for x in comparisons.values())


def verify_claims(computed, reported):
    assert set(computed) == set(reported)
    for key, value in computed.items():
        claim = reported[key]
        assert value['bitwise_exact'] == claim['bitwise_exact'], f'Incorrect byte claim:{key}'
        if 'exact' in claim:
            assert (value['numeric_error_count'] == 0) == claim['exact']
        for metric in ('rmse', 'relative_rmse', 'max_abs'):
            if metric in claim:
                if value[metric] is None:
                    assert claim[metric] is None
                else:
                    assert math.isclose(value[metric], claim[metric], rel_tol=1e-12, abs_tol=1e-15), (key, metric)


@lru_cache(maxsize=None)
def fraction_of_bits(bits):
    return Fraction(*struct.unpack('<f', struct.pack('<I', bits))[0].as_integer_ratio())


@lru_cache(maxsize=None)
def reciprocal_bits(bits):
    target = 1 / fraction_of_bits(bits)
    lo, hi = 0, 0x7f7fffff
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fraction_of_bits(mid) <= target: lo = mid
        else: hi = mid - 1
    if fraction_of_bits(lo) == target: return lo
    down, up = target - fraction_of_bits(lo), fraction_of_bits(lo + 1) - target
    return lo if down < up or (down == up and lo % 2 == 0) else lo + 1


def reciprocal_oracle(d):
    assert d.dtype == np.float32 and np.isfinite(d).all() and (d > 0).all()
    return np.array([reciprocal_bits(int(x)) for x in d.ravel().view(np.uint32)], np.uint32).view(np.float32).reshape(d.shape)


def keyed(rows, keys):
    result = {tuple(row[k] for k in keys): row for row in rows}
    assert len(result) == len(rows)
    return result


def summarize(path):
    r = read(path)
    assert r['status'] == 'completed'
    original_hash = sha(path)
    protocol_path = path.parent / 'protocol.json'
    protocol = read(protocol_path)
    assert sha(protocol_path) == r['protocol_sha256']
    assert r['pins'] == protocol['pins']
    required = {BASE, PRIOR, TAP_PRIOR, WORKER, protocol['build']}
    assert required <= set(r['pins'])
    for name, digest in r['pins'].items(): assert sha(resolve(name)) == digest, name
    assert sha(path.parent / 'source-snapshot.py') == r['pins'][WORKER]
    build_path = resolve(protocol['build'])
    assert sha(path.parent / 'build-snapshot.json') == sha(build_path)
    build, base, prior, taps_prior = map(lambda p: read(resolve(p)), (protocol['build'], BASE, PRIOR, TAP_PRIOR))
    assert build['status'] == 'completed' and build['passed'] and build['graph_check']['passed']
    assert build['tapped_graph_check']['passed']
    assert prior['status'] == 'completed' and prior['cpu_preservation_passed']
    assert taps_prior['status'] == 'completed' and taps_prior['instrumentation_valid']
    source_hashes = [build[k] for k in ('source_build_sha256', 'source_result_sha256') if k in build]
    assert source_hashes and all(h == sha(resolve(BASE)) for h in source_hashes)
    assert build['source_files'] == base['variant_files']
    seed_names = [build['seed_name']] if 'seed_name' in build else []
    if 'seed_name' in build['tapped_graph_check']: seed_names.append(build['tapped_graph_check']['seed_name'])
    assert seed_names and len(set(seed_names)) == 1
    seed = seed_names[0]
    expected_taps = taps_prior['taps'] + [seed]
    assert r['taps'] == expected_taps and len(set(expected_taps)) == 13
    assert r['runtime_provenance'] == prior['runtime_provenance'] == taps_prior['runtime_provenance']
    for name, digest in r['runtime_provenance'].items(): assert sha(resolve(name)) == digest
    assert r['prediction_guard'] == {'installed': True, 'plain_calls': 0, 'state_calls': 12, 'gil_releases': 12}
    expected_cells = {(p, a, c) for p in ('CPU', 'GPU') for a in ('original', 'candidate', 'tapped') for c in (0, 1)}
    calls = keyed(r['calls'], ('policy', 'arm', 'chunk'))
    assert set(calls) == expected_cells
    assert all(c['restoration_exact'] is True for c in calls.values())
    verified = {}
    loaded = {}
    def load(meta):
        p = resolve(meta['path'])
        key = str(p)
        if key in loaded: return loaded[key]
        assert sha(p) == meta['sha256'], key
        with np.load(p, allow_pickle=False) as archive:
            data = {name: archive[name].copy() for name in archive.files}
        assert set(data) == set(meta['shapes']) == set(meta['dtypes'])
        for name, a in data.items():
            assert list(a.shape) == meta['shapes'][name] and str(a.dtype) == meta['dtypes'][name]
            assert np.isfinite(a).all()
        verified[str(p.relative_to(ROOT))] = meta['sha256']
        loaded[key] = data
        return data
    got = {}
    expected_labels = set()
    for (policy, arm, chunk), call in calls.items():
        label = f'chunk{chunk}/{policy}-{arm}'
        for field, suffix in [('outputs', 'outputs'), ('state_after', 'state-after')]:
            expected_labels.add(label + '/' + suffix)
            assert call[field] == r['files'][label + '/' + suffix]
        outputs, states = load(call['outputs']), load(call['state_after'])
        assert set(outputs) == (set(ENDPOINTS) | set(expected_taps) if arm == 'tapped' else set(ENDPOINTS))
        assert set(states) == set(STATES)
        got[(policy, arm, chunk)] = {'outputs': outputs, 'states': states}
    assert set(r['files']) == expected_labels
    old, old_taps = {}, {}
    for chunk in (0, 1):
        load(prior['files'][f'chunk{chunk}/inputs'])
        load(prior['files'][f'chunk{chunk}/before'])
        for policy in ('CPU', 'GPU'):
            old[(policy, chunk)] = {'outputs': load(prior['files'][f'chunk{chunk}/candidate-{policy}/outputs']),
                                    'states': load(prior['files'][f'chunk{chunk}/candidate-{policy}/state-after'])}
            old_taps[(policy, chunk)] = load(taps_prior['files'][f'chunk{chunk}/{policy}-tapped/outputs'])
    controls = []
    reported_controls = keyed(r['controls'], ('policy', 'chunk'))
    assert set(reported_controls) == {(p, c) for p in ('CPU', 'GPU') for c in (0, 1)}
    for (policy, chunk), claimed in reported_controls.items():
        original, candidate, tapped = (got[(policy, arm, chunk)] for arm in ('original', 'candidate', 'tapped'))
        checks = {'original_outputs_vs_saved': compare_maps(original['outputs'], old[(policy, chunk)]['outputs']),
                  'original_states_vs_saved': compare_maps(original['states'], old[(policy, chunk)]['states']),
                  'tapped_endpoints_vs_candidate': compare_maps({n: tapped['outputs'][n] for n in ENDPOINTS}, candidate['outputs']),
                  'tapped_states_vs_candidate': compare_maps(tapped['states'], candidate['states'])}
        for key, value in checks.items(): verify_claims(value, claimed[key])
        passed = all(exact(x) for x in checks.values())
        assert claimed['passed'] == passed
        controls.append({'policy': policy, 'chunk': chunk, 'passed': passed, **checks})
    instrumentation = all(x['passed'] for x in controls)
    assert r['instrumentation_valid'] == instrumentation
    cpu_checks, errors = [], []
    reported_cpu = keyed(r['cpu_preservation'], ('chunk',))
    reported_gpu = keyed(r['gpu_vs_original_cpu'], ('chunk',))
    assert set(reported_cpu) == set(reported_gpu) == {(0,), (1,)}
    for chunk in (0, 1):
        reference = got[('CPU', 'original', chunk)]
        cpu = got[('CPU', 'candidate', chunk)]
        candidate = got[('GPU', 'candidate', chunk)]
        old_gpu = got[('GPU', 'original', chunk)]
        check = {'outputs': compare_maps(cpu['outputs'], reference['outputs']), 'states': compare_maps(cpu['states'], reference['states'])}
        for key in check: verify_claims(check[key], reported_cpu[(chunk,)][key])
        passed = all(exact(x) for x in check.values())
        assert passed == reported_cpu[(chunk,)]['passed']
        cpu_checks.append({'chunk': chunk, 'passed': passed, **check})
        current_metrics = {key: compare_maps(candidate[key], reference[key]) for key in ('outputs', 'states')}
        before_metrics = {key: compare_maps(old_gpu[key], reference[key]) for key in ('outputs', 'states')}
        for key in current_metrics: verify_claims(current_metrics[key], reported_gpu[(chunk,)][key])
        old_rmse, new_rmse = before_metrics['outputs']['y']['rmse'], current_metrics['outputs']['y']['rmse']
        cache_summary = {}
        for tag, values in [('old_gpu', before_metrics), ('candidate_gpu', current_metrics)]:
            changed = [n for n in STATES if not values['states'][n]['bitwise_exact']]
            cache_summary[tag] = {'different_arrays': len(changed), 'different_elements': sum(values['states'][n]['bitwise_error_count'] for n in STATES),
                                  'changed_arrays': changed, 'first_changed': {'name': changed[0], **values['states'][changed[0]]['first_difference']} if changed else None}
        errors.append({'chunk': chunk, 'old_gpu_vs_original_cpu': before_metrics, 'candidate_gpu_vs_original_cpu': current_metrics,
                       'y_rmse_reduction_fraction': (old_rmse - new_rmse) / old_rmse if old_rmse else None,
                       'y_rmse_delta': old_rmse - new_rmse, 'cache_summary': cache_summary})
    cpu_pass = all(x['passed'] for x in cpu_checks)
    assert r['cpu_preservation_passed'] == cpu_pass
    stage_checks = []
    if instrumentation and cpu_pass:
        claims = keyed(r['stage_checks'], ('policy', 'chunk'))
        assert set(claims) == {(p, c) for p in ('CPU', 'GPU') for c in (0, 1)}
        for (policy, chunk), claimed in claims.items():
            actual = got[(policy, 'tapped', chunk)]['outputs']
            earlier = old_taps[(policy, chunk)]
            reference = reciprocal_oracle(actual['denom_7'])
            checks = {'pre_reciprocal_vs_saved': compare_maps({n: actual[n] for n in PRE}, {n: earlier[n] for n in PRE}),
                      'seed_vs_saved_reciprocal': compare_maps({'seed': actual[seed]}, {'seed': earlier['var_5675']}),
                      'corrected_vs_rational_oracle': compare_maps({'reciprocal': actual['var_5675']}, {'reciprocal': reference}),
                      'stages_vs_original_cpu': compare_maps({n: actual[n] for n in taps_prior['taps']}, {n: old_taps[('CPU', chunk)][n] for n in taps_prior['taps']})}
            for key, value in checks.items(): verify_claims(value, claimed[key])
            domain = bool(np.all((actual['denom_7'] >= .5) & (actual['denom_7'] <= 8)))
            near = bool(np.all(np.abs(actual[seed].view(np.uint32).astype(np.int64) - reference.view(np.uint32).astype(np.int64)) <= 1))
            assert domain == claimed['denominator_domain_passed'] and near == claimed['seed_within_one_neighbor']
            stage_checks.append({'policy': policy, 'chunk': chunk, 'denominator_domain_passed': domain, 'seed_within_one_neighbor': near, **checks})
        site = all(exact(x['pre_reciprocal_vs_saved']) and exact(x['seed_vs_saved_reciprocal']) and exact(x['corrected_vs_rational_oracle']) and x['denominator_domain_passed'] and x['seed_within_one_neighbor'] for x in stage_checks)
        warm = next(x for x in stage_checks if x['policy'] == 'GPU' and x['chunk'] == 1)
        warm_exact = exact(warm['stages_vs_original_cpu'])
    else:
        assert r['stage_checks'] is None
        site = warm_exact = False
    assert r['site_gate_passed'] == site and r['warm_normalization_exact'] == warm_exact
    passed = instrumentation and cpu_pass and site and warm_exact
    assert r['passed'] == passed
    assert sha(path) == original_hash
    return {'status': 'completed', 'evidence_verified': True, 'passed': passed,
            'result': str(path.relative_to(ROOT)), 'result_sha256': original_hash,
            'protocol_sha256': sha(protocol_path), 'source_pins_verified': r['pins'],
            'runtime_pins_verified': r['runtime_provenance'], 'verified_archives': verified,
            'calls_verified': 12, 'control_cells_verified': 4, 'prediction_guard': r['prediction_guard'],
            'instrumentation_valid': instrumentation, 'cpu_preservation_passed': cpu_pass,
            'site_gate_passed': site, 'warm_normalization_exact': warm_exact,
            'controls': controls, 'cpu_preservation': cpu_checks, 'error_changes': errors, 'stage_checks': stage_checks,
            'standalone_seed_reproduction_passed': build.get('standalone_seed_reproduction_passed'),
            'scope': 'Independent saved-array audit; group-local checks only. Artifact inventory records bound by result pins; model/package payloads not rehashed here. No full-model or performance qualification.',
            'actual_device_placement': 'unverified; compute policy only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output', default='evidence/06f-gpu-reciprocal/group-summary.json')
    args = parser.parse_args()
    path, output = resolve(args.result), resolve(args.output)
    assert path.is_relative_to(ROOT / 'evidence/06f-gpu-reciprocal')
    assert output.is_relative_to(ROOT / 'evidence/06f-gpu-reciprocal') and not output.exists()
    result = summarize(path)
    result['summary_script_sha256'] = sha(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, result)
    print(json.dumps({key: result[key] for key in ('status', 'evidence_verified', 'passed', 'instrumentation_valid', 'cpu_preservation_passed', 'site_gate_passed', 'warm_normalization_exact')}, indent=2))


if __name__ == '__main__': main()
