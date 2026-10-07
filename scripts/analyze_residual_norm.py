"""Audit eight saved residual-normalization calls and characterize each reciprocal."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_boundary_reciprocal_group import (
    sha, read, resolve, save, metrics, compare_maps, exact, verify_claims,
    reciprocal_oracle, keyed, ENDPOINTS, STATES,
)
HELPER = 'scripts/summarize_boundary_reciprocal_group.py'
HELPER_SHA = '053d1a4bfd23e83f2d34775e2b9b593e3bef38b6d4bb1667ef036536e956721e'
BASE = 'evidence/06f-gpu-reciprocal/group0-1/results.json'
PRIOR = 'evidence/06f-gpu-reciprocal/group-replay-1/results.json'
FIXTURE = 'evidence/06d-gpu-rounding/group0-replay-1/results.json'
WORKER = 'scripts/probe_residual_norm_taps.py'


def distribution(values):
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(keys, counts)}


def reciprocal_analysis(cpu, gpu, denominator, reciprocal):
    results = {}
    for policy, arrays in [('CPU', cpu), ('GPU', gpu)]:
        d, r = arrays[denominator], arrays[reciprocal]
        assert d.dtype == r.dtype == np.float32 and d.shape == r.shape
        assert np.all(d > 0) and np.all(r > 0)
        reference = reciprocal_oracle(d)
        ulps = r.view(np.uint32).astype(np.int64) - reference.view(np.uint32).astype(np.int64)
        comparison = metrics(r, reference)
        results[policy] = {
            'denominator_min': float(d.min()), 'denominator_max': float(d.max()),
            'denominator_domain_0_5_to_8': bool(np.all((d >= .5) & (d <= 8))),
            'seed_within_one_neighbor': bool(np.all(np.abs(ulps) <= 1)),
            'seed_vs_exact_rational_reciprocal': comparison,
            'ulp_error_distribution': distribution(ulps),
        }
    same_d = cpu[denominator].view(np.uint32) == gpu[denominator].view(np.uint32)
    changed_r = cpu[reciprocal].view(np.uint32) != gpu[reciprocal].view(np.uint32)
    return {
        'denominator': denominator, 'reciprocal': reciprocal, 'policies': results,
        'equal_denominator_count': int(same_d.sum()),
        'different_denominator_count': int((~same_d).sum()),
        'reciprocal_differences_with_equal_denominator': int((same_d & changed_r).sum()),
        'both_policies_within_correction_contract': all(
            x['denominator_domain_0_5_to_8'] and x['seed_within_one_neighbor'] for x in results.values()),
        'cpu_reciprocal_exact': results['CPU']['seed_vs_exact_rational_reciprocal']['bitwise_exact'],
        'reference_scope': 'Exact rational 1/d, rounded to nearest FP32 ties-to-even independently for each saved policy denominator. No correctness claim about preceding square root or reduction.',
    }


def analyze(path):
    assert sha(ROOT / HELPER) == HELPER_SHA
    replay_hash = sha(path)
    r = read(path)
    assert r['status'] == 'completed'
    protocol_path = resolve(r['protocol_path'])
    protocol = read(protocol_path)
    assert sha(protocol_path) == r['protocol_sha256']
    assert protocol['pins'] == r['pins'] and protocol['expected_state_calls'] == 8
    assert {BASE, PRIOR, FIXTURE, WORKER} <= set(r['pins'])
    for name, digest in r['pins'].items(): assert sha(resolve(name)) == digest, name
    for name, digest in protocol['input_archives'].items(): assert sha(resolve(name)) == digest, name
    assert sha(path.parent / 'source-snapshot.py') == r['pins'][WORKER]
    snapshot = path.parent / 'build-snapshot.json'
    builds = [n for n, h in r['pins'].items() if h == sha(snapshot)]
    assert len(builds) == 1
    build_path = resolve(builds[0])
    b, base, prior, fixture = [read(p) for p in (build_path, ROOT / BASE, ROOT / PRIOR, ROOT / FIXTURE)]
    assert b['status'] == 'completed' and b['passed'] and b['graph_check']['passed']
    assert base['passed'] and prior['status'] == 'completed' and prior['cpu_preservation_passed']
    assert b['source_build_sha256'] == sha(ROOT / BASE) and b['source_files'] == base['variant_files']
    records = b['graph_check']['added_taps']
    taps = [x['name'] for x in records]
    assert taps == r['taps'] and len(set(taps)) == len(taps)
    stages = {x['stage']: x['name'] for x in records}
    assert len(stages) == len(records)
    assert r['runtime_provenance'] == prior['runtime_provenance']
    for name, digest in r['runtime_provenance'].items(): assert sha(resolve(name)) == digest
    assert r['prediction_guard'] == {'installed': True, 'plain_calls': 0, 'state_calls': 8, 'gil_releases': 8}
    calls = keyed(r['calls'], ('policy', 'arm', 'chunk'))
    assert set(calls) == {(p, a, c) for p in ('CPU', 'GPU') for a in ('original', 'tapped') for c in (0, 1)}
    assert all(c['restoration_exact'] is True for c in calls.values())
    verified, loaded = {}, {}
    def load(meta):
        p = resolve(meta['path'])
        if str(p) in loaded: return loaded[str(p)]
        assert sha(p) == meta['sha256']
        with np.load(p, allow_pickle=False) as z: data = {n: z[n].copy() for n in z.files}
        assert set(data) == set(meta['shapes']) == set(meta['dtypes'])
        for n, a in data.items():
            assert list(a.shape) == meta['shapes'][n] and str(a.dtype) == meta['dtypes'][n]
            assert np.isfinite(a).all()
        loaded[str(p)] = data
        verified[str(p.relative_to(ROOT))] = meta['sha256']
        return data
    got, expected_labels = {}, set()
    for (policy, arm, chunk), call in calls.items():
        label = f'chunk{chunk}/{policy}-{arm}'
        for field, suffix in [('outputs', 'outputs'), ('state_after', 'state-after')]:
            expected_labels.add(label + '/' + suffix)
            assert call[field] == r['files'][label + '/' + suffix]
        outputs, states = load(call['outputs']), load(call['state_after'])
        assert set(outputs) == (set(ENDPOINTS) | set(taps) if arm == 'tapped' else set(ENDPOINTS))
        assert set(states) == set(STATES)
        got[(policy, arm, chunk)] = {'outputs': outputs, 'states': states}
    assert set(r['files']) == expected_labels
    fixtures = {fixture['files'][f'chunk{c}/{kind}']['path']: fixture['files'][f'chunk{c}/{kind}']['sha256']
                for c in (0, 1) for kind in ('inputs', 'before')}
    assert all(protocol['input_archives'].get(n) == h for n, h in fixtures.items())
    for c in (0, 1):
        for kind in ('inputs', 'before'): load(fixture['files'][f'chunk{c}/{kind}'])
    controls = []
    reported = keyed(r['controls'], ('policy', 'chunk'))
    assert set(reported) == {(p, c) for p in ('CPU', 'GPU') for c in (0, 1)}
    for (policy, chunk), claim in reported.items():
        plain, tapped = (got[(policy, a, chunk)] for a in ('original', 'tapped'))
        for suffix in ('outputs', 'state-after'):
            meta = prior['files'][f'chunk{chunk}/{policy}-candidate/{suffix}']
            assert protocol['input_archives'].get(meta['path']) == meta['sha256']
        saved = {key: load(prior['files'][f'chunk{chunk}/{policy}-candidate/{suffix}'])
                 for key, suffix in [('outputs', 'outputs'), ('states', 'state-after')]}
        checks = {
            'original_outputs_vs_saved': compare_maps(plain['outputs'], saved['outputs']),
            'original_states_vs_saved': compare_maps(plain['states'], saved['states']),
            'tapped_endpoints_vs_original': compare_maps({n: tapped['outputs'][n] for n in ENDPOINTS}, plain['outputs']),
            'tapped_states_vs_original': compare_maps(tapped['states'], plain['states']),
        }
        for key, values in checks.items(): verify_claims(values, claim[key])
        passed = all(exact(v) for v in checks.values())
        assert claim['passed'] == passed
        controls.append({'policy': policy, 'chunk': chunk, 'passed': passed, **checks})
    control_pass = all(c['passed'] for c in controls)
    assert r['instrumentation_valid'] == r['passed'] == control_pass
    chunks = []
    if control_pass:
        comparisons = keyed(r['tap_comparisons'], ('chunk',))
        assert set(comparisons) == {(0,), (1,)}
        for chunk in (0, 1):
            cpu, gpu = (got[(p, 'tapped', chunk)]['outputs'] for p in ('CPU', 'GPU'))
            measured = compare_maps({n: gpu[n] for n in taps}, {n: cpu[n] for n in taps})
            verify_claims(measured, comparisons[(chunk,)]['comparisons'])
            first = next((x for x in records if not measured[x['name']]['bitwise_exact']), None)
            inputs = sorted({n for names in first.get('direct_inputs', {}).values() for n in names}) if first else []
            observed = {n: measured[n]['bitwise_exact'] for n in inputs if n in measured}
            chunks.append({
                'chunk': chunk, 'query_length': int(cpu[taps[0]].shape[1]),
                'first_differing_stage': first, 'stages': measured,
                'first_difference_observed_direct_inputs': observed,
                'first_difference_unobserved_direct_inputs': [n for n in inputs if n not in measured],
                'normalization_input_exact': measured[stages['normalization_input']]['bitwise_exact'],
                'reciprocal_reference': reciprocal_analysis(cpu, gpu, stages['square_root'], stages['original_reciprocal_seed']),
            })
    else:
        assert r['tap_comparisons'] is None
    assert sha(path) == replay_hash
    return {
        'status': 'completed', 'evidence_verified': True, 'controls_passed': control_pass,
        'interpretation_available': control_pass, 'script_sha256': sha(__file__),
        'analysis_helper': {'path': HELPER, 'sha256': HELPER_SHA},
        'replay': {'path': str(path.relative_to(ROOT)), 'sha256': replay_hash},
        'protocol': {'path': str(protocol_path.relative_to(ROOT)), 'sha256': sha(protocol_path)},
        'pins_verified': r['pins'], 'runtime_pins_verified': r['runtime_provenance'],
        'archive_sha256': verified, 'prediction_guard': r['prediction_guard'],
        'tap_records': records, 'controls': controls, 'chunks': chunks,
        'cpu_endpoint_state_preservation_passed': all(c['passed'] for c in controls if c['policy'] == 'CPU'),
        'actual_device_placement': 'unverified; compute policy only',
        'scope': 'Saved eight-call output-only group diagnostic. First differing exposed stage is observational and may omit intermediate operations. Reciprocal correctness applies to each saved denominator separately; neither upstream correctness nor downstream causality is inferred. Artifact inventories are bound by build pins, not payload-rehashed here. No model changes, native execution, full-model qualification or performance claim.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', default='evidence/06g-gpu-residual/replay-1/results.json')
    p.add_argument('--output', default='evidence/06g-gpu-residual/analysis.json')
    args = p.parse_args()
    replay, output = resolve(args.replay), resolve(args.output)
    assert output.is_relative_to(ROOT / 'evidence/06g-gpu-residual') and not output.exists()
    result = analyze(replay)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, result)
    print({'controls_passed': result['controls_passed'], 'first_differing_stages': [c['first_differing_stage']['name'] if c['first_differing_stage'] else None for c in result['chunks']]})


if __name__ == '__main__': main()
