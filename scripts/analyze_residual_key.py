"""Audit eight saved key-path calls, reciprocal errors, and surviving rounded differences."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_boundary_reciprocal_group import (
    sha, read, resolve, save, metrics, compare_maps, exact, verify_claims,
    keyed, ENDPOINTS, STATES,
)
from scripts.analyze_residual_norm import reciprocal_analysis
RECIPROCAL_HELPER = 'scripts/analyze_residual_norm.py'
RECIPROCAL_HELPER_SHA = '467e8eaa1919fbc3e18018c8659cd6594149f91a3e7c3c4d1d77a347e28c772d'
HELPER = 'scripts/summarize_boundary_reciprocal_group.py'
HELPER_SHA = '053d1a4bfd23e83f2d34775e2b9b593e3bef38b6d4bb1667ef036536e956721e'
BASE = 'evidence/06f-gpu-reciprocal/group0-1/results.json'
PRIOR = 'evidence/06f-gpu-reciprocal/group-replay-1/results.json'
FIXTURE = 'evidence/06d-gpu-rounding/group0-replay-1/results.json'
WORKER = 'scripts/probe_residual_key_taps.py'



ROUNDED = ('value_41', 'k_49', 'var_14415', 'var_14438', 'k_51')
ROUND_PAIRS = (('x_197', 'value_41'), ('x_207', 'k_49'),
               ('x_215', 'var_14415'), ('x_217', 'var_14438'), ('x_219', 'k_51'))


def bit_changes(a, b):
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    return np.ascontiguousarray(a).view(np.uint32) != np.ascontiguousarray(b).view(np.uint32)


def rounded_analysis(cpu, gpu):
    rows = []
    for raw, rounded in ROUND_PAIRS:
        before, after = bit_changes(gpu[raw], cpu[raw]), bit_changes(gpu[rounded], cpu[rounded])
        # Linear output may be reshaped to [1,Q,heads,width] before quantization.
        assert before.size == after.size and before.shape[:2] == after.shape[:2]
        before = before.reshape(after.shape)
        rows.append({'raw': raw, 'rounded': rounded, 'raw_differences': int(before.sum()),
                     'rounded_differences': int(after.sum()),
                     'raw_differences_masked_at_corresponding_output': int((before & ~after).sum()),
                     'raw_differences_surviving_at_corresponding_output': int((before & after).sum()),
                     'rounded_differences_with_equal_raw_bits': int((~before & after).sum()),
                     'raw_different_but_all_rounded_equal': bool(before.any() and not after.any())})
    return rows



def quantization_analysis(cpu, gpu, quotient, rounded, step, output):
    qc, qg = cpu[quotient], gpu[quotient]
    assert qc.shape == qg.shape == cpu[rounded].shape == gpu[rounded].shape
    lo, hi = np.minimum(qc.astype(np.float64), qg), np.maximum(qc.astype(np.float64), qg)
    half = np.ceil(lo - .5) + .5
    crossing = (lo < hi) & (lo <= half) & (half <= hi)
    rounded_changes = bit_changes(gpu[rounded], cpu[rounded])
    quotient_changes = bit_changes(qg, qc)
    refs = {}
    for policy, arrays in [('CPU', cpu), ('GPU', gpu)]:
        q = arrays[quotient]
        # FP32 inputs widen exactly to Python binary64; round gives nearest
        # integer ties-to-even. Restore the input sign for integer zero.
        assert np.all(np.abs(q) < 2**24)
        nearest = np.array([round(float(x)) for x in q.ravel()], np.float32).reshape(q.shape)
        nearest = np.where(nearest == 0, np.copysign(nearest, q), nearest)
        refs[policy] = metrics(arrays[rounded], nearest)
    examples = []
    for ix in np.argwhere(rounded_changes)[:12]:
        ix = tuple(map(int, ix))
        examples.append({'index': list(ix), 'quotient_cpu': float(qc[ix]), 'quotient_gpu': float(qg[ix]),
                         'rounded_cpu': float(cpu[rounded][ix]), 'rounded_gpu': float(gpu[rounded][ix]),
                         'step_cpu': float(cpu[step][ix]), 'step_gpu': float(gpu[step][ix]),
                         'output_cpu': float(cpu[output][ix]), 'output_gpu': float(gpu[output][ix]),
                         'half_boundary_spanned_or_touched': bool(crossing[ix])})
    return {'quotient': quotient, 'rounded': rounded, 'step': step, 'output': output,
            'round_vs_exact_integer_ties_even': refs,
            'quotient_differences': int(quotient_changes.sum()),
            'rounded_differences': int(rounded_changes.sum()),
            'rounded_differences_with_equal_quotient': int((rounded_changes & ~quotient_changes).sum()),
            'rounded_differences_spanning_or_touching_half': int((rounded_changes & crossing).sum()),
            'step_differences': int(bit_changes(gpu[step], cpu[step]).sum()), 'examples': examples}


def direct_input_analysis(records, measured):
    result = []
    for record in records:
        if measured[record['name']]['bitwise_exact']: continue
        inputs = sorted({n for names in record.get('direct_inputs', {}).values() for n in names})
        observed = {n: measured[n]['bitwise_exact'] for n in inputs if n in measured}
        missing = [n for n in inputs if n not in measured]
        result.append({'name': record['name'], 'stage': record['stage'],
                       'observed_inputs': observed, 'unobserved_inputs': missing,
                       'all_direct_inputs_exposed_and_equal': bool(inputs) and not missing and all(observed.values())})
    return result


def analyze(path):
    assert sha(ROOT / HELPER) == HELPER_SHA
    assert sha(ROOT / RECIPROCAL_HELPER) == RECIPROCAL_HELPER_SHA
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
    assert base['passed'] and prior['status'] == 'completed' and prior['passed'] and prior['cpu_preservation_passed']
    assert b['source_build_sha256'] == sha(ROOT / BASE) and b['source_files'] == base['variant_files']
    records = b['graph_check']['added_taps']
    taps = [x['name'] for x in records]
    assert taps == r['taps'] and len(set(taps)) == len(taps)
    assert set(ROUNDED) | {n for pair in ROUND_PAIRS for n in pair} <= set(taps)
    stages = {x['stage']: x['name'] for x in records}
    assert len(stages) == len(records)
    indices = [x['operation_index'] for x in records]
    assert indices == sorted(set(indices))
    assert {'denom_21', 'var_14324'} <= set(taps)
    reciprocal_record = next(x for x in records if x['name'] == 'var_14324')
    assert reciprocal_record['operation_type'] == 'real_div'
    assert reciprocal_record['direct_inputs']['y'] == ['denom_21']
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
        if arm == 'tapped':
            for record in records:
                actual = outputs[record['name']]
                expected = record['mil_shape']
                assert actual.dtype == np.float32 and len(actual.shape) == len(expected)
                assert all(e is None or e == a for e, a in zip(expected, actual.shape))
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
            cache_metrics = compare_maps(got[('GPU', 'original', chunk)]['states'], got[('CPU', 'original', chunk)]['states'])
            changed_caches = [name for name in STATES if not cache_metrics[name]['bitwise_exact']]
            endpoint_metrics = compare_maps(got[('GPU', 'original', chunk)]['outputs'], got[('CPU', 'original', chunk)]['outputs'])
            first_index = taps.index(first['name']) if first else None
            direct = direct_input_analysis(records, measured)
            chunks.append({
                'chunk': chunk, 'query_length': int(cpu[taps[0]].shape[1]),
                'first_differing_stage': first, 'stages': measured,
                'first_difference_observed_direct_inputs': observed,
                'first_difference_unobserved_direct_inputs': [n for n in inputs if n not in measured],
                'preceding_exact_taps': taps[:first_index] if first else taps,
                'last_exact_tap_before_first_difference': records[first_index-1] if first_index else None,
                'endpoint_comparisons': endpoint_metrics,
                'rounding_transitions': rounded_analysis(cpu, gpu),
                'quantization_boundaries': [
                    quantization_analysis(cpu, gpu, 'var_14339', 'var_14340', 'step_149', 'k_49'),
                    quantization_analysis(cpu, gpu, 'var_14453', 'var_14454', 'step_161', 'k_51'),
                ],
                'first_differing_rounded_output': next((n for n in ROUNDED if not measured[n]['bitwise_exact']), None),
                'changed_operation_direct_inputs': direct,
                'first_changed_operation_with_all_direct_inputs_exposed_and_equal': next((x for x in direct if x['all_direct_inputs_exposed_and_equal']), None),
                'reciprocal_reference': reciprocal_analysis(cpu, gpu, 'denom_21', 'var_14324'),
                'cache_comparisons': cache_metrics,
                'cache_summary': {
                    'different_arrays': len(changed_caches), 'changed_arrays': changed_caches,
                    'different_elements': sum(v['bitwise_error_count'] for v in cache_metrics.values()),
                    'first_changed': {'name': changed_caches[0], **cache_metrics[changed_caches[0]]['first_difference']} if changed_caches else None,
                },
            })
    else:
        assert r['tap_comparisons'] is None
    assert sha(path) == replay_hash
    return {
        'status': 'completed', 'evidence_verified': True, 'controls_passed': control_pass,
        'interpretation_available': control_pass, 'script_sha256': sha(__file__),
        'analysis_helper': {'path': HELPER, 'sha256': HELPER_SHA},
        'reciprocal_helper': {'path': RECIPROCAL_HELPER, 'sha256': RECIPROCAL_HELPER_SHA},
        'replay': {'path': str(path.relative_to(ROOT)), 'sha256': replay_hash},
        'protocol': {'path': str(protocol_path.relative_to(ROOT)), 'sha256': sha(protocol_path)},
        'pins_verified': r['pins'], 'runtime_pins_verified': r['runtime_provenance'],
        'archive_sha256': verified, 'prediction_guard': r['prediction_guard'],
        'tap_records': records, 'controls': controls, 'chunks': chunks,
        'cpu_endpoint_state_preservation_passed': all(c['passed'] for c in controls if c['policy'] == 'CPU'),
        'actual_device_placement': 'unverified; compute policy only',
        'scope': 'Saved eight-call output-only group diagnostic. First differing exposed stage is observational and may omit intermediate operations. Rounded transition counts compare corresponding elements and do not imply independent causal effects. Boundary ordering does not prove a direct dependency between adjacent taps; neither upstream correctness nor downstream causality is inferred. The reciprocal oracle applies only to each saved denom_21; it does not establish correctness of preceding operations or causality for later differences. Artifact inventories are bound by build pins, not payload-rehashed here. No model changes, native execution, full-model qualification or performance claim.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', default='evidence/06g-gpu-residual/key-replay-1/results.json')
    p.add_argument('--output', default='evidence/06g-gpu-residual/key-analysis.json')
    args = p.parse_args()
    replay, output = resolve(args.replay), resolve(args.output)
    assert output.is_relative_to(ROOT / 'evidence/06g-gpu-residual') and not output.exists()
    result = analyze(replay)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, result)
    print({'controls_passed': result['controls_passed'], 'first_differing_stages': [c['first_differing_stage']['name'] if c['first_differing_stage'] else None for c in result['chunks']]})


if __name__ == '__main__': main()
