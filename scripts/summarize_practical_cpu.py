"""Compare the two matched fixed-stream pilots; do not infer task accuracy."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summarize(coreml_path, torch_path, output):
    cm = json.loads(coreml_path.read_text()); pt = json.loads(torch_path.read_text())
    errors = []
    for backend, value in [('coreml', cm), ('torch', pt)]:
        if value.get('status') != 'completed' or value.get('backend') != backend:
            errors.append(f'{backend}: not completed or wrong backend')
        if len(value.get('trials', [])) != 10:
            errors.append(f'{backend}: expected one warmup and nine timed trials')
        for trial in value.get('trials', []):
            if not trial.get('disposed'): errors.append(f'{backend}: request not disposed')
            expected = 3 if trial['warmup'] else 9
            if len(trial['steps']) != expected: errors.append(f'{backend}: incomplete trial')
        if sha(coreml_path.parent / 'source-snapshot.py' if backend == 'coreml' else torch_path.parent / 'source-snapshot.py') != value['script_sha256']:
            errors.append(f'{backend}: script snapshot mismatch')
    for key in ['protocol', 'candidate_identity', 'milestone_sha256', 'fixture_sha256', 'script_sha256', 'stream_ids', 'runtime_provenance', 'source_sha256', 'versions']:
        if cm.get(key) != pt.get(key): errors.append(f'Backend mismatch: {key}')
    for key in ['system', 'machine', 'python', 'logical_cpus']:
        if cm['host_before'][key] != pt['host_before'][key]: errors.append(f'Host mismatch: {key}')
    if errors: raise ValueError('; '.join(errors))
    score_files = [coreml_path.parent / 'scores.npz', torch_path.parent / 'scores.npz']
    for data, path in zip([cm, pt], score_files):
        if sha(path) != data['scores_sha256']: raise ValueError('Score artifact hash mismatch')
    left, right = [np.load(p) for p in score_files]
    expected_keys = {f'p{length}_s{step}' for length in cm['protocol']['lengths'] for step in range(cm['protocol']['decode_tokens'] + 1)}
    if set(left.files) != expected_keys or set(right.files) != expected_keys: raise ValueError('Score coverage mismatch')
    comparisons = []
    for key in sorted(expected_keys):
        a = left[key].reshape(-1)[:151667]; b = right[key].reshape(-1)[:151667]
        if a.shape != b.shape or len(a) != 151667 or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid score shape/values')
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        top = np.argpartition(b, -2)[-2:]; vals = sorted(b[top]); margin = float(vals[-1] - vals[-2])
        top1 = int(a.argmax()) == int(b.argmax())
        maximum = float(diff.max())
        comparisons.append({'key': key, 'max_abs': maximum, 'top1_match': top1, 'reference_margin': margin,
                            'passed': maximum <= .04 and (top1 or margin <= .08)})
    def warm_stats(data, length):
        trials = [t for t in data['trials'] if not t['warmup'] and t['prefix_length'] == length]
        def describe(values):
            return {'n': len(values), 'median': statistics.median(values), 'min': min(values), 'max': max(values)}
        warm = [t for t in trials if t['repeat'] > 0]
        first = next(t for t in trials if t['repeat'] == 0)
        prefill = [t['steps'][0]['seconds'] for t in warm]
        decode = [step['seconds'] for t in warm for step in t['steps'][1:]]
        return {'first_shape_trial_prefill_seconds': first['steps'][0]['seconds'],
                'first_shape_trial_decode_seconds': [x['seconds'] for x in first['steps'][1:]],
                'prefill_seconds': describe(prefill),
                'prefill_tokens_per_second': length / statistics.median(prefill),
                'request_create_plus_prefill_seconds': describe([t['request_create_seconds'] + t['steps'][0]['seconds'] for t in warm]),
                'decode_seconds': describe(decode), 'decode_tokens_per_second': len(decode) / sum(decode),
                'per_trial_decode_totals_seconds': [sum(x['seconds'] for x in t['steps'][1:]) for t in warm],
                'per_trial_inference_plus_state_lifecycle_seconds': [t['request_create_seconds'] + sum(x['seconds'] for x in t['steps']) + t['dispose_seconds'] for t in warm]}
    rows = []
    for length in cm['protocol']['lengths']:
        a = warm_stats(cm, length); b = warm_stats(pt, length)
        rows.append({'prefix_length': length, 'coreml': a, 'torch': b,
                     'coreml_over_torch_prefill_latency': a['prefill_seconds']['median'] / b['prefill_seconds']['median'],
                     'coreml_over_torch_decode_latency': a['decode_seconds']['median'] / b['decode_seconds']['median']})
    result = {'status': 'completed', 'scope': 'One fresh process per backend; three within-process trials; first shape trial separate, repeats1-2 used for warm summaries. Descriptive pilot, not a cold-start distribution or universal usefulness gate.',
              'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [coreml_path, torch_path]},
              'script_sha256': sha(__file__), 'matching_protocol': True,
              'numeric_comparisons': comparisons, 'numerics_pass': all(x['passed'] for x in comparisons),
              'performance_pass_threshold': None, 'performance_threshold_note': 'No user SLA was specified; report relative costs without inventing a pass threshold.',
              'startup': {}, 'memory': {}, 'first_request': {}, 'rows': rows}
    for name, d in [('coreml', cm), ('torch', pt)]:
        result['startup'][name] = {k: d[k] for k in ['entry_to_imports_seconds', 'model_load_seconds', 'entry_to_ready_seconds']}
        result['memory'][name] = {'process_peak_rss_bytes': d['usage_final']['peak_rss_bytes'], 'process_peak_rss_gib': d['usage_final']['peak_rss_bytes'] / 2**30}
        t = d['trials'][0]
        result['first_request'][name] = {'prefill_seconds': t['steps'][0]['seconds'], 'request_create_seconds': t['request_create_seconds'], 'decode_seconds': [x['seconds'] for x in t['steps'][1:]]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coreml', required=True); p.add_argument('--torch', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); result = summarize((ROOT / a.coreml).resolve(), (ROOT / a.torch).resolve(), (ROOT / a.output).resolve())
    print(json.dumps({'numerics_pass': result['numerics_pass'], 'comparisons': len(result['numeric_comparisons']), 'startup': result['startup'], 'memory': result['memory'], 'ratios': [{k: v for k, v in row.items() if k not in ['coreml', 'torch']} for row in result['rows']]}, indent=2))
    return 0 if result['numerics_pass'] else 1
if __name__ == '__main__': sys.exit(main())
