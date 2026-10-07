"""Fixed-stream paired component timing of the qualified compiled CPU loader.

Runtime and graph sources remain unchanged. Instrumentation times Pipeline.predict,
including its input preparation, bridge and native prediction; it is not an
operation-level or device-placement trace.
"""
from __future__ import annotations
import time
ENTRY = time.perf_counter()
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_practical_cpu import host, usage

LENGTHS = [32, 128, 512]
REPEATS = 3
DECODE = 8
NAMES = ['embedding', *map(str, range(12)), 'head']
# Independently recorded in05-practical/comparison.json and its handoff.
PILOT_SHA256 = 'afff9529de181adf72be924ba09017eb3b7bf558362b153d0fad2c6f9b27d2e8'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if not args.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.attempt):
        parser.error('Attempt must be a lowercase path-safe slug')
    out = ROOT / 'evidence/05d-decode' / args.attempt
    out.mkdir(parents=True, exist_ok=False)
    source_hash = sha(__file__)
    (out / 'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    r = {'status': 'starting', 'script_sha256': source_hash,
         'helper_sha256': sha(ROOT/'scripts/benchmark_practical_cpu.py'),
         'host_before': host(), 'trials': [],
         'protocol': {'prefix_lengths': LENGTHS, 'repeats': REPEATS, 'decode_steps': DECODE,
          'warmup': '32-token prefill and two forced decode calls; excluded',
          'repeat0': 'Uninstrumented first-shape trial; excluded from warm comparison',
          'measured_repeats': [1, 2], 'profile_condition': '(step + repeat) % 2 == 0 for decode only',
          'pairing': 'Each prefix/step has one control and one profiled observation across repeats1/2',
          'order': 'repeat outer, ascending prefix inner; fresh request per trial',
          'compute': 'CPU_ONLY', 'torch_threads': 4, 'sampling': 'none; teacher-forced frozen stream',
          'timed_boundary': 'Entire forward; profiled calls additionally time each Pipeline.predict',
          'exclusions': 'Request creation, score checks, hashing, JSON and logging outside forward timer',
          'call_order_scope': '14 expected calls checked only on profiled decode forwards',
          'residual_scope': 'Everything outside component timers, including wrapper/list-append overhead',
          'limits': 'Single process; uncontrolled host/OS caches; condition confounded with repeat/order for each pair; descriptive sample; no operation-level attribution'}}
    def save():
        tmp = out/'results.tmp'; tmp.write_text(json.dumps(r, indent=2)+'\n'); tmp.replace(out/'results.json')
    save()
    model = None
    original_predict = None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        from coreml_port.loader_selection import create_pipeline, loader_provenance
        from scripts.runtime_provenance import runtime_provenance
        milestone_path = ROOT/'evidence/05c-compiled/milestone-rerun1.json'
        milestone = json.loads(milestone_path.read_text())
        if milestone.get('milestone_pass') is not True:
            raise ValueError('Passing compiled milestone required')
        selection = argparse.Namespace(compute='CPU_ONLY', tiny_prefix_replay=True,
            model_dir=str(ROOT/'artifacts/04d-cpu/full-candidate-6'),
            identity_manifest=str(ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json'),
            compiled_dir=str(ROOT/'artifacts/05c-compiled/full-cache-1'),
            compiled_manifest=str(ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json'))
        r['loader_provenance'] = loader_provenance(selection)
        r['runtime_provenance'] = runtime_provenance()
        r['tiny_prefix_sha256'] = sha(ROOT/'coreml_port/tiny_prefix_runtime.py')
        if r['loader_provenance'] != milestone['compiled_identity']['loader_provenance'] or \
           r['runtime_provenance'] != milestone['current_runtime_provenance'] or \
           r['tiny_prefix_sha256'] != milestone['current_tiny_prefix_runtime_sha256']:
            raise ValueError('Current loader/runtime differs from qualification')
        r['milestone_sha256'] = sha(milestone_path)
        fixture = ROOT/'evidence/04-coreml/generation-comparison.json'
        reference = json.loads(fixture.read_text())['reference']
        stream = reference['prompt_ids'] + reference['generated_ids']
        score_path = ROOT/'evidence/05-practical/coreml-pilot-1/scores.npz'
        pilot_path = score_path.parent/'results.json'
        if sha(pilot_path) != PILOT_SHA256:
            raise ValueError('Pilot results differ from independently recorded identity')
        pilot = json.loads(pilot_path.read_text())
        if pilot.get('status') != 'completed':
            raise ValueError('Completed pilot required')
        r['pilot_results_sha256'] = sha(pilot_path)
        if sha(fixture) != pilot['fixture_sha256'] or sha(score_path) != pilot['scores_sha256']:
            raise ValueError('Qualified score fixture changed')
        if pilot['candidate_identity'] != r['loader_provenance']['source_candidate_identity']:
            raise ValueError('Score fixture belongs to another candidate')
        r.update(fixture_sha256=sha(fixture), score_fixture_sha256=sha(score_path), stream_ids=stream)
        with np.load(score_path) as z:
            expected = {k: z[k].copy() for k in z.files}
        start = time.perf_counter(); model = create_pipeline(selection)
        r['initialization_seconds'] = time.perf_counter()-start
        r['entry_to_ready_seconds'] = time.perf_counter()-ENTRY
        r['package_identity'] = model.package_identity['aggregate_sha256']
        r['status'] = 'running'; save()
        original_predict = model.predict
        calls = []
        def timed_predict(name, inputs, state=None):
            start = time.perf_counter()
            try:
                return original_predict(name, inputs, state)
            finally:
                elapsed = time.perf_counter()-start
                calls.append({'name': name, 'seconds': elapsed})
        def trial(length, repeat, decode_count, warmup=False):
            req = model.request()
            t = {'prefix_length': length, 'repeat': repeat, 'warmup': warmup, 'steps': []}
            r['trials'].append(t)
            try:
                for step in range(decode_count+1):
                    profiled = not warmup and repeat > 0 and step > 0 and (step+repeat)%2 == 0
                    calls.clear()
                    model.predict = timed_predict if profiled else original_predict
                    ids = stream[:length] if step == 0 else [stream[length+step-1]]
                    started = time.perf_counter(); scores = model.forward(ids, req)
                    elapsed = time.perf_counter()-started
                    model.predict = original_predict
                    # All checks, copies, hashes and writes occur after the timer.
                    exact = bool(np.array_equal(scores, expected[f'p{length}_s{step}']))
                    record = {'step': step, 'prefix_length': length+step, 'profiled': profiled,
                              'seconds': elapsed, 'scores_exact': exact,
                              'scores_sha256': hashlib.sha256(scores.tobytes()).hexdigest(),
                              'calls': list(calls)}
                    if profiled:
                        record['component_seconds'] = sum(c['seconds'] for c in calls)
                        record['outside_predict_seconds'] = elapsed-record['component_seconds']
                        if [c['name'] for c in calls] != NAMES or record['outside_predict_seconds'] < 0:
                            raise RuntimeError('Unexpected predict call order/count or timing accounting')
                    t['steps'].append(record); save()
                    if not exact:
                        raise ValueError(f'Qualified score mismatch at {length}/{repeat}/{step}')
            finally:
                model.predict = original_predict
                req.close(); t['disposed'] = bool(req.closed and not req.states and not req.warmup_ids)
                save()
            if not t['disposed']:
                raise RuntimeError('Request disposal failed')
            print(json.dumps({'prefix': length, 'repeat': repeat, 'warmup': warmup,
                              'prefill_seconds': t['steps'][0]['seconds'],
                              'decode_median_seconds': statistics.median(s['seconds'] for s in t['steps'][1:]),
                              'all_scores_exact': all(s['scores_exact'] for s in t['steps'])}), flush=True)
        trial(32, -1, 2, True)
        for repeat in range(REPEATS):
            for length in LENGTHS:
                trial(length, repeat, DECODE)
        if sha(ROOT/'scripts/benchmark_practical_cpu.py') != r['helper_sha256'] or sha(ROOT/'coreml_port/tiny_prefix_runtime.py') != r['tiny_prefix_sha256'] or sha(__file__) != source_hash or loader_provenance(selection) != r['loader_provenance'] or runtime_provenance() != r['runtime_provenance']:
            raise ValueError('Sources changed during profiling')
        from coreml_port.prediction_guard import stats
        r['prediction_guard'] = stats()
        r.update(status='completed', passed=True)
    except BaseException as exc:
        r.update(status='failed', passed=False, failure={'type':type(exc).__name__, 'message':str(exc)})
        raise
    finally:
        if model is not None and original_predict is not None:
            model.predict = original_predict
        r['host_after'] = host(); r['usage_final'] = usage()
        r['entry_to_finish_seconds'] = time.perf_counter()-ENTRY
        save()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
