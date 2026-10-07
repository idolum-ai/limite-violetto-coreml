"""Compare fixed-stream CPU inference without changing the qualified runtimes.

One backend per fresh process. No forced disk/compiler-cache clearing: startup
is a fresh-process measurement, not a claim of cold machine/cache behavior.
"""
from __future__ import annotations
import time
PROCESS_ENTRY = time.perf_counter()
import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANDIDATE = 'artifacts/04d-cpu/full-candidate-6'
IDENTITY = 'evidence/04d-cpu/full-candidate-6/package-identity.json'
MILESTONE = 'evidence/04d-cpu/candidate6-milestone.json'
EXPECTED_ID = '5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6'
LENGTHS = [32, 128, 512]
REPEATS = 3
DECODE = 8

def digest(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()

def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return {'exit_code': r.returncode, 'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()}
    except Exception as exc:
        return {'error': str(exc)}

def host():
    return {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'system': platform.platform(), 'machine': platform.machine(),
            'python': platform.python_version(), 'logical_cpus': os.cpu_count(),
            'load_average': os.getloadavg(),
            'hardware': command(['sysctl', 'hw.model', 'hw.memsize', 'hw.ncpu']),
            'thermal': command(['pmset', '-g', 'therm']),
            'power': command(['pmset', '-g', 'batt'])}

def usage():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return {'peak_rss_bytes': int(u.ru_maxrss if sys.platform == 'darwin' else u.ru_maxrss * 1024),
            'user_cpu_seconds': u.ru_utime, 'system_cpu_seconds': u.ru_stime}

def stats(values):
    return {'n': len(values), 'median': statistics.median(values), 'min': min(values), 'max': max(values)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['coreml', 'torch'], required=True)
    p.add_argument('--evidence-dir', required=True)
    args = p.parse_args()
    out = (ROOT / args.evidence_dir).resolve()
    if not out.is_relative_to(ROOT / 'evidence/05-practical'):
        p.error('Evidence must be under evidence/05-practical')
    out.mkdir(parents=True, exist_ok=False)
    milestone = json.loads((ROOT / MILESTONE).read_text())
    if not milestone['milestone_pass']:
        raise ValueError('Qualified candidate required')
    identity = json.loads((ROOT / IDENTITY).read_text())
    if identity['aggregate_sha256'] != EXPECTED_ID:
        raise ValueError('Candidate identity changed')
    fixture = ROOT / 'evidence/04-coreml/generation-comparison.json'
    reference = json.loads(fixture.read_text())['reference']
    stream = reference['prompt_ids'] + reference['generated_ids']
    assert len(stream) >= max(LENGTHS) + DECODE
    result = {'status': 'starting', 'backend': args.backend, 'host_before': host(),
              'protocol': {'lengths': LENGTHS, 'repeats': REPEATS, 'decode_tokens': DECODE,
                           'order': 'repeat outer, ascending prefix inner', 'warmup': '32-token prefill plus 2 teacher-forced decode calls',
                           'torch_threads': 4, 'compute': 'CPU_ONLY', 'sampling': 'none; fixed teacher-forced continuation',
                           'prefill_includes': 'runtime input preparation, backbone and final score head; excludes request creation',
                           'decode_includes': 'runtime input preparation, backbone and score head; excludes sampling/logging/metrics',
                           'startup_cache_state': 'uncontrolled; fresh process, no forced filesystem/compiler-cache eviction',
                           'memory_scope': 'process ru_maxrss high-water mark, not system-wide or accelerator memory',
                           'device_placement': 'CPU_ONLY permitted; no placement profiler'},
              'candidate_identity': EXPECTED_ID, 'milestone_sha256': digest(ROOT / MILESTONE),
              'fixture_sha256': digest(fixture), 'script_sha256': digest(__file__),
              'stream_ids': stream, 'usage_before_imports': usage(), 'trials': []}
    (out / 'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    def save():
        tmp = out / 'results.tmp'
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        tmp.replace(out / 'results.json')
    save()
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from scripts.runtime_provenance import runtime_provenance
    result['runtime_provenance'] = runtime_provenance()
    if result['runtime_provenance'] != milestone['current_runtime_provenance']:
        raise ValueError('Runtime differs from the passing candidate milestone')
    if digest(ROOT / 'coreml_port/tiny_prefix_runtime.py') != milestone['current_tiny_prefix_runtime_sha256']:
        raise ValueError('Tiny-prefix adapter differs from the passing milestone')
    qualified_sources = json.loads((ROOT / 'evidence/04d-cpu/full-candidate-6/validation-CPU_ONLY-attempt.json').read_text())['source_hashes']
    if digest(ROOT / 'portable/model.py') != qualified_sources['portable/model.py']:
        raise ValueError('Portable reference differs from the qualified source')
    result['source_sha256'] = {n: digest(ROOT / n) for n in ['portable/model.py', 'coreml_port/tiny_prefix_runtime.py', 'coreml_port/package_identity.py', 'sources/model/config.json', 'sources/value-model/modeling_backbone.py']}
    result['versions'] = {'torch': torch.__version__, 'numpy': np.__version__}
    result['entry_to_imports_seconds'] = time.perf_counter() - PROCESS_ENTRY
    print('Loading', args.backend, flush=True)
    save()
    started = time.perf_counter()
    try:
        if args.backend == 'coreml':
            from coreml_port.tiny_prefix_runtime import TinyPrefixPipeline
            model = TinyPrefixPipeline('CPU_ONLY', ROOT / CANDIDATE, ROOT / IDENTITY)
            result['component_load_seconds'] = model.load_seconds
            result['package_identity'] = model.package_identity
            create = model.request
            forward = model.forward
            close = lambda req: req.close()
            disposed = lambda req: req.closed and not req.states
        else:
            from portable.model import load_model
            model = load_model('cpu')
            result['checkpoint_sha256'] = model.load_report['checkpoint_sha256']
            create = model.new_cache
            def forward(ids, req):
                with torch.inference_mode():
                    h = model(torch.tensor([ids], dtype=torch.long), req)
                    return model.scores(h[:, -1:]).numpy()
            close = lambda req: req.reset()
            disposed = lambda req: req.length == 0 and all(x is None for x in req.keys + req.values)
    except BaseException as exc:
        result.update(status='failed', failure={'stage': 'model_load', 'type': type(exc).__name__, 'message': str(exc)})
        save()
        raise
    result['model_load_seconds'] = time.perf_counter() - started
    result['entry_to_ready_seconds'] = time.perf_counter() - PROCESS_ENTRY
    result['usage_after_load'] = usage()
    result['status'] = 'running'
    save()
    arrays = {}
    def run_request(length, decode_count, repeat, warmup=False):
        start = time.perf_counter(); req = create(); creation = time.perf_counter() - start
        trial = {'prefix_length': length, 'repeat': repeat, 'warmup': warmup, 'request_create_seconds': creation, 'steps': []}
        result['trials'].append(trial)
        try:
            for step in range(decode_count + 1):
                ids = stream[:length] if step == 0 else [stream[length + step - 1]]
                t = time.perf_counter(); scores = forward(ids, req); elapsed = time.perf_counter() - t
                if not np.isfinite(scores).all(): raise FloatingPointError('Nonfinite score output')
                record = {'step': step, 'prefix_length': length + step, 'seconds': elapsed,
                          'top1': int(scores.reshape(-1)[:151667].argmax()),
                          'scores_sha256': hashlib.sha256(scores.tobytes()).hexdigest()}
                trial['steps'].append(record)
                if not warmup and repeat == 0:
                    arrays[f'p{length}_s{step}'] = scores.copy()
                save()
        finally:
            t = time.perf_counter(); close(req); trial['dispose_seconds'] = time.perf_counter() - t
            trial['disposed'] = bool(disposed(req)); trial['usage_after'] = usage(); save()
        if not trial['disposed']: raise RuntimeError('Request disposal failed')
        print('trial', length, repeat, 'prefill', trial['steps'][0]['seconds'], 'decode median', statistics.median(x['seconds'] for x in trial['steps'][1:]), flush=True)
    try:
        run_request(32, 2, -1, warmup=True)
        for repeat in range(REPEATS):
            for length in LENGTHS:
                run_request(length, DECODE, repeat)
        np.savez(out / 'scores.npz', **arrays)
        result['scores_sha256'] = digest(out / 'scores.npz')
        result['summary_by_prefix'] = {}
        for length in LENGTHS:
            trials = [t for t in result['trials'] if not t['warmup'] and t['prefix_length'] == length]
            prefill = [t['steps'][0]['seconds'] for t in trials]
            decode = [s['seconds'] for t in trials for s in t['steps'][1:]]
            result['summary_by_prefix'][str(length)] = {'prefill_seconds': stats(prefill),
                'prefill_tokens_per_second': length / statistics.median(prefill),
                'request_create_plus_prefill_seconds': stats([t['request_create_seconds'] + t['steps'][0]['seconds'] for t in trials]),
                'decode_seconds': stats(decode), 'decode_tokens_per_second': len(decode) / sum(decode)}
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        result['host_after'] = host(); result['usage_final'] = usage()
        result['entry_to_finish_seconds'] = time.perf_counter() - PROCESS_ENTRY
        save()
    print(json.dumps(result['summary_by_prefix'], indent=2), flush=True)

if __name__ == '__main__':
    main()
