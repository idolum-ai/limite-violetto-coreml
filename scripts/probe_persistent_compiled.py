"""Isolated compiled-loading pilot. No production pipeline or model edits."""
from __future__ import annotations
import time
ENTRY = time.perf_counter()
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PACKAGE = ROOT / 'artifacts/04d-cpu/full-candidate-6/0.mlpackage'
IDENTITY = ROOT / 'evidence/04d-cpu/full-candidate-6/package-identity.json'
MILESTONE = ROOT / 'evidence/04d-cpu/candidate6-milestone.json'
INPUTS = ROOT / 'evidence/04d-audit/candidate4-capacity-locator/first-failure-inputs.npz'
STATES = ROOT / 'evidence/04d-audit/candidate4-capacity-locator/first-failure-state-before.npz'
EXPECTED = '5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def inventory(path):
    return {str(p.relative_to(path)): {'size': p.stat().st_size, 'sha256': sha(p)} for p in sorted(path.rglob('*')) if p.is_file()}

def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)

def conditions():
    r = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'platform': platform.platform(), 'python': platform.python_version(), 'load_average': os.getloadavg()}
    for name, cmd in [('thermal', ['pmset', '-g', 'therm']), ('power', ['pmset', '-g', 'batt'])]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            r[name] = {'exit_code': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
        except Exception as exc: r[name] = {'error': str(exc)}
    return r

def verify_source():
    identity = json.loads(IDENTITY.read_text())
    assert identity['aggregate_sha256'] == EXPECTED
    group = next(p for p in identity['packages'] if p['name'] == '0')
    expected = {str(Path(x['path']).relative_to('0.mlpackage')): {'size': x['size'], 'sha256': x['sha256']} for x in group['files']}
    if inventory(PACKAGE) != expected: raise ValueError('Sealed group package changed')
    from scripts.runtime_provenance import runtime_provenance
    runtime = runtime_provenance(); milestone = json.loads(MILESTONE.read_text())
    if not milestone['milestone_pass'] or runtime != milestone['current_runtime_provenance']:
        raise ValueError('Qualified runtime changed')
    return {'candidate_identity': EXPECTED, 'package_files': expected, 'runtime_provenance': runtime,
            'script_sha256': sha(__file__), 'input_sha256': sha(INPUTS), 'state_sha256': sha(STATES)}

def worker(args):
    out = ROOT / 'evidence/05b-startup' / args.attempt / args.child
    out.mkdir(parents=True, exist_ok=False)
    artifact = ROOT / 'artifacts/05b-startup' / args.attempt / '0.mlmodelc'
    result = {'status': 'starting', 'mode': args.mode, 'compute': 'CPU_ONLY', 'host_before': conditions()}
    target = out / 'results.json'; save(target, result)
    try:
        result.update(verify_source())
        from coreml_port.prediction_guard import install, stats
        result['guard_initial'] = install()
        import coremltools as ct
        import numpy as np
        result['coremltools_version'] = ct.__version__
        result['entry_to_setup_seconds'] = time.perf_counter() - ENTRY
        if args.mode == 'compile':
            if artifact.exists(): raise FileExistsError(artifact)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            returned = ct.models.utils.compile_model(str(PACKAGE), destination_path=str(artifact))
            result['compile_seconds'] = time.perf_counter() - start
            if Path(returned).resolve() != artifact.resolve(): raise ValueError('Unexpected compile destination')
            result['compiled_files'] = inventory(artifact)
            result['compiled_bytes'] = sum(x['size'] for x in result['compiled_files'].values())
        else:
            record = json.loads((out.parent / 'compile/results.json').read_text())
            if inventory(artifact) != record['compiled_files']: raise ValueError('Compiled artifact changed')
            # CompiledMLModel has no get_spec; inspect the sealed source package.
            spec = ct.models.utils.load_spec(str(PACKAGE))
            result['state_names'] = [s.name for s in spec.description.state]
            if len(result['state_names']) != 8: raise ValueError('Expected 8 group0 cache states')
            start = time.perf_counter()
            model = (ct.models.MLModel(str(PACKAGE), compute_units=ct.ComputeUnit.CPU_ONLY)
                     if args.mode == 'package' else ct.models.CompiledMLModel(str(artifact), compute_units=ct.ComputeUnit.CPU_ONLY))
            result['constructor_seconds'] = time.perf_counter() - start
            result['entry_to_loaded_seconds'] = time.perf_counter() - ENTRY
            start = time.perf_counter(); state = model.make_state()
            result['make_state_seconds'] = time.perf_counter() - start
            result['entry_to_state_ready_seconds'] = time.perf_counter() - ENTRY
            result['status'] = 'loaded'; save(target, result)
            if args.replay:
                with np.load(INPUTS) as values: fixture = dict(values)
                with np.load(STATES) as values: caches = dict(values)
                inputs = {v.name: fixture[v.name] for v in spec.description.input}
                result['positions'] = fixture['positions'].tolist()
                if set(caches) != set(result['state_names']): raise ValueError('Fixture state names mismatch')
                outputs = []
                cache_after = []
                prediction_seconds = []
                # Two independent states restored from the same captured native prefix.
                for trial in range(2):
                    current = state if trial == 0 else model.make_state()
                    for name in result['state_names']: current.write_state(name, caches[name])
                    before = {name: current.read_state(name).copy() for name in result['state_names']}
                    if not all(np.array_equal(before[name], caches[name]) for name in before):
                        raise ValueError('State restoration failed')
                    start = time.perf_counter(); actual = model.predict(inputs, state=current)
                    prediction_seconds.append(time.perf_counter() - start)
                    outputs.append({k: v.copy() for k, v in actual.items()})
                    cache_after.append({name: current.read_state(name).copy() for name in result['state_names']})
                    if trial == 1:
                        del current
                np.savez(out / 'outputs.npz', **outputs[0])
                np.savez(out / 'state-after.npz', **cache_after[0])
                result['native_y_exact'] = all(np.array_equal(x['y'], fixture['expected']) for x in outputs)
                result['max_native_y_error'] = max(float(np.abs(x['y'].astype(np.float64) - fixture['expected'].astype(np.float64)).max()) for x in outputs)
                result['independent_state_repeat_exact'] = all(np.array_equal(outputs[0][k], outputs[1][k]) for k in outputs[0])
                result['independent_cache_repeat_exact'] = all(np.array_equal(cache_after[0][k], cache_after[1][k]) for k in cache_after[0])
                result['prediction_seconds'] = prediction_seconds
                result['output_sha256'] = sha(out / 'outputs.npz'); result['cache_sha256'] = sha(out / 'state-after.npz')
                result['passed'] = result['native_y_exact'] and result['independent_state_repeat_exact'] and result['independent_cache_repeat_exact']
                del outputs, cache_after, actual, before
                # State objects are released below; no private native disposal API is claimed.
            else:
                result['passed'] = True
            del state
            del model
            gc.collect()
            result['python_references_released'] = True
        result['status'] = 'completed'
        result['passed'] = result.get('passed', True)
        result['guard_final'] = stats()
        result['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['entry_to_finish_seconds'] = time.perf_counter() - ENTRY
        result['host_after'] = conditions()
        save(target, result)
        print(json.dumps({k:v for k,v in result.items() if k in ['mode','constructor_seconds','compile_seconds','make_state_seconds','passed','native_y_exact','prediction_seconds']}, indent=2), flush=True)
        return 0 if result['passed'] else 1
    except BaseException as exc:
        result.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)})
        save(target, result)
        raise

def parent(args):
    out = ROOT / 'evidence/05b-startup' / args.attempt
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'running', 'scope': 'One sealed group; fresh-process load/state construction pilot, then bounded native capture replay. No pipeline adoption.',
              'protocol': {'sequence':['package','compiled','compiled','package','package','compiled'], 'cache_state':'uncontrolled OS caches; persistent artifact path reused; first compiled opening separately reported', 'guards':'both pinned guards', 'compute':'CPU_ONLY'}, 'source':verify_source(), 'runs':[]}
    target = out / 'results.json'; save(target, result)
    def run(mode, child, replay=False):
        command = [sys.executable, str(Path(__file__).resolve()), '--attempt', args.attempt, '--mode', mode, '--child', child]
        if replay: command += ['--replay']
        start = time.perf_counter()
        with (out / (child + '.log')).open('w') as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = {'name':child,'mode':mode,'replay':replay,'exit_code':proc.returncode,'process_wall_seconds':time.perf_counter()-start,'command':command}
        result['runs'].append(record); save(target,result)
        print(child, proc.returncode, round(record['process_wall_seconds'],3), flush=True)
        if proc.returncode: raise RuntimeError('Child failed: '+child)
    try:
        run('compile','compile')
        for i, mode in enumerate(result['protocol']['sequence']): run(mode, f'load-{i}-{mode}')
        run('package','replay-package',True)
        run('compiled','replay-compiled',True)
        import numpy as np
        checks = {}
        for name in ['outputs.npz','state-after.npz']:
            with np.load(out / 'replay-package' / name) as a, np.load(out / 'replay-compiled' / name) as b:
                checks[name] = set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
        artifact = ROOT / 'artifacts/05b-startup' / args.attempt / '0.mlmodelc'
        before = json.loads((out/'compile/results.json').read_text())['compiled_files']
        result['compiled_artifact_unchanged'] = inventory(artifact)==before
        result['source_unchanged'] = verify_source()==result['source']
        result['cross_loader_exact'] = checks
        result['passed'] = all(checks.values()) and result['compiled_artifact_unchanged'] and result['source_unchanged']
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='failed',passed=False,failure={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally: save(target,result)
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',required=True); parser.add_argument('--mode',choices=['compile','package','compiled']);parser.add_argument('--child');parser.add_argument('--replay',action='store_true')
    args = parser.parse_args()
    if not args.attempt.replace('-','').isalnum() or (args.child and not args.child.replace('-','').isalnum()): parser.error('Use simple attempt and child names')
    if bool(args.mode) != bool(args.child): parser.error('--mode and --child are paired internal arguments')
    sys.exit(worker(args) if args.mode else parent(args))
