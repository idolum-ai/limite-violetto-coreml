"""Actual Core ML saved-fixture, idle/reuse and error-return guard regression."""
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import coremltools as ct
from coreml_port.prediction_guard import install, stats
from coreml_port.cleanup_guard import stats as cleanup_stats

p = argparse.ArgumentParser()
p.add_argument('--attempt', required=True)
p.add_argument('--cycles', type=int, default=12)
a = p.parse_args()
out = ROOT / 'evidence/04e-runtime' / a.attempt
out.mkdir(parents=True, exist_ok=False)
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = {'status': 'starting', 'compute': 'CPU_ONLY', 'cycles': [],
          'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
              Path(__file__), ROOT/'coreml_port/prediction_guard.py',
              ROOT/'coreml_port/native/prediction_guard.mm',
              ROOT/'coreml_port/cleanup_guard.py',
              ROOT/'artifacts/04e-runtime/prediction_guard.dylib')},
          'installed': install()}
def save():
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
save()
package = ROOT/'artifacts/04d-cpu/full-candidate-4/8.mlpackage'
report['package_files'] = {str(f.relative_to(package)): digest(f) for f in sorted(package.rglob('*')) if f.is_file()}
cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
tiny = dict(np.load(ROOT/'evidence/04d-gpu/tiny3-productrows-group8/fixture.npz'))
long = dict(np.load(ROOT/'evidence/04d-cpu/long-group-locator/first-failure-inputs.npz'))
saved_states = dict(np.load(ROOT/'evidence/04d-cpu/long-group-locator/first-failure-state-before.npz'))
names = ('x', 've', 'positions', 'tap0', 'tap12', 'tap23', 'norm_divisor')
for cycle in range(a.cycles):
    values = tiny if cycle % 2 == 0 else long
    state = cm.make_state()
    if cycle % 2:
        for spec in cm.get_spec().description.state:
            kind = 'k_cache_' if 'k_cache_' in spec.name else 'v_cache_'
            suffix = spec.name.replace('.', '_').split('_cache_')[-1]
            matches = [v for k,v in saved_states.items() if k.endswith(kind+suffix)]
            assert len(matches) == 1
            state.write_state(spec.name, matches[0])
    got = cm.predict({n: values[n].copy() for n in names}, state=state)
    expected = values['expected_y'] if cycle % 2 == 0 else values['expected']
    row = {'cycle': cycle, 'fixture': 'tiny3' if cycle % 2 == 0 else 'long1026',
           'exact': bool(np.array_equal(got['y'], expected)),
           'max_abs': float(np.max(np.abs(got['y']-expected)))}
    report['cycles'].append(row)
    del state, got
    gc.collect()
    # Mix immediate reuse with idle intervals, permitting framework reset work.
    time.sleep((0., .05, .5, 2.)[cycle % 4])
    save()
    print(row, flush=True)
embedding = ct.models.MLModel(str(ROOT/'artifacts/04d-cpu/full-candidate-4/embedding.mlpackage'), compute_units=ct.ComputeUnit.CPU_ONLY)
ids = np.array([[87555,130664,114854]], dtype=np.int32)
embedding_input = lambda tokens: {'ids': tokens, 'norm_divisor': np.array([1280.], dtype=np.float32)}
first = embedding.predict(embedding_input(ids))
before = stats()['plain_calls']
try:
    embedding.predict(embedding_input(np.ones((1,33), dtype=np.int32)))
    report['invalid_shape_rejected'] = False
except Exception as exc:
    report['invalid_shape_rejected'] = True
    report['invalid_shape_error'] = {'type': type(exc).__name__, 'message': str(exc)}
report['error_reached_native_wrapper'] = stats()['plain_calls'] > before
after = embedding.predict(embedding_input(ids))
report['stateless_repeat_after_error_exact'] = all(np.array_equal(first[k],after[k]) for k in first)
del first, after, embedding, cm
gc.collect()
time.sleep(2)
report['prediction_guard'] = stats()
report['cleanup_guard'] = cleanup_stats()
report['pass'] = (len(report['cycles']) == a.cycles and a.cycles > 0 and
                  all(v['exact'] for v in report['cycles']) and
                  report['invalid_shape_rejected'] and report['error_reached_native_wrapper'] and
                  report['stateless_repeat_after_error_exact'] and
                  report['prediction_guard']['state_calls'] >= a.cycles and
                  report['prediction_guard']['plain_calls'] >= 3 and
                  report['prediction_guard']['gil_releases'] >= a.cycles+3)
report['status'] = 'complete'
save()
print(json.dumps({'pass': report['pass'], 'prediction_guard': stats()}), flush=True)
raise SystemExit(not report['pass'])
