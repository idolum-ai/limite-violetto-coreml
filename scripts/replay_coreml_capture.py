"""Run an ordinary converted group against a saved input/state fixture."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import coremltools as ct
from coreml_port.prediction_guard import install
from scripts.runtime_provenance import runtime_provenance

p = argparse.ArgumentParser()
p.add_argument('--package', required=True)
p.add_argument('--inputs', required=True)
p.add_argument('--states')
p.add_argument('--evidence-dir', required=True)
a = p.parse_args()
install()
out = ROOT / a.evidence_dir
out.mkdir(parents=True, exist_ok=False)
package = ROOT / a.package
source = ROOT / a.inputs
saved = dict(np.load(source))
names = ('x', 've', 'positions', 'tap0', 'tap12', 'tap23', 'norm_divisor')
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = {
    'status': 'starting', 'compute': 'CPU_ONLY', 'package': str(package),
    'script_sha256': digest(Path(__file__)),
    'package_files': {str(f.relative_to(package)): digest(f) for f in sorted(package.rglob('*')) if f.is_file()},
    'input_file': str(source), 'input_sha256': digest(source),
    'positions': saved['positions'].tolist(),
    'runtime_provenance': runtime_provenance(),
}
(out / 'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
def save():
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
save()
cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
state = cm.make_state()
if a.states:
    state_path = ROOT / a.states
    values = dict(np.load(state_path))
    report.update(state_file=str(state_path), state_sha256=digest(state_path))
    for spec in cm.get_spec().description.state:
        kind = 'k_cache_' if 'k_cache_' in spec.name else 'v_cache_'
        suffix = spec.name.replace('.', '_').split('_cache_')[-1]
        matches = [value for key, value in values.items() if key.endswith(kind + suffix)]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous or missing state for {spec.name}')
        state.write_state(spec.name, matches[0])
got = cm.predict({n: saved[n] for n in names}, state=state)
def compare(actual, expected):
    bad = np.argwhere(actual != expected)
    return {'different_elements': int(len(bad)),
            'max_abs': float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))),
            'exact': bool(np.array_equal(actual, expected))}
report['y'] = compare(got['y'], saved['expected'] if 'expected' in saved else saved['expected_y'])
tap_key = next((k for k in ('expected_tap23', 'expected_tap23_out') if k in saved), None)
if tap_key:
    report['tap23'] = compare(got['tap23_out'], saved[tap_key])
report['pass'] = report['y']['exact'] and report.get('tap23', {'exact': True})['exact']
report['status'] = 'complete'
save()
np.savez_compressed(out / 'outputs.npz', **got)
print(json.dumps({k: report[k] for k in ('positions', 'y', 'pass')}), flush=True)
raise SystemExit(not report['pass'])
