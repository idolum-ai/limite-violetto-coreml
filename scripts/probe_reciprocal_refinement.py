"""Bounded reciprocal reproduction and compensated-Newton diagnostic.

Two isolated models x two policies x four frozen cases = 16 predictions.
Numerical failures are recorded through all calls; a failure to reproduce the
saved reciprocal prevents any correction claim. Never changes a group model.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'evidence/06e-gpu-normalization/replay-1/results.json'
HELPER = ROOT / 'coreml_port/source_exp.py'
POLICIES = ('CPU', 'GPU')
ARMS = ('control', 'diagnostic')
CASE_NAMES = ('warm-common', 'cold-cpu', 'cold-gpu', 'directed32')
LABELS = tuple(f'chunk{c}/{p}-tapped/outputs' for c in (0, 1) for p in POLICIES)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    assert isinstance(value, dict)
    return value


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def inventory(path):
    result = {}
    for p in sorted(path.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlink not allowed: {p}')
        if p.is_file():
            result[str(p.relative_to(path))] = sha(p)
        elif not p.is_dir():
            raise ValueError(f'Nonregular package file: {p}')
    return result


def float_of_bits(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def rational_of_bits(bits):
    return Fraction(*float_of_bits(bits).as_integer_ratio())


def nearest_positive_fp32_bits(value):
    """Exact positive-rational rounding, including ties-to-even; no FP divide."""
    assert value > 0 and value <= rational_of_bits(0x7f7fffff)
    low, high = 0, 0x7f7fffff
    while low < high:
        middle = (low + high + 1) // 2
        if rational_of_bits(middle) <= value:
            low = middle
        else:
            high = middle - 1
    lower = rational_of_bits(low)
    if lower == value:
        return low
    upper = rational_of_bits(low + 1)
    down, up = value - lower, upper - value
    return low if down < up or (down == up and low % 2 == 0) else low + 1


def oracle(d):
    import numpy as np
    assert d.dtype == np.float32 and np.isfinite(d).all() and (d > 0).all()
    output = np.empty(d.shape, dtype=np.uint32)
    for index in np.ndindex(d.shape):
        exact_d = Fraction(*float(d[index]).as_integer_ratio())
        output[index] = nearest_positive_fp32_bits(1 / exact_d)
    return output.view(np.float32)


def array_sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def compare(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {'bitwise_exact': False, 'shape_or_dtype_mismatch': True,
                'actual_shape': list(actual.shape), 'actual_dtype': str(actual.dtype)}
    a, b = np.ascontiguousarray(actual).view(np.uint32), np.ascontiguousarray(expected).view(np.uint32)
    bad = a != b
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    examples = [{'index': list(map(int, ix)), 'actual_bits': f'0x{int(a[tuple(ix)]):08x}',
                 'expected_bits': f'0x{int(b[tuple(ix)]):08x}'} for ix in np.argwhere(bad)[:8]]
    return {'bitwise_exact': not bool(bad.any()), 'different_elements': int(bad.sum()),
            'finite': finite, 'examples': examples,
            'max_abs': float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if finite else None}


def prepare_cases(source, archive_pins):
    import numpy as np
    arrays = {}
    for label in LABELS:
        metadata = source['files'][label]
        assert metadata == archive_pins[label]
        path = ROOT / metadata['path']
        assert sha(path) == metadata['sha256']
        with np.load(path, allow_pickle=False) as archive:
            pair = {name: archive[name].copy() for name in ('denom_7', 'var_5675')}
        for name, a in pair.items():
            assert a.dtype == np.float32 and list(a.shape) == metadata['shapes'][name]
            assert str(a.dtype) == metadata['dtypes'][name] and np.isfinite(a).all()
            assert a.shape == (1, 32 if label.startswith('chunk0') else 12, 1)
        arrays[label] = pair
    warm_cpu = arrays['chunk1/CPU-tapped/outputs']
    warm_gpu = arrays['chunk1/GPU-tapped/outputs']
    assert compare(warm_cpu['denom_7'], warm_gpu['denom_7'])['bitwise_exact']
    assert not compare(warm_cpu['var_5675'], warm_gpu['var_5675'])['bitwise_exact'], 'Saved warm drift absent'
    cases = [dict(name='warm-common', d=warm_cpu['denom_7'],
                  saved={'CPU': warm_cpu['var_5675'], 'GPU': warm_gpu['var_5675']}),
             dict(name='cold-cpu', d=arrays['chunk0/CPU-tapped/outputs']['denom_7'],
                  saved={'CPU': arrays['chunk0/CPU-tapped/outputs']['var_5675']}),
             dict(name='cold-gpu', d=arrays['chunk0/GPU-tapped/outputs']['denom_7'],
                  saved={'GPU': arrays['chunk0/GPU-tapped/outputs']['var_5675']})]
    directed = []
    for value in (1., 2., 4.):
        x = np.float32(value)
        directed.extend([np.nextafter(x, np.float32(-np.inf)), x, np.nextafter(x, np.float32(np.inf))])
    joined = np.concatenate([c['d'].ravel() for c in cases])
    for x in (joined.min(), joined.max()):
        directed.extend([np.nextafter(x, np.float32(-np.inf)), x, np.nextafter(x, np.float32(np.inf))])
    directed.extend(warm_cpu['denom_7'].ravel())
    directed.extend(map(np.float32, (1.25, 1.5, 1.75, 2.5, 3.)))
    assert len(directed) == 32
    cases.append(dict(name='directed32', d=np.asarray(directed, dtype=np.float32).reshape(1, 32, 1), saved={}))
    assert tuple(c['name'] for c in cases) == CASE_NAMES
    for c in cases:
        assert np.all(c['d'] >= .5) and np.all(c['d'] <= 8), 'Outside frozen safe denominator range'
        c['reference'] = oracle(c['d'])
    return cases


def host_check(cases):
    import numpy as np
    import torch
    from coreml_port.source_exp import fma
    results = []
    for case in cases:
        with torch.inference_mode():
            d = torch.from_numpy(case['d'].copy())
            r0 = torch.ones_like(d) / d
            e = fma(-d, r0, torch.ones_like(d))
            r1 = fma(r0, e, r0)
        arrays = {'r0': r0.numpy().copy(), 'residual': e.numpy().copy(), 'r1': r1.numpy().copy()}
        assert all(np.isfinite(x).all() for x in arrays.values())
        results.append({'case': case['name'], 'control_vs_oracle': compare(arrays['r0'], case['reference']),
                        'refined_vs_oracle': compare(arrays['r1'], case['reference']),
                        'cpu_bytes_preserved': compare(arrays['r1'], arrays['r0'])})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = read(protocol_path)
    assert args.attempt and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.attempt)
    assert args.attempt == protocol['attempt']
    assert protocol['expected_predictions'] == 16
    assert protocol['case_names'] == list(CASE_NAMES)
    assert protocol['arms'] == list(ARMS) and protocol['policies'] == list(POLICIES)
    assert protocol['script_sha256'] == sha(__file__)
    assert protocol['source_result_sha256'] == sha(SOURCE)
    assert protocol['helper_sha256'] == sha(HELPER)
    assert protocol['runtime_provenance_source_sha256'] == sha(ROOT / 'scripts/runtime_provenance.py')
    design = ROOT / 'evidence/06e-gpu-normalization/reciprocal-design.md'
    assert protocol['design_sha256'] == sha(design)
    source = read(SOURCE)
    assert source['status'] == 'completed' and source['passed'] and source['instrumentation_valid']
    assert len(source['controls']) == 4 and all(x['passed'] for x in source['controls'])
    out = ROOT / 'evidence/06e-gpu-normalization' / args.attempt
    artifacts = ROOT / 'artifacts/06e-gpu-normalization' / args.attempt
    if out.exists() or artifacts.exists():
        raise FileExistsError('Fresh evidence and artifact paths required')
    out.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    for name, path in [('source-snapshot.py', Path(__file__)), ('helper-snapshot.py', HELPER),
                       ('protocol.json', protocol_path), ('source-replay-snapshot.json', SOURCE),
                       ('design-snapshot.md', design), ('runtime-provenance-snapshot.py', ROOT / 'scripts/runtime_provenance.py')]:
        (out / name).write_bytes(path.read_bytes())
    pins = {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), HELPER, protocol_path, SOURCE,
                                                    design, ROOT / 'scripts/runtime_provenance.py')}
    report = {'status': 'starting', 'passed': False, 'pins': pins, 'protocol': protocol,
              'source_archives': protocol['source_archives'], 'cases': [], 'models': {}, 'calls': [],
              'actual_device_placement': 'unverified; compute policy only',
              'scope': 'Standalone bounded reciprocal diagnostic; no group or full-model qualification'}
    def update(): save(out / 'results.json', report)
    update()
    model = None
    try:
        import coremltools as ct
        import numpy as np
        import torch
        from torch import nn
        from coreml_port.source_exp import fma
        from coreml_port.prediction_guard import install, stats
        from coreml_port.cleanup_guard import stats as cleanup_stats
        from scripts.runtime_provenance import runtime_provenance
        torch.set_num_threads(4)
        cases = prepare_cases(source, protocol['source_archives'])
        assert {c['name']: array_sha(c['d']) for c in cases} == protocol['case_input_sha256']
        report['runtime_provenance'] = runtime_provenance()
        assert report['runtime_provenance'] == source['runtime_provenance']
        report['host_preflight'] = host_check(cases)
        assert all(x[k]['bitwise_exact'] for x in report['host_preflight']
                   for k in ('control_vs_oracle', 'refined_vs_oracle', 'cpu_bytes_preserved'))
        def archive(relative, arrays):
            path = out / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                    'shapes': {n: list(a.shape) for n, a in arrays.items()},
                    'dtypes': {n: str(a.dtype) for n, a in arrays.items()}}
        for case in cases:
            meta = archive(f'cases/{case["name"]}.npz',
                           {'d': case['d'], 'reference': case['reference'],
                            **{f'saved_{p}': v for p, v in case['saved'].items()}})
            report['cases'].append({'name': case['name'], 'archive': meta,
                                    'input_sha256': array_sha(case['d']),
                                    'saved_reproduction_policies': list(case['saved'])})
        update()
        install()
        assert stats() == {'installed': True, 'plain_calls': 0, 'state_calls': 0, 'gil_releases': 0}
        class Control(nn.Module):
            def forward(self, d): return torch.ones_like(d) / d
        class Diagnostic(nn.Module):
            def forward(self, d):
                r0 = torch.ones_like(d) / d
                e = fma(-d, r0, torch.ones_like(d))
                return r0, e, fma(r0, e, r0)
        for arm, graph in [('control', Control()), ('diagnostic', Diagnostic())]:
            with torch.inference_mode():
                traced = torch.jit.trace(graph.eval(), torch.from_numpy(cases[0]['d'].copy()), check_trace=False)
            names = ['r0'] if arm == 'control' else ['r0', 'residual', 'r1']
            converted = ct.convert(traced, inputs=[ct.TensorType(name='d', shape=(1, ct.RangeDim(1, 32, default=12), 1), dtype=np.float32)],
                                   outputs=[ct.TensorType(name=n, dtype=np.float32) for n in names],
                                   convert_to='mlprogram', compute_precision=ct.precision.FLOAT32,
                                   minimum_deployment_target=ct.target.macOS15,
                                   pass_pipeline=ct.PassPipeline.EMPTY, skip_model_load=True)
            spec = converted.get_spec()
            assert set(spec.mlProgram.functions) == {'main'}
            function = spec.mlProgram.functions['main']
            assert set(function.block_specializations) == {'CoreML8'}
            operations = list(function.block_specializations['CoreML8'].operations)
            counts = dict(Counter(o.type for o in operations))
            assert counts.get('real_div', 0) == 1 and not counts.get('inverse', 0)
            assert all(o.type in {'const', 'fill_like', 'real_div', 'mul', 'sub', 'add', 'identity'} and not o.blocks for o in operations)
            assert all(v.type.tensorType.dataType == 11 for o in operations if o.type != 'const' for v in o.outputs)
            division = next(o for o in operations if o.type == 'real_div')
            package = artifacts / f'{arm}.mlpackage'
            converted.save(str(package))
            compiled = artifacts / f'{arm}.mlmodelc'
            start = time.perf_counter()
            returned = ct.models.utils.compile_model(str(package), destination_path=str(compiled))
            assert Path(returned).resolve() == compiled.resolve()
            report['models'][arm] = {'package': str(package.relative_to(ROOT)), 'compiled': str(compiled.relative_to(ROOT)),
                                     'package_files': inventory(package), 'compiled_files': inventory(compiled),
                                     'compile_seconds': time.perf_counter() - start, 'operation_counts': counts,
                                     'division_operation_proto_text': str(division),
                                     'function_input_proto_text': [str(v) for v in function.inputs],
                                     'operation_sha256': hashlib.sha256(b''.join(o.SerializeToString(deterministic=True) for o in operations)).hexdigest()}
            update()
            del converted, traced, graph
        assert stats() == {'installed': True, 'plain_calls': 0, 'state_calls': 0, 'gil_releases': 0}
        native = {}
        for policy in POLICIES:
            unit = ct.ComputeUnit.CPU_ONLY if policy == 'CPU' else ct.ComputeUnit.CPU_AND_GPU
            for arm in ARMS:
                model = ct.models.CompiledMLModel(str(ROOT / report['models'][arm]['compiled']), compute_units=unit)
                for case in cases:
                    payload = {'d': case['d'].copy()}
                    actual = {n: a.copy() for n, a in model.predict(payload).items()}
                    native[(policy, arm, case['name'])] = actual
                    report['calls'].append({'policy': policy, 'arm': arm, 'case': case['name'],
                                            'archive': archive(f'native/{policy}/{arm}/{case["name"]}.npz', actual),
                                            'input_preserved': compare(payload['d'], case['d']),
                                            'outputs_match_schema': set(actual) == ({'r0'} if arm == 'control' else {'r0', 'residual', 'r1'}),
                                            'all_outputs_finite': all(bool(np.isfinite(a).all()) for a in actual.values())})
                    update()
                del model
                model = None
                gc.collect()
        report['prediction_guard'] = stats()
        report['cleanup_guard'] = cleanup_stats()
        assert stats() == {'installed': True, 'plain_calls': 16, 'state_calls': 0, 'gil_releases': 16}
        reproduction, diagnostic_controls, refinements, cpu_preservation, cpu_oracle = [], [], [], [], []
        for case in cases:
            name = case['name']
            for policy in POLICIES:
                control = native[(policy, 'control', name)]
                diagnostic = native[(policy, 'diagnostic', name)]
                if policy in case['saved']:
                    reproduction.append({'case': name, 'policy': policy, **compare(control['r0'], case['saved'][policy])})
                diagnostic_controls.append({'case': name, 'policy': policy, **compare(diagnostic['r0'], control['r0'])})
                refinements.append({'case': name, 'policy': policy, **compare(diagnostic['r1'], case['reference'])})
                if policy == 'CPU':
                    cpu_preservation.append({'case': name, **compare(diagnostic['r1'], control['r0'])})
                    cpu_oracle.append({'case': name, **compare(control['r0'], case['reference'])})
        checks = {'saved_control_reproduction': reproduction, 'diagnostic_baseline_preservation': diagnostic_controls,
                  'refinement_vs_exact_rational_oracle': refinements, 'cpu_reference_preservation': cpu_preservation,
                  'cpu_control_vs_exact_rational_oracle': cpu_oracle}
        gates = {name: all(row['bitwise_exact'] for row in rows) for name, rows in checks.items()}
        gates['all_calls_finite_schema_and_input_preserved'] = all(x['all_outputs_finite'] and x['outputs_match_schema'] and x['input_preserved']['bitwise_exact'] for x in report['calls'])
        report.update(comparisons=checks, gates=gates, native_predictions=len(report['calls']),
                      reproduced=bool(gates['saved_control_reproduction'] and gates['diagnostic_baseline_preservation']))
        for rec in report['models'].values():
            assert inventory(ROOT / rec['package']) == rec['package_files']
            assert inventory(ROOT / rec['compiled']) == rec['compiled_files']
        assert all(sha(ROOT / p) == h for p, h in pins.items())
        assert all(sha(ROOT / m['path']) == m['sha256'] for m in protocol['source_archives'].values())
        assert all(sha(ROOT / x['archive']['path']) == x['archive']['sha256'] for x in report['cases'] + report['calls'])
        assert runtime_provenance() == report['runtime_provenance']
        passed = all(gates.values()) and len(report['calls']) == 16
        report.update(status='completed', passed=passed, bounded_refinement_gate_passed=passed,
                      decision='bounded_candidate_passes_next_group_gate_required' if passed else
                               'nonreproduction_no_correction_claim' if not report['reproduced'] else 'reproduced_but_refinement_or_preservation_failed')
    except BaseException as exc:
        report.update(status='failed', passed=False, error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        model = None
        gc.collect()
        update()
    print(json.dumps({key: report[key] for key in ('status', 'passed', 'native_predictions', 'decision', 'gates')}, indent=2))


if __name__ == '__main__':
    main()
