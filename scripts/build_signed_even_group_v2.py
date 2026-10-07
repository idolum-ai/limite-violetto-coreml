"""Create original group-0 signed-even v2 using division/multiplication sign transfer.

Only selected MIL round operations change. No model is loaded or predicted;
--compile is an explicit optional native compilation step for the root runner.
Numerical/domain qualification is separate from this structural build check.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'artifacts/04d-cpu/full-candidate-6'
SOURCE_ID = '5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6'
HELPER = ROOT / 'scripts/signed_even_round_v2.py'
HELPER_SHA256 = '298afbcd964531c0cb39a47e3b939da5306e9af585eec0e761a54d78252cd490'
PURE = frozenset({'const', 'abs', 'floor', 'sub', 'mul', 'real_div', 'equal',
                  'greater', 'cast', 'select', 'add'})
EXPECTED_NONCONST = {'abs': 1, 'floor': 2, 'sub': 2, 'mul': 2, 'real_div': 2,
                     'equal': 2, 'greater': 1, 'cast': 1, 'select': 2, 'add': 1}
MODEL_FILE = 'Data/com.apple.CoreML/model.mlmodel'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wire(message):
    return message.SerializeToString(deterministic=True)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def inventory(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def arguments(op):
    return [a.name for binding in op.inputs.values() for a in binding.arguments if a.name]


def block_of(spec):
    assert set(spec.mlProgram.functions) == {'main'}
    function = spec.mlProgram.functions['main']
    assert set(function.block_specializations) == {'CoreML8'}
    return function.block_specializations['CoreML8']


def make_template():
    assert sha(HELPER) == HELPER_SHA256, 'Signed-even helper differs from reviewed source'
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import get_new_symbol
    from scripts.signed_even_round_v2 import mil_round_signed_even_v2

    @mb.program(input_specs=[mb.TensorSpec(shape=(get_new_symbol(),))],
                opset_version=ct.target.macOS15)
    def program(z):
        return mil_round_signed_even_v2(mb, z)

    model = ct.convert(program, source='milinternal', convert_to='mlprogram',
                       minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT32,
                       pass_pipeline=ct.PassPipeline.EMPTY, skip_model_load=True)
    spec = model.get_spec()
    block = block_of(spec)
    assert len(block.outputs) == 1
    ops = list(block.operations)
    assert all(op.type in PURE and not op.blocks and len(op.outputs) == 1 for op in ops)
    assert dict(Counter(op.type for op in ops if op.type != 'const')) == EXPECTED_NONCONST
    for op in ops:
        tensor = op.outputs[0].type.tensorType
        if op.type == 'const':
            assert tensor.rank == 0 and not tensor.dimensions
        else:
            assert tensor.rank == 1 and tensor.dimensions[0].HasField('unknown')
    return ops, spec.mlProgram.functions['main'].inputs[0].name, block.outputs[0]


def replace_rounds(spec, template, selected_sites):
    """Pure protobuf transformation, with byte preservation outside replacements."""
    original = list(block_of(spec).operations)
    assert len(original) == 42498 and not any(op.blocks for op in original)
    rounds = {op.outputs[0].name: index for index, op in enumerate(original) if op.type == 'round'}
    assert len(rounds) == 131
    selected = set(rounds) if not selected_sites else set(selected_sites)
    if not selected <= rounds.keys():
        raise ValueError(f'Unknown round outputs: {sorted(selected - rounds.keys())}')
    assert selected
    variant = copy.deepcopy(spec)
    block = block_of(variant)
    template_ops, template_input, template_output = template
    all_names = {v.name for op in original for v in op.outputs}
    original_values = {v.name: v for op in original for v in op.outputs}
    original_values.update({v.name: v for v in spec.mlProgram.functions['main'].inputs})
    assert len(all_names) == sum(len(op.outputs) for op in original)
    external = {v.name for v in spec.mlProgram.functions['main'].inputs}
    rewritten = []
    maps = []
    untouched = []
    for index, old in enumerate(original):
        if old.type != 'round' or old.outputs[0].name not in selected:
            rewritten.append(copy.deepcopy(old))
            untouched.append((index, len(rewritten) - 1))
            continue
        assert len(old.outputs) == 1 and set(old.inputs) == {'x'}
        binding = old.inputs['x'].arguments
        assert len(binding) == 1 and binding[0].name
        old_out = old.outputs[0]
        assert old_out.type.tensorType.dataType == 11  # MIL FLOAT32
        assert wire(original_values[binding[0].name].type) == wire(old_out.type)
        prefix = f'signed_even_site_{index}_'
        names = {template_input: binding[0].name}
        for op in template_ops:
            output = op.outputs[0].name
            names[output] = old_out.name if output == template_output else prefix + output
            assert names[output] == old_out.name or names[output] not in all_names
            all_names.add(names[output])
        replacements = []
        for op in template_ops:
            new = copy.deepcopy(op)
            for inp in new.inputs.values():
                for argument in inp.arguments:
                    if argument.name:
                        argument.name = names[argument.name]
            output = new.outputs[0]
            output.name = names[output.name]
            if op.type != 'const':
                tensor = output.type.tensorType
                tensor.rank = old_out.type.tensorType.rank
                del tensor.dimensions[:]
                tensor.dimensions.extend(old_out.type.tensorType.dimensions)
            if output.name == old_out.name:
                output.CopyFrom(old_out)
            assert 'name' in new.attributes
            new.attributes['name'].immediateValue.tensor.strings.values[0] = output.name
            assert new.type in PURE and not new.blocks
            replacements.append(new)
        assert replacements[-1].outputs[0].name == old_out.name
        assert wire(replacements[-1].outputs[0]) == wire(old_out)
        maps.append({'original_index': index, 'input': binding[0].name, 'output': old_out.name,
                     'original_operation_sha256': digest(wire(old)),
                     'original_output_type_sha256': digest(wire(old_out.type)),
                     'replacement_start': len(rewritten), 'replacement_count': len(replacements),
                     'replacement_sha256': digest(b''.join(wire(op) for op in replacements)),
                     'names': names, 'operations': dict(sorted(Counter(op.type for op in replacements).items()))})
        rewritten.extend(replacements)
    del block.operations[:]
    block.operations.extend(rewritten)
    for old_index, new_index in untouched:
        assert wire(original[old_index]) == wire(block.operations[new_index])
    available = set(external)
    for op in block.operations:
        assert set(arguments(op)) <= available, f'Unavailable input for {op.type}'
        for value in op.outputs:
            assert value.name not in available
            available.add(value.name)
    assert set(block.outputs) <= available
    before_shell, after_shell = copy.deepcopy(spec), copy.deepcopy(variant)
    del block_of(before_shell).operations[:]
    del block_of(after_shell).operations[:]
    assert wire(before_shell) == wire(after_shell)
    before_counts = dict(sorted(Counter(op.type for op in original).items()))
    after_counts = dict(sorted(Counter(op.type for op in block.operations).items()))
    assert after_counts.get('round', 0) == 131 - len(selected)
    check = {'passed': True, 'before': before_counts, 'after': after_counts,
             'original_round_count': 131, 'replaced_round_count': len(selected),
             'remaining_round_count': after_counts.get('round', 0), 'site_maps': maps,
             'unchanged_operation_count': len(untouched),
             'all_other_operations_byte_identical': True,
             'all_non_operation_metadata_identical': True,
             'original_output_names_and_types_preserved': True,
             'weights_state_interfaces_and_state_operations_unchanged': True,
             'only_allowlisted_pure_operations_inserted': True,
             'topological_references_valid': True,
             'numerical_qualification': 'not performed by structural builder',
             'domain': 'Finite FP32 quotient inputs; full Q exponent/step computation unchanged; numerical/domain gates separate'}
    return variant, check


def build(attempt, sites=(), compile_package=False):
    assert sha(HELPER) == HELPER_SHA256, 'Signed-even helper differs from reviewed source'
    if not attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in attempt):
        raise ValueError('Fresh lowercase path-safe attempt required')
    evidence = ROOT / 'evidence/06d-gpu-rounding' / attempt
    artifact = ROOT / 'artifacts/06d-gpu-rounding' / attempt
    if evidence.exists() or artifact.exists():
        raise FileExistsError('Fresh artifact/evidence paths required')
    evidence.mkdir(parents=True)
    artifact.mkdir(parents=True)
    (evidence / 'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    (evidence / 'helper-snapshot.py').write_bytes(HELPER.read_bytes())
    report = {'status': 'starting', 'group': '0', 'script_sha256': sha(__file__),
              'helper_sha256': sha(HELPER), 'selected_sites': list(sites) or 'all',
              'arithmetic_policy': 'explicit-signed-ties-even-v2-round-only',
              'helper_source': str(HELPER.relative_to(ROOT)),
              'builder_source': str(Path(__file__).resolve().relative_to(ROOT)),
              'source_candidate_identity': SOURCE_ID}
    def save():
        temp = evidence / 'results.tmp'
        temp.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        temp.replace(evidence / 'results.json')
    save()
    try:
        import coremltools as ct
        from coreml_port.package_identity import verify_candidate_identity
        identity = verify_candidate_identity(SOURCE, ROOT / 'evidence/04d-cpu/full-candidate-6/package-identity.json')
        assert identity['aggregate_sha256'] == SOURCE_ID
        src = SOURCE / '0.mlpackage'
        source_files = inventory(src)
        spec = ct.models.utils.load_spec(str(src))
        template = make_template()
        assert sha(HELPER) == report['helper_sha256']
        report['template_sha256'] = digest(b''.join(wire(op) for op in template[0]))
        variant, report['graph_check'] = replace_rounds(spec, template, sites)
        destination = artifact / '0.mlpackage'
        shutil.copytree(src, destination)
        (destination / MODEL_FILE).write_bytes(wire(variant))
        assert wire(ct.models.utils.load_spec(str(destination))) == wire(variant)
        variant_files = inventory(destination)
        assert source_files.keys() == variant_files.keys()
        changed = [name for name in source_files if source_files[name] != variant_files[name]]
        assert changed == [MODEL_FILE]
        report.update(package=str(destination.relative_to(ROOT)), source_files=source_files,
                      variant_files=variant_files, changed_package_files=changed)
        save()
        if compile_package:
            from coreml_port.prediction_guard import install, stats
            report['guard_before_compile'] = install()
            compiled = artifact / '0.mlmodelc'
            started = time.perf_counter()
            returned = ct.models.utils.compile_model(str(destination), destination_path=str(compiled))
            assert Path(returned).resolve() == compiled.resolve()
            report.update(compiled=str(compiled.relative_to(ROOT)), compiled_files=inventory(compiled),
                          compile_seconds=time.perf_counter() - started, guard_after_compile=stats())
            assert report['guard_after_compile'] == report['guard_before_compile']
        assert inventory(src) == source_files
        assert sha(__file__) == report['script_sha256'] and sha(HELPER) == report['helper_sha256']
        report.update(status='completed', passed=True)
    except BaseException as exc:
        report.update(status='failed', passed=False, error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--group', choices=['0'], default='0')
    parser.add_argument('--site', action='append', default=[], help='Original round output name; repeat to select multiple sites; default all131')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    result = build(args.attempt, args.site, args.compile)
    print(json.dumps({key: result[key] for key in ('status', 'passed', 'group', 'package', 'helper_sha256')}))


if __name__ == '__main__':
    main()
