"""Isolated MIL ablation: preserve shapes, drop unused value-product matmuls."""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def wire(message):return message.SerializeToString(deterministic=True)
def inventory(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def refs(block):
    yield from block.outputs
    for op in block.operations:
        for value in op.inputs.values():
            for arg in value.arguments:
                if arg.name:yield arg.name
        for nested in op.blocks:yield from refs(nested)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--compile',action='store_true');a=p.parse_args()
    if not a.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt):p.error('Invalid attempt')
    out=ROOT/'evidence/05e-product'/a.attempt;out.mkdir(parents=True,exist_ok=False)
    artifact=ROOT/'artifacts/05e-product'/a.attempt;artifact.mkdir(parents=True,exist_ok=False)
    (out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    r={'status':'starting','script_sha256':sha(__file__),'scope':'Harness-only one query, absolute position 32..2047; package interface unchanged; no runtime adoption'}
    def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    save()
    try:
        import coremltools as ct
        from coreml_port.package_identity import verify_candidate_identity
        source=ROOT/'artifacts/04d-cpu/full-candidate-6'
        ident=verify_candidate_identity(source,ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json')
        assert ident['aggregate_sha256']=='5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6'
        r['source_candidate_identity']=ident['aggregate_sha256']
        src=source/'0.mlpackage';spec=ct.models.utils.load_spec(str(src));variant=copy.deepcopy(spec)
        assert set(spec.mlProgram.functions)=={'main'}
        assert set(spec.mlProgram.functions['main'].block_specializations)=={'CoreML8'}
        block=variant.mlProgram.functions['main'].block_specializations['CoreML8']
        assert not any(op.blocks for op in block.operations), 'Nested graph needs separate analysis'
        ops=list(block.operations);by_name={v.name:op for op in ops for v in op.outputs}
        candidates=[];removed=set();r['rewires']=[]
        def ref(op,key):
            args=op.inputs[key].arguments;assert len(args)==1 and args[0].name
            return args[0].name
        def ints(name):return list(by_name[name].attributes['val'].immediateValue.tensor.ints.values)
        for op in ops:
            if op.type!='concat' or len(op.inputs['values'].arguments)!=16:continue
            names=[v.name for v in op.inputs['values'].arguments];products=[by_name[n] for n in names]
            if not all(v.type=='matmul' for v in products):continue
            assert ints(ref(op,'axis'))==[2]
            assert len(set(ref(v,'y') for v in products))==1
            slices=[by_name[ref(v,'x')] for v in products]
            assert all(v.type=='slice_by_index' for v in slices)
            assert len(set(ref(v,'x') for v in slices))==1
            for i,v in enumerate(slices):
                assert ints(ref(v,'begin'))[2]==2*i and ints(ref(v,'end'))[2]==2*i+2
            for v in products:
                assert len(v.outputs)==1
                dims=v.outputs[0].type.tensorType.dimensions
                assert dims[0].constant.size==1 and dims[1].constant.size==10 and dims[3].constant.size==128
            candidates.append(op.outputs[0].name);removed.update(names[1:])
            r['rewires'].append({'concat':op.outputs[0].name,'original_inputs':names,'new_inputs':[names[0]]*16,'first_matmul_sha256':hashlib.sha256(wire(products[0])).hexdigest(),'first_slice_sha256':hashlib.sha256(wire(slices[0])).hexdigest()})
            for arg in op.inputs['values'].arguments:arg.name=names[0]
        assert len(candidates)==4 and len(removed)==60
        assert not (set(refs(block)) & removed), 'Deleted matmul output still consumed'
        survivors=[op for op in ops if not any(v.name in removed for v in op.outputs)]
        del block.operations[:];block.operations.extend(survivors)
        before=spec.mlProgram.functions['main'].block_specializations['CoreML8']
        unchanged=0
        original_survivors=[op for op in before.operations if not any(v.name in removed for v in op.outputs)]
        assert len(original_survivors)==len(block.operations)
        for original,op in zip(original_survivors,block.operations):
            if not any(v.name in candidates for v in op.outputs):
                assert wire(op)==wire(original);unchanged+=1
        base_shell=copy.deepcopy(spec);new_shell=copy.deepcopy(variant)
        del base_shell.mlProgram.functions['main'].block_specializations['CoreML8'].operations[:]
        del new_shell.mlProgram.functions['main'].block_specializations['CoreML8'].operations[:]
        assert wire(base_shell)==wire(new_shell), 'Non-operation metadata changed'
        counts=lambda b:dict(sorted(collections.Counter(o.type for o in b.operations).items()))
        r['graph_check']={'passed':True,'before':counts(before),'after':counts(block),'removed_matmuls':sorted(removed),'unchanged_surviving_ops':unchanged,'changed_concats':4,'all_non_operation_metadata_identical':True,'first_pair_geometry':[1,10,2,2048]}
        assert r['graph_check']['before']['matmul']==68 and r['graph_check']['after']['matmul']==8
        dst=artifact/'0.mlpackage';shutil.copytree(src,dst)
        specpath=dst/'Data/com.apple.CoreML/model.mlmodel';specpath.write_bytes(variant.SerializeToString())
        assert wire(ct.models.utils.load_spec(str(dst)))==wire(variant)
        si,di=inventory(src),inventory(dst);changed=[k for k in si if si[k]!=di[k]]
        assert si.keys()==di.keys() and changed==['Data/com.apple.CoreML/model.mlmodel']
        r.update(package=str(dst.relative_to(ROOT)),source_files=si,variant_files=di,changed_package_files=changed)
        save()
        if a.compile:
            from coreml_port.prediction_guard import install
            install()
            start=time.perf_counter();compiled=artifact/'0.mlmodelc'
            ct.models.utils.compile_model(str(dst),destination_path=str(compiled))
            r.update(compile_seconds=time.perf_counter()-start,compiled=str(compiled.relative_to(ROOT)),compiled_files=inventory(compiled))
        assert sha(__file__)==r['script_sha256'] and inventory(src)==si
        r.update(status='completed',passed=True)
    except BaseException as e:r.update(status='failed',passed=False,error={'type':type(e).__name__,'message':str(e)});raise
    finally:save()
    print(json.dumps({'passed':True,'removed_matmuls':60,'compiled':r.get('compiled'),'compile_seconds':r.get('compile_seconds')},indent=2))
if __name__=='__main__':main()
