"""Export and summarize only a completed owned worker's Core ML trace table.

Never records or attaches. Requires the exact completed two-call replay worker
and resolves target scope from the trace TOC before requesting event rows.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
XCTRACE=Path('/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace')
WORKER='scripts/trace_gpu_replay.py'
WORKER_SHA256='f1db086cdbeb3b8bd3a4a0220551eb7479e42ecac18da185129cf1e8f145ef8c'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def resolve(value):
    p=Path(value).expanduser();return (p if p.is_absolute() else ROOT/p).resolve()


def read(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict):raise ValueError('Expected JSON object')
    return value


def local(tag):return tag.rsplit('}',1)[-1]


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def validate_worker(path):
    worker=read(path)
    if worker.get('status')!='completed' or worker.get('passed') is not True:
        raise ValueError('Owned replay worker is not complete and passing')
    if type(worker.get('pid')) is not int or worker['pid']<=0 or worker.get('policy') not in {'CPU','GPU'}:
        raise ValueError('Worker PID or policy is missing')
    if worker.get('script_sha256')!=WORKER_SHA256 or sha(path.parent/'source-snapshot.py')!=WORKER_SHA256:
        raise ValueError('Worker source differs from the reviewed isolated replay')
    if worker.get('prediction_guard')!={'installed':True,'plain_calls':0,'state_calls':2,'gil_releases':2}:
        raise ValueError('Worker did not report exactly two guarded state predictions')
    calls=worker.get('calls',[])
    if len(calls)!=2 or [x.get('chunk') for x in calls]!=[0,1]:raise ValueError('Expected exactly chunks 0 and 1')
    for call in calls:
        if call.get('passed') is not True or call.get('state_reference_released') is not True:
            raise ValueError('Replay or state-reference release failed')
        for key in ('restoration','inputs_unchanged','outputs','state_after'):
            if call.get(key,{}).get('exact') is not True:raise ValueError(f'Worker {key} is not exact')
        if call.get('prediction_begin',{}).get('pid')!=worker['pid'] or call.get('prediction_end',{}).get('pid')!=worker['pid']:
            raise ValueError('Worker prediction event PID differs')
    if worker.get('source_and_compiled_preserved') is not True or worker.get('fixture_preserved') is not True:
        raise ValueError('Worker did not preserve its source/cache/fixture cohort')
    return worker


def run_export(trace,output,selector):
    command=[str(XCTRACE),'export','--input',str(trace),*selector,'--output',str(output)]
    process=subprocess.run(command,capture_output=True,text=True)
    record={'command':command,'exit_code':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
    if process.returncode or not output.is_file():
        raise RuntimeError(f'xctrace export failed: {record}')
    return record


def select_owned_table(toc,pid):
    """Require one normally exited launched process before using SINGLE scope."""
    root=ET.parse(toc).getroot()
    runs=[node for node in root if local(node.tag)=='run']
    if len(runs)!=1:raise ValueError('Expected exactly one trace run')
    run=runs[0]
    targets=[node for node in run.iter() if local(node.tag)=='target']
    processes=[node for target in targets for node in target.iter() if local(node.tag)=='process']
    if len(processes)!=1:raise ValueError('SINGLE must denote exactly one launched process')
    owned=[]
    for target in targets:
        for node in target.iter():
            if local(node.tag)=='process' and node.get('pid')==str(pid):owned.append(node)
    if len(owned)!=1:raise ValueError('Trace target metadata does not uniquely identify the owned worker PID')
    target=owned[0]
    if target.get('type')!='launched' or target.get('return-exit-status')!='0':
        raise ValueError('Trace is not an explicitly launched target')
    if 'scripts/trace_gpu_replay.py' not in target.get('arguments',''):
        raise ValueError('Trace target arguments do not identify the owned replay worker')
    tables=[node for node in run.iter() if local(node.tag)=='table' and node.get('schema')=='coreml-os-signpost']
    scoped=[node for node in tables if node.get('target-pid') in {str(pid),'SINGLE'}]
    if len(scoped)!=1:raise ValueError('No unique own-target Core ML table; ALL/unattributed tables are forbidden')
    table=scoped[0];number=run.get('number')
    if not number or not number.isdigit():raise ValueError('Trace run number is malformed')
    scope=table.get('target-pid')
    xpath=f'/trace-toc/run[@number="{number}"]/data/table[@schema="coreml-os-signpost" and @target-pid="{scope}"]'
    return xpath,{'run_number':number,'pid':pid,'target':dict(target.attrib),'table_attributes':dict(table.attrib)}


def decode_cells(xml_path):
    """Resolve standard xctrace id/ref cells while preserving raw/formatted text.

    The real CPU/GPU exports verified this schema but contained no rows.
    Unknown row representations fail instead of guessing devices.
    """
    root=ET.parse(xml_path).getroot()
    ids={node.get('id'):node for node in root.iter() if node.get('id')}
    schemas=[node for node in root.iter() if local(node.tag)=='schema' and node.get('name')=='coreml-os-signpost']
    if len(schemas)!=1:raise ValueError('Expected one exported Core ML schema')
    columns=[]
    for col in schemas[0]:
        if local(col.tag) not in {'col','column'}:continue
        mnemonic=col.get('mnemonic')
        if mnemonic is None:
            for child in col:
                if local(child.tag)=='mnemonic':mnemonic=child.text
        if not mnemonic:raise ValueError('Exported column mnemonic missing')
        columns.append(mnemonic)
    required={'start','duration','event-label','model-name','parent-model-name','event-type'}
    if not required<=set(columns):raise ValueError(f'Exported Core ML columns missing: {sorted(required-set(columns))}')
    def cell_value(cell):
        visited=set()
        while cell.get('ref'):
            ref=cell.get('ref')
            if ref in visited or ref not in ids:raise ValueError('Unresolved/cyclic XML cell reference')
            visited.add(ref);cell=ids[ref]
        return {'type':local(cell.tag),'raw':''.join(cell.itertext()).strip(),'formatted':cell.get('fmt')}
    rows=[]
    for row in root.iter():
        if local(row.tag)!='row':continue
        cells=list(row)
        if len(cells)!=len(columns):raise ValueError('Exported row does not match schema width')
        rows.append({name:cell_value(cell) for name,cell in zip(columns,cells)})
    return rows,columns


def label(cell):return cell['formatted'] if cell.get('formatted') is not None else cell['raw']


def summarize_rows(rows):
    labels={};dispatch=[];predictions=[];unrecognized=[]
    for row in rows:
        name=label(row['event-label']);kind=label(row['event-type'])
        labels[(kind,name)]=labels.get((kind,name),0)+1
        if name=='Prediction':predictions.append(row)
        if kind=='model-dispatch-event':
            dispatch.append(row)
            if name not in {'CPU','GPU Request'}:unrecognized.append(name)
    cpu=[r for r in dispatch if label(r['event-label'])=='CPU']
    gpu=[r for r in dispatch if label(r['event-label'])=='GPU Request']
    return {'event_inventory':[{'event_type':k,'event_label':n,'count':c} for (k,n),c in sorted(labels.items())],
            'event_row_count':len(rows),'expected_worker_prediction_count':2,
            'two_prediction_intervals_present':len(predictions)==2,
            'placement_verification':'positive_dispatch_observed' if dispatch else 'unverified_no_dispatch_events',
            'prediction_interval_count':len(predictions),'cpu_dispatch_interval_count':len(cpu),
            'gpu_request_interval_count':len(gpu),'actual_cpu_dispatch_observed':bool(cpu),
            'actual_gpu_request_execution_observed':bool(gpu),'unrecognized_dispatch_labels':sorted(set(unrecognized)),
            'dispatch_intervals':dispatch,
            'placement_scope':'Own-PID model/run only; no per-call or per-operation placement attribution',
            'limits':['No placement claim follows from compute policy alone.','Missing dispatch rows do not prove device absence.',
                      'GPU Request intervals are workload spans, not exclusive GPU busy time.','The installed schema does not expose a public operation-ID column.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',required=True);parser.add_argument('--worker-result',required=True);parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    if not args.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.attempt):parser.error('Fresh path-safe attempt required')
    trace=resolve(args.trace);worker_path=resolve(args.worker_result)
    if not trace.is_relative_to(ROOT/'evidence/06c-gpu-localization') or not worker_path.is_relative_to(ROOT/'evidence/06c-gpu-localization'):
        raise ValueError('Only explicitly selected localization-owned trace/worker artifacts are allowed')
    worker=validate_worker(worker_path)
    out=ROOT/'evidence/06c-gpu-localization'/args.attempt;out.mkdir(parents=True,exist_ok=False)
    (out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    record={'status':'starting','script_sha256':sha(__file__),'worker_result':str(worker_path),'worker_result_sha256':sha(worker_path),
            'trace':str(trace),'policy':worker['policy'],'own_pid':worker['pid'],'exports':[]}
    save(out/'results.json',record)
    try:
        toc=out/'toc.xml';record['exports'].append(run_export(trace,toc,['--toc']))
        xpath,scope=select_owned_table(toc,worker['pid']);record['verified_scope']=scope;record['toc_sha256']=sha(toc)
        data=out/'coreml-own-pid.xml';record['exports'].append(run_export(trace,data,['--xpath',xpath]))
        record['events_sha256']=sha(data);rows,columns=decode_cells(data);record['columns']=columns
        record['observations']=summarize_rows(rows)
        if sha(worker_path)!=record['worker_result_sha256']:raise ValueError('Worker result changed during export')
        record.update(status='completed',export_scope_verified=True,
                      worker_predictions_verified=2,
                      actual_placement_verified=bool(record['observations']['dispatch_intervals']),
                      interpretation='Positive runtime dispatch observations only; do not equate no rows with no device use.')
    except BaseException as exc:
        record.update(status='failed',export_scope_verified=False,error={'type':type(exc).__name__,'message':str(exc)});raise
    finally:save(out/'results.json',record)
    print(json.dumps({'status':record['status'],'own_pid':worker['pid'],'observations':record.get('observations',{})},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
