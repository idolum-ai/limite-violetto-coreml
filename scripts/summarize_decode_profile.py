"""Validate paired decode profiling evidence and summarize component timings."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
ROOT = Path(__file__).resolve().parents[1]
NAMES = ['embedding', *map(str, range(12)), 'head']

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def stats(values):
    return {'n':len(values), 'median':statistics.median(values),
            'min':min(values), 'max':max(values), 'sum':sum(values)}

def summarize(r):
    if r.get('status') != 'completed' or r.get('passed') is not True:
        raise ValueError('Completed passing profile required')
    trials=r['trials']
    expected={(-1,32), *((rep,n) for rep in range(3) for n in [32,128,512])}
    if len(trials)!=10 or {(t['repeat'],t['prefix_length']) for t in trials}!=expected:
        raise ValueError('Trial grid missing or duplicated')
    count=0; pairs={}; profiled=[]; controls=[]; by_prefix={}
    for t in trials:
        warm=t['repeat']==-1
        if t['warmup']!=warm or t.get('disposed') is not True:
            raise ValueError('Warmup/disposal check failed')
        if [s['step'] for s in t['steps']]!=list(range(3 if warm else 9)):
            raise ValueError('Missing/duplicate step')
        for step in t['steps']:
            count+=1
            if step.get('scores_exact') is not True or step['seconds']<=0:
                raise ValueError('Score parity or timing failed')
            expected_profile=not warm and t['repeat']>0 and step['step']>0 and (step['step']+t['repeat'])%2==0
            if step['profiled']!=expected_profile:
                raise ValueError('Instrumentation schedule differs from frozen protocol')
            if not expected_profile and step['calls']:
                raise ValueError('Control unexpectedly instrumented')
            if expected_profile:
                if [c['name'] for c in step['calls']]!=NAMES or any(c['seconds']<=0 for c in step['calls']):
                    raise ValueError('Profiled component order/count/timing differs')
                component=sum(c['seconds'] for c in step['calls'])
                if abs(component-step['component_seconds'])>1e-10 or abs(step['seconds']-component-step['outside_predict_seconds'])>1e-10 or step['outside_predict_seconds']<0:
                    raise ValueError('Timing accounting differs')
                profiled.append(step)
            if not warm and t['repeat']>0 and step['step']>0:
                key=(t['prefix_length'],step['step'])
                pair=pairs.setdefault(key,{})
                label='profiled' if expected_profile else 'control'
                if label in pair:raise ValueError('Duplicate pair condition')
                pair[label]=step
                if not expected_profile:controls.append(step)
    if count!=84 or len(pairs)!=24 or len(profiled)!=24 or len(controls)!=24:
        raise ValueError('Forward/pair counts differ from protocol')
    deltas=[]; ratios=[]; paired_rows=[]
    for (prefix,step),pair in sorted(pairs.items()):
        if set(pair)!= {'control','profiled'} or pair['control']['scores_sha256']!=pair['profiled']['scores_sha256']:
            raise ValueError('Matched pair absent or score bytes differ')
        c=pair['control']['seconds'];p=pair['profiled']['seconds']
        deltas.append(p-c);ratios.append(p/c)
        paired_rows.append({'prefix':prefix,'step':step,'control_seconds':c,'profiled_seconds':p,'difference_seconds':p-c,'ratio':p/c})
    for prefix in [32,128,512]:
        rows=[p for p in paired_rows if p['prefix']==prefix]
        by_prefix[str(prefix)]={'control_seconds':stats([p['control_seconds'] for p in rows]),
                               'profiled_seconds':stats([p['profiled_seconds'] for p in rows]),
                               'paired_ratio':stats([p['ratio'] for p in rows]),
                               'paired_difference_seconds':stats([p['difference_seconds'] for p in rows]),
                               'control_tokens_per_second':len(rows)/sum(p['control_seconds'] for p in rows)}
        observations=[pair['profiled'] for (length,_),pair in pairs.items() if length==prefix]
        subtotal=sum(item['seconds'] for item in observations)
        amounts={name:sum(c['seconds'] for item in observations for c in item['calls'] if c['name']==name) for name in NAMES}
        by_prefix[str(prefix)]['component_shares']={
            'embedding':amounts['embedding']/subtotal,
            'backbone_groups':sum(amounts[str(i)] for i in range(12))/subtotal,
            'head':amounts['head']/subtotal,
            'outside_predict':sum(item['outside_predict_seconds'] for item in observations)/subtotal}
    total=sum(s['seconds'] for s in profiled)
    components={}
    for name in NAMES:
        values=[next(c['seconds'] for c in s['calls'] if c['name']==name) for s in profiled]
        components[name]={**stats(values),'share_of_profiled_forward_time':sum(values)/total}
    grouped={'embedding':components['embedding']['sum'],'backbone_groups':sum(components[str(i)]['sum'] for i in range(12)),
             'head':components['head']['sum'],'outside_predict':sum(s['outside_predict_seconds'] for s in profiled)}
    if abs(sum(grouped.values())-total)>1e-8:raise ValueError('Grouped timing does not cover total')
    return {'status':'pass','passed':True,'exact_score_comparisons':count,'disposed_requests':len(trials),
            'matched_pairs':len(pairs),'profiled_component_calls':len(profiled)*14,
            'by_prefix':by_prefix,'components':components,
            'grouped_cost':{k:{'seconds':v,'share':v/total} for k,v in grouped.items()},
            'profiled_forward_seconds':stats([s['seconds'] for s in profiled]),
            'paired_difference_seconds':stats(deltas),'paired_profile_over_control_ratio':stats(ratios),
            'pairs':paired_rows,
            'limits':['Small single-process descriptive sample, with uncontrolled host/caches.',
                      'Condition is confounded with repeat/order for a given position; paired timing difference is not an isolated causal estimate of timer overhead.',
                      'Component timing includes Python preparation, bridge, native prediction and returned-array conversion; it does not isolate an internal operator.',
                      'Outside-predict time includes profiler wrapper and list-append overhead.',
                      'First-shape repeat0, initial warmup and all prefill calls are excluded from warm decode summaries.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.input);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    r=json.loads(source.read_text());summary=summarize(r)
    if sha(source.parent/'source-snapshot.py')!=r['script_sha256'] or sha(ROOT/'scripts/profile_compiled_decode.py')!=r['script_sha256']:
        raise ValueError('Profiler source/snapshot differs')
    qualified=ROOT/'evidence/05c-compiled/milestone-rerun1.json';m=json.loads(qualified.read_text())
    if sha(qualified)!=r['milestone_sha256'] or r['loader_provenance']!=m['compiled_identity']['loader_provenance'] or r['runtime_provenance']!=m['current_runtime_provenance']:
        raise ValueError('Qualification provenance differs')
    summary.update(profile_sha256=sha(source),aggregator_sha256=sha(__file__),protocol=r['protocol'],
                   loader_provenance=r['loader_provenance'],host_before=r['host_before'],host_after=r['host_after'])
    out.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'passed':True,'grouped_cost':summary['grouped_cost'],'by_prefix':summary['by_prefix']},indent=2))

if __name__=='__main__':main()
