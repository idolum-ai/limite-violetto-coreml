"""Initial backend qualification and reference-only shape calibration."""
import json,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from portable.model import load_model

def metrics(a,b):
    a=a.cpu().double();b=b.cpu().double();d=a-b
    return {'max_abs':d.abs().max().item(),'rmse':d.square().mean().sqrt().item(),
            'relative_rmse':(d.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-12)).item(),
            'cosine':torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0).item()}

if __name__=='__main__':
    torch.set_num_threads(4)
    device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
    out=ROOT/'evidence/03-portable'
    print('Loading',device,flush=True);t=time.perf_counter();m=load_model(device)
    (out/'weight-mapping.json').write_text(json.dumps(m.load_report,indent=2)+'\n')
    print('Loaded in',time.perf_counter()-t,flush=True)
    cases=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases']
    short=cases[0]['input_ids']
    long=(short*24)[:1026]
    results={'device':device,'torch':torch.__version__,'mps_available':torch.backends.mps.is_available(),'calibration':[]}
    for name,ids,n in [('short',short+[1],len(short)),('window',long,1025)]:
        x=torch.tensor([ids],device=device)
        t=time.perf_counter();a=m.reference_hidden(x[:,:n]);s=m.scores(a[:,-1:]);
        b=m.reference_hidden(x);bs=m.scores(b[:,n-1:n]);
        entry={'case':name,'prefix_length':n,'hidden':metrics(a[:,-1:],b[:,n-1:n]),'scores':metrics(s,bs),'seconds':time.perf_counter()-t}
        if name=='short':
            repeat=m.reference_hidden(x[:,:n]);entry['repeat_hidden']=metrics(a,repeat)
        results['calibration'].append(entry)
        print(json.dumps(entry),flush=True)
    ac=json.loads((out/'ACCEPTANCE.json').read_text())
    sm=max(c['scores']['max_abs'] for c in results['calibration'])
    hm=max(c['hidden']['relative_rmse'] for c in results['calibration'])
    results['frozen_limits']={'score_max_abs':min(.30,max(.04,4*sm)), 'hidden_relative_rmse':min(.10,max(.01,4*hm)), 'hidden_min_cosine':.995,'calibration_valid':4*sm<=.30 and 4*hm<=.10}
    results['peak_process_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
    (out/f'calibration-{device}.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results['frozen_limits']),flush=True)
