"""Recompute finite rounding laws from the controlled, sealed raw key captures."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=ROOT/a.output
 if out.exists():raise FileExistsError(out)
 src=ROOT/'evidence/06c-gpu-localization/raw-key-replay-1/results.json';r=json.loads(src.read_text())
 assert r['status']=='completed' and r['instrumentation_valid'] and r['previous_tap_compatible']
 report={'script_sha256':sha(__file__),'raw_result_sha256':sha(src),'scope':'Observed finite raw-stage arrays only; no actual-device or full-model-fix claim','archive_hashes':{},'stages':[],'equal_quotient_disagreements':[],'step_differences':[]}
 def load(c,policy):
  info=r['files'][f'chunk{c}/{policy}-tapped/outputs'];path=ROOT/info['path'];assert sha(path)==info['sha256'];report['archive_hashes'][str(path.relative_to(ROOT))]=info['sha256']
  with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
 def away(x):return np.sign(x)*np.floor(np.abs(x)+.5)
 for c in (0,1):
  arrays={pol:load(c,pol) for pol in ('CPU','GPU')}
  for pol,z in arrays.items():
   for raw,rounded in [('x_9','value_5'),('x_27','var_2083'),('x_29','var_2106'),('x_31','k_9')]:
    x=z[raw].astype(np.float64).reshape(z[rounded].shape);actual=z[rounded].astype(np.float64)
    assert np.isfinite(x).all() and np.isfinite(actual).all()
    _,exponent=np.frexp(np.maximum(abs(x),1e-30));step=np.ldexp(np.ones_like(x),exponent-8);q=x/step
    even=np.rint(q)*step;other=away(q)*step
    report['stages'].append({'chunk':c,'policy':pol,'raw':raw,'rounded':rounded,'elements':x.size,'ties_even_mismatches':int(np.count_nonzero(actual!=even)),'ties_away_mismatches':int(np.count_nonzero(actual!=other))})
  cpu,gpu=arrays['CPU'],arrays['GPU'];same=cpu['var_2121']==gpu['var_2121'];diff=cpu['var_2122']!=gpu['var_2122'];mask=same&diff;q=cpu['var_2121'].astype(np.float64)
  assert np.all(np.mod(np.abs(q[mask]),1)==.5)
  assert np.array_equal(cpu['var_2122'],np.rint(cpu['var_2121'].astype(np.float64)))
  assert np.array_equal(gpu['var_2122'],away(gpu['var_2121'].astype(np.float64)))
  report['equal_quotient_disagreements'].append({'chunk':c,'equal_quotients':int(same.sum()),'different_rounds':int(mask.sum()),'all_exact_half_ties':True})
  for ix in np.argwhere(cpu['step_23']!=gpu['step_23']):
   idx=tuple(ix);report['step_differences'].append({'chunk':c,'index':ix.tolist(),**{pol:{name:float(z[name][idx]) for name in ['x_31','step_23','var_2121','var_2122','k_9']} for pol,z in arrays.items()}})
 report['all_observed_stages_match_policy_rounding_law']=all(row['ties_even_mismatches' if row['policy']=='CPU' else 'ties_away_mismatches']==0 for row in report['stages'])
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'all_observed_stages_match_policy_rounding_law':report['all_observed_stages_match_policy_rounding_law'],'equal_quotient_disagreements':report['equal_quotient_disagreements'],'step_differences':report['step_differences']},indent=2))
if __name__=='__main__':main()
