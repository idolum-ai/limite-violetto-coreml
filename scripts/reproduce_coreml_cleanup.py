"""Small allocator-debug reproducer for Core ML Tools input-array cleanup."""
import argparse,faulthandler,json,sys,time,ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));faulthandler.enable()
p=argparse.ArgumentParser();p.add_argument('--guard',action='store_true');p.add_argument('--iterations',type=int,default=50);a=p.parse_args()
import numpy as np,coremltools as ct
if a.guard:
    from coreml_port.cleanup_guard import install
    print('guard',install(),flush=True)
m=ct.models.MLModel(str(ROOT/'artifacts/04-coreml/rounding-v2/rounding.mlpackage'),compute_units=ct.ComputeUnit.CPU_ONLY)
for i in range(a.iterations):
    x=np.full(50517,i/32,dtype=np.float32);y=m.predict({'x':x})['y'];del x,y
    if i%10==0:print('iteration',i,flush=True)
    time.sleep(.03)
print('waiting for delayed cleanup with model alive',flush=True);time.sleep(15)
if a.guard:
    from coreml_port.cleanup_guard import stats
    print('stats',stats(),flush=True)
del m;print('completed',flush=True)
