"""Deterministic actual-class regression: delayed release on a native worker."""
import argparse,ctypes,faulthandler,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));faulthandler.enable()
p=argparse.ArgumentParser();p.add_argument('--guard',action='store_true');p.add_argument('--count',type=int,default=10);a=p.parse_args()
import numpy as np,coremltools
if a.guard:
    from coreml_port.cleanup_guard import install,stats
    print('guard',install(),flush=True)
lib=ctypes.PyDLL(str(ROOT/'artifacts/04b-reliability/cleanup_reproducer.dylib'));lib.violetto_release_array_on_worker.argtypes=[ctypes.py_object];lib.violetto_release_array_on_worker.restype=None
for i in range(a.count):lib.violetto_release_array_on_worker(np.ones((1,32,1280),dtype=np.float32))
print('scheduled',a.count,flush=True);time.sleep(2)
if a.guard:
    deadline=time.monotonic()+30
    while stats()['calls']<a.count and time.monotonic()<deadline:time.sleep(.05)
    print('stats',stats(),flush=True)
    assert stats()['calls']==a.count and stats()['background_calls']==a.count
print('completed',flush=True)
