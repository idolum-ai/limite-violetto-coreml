"""Download the public pinned checkpoint; verify size/hash before publishing it."""
from pathlib import Path
import datetime
import hashlib
import json
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'sources/model/model.safetensors'
SIZE=2070811520
SHA='9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5'
REV='9402e422e87b220507963cd42c997459c1e87b43'
URL=f'https://huggingface.co/paradigma-inc/limite-1b-violetto/resolve/{REV}/model.safetensors'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

start=time.perf_counter()
if TARGET.exists():
    if TARGET.stat().st_size!=SIZE or digest(TARGET)!=SHA:
        raise RuntimeError('Existing checkpoint fails identity check; refusing overwrite')
    print('Existing checkpoint identity verified',flush=True)
else:
    partial=TARGET.with_suffix('.safetensors.part')
    offset=partial.stat().st_size if partial.exists() else 0
    headers={'User-Agent':'violetto-portable-validation/1'}
    if offset:headers['Range']=f'bytes={offset}-'
    with urllib.request.urlopen(urllib.request.Request(URL,headers=headers),timeout=60) as response:
        if offset and response.status!=206:offset=0
        if response.status==206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
            raise RuntimeError('Invalid resume range')
        downloaded=offset;next_report=downloaded+128*1024*1024
        with partial.open('ab' if offset else 'wb') as f:
            while True:
                chunk=response.read(8*1024*1024)
                if not chunk:break
                f.write(chunk);downloaded+=len(chunk)
                if downloaded>SIZE:raise RuntimeError('Download exceeds pinned size')
                if downloaded>=next_report:
                    print(f'{downloaded}/{SIZE} bytes ({downloaded/SIZE:.1%})',flush=True)
                    next_report=downloaded+128*1024*1024
    if partial.stat().st_size!=SIZE:raise RuntimeError('Incomplete download; resumable .part retained')
    if digest(partial)!=SHA:raise RuntimeError('SHA256 mismatch; file not promoted')
    partial.rename(TARGET)
result={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':URL,
        'revision':REV,'path':str(TARGET.relative_to(ROOT)),'bytes':SIZE,'sha256':SHA,
        'size_verified':True,'sha256_verified':True,'elapsed_seconds':time.perf_counter()-start}
(ROOT/'evidence/03-portable/checkpoint-download.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
