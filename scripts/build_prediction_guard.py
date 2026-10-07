"""Build the isolated experimental Core ML prediction GIL guard."""
import hashlib,json,subprocess,sysconfig
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'coreml_port/native/prediction_guard.mm'
output=ROOT/'artifacts/04e-runtime/prediction_guard.dylib'
evidence=ROOT/'evidence/04e-runtime/prediction-guard-build.json'
output.parent.mkdir(parents=True,exist_ok=True);evidence.parent.mkdir(parents=True,exist_ok=True)
command=['xcrun','clang++','-dynamiclib','-fPIC','-std=c++17','-O2','-fobjc-exceptions',
         '-I'+sysconfig.get_paths()['include'],str(source),'-framework','CoreML',
         '-framework','Foundation','-lobjc','-undefined','dynamic_lookup','-o',str(output)]
subprocess.run(command,check=True)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'command':command,'source_sha256':digest(source),'binary_sha256':digest(output),
        'status':'built; experimental; not integrated'}
evidence.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
