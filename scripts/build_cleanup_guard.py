"""Build the narrowly scoped Core ML Tools 9.0 destructor compatibility guard."""
import hashlib,json,subprocess,sysconfig
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'coreml_port/native/cleanup_guard.c';output=ROOT/'artifacts/04b-reliability/cleanup_guard.dylib';output.parent.mkdir(exist_ok=True,parents=True)
cmd=['xcrun','clang','-shared','-fPIC','-std=c11','-O2','-I'+sysconfig.get_paths()['include'],str(source),'-lobjc','-undefined','dynamic_lookup','-o',str(output)]
subprocess.run(cmd,check=True)
result={'command':cmd,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(output.read_bytes()).hexdigest()};(ROOT/'evidence/04b-reliability/guard-build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
