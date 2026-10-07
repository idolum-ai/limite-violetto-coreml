"""Pin and load a project-local GIL guard for Core ML Tools array cleanup."""
import ctypes,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BRIDGE_SHA256='915f478a3f2a392077012f8dacd9ae433579cbb299019ef2df23217ce536ee67'
_library=None

def install():
    global _library
    if _library is None:
        import coremltools as ct
        bridge=Path(ct.__file__).parent/'libcoremlpython.so'
        if sys.version_info[:2]!=(3,12) or ct.__version__!='9.0' or hashlib.sha256(bridge.read_bytes()).hexdigest()!=BRIDGE_SHA256:
            raise RuntimeError('Cleanup guard requires the recorded CPython 3.12/Core ML Tools 9.0 bridge; requalify a different environment.')
        path=ROOT/'artifacts/04b-reliability/cleanup_guard.dylib'
        if not path.exists():raise RuntimeError('Build the cleanup guard with .venv/bin/python scripts/build_cleanup_guard.py')
        lib=ctypes.PyDLL(str(path));lib.violetto_install_cleanup_guard.restype=ctypes.c_int
        code=lib.violetto_install_cleanup_guard()
        if code:raise RuntimeError(f'Core ML cleanup guard installation failed: {code}')
        for name in ['calls','background_calls','shutdown_skips']:getattr(lib,'violetto_guard_'+name).restype=ctypes.c_ulong
        _library=lib
    return stats()

def stats():
    return {name:int(getattr(_library,'violetto_guard_'+name)()) for name in ['calls','background_calls','shutdown_skips']} if _library else {'installed':False}
