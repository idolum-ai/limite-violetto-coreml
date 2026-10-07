"""Pinned, process-local GIL release around synchronous Core ML prediction.

Python input/output conversion stays under the GIL. Only the two native
prediction selectors release it, allowing guarded background cleanup to finish.
Qualification is limited to this recorded macOS and bridge combination.
"""
import ctypes
import hashlib
import json
import platform
from pathlib import Path
from . import cleanup_guard

ROOT = Path(__file__).resolve().parents[1]
_library = None


def install():
    global _library
    if _library is None:
        cleanup_guard.install()
        if platform.mac_ver()[0] != '26.5.2' or platform.machine() != 'arm64':
            raise RuntimeError('Prediction guard requires the qualified macOS 26.5.2 arm64 environment.')
        binary = ROOT / 'artifacts/04e-runtime/prediction_guard.dylib'
        record = ROOT / 'evidence/04e-runtime/prediction-guard-build.json'
        source = ROOT / 'coreml_port/native/prediction_guard.mm'
        if not binary.exists() or not record.exists():
            raise RuntimeError('Build the prediction guard with .venv/bin/python scripts/build_prediction_guard.py')
        build = json.loads(record.read_text())
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        if digest(binary) != build['binary_sha256'] or digest(source) != build['source_sha256']:
            raise RuntimeError('Prediction guard source/binary differs from its build record; rebuild and requalify.')
        lib = ctypes.PyDLL(str(binary))
        lib.violetto_install_prediction_guard.argtypes = []
        lib.violetto_install_prediction_guard.restype = ctypes.c_int
        code = lib.violetto_install_prediction_guard()
        if code:
            raise RuntimeError(f'Prediction guard installation failed: {code}')
        for name in ('plain_calls', 'state_calls', 'gil_releases'):
            fn = getattr(lib, 'violetto_prediction_' + name)
            fn.argtypes = []
            fn.restype = ctypes.c_ulong
        _library = lib
    return stats()


def stats():
    if _library is None:
        return {'installed': False}
    return {'installed': True, **{
        n: int(getattr(_library, 'violetto_prediction_' + n)())
        for n in ('plain_calls', 'state_calls', 'gil_releases')
    }}
