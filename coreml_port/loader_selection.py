"""Explicit selection of source packages or a verified persistent compiled cache."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .tiny_prefix_runtime import select_pipeline

ROOT = Path(__file__).resolve().parents[1]
LOADER_SOURCES = ('coreml_port/compiled_runtime.py', 'coreml_port/compiled_identity.py',
                  'coreml_port/loader_selection.py', 'coreml_port/package_identity.py')

def add_loader_arguments(parser):
    parser.add_argument('--compiled-dir', help='Opt into a persistent compiled cache; requires its manifest and tiny-prefix replay.')
    parser.add_argument('--compiled-manifest', help='Identity manifest binding all14 compiled models to their sealed source packages and environment.')

def _validate(args):
    compiled = getattr(args, 'compiled_dir', None)
    manifest = getattr(args, 'compiled_manifest', None)
    if bool(compiled) != bool(manifest):
        raise ValueError('--compiled-dir and --compiled-manifest must be supplied together.')
    if compiled:
        if not getattr(args, 'tiny_prefix_replay', False):
            raise ValueError('Compiled loading requires explicit --tiny-prefix-replay.')
        if getattr(args, 'compute', 'CPU_ONLY') != 'CPU_ONLY':
            raise ValueError('Persistent compiled loading is qualified only for CPU_ONLY.')
        if not getattr(args, 'model_dir', None) or not getattr(args, 'identity_manifest', None):
            raise ValueError('Compiled loading requires explicit source --model-dir and --identity-manifest.')
    return bool(compiled)

def runtime_adapter_name(args):
    if _validate(args): return 'compiled_tiny_prefix_replay'
    return 'tiny_prefix_replay' if args.tiny_prefix_replay else 'standard_pipeline'

def loader_provenance(args):
    """Describe requested identity; completed gates also verify bytes before load.

    This metadata function itself does not load models or verify all cache bytes.
    The compiled pipeline and qualification aggregator perform that verification.
    """
    if not _validate(args): return {'mode': 'mlpackage'}
    path = Path(args.compiled_manifest).expanduser().resolve(strict=True)
    manifest = json.loads(path.read_text())
    return {'mode': 'persistent_compiled',
            'source_candidate_identity': manifest['source_candidate_identity'],
            'compiled_aggregate_sha256': manifest['aggregate_sha256'],
            'compiled_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'compiled_dir': str(Path(args.compiled_dir).expanduser().resolve(strict=True)),
            'compiled_manifest': str(path),
            'sources': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in LOADER_SOURCES}}

def create_pipeline(args, compute=None, model_dir=None, identity_manifest=None):
    selected = compute or getattr(args, 'compute', 'CPU_ONLY')
    source = model_dir if model_dir is not None else args.model_dir
    source_identity = identity_manifest if identity_manifest is not None else args.identity_manifest
    if _validate(args):
        from .compiled_runtime import CompiledTinyPrefixPipeline
        return CompiledTinyPrefixPipeline(selected, source, source_identity,
                                          compiled_dir=args.compiled_dir,
                                          compiled_manifest=args.compiled_manifest)
    return select_pipeline(args.tiny_prefix_replay)(selected, source, source_identity)
