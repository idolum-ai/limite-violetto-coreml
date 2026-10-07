"""Create or verify a deterministic identity manifest for a Core ML candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreml_port.package_identity import (  # noqa: E402
    PackageIdentityError,
    inventory_candidate,
    verify_candidate_identity,
    write_candidate_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, help="Directory containing exactly the 14 candidate .mlpackages.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-manifest", metavar="PATH", help="Inventory packages and write a JSON identity manifest.")
    action.add_argument("--check-manifest", metavar="PATH", help="Verify the candidate tree against an existing manifest.")
    args = parser.parse_args()

    try:
        if args.write_manifest:
            result = write_candidate_manifest(args.model_root, args.write_manifest)
            print(json.dumps({"status": "written", "manifest": str(Path(args.write_manifest)), "aggregate_sha256": result["aggregate_sha256"], "package_count": len(result["packages"])}, indent=2))
        else:
            result = verify_candidate_identity(args.model_root, args.check_manifest)
            print(json.dumps({"status": "verified", "aggregate_sha256": result["aggregate_sha256"], "package_count": len(result["packages"])}, indent=2))
    except PackageIdentityError as exc:
        print(f"candidate identity rejected: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"candidate identity operation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
