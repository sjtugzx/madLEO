"""Build the flat (folder-free) upload layout for the Figshare deposit.

Figshare stores item files as a flat list, which mixes files of different
directories and collides the shared basenames of the evidence parquets
(``tle/sentinel-3a.parquet`` vs ``orbit/sentinel-3a.parquet`` vs
``slr/sentinel-3a.parquet``).  This script copies ``dataset/`` into an
upload directory where every file name encodes its logical path with
double underscores:

    mission_reported/annotations/event_windows.csv
        -> mission_reported__annotations__event_windows.csv
    mission_reported/evidence/tle/sentinel-3a.parquet
        -> mission_reported__evidence__tle__sentinel-3a.parquet

Double underscore is the separator because satellite identifiers contain
single dashes (``sentinel-3a``), so a dash separator would be ambiguous.
The structured ``dataset/`` tree is never modified; the upload copy is
byte-verified against the source (size + SHA-256 per file).

The same convention is documented in ``dataset/docs/metadata.md`` and the
Data Records section of the paper.  Example::

    python3 scripts/data_pipeline/alignment/flat_release_export.py \
        --output results/figshare_flat
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path  # noqa: E402

DATASET_ROOT = REPO_ROOT / "dataset"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flat_name(relative: Path) -> str:
    """Encode a logical path as one flat file name (``__`` separators)."""
    return "__".join(relative.parts)


def build_flat(dataset_root: Path, output: Path) -> list[tuple[str, int]]:
    ensure_directory(output)
    mapping: list[tuple[str, int]] = []
    names: set[str] = set()
    for path in sorted(p for p in dataset_root.rglob("*") if p.is_file() and not p.name.startswith(".")):
        relative = path.relative_to(dataset_root)
        target_name = flat_name(relative)
        if target_name in names:
            raise SystemExit(f"flat-name collision: {target_name}")
        names.add(target_name)
        target = output / target_name
        target.write_bytes(path.read_bytes())
        if target.stat().st_size != path.stat().st_size or _sha256(target) != _sha256(path):
            raise SystemExit(f"copy verification failed: {relative}")
        mapping.append((str(relative), path.stat().st_size))
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--output", default=str(REPO_ROOT / "results" / "figshare_flat"))
    args = parser.parse_args()
    dataset_root = Path(resolve_repo_path(args.dataset_root))
    output = Path(resolve_repo_path(args.output))
    mapping = build_flat(dataset_root, output)
    total = sum(size for _, size in mapping)
    print(f"wrote {len(mapping)} files, {total/1e9:.2f} GB -> {output}")
    for relative, size in mapping:
        print(f"  {flat_name(Path(relative)):64s} {size:>12,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
