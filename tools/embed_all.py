"""Embed every captured image with CLIP and cache the result.

Embeddings are expensive on CPU and reused by grouping, calibration and the
final dataset build, so they are computed once and written to disk.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, encoders  # noqa: E402
from src.ingest import records_to_json, scan_images  # noqa: E402


def main() -> int:
    records, notes = scan_images()
    for note in notes:
        print(f"  note: {note}")
    if not records:
        print("no images ingested", file=sys.stderr)
        return 1

    print(f"\ningested {len(records)} images across "
          f"{len({(r.floor, r.side) for r in records})} floor/side corridors")

    out = config.DERIVED_DIR
    out.mkdir(parents=True, exist_ok=True)

    paths = [config.IMAGES_DIR / r.rel_path for r in records]
    print(f"embedding with {config.CLIP_MODEL} (first run downloads weights)...")
    start = time.time()
    embeddings, failed = encoders.embed_images(paths)
    elapsed = time.time() - start

    if failed:
        print(f"  {len(failed)} images failed to load")
        keep = [i for i in range(len(records)) if i not in set(failed)]
        records = [records[i] for i in keep]

    np.save(out / "all_embeddings.npy", embeddings)
    (out / "all_index.json").write_text(
        json.dumps(records_to_json(records), indent=2), encoding="utf-8"
    )

    print(f"embedded {embeddings.shape[0]} images, dim {embeddings.shape[1]}, "
          f"in {elapsed:.1f}s ({elapsed / max(embeddings.shape[0], 1):.2f}s each)")
    print(f"cached to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
