"""Entry point - build the InVision store dataset.

Prerequisites (cached, run once):
    python tools/extract_zips.py
    python tools/embed_all.py
    python tools/ocr_all.py

Then:
    python build_dataset.py
"""

from __future__ import annotations

import sys

from src.pipeline import build_dataset


def main() -> int:
    try:
        r = build_dataset()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ips = r["images_per_store"]
    print(f"images ingested   : {r['images_ingested']}")
    print(f"stores recovered  : {r['stores_recovered']}")
    print(f"images per store  : min {ips['min']}  median {ips['median']}  "
          f"mean {ips['mean']}  max {ips['max']}")
    print(f"gallery vectors   : {r['gallery_vectors']} (dim {r['embedding_dim']})")
    print(f"query vectors     : {r['query_vectors']}")
    print(f"needing review    : {r['stores_needing_review']}")
    print(f"unnamed stores    : {r['stores_unnamed']}")
    print(f"duplicate names   : {len(r['duplicate_names'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
