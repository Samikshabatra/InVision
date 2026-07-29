"""Run OCR over every captured image and cache the detections.

Slow on CPU (roughly a second per image), so results are written once and
reused by grouping and labelling.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, encoders  # noqa: E402


def main() -> int:
    index_path = config.DERIVED_DIR / "all_index.json"
    if not index_path.exists():
        print("run tools/embed_all.py first", file=sys.stderr)
        return 1

    records = json.loads(index_path.read_text(encoding="utf-8"))
    out_path = config.DERIVED_DIR / "ocr_raw.json"
    cached = {}
    if out_path.exists():
        cached = json.loads(out_path.read_text(encoding="utf-8"))

    todo = [r for r in records if r["rel_path"] not in cached]
    print(f"{len(records)} images, {len(cached)} cached, {len(todo)} to process")

    start = time.time()
    for i, rec in enumerate(todo, 1):
        rel = rec["rel_path"]
        cached[rel] = encoders.ocr_image(config.IMAGES_DIR / rel)
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - start) / i
            print(f"  {i}/{len(todo)}  {rate:.2f}s/image  "
                  f"eta {(len(todo) - i) * rate / 60:.1f}min", flush=True)

    out_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
    with_text = sum(1 for v in cached.values() if v.get("text"))
    print(f"\ndone in {(time.time() - start) / 60:.1f}min")
    print(f"images with readable text: {with_text}/{len(cached)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
