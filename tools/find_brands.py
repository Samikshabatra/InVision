"""Search the cached OCR for brands the alignment reported as never captured.

If the signage text is present, the photographs exist and the loss happened in
grouping or alignment - not in the capture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.directory import normalise  # noqa: E402

TARGETS = [
    "NIKE", "SPEEDO", "CELIO", "ORIGEM", "ALDO", "SHAYA", "ZEISS",
    "DOGFATHER", "TACO BELL", "WOW MOMO", "WOW CHINA", "DINDIGUL",
    "FISH N CHIPS", "SUBWAY", "CASIO", "GO COLORS", "ONEPLUS",
    "BURGER KING", "SHIVSAGAR", "MOD", "BOBA", "DRICKLE", "BELGIAN",
]


def main() -> int:
    ocr = json.loads((config.DERIVED_DIR / "ocr_raw.json").read_text(encoding="utf-8"))
    index = json.loads((config.DERIVED_DIR / "all_index.json").read_text(encoding="utf-8"))
    folder_of = {r["rel_path"]: r["folder"] for r in index}

    stores = json.loads((config.DERIVED_DIR / "stores.json").read_text(encoding="utf-8"))
    assigned = {img: s["name"] for s in stores for img in s["images"]}

    print(f"{'brand':<16} {'hits':>5}  where")
    print("-" * 72)
    for target in TARGETS:
        needle = normalise(target)
        hits = []
        for rel, entry in ocr.items():
            words = [normalise(t.get("text", "")) for t in entry.get("tokens", [])]
            joined = normalise(entry.get("text", ""))
            if any(needle and (needle in w or w in needle and len(w) >= 4) for w in words if w) \
                    or (needle and needle in joined):
                hits.append(rel)
        if hits:
            groups = sorted({assigned.get(h, "?") for h in hits})
            folders = sorted({folder_of.get(h, "?") for h in hits})
            print(f"{target:<16} {len(hits):>5}  {', '.join(folders)}  ->  "
                  f"grouped as: {', '.join(groups[:4])}")
        else:
            print(f"{target:<16} {0:>5}  not found in any OCR read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
