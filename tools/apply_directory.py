"""Align recovered stores to the official directory and report the result.

Dry run by default so the alignment can be inspected before it rewrites names.
Pass --write to apply the corrections to stores.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.directory import build_corrections  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="apply corrections to stores.json")
    args = parser.parse_args()

    stores_path = config.DERIVED_DIR / "stores.json"
    stores = json.loads(stores_path.read_text(encoding="utf-8"))
    result = build_corrections(stores)
    corrections = result["corrections"]

    print("CORRIDOR ALIGNMENT")
    print(f"{'floor':<7} {'side':<6} {'row':<7} {'direction':<14} "
          f"{'score':>6} {'walk':>5} {'map':>5}")
    print("-" * 60)
    for row in result["alignment"]:
        print(f"{row['floor']:<7} {row['side']:<6} {row['row']:<7} {row['direction']:<14} "
              f"{row['align_score']:>6.1f} {row['recovered']:>5} {row['official_units']:>5}")

    renamed = [(sid, c) for sid, c in corrections.items() if c["official_name"]]
    strong = [c for _, c in renamed if c["confidence"] >= 0.55]
    print(f"\n{len(renamed)}/{len(stores)} stores aligned to a directory unit")
    print(f"{len(strong)} with a confident name match, "
          f"{len(renamed) - len(strong)} placed by ordering alone")

    print("\nSAMPLE CORRECTIONS")
    by_store = {s["store_id"]: s for s in stores}
    shown = 0
    for sid, c in corrections.items():
        if not c["official_name"]:
            continue
        old = by_store[sid]["name"]
        if normal(old) == normal(c["official_name"]):
            continue
        flag = "" if c["confidence"] >= 0.55 else "   [order only]"
        print(f"  {old:<16} -> {c['official_name']:<28} {c['confidence']:.2f}{flag}")
        shown += 1
        if shown >= 30:
            break

    missing = [
        u for row in result["alignment"]
        for u in row.get("directory_units_not_captured", [])
    ]
    if missing:
        print(f"\nDIRECTORY UNITS NEVER PHOTOGRAPHED ({len(missing)})")
        print("  " + ", ".join(missing[:24]) + (" ..." if len(missing) > 24 else ""))

    out = config.DERIVED_DIR / "name_corrections.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")

    if args.write:
        for store in stores:
            c = corrections.get(store["store_id"])
            if c and c["official_name"]:
                store["ocr_name"] = store["name"]
                store["name"] = c["official_name"]
                store["name_source"] = "directory"
                store["name_confidence"] = c["confidence"]
                store["needs_review"] = c["confidence"] < 0.55
                store["review_reason"] = c["note"]
            else:
                store["name_source"] = "ocr"
                store["needs_review"] = True
                store["review_reason"] = "not aligned to any directory unit"
        stores_path.write_text(json.dumps(stores, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"applied corrections to {stores_path}")
    else:
        print("dry run - re-run with --write to apply")
    return 0


def normal(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


if __name__ == "__main__":
    raise SystemExit(main())
