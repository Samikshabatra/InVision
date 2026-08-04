"""Diagnostic probe: capture-gap distribution and OCR legibility sample.

Answers two questions before any dataset is built:
  1. Do consecutive shots cluster into per-store bursts?
  2. Is the storefront signage readable enough to auto-label stores?
"""

from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES = PROJECT_ROOT / "data" / "Images"
STAMP = re.compile(r"(\d{8})_(\d{6})")


def parse_time(path: Path) -> datetime | None:
    m = STAMP.search(path.stem)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def gap_report() -> None:
    print("=" * 60)
    print("CAPTURE GAP ANALYSIS (seconds between consecutive shots)")
    print("=" * 60)
    for folder in sorted(IMAGES.iterdir()):
        if not folder.is_dir():
            continue
        times = sorted(t for t in (parse_time(p) for p in folder.glob("*.jpg")) if t)
        if len(times) < 2:
            print(f"{folder.name:<12} {len(times):>3} images  (too few for gaps)")
            continue
        gaps = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
        burst = sum(1 for g in gaps if g <= 8)
        print(
            f"{folder.name:<12} {len(times):>3} images  "
            f"median gap {statistics.median(gaps):>6.1f}s  "
            f"max {max(gaps):>6.0f}s  "
            f"gaps<=8s: {burst}/{len(gaps)}"
        )


def ocr_report(sample: int = 12) -> None:
    print()
    print("=" * 60)
    print(f"OCR LEGIBILITY SAMPLE ({sample} images)")
    print("=" * 60)
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("rapidocr_onnxruntime not importable")
        return

    engine = RapidOCR()
    paths = sorted((IMAGES / "3rd Left").glob("*.jpg"))[:sample]
    for p in paths:
        try:
            result, _ = engine(str(p))
        except Exception as exc:
            print(f"{p.name}  ERROR {exc}")
            continue
        if not result:
            print(f"{p.name}  <no text>")
            continue
        tokens = [(d[1], float(d[2])) for d in result if len(d) >= 3 and float(d[2]) >= 0.5]
        tokens.sort(key=lambda t: -t[1])
        shown = ", ".join(f"{t}({s:.2f})" for t, s in tokens[:5])
        print(f"{p.name}  {shown or '<all low confidence>'}")


if __name__ == "__main__":
    if not IMAGES.exists():
        print(f"missing {IMAGES}", file=sys.stderr)
        raise SystemExit(1)
    gap_report()
    ocr_report()
