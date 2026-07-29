"""Locate a single storefront photograph.

Usage:
    python predict.py path/to/photo.jpg
    python predict.py path/to/photo.jpg --top-k 3 --visual-weight 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.localizer import DEFAULT_VISUAL_WEIGHT, Localizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate a storefront photograph.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--visual-weight", type=float, default=DEFAULT_VISUAL_WEIGHT)
    parser.add_argument("--no-text", action="store_true", help="disable the OCR signal")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"error: no such image: {args.image}", file=sys.stderr)
        return 1

    loc = Localizer()
    predictions = loc.locate(
        args.image,
        top_k=args.top_k,
        visual_weight=args.visual_weight,
        use_text=not args.no_text,
    )

    if not predictions:
        print("no match found")
        return 1

    top = predictions[0]
    print(f"\nlocation: {top.name}  ({top.floor_name} floor)")
    print(f"decided : {top.decided_by}"
          + ("" if top.has_images else "  (no reference photos for this unit)"))
    print(f"score   : {top.score:.3f}  (visual {top.visual_score:.3f}, text {top.text_score:.3f})")

    runner_up = predictions[1].score if len(predictions) > 1 else 0.0
    margin = top.score - runner_up
    if margin < 0.03:
        print("warning : margin over the runner-up is thin; treat as uncertain")

    print(f"\n{'rank':<5} {'store':<20} {'floor':<8} {'score':>7} {'visual':>7} {'text':>6}")
    print("-" * 60)
    for i, p in enumerate(predictions, 1):
        print(f"{i:<5} {p.name:<22} {p.floor_name:<8} {p.score:>7.3f} "
              f"{p.visual_score:>7.3f} {p.text_score:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
