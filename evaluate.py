"""Evaluate the localiser on the held-out query split, with an ablation.

Reports top-k retrieval accuracy for each signal in isolation and fused, so the
contribution of the text channel is visible rather than assumed.

Read the accuracy caveat printed at the end before quoting these numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src import config
from src.localizer import Localizer
from src.mall_graph import MallGraph


def load_queries():
    derived = config.DERIVED_DIR
    embeddings = np.load(derived / "query_embeddings.npy")
    index = json.loads((derived / "query_index.json").read_text(encoding="utf-8"))
    ocr = json.loads((derived / "ocr_raw.json").read_text(encoding="utf-8"))
    return embeddings, index, ocr


def rank_stores(loc: Localizer, query_vec, ocr_entry, visual_weight: float, use_text: bool):
    visual, _ = loc.visual_scores(query_vec)
    text = loc.text_scores(ocr_entry=ocr_entry) if use_text else {}

    w_v = visual_weight if text else 1.0
    w_t = (1.0 - visual_weight) if text else 0.0

    scored = [
        (unit_id, w_v * visual.get(unit_id, 0.0) + w_t * text.get(unit_id, 0.0))
        for unit_id in loc.units
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in scored]


def accuracy_at_k(loc, embeddings, index, ocr, visual_weight, use_text, ks=(1, 3, 5)):
    hits = {k: 0 for k in ks}
    floor_hits = 0
    for row, entry in enumerate(index):
        ranked = rank_stores(loc, embeddings[row], ocr.get(entry["image"], {}), visual_weight, use_text)
        truth = loc.graph.unit_for_store(entry["store_id"])
        if truth is None:
            continue
        for k in ks:
            if truth in ranked[:k]:
                hits[k] += 1
        # Floor accuracy matters independently: routing to the wrong floor is
        # the failure a user actually notices.
        if ranked and loc.units[ranked[0]]["floor"] == loc.units[truth]["floor"]:
            floor_hits += 1
    n = len(index)
    return {f"top{k}": hits[k] / n for k in ks} | {"floor": floor_hits / n, "n": n}


def main() -> int:
    try:
        embeddings, index, ocr = load_queries()
    except FileNotFoundError as exc:
        print(f"error: {exc}\nrun build_dataset.py first", file=sys.stderr)
        return 1

    if not len(index):
        print("no held-out queries to evaluate", file=sys.stderr)
        return 1

    loc = Localizer()
    print(f"gallery: {loc.gallery.shape[0]} vectors over {len(loc.rows_by_unit)} units "
          f"({len(loc.units)} directory units are candidates)")
    print(f"queries: {len(index)}\n")

    configs = [
        ("visual only (CLIP)", 1.0, False),
        ("text only (OCR)", 0.0, True),
        ("fusion 0.5 / 0.5", 0.5, True),
        ("fusion 0.7 / 0.3", 0.7, True),
        ("fusion 0.85 / 0.15", 0.85, True),
    ]

    print(f"{'configuration':<22} {'top-1':>7} {'top-3':>7} {'top-5':>7} {'floor':>7}")
    print("-" * 54)
    best = None
    for label, weight, use_text in configs:
        r = accuracy_at_k(loc, embeddings, index, ocr, weight, use_text)
        print(f"{label:<22} {r['top1']:>7.1%} {r['top3']:>7.1%} {r['top5']:>7.1%} {r['floor']:>7.1%}")
        if best is None or r["top1"] > best[1]["top1"]:
            best = (label, r)

    print("-" * 54)
    print(f"best top-1: {best[0]} at {best[1]['top1']:.1%}\n")

    print("CAVEAT: every query image was captured seconds from its own gallery")
    print("images, at nearly the same angle and lighting, so these figures are")
    print("optimistic. A fair number needs photographs taken on a different day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
