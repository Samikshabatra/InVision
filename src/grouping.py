"""Steps 2-3 - recover store identity from unlabelled corridor walks.

The captured photographs carry no store labels, so stores are reconstructed by
segmenting each corridor walk into runs of consecutive shots that show the same
shopfront, then naming each run from its signage text.

Two signals drive segmentation:
  * shared brand text between consecutive frames (primary, high precision)
  * CLIP embedding similarity (fallback, for frames OCR could not read)

Brand text is preferred because the calibrated similarity threshold sits close
to the median adjacent similarity, making embeddings alone unreliable here.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np

from . import config

# Generic mall and promotional vocabulary that must never name a store.
STOPWORDS = {
    "exit", "entry", "entrance", "open", "opens", "closed", "sale", "off",
    "new", "now", "free", "offer", "offers", "discount", "upto", "onwards",
    "shop", "store", "stores", "outlet", "floor", "level", "ground", "lift",
    "escalator", "stairs", "toilet", "washroom", "restroom", "parking",
    "welcome", "here", "buy", "get", "and", "the", "for", "with", "all",
    "only", "more", "best", "top", "big", "mega", "super", "collection",
    "delivery", "pickup", "menu", "price", "prices", "rs", "inr", "flat",
    "min", "max", "you", "your", "our", "any", "was", "per", "avilable",
    "available", "coming", "soon", "closing", "opening", "hours", "time",
    # Sale signage, including the non-English wording used on facade posters.
    "soldes", "saldi", "rebajas", "sconti", "season", "finale", "fnale",
    "end", "ends", "ending", "clearance", "styles", "sizes", "starting",
    "everything", "selected", "assured", "exchange", "order", "ready",
    "here", "shopping", "brands", "range", "wide", "special", "limited",
    # The mall's own branding appears on every corridor and is not a store.
    "nexus", "mall", "select",
}

_ALPHA = re.compile(r"[^A-Za-z]+")


@dataclass
class StoreGroup:
    store_id: str
    name: str
    name_confidence: float
    floor: int
    floor_name: str
    side: str
    seq: int
    images: list[str] = field(default_factory=list)
    brand_tokens: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reason: str = ""
    ocr_name: str = ""
    name_source: str = "ocr"
    row: str = ""          # "top" / "bottom" as drawn on the floor map
    map_index: int = -1    # position along that row, west to east
    row_length: int = 0

    @property
    def image_count(self) -> int:
        return len(self.images)


def brand_tokens(ocr_entry: dict, min_score: float = 0.6) -> dict[str, dict]:
    """Extract candidate brand words from one image's OCR detections.

    Returns token -> {score, rel_height}, where rel_height is the detection's
    cap height as a fraction of the tallest text in the same photograph. The
    shopfront name is almost always the largest lettering on the facade, so
    relative height separates it from promotional copy that repeats just as
    often across a store's frames.
    """
    detections = ocr_entry.get("tokens", [])
    heights = [
        float(d.get("height", 0.0))
        for d in detections
        if float(d.get("score", 0.0)) >= min_score
    ]
    tallest = max(heights) if heights else 0.0

    out: dict[str, dict] = {}
    for det in detections:
        score = float(det.get("score", 0.0))
        if score < min_score:
            continue
        rel = (float(det.get("height", 0.0)) / tallest) if tallest > 0 else 0.0
        for raw in det.get("text", "").split():
            word = _ALPHA.sub("", raw).lower()
            if len(word) < config.MIN_BRAND_TOKEN_LEN or word in STOPWORDS:
                continue
            prev = out.get(word)
            if prev is None:
                out[word] = {"score": score, "rel_height": round(rel, 3)}
            else:
                prev["score"] = max(prev["score"], score)
                prev["rel_height"] = max(prev["rel_height"], round(rel, 3))
    return out


def _shares_brand(a: dict[str, dict], b: dict[str, dict], min_score: float = 0.75) -> bool:
    """True when two frames agree on at least one confident brand word."""
    for word, info_a in a.items():
        info_b = b.get(word)
        if info_b is not None and min(info_a["score"], info_b["score"]) >= min_score:
            return True
    return False


def segment_corridor(rows: list[int], embeddings: np.ndarray, tokens: list[dict]) -> list[list[int]]:
    """Split one corridor walk into runs of frames showing the same store."""
    if not rows:
        return []

    groups: list[list[int]] = [[rows[0]]]
    for prev, curr in zip(rows, rows[1:]):
        sim = float(embeddings[prev] @ embeddings[curr])
        same = _shares_brand(tokens[prev], tokens[curr]) or sim >= config.GROUPING_SIMILARITY
        if same:
            groups[-1].append(curr)
        else:
            groups.append([curr])
    return groups


def name_group(indices: list[int], tokens: list[dict]) -> tuple[str, float, list[str]]:
    """Name a store from the signage words its frames agree on.

    Words appearing in several frames of the same run are far more likely to be
    the shopfront name than one-off promotional text, so frequency across the
    run is weighted ahead of raw OCR confidence.
    """
    ranked = rank_brand_words([tokens[i] for i in indices])
    if not ranked:
        return "", 0.0, []
    top, score = ranked[0]
    return top.upper(), round(score, 3), [w.upper() for w, _ in ranked[:6]]


def rank_brand_words(frames: list[dict[str, dict]]) -> list[tuple[str, float]]:
    """Rank signage words across a set of frames, most brand-like first.

    Height dominates because the shopfront name is the largest lettering;
    repetition across frames and OCR confidence break ties.
    """
    if not frames:
        return []

    freq: Counter[str] = Counter()
    best_score: dict[str, float] = {}
    best_height: dict[str, float] = {}
    for frame in frames:
        for word, info in frame.items():
            freq[word] += 1
            best_score[word] = max(best_score.get(word, 0.0), info["score"])
            best_height[word] = max(best_height.get(word, 0.0), info["rel_height"])

    if not freq:
        return []

    def rank(word: str) -> float:
        return (
            best_height[word] * 0.50
            + (freq[word] / len(frames)) * 0.30
            + best_score[word] * 0.20
        )

    return sorted(((w, rank(w)) for w in freq), key=lambda x: x[1], reverse=True)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "unknown"


def build_groups(records: list[dict], embeddings: np.ndarray, ocr: dict) -> list[StoreGroup]:
    """Reconstruct every store across all corridors."""
    tokens = [brand_tokens(ocr.get(r["rel_path"], {})) for r in records]

    by_corridor: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        by_corridor[(rec["floor"], rec["side"])].append(i)
    for key in by_corridor:
        by_corridor[key].sort(key=lambda i: records[i]["order"])

    stores: list[StoreGroup] = []
    used_ids: set[str] = set()

    for (floor, side), rows in sorted(by_corridor.items()):
        runs = segment_corridor(rows, embeddings, tokens)
        for seq, run in enumerate(runs):
            name, confidence, candidates = name_group(run, tokens)
            display = name or f"UNKNOWN {floor}-{side}-{seq}"

            base = f"{_slug(display)}-f{floor}-{side[0]}"
            store_id = base
            n = 2
            while store_id in used_ids:
                store_id = f"{base}-{n}"
                n += 1
            used_ids.add(store_id)

            reasons = []
            if not name:
                reasons.append("no readable brand text")
            if confidence < 0.6:
                reasons.append(f"low name confidence ({confidence})")
            if len(run) < 2:
                reasons.append("single image")

            stores.append(
                StoreGroup(
                    store_id=store_id,
                    name=display,
                    name_confidence=confidence,
                    floor=floor,
                    floor_name=config.FLOOR_DISPLAY_NAMES.get(floor, str(floor)),
                    side=side,
                    seq=seq,
                    images=[records[i]["rel_path"] for i in run],
                    brand_tokens=candidates,
                    needs_review=bool(reasons),
                    review_reason="; ".join(reasons),
                )
            )

    return stores


def groups_to_json(groups: list[StoreGroup]) -> list[dict]:
    return [asdict(g) for g in groups]
