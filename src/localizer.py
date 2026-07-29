"""The localiser - turn a storefront photograph into a place on the map.

Candidates are the units on the floor directory, not the groups of photographs
recovered from the survey walks. That distinction matters: the walks missed some
shopfronts and mislabelled others, and a localiser restricted to what they
captured can never name a store they got wrong. Photographs supply the visual
half of the evidence; the directory supplies the list of answers.

Two signals are fused:

  visual - cosine similarity to a unit's reference photographs, aggregated by
           best match, which is standard for visual place recognition
  text   - agreement between signage read off the query and the unit's official
           name from the directory

They fail differently. Two outlets of one brand look alike and read alike, while
a shopfront shot at a bad angle may read poorly yet still match visually.

Legible signage outranks appearance. A photograph whose fascia clearly reads one
brand is that brand, even when the nearest reference image belongs to the shop
next door - which is exactly what went wrong when the gallery drove the decision
alone, because a mislabelled group taught the wrong name. Text also reaches
units that were never photographed, which visual retrieval can never do.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from . import config, encoders
from .directory import similarity as name_similarity
from .grouping import brand_tokens
from .mall_graph import MallGraph

# Weight on the visual signal; the remainder goes to signage text.
DEFAULT_VISUAL_WEIGHT = 0.7

# Signage this legible decides the answer on its own...
TEXT_OVERRIDE = 0.42

# ...provided it is legible at all. A survey shot often frames two shopfronts
# at once and both names read clearly; the taller lettering is the better guess
# but not a safe one, so anything inside this margin is reported as a close call
# and the runner-up is offered alongside.
TEXT_OVERRIDE_MARGIN = 0.02
TEXT_CONFIDENT_MARGIN = 0.12

FUZZY_MATCH_RATIO = 0.82


@dataclass
class Prediction:
    unit_id: str
    name: str
    floor: int
    floor_name: str
    row: str
    index: int
    score: float
    visual_score: float
    text_score: float
    has_images: bool
    decided_by: str          # "signage" | "fusion"
    best_match_image: str = ""

    # Kept so existing callers that expect a store identifier keep working.
    @property
    def store_id(self) -> str:
        return self.unit_id


def _fuzzy_contains(word: str, vocabulary: set[str]) -> float:
    """Loose word match tolerant of the character clipping OCR introduces."""
    if word in vocabulary:
        return 1.0
    best = 0.0
    for candidate in vocabulary:
        if len(word) >= 4 and (word in candidate or candidate in word):
            best = max(best, 0.9)
            continue
        if abs(len(word) - len(candidate)) <= 3:
            ratio = SequenceMatcher(None, word, candidate).ratio()
            if ratio >= FUZZY_MATCH_RATIO:
                best = max(best, ratio)
    return best


class Localizer:
    def __init__(self, derived_dir: Path | None = None, graph: MallGraph | None = None):
        derived = Path(derived_dir or config.DERIVED_DIR)

        self.graph = graph or MallGraph.from_derived()
        self.units = self.graph.units
        self.aliases = self.graph.directory.get("aliases", {})

        self.gallery = np.load(derived / "gallery_embeddings.npy")
        self.index = json.loads((derived / "gallery_index.json").read_text(encoding="utf-8"))

        # Gallery rows grouped by the directory unit they belong to.
        self.rows_by_unit: dict[str, list[int]] = {}
        for row, entry in enumerate(self.index):
            unit_id = self.graph.unit_for_store(entry["store_id"])
            if unit_id:
                self.rows_by_unit.setdefault(unit_id, []).append(row)

        # Name words per unit, for loose matching when the full-name comparison
        # is weak (multi-word brands read one word at a time).
        self.words_by_unit: dict[str, set[str]] = {}
        for unit_id, unit in self.units.items():
            variants = {unit["name"]} | set(self.aliases.get(unit["name"], []))
            words = set()
            for variant in variants:
                for word in re.split(r"[^A-Za-z0-9]+", variant.lower()):
                    if len(word) >= config.MIN_BRAND_TOKEN_LEN:
                        words.add(word)
            self.words_by_unit[unit_id] = words

    def embed_query(self, image_path: Path) -> np.ndarray:
        vectors, failed = encoders.embed_images([Path(image_path)])
        if failed or not len(vectors):
            raise ValueError(f"could not read image: {image_path}")
        return vectors[0]

    def visual_scores(self, query_vector: np.ndarray) -> tuple[dict[str, float], dict[str, str]]:
        sims = self.gallery @ query_vector
        scores: dict[str, float] = {}
        best_image: dict[str, str] = {}
        for unit_id, rows in self.rows_by_unit.items():
            best = int(max(rows, key=lambda r: sims[r]))
            scores[unit_id] = float(sims[best])
            best_image[unit_id] = self.index[best]["image"]
        return scores, best_image

    def text_scores(self, image_path: Path | None = None, ocr_entry: dict | None = None) -> dict[str, float]:
        """Match query signage against every unit's official name.

        Weighted by cap height, because the fascia of the shop you are standing
        in front of is photographed far larger than a neighbour's caught at the
        edge of frame.
        """
        if ocr_entry is None:
            if image_path is None:
                raise ValueError("provide either image_path or ocr_entry")
            ocr_entry = encoders.ocr_image(Path(image_path))
        tokens = brand_tokens(ocr_entry)
        if not tokens:
            return {}

        scores: dict[str, float] = {}
        for unit_id, unit in self.units.items():
            best = 0.0
            for word, info in tokens.items():
                weight = 0.25 + 0.75 * float(info.get("rel_height", 0.0))
                match = name_similarity(word, unit["name"], self.aliases)
                if match < 0.6:
                    match = max(match, _fuzzy_contains(word, self.words_by_unit[unit_id]))
                best = max(best, match * weight)
            scores[unit_id] = best
        return scores

    def locate(
        self,
        image_path: Path,
        top_k: int = 5,
        visual_weight: float = DEFAULT_VISUAL_WEIGHT,
        use_text: bool = True,
        ocr_entry: dict | None = None,
    ) -> list[Prediction]:
        query = self.embed_query(image_path)
        visual, best_image = self.visual_scores(query)
        text = self.text_scores(image_path, ocr_entry) if use_text else {}

        w_visual = visual_weight if text else 1.0
        w_text = (1.0 - visual_weight) if text else 0.0

        ranked: list[Prediction] = []
        for unit_id, unit in self.units.items():
            v = visual.get(unit_id, 0.0)
            t = text.get(unit_id, 0.0)
            ranked.append(Prediction(
                unit_id=unit_id,
                name=unit["name"],
                floor=unit["floor"],
                floor_name=unit["floor_name"],
                row=unit["row"],
                index=unit["index"],
                score=round(w_visual * v + w_text * t, 4),
                visual_score=round(v, 4),
                text_score=round(t, 4),
                has_images=unit["image_count"] > 0,
                decided_by="fusion",
                best_match_image=best_image.get(unit_id, ""),
            ))

        ranked.sort(key=lambda p: p.score, reverse=True)

        # A clearly legible fascia settles it. Fusion cannot: a unit with no
        # reference photographs scores zero on the visual channel however plain
        # its sign, and a mislabelled gallery actively pulls toward the shop
        # next door.
        if text:
            by_text = sorted(ranked, key=lambda p: p.text_score, reverse=True)
            top, second = by_text[0], (by_text[1] if len(by_text) > 1 else None)
            margin = top.text_score - (second.text_score if second else 0.0)
            if top.text_score >= TEXT_OVERRIDE and margin >= TEXT_OVERRIDE_MARGIN:
                confident = margin >= TEXT_CONFIDENT_MARGIN
                top.decided_by = "signage" if confident else "signage (close call)"
                top.score = round(max(top.score, 0.5 + 0.5 * top.text_score), 4)
                rest = [p for p in ranked if p.unit_id != top.unit_id]
                # Keep the rival reading directly behind the winner so the
                # interface can offer it as a one-tap correction.
                if second is not None and not confident:
                    rest = [p for p in rest if p.unit_id != second.unit_id]
                    rest.insert(0, second)
                ranked = [top] + rest

        return ranked[:top_k]
