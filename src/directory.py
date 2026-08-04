"""Align recovered store groups against the official floor directory.

The capture walks recover stores in order but name them from OCR, which clips
logo characters and sometimes reads promotional copy instead of the brand. The
directory maps give the true brand and true order for every unit on a floor.

Matching the two is a sequence alignment problem, not a lookup: a walk may skip
units, merge visually similar neighbours, or run in either direction along
either row. So each corridor is aligned against each map row in both directions
with Needleman-Wunsch, and the highest-scoring assignment wins. Order is the
strongest evidence available - it disambiguates cases where the OCR name alone
is useless.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from . import config

# Gap penalty is deliberately mild: skipped and merged units are expected, so
# alignment should tolerate them rather than force bad pairings.
GAP_PENALTY = 0.12

# Below this similarity a pairing is recorded but flagged rather than trusted.
CONFIDENT_MATCH = 0.55

_NORM = re.compile(r"[^a-z0-9]+")


def normalise(name: str) -> str:
    return _NORM.sub("", name.lower())


def similarity(recovered: str, official: str, aliases: dict[str, list[str]] | None = None) -> float:
    """How well an OCR-recovered name matches an official brand name.

    Substring containment is scored highly because the characteristic OCR
    failure is losing leading or trailing characters ('ESTSIDE' for 'WESTSIDE',
    'DASICS' for 'ASICS'), which wrecks a plain ratio.

    Aliases cover the opposite problem: a fascia whose signage spells out what
    the directory abbreviates. 'MARKSSPENCER' and 'M&S' share almost no
    characters, so no amount of fuzzy matching connects them - only a listed
    alternate spelling can.
    """
    best_alias = 0.0
    for alt in (aliases or {}).get(official, []):
        best_alias = max(best_alias, _similarity_one(recovered, alt))
    return max(_similarity_one(recovered, official), best_alias)


def _similarity_one(recovered: str, official: str) -> float:
    a, b = normalise(recovered), normalise(official)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    best = SequenceMatcher(None, a, b).ratio()
    if len(a) >= 4 and a in b:
        best = max(best, 0.94)
    if len(b) >= 4 and b in a:
        best = max(best, 0.94)

    # Compare against each word of a multi-word brand, so 'PENN' matches
    # 'WILLIAM PENN' and 'BODY' matches 'THE BODY SHOP'. Only exact or
    # containment hits count: scoring weak per-word ratios here inflates
    # unrelated pairs and pulls the alignment off by one.
    for word in official.split():
        w = normalise(word)
        if len(w) >= 3 and len(a) >= 3:
            if a == w:
                best = max(best, 0.92)
            elif a in w or w in a:
                best = max(best, 0.86)
    return best


def align(recovered: list[str], official: list[str],
          aliases: dict[str, list[str]] | None = None
          ) -> tuple[float, list[tuple[int | None, int | None]]]:
    """Needleman-Wunsch alignment of two ordered name sequences."""
    n, m = len(recovered), len(official)
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] - GAP_PENALTY
    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] - GAP_PENALTY

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores[i][j] = max(
                scores[i - 1][j - 1] + similarity(recovered[i - 1], official[j - 1], aliases),
                scores[i - 1][j] - GAP_PENALTY,
                scores[i][j - 1] - GAP_PENALTY,
            )

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diagonal = scores[i - 1][j - 1] + similarity(recovered[i - 1], official[j - 1], aliases)
            if abs(scores[i][j] - diagonal) < 1e-9:
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(scores[i][j] - (scores[i - 1][j] - GAP_PENALTY)) < 1e-9:
            pairs.append((i - 1, None))
            i -= 1
            continue
        pairs.append((None, j - 1))
        j -= 1

    pairs.reverse()
    return scores[n][m], pairs


@dataclass
class CorridorMatch:
    side: str
    row: str
    reversed_row: bool
    score: float
    pairs: list[tuple[int | None, int | None]]
    official: list[str]


def load_directory(path: Path | None = None) -> dict:
    target = Path(path or config.DATA_DIR / "mall_directory.json")
    return json.loads(target.read_text(encoding="utf-8"))


def match_floor(recovered_by_side: dict[str, list[str]], rows: dict[str, list[str]],
                aliases: dict[str, list[str]] | None = None) -> list[CorridorMatch]:
    """Find the best side->row assignment, trying both rows and both directions.

    The capture walks did not use a consistent convention: one floor's 'left'
    is the map's top row read west to east, another's is the bottom row read
    east to west. Rather than assume, every combination is scored.
    """
    sides = [s for s in ("left", "right") if s in recovered_by_side]
    row_names = list(rows)

    best_total = None
    best_matches: list[CorridorMatch] = []

    assignments = [dict(zip(sides, row_names)), dict(zip(sides, reversed(row_names)))]
    for assignment in assignments:
        if len(set(assignment.values())) != len(assignment):
            continue
        total = 0.0
        matches: list[CorridorMatch] = []
        for side, row in assignment.items():
            candidates = []
            for flip in (False, True):
                official = list(reversed(rows[row])) if flip else list(rows[row])
                score, pairs = align(recovered_by_side[side], official, aliases)
                candidates.append((score, flip, pairs, official))
            score, flip, pairs, official = max(candidates, key=lambda c: c[0])
            total += score
            matches.append(CorridorMatch(side, row, flip, score, pairs, official))
        if best_total is None or total > best_total:
            best_total, best_matches = total, matches

    return best_matches


def build_corrections(stores: list[dict], directory: dict | None = None) -> dict:
    """Map every recovered store to its official brand where alignment allows."""
    directory = directory or load_directory()
    aliases = directory.get("aliases", {})
    by_floor = {f["floor"]: f for f in directory["floors"]}

    corrections: dict[str, dict] = {}
    summary = []

    floors = sorted({s["floor"] for s in stores})
    for floor in floors:
        entry = by_floor.get(floor)
        if not entry:
            continue

        recovered_by_side: dict[str, list[str]] = {}
        store_ids: dict[str, list[str]] = {}
        for side in ("left", "right"):
            on_side = sorted(
                (s for s in stores if s["floor"] == floor and s["side"] == side),
                key=lambda s: s["seq"],
            )
            if on_side:
                recovered_by_side[side] = [s["name"] for s in on_side]
                store_ids[side] = [s["store_id"] for s in on_side]

        if not recovered_by_side:
            continue

        matched = 0
        unmatched_official: list[str] = []
        for match in match_floor(recovered_by_side, entry["rows"], aliases):
            ids = store_ids[match.side]
            for rec_idx, off_idx in match.pairs:
                if rec_idx is None:
                    unmatched_official.append(match.official[off_idx])
                    continue
                if off_idx is None:
                    corrections[ids[rec_idx]] = {
                        "official_name": None,
                        "confidence": 0.0,
                        "row": match.row,
                        "map_index": None,
                        "row_length": len(match.official),
                        "note": "no directory unit aligned to this group",
                    }
                    continue
                official = match.official[off_idx]
                score = similarity(recovered_by_side[match.side][rec_idx], official, aliases)
                # off_idx indexes the possibly-reversed row, so convert back to
                # the map's own west-to-east ordering.
                canonical = (
                    len(match.official) - 1 - off_idx if match.reversed_row else off_idx
                )
                corrections[ids[rec_idx]] = {
                    "official_name": official,
                    "confidence": round(score, 3),
                    "row": match.row,
                    "map_index": canonical,
                    "row_length": len(match.official),
                    "note": "" if score >= CONFIDENT_MATCH else "weak name match; ordering only",
                }
                if score >= CONFIDENT_MATCH:
                    matched += 1

            summary.append({
                "floor": floor,
                "side": match.side,
                "row": match.row,
                "direction": "east-to-west" if match.reversed_row else "west-to-east",
                "align_score": round(match.score, 2),
                "recovered": len(ids),
                "official_units": len(match.official),
            })

        summary[-1]["confident_matches_on_floor"] = matched
        if unmatched_official:
            summary[-1]["directory_units_not_captured"] = unmatched_official

    return {"corrections": corrections, "alignment": summary}
