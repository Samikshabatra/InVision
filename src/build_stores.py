"""Build the store list by aligning photographs directly to directory units.

Replaces the older segment-then-name approach, which merged neighbouring
shopfronts and lost every store but one inside each merged blob.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

from . import config
from .directory import load_directory
from .frame_align import TEXT_EVIDENCE, best_row_assignment, evidence_for
from .grouping import StoreGroup, brand_tokens, rank_brand_words


_UNSET = object()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unit"


def load_review() -> dict[str, str | None]:
    """Human corrections from the review pass, keyed by image path.

    These are ground truth. Everything the aligner infers is a guess about
    which shopfront a photograph shows; a reviewer has actually looked.
    """
    path = config.DATA_DIR / "label_review.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("assignments", {})


def _index_of_unit(unit_id: str, names: list[str], floor: int, row: str) -> int | None:
    """Locate a reviewed unit id within the row currently being built.

    Unit ids are minted as slug-f<floor>-<row initial>, so a correction that
    points at the other row or another floor cannot be honoured here.
    """
    suffix = f"-f{floor}-{row[0]}"
    if not unit_id.endswith(suffix) and f"-f{floor}-{row[0]}-" not in unit_id:
        return None
    base = unit_id[: unit_id.index(suffix)] if suffix in unit_id else unit_id
    for index, name in enumerate(names):
        if _slug(name) == base:
            return index
    return None


def build_stores(records: list[dict], embeddings: np.ndarray, ocr: dict,
                 directory: dict | None = None) -> tuple[list[StoreGroup], list[dict]]:
    """Return one StoreGroup per photographed directory unit, plus a report."""
    directory = directory or load_directory()
    aliases = directory.get("aliases", {})
    by_floor = {f["floor"]: f for f in directory["floors"]}

    tokens = [brand_tokens(ocr.get(r["rel_path"], {})) for r in records]
    reviewed = load_review()

    corridors: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        corridors[(rec["floor"], rec["side"])].append(i)
    for key in corridors:
        corridors[key].sort(key=lambda i: records[i]["order"])

    stores: list[StoreGroup] = []
    summary: list[dict] = []
    used_ids: set[str] = set()

    # Each floor's two walks must claim different rows.
    for floor in sorted({f for f, _ in corridors}):
        entry = by_floor.get(floor)
        if not entry:
            continue
        rows = entry["rows"]
        claimed: list[str] = []

        sides = [s for s in ("left", "right") if (floor, s) in corridors]
        for side in sides:
            frames = corridors[(floor, side)]
            available = [r for r in rows if r not in claimed] or list(rows)
            row, flipped, assignment, score = best_row_assignment(
                [tokens[i] for i in frames],
                embeddings[frames],
                rows,
                aliases,
                allowed_rows=available,
            )
            claimed.append(row)
            names = rows[row]

            by_unit: dict[int, list[int]] = defaultdict(list)
            for local, unit_index in enumerate(assignment):
                frame = frames[local]
                # A human decision beats anything inferred. An explicit null
                # means the frame shows no shopfront and is dropped entirely.
                override = reviewed.get(records[frame]["rel_path"], _UNSET)
                if override is not _UNSET:
                    if override is None:
                        continue
                    resolved = _index_of_unit(override, names, floor, row)
                    if resolved is None:
                        continue
                    unit_index = resolved
                by_unit[unit_index].append(frame)

            for unit_index in sorted(by_unit):
                members = by_unit[unit_index]
                name = names[unit_index]
                evidence = evidence_for(tokens, members, name, aliases)

                base = f"{_slug(name)}-f{floor}-{row[0]}"
                store_id, n = base, 2
                while store_id in used_ids:
                    store_id = f"{base}-{n}"
                    n += 1
                used_ids.add(store_id)

                ranked = rank_brand_words([tokens[i] for i in members])
                stores.append(StoreGroup(
                    store_id=store_id,
                    name=name,
                    name_confidence=round(float(evidence), 3),
                    floor=floor,
                    floor_name=entry["name"],
                    side=side,
                    seq=unit_index,
                    images=[records[i]["rel_path"] for i in members],
                    brand_tokens=[w.upper() for w, _ in ranked[:6]],
                    needs_review=evidence < TEXT_EVIDENCE,
                    review_reason="" if evidence >= TEXT_EVIDENCE
                                  else "placed by walk order; signage not legible",
                    ocr_name=(ranked[0][0].upper() if ranked else ""),
                    name_source="directory",
                    row=row,
                    map_index=unit_index,
                    row_length=len(names),
                ))

            corrected = sum(
                1 for i in frames if records[i]["rel_path"] in reviewed
            )
            summary.append({
                "reviewed_frames": corrected,
                "floor": floor,
                "side": side,
                "row": row,
                "direction": "east-to-west" if flipped else "west-to-east",
                "score_per_frame": round(score, 3),
                "frames": len(frames),
                "units_on_row": len(names),
                "units_photographed": len(by_unit),
                "units_missed": [n for k, n in enumerate(names) if k not in by_unit],
            })

    stores.sort(key=lambda s: (s.floor, s.row, s.map_index))
    return stores, summary
