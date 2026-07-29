"""Step 1 - read the flat capture folders into ordered image records.

The photographs carry no store labels; the only structure available is the
folder (floor + corridor side) and the capture timestamp, which preserves the
order in which the corridor was walked. That walking order is what later steps
segment into per-store groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from . import config

_FOLDER = re.compile(config.FOLDER_PATTERN, re.IGNORECASE)
_STAMP = re.compile(config.TIMESTAMP_PATTERN)


@dataclass
class ImageRecord:
    rel_path: str
    folder: str
    floor: int
    floor_name: str
    side: str
    captured_at: str
    order: int  # position along the corridor walk, 0-based within folder


def parse_folder(name: str) -> tuple[int, str] | None:
    """Map a capture folder name to (real floor, side), applying the offset."""
    m = _FOLDER.match(name.strip())
    if not m:
        return None
    ordinal = int(m.group(1))
    floor = config.FOLDER_ORDINAL_TO_FLOOR.get(ordinal)
    if floor is None:
        return None
    return floor, m.group(2).lower()


def parse_timestamp(stem: str) -> datetime | None:
    m = _STAMP.search(stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def scan_images(images_dir: Path | None = None) -> tuple[list[ImageRecord], list[str]]:
    """Return ordered image records plus notes about anything skipped."""
    root = Path(images_dir or config.IMAGES_DIR)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    records: list[ImageRecord] = []
    notes: list[str] = []

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        parsed = parse_folder(folder.name)
        if parsed is None:
            notes.append(f"skipped unrecognised folder: {folder.name}")
            continue
        floor, side = parsed

        files = [p for p in folder.iterdir() if p.suffix.lower() in config.IMAGE_EXTS]
        timed: list[tuple[datetime, Path]] = []
        for path in files:
            ts = parse_timestamp(path.stem)
            if ts is None:
                notes.append(f"skipped file without timestamp: {folder.name}/{path.name}")
                continue
            timed.append((ts, path))

        if len(timed) < config.MIN_IMAGES_PER_FLOOR_SIDE:
            notes.append(
                f"excluded {folder.name}: only {len(timed)} images "
                f"(minimum {config.MIN_IMAGES_PER_FLOOR_SIDE})"
            )
            continue

        timed.sort(key=lambda t: t[0])
        for order, (ts, path) in enumerate(timed):
            records.append(
                ImageRecord(
                    rel_path=f"{folder.name}/{path.name}",
                    folder=folder.name,
                    floor=floor,
                    floor_name=config.FLOOR_DISPLAY_NAMES.get(floor, str(floor)),
                    side=side,
                    captured_at=ts.isoformat(),
                    order=order,
                )
            )

    return records, notes


def records_to_json(records: list[ImageRecord]) -> list[dict]:
    return [asdict(r) for r in records]
