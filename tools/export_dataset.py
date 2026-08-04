"""Materialise the recovered stores as a browsable, labelled image dataset.

The derived artefacts are vectors and indices, which are what the localiser
needs but are impossible to inspect by eye. This writes the same grouping as a
conventional dataset tree so the labels can be checked and corrected:

    data/dataset/
        gallery/<store_id>/*.jpg
        query/<store_id>/*.jpg
        labels.csv
        stores.csv
        DATASET_CARD.md

Images are re-encoded down to MAX_SIDE so the export stays small enough to copy
around; the originals under data/Images/ are never modified.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

MAX_SIDE = 1024
JPEG_QUALITY = 90


def load_split(name: str) -> dict[str, list[str]]:
    path = config.DERIVED_DIR / f"{name}_index.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row["store_id"], []).append(row["image"])
    return out


def export_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        longest = max(img.size)
        if longest > MAX_SIDE:
            scale = MAX_SIDE / longest
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)),
                Image.LANCZOS,
            )
        img.save(dst, "JPEG", quality=JPEG_QUALITY)


def main() -> int:
    stores = json.loads((config.DERIVED_DIR / "stores.json").read_text(encoding="utf-8"))
    by_id = {s["store_id"]: s for s in stores}
    splits = {"gallery": load_split("gallery"), "query": load_split("query")}

    out = config.DATA_DIR / "dataset"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    label_rows = []
    exported = 0
    for split, mapping in splits.items():
        for store_id, images in mapping.items():
            store = by_id[store_id]
            for n, rel in enumerate(images, 1):
                dst = out / split / store_id / f"{store_id}_{n:02d}.jpg"
                export_image(config.IMAGES_DIR / rel, dst)
                exported += 1
                label_rows.append({
                    "path": str(dst.relative_to(out)).replace("\\", "/"),
                    "split": split,
                    "store_id": store_id,
                    "store_name": store["name"],
                    "floor": store["floor"],
                    "floor_name": store["floor_name"],
                    "side": store["side"],
                    "corridor_seq": store["seq"],
                    "needs_review": store["needs_review"],
                    "source_image": rel,
                })

    label_rows.sort(key=lambda r: (r["floor"], r["side"], r["corridor_seq"], r["split"]))
    _write_csv(out / "labels.csv", label_rows)

    store_rows = [{
        "store_id": s["store_id"],
        "store_name": s["name"],
        "name_confidence": s["name_confidence"],
        "floor": s["floor"],
        "floor_name": s["floor_name"],
        "side": s["side"],
        "corridor_seq": s["seq"],
        "n_images": len(s["images"]),
        "needs_review": s["needs_review"],
        "review_reason": s["review_reason"],
        "alt_names": " | ".join(s["brand_tokens"][:4]),
    } for s in stores]
    store_rows.sort(key=lambda r: (r["floor"], r["side"], r["corridor_seq"]))
    _write_csv(out / "stores.csv", store_rows)

    (out / "DATASET_CARD.md").write_text(_card(stores, label_rows), encoding="utf-8")

    size_mb = sum(p.stat().st_size for p in out.rglob("*.jpg")) / 1e6
    print(f"exported {exported} images across {len(by_id)} stores -> {out}")
    print(f"  gallery: {sum(1 for r in label_rows if r['split'] == 'gallery')}")
    print(f"  query  : {sum(1 for r in label_rows if r['split'] == 'query')}")
    print(f"  size   : {size_mb:.0f} MB")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _card(stores: list[dict], rows: list[dict]) -> str:
    review = [s for s in stores if s["needs_review"]]
    corridors = sorted({(s["floor"], s["floor_name"], s["side"]) for s in stores})
    lines = [
        "# InVision store dataset",
        "",
        "Storefront photographs of a shopping mall, grouped by store, for",
        "vision-based indoor localisation.",
        "",
        "## Provenance",
        "",
        "Captured as six corridor walks (three floors x two sides), one photo",
        "burst per storefront. The source photographs carry **no store labels** -",
        "only a folder (floor + side) and a capture timestamp. Store identity was",
        "reconstructed by segmenting each walk into runs of consecutive frames",
        "showing the same shopfront, then naming each run from its signage text.",
        "",
        "Capture folders are named by ordinal where the mall's \"1st\" is the",
        "ground floor, so folder ordinals map one level down.",
        "",
        "## Contents",
        "",
        f"- Stores: **{len(stores)}**",
        f"- Images: **{len(rows)}** "
        f"({sum(1 for r in rows if r['split'] == 'gallery')} gallery, "
        f"{sum(1 for r in rows if r['split'] == 'query')} query)",
        f"- Corridors: **{len(corridors)}**",
        "",
        "| Floor | Side | Stores |",
        "| --- | --- | --- |",
    ]
    for floor, floor_name, side in corridors:
        n = sum(1 for s in stores if s["floor"] == floor and s["side"] == side)
        lines.append(f"| {floor_name} | {side} | {n} |")

    lines += [
        "",
        "## Known limitations",
        "",
        "**Labels are inferred, not verified.** Store names come from OCR of the",
        "signage, ranked by text height. Recurring failure modes:",
        "",
        "- leading characters absorbed by the logo mark (`ESTSIDE` = Westside,",
        "  `DASICS` = Asics)",
        "- visually similar food-court stalls merged into one store",
        "- generic facade words winning over the brand name",
        "",
        f"{len(review)} stores are flagged `needs_review` in `stores.csv`, but the",
        "failures above are **not** flagged - they need visual checking.",
        "",
        "**The query split understates difficulty.** Each held-out image was shot",
        "seconds from its own gallery images at nearly the same angle and lighting,",
        "so retrieval accuracy measured on it will be optimistic. A fair evaluation",
        "needs freshly captured photographs taken on a different day.",
        "",
        "Two upper floors were excluded at ingest for having too few images",
        "(1-5 each) to support recognition or routing.",
        "",
        "## Files",
        "",
        "- `gallery/<store_id>/` - reference images the localiser matches against",
        "- `query/<store_id>/` - held-out evaluation images",
        "- `labels.csv` - one row per image with full store metadata",
        "- `stores.csv` - one row per store, including review flags and alternate",
        "  name candidates from OCR",
        "",
        "Images are resized to a longest side of "
        f"{MAX_SIDE}px; originals are unmodified under `data/Images/`.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
