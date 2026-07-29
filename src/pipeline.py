"""Steps 4-5 - held-out split, artefact writing, and the sanity report.

Consumes the cached embeddings and OCR produced by tools/embed_all.py and
tools/ocr_all.py, reconstructs stores, and emits the dataset the localiser
will query against.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import config
from .build_stores import build_stores
from .grouping import StoreGroup, brand_tokens, groups_to_json, rank_brand_words


def load_cached(derived: Path) -> tuple[list[dict], np.ndarray, dict]:
    index_path = derived / "all_index.json"
    emb_path = derived / "all_embeddings.npy"
    ocr_path = derived / "ocr_raw.json"

    missing = [p.name for p in (index_path, emb_path, ocr_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing cached artefacts: {', '.join(missing)}. "
            "Run tools/embed_all.py then tools/ocr_all.py first."
        )

    records = json.loads(index_path.read_text(encoding="utf-8"))
    embeddings = np.load(emb_path)
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    return records, embeddings, ocr


def split_store_images(store: StoreGroup) -> tuple[list[str], list[str]]:
    """Hold out one image as a query, but only when the store can spare it.

    A store with fewer than MIN_IMAGES_PER_STORE frames keeps everything in the
    gallery; stripping it would leave too little to match against.
    """
    images = list(store.images)
    if len(images) < config.MIN_IMAGES_PER_STORE:
        return images, []
    return images[:-config.QUERY_PER_STORE], images[-config.QUERY_PER_STORE:]


def build_dataset(derived: Path | None = None) -> dict:
    out = Path(derived or config.DERIVED_DIR)
    records, embeddings, ocr = load_cached(out)

    row_of = {r["rel_path"]: i for i, r in enumerate(records)}
    # Photographs are aligned straight onto directory units; segmentation and
    # naming are one step, so neighbouring shopfronts can no longer merge.
    stores, alignment = build_stores(records, embeddings, ocr)

    gallery_rows, gallery_index = [], []
    query_rows, query_index = [], []
    for store in stores:
        gal, qry = split_store_images(store)
        for rel in gal:
            gallery_rows.append(row_of[rel])
            gallery_index.append({"store_id": store.store_id, "image": rel})
        for rel in qry:
            query_rows.append(row_of[rel])
            query_index.append({"store_id": store.store_id, "image": rel})

    gallery_emb = embeddings[gallery_rows] if gallery_rows else np.zeros((0, embeddings.shape[1]), np.float32)
    query_emb = embeddings[query_rows] if query_rows else np.zeros((0, embeddings.shape[1]), np.float32)

    # Per-store signage text, used as the text side of the fusion scorer.
    ocr_by_store = {}
    for store in stores:
        gal, _ = split_store_images(store)
        words: list[str] = []
        for rel in gal:
            words.extend(ocr.get(rel, {}).get("text", "").split())
        ocr_by_store[store.store_id] = " ".join(dict.fromkeys(words))

    # Brand vocabulary for text matching, built from gallery frames only.
    # Deriving it from every frame would leak the held-out query into the
    # vocabulary it is later scored against, inflating text accuracy.
    brand_vocab = {}
    for store in stores:
        gal, _ = split_store_images(store)
        frames = [brand_tokens(ocr.get(rel, {})) for rel in gal]
        words = [w.upper() for w, _ in rank_brand_words(frames)[:6]]
        # Fold in the official brand name. This is external directory
        # knowledge, not evaluation data, and it repairs the clipped reads
        # that OCR produces from logo lettering.
        if store.name_source == "directory":
            words += [
                w for w in re.split(r"[^A-Za-z0-9]+", store.name.upper())
                if len(w) >= config.MIN_BRAND_TOKEN_LEN
            ]
        brand_vocab[store.store_id] = list(dict.fromkeys(words))

    np.save(out / "gallery_embeddings.npy", gallery_emb)
    np.save(out / "query_embeddings.npy", query_emb)
    _write_json(out / "stores.json", groups_to_json(stores))
    _write_json(out / "gallery_index.json", gallery_index)
    _write_json(out / "query_index.json", query_index)
    _write_json(out / "ocr_by_store.json", ocr_by_store)
    _write_json(out / "brand_vocab.json", brand_vocab)
    _write_json(out / "directory_alignment.json", alignment)

    report = _build_report(stores, gallery_emb, query_emb, records, ocr)
    _write_json(out / "report.json", report)
    (out / "report.md").write_text(_render_report(report), encoding="utf-8")
    return report


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_report(stores, gallery_emb, query_emb, records, ocr) -> dict:
    per_corridor: dict[str, int] = {}
    for s in stores:
        key = f"{s.floor_name} / {s.side}"
        per_corridor[key] = per_corridor.get(key, 0) + 1

    counts = [s.image_count for s in stores]
    review = [s for s in stores if s.needs_review]
    singles = [s for s in stores if s.image_count == 1]
    unnamed = [s for s in stores if s.name.startswith("UNKNOWN")]

    # Same brand name recovered on more than one corridor - either a genuine
    # second outlet or a segmentation error, and either way worth checking.
    by_name: dict[str, list[str]] = {}
    for s in stores:
        by_name.setdefault(s.name, []).append(s.store_id)
    duplicates = {n: ids for n, ids in by_name.items() if len(ids) > 1 and not n.startswith("UNKNOWN")}

    return {
        "images_ingested": len(records),
        "images_with_ocr_text": sum(1 for r in records if ocr.get(r["rel_path"], {}).get("text")),
        "stores_recovered": len(stores),
        "stores_per_corridor": per_corridor,
        "images_per_store": {
            "min": min(counts) if counts else 0,
            "median": int(np.median(counts)) if counts else 0,
            "mean": round(float(np.mean(counts)), 2) if counts else 0,
            "max": max(counts) if counts else 0,
        },
        "gallery_vectors": int(gallery_emb.shape[0]),
        "query_vectors": int(query_emb.shape[0]),
        "embedding_dim": int(gallery_emb.shape[1]) if gallery_emb.size else 0,
        "stores_needing_review": len(review),
        "stores_single_image": len(singles),
        "stores_unnamed": len(unnamed),
        "duplicate_names": duplicates,
        "review_sample": [
            {"store_id": s.store_id, "name": s.name, "images": s.image_count, "why": s.review_reason}
            for s in review[:25]
        ],
    }


def _render_report(r: dict) -> str:
    ips = r["images_per_store"]
    lines = [
        "# InVision dataset report",
        "",
        "Stores were reconstructed from unlabelled corridor walks; every count",
        "below is derived, not ground truth. Verify before quoting in the report.",
        "",
        f"- Images ingested: **{r['images_ingested']}**",
        f"- Images with readable signage: **{r['images_with_ocr_text']}**",
        f"- Stores recovered: **{r['stores_recovered']}**",
        f"- Images per store: min {ips['min']}, median {ips['median']}, mean {ips['mean']}, max {ips['max']}",
        f"- Gallery vectors: **{r['gallery_vectors']}** (dim {r['embedding_dim']})",
        f"- Held-out query vectors: **{r['query_vectors']}**",
        "",
        "## Stores per corridor",
        "",
    ]
    for corridor, count in sorted(r["stores_per_corridor"].items()):
        lines.append(f"- {corridor}: {count}")

    lines += [
        "",
        "## Quality flags",
        "",
        f"- Stores needing review: **{r['stores_needing_review']}**",
        f"- Stores with a single image: {r['stores_single_image']}",
        f"- Stores with no readable name: {r['stores_unnamed']}",
        f"- Brand names appearing on more than one corridor: {len(r['duplicate_names'])}",
        "",
        "## Review sample",
        "",
    ]
    if not r["review_sample"]:
        lines.append("- none")
    for item in r["review_sample"]:
        lines.append(f"- `{item['store_id']}` ({item['images']} img) - {item['why']}")
    return "\n".join(lines) + "\n"
