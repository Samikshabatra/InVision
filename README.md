# InVision — Indoor Navigation for Nexus Mall, Koramangala

A vision-based indoor navigation system for **Nexus Mall, Koramangala (Bengaluru)**.
Point your phone at any storefront and InVision recognises where you are standing,
then routes you to any other store in the mall.

The recognition model was trained on our own data: corridor walks captured across
the mall's floors, with **no store labels** — store identity was reconstructed
directly from the photographs and the official floor directory.

## What it does

1. **Recognise** — a storefront photograph is matched to a unit on the floor
   directory by fusing two signals: CLIP visual similarity to reference images,
   and agreement between OCR'd signage text and the official brand name.
2. **Route** — every unit on every floor map becomes a node in a navigation
   graph (walkways, escalators, gates), so InVision can give turn-by-turn
   directions between any two stores — even ones no one photographed.

## How it was built

The pipeline runs bottom-up, and this repository's commit history follows the
same order:

| Stage | What happens |
| --- | --- |
| **Ingestion** | Read the flat corridor-walk folders into ordered image records (floor + side + capture time). |
| **Encoders** | CLIP image embeddings (`clip-vit-base-patch32`) + RapidOCR signage text, cached once. |
| **Store recovery** | Segment each walk into per-store bursts using shared signage text, with embedding similarity as a fallback. |
| **Directory alignment** | Align recovered stores to the official floor directory with Needleman–Wunsch sequence alignment, recovering true brand names and order. |
| **Dataset build** | Hold out a query split, write gallery/query embeddings, brand vocabulary, and a sanity report. |
| **Localizer** | Fuse visual + text evidence to name a query photograph. |
| **Navigation graph** | Build the routable mall graph from the directory maps. |
| **Evaluation** | Top-k retrieval accuracy per signal and fused, with an ablation. |
| **Web + API** | FastAPI service wrapping the localizer and router, with a map interface. |

## Dataset at a glance

- **289** storefront photographs across **6 corridors** (3 floors × 2 sides)
- **83** stores recovered from unlabelled walks, named via OCR + directory
- **234** gallery vectors / **55** held-out query vectors (512-dim CLIP)

> Labels are *inferred*, not hand-verified. See
> [`data/dataset/DATASET_CARD.md`](data/dataset/DATASET_CARD.md) for provenance
> and known limitations.

## Quickstart

```bash
pip install -r requirements.txt   # torch, transformers, rapidocr-onnxruntime, fastapi, uvicorn, numpy, pillow

# Locate a single storefront photo
python predict.py path/to/photo.jpg --top-k 3

# Route between two stores
python route.py --from-store SWAROVSKI --to DECATHLON

# Run the web app + API
python api.py            # then open http://localhost:8000
```

Rebuilding the derived artefacts from scratch requires the raw capture set
(`data/Images/`, ~730 MB), which is **not tracked in git** — available on
request. The curated labelled dataset (`data/dataset/`) and the derived
embeddings/indexes needed to run the localizer *are* included.

```bash
# Full rebuild (needs the raw images):
python tools/extract_zips.py
python tools/embed_all.py
python tools/ocr_all.py
python build_dataset.py
```

## Repository layout

```
src/            core pipeline (ingest → encoders → grouping → alignment → localizer → graph)
tools/          one-off scripts: embedding/OCR caching, calibration, dataset export, reports
data/dataset/   curated, labelled image dataset (gallery + query splits)
data/derived/   cached embeddings, indexes, and the sanity report
web/            map interface served by the API
predict.py      locate one storefront photo
route.py        route between two stores
evaluate.py     held-out retrieval evaluation + ablation
api.py          FastAPI web service
```

## Documentation

- [`InVision_Project_Report.pdf`](InVision_Project_Report.pdf) — full design writeup
- [`InVision_Team_Guide.pdf`](InVision_Team_Guide.pdf) — task division and walkthrough
