# InVision store dataset

Storefront photographs of a shopping mall, grouped by store, for
vision-based indoor localisation.

## Provenance

Captured as six corridor walks (three floors x two sides), one photo
burst per storefront. The source photographs carry **no store labels** -
only a folder (floor + side) and a capture timestamp. Store identity was
reconstructed by segmenting each walk into runs of consecutive frames
showing the same shopfront, then naming each run from its signage text.

Capture folders are named by ordinal where the mall's "1st" is the
ground floor, so folder ordinals map one level down.

## Contents

- Stores: **85**
- Images: **276** (217 gallery, 59 query)
- Corridors: **6**

| Floor | Side | Stores |
| --- | --- | --- |
| Ground | left | 11 |
| Ground | right | 14 |
| First | left | 15 |
| First | right | 17 |
| Second | left | 15 |
| Second | right | 13 |

## Known limitations

**Labels are inferred, not verified.** Store names come from OCR of the
signage, ranked by text height. Recurring failure modes:

- leading characters absorbed by the logo mark (`ESTSIDE` = Westside,
  `DASICS` = Asics)
- visually similar food-court stalls merged into one store
- generic facade words winning over the brand name

11 stores are flagged `needs_review` in `stores.csv`, but the
failures above are **not** flagged - they need visual checking.

**The query split understates difficulty.** Each held-out image was shot
seconds from its own gallery images at nearly the same angle and lighting,
so retrieval accuracy measured on it will be optimistic. A fair evaluation
needs freshly captured photographs taken on a different day.

Two upper floors were excluded at ingest for having too few images
(1-5 each) to support recognition or routing.

## Files

- `gallery/<store_id>/` - reference images the localiser matches against
- `query/<store_id>/` - held-out evaluation images
- `labels.csv` - one row per image with full store metadata
- `stores.csv` - one row per store, including review flags and alternate
  name candidates from OCR

Images are resized to a longest side of 1024px; originals are unmodified under `data/Images/`.
