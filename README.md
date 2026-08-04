<div align="center">

# InVision

**Indoor mall navigation from a photograph of a shopfront.**

![Python](https://img.shields.io/badge/python-3.11+-7F8C99?style=flat-square)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-4C6EF5?style=flat-square)
![OCR](https://img.shields.io/badge/OCR-RapidOCR-1D9E75?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139-0E7C6B?style=flat-square)
![Units](https://img.shields.io/badge/directory-104%20units%20%C2%B7%205%20floors-8A8F98?style=flat-square)
![Top-1](https://img.shields.io/badge/top--1-67.3%25-3FA45B?style=flat-square)

</div>

---

Indoor positioning is the awkward gap in navigation. GPS stops at the door, and
the alternatives all want infrastructure: beacons bolted to ceilings, wifi
fingerprints resurveyed every time a shop refits, a floor plan nobody has in a
machine-readable form.

A shopping mall already carries a dense, maintained, human-readable position
signal on every wall. It is called signage. InVision reads it: photograph any
shopfront, and it works out which unit of the directory you are standing at, then
routes you from there to wherever you want to go, across floors and escalators.

```
photo -> CLIP embedding ─┐
                         ├─> fused score -> unit id -> graph route -> directions
photo -> OCR signage ────┘
```

**Two signals, deliberately.** Visual similarity alone confuses the many
storefronts that are glass, white, and lit the same way. Signage text alone fails
whenever a logo mark eats the first letter or a survey shot frames two shops at
once. Fusing them at 0.7 visual to 0.3 text beats either signal alone by nine
points, and legible signage is allowed to override the visual channel outright.

## What it does

- **Locates a shopper from one photograph.** A CLIP embedding is matched against
  a gallery of storefront images, OCR tokens from the signage are matched against
  the directory's brand names with fuzzy word matching, and the two scores are
  fused per unit.
- **Says when it is unsure, rather than guessing confidently.** A result is
  reported as confident when signage decided it outright, or when the fused
  margin over the runner-up is at least 0.03. Anything closer is returned as a
  close call with the alternatives offered.
- **Routes across the whole building.** The floor directory becomes a graph of
  268 nodes: shopfront units, walkway points, gates and escalator banks on every
  level. Dijkstra over walking-equivalent metres produces turn-by-turn directions
  with landmarks, floor changes, and an estimated distance.
- **Draws the plan.** Each floor on the route is rendered as an SVG of the real
  directory layout, with the walk drawn on it and pins at both ends.
- **Walks a shopping list in the cheapest order.** Give it up to eight stops and
  it solves the visiting order exactly with Held-Karp, which on a two-row floor
  plan regularly beats nearest-next by a comfortable margin.
- **Checks progress mid-walk.** Shoot a second shopfront part way along and it
  judges that sighting against the planned line: on route with the distance
  remaining, or off route with a fresh plan from wherever you actually are.
- **Remembers where you are.** Position carries between routes, so walking to one
  shop and then thinking of another does not need a second photograph. Every
  position records how it was learned: recognised, declared, assumed after a
  route, or carried over from earlier.
- **Reads the directions aloud**, because a shopper walking a corridor is not
  looking at a screen.

## Localisation, in detail

`src/localizer.py` scores every directory unit and fuses two channels.

**Visual.** The query image is embedded with `openai/clip-vit-base-patch32`. Each
unit's score is the best cosine similarity across its gallery vectors, so a unit
photographed once is not penalised against one photographed fourteen times.

**Text.** RapidOCR returns tokens with bounding boxes. Tokens are ranked by text
height, on the reasoning that the tallest lettering on a shopfront is the brand
name. Each token is matched against the unit's name and its aliases, tolerant of
the character clipping OCR introduces: a substring match on words of four letters
or more, or a sequence ratio of at least 0.82.

**Fusion and override.** The default weighting is 0.7 visual to 0.3 text. Signage
scoring above 0.42 decides the answer on its own, because a legible brand name is
stronger evidence than any amount of glass-and-white similarity. That override is
withheld when a second unit is within 0.02 of it, which is the survey shot that
frames two shopfronts at once, and the result is reported as a close call.

Every threshold in that description is a named constant at the top of the module,
not a number buried in an expression.

## Results

Measured on the held-out query split, 55 images over 83 recognisable units, with
all 104 directory units as candidates:

| Configuration | top-1 | top-3 | top-5 | correct floor |
|---|---|---|---|---|
| Visual only (CLIP) | 58.2% | 74.5% | 80.0% | 83.6% |
| Text only (OCR) | 58.2% | 63.6% | 65.5% | 74.5% |
| Fusion 0.5 / 0.5 | 60.0% | 72.7% | 76.4% | 78.2% |
| **Fusion 0.7 / 0.3** | **67.3%** | **80.0%** | **87.3%** | **87.3%** |
| Fusion 0.85 / 0.15 | 65.5% | 80.0% | 83.6% | 89.1% |

```bash
python evaluate.py
```

**These numbers are optimistic and the evaluation says so before it prints
them.** Every held-out image was captured seconds from its own gallery images, at
nearly the same angle and under the same lighting. A fair figure needs
photographs taken on a different day, ideally in different light and with the
crowd that a real shopper photographs through. Treat the ablation as the finding
here, not the absolute accuracy: the text channel is worth nine points of top-1
and seven points of correct-floor, and it is worth most where the visual channel
is weakest.

## The dataset, and why it is not published

**The photographs in this project were captured with the written permission of
mall management, and that permission is why the imagery is not distributed.**

Photographing shopfronts inside a private building is not a thing you may simply
do. Before any capture, a letter was written to the mall's management office
setting out what the project was, what would be photographed, that no shoppers or
staff were the subject, and how the images would be used. Permission was granted
on that basis, for that use.

That constrains what this repository can contain, for reasons worth stating
plainly:

- **The images are of a real, identifiable, occupied building.** A complete
  photographic index of every shopfront on every floor, aligned to a floor plan
  and searchable, is exactly the artefact a building's security team would prefer
  did not exist publicly. The routing graph makes it more useful still, which is
  the point of the project and also the problem.
- **Incidental capture is unavoidable.** Corridor walks catch reflections,
  shoppers in the background, staff at counters, and occasionally a till or a
  screen. Nobody in those frames consented to publication, and none of them were
  the subject of the shot.
- **The permission was for a project, not for redistribution.** Publishing the
  set would exceed what was asked for and granted, whatever the licence on the
  code says.

So the repository ships the derived artefacts and not the source photographs:

| In the repository | Not in the repository |
|---|---|
| CLIP embeddings (`data/derived/*.npy`) | The corridor walk, 732 MB of raw frames |
| Gallery and query indexes, OCR output | The curated gallery and query images |
| The floor directory and the routing graph | Any frame containing a person |
| Dataset card, labels and store metadata | |

The embeddings are 512-dimensional float vectors. They are enough to run and
evaluate the localiser, which is why everything in this repository still works
without a single photograph present, and they are not enough to reconstruct the
photographs they came from.

**If you want to run this on your own building, capture your own set, and ask
first.** `tools/` contains the whole pipeline that turns a folder of corridor
walks into the derived artefacts, and `data/dataset/DATASET_CARD.md` documents
the format it expects.

## How the dataset was reconstructed

The source photographs carry no labels. Six corridor walks, three floors by two
sides, one burst per storefront, with nothing but a folder name and a capture
timestamp. Store identity was recovered rather than recorded:

1. **Segment each walk** into runs of consecutive frames showing the same
   shopfront, using CLIP similarity between neighbours.
2. **Name each run** from the tallest signage text its frames contain.
3. **Align the recovered stores to the printed floor directory**, which is the
   authority on what exists and where. A unit with photographs can be recognised;
   a unit without can still be routed to.
4. **Flag what needs a human.** Stores with a single image, no legible name, or a
   brand appearing on more than one corridor are marked for review, and
   `/review` is a small interface for stepping through the frames in walk order
   and correcting the assignment.

The failure modes are documented in the dataset card rather than smoothed over:
leading characters absorbed by a logo mark (`ESTSIDE` for Westside, `DASICS` for
Asics), visually similar food-court stalls merged, and generic facade words
beating the brand name.

## Routing

`src/mall_graph.py` builds the graph from the directory, not from the
photographs. Every floor is a rectangular walkway around a central atrium:

```
front gate (west)                                back gate (east)
    |                                                    |
    |   +--------------- TOP ROW ---------------------+  |
    +---|                                             |--+
    |   |     [esc west]    (atrium)    [esc east]    |  |
    +---|                                             |--+
        +------------- BOTTOM ROW --------------------+
```

Shoppers cross between rows at four points: the gates at each end and the two
escalator banks either side of the atrium. Both banks serve every level, so the
router picks whichever is nearer. Distances are modelled from an 8 m average
storefront frontage, a 34 m corridor width, and a 25 m walking-equivalent cost
per escalator level, which makes ordering and direction reliable while metre
figures stay estimates.

**Multi-stop trips** are solved exactly. Held-Karp over the stop set costs
2^n·n², which at the eight-stop ceiling runs in about 6 ms, against 45 seconds
for brute-force permutations of the same eight stops. It is exact rather than
greedy because nearest-next gets a two-row floor plan wrong often enough to
matter: a four-stop list in testing came out 266 m ordered against 339 m as
written, a saving of 73 m.

**Progress checks** compare a mid-walk sighting against the planned line. One
Dijkstra sweep from the sighting answers both how far the destination still is
and how far the plan is, giving three states: arrived within 6 m, on route within
12 m of the line and no further from the goal, or off route with a fresh plan.
The plan is recomputed server side from the two endpoints rather than carried by
the client, so a stale payload cannot move the line being measured against.

## Project layout

```
CV_Project/
├── api.py                 # the web service: locate, route, tour, progress, review
├── predict.py             # locate one photograph from the command line
├── route.py               # route between two stores, by name or by photograph
├── evaluate.py            # the ablation table above
├── build_dataset.py       # reconstruct stores from the corridor walks
├── src/
│   ├── localizer.py       # the fusion: CLIP + OCR -> ranked units
│   ├── encoders.py        # CLIP and RapidOCR, both loaded lazily
│   ├── mall_graph.py      # the directory as a graph; routing, tours, progress
│   ├── directory.py       # the floor directory and its aliases
│   ├── pipeline.py        # walk segmentation, store recovery, the report
│   ├── grouping.py        # brand tokens, run detection
│   ├── frame_align.py     # aligning recovered stores to directory units
│   ├── ingest.py          # reading the capture folders
│   ├── build_stores.py    # the exported dataset
│   └── config.py          # every path and threshold in one place
├── tools/                 # one-shot pipeline stages, cached
│   ├── extract_zips.py    ├── embed_all.py      ├── ocr_all.py
│   ├── apply_directory.py ├── find_brands.py    ├── export_dataset.py
│   ├── calibrate_threshold.py                   └── probe_data.py
├── web/
│   ├── index.html         # the interface: one page, no build step, no framework
│   ├── review.html        # the label review pass
│   └── atrium.png
└── data/
    ├── derived/           # embeddings, indexes, OCR output, the report
    ├── dataset/           # the dataset card, labels and store metadata
    └── mall_directory.json
```

## Running it

You need Python 3.11 or newer.

```bash
pip install -r requirements.txt
# torch is a large download; the CPU wheel is enough:
pip install torch --index-url https://download.pytorch.org/whl/cpu

python api.py
```

Open <http://127.0.0.1:8000>. Drop a photograph of a shopfront onto the sighting
panel, or paste one from the clipboard. The header lights up with the unit it
recognised and how confident it was. Choose a destination, or add several to the
list, and the plan and directions appear.

`/review` is the label review interface: every captured frame in walk order, with
its current unit assignment and the OCR tokens that produced it.

**The command line does the same things** without the browser:

```bash
python predict.py shot.jpg --top-k 3
python predict.py shot.jpg --no-text          # visual channel only
python route.py --from-store SWAROVSKI --to DECATHLON
python route.py --from-photo shot.jpg --to DECATHLON
```

**Rebuilding the dataset** from a folder of corridor walks, which needs the
photographs and so is only useful on your own capture:

```bash
python tools/extract_zips.py     # unpack the walks
python tools/embed_all.py        # CLIP over every frame, cached
python tools/ocr_all.py          # RapidOCR over every frame, cached
python build_dataset.py          # segment, recover stores, write the report
```

CLIP and RapidOCR are both loaded lazily, on first use, so the service starts in
about a second and only pays for the model when a photograph actually arrives.

## The HTTP contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The interface. |
| `GET` | `/api/stores` | Every routable unit on the directory. |
| `GET` | `/api/layout` | Floor geometry for the plan drawing. |
| `POST` | `/api/locate` | A photograph in, ranked units out, with a confidence verdict. |
| `POST` | `/api/route` | Origin and destination in, steps and map legs out. |
| `POST` | `/api/tour` | An origin and up to eight stops, ordered and routed. |
| `POST` | `/api/progress` | A mid-walk photograph judged against the plan. |
| `GET` | `/review` | The label review interface. |
| `GET` | `/api/review` | Every frame in walk order with its assignment. |
| `POST` | `/api/review` | Save corrected assignments. |
| `GET` | `/api/thumb` | A cached thumbnail of a captured frame, path-guarded. |

`/api/locate` returns the ranked predictions plus `confident`, `margin` and
`decided_by`, which is `signage` when the text override fired and `fusion`
otherwise. The interface uses `confident` to decide whether to offer the
alternatives.

## Limitations

Found by using it, not imagined while designing it.

- **The accuracy figure is optimistic** for the reason the evaluation prints, and
  a same-day query split is the single biggest weakness in the measurement.
- **There is no open-set rejection.** A photograph of the floor, a ceiling, or a
  person still returns the closest storefront, sometimes with a respectable
  score. The system cannot currently say "that is not a shopfront".
- **Distances are modelled, not measured.** Storefront frontages are assumed
  equal at 8 m. Ordering, direction and floor changes are reliable; the metre
  figures are estimates and are presented as "about".
- **21 of 104 directory units have no photographs**, mostly on the two upper
  floors that were excluded at ingest for having too few frames. They can be
  routed to but never recognised, which is the right asymmetry but still a gap.
- **Store labels are inferred, not verified.** OCR named every store, and the
  known failure modes in the dataset card are not all flagged in the data.
- **Nothing survives a refit.** A shop that changes its signage stops being
  recognisable until its gallery is recaptured, and there is no mechanism for
  noticing that has happened.
- **The progress check trusts one frame.** A single mid-walk photograph that is
  misread will reroute the shopper confidently from the wrong place.
- **Position memory assumes the walk was made.** Asking for a new destination
  while a route is on screen moves the position to that route's far end. It says
  so and offers one click back, but it is an assumption, not evidence.

## Things I would add next

- **Sequence localisation.** Two or three frames while walking, filtered through
  the adjacency graph, so physically impossible jumps are ruled out. This is the
  single change that would turn a per-image classifier into a positioning system.
- **Geometric verification.** SIFT or LoFTR keypoint matching with RANSAC to
  re-rank CLIP's top five, which is the standard visual place recognition move
  this pipeline is missing.
- **Open-set rejection**, so a photograph of a ceiling is refused rather than
  matched.
- **Landmark thumbnails in the directions.** The steps already name what you
  walk past; showing the gallery image of the next landmark would let a shopper
  confirm they are in the right corridor.
- **A robustness benchmark**: the query set corrupted with motion blur, glare,
  low light and occlusion, reported per corruption level. Cheap to generate, and
  much closer to what a phone in a mall actually produces.
- **Amenities in the graph.** Washrooms, lifts, exits and parking, which is
  realistically the most common thing anyone wants to find, and step-free routing
  for anyone who cannot use an escalator.

## Cost

Zero. CLIP and RapidOCR run locally on CPU, the graph is NetworkX, the service is
FastAPI, the interface is one HTML file with no build step and no framework, and
nothing calls a paid API. The only expensive thing about this project was the
walking.
