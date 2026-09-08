"""What each stage of a lookup actually costs, on this machine's CPU.

The project claims a photograph in and directions out with nothing paid to an
API. That claim is only worth anything if the wait is short, so the wait is
measured rather than asserted.

Every figure is a median over several runs on real query photographs, taken
after the models are warm. Model load is reported separately, because it
happens once per process and the service pays it lazily on the first
photograph rather than at startup.

    python tools/timings.py            # 5 photographs, 3 runs each
    python tools/timings.py --images 10 --runs 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, encoders  # noqa: E402
from src.localizer import Localizer  # noqa: E402
from src.mall_graph import MallGraph  # noqa: E402


def timed(fn, runs: int) -> float:
    """Median wall-clock milliseconds over `runs` calls."""
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def query_images(limit: int) -> list[Path]:
    index_path = config.DERIVED_DIR / "query_index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    paths = []
    for record in index:
        candidate = config.IMAGES_DIR / record["image"]
        if candidate.is_file():
            paths.append(candidate)
        if len(paths) >= limit:
            break
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=5,
                        help="how many query photographs to time")
    parser.add_argument("--runs", type=int, default=3,
                        help="runs per photograph, median reported")
    args = parser.parse_args()

    rows: list[tuple[str, float, str]] = []

    start = time.perf_counter()
    graph = MallGraph.from_derived()
    rows.append(("Build the graph", (time.perf_counter() - start) * 1000,
                 f"{graph.graph.number_of_nodes()} nodes, once at startup"))

    start = time.perf_counter()
    localizer = Localizer(graph=graph)
    rows.append(("Load the gallery", (time.perf_counter() - start) * 1000,
                 "embeddings and the directory, once at startup"))

    images = query_images(args.images)
    if not images:
        print("No query photographs on this machine, so the per-photo stages "
              "cannot be timed.\nThe capture is not distributed with the "
              "repository; see the dataset card.", file=sys.stderr)
        return 1

    # Both models load lazily on first use. That first call is the cold start a
    # shopper pays once, and it is reported on its own rather than smeared
    # across the per-photograph figures.
    start = time.perf_counter()
    localizer.embed_query(images[0])
    rows.append(("Load CLIP (first photo only)", (time.perf_counter() - start) * 1000,
                 config.CLIP_MODEL))

    start = time.perf_counter()
    encoders.ocr_image(images[0])
    rows.append(("Load RapidOCR (first photo only)", (time.perf_counter() - start) * 1000,
                 "detection and recognition models"))

    def median_over_images(fn) -> float:
        return statistics.median(timed(lambda: fn(path), args.runs) for path in images)

    rows.append(("Embed one photograph", median_over_images(localizer.embed_query),
                 "CLIP forward pass, warm"))
    rows.append(("Read the signage", median_over_images(encoders.ocr_image),
                 "RapidOCR, warm"))

    vectors = [localizer.embed_query(path) for path in images]
    rows.append(("Score every unit visually",
                 statistics.median(timed(lambda v=v: localizer.visual_scores(v), args.runs)
                                   for v in vectors),
                 f"{len(graph.units)} units against the gallery"))

    # text_scores runs OCR itself when handed a path, so the reading is passed
    # in: this row is the matching, not the matching plus the OCR above it.
    readings = [encoders.ocr_image(path) for path in images]
    rows.append(("Score every unit on text",
                 statistics.median(
                     timed(lambda entry=entry: localizer.text_scores(ocr_entry=entry),
                           args.runs)
                     for entry in readings),
                 "fuzzy name matching over OCR tokens already read"))
    rows.append(("Locate, end to end", median_over_images(localizer.locate),
                 "embed, OCR, score both channels, fuse"))

    units = list(graph.units)
    origin, destination = units[0], units[-1]
    rows.append(("Route between two units",
                 timed(lambda: graph.route(origin, destination), args.runs * 10),
                 "Dijkstra plus the written directions"))
    rows.append(("Nearby, 45 m",
                 timed(lambda: graph.nearby(origin), args.runs * 10),
                 "one capped sweep"))
    stops = units[1:5]
    rows.append(("Order a four stop trip",
                 timed(lambda: graph.tour(origin, stops), args.runs * 10),
                 "Held-Karp, exact"))

    print(f"\nMedian of {args.runs} runs over {len(images)} query photographs.\n")
    print("| Stage | Median | Notes |")
    print("|---|---|---|")
    for name, ms, note in rows:
        if ms >= 1000:
            figure = f"{ms / 1000:.1f} s"
        elif ms >= 10:
            figure = f"{ms:.0f} ms"
        else:
            # Rounding the graph work to a whole millisecond reports it as free,
            # which is a stronger claim than the measurement supports.
            figure = f"{ms:.2f} ms"
        print(f"| {name} | {figure} | {note} |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
