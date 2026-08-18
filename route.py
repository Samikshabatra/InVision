"""Route between two stores, by name or by photograph.

Usage:
    python route.py --from-store SWAROVSKI --to DECATHLON
    python route.py --from-photo shot.jpg --to DECATHLON
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.mall_graph import MallGraph


def resolve(graph: MallGraph, query: str) -> str | None:
    matches = graph.find_stores(query)
    if not matches:
        print(f"no store matching '{query}'", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(f"'{query}' matches {len(matches)} stores:", file=sys.stderr)
        for s in matches[:8]:
            print(f"  {s['unit_id']:<26} {s['name']} "
                  f"({s['floor_name']} floor, {s['row']} row)", file=sys.stderr)
        print("re-run with a full unit id to disambiguate", file=sys.stderr)
        return matches[0]["unit_id"]
    return matches[0]["unit_id"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Route between mall stores.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-store", help="origin store name or id")
    source.add_argument("--from-photo", type=Path, help="photo of where you are standing")
    parser.add_argument("--to", required=True, help="destination store name or id")
    args = parser.parse_args()

    graph = MallGraph.from_derived()

    if args.from_photo:
        from src.localizer import Localizer

        if not args.from_photo.exists():
            print(f"no such image: {args.from_photo}", file=sys.stderr)
            return 1
        predictions = Localizer(graph=graph).locate(args.from_photo, top_k=2)
        if not predictions:
            print("could not recognise that photo", file=sys.stderr)
            return 1
        origin_id = predictions[0].unit_id
        print(f"recognised: {predictions[0].name} "
              f"(score {predictions[0].score:.3f}, by {predictions[0].decided_by})")
        if len(predictions) > 1 and predictions[0].score - predictions[1].score < 0.03:
            print(f"  uncertain - could also be {predictions[1].name}")
    else:
        origin_id = resolve(graph, args.from_store)
        if origin_id is None:
            return 1

    destination_id = resolve(graph, args.to)
    if destination_id is None:
        return 1

    route = graph.route(origin_id, destination_id)

    print()
    for i, step in enumerate(route.steps, 1):
        print(f"  {i}. {step.text}")
    levels = max(len(route.floors_traversed) - 1, 0)
    print(f"\n  total: ~{route.total_distance_m:.0f}m, "
          f"about {max(1, round(route.minutes))} min on foot", end="")
    if levels:
        print(f", crossing {levels} floor level{'s' if levels != 1 else ''}")
    else:
        print()

    for caveat in route.caveats:
        print(f"  note: {caveat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
