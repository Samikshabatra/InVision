"""The navigation graph - built from the official floor directory.

The directory maps, not the photographs, define the mall. Every unit listed on
a floor map becomes a node here, whether or not anyone photographed it, so the
app can route to Nike or the IMAX ticket counter even though no reference image
of either exists. Photographs only add the ability to *recognise* where a
shopper is standing; they never define where things are.

Each floor is a rectangular walkway around a central atrium:

    front gate (west)                                back gate (east)
        |                                                    |
        |   +--------------- TOP ROW ---------------------+  |
        +---|                                             |--+
        |   |     [esc west]    (atrium)    [esc east]    |  |
        +---|                                             |--+
            +------------- BOTTOM ROW --------------------+

Shoppers cross between rows at four points: the gates at each end and the two
escalator banks either side of the atrium. Both banks serve every floor, so the
router picks whichever is nearer.

Distances are modelled from average storefront width rather than measured, so
ordering and direction are reliable while metre figures are estimates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from . import config
from .directory import load_directory

STORE_SPACING_M = 8.0     # average storefront frontage
STORE_STUB_M = 2.0        # walkway centreline to shop entrance
CORRIDOR_WIDTH_M = 34.0   # across the atrium, top row to bottom row
ESCALATOR_COST_M = 25.0   # walking-equivalent cost of one level

GATE_WEST_X, GATE_EAST_X = 0.0, 1.0
ESC_WEST_X, ESC_EAST_X = 0.28, 0.72

CROSS_POINTS = {
    "front gate": GATE_WEST_X,
    "west escalator": ESC_WEST_X,
    "east escalator": ESC_EAST_X,
    "back gate": GATE_EAST_X,
}
ESCALATOR_BANKS = ("west escalator", "east escalator")
ROWS = ("top", "bottom")


@dataclass
class Step:
    kind: str  # depart | walk | cross | escalator | arrive
    text: str
    floor: int | None = None
    distance_m: float = 0.0


@dataclass
class Route:
    steps: list[Step]
    total_distance_m: float
    floors_traversed: list[int]
    caveats: list[str]
    map_legs: list[dict] = field(default_factory=list)


def unit_node(unit_id: str) -> str:
    return f"unit:{unit_id}"


def walk_node(floor: int, row: str, x: float) -> str:
    return f"walk:{floor}:{row}:{x:.4f}"


def cross_node(floor: int, name: str) -> str:
    return f"cross:{floor}:{name}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unit"


class MallGraph:
    def __init__(self, directory: dict, stores: list[dict] | None = None):
        self.directory = directory
        self.units: dict[str, dict] = {}
        self.store_to_unit: dict[str, str] = {}
        self._floor_length: dict[int, float] = {}
        self.graph = nx.Graph()

        self._build_units()
        if stores:
            self._attach_stores(stores)
        self._build_graph()

    @classmethod
    def from_derived(cls, derived: Path | None = None) -> "MallGraph":
        derived = Path(derived or config.DERIVED_DIR)
        stores_path = derived / "stores.json"
        stores = json.loads(stores_path.read_text(encoding="utf-8")) if stores_path.exists() else []
        return cls(load_directory(), stores)

    # ---------- units ----------

    def _build_units(self) -> None:
        for entry in self.directory["floors"]:
            floor = entry["floor"]
            rows = entry["rows"]
            longest = max((len(v) for v in rows.values()), default=2)
            self._floor_length[floor] = max(longest, 2) * STORE_SPACING_M

            for row in ROWS:
                names = rows.get(row, [])
                for index, name in enumerate(names):
                    base = f"{_slug(name)}-f{floor}-{row[0]}"
                    unit_id = base
                    n = 2
                    while unit_id in self.units:
                        unit_id = f"{base}-{n}"
                        n += 1
                    self.units[unit_id] = {
                        "unit_id": unit_id,
                        "name": name,
                        "floor": floor,
                        "floor_name": entry["name"],
                        "row": row,
                        "index": index,
                        "row_length": len(names),
                        "x": (index + 0.5) / len(names),
                        "y": 0.0 if row == "top" else 1.0,
                        "image_count": 0,
                        "store_ids": [],
                    }

    def _attach_stores(self, stores: list[dict]) -> None:
        """Link recovered photo groups to the directory units they matched.

        A unit with photographs can be recognised from a snapshot; one without
        can still be navigated to, it just cannot be the starting point.
        """
        by_position = {
            (u["floor"], u["row"], u["index"]): u for u in self.units.values()
        }
        for store in stores:
            index = store.get("map_index")
            row = store.get("row")
            if index is None or index < 0 or row not in ROWS:
                continue
            unit = by_position.get((store["floor"], row, index))
            if unit is None:
                continue
            unit["store_ids"].append(store["store_id"])
            unit["image_count"] += len(store.get("images", []))
            self.store_to_unit[store["store_id"]] = unit["unit_id"]

    def unit_for_store(self, store_id: str) -> str | None:
        return self.store_to_unit.get(store_id)

    # ---------- graph ----------

    def _build_graph(self) -> None:
        floors = sorted({u["floor"] for u in self.units.values()})

        for floor in floors:
            for name, x in CROSS_POINTS.items():
                self.graph.add_node(
                    cross_node(floor, name),
                    kind="escalator" if name in ESCALATOR_BANKS else "gate",
                    floor=floor, x=x, y=0.5, label=name,
                )

            for row in ROWS:
                in_row = [u for u in self.units.values()
                          if u["floor"] == floor and u["row"] == row]
                xs = sorted({u["x"] for u in in_row} | set(CROSS_POINTS.values()))

                for x in xs:
                    self.graph.add_node(
                        walk_node(floor, row, x), kind="walk",
                        floor=floor, row=row, x=x, y=0.0 if row == "top" else 1.0,
                    )
                for a, b in zip(xs, xs[1:]):
                    self.graph.add_edge(
                        walk_node(floor, row, a), walk_node(floor, row, b),
                        weight=(b - a) * self._floor_length[floor],
                    )
                for name, x in CROSS_POINTS.items():
                    self.graph.add_edge(
                        walk_node(floor, row, x), cross_node(floor, name),
                        weight=CORRIDOR_WIDTH_M / 2,
                    )
                for unit in in_row:
                    self.graph.add_node(unit_node(unit["unit_id"]), kind="unit", **unit)
                    self.graph.add_edge(
                        unit_node(unit["unit_id"]),
                        walk_node(floor, row, unit["x"]),
                        weight=STORE_STUB_M,
                    )

        for lower, upper in zip(floors, floors[1:]):
            for bank in ESCALATOR_BANKS:
                self.graph.add_edge(
                    cross_node(lower, bank), cross_node(upper, bank),
                    weight=ESCALATOR_COST_M,
                )

    # ---------- routing ----------

    def resolve(self, identifier: str) -> str | None:
        """Accept a unit id, a recovered store id, or a store name."""
        if identifier in self.units:
            return identifier
        if identifier in self.store_to_unit:
            return self.store_to_unit[identifier]
        matches = self.find_units(identifier)
        return matches[0]["unit_id"] if matches else None

    def route(self, origin: str, destination: str) -> Route:
        a, b = self.resolve(origin), self.resolve(destination)
        if a is None:
            raise KeyError(f"unknown origin: {origin}")
        if b is None:
            raise KeyError(f"unknown destination: {destination}")
        if a == b:
            return Route([Step("arrive", "You are already there.")], 0.0, [], [])

        path = nx.shortest_path(self.graph, unit_node(a), unit_node(b), weight="weight")
        distance = nx.shortest_path_length(
            self.graph, unit_node(a), unit_node(b), weight="weight"
        )
        return self._describe(path, distance, a, b)

    def _describe(self, path, distance, origin_id, destination_id) -> Route:
        origin, destination = self.units[origin_id], self.units[destination_id]
        nodes = [self.graph.nodes[n] for n in path]

        steps = [Step(
            "depart",
            f"You are at {origin['name']}, on the {origin['floor_name'].lower()} floor.",
            floor=origin["floor"],
        )]

        run_start = run_floor = run_row = None
        consumed: set[int] = set()

        def flush(end_x):
            nonlocal run_start
            if run_start is None or end_x is None:
                return
            delta = end_x - run_start
            metres = abs(delta) * self._floor_length.get(run_floor, 100.0)
            if metres >= 3:
                heading = "the back gate" if delta > 0 else "the front gate"
                marks = self._landmarks(run_floor, run_row, run_start, end_x)
                extra = f" Past {', '.join(marks)}." if marks else ""
                steps.append(Step(
                    "walk", f"Walk toward {heading} for about {metres:.0f}m.{extra}",
                    floor=run_floor, distance_m=metres,
                ))
            run_start = None

        for i, data in enumerate(nodes):
            kind = data.get("kind")

            if kind in ("walk", "unit"):
                x, row, floor = data.get("x"), data.get("row"), data.get("floor")
                if run_start is None or floor != run_floor or row != run_row:
                    flush(None)
                    run_start, run_floor, run_row = x, floor, row
                nxt = nodes[i + 1] if i + 1 < len(nodes) else None
                if nxt is None or nxt.get("kind") not in ("walk", "unit") \
                        or nxt.get("row") != row or nxt.get("floor") != floor:
                    flush(x)
                    run_start = run_floor = run_row = None

            elif kind in ("gate", "escalator"):
                if i in consumed:
                    continue
                label = data["label"]
                end = i
                while (end + 1 < len(nodes)
                       and nodes[end + 1].get("kind") == "escalator"
                       and nodes[end + 1].get("floor") != nodes[end].get("floor")):
                    end += 1
                    consumed.add(end)

                if end > i:
                    levels = nodes[end]["floor"] - data["floor"]
                    direction = "up" if levels > 0 else "down"
                    name = config.FLOOR_DISPLAY_NAMES.get(
                        nodes[end]["floor"], str(nodes[end]["floor"]))
                    plural = "" if abs(levels) == 1 else f" {abs(levels)} levels"
                    steps.append(Step(
                        "escalator",
                        f"Take the {label} {direction}{plural} to the {name.lower()} floor.",
                        floor=nodes[end]["floor"],
                        distance_m=ESCALATOR_COST_M * abs(levels),
                    ))
                else:
                    before = nodes[i - 1] if i > 0 else None
                    after = nodes[i + 1] if i + 1 < len(nodes) else None
                    if before and after and before.get("row") and after.get("row") \
                            and before["row"] != after["row"]:
                        steps.append(Step(
                            "cross", f"Cross to the other side at the {label}.",
                            floor=data.get("floor"), distance_m=CORRIDOR_WIDTH_M,
                        ))

        steps.append(Step(
            "arrive", f"You have arrived at {destination['name']}.",
            floor=destination["floor"],
        ))

        caveats: list[str] = []

        floors_traversed = list(dict.fromkeys(
            d["floor"] for d in nodes if d.get("floor") is not None
        ))
        return Route(steps, round(distance, 1), floors_traversed, caveats,
                     self._map_legs(nodes))

    def _map_legs(self, nodes: list[dict]) -> list[dict]:
        legs: list[dict] = []
        for data in nodes:
            floor, x, y = data.get("floor"), data.get("x"), data.get("y")
            if floor is None or x is None or y is None:
                continue
            point = [round(float(x), 4), round(float(y), 4)]
            if legs and legs[-1]["floor"] == floor:
                if legs[-1]["path"][-1] != point:
                    legs[-1]["path"].append(point)
            else:
                legs.append({"floor": floor, "path": [point]})
        return [leg for leg in legs if len(leg["path"]) > 1]

    def _landmarks(self, floor, row, a, b, limit: int = 3) -> list[str]:
        low, high = sorted((a, b))
        between = [
            (u["x"], u["name"]) for u in self.units.values()
            if u["floor"] == floor and u["row"] == row and low < u["x"] < high
        ]
        between.sort(reverse=b < a)
        return [name for _, name in between[:limit]]

    # ---------- presentation ----------

    def layout(self) -> dict:
        out: dict[str, dict] = {}
        for unit in self.units.values():
            key = str(unit["floor"])
            if key not in out:
                out[key] = {
                    "floor": unit["floor"],
                    "floor_name": unit["floor_name"],
                    "stores": [],
                    "crossings": [
                        {"label": name, "x": x, "y": 0.5,
                         "kind": "escalator" if name in ESCALATOR_BANKS else "gate"}
                        for name, x in CROSS_POINTS.items()
                    ],
                }
            out[key]["stores"].append({
                "store_id": unit["unit_id"],
                "name": unit["name"],
                "row": unit["row"],
                "index": unit["index"],
                "row_length": unit["row_length"],
                "x": round(unit["x"], 4),
                "y": unit["y"],
                "has_images": unit["image_count"] > 0,
            })
        for floor in out.values():
            floor["stores"].sort(key=lambda s: (s["row"], s["x"]))
        return out

    def find_units(self, query: str) -> list[dict]:
        q = query.strip().lower()
        return sorted(
            (u for u in self.units.values() if q in u["name"].lower()),
            key=lambda u: (len(u["name"]), u["floor"], u["index"]),
        )

    # Backwards-compatible aliases used by the CLI.
    find_stores = find_units

    @property
    def stores(self) -> dict[str, dict]:
        return self.units
