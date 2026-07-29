"""Choose the adjacent-similarity threshold that segments corridor walks into
per-store groups.

Uses two hand-verified bursts from 3rd Left as ground truth: the Timezone shots
(170711/170726/170745) and the Decathlon shots (170859/170912/170924) must each
stay whole, and must not merge with their neighbours.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

TRUTH_GROUPS = [
    ["20260709_170711.jpg", "20260709_170726.jpg", "20260709_170745.jpg"],
    ["20260709_170859.jpg", "20260709_170912.jpg", "20260709_170924.jpg"],
]


def load():
    emb = np.load(config.DERIVED_DIR / "all_embeddings.npy")
    idx = json.loads((config.DERIVED_DIR / "all_index.json").read_text(encoding="utf-8"))
    return emb, idx


def corridors(idx):
    groups = defaultdict(list)
    for i, rec in enumerate(idx):
        groups[rec["folder"]].append(i)
    for folder in groups:
        groups[folder].sort(key=lambda i: idx[i]["order"])
    return groups


def segment(sims: list[float], threshold: float) -> list[int]:
    """Return a group id per image given adjacent similarities."""
    ids = [0]
    for s in sims:
        ids.append(ids[-1] if s >= threshold else ids[-1] + 1)
    return ids


def main() -> int:
    emb, idx = load()
    corr = corridors(idx)

    all_sims = []
    per_corridor = {}
    for folder, rows in corr.items():
        sims = [float(emb[rows[i]] @ emb[rows[i + 1]]) for i in range(len(rows) - 1)]
        per_corridor[folder] = (rows, sims)
        all_sims.extend(sims)

    arr = np.array(all_sims)
    print("ADJACENT COSINE SIMILARITY across all corridors")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{q:<3} {np.percentile(arr, q):.3f}")
    print(f"  mean {arr.mean():.3f}   min {arr.min():.3f}   max {arr.max():.3f}")

    print("\nGROUP COUNTS BY THRESHOLD")
    print(f"{'thresh':>7} {'groups':>7} {'avg imgs/store':>15}  truth-ok")
    for t in (0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94):
        total = 0
        name_to_group = {}
        for folder, (rows, sims) in per_corridor.items():
            ids = segment(sims, t)
            for local, row in enumerate(rows):
                name = Path(idx[row]["rel_path"]).name
                name_to_group[(folder, name)] = f"{folder}#{ids[local]}"
            total += len(set(ids))
        ok = all(
            len({name_to_group.get(("3rd Left", n)) for n in group}) == 1
            for group in TRUTH_GROUPS
        )
        distinct = len({name_to_group.get(("3rd Left", n)) for g in TRUTH_GROUPS for n in g})
        print(f"{t:>7.2f} {total:>7} {len(idx) / total:>15.2f}  "
              f"{'yes' if ok and distinct == 2 else 'no'}")

    print("\n3rd Left walk, adjacent similarity (marking known truth groups)")
    rows, sims = per_corridor["3rd Left"]
    truth_lookup = {n: f"T{i + 1}" for i, g in enumerate(TRUTH_GROUPS) for n in g}
    for i, row in enumerate(rows[:26]):
        name = Path(idx[row]["rel_path"]).name
        tag = truth_lookup.get(name, "  ")
        sim = f"{sims[i]:.3f}" if i < len(sims) else "  -  "
        print(f"  {tag:<3} {name}  next-sim {sim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
