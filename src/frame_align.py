"""Assign individual photographs to directory units.

Earlier the pipeline segmented each walk into groups first and only then matched
those groups to the floor directory. That order loses information: whenever two
neighbouring shopfronts looked alike the segmenter merged them, and the merged
blob could only carry one name, so every other store inside it vanished - a food
court collapsed four stalls into one.

Segmenting and naming are really one problem, and the directory already answers
half of it: the exact units on a row, in order. So frames are matched directly
to units.

The method is anchor-and-fill rather than a single global optimisation. A walk
produces two very different kinds of frame: shots where the fascia is legible,
which identify a unit almost beyond doubt, and shots of interiors, crowds,
promotional boards and wayfinding signage, whose text is actively misleading. A
scoring scheme that lets every frame vote equally lets the misleading ones drag
whole runs onto the wrong store.

So:
  1. Frames whose signage clearly names one unit become anchors.
  2. Anchors must appear in the order the row runs; the best-supported
     non-decreasing subsequence is kept and out-of-order anchors are dropped,
     which discards reflections and directory boards naming distant shops.
  3. Every remaining frame joins its nearest surviving anchor.

Ambiguous frames therefore inherit an identity from confident neighbours instead
of inventing one.
"""

from __future__ import annotations

import numpy as np

from .directory import similarity

# A frame anchors a unit only if its signage matches this strongly...
ANCHOR_MIN = 0.55

# ...and beats every other unit on the row by this margin, so a word shared by
# two shopfronts cannot anchor either.
ANCHOR_MARGIN = 0.10

# Below this, a unit's assignment rests on walk order rather than a legible read.
TEXT_EVIDENCE = 0.40


def text_scores(frame_tokens: dict, unit_names: list[str], aliases: dict) -> np.ndarray:
    """Score one frame against every unit on the row.

    Height dominates the score. A wide shot often catches the neighbouring
    shopfront too, and both names then match at full confidence; what separates
    them is that the fascia you are standing in front of is photographed much
    larger. Weighting by cap height turns that ambiguity into a clear margin.
    """
    scores = np.zeros(len(unit_names), dtype=np.float32)
    if not frame_tokens:
        return scores
    for j, name in enumerate(unit_names):
        best = 0.0
        for word, info in frame_tokens.items():
            sim = similarity(word, name, aliases)
            if sim <= 0:
                continue
            weight = 0.18 + 0.82 * float(info.get("rel_height", 0.0))
            best = max(best, sim * weight)
        scores[j] = best
    return scores


def _anchors(matrix: np.ndarray) -> list[tuple[int, int, float]]:
    """Frames that name exactly one unit convincingly: (frame, unit, score)."""
    found = []
    for i, row in enumerate(matrix):
        if not row.size:
            continue
        order = np.argsort(row)[::-1]
        best_j = int(order[0])
        best = float(row[best_j])
        runner_up = float(row[order[1]]) if row.size > 1 else 0.0
        if best >= ANCHOR_MIN and (best - runner_up) >= ANCHOR_MARGIN:
            found.append((i, best_j, best))
    return found


def _monotonic_subset(anchors: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Keep the best-supported anchors that respect the row's order.

    A walk passes units in order, so anchors must too. Maximising total score
    over non-decreasing unit indices drops the stragglers - typically a
    reflection, or a directory board naming a shop on the far side of the mall.
    """
    if not anchors:
        return []
    n = len(anchors)
    best = [a[2] for a in anchors]
    prev = [-1] * n
    for i in range(n):
        for k in range(i):
            if anchors[k][1] <= anchors[i][1] and best[k] + anchors[i][2] > best[i]:
                best[i] = best[k] + anchors[i][2]
                prev[i] = k
    end = int(np.argmax(best))
    chain = []
    while end != -1:
        chain.append(anchors[end])
        end = prev[end]
    return list(reversed(chain))


def align_walk(
    frame_tokens: list[dict],
    embeddings: np.ndarray,
    unit_names: list[str],
    aliases: dict,
) -> tuple[float, list[int]]:
    """Map each frame of one walk to a unit index, preserving order."""
    n, m = len(frame_tokens), len(unit_names)
    if n == 0 or m == 0:
        return 0.0, []

    matrix = np.vstack([text_scores(t, unit_names, aliases) for t in frame_tokens])
    kept = _monotonic_subset(_anchors(matrix))

    if not kept:
        # Nothing legible: spread the walk evenly so ordering is at least sane.
        return 0.0, [min(m - 1, (i * m) // max(n, 1)) for i in range(n)]

    # An anchor read its own fascia, so its unit is fixed. Everything else is
    # filled in around them.
    anchored = {frame: unit for frame, unit, _ in kept}
    assignment = [None] * n
    for frame, unit in anchored.items():
        assignment[frame] = unit

    # Unanchored frames join the nearer neighbouring anchor, and are clamped
    # between the two anchors that bracket them so the fill can never place a
    # frame outside the stretch of row it was walked through.
    frames = sorted(anchored)
    for i in range(n):
        if assignment[i] is not None:
            continue
        before = [f for f in frames if f <= i]
        after = [f for f in frames if f >= i]
        if before and after:
            lo, hi = before[-1], after[0]
            chosen = lo if (i - lo) <= (hi - i) else hi
            assignment[i] = min(max(anchored[chosen], anchored[lo]), anchored[hi])
        elif before:
            assignment[i] = anchored[before[-1]]
        else:
            assignment[i] = anchored[after[0]]

    total = sum(a[2] for a in kept)
    return total, assignment


def best_row_assignment(
    frame_tokens: list[dict],
    embeddings: np.ndarray,
    rows: dict[str, list[str]],
    aliases: dict,
    allowed_rows: list[str] | None = None,
) -> tuple[str, bool, list[int], float]:
    """Pick the row and walk direction that explain this corridor best.

    The capture used no consistent convention - one floor's 'left' walk runs the
    top row west to east, another's runs the bottom row east to west - so every
    combination is scored and the strongest wins.
    """
    best = None
    for row in (allowed_rows or list(rows)):
        names = rows.get(row) or []
        if not names:
            continue
        for flip in (False, True):
            ordered = list(reversed(names)) if flip else list(names)
            score, assignment = align_walk(frame_tokens, embeddings, ordered, aliases)
            normalised = score / max(len(frame_tokens), 1)
            if best is None or normalised > best[3]:
                canonical = [(len(ordered) - 1 - a) if flip else a for a in assignment]
                best = (row, flip, canonical, normalised)
    return best if best else ("top", False, [], 0.0)


def evidence_for(frame_tokens: list[dict], indices: list[int], name: str, aliases: dict) -> float:
    """Strongest signage evidence among the frames assigned to a unit."""
    best = 0.0
    for i in indices:
        for word, info in frame_tokens[i].items():
            sim = similarity(word, name, aliases)
            weight = 0.18 + 0.82 * float(info.get("rel_height", 0.0))
            best = max(best, sim * weight)
    return best
