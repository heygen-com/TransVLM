"""Merge overlap-produced duplicate segments from adjacent sliding windows.

Uses the same semantics as the training pipeline: ``merge_segments(mode='ss', eps=0.02)``.
A plain sort-and-fold-adjacent implementation is NOT equivalent and produces different
boundaries -- see the width-dependent expansion rules below for why.

Semantics (per ``ShotBoundary.to_segment_level(eps)`` +
``_compare_segment_segment``):

* Point-level segments (``start == end``): expanded to ``[start-eps, end+eps]``.
* Narrow segments (``0 < end-start < 2*eps``): re-centered to
  ``[mid-eps, mid+eps]``.
* Wide segments (``end-start >= 2*eps``): **NOT expanded** — kept as-is.
* Overlap test: ``max(a.start, b.start) <= min(a.end, b.end)`` (touching
  counts). If overlap, merge b into a by union.

Walks the list in **insertion order**, not sorted. Time complexity is O(n²);
acceptable for the small per-video segment counts we see (5-100).

**Semantics diff vs a naive sort-and-fold implementation**: naive implementation expanded
BOTH sides by ±eps regardless of segment width, effectively merging any gap
≤ 2*eps. Real training pipeline only expands narrow/point segments; wide segments
(post ±tolerance expansion in the evaluator, all segments are wide) fall
through to plain touching-overlap check. Confirmed by tracing
``ShotBoundary.to_segment_level`` at
the training pipeline's merge routine.
"""

from __future__ import annotations

from transvlm.data.types import SegmentPrediction


def _to_segment_level(start: float, end: float, eps: float) -> tuple[float, float]:
    """Port of training pipeline ``ShotBoundary.to_segment_level(eps)`` (in
    the training pipeline's merge routine).

    Semantics:
    - point-level (``start == end``): expand to ``[start-eps, end+eps]``.
    - narrow segment (``0 < end-start < 2*eps``): re-center to
      ``[mid-eps, mid+eps]`` (width 2*eps).
    - wide segment (``end-start >= 2*eps``): return unchanged. **No expansion.**
      This asymmetry is easy to get wrong: expanding wide segments too would merge
      genuinely distinct transitions that happen to sit within eps of each other.
    """
    if end < start:
        start, end = end, start
    diff = end - start
    if diff == 0:
        return (start - eps, end + eps)
    if diff < 2 * eps:
        mid = (start + end) / 2.0
        return (mid - eps, mid + eps)
    return (start, end)


def _overlaps_within_eps(
    a: SegmentPrediction,
    b: SegmentPrediction,
    eps: float,
) -> bool:
    """True iff ``a`` and ``b`` overlap after each is canonicalized by
    ``_to_segment_level(eps)``. Matches training pipeline
    ``ShotBoundaryComparer._compare_segment_segment`` (touching endpoints
    count — ``s <= e``, not strict less).
    """
    a_lo, a_hi = _to_segment_level(a.start_time, a.end_time, eps)
    b_lo, b_hi = _to_segment_level(b.start_time, b.end_time, eps)
    s = max(a_lo, b_lo)
    e = min(a_hi, b_hi)
    return s <= e


def merge_shots(shots: list[SegmentPrediction], eps: float = 0.02) -> list[SegmentPrediction]:
    """Any-pair-overlap merger. O(n²). Preserves insertion order of the survivor.

    Matches the training pipeline's ``merge_segments(mode='ss', eps=0.02)``
    used by paper's Ours branch in
    the training pipeline output converter.

    Args:
        shots: candidate predictions in any order. ``start_time > end_time`` is
            transparently flipped.
        eps: tolerance in seconds. The training pipeline default is 0.02; the previous
            earlier default of 0.05 was too permissive (over-merged nearby boundaries).

    Returns:
        Merged list; survivor insertion order preserved. Widths union-expanded
        so the widened kept segment covers all absorbed candidates.
    """
    if not shots:
        return []

    # Normalize (flip inverted).
    normalized: list[SegmentPrediction] = []
    for s in shots:
        start, end = s.start_time, s.end_time
        if start > end:
            start, end = end, start
        normalized.append(
            SegmentPrediction(
                start_time=start,
                end_time=end,
                confidence=s.confidence,
                source_window=s.source_window,
            )
        )

    merged: list[SegmentPrediction] = []
    for cand in normalized:
        absorbed = False
        for i, kept in enumerate(merged):
            if _overlaps_within_eps(kept, cand, eps):
                # Union — widen kept to cover cand.
                merged[i] = SegmentPrediction(
                    start_time=min(kept.start_time, cand.start_time),
                    end_time=max(kept.end_time, cand.end_time),
                    confidence=kept.confidence,
                    source_window=kept.source_window,
                )
                absorbed = True
                break
        if not absorbed:
            merged.append(cand)
    return merged
