"""Three functions vendored from the toolbox the published numbers were computed with.

Vendored rather than reimplemented on purpose. These are the matcher, the NMS merge and the
shot-boundary value types that decide *which* predictions count as matching *which* ground truth.
A reimplementation that looks equivalent but rounds or orders differently produces numbers that
are plausible and wrong -- the exact failure mode this whole package exists to avoid. Copying the
originals byte for byte makes the scoring exact by construction, not by argument.

Contents and their upstream files:

* ``segment_operations.py`` -- ``merge_segments`` and friends. Upstream: ``the toolbox``. stdlib only.
* ``list_match.py`` -- ``match_lists_v3``, the bare-overlap many-to-many matcher. Upstream:
  the toolbox.
* ``shot_boundary_types.py`` -- ``ShotBoundary``, ``ShotBoundaryComparer``, ``ShotBoundaryMerger``.
  Upstream: the toolbox. stdlib only.

**One deliberate deviation from byte-identical.** ``list_match.py`` had ``from tqdm import tqdm``
at module scope, which made ``tqdm`` a hard dependency of the scorer. Every call on this path
passes ``show_progress=False``, so the three ``tqdm(...)`` sites are unreachable here; the import
moved inside those branches. Nothing else in any of the three files was touched, and the scoring
path is unchanged. ``test_vendored_integrity.py`` pins the current hashes so any further edit is
caught.
"""
