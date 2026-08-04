# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Split a video duration into overlapping sliding windows.

Same semantics as the helper the training pipeline used, kept self-contained here rather than
imported, because it is twenty lines.

The released checkpoint was evaluated with ``window_size=10.0, stride=9.0, strict_tail=False``.
``strict_tail`` matters more than it looks: with ``True`` a trailing span shorter than one window
is dropped, so the end of every video that is not an exact multiple of the stride goes unscored.

Callers pass the three floats directly; there is deliberately no config object for a function
that takes three numbers.
"""

from __future__ import annotations


def get_sliding_window_segments(
    start: float,
    end: float,
    window_size: float,
    stride: float,
    strict_tail: bool = True,
) -> list[tuple[float, float]]:
    """Generate sliding-window ``(seg_start, seg_end)`` intervals over ``[start, end)``.

    * If ``end - start < window_size`` and ``strict_tail`` is True, returns an
      empty list. If ``strict_tail`` is False, returns ``[(start, end)]``.
    * Otherwise fills full windows spaced ``stride`` apart, then appends one
      tail window that ends exactly at ``end`` (if ``strict_tail=True``) or
      that starts at the next stride tick and ends at ``end``
      (``strict_tail=False``).
    """
    if stride <= 0:
        raise ValueError(f'stride must be positive, got {stride}')
    if window_size <= 0:
        raise ValueError(f'window_size must be positive, got {window_size}')

    segments: list[tuple[float, float]] = []
    total = end - start
    count = int((total - window_size) // stride) + 1 if window_size <= total else 0
    for i in range(count):
        seg_start = start + i * stride
        segments.append((seg_start, seg_start + window_size))

    if segments:
        last_start, last_end = segments[-1]
        if last_end < end:
            if strict_tail:
                tail_start = end - window_size
                if tail_start > last_start:
                    segments.append((tail_start, end))
            else:
                tail_start = last_start + stride
                if tail_start >= end:
                    raise AssertionError(f'tail_start ({tail_start}) >= end ({end}); check inputs')
                segments.append((tail_start, end))
    elif not strict_tail:
        segments.append((start, end))
    return segments
