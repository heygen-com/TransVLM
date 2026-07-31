from typing import List, Optional, Tuple

Segment = Tuple[float, float]


def _normalize_segment(segment: Segment) -> Segment:
    """Ensure the segment is ordered as (start, end)."""
    start, end = segment
    return (start, end) if start <= end else (end, start)


def _get_intersection_range(segment1: Segment, segment2: Segment) -> Optional[Segment]:
    """Return the closed intersection range if it exists, otherwise None."""
    start1, end1 = _normalize_segment(segment1)
    start2, end2 = _normalize_segment(segment2)
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    if inter_start <= inter_end:
        return (inter_start, inter_end)
    return None


def calculate_segment_intersection(segment1: Segment, segment2: Segment) -> List[Segment]:
    """
    Calculate the intersection of two closed intervals.

    Returns a list containing the intersection segment if it exists,
    otherwise an empty list.
    """
    intersection = _get_intersection_range(segment1, segment2)
    return [intersection] if intersection else []


def calculate_segment_union(segment1: Segment, segment2: Segment) -> List[Segment]:
    """
    Calculate the union of two closed intervals.

    If the segments overlap (including touching endpoints), return a single merged
    segment. Otherwise, return the two original segments ordered by start time to
    keep the result deterministic regardless of input order.
    """
    seg1 = _normalize_segment(segment1)
    seg2 = _normalize_segment(segment2)
    intersection = _get_intersection_range(seg1, seg2)
    if intersection:
        merged = (min(seg1[0], seg2[0]), max(seg1[1], seg2[1]))
        return [merged]
    # No overlap: return the original segments sorted by start then end.
    return sorted([seg1, seg2], key=lambda s: (s[0], s[1]))


def calculate_segment_difference(segment1: Segment, segment2: Segment) -> List[Segment]:
    """
    Calculate the difference of two closed intervals: segment1 - segment2.

    Order matters here. If there is no intersection, return [segment1].
    If there is an intersection, remove it from segment1 and return the remaining
    portion(s) (one or two segments).
    """
    seg1 = _normalize_segment(segment1)
    seg2 = _normalize_segment(segment2)
    intersection = _get_intersection_range(seg1, seg2)
    if not intersection:
        return [seg1]

    inter_start, inter_end = intersection
    start1, end1 = seg1
    result: List[Segment] = []

    # Left remainder
    if start1 < inter_start:
        result.append((start1, inter_start))
    # Right remainder
    if inter_end < end1:
        result.append((inter_end, end1))

    return result


# Others
def get_sliding_window_segments(start, end, window_size, stride, strict_tail=True):
    """
    Generate sliding window segments from start to end, with given window_size and stride.

    Args:
        start (float): Start time.
        end (float): End time (exclusive).
        window_size (float): Size of the window.
        stride (float): Stride.
        strict_tail (bool): Whether to strictly fill the tail.

    Returns:
        List[Tuple[float, float]]: List of (segment_start, segment_end) tuples.
    """
    segments = []
    count = int((end - start - window_size) // stride) + 1 if window_size <= end - start else 0
    for i in range(count):
        seg_start = start + i * stride
        seg_end = seg_start + window_size
        segments.append((seg_start, seg_end))
    # Optionally, cover the tail if end is not included
    if segments:
        last_start, last_end = segments[-1]
        if last_end < end:
            if strict_tail:
                tail_start = end - window_size
                if tail_start > last_start:
                    segments.append((tail_start, end))
            else:
                tail_start = last_start + stride
                tail_end = end
                assert tail_start < tail_end, f"tail_start ({tail_start}) must be less than tail_end ({tail_end})"
                segments.append((tail_start, tail_end))
    else:
        if strict_tail:
            print("strict_tail is True, but video length is less than window_size, return empty segments")
        else:
            segments.append((start, end))
    return segments


def merge_segments(segment_list, segment_cmpr_func, segment_mrg_func):
    """
    Simple NMS for list of segments.

    segment_list: List of segments (or any object)
    segment_cmpr_func: function(a, b) -> bool, True if a and b should be merged
    segment_mrg_func: function(a, b) -> merged instance

    Returns: filtered and merged segment_list, order preserved as much as possible
    """
    if not segment_list:
        return []

    result = []
    used = [False] * len(segment_list)

    for i, segment_a in enumerate(segment_list):
        if used[i]:
            continue
        merged_segment = segment_a
        for j in range(i + 1, len(segment_list)):
            if used[j]:
                continue
            segment_b = segment_list[j]
            if segment_cmpr_func(merged_segment, segment_b):
                merged_segment = segment_mrg_func(merged_segment, segment_b)
                used[j] = True
        used[i] = True
        result.append(merged_segment)
    return result
