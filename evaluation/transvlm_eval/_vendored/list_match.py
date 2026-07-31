from typing import List, Tuple, Any, Callable, Set, Optional


def _get_unmatched_indices(total_len: int, matched_indices: Set[int]) -> List[int]:
    """
    Get indices that are not in the matched set.
    """
    return [i for i in range(total_len) if i not in matched_indices]


def _log_match_info(
    len_list1: int,
    len_list2: int,
    len_matched: int,
    len_remains1: int,
    len_remains2: int,
) -> None:
    """
    Log information about the matching process.
    """
    print("-" * 30)
    print("Match Statistics:")
    print(f"List1 Length: {len_list1}")
    print(f"List2 Length: {len_list2}")
    print(f"Matched Pairs: {len_matched}")
    print(f"List1 Remains: {len_remains1}")
    print(f"List2 Remains: {len_remains2}")
    print("-" * 30)


def match_lists(
    list1: List[Any],
    list2: List[Any],
    eq_func: Callable[[Any, Any], bool],
    reuse_list1_item: bool = False,
    reuse_list2_item: bool = False,
    show_progress: bool = False,
    show_info: bool = False,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match elements from two lists based on an equality function.

    Args:
        list1: The first list to match.
        list2: The second list to match.
        eq_func: A function that takes two arguments (one from list1, one from list2)
                 and returns True if they match, False otherwise.
        reuse_list1_item: If True, an item in list1 can be matched with multiple items in list2.
                          Default is False.
        reuse_list2_item: If True, an item in list2 can be matched with multiple items in list1.
                          Default is False.
        show_progress: If True, displays a progress bar during matching. Default is False.
        show_info: If True, logs matching statistics after completion. Default is False.

    Returns:
        A tuple containing:
        - matched: A list of tuples, where each tuple contains (index_in_list1, index_in_list2).
        - list1_remains: A list of indices from list1 that were not matched.
        - list2_remains: A list of indices from list2 that were not matched.
    """
    matched: List[Tuple[int, int]] = []
    matched_indices_1: Set[int] = set()
    matched_indices_2: Set[int] = set()

    # Determine the iterator for list1, potentially wrapping with tqdm
    iterator = range(len(list1))
    if show_progress:
        from tqdm import tqdm  # local: only needed when progress is requested

        iterator = tqdm(iterator, total=len(list1), desc="Matching")

    for i in iterator:
        item1 = list1[i]

        # Iterate through list2 to find matches
        for j, item2 in enumerate(list2):
            # If list2 items cannot be reused and this index is already matched, skip it
            if not reuse_list2_item and j in matched_indices_2:
                continue

            # Check for equality
            if eq_func(item1, item2):
                matched.append((i, j))
                matched_indices_1.add(i)
                matched_indices_2.add(j)

                # If list1 items cannot be reused, stop looking for matches for this item1
                # after the first match is found.
                if not reuse_list1_item:
                    break

    # Identify unmatched indices
    list1_remains = _get_unmatched_indices(len(list1), matched_indices_1)
    list2_remains = _get_unmatched_indices(len(list2), matched_indices_2)

    # print(f"show_info: {show_info}")
    if show_info:
        _log_match_info(
            len(list1),
            len(list2),
            len(matched),
            len(list1_remains),
            len(list2_remains),
        )

    return matched, list1_remains, list2_remains


def match_lists_v2(
    list1: List[Any],
    list2: List[Any],
    eq_func: Callable[[Any, Any], bool],
    show_progress: bool = False,
    show_info: bool = False,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match elements from two lists based on an equality function.

    Args:
        list1: The first list to match.
        list2: The second list to match.
        eq_func: A function that takes two arguments (one from list1, one from list2) and returns True if they match, False otherwise.
        show_progress: If True, displays a progress bar during matching. Default is False.
        show_info: If True, logs matching statistics after completion. Default is False.

    Returns:
        A tuple containing:
        - matched: A list of tuples, where each tuple contains (index_in_list1, index_in_list2).
        - list1_remains: A list of indices from list1 that were not matched.
        - list2_remains: A list of indices from list2 that were not matched.
    """
    matched: List[Tuple[int, int]] = []
    matched_indices_1: Set[int] = set()
    matched_indices_2: Set[int] = set()

    # Determine the iterator for list1, potentially wrapping with tqdm
    iterator = range(len(list1))
    if show_progress:
        from tqdm import tqdm  # local: only needed when progress is requested

        iterator = tqdm(iterator, total=len(list1), desc="Matching")

    for i in iterator:
        item1 = list1[i]
        for j, item2 in enumerate(list2):
            if eq_func(item1, item2):
                matched.append((i, j))
                matched_indices_1.add(i)
                matched_indices_2.add(j)

    # Identify unmatched indices
    list1_remains = _get_unmatched_indices(len(list1), matched_indices_1)
    list2_remains = _get_unmatched_indices(len(list2), matched_indices_2)
    matched_unique = list(set(matched))
    matched_unique = sorted(matched_unique, key=lambda x: (x[0], x[1]))

    # print(f"show_info: {show_info}")
    if show_info:
        print(f"matched: {matched}")
        print(f"matched_unique: {matched_unique}")
        _log_match_info(
            len(list1),
            len(list2),
            len(matched),
            len(list1_remains),
            len(list2_remains),
        )

    return matched_unique, list1_remains, list2_remains


def match_lists_v3(
    list1: List[Any],
    list2: List[Any],
    get_start: Callable[[Any], float],
    get_end: Callable[[Any], float],
    show_progress: bool = False,
    show_info: bool = False,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match elements from two lists using a two-pointer algorithm for interval intersection.

    This function is optimized for the case where:
    1. Both lists contain interval-like objects with start_time and end_time.
    2. Both lists are sorted by start_time (or can be sorted).
    3. Intervals within each list are non-overlapping (already merged).

    The algorithm achieves O((n+m) log(n+m) + k) time complexity where k is the
    number of matched pairs, compared to O(n*m) for the naive approach.

    Args:
        list1: The first list to match (e.g., ground truth intervals).
        list2: The second list to match (e.g., predicted intervals).
        get_start: A function that extracts the start time from an element.
        get_end: A function that extracts the end time from an element.
        show_progress: If True, displays a progress bar during matching. Default is False.
        show_info: If True, logs matching statistics after completion. Default is False.

    Returns:
        A tuple containing:
        - matched: A list of tuples, where each tuple contains (index_in_list1, index_in_list2).
        - list1_remains: A list of indices from list1 that were not matched.
        - list2_remains: A list of indices from list2 that were not matched.

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Interval:
        ...     start_time: float
        ...     end_time: float
        >>> list1 = [Interval(0.0, 1.0), Interval(2.0, 3.0), Interval(5.0, 6.0)]
        >>> list2 = [Interval(0.5, 1.5), Interval(4.0, 5.5)]
        >>> matched, rem1, rem2 = match_lists_v3(
        ...     list1, list2,
        ...     get_start=lambda x: x.start_time,
        ...     get_end=lambda x: x.end_time,
        ... )
        >>> # Interval(0, 1) overlaps with Interval(0.5, 1.5)
        >>> # Interval(5, 6) overlaps with Interval(4.0, 5.5)
        >>> matched
        [(0, 0), (2, 1)]
    """
    if not list1 or not list2:
        # Handle empty lists
        list1_remains = list(range(len(list1)))
        list2_remains = list(range(len(list2)))
        if show_info:
            _log_match_info(len(list1), len(list2), 0, len(list1_remains), len(list2_remains))
        return [], list1_remains, list2_remains

    # Create index-sorted lists: (original_index, start, end)
    # Sort by start time, then by end time for tie-breaking
    indexed1: List[Tuple[int, float, float]] = [
        (i, get_start(item), get_end(item)) for i, item in enumerate(list1)
    ]
    indexed2: List[Tuple[int, float, float]] = [
        (i, get_start(item), get_end(item)) for i, item in enumerate(list2)
    ]

    # Sort both lists by start time
    indexed1.sort(key=lambda x: (x[1], x[2]))
    indexed2.sort(key=lambda x: (x[1], x[2]))

    matched: List[Tuple[int, int]] = []
    matched_indices_1: Set[int] = set()
    matched_indices_2: Set[int] = set()

    # Two-pointer algorithm:
    # For each interval in list1, find all overlapping intervals in list2.
    # Since both lists are sorted by start time and intervals within each list
    # are non-overlapping, we can use a sliding window approach.

    n1, n2 = len(indexed1), len(indexed2)

    # For each interval i in list1, we maintain a pointer j that indicates
    # the first interval in list2 that could potentially overlap with i.
    # An interval j can be skipped if its end_time < start_time of i.

    j_start = 0  # Starting pointer for list2

    # Determine the iterator for list1, potentially wrapping with tqdm
    iterator = range(n1)
    if show_progress:
        from tqdm import tqdm  # local: only needed when progress is requested

        iterator = tqdm(iterator, total=n1, desc="Matching (v3 two-pointer)")

    for i in iterator:
        orig_i, start_i, end_i = indexed1[i]

        # Move j_start forward to skip intervals in list2 that end before list1[i] starts
        # These intervals cannot overlap with any remaining intervals in list1
        # (since list1 is also sorted by start time)
        while j_start < n2 and indexed2[j_start][2] < start_i:
            j_start += 1

        # Now scan from j_start to find all overlapping intervals
        # Stop when list2[j].start > list1[i].end (no more overlaps possible)
        j = j_start
        while j < n2:
            orig_j, start_j, end_j = indexed2[j]

            # Check if list2[j] starts after list1[i] ends -> no more overlaps
            if start_j > end_i:
                break

            # Check for overlap: max(start_i, start_j) <= min(end_i, end_j)
            if max(start_i, start_j) <= min(end_i, end_j):
                matched.append((orig_i, orig_j))
                matched_indices_1.add(orig_i)
                matched_indices_2.add(orig_j)

            j += 1

    # Identify unmatched indices
    list1_remains = _get_unmatched_indices(len(list1), matched_indices_1)
    list2_remains = _get_unmatched_indices(len(list2), matched_indices_2)

    # Sort matched pairs by (index_in_list1, index_in_list2)
    matched.sort(key=lambda x: (x[0], x[1]))

    if show_info:
        _log_match_info(
            len(list1),
            len(list2),
            len(matched),
            len(list1_remains),
            len(list2_remains),
        )

    return matched, list1_remains, list2_remains
