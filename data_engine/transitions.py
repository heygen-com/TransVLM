"""Transition names and FFmpeg mapping for the data engine.

The 59 supported names match the historical implementation: one hard cut plus
58 FFmpeg ``xfade`` transitions. Order is preserved because it determines the
random draw order when transition types are sampled without a seed.
"""

CUT = "cut"

# The 58 gradual transitions, in the original order.
LONG_TRANSITIONS = (
    "fade",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "slideleft",
    "slideright",
    "slideup",
    "slidedown",
    "circlecrop",
    "rectcrop",
    "distance",
    "fadeblack",
    "fadewhite",
    "radial",
    "smoothleft",
    "smoothright",
    "smoothup",
    "smoothdown",
    "circleopen",
    "circleclose",
    "vertopen",
    "vertclose",
    "horzopen",
    "horzclose",
    "dissolve",
    "pixelize",
    "diagtl",
    "diagtr",
    "diagbl",
    "diagbr",
    "hlslice",
    "hrslice",
    "vuslice",
    "vdslice",
    "hblur",
    "fadegrays",
    "wipetl",
    "wipetr",
    "wipebl",
    "wipebr",
    "squeezeh",
    "squeezev",
    "zoomin",
    "fadefast",
    "fadeslow",
    "hlwind",
    "hrwind",
    "vuwind",
    "vdwind",
    "coverleft",
    "coverright",
    "coverup",
    "coverdown",
    "revealleft",
    "revealright",
    "revealup",
    "revealdown",
)

# All 59 names in the original order.
ALL_TRANSITIONS = (CUT,) + LONG_TRANSITIONS

# The hard cut is rendered with the concat filter, not with xfade.
FAST_TRANSITIONS = (CUT,)

# Compatibility aliases for callers that used the historical list names.
AVAILABLE_TRANSITIONS = ALL_TRANSITIONS
AVAILABLE_LONG_TRANSITIONS = LONG_TRANSITIONS
AVAILABLE_FAST_TRANSITIONS = FAST_TRANSITIONS

# Every non-cut name is passed verbatim to FFmpeg's xfade filter. This mapping
# makes the rendered-filter relationship explicit and is checked by tests.
XFRAME_MAPPING = {name: name for name in LONG_TRANSITIONS}
XFRAME_MAPPING[CUT] = "concat"

TRANSITION_KIND = {name: ("cut" if name == CUT else "xfade") for name in ALL_TRANSITIONS}


def is_supported(name: str) -> bool:
    """Return whether *name* is one of the 59 supported transitions."""
    return name in TRANSITION_KIND


def xfade_name(name: str) -> str:
    """Return the FFmpeg xfade transition name.

    Raises:
        ValueError: if *name* is unknown or is the hard cut, which uses concat.
    """
    if name not in TRANSITION_KIND:
        raise ValueError(f"Unsupported transition: {name!r}")
    if name == CUT:
        raise ValueError("The hard cut is rendered with concat, not xfade")
    return XFRAME_MAPPING[name]
