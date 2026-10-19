"""Self-contained shot-transition synthesis engine.

Public entry points:
    generate_transition_video  compose clips with cuts and xfade transitions
    get_video_info             probe duration and frame rate with ffprobe
    ALL_TRANSITIONS            the 59 supported transition names
"""

from .engine import generate_transition_video
from .media import get_video_info
from .transitions import ALL_TRANSITIONS, CUT, LONG_TRANSITIONS

__all__ = ["ALL_TRANSITIONS", "CUT", "LONG_TRANSITIONS", "generate_transition_video", "get_video_info"]
