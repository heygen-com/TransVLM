from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Any, Dict

# Define constants as requested
AVAILABLE_TRANSITIONS = [
    "cut",  #! cut
    "radial",
    "slice",
    "reveal",
    "dissolve",
    "fade",
    "others",
    "wipe",
    "crop",
    "slide",
    "wind",
    "smooth",
    "cover",
    "squeeze",
    "others",  #! others
]
AVAILABLE_DIRECTION = [
    "tl",
    "down",
    "open",
    "up",
    "global",
    "bl",
    "tr",
    "close",
    "right",
    "br",
    "left",
]

# Define Enums based on the constants
# Using functional API to create Enums dynamically based on the lists
# Type hint as Any to avoid "Variable not allowed in type expression" linter errors
ShotBoundaryLabel: Any = Enum(
    "ShotBoundaryLabel",
    {item.upper(): item for item in AVAILABLE_TRANSITIONS},
    type=str,
)
ShotBoundaryDirection: Any = Enum(
    "ShotBoundaryDirection",
    {item.upper(): item for item in AVAILABLE_DIRECTION},
    type=str,
)


def _validate_float_range(value: Optional[float], min_val: float, max_val: float, name: str) -> None:
    """
    Validate that a float value is within a specified range [min_val, max_val].
    """
    if value is not None:
        if not isinstance(value, (float, int)):
            raise TypeError(f"{name} must be a float or int, got {type(value)}")
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def _ensure_enum(enum_cls: Any, value: Any, name: str) -> Any:
    """
    Ensure the value is a valid member of the given Enum class.
    Tries to convert from string value if necessary.
    """
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            pass
    raise ValueError(f"Invalid value '{value}' for {name}. Expected one of {[e.value for e in enum_cls]}")


def _ensure_type(value: Any, expected_type: Any, name: str) -> Any:
    """
    Basic type checking helper.
    """
    if value is None:
        return None
    # Handle List[str] generic type check roughly
    if hasattr(expected_type, "__origin__") and expected_type.__origin__ is list:
        if not isinstance(value, list):
            raise TypeError(f"{name} must be a list, got {type(value)}")
        # Check element types if needed, but might be expensive for large lists.
        # For now assume list is enough or check first element.
        return value

    if not isinstance(value, expected_type):
        # Allow int for float
        if expected_type is float and isinstance(value, int):
            return float(value)
        raise TypeError(f"{name} must be of type {expected_type}, got {type(value)}")
    return value


@dataclass(frozen=True, slots=True)
class ShotBoundary:
    """
    Data class representing a shot boundary in a video.
    """

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    label: Any = None  # Typed as Any to handle dynamic Enum
    direction: Any = None  # Typed as Any to handle dynamic Enum
    attributes: Any = None
    confidence: Any = None

    def __post_init__(self):
        """
        Perform validation after initialization.
        """
        # Validate confidence range
        if not isinstance(self.confidence, list):
            _validate_float_range(self.confidence, 0.0, 1.0, "confidence")

        # Additional type checks for fields to ensure class consistency
        # Since frozen=True, we can't modify self, but we can check values.
        # Note: dataclass generated __init__ doesn't do type coercion, so we check types here.
        # We assume _ensure_type logic or similar was applied before or we check here.
        if self.start_time is not None and not isinstance(self.start_time, (float, int)):
            raise TypeError(f"start_time must be float, got {type(self.start_time)}")
        if self.end_time is not None and not isinstance(self.end_time, (float, int)):
            raise TypeError(f"end_time must be float, got {type(self.end_time)}")

        # Check start_time <= end_time if both are provided
        if self.start_time is not None and self.end_time is not None:
            if self.start_time > self.end_time:
                raise ValueError(f"start_time ({self.start_time}) must be <= end_time ({self.end_time})")

        # Auto-convert label if it's a string, then check type
        if self.label is not None and not isinstance(self.label, list):
            if isinstance(self.label, str):
                # Attempt conversion
                converted_label = _ensure_enum(ShotBoundaryLabel, self.label, "label")
                object.__setattr__(self, "label", converted_label)

            if not isinstance(self.label, ShotBoundaryLabel):
                raise TypeError(f"label must be ShotBoundaryLabel, got {type(self.label)}")

        # Auto-convert direction if it's a string, then check type
        if self.direction is not None and not isinstance(self.direction, list):
            if isinstance(self.direction, str):
                # Attempt conversion
                converted_direction = _ensure_enum(ShotBoundaryDirection, self.direction, "direction")
                object.__setattr__(self, "direction", converted_direction)

            if not isinstance(self.direction, ShotBoundaryDirection):
                raise TypeError(f"direction must be ShotBoundaryDirection, got {type(self.direction)}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the object to a dictionary.
        """
        if not isinstance(self.label, list):
            label_value = self.label.value if self.label else None
        else:
            label_value = []
            for i in self.label:
                label_value.append(i.value if i else None)

        if not isinstance(self.direction, list):
            direction_value = self.direction.value if self.direction else None
        else:
            direction_value = []
            for i in self.direction:
                direction_value.append(i.value if i else None)

        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "label": label_value,
            "direction": direction_value,
            "attributes": self.attributes,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShotBoundary":
        """
        Create an instance from a dictionary.
        Handles missing keys by assigning None.
        Performs type conversion and validation.
        """
        # Extract values with default None
        start_time = data.get("start_time")
        if isinstance(start_time, str):
            start_time = float(start_time)
        end_time = data.get("end_time")
        if isinstance(end_time, str):
            end_time = float(end_time)
        label_raw = data.get("label")
        direction_raw = data.get("direction")
        attributes = data.get("attributes")
        confidence = data.get("confidence")

        # Type conversion/checking
        start_time = _ensure_type(start_time, float, "start_time")
        end_time = _ensure_type(end_time, float, "end_time")
        attributes = _ensure_type(attributes, List[str], "attributes")
        confidence = _ensure_type(confidence, float, "confidence")

        # Convert Enum values
        if not isinstance(label_raw, list):
            label = _ensure_enum(ShotBoundaryLabel, label_raw, "label")
        else:
            label = []
            for i in label_raw:
                label.append(_ensure_enum(ShotBoundaryLabel, i, "label"))
        if not isinstance(direction_raw, list):
            direction = _ensure_enum(ShotBoundaryDirection, direction_raw, "direction")
        else:
            direction = []
            for i in direction_raw:
                direction.append(_ensure_enum(ShotBoundaryDirection, i, "direction"))

        return cls(
            start_time=start_time,
            end_time=end_time,
            label=label,
            direction=direction,
            attributes=attributes,
            confidence=confidence,
        )

    def to_point_level(self) -> "ShotBoundary":
        """
        Convert the shot boundary to a point level boundary (start_time == end_time).
        If start_time != end_time, returns a new instance with start/end at midpoint.
        Otherwise returns self.
        """
        if self.start_time is None or self.end_time is None:
            return self

        diff = self.end_time - self.start_time
        if diff != 0:
            mid = (self.start_time + self.end_time) / 2.0
            # Explicitly create new instance instead of using replace
            return ShotBoundary(
                start_time=mid,
                end_time=mid,
                label=self.label,
                direction=self.direction,
                attributes=list(self.attributes) if self.attributes is not None else None,
                confidence=self.confidence,
            )
        return self

    def to_segment_level(self, eps: float) -> "ShotBoundary":
        """
        Convert the shot boundary to a segment level boundary.
        If start_time == end_time, returns a new instance expanded by eps.
        Otherwise returns self.
        """
        if eps <= 0:
            raise ValueError(f"eps must be greater than 0, got {eps}")

        if self.start_time is None or self.end_time is None:
            return self

        diff = self.end_time - self.start_time
        if diff == 0:
            # Explicitly create new instance instead of using replace
            return ShotBoundary(
                start_time=self.start_time - eps,
                end_time=self.end_time + eps,
                label=self.label,
                direction=self.direction,
                attributes=list(self.attributes) if self.attributes is not None else None,
                confidence=self.confidence,
            )
        elif diff < (eps * 2):
            mid = (self.start_time + self.end_time) / 2.0
            return ShotBoundary(
                start_time=mid - eps,
                end_time=mid + eps,
                label=self.label,
                direction=self.direction,
                attributes=list(self.attributes) if self.attributes is not None else None,
                confidence=self.confidence,
            )
        return self

    def is_point_level(self) -> bool:
        """
        Check if the boundary is point level (start_time == end_time).
        """
        if self.start_time is None or self.end_time is None:
            return False
        return self.start_time == self.end_time

    def is_segment_level(self) -> bool:
        """
        Check if the boundary is segment level (start_time != end_time).
        """
        if self.start_time is None or self.end_time is None:
            return False
        return self.start_time != self.end_time


class ShotBoundaryComparer:
    """
    Helper class to compare two ShotBoundary objects based on different modes.
    """

    def _check_shot_boundary(self, sb: ShotBoundary) -> None:
        """
        Check if the shot boundary is valid.
        """
        if sb.start_time is None or sb.end_time is None:
            raise ValueError(f"sb.start_time and sb.end_time must not be None, got {sb.start_time} and {sb.end_time}")

    def _compare_point_point(self, sb1: ShotBoundary, sb2: ShotBoundary) -> bool:
        """
        Compare two point-level ShotBoundaries.
        Returns True if their times are equal.
        """
        if not sb1.is_point_level() or not sb2.is_point_level():
            raise ValueError(f"sb1 and sb2 must be point-level, got {sb1.is_point_level()} and {sb2.is_point_level()}")

        self._check_shot_boundary(sb1)
        self._check_shot_boundary(sb2)

        return sb1.start_time == sb2.start_time and sb1.end_time == sb2.end_time  # type: ignore

    def _compare_point_segment(self, sb1: ShotBoundary, sb2: ShotBoundary) -> bool:
        """
        Compare a point-level sb1 with a segment-level sb2.
        Returns True if sb1 is within sb2's range.
        """
        if not sb1.is_point_level() or not sb2.is_segment_level():
            raise ValueError(
                f"sb1 must be point-level and sb2 must be segment-level, got {sb1.is_point_level()} and {sb2.is_segment_level()}"
            )

        self._check_shot_boundary(sb1)
        self._check_shot_boundary(sb2)

        return (sb2.start_time <= sb1.start_time <= sb2.end_time) and (sb2.start_time <= sb1.end_time <= sb2.end_time)  # type: ignore

    def _compare_segment_point(self, sb1: ShotBoundary, sb2: ShotBoundary) -> bool:
        """
        Compare a segment-level sb1 with a point-level sb2.
        Returns True if sb2 is within sb1's range.
        """
        if not sb1.is_segment_level() or not sb2.is_point_level():
            raise ValueError(
                f"sb1 must be segment-level and sb2 must be point-level, got {sb1.is_segment_level()} and {sb2.is_point_level()}"
            )

        self._check_shot_boundary(sb1)
        self._check_shot_boundary(sb2)

        return (sb1.start_time <= sb2.start_time <= sb1.end_time) and (sb1.start_time <= sb2.end_time <= sb1.end_time)  # type: ignore

    def _compare_segment_segment(self, sb1: ShotBoundary, sb2: ShotBoundary) -> bool:
        """
        Compare two segment-level ShotBoundaries.
        Returns True if they overlap.
        """
        if not sb1.is_segment_level() or not sb2.is_segment_level():
            raise ValueError(
                f"sb1 must be segment-level and sb2 must be segment-level, got {sb1.is_segment_level()} and {sb2.is_segment_level()}"
            )

        self._check_shot_boundary(sb1)
        self._check_shot_boundary(sb2)

        s = max(sb1.start_time, sb2.start_time)  # type: ignore
        e = min(sb1.end_time, sb2.end_time)  # type: ignore
        return s <= e

    def __call__(
        self,
        sb1: ShotBoundary,
        sb2: ShotBoundary,
        mode: str,
        eps: float = 0.02,
        check_label: bool = False,
        check_direction: bool = False,
    ) -> bool:
        """
        Compare two ShotBoundaries based on the specified mode.
        Modes:
        - "pp": point-point comparison
        - "ps": point-segment comparison
        - "sp": segment-point comparison
        - "ss": segment-segment comparison

        Args:
            sb1: The first ShotBoundary to compare.
            sb2: The second ShotBoundary to compare.
            mode: The comparison mode ("pp", "ps", "sp", "ss").
            eps: Epsilon value for segment conversion. Defaults to 0.02.
            check_label: If True, checks if labels match before comparing times. Defaults to False.
            check_direction: If True, checks if directions match before comparing times. Defaults to False.

        Returns:
            bool: True if the boundaries match according to the criteria, False otherwise.
        """
        if check_label:
            if sb1.label != sb2.label:
                return False

        if check_direction:
            if sb1.direction != sb2.direction:
                return False

        if mode == "pp":
            return self._compare_point_point(sb1.to_point_level(), sb2.to_point_level())
        elif mode == "ps":
            return self._compare_point_segment(sb1.to_point_level(), sb2.to_segment_level(eps))
        elif mode == "sp":
            return self._compare_segment_point(sb1.to_segment_level(eps), sb2.to_point_level())
        elif mode == "ss":
            return self._compare_segment_segment(sb1.to_segment_level(eps), sb2.to_segment_level(eps))
        else:
            raise ValueError(f"Invalid mode: {mode}")


class ShotBoundaryMerger:
    """
    Helper class to merge two ShotBoundary objects.
    """

    def _merge_start_time(self, sb1: ShotBoundary, sb2: ShotBoundary) -> Optional[float]:
        sb1_start_time = sb1.start_time
        sb2_start_time = sb2.start_time
        if sb1_start_time is None and sb2_start_time is None:
            return None
        if sb1_start_time is None:
            return sb2_start_time
        if sb2_start_time is None:
            return sb1_start_time
        return min(sb1_start_time, sb2_start_time)

    def _merge_end_time(self, sb1: ShotBoundary, sb2: ShotBoundary) -> Optional[float]:
        sb1_end_time = sb1.end_time
        sb2_end_time = sb2.end_time
        if sb1_end_time is None and sb2_end_time is None:
            return None
        if sb1_end_time is None:
            return sb2_end_time
        if sb2_end_time is None:
            return sb1_end_time
        return max(sb1_end_time, sb2_end_time)

    def _merge_to_list(self, a, b) -> List[Any]:
        if not isinstance(a, list):
            a = [a]
        if not isinstance(b, list):
            b = [b]
        return a + b

    def _merge_label(self, sb1: ShotBoundary, sb2: ShotBoundary) -> List[Any]:
        return self._merge_to_list(sb1.label, sb2.label)

    def _merge_direction(self, sb1: ShotBoundary, sb2: ShotBoundary) -> List[Any]:
        return self._merge_to_list(sb1.direction, sb2.direction)

    def _merge_attributes(self, sb1: ShotBoundary, sb2: ShotBoundary) -> List[Any]:
        return self._merge_to_list(sb1.attributes, sb2.attributes)

    def _merge_confidence(self, sb1: ShotBoundary, sb2: ShotBoundary) -> List[Any]:
        return self._merge_to_list(sb1.confidence, sb2.confidence)

    def __call__(self, sb1: ShotBoundary, sb2: ShotBoundary) -> ShotBoundary:
        """
        Merge two ShotBoundary objects.
        """
        sb_start_time = self._merge_start_time(sb1, sb2)
        sb_end_time = self._merge_end_time(sb1, sb2)
        sb_label = self._merge_label(sb1, sb2)
        sb_direction = self._merge_direction(sb1, sb2)
        sb_attributes = self._merge_attributes(sb1, sb2)
        sb_confidence = self._merge_confidence(sb1, sb2)
        return ShotBoundary(
            start_time=sb_start_time,
            end_time=sb_end_time,
            label=sb_label,
            direction=sb_direction,
            attributes=sb_attributes,
            confidence=sb_confidence,
        )


if __name__ == "__main__":
    # Test logic
    data = {
        "start_time": 0.0,
        "end_time": 1.0,
        "label": "cut",
        "direction": "left",
        "attributes": ["attribute1", "attribute2"],
        "confidence": 0.9,
    }
    shot_boundary = ShotBoundary.from_dict(data)
    print(shot_boundary)
    print(shot_boundary.to_dict())

    shot_boundary_2 = ShotBoundary(
        start_time=0.5,
        end_time=1.5,
        label="fade",
        direction="right",
        attributes=["attribute3", "attribute4"],
        confidence=0.8,
    )

    print(shot_boundary_2.to_point_level())
    print(shot_boundary_2.to_segment_level(0.02))
    print(shot_boundary_2)
    print(shot_boundary_2.is_point_level())
    print(shot_boundary_2.is_segment_level())

    comparer = ShotBoundaryComparer()
    # pp comparison (midpoints: 0.5 vs 1.0) -> False
    print(f"pp: {comparer(shot_boundary, shot_boundary_2, 'pp')}")
    # ps
    print(f"ps: {comparer(shot_boundary, shot_boundary_2, 'ps')}")
    # sp
    print(f"sp: {comparer(shot_boundary, shot_boundary_2, 'sp')}")
    # ss comparison (overlap [0, 1] and [0.5, 1.5] -> [0.5, 1]) -> True
    print(f"ss: {comparer(shot_boundary, shot_boundary_2, 'ss')}")
