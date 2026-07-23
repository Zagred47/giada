from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventKind(str, Enum):
    AXONAL_SPIKE = "axonal_spike"
    SOMATIC_SPIKE = "somatic_spike"
    BACKPROPAGATING_AP = "backpropagating_ap"
    CALCIUM_SPIKE = "calcium_spike"
    NMDA_SPIKE = "nmda_spike"
    NMDA_PLATEAU = "nmda_plateau"


@dataclass(frozen=True)
class EventLabel:
    """A teacher event expressed relative to the enclosing 1 ms step."""

    kind: EventKind
    segment_id: int
    region: str
    onset_offset_ms: float
    peak_offset_ms: float
    offset_offset_ms: float
    amplitude: float
    duration_ms: float
    detector_version: str
    confidence: Optional[float] = None

    def validate(self, step_ms: float = 1.0) -> None:
        if self.segment_id < 0:
            raise ValueError("segment_id must be non-negative")
        if not (
            0.0
            <= self.onset_offset_ms
            <= self.peak_offset_ms
            <= self.offset_offset_ms
            <= step_ms
        ):
            raise ValueError("event offsets must be ordered inside the step")
        if self.duration_ms < 0.0:
            raise ValueError("duration_ms must be non-negative")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.detector_version:
            raise ValueError("detector_version is required")
