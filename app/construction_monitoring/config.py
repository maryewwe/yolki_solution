from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MatchingConfig:
    """Runtime controls for one observation.

    The defaults are engineering defaults for the prototype, not regulatory
    requirements. They must be calibrated with the CV team before production.
    """

    confidence_threshold: float = 0.25
    accept_legacy_class_strings: bool = True
    flag_possible_unexpected: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")


@dataclass(frozen=True)
class TemporalConfig:
    """Controls persistence checks across observations.

    window_seconds=None means that the caller already supplied the desired
    observation window. No fixed camera interval is assumed.
    """

    window_seconds: Optional[float] = None
    min_observations: int = 3
    persistence_ratio: float = 0.75

    def __post_init__(self) -> None:
        if self.window_seconds is not None and self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive or None")
        if self.min_observations < 1:
            raise ValueError("min_observations must be at least 1")
        if not 0.0 < self.persistence_ratio <= 1.0:
            raise ValueError("persistence_ratio must be in (0, 1]")
