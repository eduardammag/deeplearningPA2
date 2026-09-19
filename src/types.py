"""Tipos compartilhados entre as etapas do pipeline."""
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Detection:
    frame: int
    bbox: tuple[float, float, float, float]
    score: float = 1.0
    gt_id: Optional[int] = None
    visibility: float = 1.0

@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    hits: int = 1
    missed: int = 0
    age: int = 1
    gt_id: Optional[int] = None
    history: list[tuple[float, float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.history:
            self.history.append(self.bbox)
