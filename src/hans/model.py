from __future__ import annotations

from dataclasses import dataclass
from PIL import Image
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np


@dataclass
class Question:
    id: int
    collection_id: str
    prompt: str
    answers: list[str]
    img: Image

@dataclass
class Participant:
    name: str
    id: int


@dataclass
class Round:
    question: Question
    duration: int
    participants: list[Participant]
    answer_positions: np.ndarray  # Shape: (len(question.answers), 2)
    radius: float
