from __future__ import annotations

from dataclasses import dataclass
from PIL import Image
import json
import base64
from io import BytesIO

import numpy as np


@dataclass
class Question:
    id: int
    collection_id: str
    prompt: str
    answers: list[str]
    img: Image

    @classmethod
    def from_json(cls, data: str) -> Question:
        data = json.loads(data)
        return cls(
            id=data["id"],
            collection_id=data["collection_id"],
            prompt=data["prompt"],
            answers=data["answers"],
            img=Image.open(BytesIO(base64.b64decode(data["img"])))
        )

    def to_json(self) -> str:
        stream = BytesIO()
        self.img.save(stream, self.img.format)

        return json.dumps({
            "id": self.id,
            "collection_id": self.collection_id,
            "prompt": self.prompt,
            "answers": self.answers,
            # Not the most efficient solution but, right now, the most convenient ;)
            "img": base64.b64encode(stream.getvalue()).decode()
        })


@dataclass
class Participant:
    name: str
    id: int

    @classmethod
    def from_json(cls, data: str) -> Participant:
        return cls(**json.loads(data))

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "id": self.id
        })


@dataclass
class Round:
    question: Question
    duration: float
    participants: list[Participant]
    answer_positions: np.ndarray  # Shape: (len(question.answers), 2)
    radius: float

    @classmethod
    def from_json(cls, data: str) -> Round:
        data = json.loads(data)
        return cls(
            question=Question.from_json(data["question"]),
            # TODO: check why this is not an int
            duration=int(data["duration"]),
            participants=[
                Participant.from_json(participant_data) for participant_data in data["participants"]
            ],
            answer_positions=np.array(data["answer_positions"]),
            radius=data["radius"]
        )

    def to_json(self) -> str:
        return json.dumps({
            "question": self.question.to_json(),
            "duration": self.duration,
            "participants": [
                participant.to_json() for participant in self.participants
            ],
            "answer_positions": [list(position) for position in self.answer_positions],
            "radius": self.radius
        })
