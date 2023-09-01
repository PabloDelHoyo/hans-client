from __future__ import annotations

from dataclasses import dataclass
from PIL import Image
from io import BytesIO
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from client import HansPlatform


@dataclass
class Question:
    id: int
    collection_id: str
    prompt: str
    answers: list[str]
    img: Image

    @classmethod
    def from_hans_platform(cls, platform: HansPlatform, setup_msg):
        collection_id = setup_msg["collection_id"]
        question_id = setup_msg["question_id"]

        # TODO: handle possible errors (timeout or missing json)
        response = platform.get(f"question/{collection_id}/{question_id}")
        data = response.json()

        # NOTE: we could load the image when the question starts.
        # The main downside is that if, for whatever reason, we are not able to
        # load the it, there is no way to tell the server.
        #
        # I guess the the server knows that everything went right
        # if it has received the corresponding ready messages.
        image_response = platform.get(f"question/{collection_id}/{question_id}/image")
        img = Image.open(BytesIO(image_response.content))

        return cls(
            id=question_id,
            collection_id=collection_id,
            prompt=data["prompt"],
            answers=data["answers"],
            img=img,
        )


@dataclass
class Participant:
    name: str
    id: int


@dataclass
class Round:
    question: Question
    duration: int
    participants: list[Participant]
