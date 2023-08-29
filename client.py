from __future__ import annotations

from typing import Optional
import json
from PIL import Image
from io import BytesIO
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import time
from datetime import datetime

import requests
import paho.mqtt.client as mqtt

from exceptions import CannotStartRoundException
from position_codec import PositionCodec

if TYPE_CHECKING:
    import numpy as np
    from game_loop import GameLoopThread


class Brain:
    def __init__(self, round):
        self.round = round

    def render(self, hans_client: HansClient):
        """This is where all code in which message packets are sent must go.
        Right now, those packets only contain position information.

        The rate at which this method is called may vary. That will depend on the work
        done by self.udpate()
        """

    def update(self, delta: float, sync_ratio: float):
        """All code related to the calculation of the next position.

        The rate which this method is called is guaranteed to be constant so delta is fixed.
        """


@dataclass
class Question:
    id: int
    collection_id: str
    prompt: str
    answers: List[str]
    img: Image

    @classmethod
    def from_setup_msg(cls, session, api_base, setup_msg_payload):
        collection_id = setup_msg_payload["collection_id"]
        question_id = setup_msg_payload["question_id"]

        # TODO: handle possible errors (timeout or missing json)
        response = session.get(f"{api_base}/question/{collection_id}/{question_id}")
        data = response.json()

        # TODO: we could load the image when the question starts.
        # The main downside is that if, for whatever reason, we are not able to
        # load the it, there is no way to tell the server.
        #
        # I guess the the server knows that everything went right
        # if it has received the corresponding ready messages.
        image_response = session.get(
            f"{api_base}/question/{collection_id}/{question_id}/image"
        )

        img = Image.open(BytesIO(image_response.content))

        return cls(
            id=question_id,
            collection_id=collection_id,
            prompt=data["prompt"],
            answers=data["answers"],
            img=img,
        )


@dataclass
class Round:
    question: Question
    duration: int


class HansClient:
    def __init__(self, platform: HansPlatform, pcodec: PositionCodec):
        self._platform = platform
        self.pcodec = pcodec

    def send_position(self, position: np.array, encode=True):
        # TODO: check if the date format is the one expected by the hans platform
        if encode:
            position = self.pcodec.encode(position)

        self._platform._publish(
            self._platform.update_topic,
            {
                "data": {"position": list(position)},
                "timeStamp": datetime.now().isoformat(),
            },
        )


class HansPlatform:
    def __init__(
        self, client_name: str, game_loop: GameLoopThread, session_id: int = 1
    ):
        self.client_name = client_name

        self._host = ""
        self._api_base = ""
        self._port = 3000
        self._broker_port = 9001
        self.client_id = None

        self._session_id = str(session_id)
        self._session_topic = f"swarm/session/{self._session_id}"

        self.control_topic = ""
        self.update_topic = ""

        self._session = None

        self._mqttc = mqtt.Client(transport="websockets", clean_session=True)
        self._mqttc.on_connect = self._on_connect
        self._mqttc.on_message = self._on_message

        self._game_loop = game_loop

        self._current_question = None

    def connect(
        self, host: str, port: Optional[int] = None, broker_port: Optional[int] = None
    ):
        self._host = host
        self._port = port or self._port
        self._broker_port = broker_port or self._broker_port

        self._api_base = f"http://{self._host}:{self._port}/api/"

        self._session = requests.Session()

        # Send login request
        req = self._session.post(
            f"{self._api_base}/session/{self._session_id}/participants",
            json={"user": self.client_name},
        )

        if req.content == b"Participant already joined session":
            raise ValueError(
                f"There already exists an user with the name {self.client_name}"
            )
        elif req.content == b"Session not found":
            raise ValueError(f"There does not exist session with id {self._session_id}")

        self.client_id = str(req.json()["id"])
        self.control_topic = f"{self._session_topic}/control/{self.client_id}"
        self.update_topic = f"{self._session_topic}/updates/{self.client_id}"

        self._mqttc.connect(self._host, self._broker_port)

    def listen(self, *args, **kwargs):
        """Listen to incoming MQTT requests and start the game loop thread"""
        self._game_loop.start()
        self._mqttc.loop_forever(*args, **kwargs)

    def _send_ready_msg(self):
        self._mqttc.publish(
            self.control_topic,
            payload=json.dumps(
                {
                    "type": "ready",
                    "participant": self.client_id,
                    "session": self._session_id,
                }
            ),
        )

    def _on_connect(self, client, userdata, flags, rc):
        self._mqttc.subscribe(f"{self._session_topic}/control/#")
        self._mqttc.subscribe(f"{self._session_topic}/updates/#")

        # This must be sent so that the client's name appears to the admin in the text area
        # where all connected clients are shown
        self._publish(
            self.control_topic,
            {
                "type": "join",
                "participant": self.client_id,
                "session": self._session_id,
            },
        )

    def _on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload)
        print(f"Recieved from {msg.topic}: {payload}")

        # Only control messages contain the "type" value. If in the future, that does not
        # hold, it is very important to change it here
        is_control = "type" in payload

        if is_control:
            self._handle_control_msgs(payload)
        else:
            # Right now, update messsages are sent when the users are responding. If it were
            # not the case, we would have to keep track of the state in which the client is
            pass

    def _handle_control_msgs(self, payload):
        if payload["type"] == "setup":
            self._current_question = Question.from_setup_msg(
                self._session, self._api_base, payload
            )

            print(f"The question has changed '{self._current_question.prompt}'")

            # I think this is to inform that everything went right
            self._publish(
                self.control_topic,
                {
                    "type": "ready",
                    "participant": self.client_id,
                    "session": self._session_id,
                },
            )
        elif payload["type"] == "start":
            print(f"A round has started")
            if self._current_question is None:
                raise CannotStartRoundException("The question has not been set")

            new_round = Round(self._current_question, payload["duration"])
            hans_client = HansClient(
                self, PositionCodec(num_answers=len(new_round.question.answers))
            )
            self._game_loop.new_loop(
                Round(self._current_question, payload["duration"]), hans_client
            )
        elif payload["type"] == "stop":
            self._game_loop.stop()

    def _publish(self, topic: str, payload):
        self._mqttc.publish(topic, payload=json.dumps(payload))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session is not None:
            self._session.close()