from __future__ import annotations

from typing import Optional
import json
from typing import TYPE_CHECKING
from datetime import datetime

import requests
import paho.mqtt.client as mqtt

from model import Question, Round, Participant
from exceptions import CannotStartRoundException
from position_codec import PositionCodec

if TYPE_CHECKING:
    from sys import ExcInfo
    import numpy as np
    from loop import Loop, LoopThread

TOPIC_BASE = "swarm/session/{session_id}"
API_BASE = "http://{host}:{port}/api/"

CONTROL_TOPIC = "{topic_base}/control/{client_id}"
UPDATES_TOPIC = "{topic_base}/updates/{client_id}"


def raise_from_exc_info(exc_info: ExcInfo):
    _, value, traceback = exc_info
    raise value.with_traceback(traceback)


class HansClient:
    def __init__(self, platform: HansPlatform, pcodec: PositionCodec):
        self._platform = platform
        self.pcodec = pcodec

    def send_position(self, position: np.ndarray, encode=True):
        # TODO: check if the date format is the one expected by the hans platform
        if encode:
            position = self.pcodec.encode(position)

        self._platform.publish(
            "updates",
            {
                "data": {"position": list(position)},
                "timeStamp": datetime.now().isoformat(),
            },
        )

    @property
    def id(self):
        return self._platform.client_id


class HansPlatform:
    def __init__(self, client_name: str, loop: LoopThread, session_id: int = 1):
        self.client_name = client_name
        self.client_id = None

        self._api_base = ""

        self._session_id = str(session_id)
        self._session_topic = TOPIC_BASE.format(session_id=session_id)

        self._control_topic = ""
        self._update_topic = ""

        self._session: Optional[requests.Session] = None

        self._mqttc = mqtt.Client(transport="websockets", clean_session=True)
        self._mqtt_connected = False
        self._mqttc.on_connect = self._on_connect
        self._mqttc.on_message = self._on_message

        self._loop_thread = loop
        self._loop_thread.add_exc_handler(lambda: self._mqtt_disconnect())

        # TODO: the logic for setting a question is intermingled in this class. This
        # should be refactored
        self._current_question: Optional[Question] = None

    def connect(self, host: str, port: int = 3000, broker_port: int = 9001):
        self._api_base = API_BASE.format(host=host, port=port)

        self._session = requests.Session()

        # Send login request
        req = self.post(
            f"/session/{self._session_id}/participants",
            json={"user": self.client_name},
        )

        if req.content == b"Participant already joined session":
            raise ValueError(
                f"There already exists an user with the name {self.client_name}"
            )
        elif req.content == b"Session not found":
            raise ValueError(f"There does not exist session with id {self._session_id}")

        self.client_id = str(req.json()["id"])

        self._control_topic = CONTROL_TOPIC.format(
            topic_base=self._session_topic, client_id=self.client_id
        )
        self._update_topic = UPDATES_TOPIC.format(
            topic_base=self._session_topic, client_id=self.client_id
        )

        self._mqttc.connect(host, broker_port)

    def get(self, endpoint: str, *req_args, **req_kwargs) -> requests.Response:
        return self._session.get(
            f"{self._api_base}/{endpoint}", *req_args, **req_kwargs
        )

    def post(self, endpoint: str, *req_args, **req_kwargs) -> requests.Response:
        return self._session.post(
            f"{self._api_base}/{endpoint}", *req_args, **req_kwargs
        )

    def publish(self, topic: str, payload):
        if topic == "control":
            send_topic = self._control_topic
        elif topic == "updates":
            send_topic = self._update_topic
        else:
            raise ValueError("Incorrect topic")

        self._mqttc.publish(send_topic, payload=json.dumps(payload))

    def listen(self, *args, **kwargs):
        """Listen to incoming MQTT requests and start the game loop thread"""
        self._loop_thread.start()
        self._mqtt_connected = True
        self._mqttc.loop_forever(*args, **kwargs)
        if self._loop_thread.exc_info is not None:
            raise_from_exc_info(self._loop_thread.exc_info)

    def disconnect(self):
        self._mqtt_disconnect()
        if self._session is not None:
            self._session.close()
        if self._loop_thread.is_alive():
            self._loop_thread.quit()

    def _mqtt_disconnect(self):
        if self._mqttc is not None and self._mqtt_connected:
            self._mqtt_connected = False
            self._mqttc.disconnect()

    def _send_ready_msg(self):
        self.publish(
            "control",
            payload=json.dumps(
                {
                    "type": "ready",
                    "participant": self.client_id,
                    "session": self._session_id,
                }
            ),
        )

    def _on_connect(self, client, userdata, flags, rc):
        self._mqttc.subscribe(
            CONTROL_TOPIC.format(topic_base=self._session_topic, client_id="#")
        )
        self._mqttc.subscribe(
            UPDATES_TOPIC.format(topic_base=self._session_topic, client_id="#")
        )

        # This must be sent so that the client's name appears to the admin in the text area
        # where all connected clients are shown
        self.publish(
            "control",
            {
                "type": "join",
                "participant": self.client_id,
                "session": self._session_id,
            },
        )

    def _on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload)

        # Only control messages contain the "type" value. If in the future, that does not
        # hold, it is very important to change it here
        is_control = "type" in payload

        if is_control:
            self._handle_control_msgs(payload)
        else:
            # Right now, update messsages are sent when the users are responding. If it were
            # not the case, we would have to keep track of the state in which the client is
            participant_id = int(msg.topic.split("/")[-1])
            self._loop_thread.on_changed_position(participant_id, payload["data"])

    def _handle_control_msgs(self, payload):
        if payload["type"] == "setup":
            self._current_question = Question.from_hans_platform(self, payload)

            # I think this is to inform that everything went right
            self.publish(
                "control",
                {
                    "type": "ready",
                    "participant": self.client_id,
                    "session": self._session_id,
                },
            )
        elif payload["type"] == "start":
            if self._current_question is None:
                raise CannotStartRoundException("The question has not been set")

            participants = self._all_participants()
            new_round = Round(self._current_question, payload["duration"], participants)
            hans_client = HansClient(
                self, PositionCodec(num_answers=len(new_round.question.answers))
            )
            self._loop_thread.new_loop(new_round, hans_client)
        elif payload["type"] == "stop":
            self._loop_thread.stop()

    def _all_participants(self) -> list[str]:
        req = self.post(
            f"session/{self._session_id}/allParticipants",
            json={"user": "admin", "pass": "admin"},
        )

        return [Participant(user["username"], user["id"]) for user in req.json()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
