from __future__ import annotations

from typing import Optional
import logging
import json
from typing import TYPE_CHECKING
from datetime import datetime, timezone

import requests
import numpy as np
import paho.mqtt.client as mqtt

from . import utils
from .model import Question, Round, Participant
from .exceptions import CannotStartRoundException
from .position_codec import PositionCodec

if TYPE_CHECKING:
    from sys import ExcInfo
    from .loop import LoopThread

TOPIC_BASE = "swarm/session/{session_id}"
API_BASE = "http://{host}:{port}/api"

CONTROL_TOPIC = "{topic_base}/control/{client_id}"
UPDATES_TOPIC = "{topic_base}/updates/{client_id}"

logger = logging.getLogger(__name__)


def raise_from_exc_info(exc_info: ExcInfo):
    _, value, traceback = exc_info
    raise value.with_traceback(traceback)


class HansClient:
    def __init__(self, platform: HansPlatform, pcodec: PositionCodec):
        self._platform = platform
        self.pcodec = pcodec

    def send_position(self, position: np.ndarray, encode=True):
        if encode:
            try:
                position = self.pcodec.encode(position)
            except np.linalg.LinAlgError:
                logger.warning("Cannot encode %s. The position won't be sent", position)
                return

        self._platform.publish(
            "updates",
            {
                "data": {"position": list(position)},
                "timeStamp": datetime.now().astimezone(timezone.utc).isoformat(),
            },
        )

    @property
    def id(self):
        return self._platform.client_id


class HansPlatform:
    def __init__(
            self,
            client_name: str,
            loop: LoopThread,
            session_id: int = 1,
            *,
            hexagon_radius: float=340):

        self.client_name = client_name
        self.client_id = None

        self._hexagon_radius = hexagon_radius

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

        self._current_question: Optional[Question] = None

    def connect(self, host: str, port: int = 3000, broker_port: int = 9001):
        self._api_base = API_BASE.format(host=host, port=port)

        self._session = requests.Session()

        # Send login request
        req = self.post(
            f"session/{self._session_id}/participants",
            json={"user": self.client_name},
        )

        if req.content == b"Participant already joined session":
            raise ValueError(
                f"There already exists an user with the name {self.client_name}"
            )
        elif req.content == b"Session not found":
            raise ValueError(
                f"There does not exist session with id {self._session_id}")

        self.client_id = req.json()["id"]

        self._control_topic = CONTROL_TOPIC.format(
            topic_base=self._session_topic, client_id=self.client_id
        )
        self._update_topic = UPDATES_TOPIC.format(
            topic_base=self._session_topic, client_id=self.client_id
        )

        logger.info("Connecting to MQTT broker at %s:%s", host, broker_port)

        self._mqttc.connect(host, broker_port)

    def get(self, endpoint: str, **req_kwargs) -> requests.Response:
        uri = f"{self._api_base}/{endpoint}"
        logger.debug("Sending GET request to %s", uri)
        return self._session.get(
            uri, **req_kwargs
        )

    def post(self, endpoint: str, json=None, **req_kwargs) -> requests.Response:
        uri = f"{self._api_base}/{endpoint}"

        payload_debug = "empty payload" if json is None else f"payload {json}"
        logger.debug("Sending POST request to %s with %s", uri, payload_debug)

        return self._session.post(
            uri, json=json, **req_kwargs
        )

    def publish(self, topic: str, payload):
        if topic == "control":
            send_topic = self._control_topic
        elif topic == "updates":
            send_topic = self._update_topic
        else:
            raise ValueError("Incorrect topic")

        payload = json.dumps(payload)
        logger.debug("Publishing to topic '%s' with payload '%s'",
                     topic, payload)
        self._mqttc.publish(send_topic, payload=payload)

    def listen(self, *args, **kwargs):
        """Listen to incoming MQTT requests and start the game loop thread"""

        logger.info("Start listening for incoming MQTT packets")
        self._loop_thread.start()
        self._mqtt_connected = True
        self._mqttc.loop_forever(*args, **kwargs)
        if self._loop_thread.exc_info is not None:
            raise_from_exc_info(self._loop_thread.exc_info)

    def disconnect(self):
        logger.info("Disconnecting from MQTT broker")
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
        control_topic = CONTROL_TOPIC.format(
            topic_base=self._session_topic, client_id="#")
        logger.debug("Subscribing to %s", control_topic)
        self._mqttc.subscribe(control_topic)

        updates_topic = UPDATES_TOPIC.format(
            topic_base=self._session_topic, client_id="#")
        logger.debug("Subscribing to %s", updates_topic)
        self._mqttc.subscribe(updates_topic)

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
            logger.debug(
                "Received control msg from '%s' with payload '%s'", msg.topic, payload
            )
            self._handle_control_msgs(payload)
        else:
            # Right now, update messsages are sent when the users are responding. If it were
            # not the case, we would have to keep track of the state in which the client is
            participant_id = int(msg.topic.split("/")[-1])
            self._loop_thread.on_changed_position(
                participant_id, payload["data"])

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
                raise CannotStartRoundException(
                    "The question has not been set")

            participants = self._all_participants()
            answer_positions = utils.calculate_answer_points(
                len(self._current_question.answers), self._hexagon_radius
            )

            new_round = Round(self._current_question,
                              payload["duration"], participants,
                              answer_positions)

            hans_client = HansClient(
                self, PositionCodec(answer_positions)
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
