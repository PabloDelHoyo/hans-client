from __future__ import annotations

from typing import Optional
import logging
import json
from typing import TYPE_CHECKING
from datetime import datetime, timezone
from PIL import Image
from io import BytesIO

import requests
import numpy as np
import paho.mqtt.client as mqtt

from . import utils
from .model import Question, Round, Participant
from .exceptions import CannotStartRoundException
from .position_codec import PositionCodec

if TYPE_CHECKING:
    from sys import ExcInfo
    from .agent import AgentManager

TOPIC_BASE = "swarm/session/{session_id}"
API_BASE = "http://{host}:{port}/api"

CONTROL_TOPIC = "{topic_base}/control/{client_id}"
UPDATES_TOPIC = "{topic_base}/updates/{client_id}"

logger = logging.getLogger(__name__)


def _raise_from_exc_info(exc_info: ExcInfo):
    _, value, traceback = exc_info
    raise value.with_traceback(traceback)


def _get(req_session: requests.Session,
         api_base: str,
         endpoint: str, **req_kwargs) -> requests.Response:
    uri = f"{api_base}/{endpoint}"
    logger.debug("Sending GET request to %s", uri)
    return req_session.get(
        uri, **req_kwargs
    )


def _post(req_session: requests.Session,
          api_base: str,
          endpoint: str, json=None, **req_kwargs) -> requests.Response:

    uri = f"{api_base}/{endpoint}"

    payload_debug = "empty payload" if json is None else f"payload {json}"
    logger.debug("Sending POST request to %s with %s", uri, payload_debug)

    return req_session.post(
        uri, json=json, **req_kwargs
    )


class HansClient:
    """Wrapper which only expose the needed functionality from HansPlatform to a Loop instance"""

    def __init__(self, api_wrapper: _HansApiWrapper, pcodec: PositionCodec):
        self._api_wrapper = api_wrapper
        self.pcodec = pcodec

    def send_position(self, position: np.ndarray, encode=True):
        if encode:
            try:
                position = self.pcodec.encode(position)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Cannot encode %s. The position won't be sent", position)
                return

        self._api_wrapper.publish(
            "updates",
            {
                "data": {
                    "position": list(position),
                    "timeStamp": datetime.now().astimezone(timezone.utc).isoformat(),
                },
            },
        )

    @property
    def id(self):
        return self._api_wrapper.client_id


class _HansApiWrapper:

    def __init__(
            self,
            req_session: requests.Session,
            mqttc: mqtt.Client,
            session_id: int,
            api_base: str,
            client_id: str,
            publish_topics: dict[str, str],
            subscribe_topics: list[str]):

        self._req_session = req_session
        self._mqttc = mqttc
        self._session_id = session_id
        self._api_base = api_base
        self._client_id = client_id
        self._publish_topics = publish_topics
        self._subscribe_topics = subscribe_topics

    @classmethod
    def from_connection(
        cls,
        client_name: str,
        session_id: int,
        api_host: str,
    ):

        mqttc = mqtt.Client(transport="websockets", clean_session=True)
        req_session = requests.Session()

        session_id = str(session_id)
        session_topic = TOPIC_BASE.format(session_id=session_id)

        api_base = api_host + "/api"

        req = _post(
            req_session,
            api_base,
            f"session/{session_id}/participants",
            json={"user": client_name}
        )

        if req.content == b"Participant already joined session":
            raise ValueError(
                f"There already exists an user with the name {client_name}"
            )
        elif req.content == b"Session not found":
            raise ValueError(
                f"There does not exist session with id {session_id}")

        client_id = req.json()["id"]

        publish_topics = {
            "control": CONTROL_TOPIC.format(
                topic_base=session_topic, client_id=client_id
            ),
            "updates": UPDATES_TOPIC.format(
                topic_base=session_topic, client_id=client_id
            )
        }

        subscribe_topics = [
            CONTROL_TOPIC.format(
                topic_base=session_topic, client_id="#"
            ),
            UPDATES_TOPIC.format(
                topic_base=session_topic, client_id="#"
            )
        ]

        return cls(
            req_session=req_session,
            mqttc=mqttc,
            session_id=session_id,
            api_base=api_base,
            client_id=client_id,
            publish_topics=publish_topics,
            subscribe_topics=subscribe_topics
        )

    def set_offline(self):
        self.post(f"session/{self._session_id}/participants/{self.client_id}")
        self.publish("control", {
            "type": "leave",
            "participant": self._client_id,
            "session": self._session_id
        })

    def disconnect(self):
        self.set_offline()
        self._req_session.close()
        if self._mqttc.is_connected:
            self._mqttc.disconnect()

    def get_question_from_id(self, collection_id, question_id):
        # TODO: handle possible errors (timeout or missing json)
        response = self.get(f"question/{collection_id}/{question_id}")
        data = response.json()

        # NOTE: we could load the image when the question starts.
        # The main downside is that if, for whatever reason, we are not able to
        # load the it, there is no way to tell the server.
        #
        # I guess the the server knows that everything went right
        # if it has received the corresponding ready messages.
        image_response = self.get(
            f"question/{collection_id}/{question_id}/image")
        img = Image.open(BytesIO(image_response.content))

        return Question(
            id=question_id,
            collection_id=collection_id,
            prompt=data["question"],
            answers=data["answers"],
            img=img,
        )

    def get_all_participants(self) -> list[Participant]:
        req = self.post(
            f"session/{self._session_id}/allParticipants",
            json={"user": "admin", "pass": "admin"},
        )

        return [Participant(user["username"], user["id"]) for user in req.json()]

    def get(self, endpoint: str, **req_kwargs) -> requests.Response:
        return _get(self._req_session, self._api_base, endpoint, **req_kwargs)

    def post(self, endpoint: str, json=None, **req_kwargs) -> requests.Response:
        return _post(self._req_session, self._api_base, endpoint, json, **req_kwargs)

    def publish(self, topic: str, payload):
        try:
            send_topic = self._publish_topics[topic]
        except KeyError:
            raise ValueError("Incorrect topic") from None

        payload = json.dumps(payload)
        logger.debug("Publishing to topic '%s' with payload '%s'",
                     send_topic, payload)
        self._mqttc.publish(send_topic, payload=payload)

    def send_join_msg(self):
        self.publish(
            "control",
            {
                "type": "join",
                "participant": self.client_id,
                "session": self._session_id,
            },
        )

    def send_ready_msg(self):
        self.publish(
            "control",
            {
                "type": "ready",
                "participant": self._client_id,
                "session": self._session_id,
            },
        )

    @property
    def mqttc(self) -> mqtt.Client:
        return self._mqttc

    @property
    def client_id(self) -> int:
        return self._client_id

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def subscribe_topics(self) -> list[str]:
        return self._subscribe_topics


class HansPlatform:
    def __init__(
        self,
        client_name: str,
        agent_manager: AgentManager,
        *,
        hexagon_radius: float = 340
    ):

        self.client_name = client_name
        self._hexagon_radius = hexagon_radius
        self._connected = False

        self._api_wrapper: Optional[_HansApiWrapper] = None

        self._agent_manager = agent_manager

        self._current_question: Optional[Question] = None

    def connect(
        self,
        api_host: str,
        broker_host: str,
        broker_port: int = 9001,
        session_id: int = 1
    ):
        logger.info("Connecting to MQTT broker at %s:%s",
                    broker_host, broker_port
                    )

        self._api_wrapper = _HansApiWrapper.from_connection(
            self.client_name,
            session_id,
            api_host
        )
        mqttc = self._api_wrapper.mqttc

        mqttc.on_connect = self._on_connect
        mqttc.on_message = self._on_message

        self._connected = True
        mqttc.connect(broker_host, broker_port)

    def listen(self, *args, **kwargs):
        """Listen to incoming MQTT requests and start the game loop thread"""

        logger.info("Start listening for incoming MQTT packets")

        # If there is an error inside the thread, the client will disconnect from the
        # platform
        self._agent_manager.start_thread(
            self.client_name, lambda: self.disconnect())
        self._api_wrapper.mqttc.loop_forever(*args, **kwargs)
        if self._agent_manager.exc_info is not None:
            _raise_from_exc_info(self._agent_manager.exc_info)

    def disconnect(self):
        if not self._connected:
            return
        self._connected = False

        logger.info("Disconnecting from MQTT broker")
        self._api_wrapper.disconnect()
        if self._agent_manager.is_thread_alive():
            self._agent_manager.quit()

    def _on_connect(self, client, userdata, flags, rc):
        mqttc = self._api_wrapper.mqttc

        for topic in self._api_wrapper.subscribe_topics:
            logger.debug("Subscribing to %s", topic)
            mqttc.subscribe(topic)

        self._api_wrapper.send_join_msg()

        # set the question once the bot has connected to the mqtt broker
        res = self._api_wrapper.get(
            f"session/{self._api_wrapper.session_id}").json()
        self._set_current_question(res["collection_id"], res["question_id"])

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

            # the backend is the one who publishes events to the topic under the 0 id. Right
            # now, its update messages can be safely ignored for
            if participant_id == 0:
                return

            self._agent_manager.on_changed_position(
                participant_id, np.array(payload["data"]["position"])
            )

    def _handle_control_msgs(self, payload):
        if payload["type"] == "setup":
            self._set_current_question(
                payload["collection_id"], payload["question_id"])
        elif payload["type"] == "start":
            if self._current_question is None:
                raise CannotStartRoundException(
                    "The question has not been set"
                )

            participants = self._api_wrapper.get_all_participants()
            answer_positions = utils.calculate_answer_points(
                len(self._current_question.answers), self._hexagon_radius
            )

            new_round = Round(self._current_question,
                              payload["duration"], participants,
                              answer_positions, self._hexagon_radius)

            hans_client = HansClient(
                self._api_wrapper, PositionCodec(answer_positions)
            )

            self._agent_manager.start_session(new_round, hans_client)
            logger.info("The round has started")
        elif payload["type"] == "stop":
            self._agent_manager.finish_session()
            logger.info("The round has stopped")

    def _set_current_question(self, collection_id, question_id):
        self._current_question = self._api_wrapper.get_question_from_id(
            collection_id, question_id
        )
        self._api_wrapper.send_ready_msg()
        logger.info("The question has changed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
