from typing import Optional
import json
from PIL import Image
from io import BytesIO
from dataclasses import dataclass
from typing import List
import threading
import time

import requests
import paho.mqtt.client as mqtt

from exceptions import CannotStartRoundException


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
        response = session.get(
            f"{api_base}/question/{collection_id}/{question_id}"
        )
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

        return cls(id=question_id,
                   collection_id=collection_id,
                   prompt=data["prompt"],
                   answers=data["answers"],
                   img=img
                )

@dataclass
class Round:
    question: Question
    duration: int


# TODO: where should state go???

class Brain:

    def __init__(self, round):
        self.round = round

    def render(self, hans_client):
        """ This is where all code in which message packets are sent must go. 
        Right now, those packets only contain position information.

        The rate at which this method is called may vary. That will depend on the work
        done by self.udpate() 
        """
        
    # NOTE: should round be a paramater to update or an attribute?
    def update(self, delta):
        """ All code related to the calculation of the next position.
        
        The rate which this method is called is guaranteed to be constant so delta is fixed.
        """

# https://gafferongames.com/post/fix_your_timestep/
# https://gameprogrammingpatterns.com/game-loop.html

class GameLoopThread(threading.Thread):

    def __init__(self, brain_cls, fps=10, tps=20, brain_kwargs={}):
        super().__init__()

        self._brain_cls = brain_cls
        self._brain_kwargs = brain_kwargs

        # This will be set by HansClient
        self._hans_client = None

        self._max_frame_time = 1 / fps
        self._delta = 1 / tps

        # A boolean flag would have been enough but that assumes that
        # we are working with a Python implementation which uses the GIL.
        # Thererfore, event is more robust
        self._current_loop_quit = threading.Event()
        self._continue = threading.Event()
        self._thread_quit = threading.Event()

        # These will be set when a new loop is called
        self._current_brain = None
    
    def stop(self):
        """ Stops and clears the currently executing game loop. This implies that the
         state of the game loop is lost (but not the one of the brain). 
         This method must be called before calling new loop. """

        self._current_loop_quit.set()
        self._continue.clear()
    
    def new_loop(self, round):
        """ Creates a new game loop in this thread for the given round. 
        A new instance of the brain will be created.
        You have to call this method in order for a game loop to run. Otherwise, the thread
        will stay idle.
        """

        self._current_brain = self._brain_cls(round=round, **self._brain_kwargs)

        self._continue.set()
        self._current_loop_quit.clear()

    def quit(self):
        """ Exits the thread """

        self._thread_quit.set()
        self._current_loop_quit.set()
        self._continue.set()
    
    def run(self):
        while self._continue.wait():
            if self._thread_quit.is_set():
                break

            self._run_loop()

    def _run_loop(self):
        current_time = time.monotonic()
        accumulator = 0
        
        while not self._current_loop_quit.is_set():
            new_time = time.monotonic()
            frame_time = new_time - current_time
            current_time = new_time

            accumulator += frame_time
            while accumulator >= self._delta and not self._current_loop_quit.is_set():
                self._current_brain.update(self._delta)
                accumulator -= self._delta
        
            if not self._current_loop_quit.is_set():
                self._current_brain.render(self._hans_client)
            
            remaining_frame_time = self._max_frame_time - (time.monotonic() - current_time)
            if remaining_frame_time > 0:
                # We employ Event instead of time.sleep() because in that way,
                # if stop() or quit() is called then this thread can exit as soon as
                # the scheduler decides
                self._current_loop_quit.wait(remaining_frame_time)

class HansClient:

    def __init__(self, name: str, game_loop, session_id: int=1):
        self.name = name

        self._host = ""
        self._api_base = ""
        self._port = 3000
        self._broker_port = 9001
        self.client_id = None

        self._session_id = str(session_id)
        self._session_topic = f"swarm/session/{self._session_id}"

        self._session = None

        self._mqttc = mqtt.Client(transport="websockets", clean_session=True)
        self._mqttc.on_connect = self._on_connect
        self._mqttc.on_message = self._on_message

        # the last question selected by the admin
        self._current_question = None

        self._game_loop = game_loop
        
        # TODO: think if there is an alternative way to provide the sending functionality
        # to the game loop. We only need a reference to an instance in this class because
        # we need a way to send messages from the game loop and right now HansClient is 
        # the only one which can do that. But it also does other things
        self._game_loop._hans_client = self

    def connect(self, host: str, port: Optional[int]=None, broker_port: Optional[int]=None):
        self._host = host
        self._port = port or self._port
        self._broker_port = broker_port or self._broker_port

        self._api_base = f"http://{self._host}:{self._port}/api/"

        self._session = requests.Session()

        # Send login request
        req = self._session.post(
            f"{self._api_base}/session/{self._session_id}/participants", 
            json={
                "user": self.name
            }
        )

        if req.content == b"Participant already joined session":
            raise ValueError(f"There already exists an user with the name {self.name}")
        elif req.content == b"Session not found":
            raise ValueError(f"There does not exist session with id {self._session_id}")

        self.client_id = str(req.json()["id"])
        self._control_topic = f"{self._session_topic}/control/{self.client_id}"
        print(f"Control topic: {self._control_topic}")

        self._mqttc.connect(self._host, self._broker_port)
    
    def listen(self, *args, **kwargs):
        """ Listen to incoming MQTT requests and start the game loop thread"""
        self._game_loop.start()
        self._mqttc.loop_forever(*args, **kwargs)
    
    def _send_ready_msg(self):
        self._mqttc.publish(self._control_topic, payload=json.dumps({
            "type": "ready",
            "participant": self.client_id,
            "session": self._session_id
        }))
    
    def _on_connect(self, client, userdata, flags, rc):
        self._mqttc.subscribe(f"{self._session_topic}/control/#")
        self._mqttc.subscribe(f"{self._session_topic}/updates/#")

        # This must be sent so that the client's name appears to the admin in the text area
        # where all connected clients are shown
        self._publish(self._control_topic, {
            "type": "join",
            "participant": self.client_id,
            "session": self._session_id
        })

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
            self._current_question = Question.from_setup_msg(self._session, 
                                                                self._api_base,
                                                                payload)

            print(f"Changed question to {self._current_question}")

            # I think this is to inform that everything went right
            self._publish(self._control_topic, {
                "type": "ready",
                "participant": self.client_id,
                "session": self._session_id
            })
        elif payload["type"] == "start":
            print(f"From start: {payload}")
            if self._current_question is None:
                raise CannotStartRoundException("The question has not been set")
            self._game_loop.new_loop(Round(self._current_question, payload["duration"]))
        elif payload["type"] == "stop":
            self._game_loop.stop()
    
    def _publish(self, topic, payload):
        self._mqttc.publish(topic, payload=json.dumps(payload))

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session is not None:
            self._session.close()
