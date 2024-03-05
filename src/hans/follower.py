from __future__ import annotations

import threading
from typing import Any, Callable, cast, TYPE_CHECKING
from numpy import ndarray

import zmq

from .loop import Loop, GameLoop, GameLoopManager, LoopWithScheduler
from .state import State, StateSnapshot
from .coro import Scheduler
from .exceptions import FollowerException
from .thread_loop_manager import ThreadLoopManager

if TYPE_CHECKING:
    from .model import Round
    from .client import HansClient

# TODO: this exact definition is in leader. Refactor the code so that we don't duplicate
# it
JSONDict = dict[str, Any]


class Follower(LoopWithScheduler):

    def __init__(
        self,
        client: HansClient,
        round: Round,
        coro_scheduler: Scheduler,
        send_buffer: list[str]
    ):
        super().__init__(coro_scheduler)

        self.client = client
        self.round = round
        self.snapshot: StateSnapshot | None = None

        self._send_buffer = send_buffer

    def on_message_receive(self, data: str):
        """
        This method will be called for each message the Leader receives from a child agent.
        It will always be called after update() so there is no need to use synchronization
        mechanisms if the same value is being written here and in another method from Loop.
        """

    def send_msg(self, data: str):
        """Send 'data' to the leader"""

        self._send_buffer.append(data)


class DealerRouter:

    def __init__(self, context: zmq.Context, is_designated: bool = False):
        self._socket = cast(zmq.Socket, context.socket(zmq.DEALER))
        self.is_designated = is_designated

    def connect(self, zmq_endpoint: str):
        self._socket.connect(zmq_endpoint)

    def send_json(self, data: JSONDict):
        self._socket.send_json(data)

    def recv_json(self, flags: int = 0) -> JSONDict:
        return self._socket.recv_json(flags)

    def designated_send_json(self, data: JSONDict):
        """Send the data only if the is_designated data is set to true"""
        if self.is_designated:
            self.send_json(data)


class _FollowerWrapper(Loop):

    def __init__(
        self,
        follower: Follower,
        state: State,
        socket: DealerRouter,
        send_buffer: list[str]
    ):
        self._follower = follower
        self._socket = socket
        self._send_buffer = send_buffer

        self.state = state

    def setup(self, **kwargs):
        self._follower.setup(**kwargs)

    def fixed_update(self, delta: float, sync_ratio: float):
        self._designated_send_position()
        self._follower.fixed_update(delta, sync_ratio)

    def update(self, delta: float):
        self._designated_send_position()
        self._follower.update(delta)
        messages = self._read_all_messages()
        for message in messages:
            self._follower.on_message_receive(message)
        for send_message in self._send_buffer:
            self._socket.send_json({
                "type": "agent_communication",
                "data": send_message
            })
        self._send_buffer = []

    def close(self):
        self._follower.close()

    def _read_all_messages(self) -> list[str]:
        messages = []
        while True:
            try:
                packet = self._socket.recv_json(zmq.DONTWAIT)
                if "type" not in packet or "data" not in packet:
                    continue

                if packet["type"] != "agent_communication":
                    continue

                messages.append(packet["data"])
            except zmq.Again:
                return messages

    def _designated_send_position(self):
        self._socket.designated_send_json({
            "type": "position",
            "data": list(map(list, self.state.get_snapshot().all_positions))
        })


class FollowerManager(ThreadLoopManager):

    def __init__(
        self,
        follower_cls: type[Follower],
        follower_kwargs={},
        game_loop_kwargs={},
        zmq_listen_addr: str = "ipc:///tmp/hansleader.ipc",
        zmq_context: zmq.Context | None = None
    ):
        """
        zmq_listen_addr:
            zmq endpoint used to communicate with the leader
        zmq_context:
            the zmq context from which the leader socket will be created. If the value is 'None'
            then a new context will be created. The ZMQ documentation states that there must
            be one context per process
        """
        zmq_context = zmq_context if zmq_context is not None else zmq.Context()
        self._zmq_listen_addr = zmq_listen_addr
        self._socket = DealerRouter(zmq_context)

        self._follower_cls = follower_cls
        self._game_loop_kwargs = game_loop_kwargs

        self._manager = GameLoopManager(follower_kwargs)
        self._thread: threading.Thread | None = None

    def start_session(self, round: Round, hans_client: HansClient):
        if not self._is_connected_to_leader:
            raise FollowerException(
                f"The agent {hans_client.name} is not connected to a leader"
            )

        self._socket.designated_send_json({
            "type": "control",
            "data": "start"
        })

        participant_ids = [
            participant.id for participant in round.participants]
        state = State(hans_client.pcodec, participant_ids, hans_client.id)

        send_buffer = []
        scheduler = Scheduler()

        follower = self._follower_cls(
            hans_client, round, scheduler, send_buffer
        )
        self._follower = _FollowerWrapper(
            follower,
            state,
            self._socket,
            send_buffer
        )

        game_loop = GameLoop(
            self._follower,
            scheduler,
            **self._game_loop_kwargs
        )

        self._manager.set_game_loop(game_loop)

    def start_thread(self, agent_name: str, exc_handler: Callable[[None], None] | None = None):
        self._manager.add_exc_handler(exc_handler)
        self._thread = threading.Thread(target=self._run, args=(agent_name,))
        self._thread.start()

    def _run(self, agent_name: str):
        """Called before the mqtt loop starts"""
        self._socket.connect(self._zmq_listen_addr)

        self._socket.send_json({
            "type": "connect",
            "data": {
                "name": agent_name
            }
        })
        packet = self._socket.recv_json()
        if packet["type"] != "success":
            raise FollowerException(
                f"Error while connecting to leader: {packet['data']}"
            )
        self._socket.is_designated = packet["data"]["is_designated"]
        self._is_connected_to_leader = True

        self._manager.run()

    def on_position_change(self, participant_id: int, position: ndarray):
        self._follower.state.update(participant_id, position)

    def quit(self):
        self._manager.quit()

    def finish_session(self):
        """Stops and removes the currently executing agent."""

        self._socket.designated_send_json({
            "type": "control",
            "data": "stop"
        })

        self._manager.stop()

    def add_exc_handler(self, exc_handler: Callable[[None], None]):
        """Sets the handler that will be called when there is an exception in the loop"""

        self._manager.add_exc_handler(exc_handler)

    def is_thread_alive(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def exc_info(self):
        return self._manager.exc_info
