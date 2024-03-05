from __future__ import annotations

from typing import Any, Iterable, Callable, cast, TYPE_CHECKING
from dataclasses import dataclass, field
import json
import logging

import zmq
import numpy as np

from hans.coro import LoopCoroutine, Scheduler

from .loop import Loop, GameLoop, LoopWithScheduler

if TYPE_CHECKING:
    from .loop import Round

JSONDict = dict[str, Any]
logger = logging.getLogger(__name__)


class RouterSocket:

    def __init__(self, context: zmq.Context):
        self._socket = cast(zmq.Socket, context.socket(zmq.ROUTER))

    def bind(self, addr: str):
        """Set the address where the leader will listen for commands"""
        self._socket.bind(addr)

    def recv_json(self, flags: int = 0) -> tuple[bytes, JSONDict]:
        # TODO: add check for the case in which you cannot decode the message
        ident, data = self._socket.recv_multipart(flags)
        return ident, json.loads(data)

    def send_json(self, ident: bytes, msg: JSONDict, flags: int = 0):
        self.send(ident, json.dumps(msg).encode(), flags)

    def send_string(self, ident: bytes, msg: str, flags: int = 0):
        self.send(ident, msg.encode(), flags)

    def send(self, ident: bytes, data: bytes, flags: int = 0):
        self._socket.send_multipart([ident, data], flags)


# The data is a string instead of bytes because the packages are being serialized to JSON and
# the data attribute which appears here is an attribute of the serialized JSON object.

@dataclass
class SendMessage:
    agent_name: str
    data: str


class _SendMessageBuffer:

    def __init__(self):
        self._queue: list[SendMessage] = []

    def send_msg(self, agent_names: str | Iterable[str], data: str):
        if isinstance(agent_names, str):
            self._queue.append(SendMessage(agent_names, data))
        else:
            self._queue += [SendMessage(agent_name, data)
                            for agent_name in agent_names]

    def clear(self):
        self._queue = []

    def __iter__(self):
        return iter(self._queue)


class Leader(LoopWithScheduler):

    def __init__(
        self,
        round: Round,
        agents_names: list[str],
        send_buffer: _SendMessageBuffer,
        coro_scheduler: Scheduler
    ):
        super().__init__(coro_scheduler)

        self.round = round
        self.agent_names = agents_names
        # TODO: the len of this vector must be equal to the number of participants in the session (info
        # which is obtained from round)
        self.positions = np.zeros(4)
        self._send_buffer = send_buffer

    def on_message_received(self, agent_name: str, data: str):
        """
        This method will be called for each message the Leader receives from a child agent.
        It will always be called after update() so there is no need to use synchronization
        mechanisms if the same value is being written here and in another method from Loop.
        """

    def send_msg(self, agent_names: str | Iterable[str], data: str):
        """
        Send a message with payload 'data' to the set of agents identified by 'agent_names'
        """

        self._send_buffer.send_msg(agent_names, data)

    def broadcast(self, data: str):
        """
        Send a message with payload 'data' to all connected agents
        """

        self._send_buffer.send_msg(self.agent_names, data)


@dataclass
class ReceivedMessage:
    agent_name: str
    ty: str
    data: bytes


@dataclass
class LastMessages:
    # Because we are not trying to synchronize a complex physical simulation, we are only interested
    # in the most recent position of each player
    state: ReceivedMessage | None = None
    agent_communication: list[ReceivedMessage] = field(default_factory=list)
    # Right now, there is only one control message we can receive
    control: ReceivedMessage | None = None


class _LeaderWrapper(Loop):

    def __init__(
        self,
        leader: Leader,
        socket: RouterSocket,
        ident_name: IdentNameMap,
        send_buffer: _SendMessageBuffer,
    ):
        self._socket = socket
        self._leader = leader
        self._send_buffer = send_buffer

        self._ident_name = ident_name
        self._comm_messages: list[ReceivedMessage] = []

        # Called when the session has finished
        self.on_session_finish: Callable[[None], None] | None = None

    def setup(self, **kwargs):
        self._leader.setup(**kwargs)

    def fixed_update(self, delta: float, sync_ratio: float):
        last_messages = self._last_messages()
        self._process_last_messages(last_messages)
        self._leader.fixed_update(delta, sync_ratio)

    def update(self, delta: float):
        last_messages = self._last_messages()
        self._process_last_messages(last_messages)

        self._leader.update(delta)

        for msg in self._comm_messages:
            self._leader.on_message_received(msg.agent_name, msg.data)
        self._comm_messages = []

        for msg in self._send_buffer:
            self._socket.send_json(
                self._ident_name.get_ident(msg.agent_name), {
                    "type": "agent_communication",
                    "data": msg.data
                }
            )
        self._send_buffer.clear()

    def close(self):
        self._leader.close()

    def _process_last_messages(self, last_messages: LastMessages):
        if last_messages.control is not None:
            self.on_session_finish()

        if last_messages.state is not None:
            self._leader.positions = list(
                map(np.array, last_messages.state.data))

        self._comm_messages += last_messages.agent_communication

    def _last_messages(self) -> LastMessages:
        return self._multiplex_messages(self._read_all_messages())

    def _multiplex_messages(self, messages: list[ReceivedMessage]) -> LastMessages:
        last_messages = LastMessages()
        for message in messages:
            if message.ty == "position":
                last_messages.state = message
            elif message.ty == "agent_communication":
                last_messages.agent_communication.append(message)
            elif message.ty == "control":
                last_messages.control = message
        return last_messages

    def _read_all_messages(self) -> list[ReceivedMessage]:
        # TODO: think on a good condition which guarantees an upper bound on the amount of
        # time spent retrieving messages
        messages = []
        while True:
            try:
                ident, packet = self._socket.recv_json(zmq.DONTWAIT)
                if (
                    not self._ident_name.has_ident(ident) or
                    "type" not in packet or
                    "data" not in packet
                ):
                    continue

                message = ReceivedMessage(
                    self._ident_name.get_name(ident),
                    packet["type"],
                    packet["data"]
                )

                messages.append(message)

                if message.ty == "control" and message.data == "stop":
                    return messages

            except zmq.Again:
                return messages


class IdentNameMap:

    def __init__(self):
        self._name_to_ident: dict[str, bytes] = {}
        self._ident_to_name: dict[bytes, str] = {}

    def add(self, ident: bytes, name: str):
        self._name_to_ident[name] = ident
        self._ident_to_name[ident] = name

    def __len__(self):
        return len(self._name_to_ident)

    def names(self) -> list[str]:
        return list(self._name_to_ident.keys())

    def idents(self) -> list[str]:
        return list(self._name_to_ident.values())

    def get_name(self, ident: bytes) -> str:
        return self._ident_to_name[ident]

    def get_ident(self, name: str) -> bytes:
        return self._name_to_ident[name]

    def has_name(self, name: str) -> bool:
        return name in self._name_to_ident

    def has_ident(self, ident: bytes) -> bool:
        return ident in self._ident_to_name


class LeaderManager:

    def __init__(
        self,
        leader_cls: type[Leader],
        leader_kwargs={},
        game_loop_kwargs={},
        zmq_context: zmq.Context | None = None
    ):
        """
        zmq_context:
            the zmq context from which the leader socket will be created. If the value is 'None'
            then a new context will be created. The ZMQ documentation states that there must
            be one context per process
        """
        zmq_context = zmq_context if zmq_context is not None else zmq.Context()

        self._socket = RouterSocket(zmq_context)

        self._leader_cls = leader_cls
        self._leader_kwargs = leader_kwargs
        self._game_loop_kwargs = game_loop_kwargs

        self._ident_name = IdentNameMap()

    def bind(self, zmq_listen_addr: str = "ipc:///tmp/hansleader.ipc"):
        """Set the address where the leader will listen for commands. It is a zqm endpoint.
        By default, an endpoint with transport 'ipc' will be bound"""
        self._socket.bind(zmq_listen_addr)

    def start(self):
        while True:
            logger.info("Waiting until the round starts")

            round = self._wait_for_session()

            logger.info("The session has started")
            self.start_session(round)

    def _wait_for_session(self) -> Round:
        while True:
            ident, msg = self._socket.recv_json()

            if msg["type"] == "connect":
                name = msg["data"]["name"]
                if self._ident_name.has_name(name):
                    self._socket.send_json(ident, {
                        "type": "error",
                        "data": f"There already exists an agent with the name {name}"
                    })
                else:
                    logger.info("The agent '%s' has joined", name)
                    # Right now, the first agent which connects will be designated one
                    self._socket.send_json(ident, {
                        "type": "success",
                        "data": {
                            "is_designated": len(self._ident_name) == 0
                        }
                    })
                    self._ident_name.add(ident, name)
            elif msg["type"] == "control" and msg["data"] == "start":
                # TODO: Retrieve the static information of a session (i.e the round) only
                # from the designated agent. If that data is received from another agent
                # which is not the designated one, return an error
                break
            else:
                self._socket.send_json(ident, {
                    "type": "error",
                    "data": f"Unknown message type {msg['type']}"
                })

    def start_session(self, round: Round):
        send_buffer = _SendMessageBuffer()
        scheduler = Scheduler()

        leader = _LeaderWrapper(
            self._leader_cls(round, self._ident_name.names(),
                             send_buffer, scheduler),
            self._socket,
            self._ident_name,
            send_buffer
        )

        game_loop = GameLoop(leader, scheduler, **self._game_loop_kwargs)
        leader.on_session_finish = game_loop.signal_quit
        game_loop.run(**self._leader_kwargs)
