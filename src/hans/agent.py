from __future__ import annotations

import threading
import numpy as np
from typing import Callable, TYPE_CHECKING

from .loop import Loop, GameLoop, GameLoopManager, LoopWithScheduler
from .state import State
from .coro import Scheduler
from .thread_loop_manager import ThreadLoopManager

if TYPE_CHECKING:
    from .client import HansClient
    from .state import StateSnapshot
    from .model import Round


class Agent(LoopWithScheduler):
    """All the logic to control a client must be included in a subclass
    inheriting from this one. It is not necessary to override the constructor
    """

    def __init__(self, round: Round, client: HansClient, coro_scheduler: Scheduler):
        super().__init__(coro_scheduler)

        self.round = round
        self.client = client

        # TODO: even though snapshot is None when the agent is created, when the client
        # uses it it is not possible that it has that value. If the user were using a typechecker
        # it will complain that this variable may be None, which from his perspective it won't be
        # possible if the calling code is working properly.
        # Think on some way of avoiding that inconvenient to the user.
        self.snapshot: StateSnapshot | None = None


class _AgentWrapper(Loop):
    """This is a wrapper for an Agent. It is necessary to get snapshots every time
    update() or fixed_update() is called"""

    def __init__(self, agent: Agent, state: State):
        self._agent = agent
        self.state = state

    def setup(self, **kwargs):
        self._agent.setup(**kwargs)

    def update(self, delta: float):
        # The time it takes get a snapshot is negible, so I don't think it is worth it
        # to add it to delta
        self._agent.snapshot = self.state.get_snapshot()
        self._agent.update(delta)

    def fixed_update(self, delta: float, sync_ratio: float):
        # The time it takes get a snapshot is negible, so I don't think it is worth it
        # to add it to delta
        self._agent.snapshot = self.state.get_snapshot()
        self._agent.fixed_update(delta, sync_ratio)

    def close(self):
        self._agent.close()


class AgentManager(ThreadLoopManager):
    """This class is in charge of running an agent when a session starts and stopping
    its execution when the session finishes."""

    def __init__(
        self,
        agent_cls: type[Agent],
        agent_kwargs={},
        game_loop_kwargs={}
    ):
        super().__init__()

        self._manager = GameLoopManager(agent_kwargs)

        self._agent_cls = agent_cls
        self._game_loop_kwargs = game_loop_kwargs

        self._thread: threading.Thread | None = None

        # All of these variables will be initialized when start_session is called
        # The agent which is being currently being executed
        self._agent: _AgentWrapper | None = None

    def start_session(self, round: Round, hans_client: HansClient):
        participant_ids = [
            participant.id for participant in round.participants]
        state = State(hans_client.pcodec, participant_ids, hans_client.id)

        coro_scheduler = Scheduler()

        agent = self._agent_cls(
            round=round,
            client=hans_client,
            coro_scheduler=coro_scheduler
        )

        self._agent = _AgentWrapper(agent, state)
        game_loop = GameLoop(
            self._agent,
            coro_scheduler,
            **self._game_loop_kwargs
        )

        self._manager.set_game_loop(game_loop)

    def start_thread(self, agent_name: str, exc_handler: Callable[[None], None] | None = None):
        self._manager.add_exc_handler(exc_handler)
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        self._manager.run()

    def quit(self):
        self._manager.quit()

    def on_changed_position(self, participant_id: str, position: np.ndarray):
        """Called every time a participant changes their position and a message is sent through
        the appropiate topic"""

        self._agent.state.update(participant_id, position)

    def finish_session(self):
        """Stops and removes the currently executing agent."""

        self._manager.stop()

    def is_thread_alive(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def exc_info(self):
        return self._manager.exc_info
