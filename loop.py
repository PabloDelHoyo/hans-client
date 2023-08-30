from __future__ import annotations

# https://gafferongames.com/post/fix_your_timestep/
# https://gameprogrammingpatterns.com/game-loop.html

# In those articles it is recommended to interpolete between two frames. For this application,
# I don't think it is worth the trouble. The interpolation solves two problems.
#   1. If the fps is greater than the tps, then the movement is stuttered. The interpolation
#      helps to make smoother. Therefore, this might be useful for the frontend altough, again,
#      it is not the trouble
#
#   2. Because the updating and "rendering" happen at different rates, there will usually be
#      the same number of updates between two renders (assume tps > fps ie. 60 and 50). However
#      the state we should draw is not the one which is indicated by the simulation but
#      the one of the few ms which separate the last update and the render. Aditionally,
#      these remains accumulate which have the effect of performing more simulations between
#      renders so it gives the impression that the simulation has run faster. This problem
#      can be avoided with lerp.

import threading
import time
import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from position_codec import PositionCodec
    from client import HansClient, Round


class State:
    def __init__(self, state: Dict[int, np.ndarray], client_id: int):
        self._state = state
        self._client_id = client_id

    @property
    def all_positions(self) -> Dict[int, np.ndarray]:
        return self._state

    @property
    def other_participants_positions(self) -> Dict[int, np.ndarray]:
        return {
            participant_id: position
            for participant_id, position in self._state.items()
            if participant_id != self._client_id
        }


class StateHandler:
    """Class to update the global state given individual delta updates. In a real
    game, we would poll the I/O system to get the latest network messages"""

    def __init__(
        self, pcodec: PositionCodec, participant_ids: List[int], client_id: int
    ):
        self._client_id = client_id
        self._pcodec = pcodec

        # All participants start at the postition (0, 0)
        self._state = {
            participant_id: np.zeros(2) for participant_id in participant_ids
        }

        # We don't want the state to be updated when it is being copy in "get_state"
        self._lock = threading.Lock()

    def update_state(self, participant_id: int, position: np.ndarray):
        with self._lock:
            self._state[participant_id] = self._pcodec.decode(position)

    def get_state(self) -> State:
        """Copy the current state. This is safe to call when there are multiple threads"""

        with self._lock:
            return State(
                {
                    participant_id: position.copy()
                    for participant_id, position in self._state.items()
                },
                self._client_id,
            )


class Loop:
    def __init__(self, round):
        self.round = round

    def render(self, hans_client: HansClient, sync_ratio: float):
        """This is where all code in which message packets are sent must go.
        Right now, those packets only contain position information.

        The rate at which this method is called may vary. That will depend on the work
        done by self.udpate()
        """

    def update(self, state: State, delta: float):
        """All code related to the calculation of the next position.

        The rate which this method is called is guaranteed to be constant so delta is fixed.
        """


class LoopThread(threading.Thread):
    def __init__(self, loop_cls, fps=20, tps=20, loop_kwargs={}):
        super().__init__()

        self.fps = fps
        self.tps = tps

        self._loop_cls = loop_cls
        self._loop_kwargs = loop_kwargs

        self._current_hans_client = None

        self._max_frame_time = 1 / fps
        self._delta = 1 / tps

        # A boolean flag would have been enough but that assumes that
        # we are working with a Python implementation which uses the GIL.
        # Thererfore, event is more robust
        self._current_loop_quit = threading.Event()
        self._continue = threading.Event()
        self._thread_quit = threading.Event()

        # These will be set when a new loop is called
        self._current_loop: Optional[Loop] = None

        self._current_state_handler: Optional[StateHandler] = None

    def stop(self):
        """Stops and clears the currently executing loop. This implies that the
        state of the loop is lost (but not the one of the Loop instance).
        This method must be called before calling new loop."""

        self._current_loop_quit.set()
        self._continue.clear()

    def new_loop(self, round: Round, hans_client: HansClient):
        """Creates a new game loop in this thread for the given round.
        A new instance of the loop will be created.
        You have to call this method in order for a game loop to run. Otherwise, the thread
        will stay idle.
        """

        self._current_loop = self._loop_cls(round=round, **self._loop_kwargs)
        self._current_hans_client = hans_client

        participant_ids = [participant.id for participant in round.participants]
        self._current_state_handler = StateHandler(
            hans_client.pcodec, participant_ids, hans_client.id
        )

        self._continue.set()
        self._current_loop_quit.clear()

    def quit(self):
        """Exits the thread"""

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
            # In the article, the frame time is upper bounded (0.25ms). I think
            # this is to avoid "the spiral of hell" introduced in the article

            while accumulator >= self._delta and not self._current_loop_quit.is_set():
                state = self._current_state_handler.get_state()
                self._current_loop.update(state, self._delta)
                accumulator -= self._delta

            if not self._current_loop_quit.is_set():
                self._current_loop.render(
                    self._current_hans_client, accumulator / self._delta
                )

            remaining_frame_time = self._max_frame_time - (
                time.monotonic() - current_time
            )
            if remaining_frame_time > 0:
                # We employ Event instead of time.sleep() because in that way,
                # if stop() or quit() is called then this thread can exit as soon as
                # the scheduler decides
                self._current_loop_quit.wait(remaining_frame_time)

    def on_changed_position(self, participant_id: int, data):
        """Called every time a participant changes their position and a message is sent through
        the appropiate topic"""

        # the backend is the one who publishes events to the topic under the 0 id. Right
        # now, its update messages can be safely ignored for
        if participant_id == 0:
            return

        position = np.array(data["position"])
        self._current_state_handler.update_state(participant_id, position)
