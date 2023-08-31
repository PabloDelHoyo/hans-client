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
from typing import Optional, Callable, TYPE_CHECKING

from state import State

if TYPE_CHECKING:
    from sys import OptExcInfo

    from state import StateSnapshot
    from client import HansClient, Round




class Loop:
    def __init__(self, round):
        self.round = round

    def render(self, hans_client: HansClient, sync_ratio: float):
        """This is where all code in which message packets are sent must go.
        Right now, those packets only contain position information.

        The rate at which this method is called may vary. That will depend on the work
        done by self.udpate()
        """

    def update(self, snapshot: StateSnapshot, delta: float):
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

        self._current_state: Optional[State] = None

        # Called when an exception happens
        self._exc_handler: Optional[Callable[[None], None]] = None

        # Stores the exception info in case one is reaised
        self.exc_info: OptExcInfo = None

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
        self._current_state= State(
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
        try:
            while self._continue.wait():
                if self._thread_quit.is_set():
                    break
                self._run_loop()
        except Exception:
            import sys

            self.exc_info = sys.exc_info()
            self._exc_handler()

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
                snapshot = self._current_state.get_snapshot()
                self._current_loop.update(snapshot, self._delta)
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
        self._current_state.update(participant_id, position)

    def add_exc_handler(self, exc_handler: Callable[[None], None]):
        """Sets the handler that will be called when there is an exception in the loop"""

        self._exc_handler = exc_handler
