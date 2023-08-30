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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client import HansClient, Round

class Loop:
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

class LoopThread(threading.Thread):
    def __init__(self, brain_cls, fps=20, tps=20, brain_kwargs={}):
        super().__init__()

        self.fps = fps
        self.tps = tps

        self._brain_cls = brain_cls
        self._brain_kwargs = brain_kwargs

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
        self._current_brain = None

    def stop(self):
        """Stops and clears the currently executing game loop. This implies that the
        state of the game loop is lost (but not the one of the brain).
        This method must be called before calling new loop."""

        self._current_loop_quit.set()
        self._continue.clear()

    def new_loop(self, round: Round, hans_client: HansClient):
        """Creates a new game loop in this thread for the given round.
        A new instance of the brain will be created.
        You have to call this method in order for a game loop to run. Otherwise, the thread
        will stay idle.
        """

        self._current_brain = self._brain_cls(round=round, **self._brain_kwargs)
        self._current_hans_client = hans_client

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
            # this is to avoid
            while accumulator >= self._delta and not self._current_loop_quit.is_set():
                self._current_brain.update(self._delta)
                accumulator -= self._delta

            if not self._current_loop_quit.is_set():
                self._current_brain.render(
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
