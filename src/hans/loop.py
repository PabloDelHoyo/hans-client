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
from typing import Optional, Callable, Coroutine, TYPE_CHECKING

from .state import State
from .coro import Scheduler, WaitTask, LoopCoroutine

if TYPE_CHECKING:
    from sys import OptExcInfo

    from .state import StateSnapshot
    from .client import HansClient, Round


class Loop:
    """All the logic to control a client must be included in a subclass
    inheriting from this one. It is not necessary to override the constructor
    """

    def __init__(self, round: Round, client: HansClient, coro_scheduler: Scheduler):
        self.round = round
        self.client = client

        self._coro_scheduler = coro_scheduler

    def start_coroutine(self, coro: LoopCoroutine, after: float = 0):
        """Schedules a coroutine after 'after' seconds"""

        self._coro_scheduler.add_task(WaitTask.from_sleep_time(coro, after))

    def setup(self, **kwargs):
        """This is where all initialization code should go. Positional arguments
        are not allowed"""

    def update(self, snapshot: StateSnapshot, delta: float):
        """This method will be tried to be called at a fixed rate determined by FPS
        but that is not guaranteed.  That depends on the work being done by the method and
        external factors like system load. Therefore, delta time will vary but it will
        never exceed max_delta_time

        This is the recommended place to send the agent position.
        """

    def fixed_update(self, snapshot: StateSnapshot, delta: float, sync_ratio: float):
        """This method will be called at a fixed rate, determined by TPS.
        The rate at which this method is called is guaranteed to be constant so delta is fixed.
        This method is very useful for calculation which require a fixed timestep to work
        properly.
        """

    def close(self):
        """Called when a question finishes. It is guaranted no update() nor render() calls
        will happen after this method is called"""


class LoopThread(threading.Thread):
    """This represents the thread which will run the game loop

    loop_cls: Class inhereting from Loop whose method will be called according to the logic of a game
    loop
    fps: maximum number of times the method Loop.update() will be called in a second
    tps: number of time Loop.fixed_update() will be called in a second
    max_delta_time: upper bound for deltatime. This has the following advantages
      (https://docs.unity3d.com/ScriptReference/Time-maximumDeltaTime.html)
        - The number of fixed updates is bounded, which is necessary in those situations where
        there are hitches and we may fall into the spiral of hell
        - You have more guarantees over the varying delta time

        max_delta_time will always be as large as the time it takes a fixed update to run. If it
        were not the case, the upper bound would not longer be max_delta_time

    """

    def __init__(
        self,
        loop_cls: type[Loop],
        fps=20,
        tps=20,
        max_delta_time=0.3333,
        loop_kwargs={},
    ):
        super().__init__()

        self.fps = fps
        self.tps = tps
        self.max_delta_time = max_delta_time

        self._loop_cls = loop_cls
        self._loop_kwargs = loop_kwargs

        self._frame_time = 1 / fps
        self._fixed_delta = 1 / tps

        # max_delta_time has to be as large as fixed delta
        self._max_delta_time = max(max_delta_time, self._fixed_delta)

        # signals that the currently running loop must stop
        self._current_loop_quit = threading.Event()

        # blocks the thread until a new loop is created
        self._continue = threading.Event()

        # signals whether this thread should be killed
        self._thread_quit = threading.Event()

        # to avoid race conditions when calling close
        self._loop_finished = threading.Event()

        # These will be set when a new loop is called
        self._current_loop: Optional[Loop] = None

        self._current_state: Optional[State] = None

        # Called when an exception happens
        self._exc_handler: Optional[Callable[[None], None]] = None

        self._coro_scheduler = Scheduler()

        # Stores the exception info in case one is reaised
        self.exc_info: OptExcInfo = None

    def stop(self):
        """Stops and clears the currently executing loop. This implies that the
        state of the loop is lost (but not the one of the Loop instance).
        This method must be called before calling new loop."""

        self._current_loop_quit.set()
        self._continue.clear()

        self._loop_finished.wait()
        self._current_loop.close()

    def new_loop(self, round: Round, hans_client: HansClient):
        """Creates a new game loop in this thread for the given round.
        A new instance of the loop will be created.
        You have to call this method in order for a game loop to run. Otherwise, the thread
        will stay idle.
        """

        self._current_loop = self._loop_cls(
            round=round, client=hans_client, coro_scheduler=self._coro_scheduler
        )

        self._current_loop.setup(**self._loop_kwargs)

        participant_ids = [participant.id for participant in round.participants]
        self._current_state = State(hans_client.pcodec, participant_ids, hans_client.id)

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
                self._loop_finished.clear()
                if self._thread_quit.is_set():
                    break
                self._run_loop()
                self._loop_finished.set()
        except Exception:
            import sys

            self.exc_info = sys.exc_info()
            self._exc_handler()

    def _run_loop(self):
        current_time = time.monotonic()
        accumulator = 0

        while not self._current_loop_quit.is_set():
            # Calcuate the time the previous tick took
            new_time = time.monotonic()
            frame_time = new_time - current_time
            current_time = new_time

            # Bound the time time it took to avoid "spiral of hell"
            frame_time = min(frame_time, self._max_delta_time)
            accumulator += frame_time

            while (
                accumulator >= self._fixed_delta
                and not self._current_loop_quit.is_set()
            ):
                snapshot = self._current_state.get_snapshot()
                self._current_loop.fixed_update(
                    snapshot, self._fixed_delta, accumulator / self._fixed_delta
                )
                accumulator -= self._fixed_delta

            if not self._current_loop_quit.is_set():
                # We have to provide the actual time which have elapsed between update() and
                # update(). For that, we need to take into account the time it took to call
                # fixed_update()
                snapshot = self._current_state.get_snapshot()
                delta = frame_time + (time.monotonic() - current_time)

                # Do not forget to bound again
                delta = min(self._max_delta_time, delta)
                self._current_loop.update(snapshot, delta)

            # Unity schedules a coroutine other than "yield WaitForEndOfFrame" after
            # a call to update. Here we do the same
            self._coro_scheduler.step()

            remaining_frame_time = self._frame_time - (time.monotonic() - current_time)
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
