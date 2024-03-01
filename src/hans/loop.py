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

from .state import State
from .coro import Scheduler, WaitTask, LoopCoroutine

if TYPE_CHECKING:
    from sys import OptExcInfo

    from .state import StateSnapshot
    from .client import HansClient, Round


class Loop:

    def start_coroutine(self, coro: LoopCoroutine, after: float = 0):
        """Schedules a coroutine after 'after' seconds"""

    def setup(self, **kwargs):
        """This is where all initialization code should go. Positional arguments
        are not allowed"""

    def update(self, delta: float):
        """This method will be tried to be called at a fixed rate determined by FPS
        but that is not guaranteed.  That depends on the work being done by the method and
        external factors like system load. Therefore, delta time will vary but it will
        never exceed max_delta_time

        This is the recommended place to send the agent position.
        """

    def fixed_update(self, delta: float, sync_ratio: float):
        """This method will be called at a fixed rate, determined by TPS.
        The rate at which this method is called is guaranteed to be constant so delta is fixed.
        This method is very useful for calculation which require a fixed timestep to work
        properly.
        """

    def close(self):
        """Called when a question finishes. It is guaranted no update() nor fixed_update() calls
        will happen after this method is called"""


class LoopWithScheduler(Loop):

    def __init__(self, coro_scheduler: Scheduler):
        self._coro_scheduler = coro_scheduler

    def start_coroutine(self, coro: LoopCoroutine, after: float = 0):
        self._coro_scheduler.add_task(WaitTask.from_sleep_time(coro, after))


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


class GameLoop:
    """
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
        loop: Loop,
        coro_scheduler: Scheduler,
        fps=20,
        tps=20,
        max_delta_time=0.33333,
    ):
        self.fps = fps
        self.tps = tps
        self.frame_time = 1 / fps
        self.fixed_delta = 1 / tps

        # max_delta_time has to be as large as fixed delta
        self.max_delta_time = max(max_delta_time, self.fixed_delta)

        self._loop = loop
        self._coro_scheduler = coro_scheduler

        # Signals that the game loop must finish
        self._quit = threading.Event()

        # Set when the game loop has completely finished
        self._completely_finished = threading.Event()

    def run(self, **kwargs):
        if self._quit.is_set():
            raise ValueError(
                "The game loop has already finished and cannot be started again"
            )

        self._loop.setup(**kwargs)
        current_time = time.monotonic()
        accumulator = 0

        while not self._quit.is_set():
            # Calcuate the time the previous tick took
            new_time = time.monotonic()
            frame_time = new_time - current_time
            current_time = new_time

            # Bound the time time it took to avoid "spiral of hell"
            frame_time = min(frame_time, self.max_delta_time)
            accumulator += frame_time

            while accumulator >= self.fixed_delta:
                self._loop.fixed_update(
                    self.fixed_delta, accumulator / self.fixed_delta
                )
                accumulator -= self.fixed_delta

            # We have to provide the actual time which have elapsed between update() and
            # update(). For that, we need to take into account the time it took to call
            # fixed_update()
            delta = frame_time + (time.monotonic() - current_time)

            # Do not forget to bound again
            delta = min(self.max_delta_time, delta)
            self._loop.update(delta)

            # Unity schedules a coroutine other than "yield WaitForEndOfFrame" after
            # a call to update. Here we do the same
            self._coro_scheduler.step()

            remaining_frame_time = self.frame_time - \
                (time.monotonic() - current_time)
            if remaining_frame_time > 0:
                # We employ Event instead of time.sleep() because in that way,
                # if stop() or quit() is called then this thread can exit as soon as
                # the scheduler decides
                self._quit.wait(remaining_frame_time)

        self._loop.close()
        self._completely_finished.set()

    def has_finished(self):
        return self._completely_finished.is_set()

    # TODO: consdier making a method which just waits for the game loop to finish
    def quit(self, timeout: float | None = None):
        """Quits the game loop. If it is called several times, only the first call will
        call Loop.close(). Timeout is the maximum number of seconds to wait before this
        method returns. If it None, it does not return until the loop has completely finished"""

        self._quit.set()
        # Wait until the game loop has completely finished
        self._completely_finished.wait(timeout)

    def signal_quit(self):
        """Signals the game loop to finish. It does not wait until the loop completely finishes.
        This is the method that should be used in case you are controlling the game loop
        from the same thread where the loop is running. If 'quit()' were used instead, a deadlock would
        happen"""

        self._quit.set()


class GameLoopManager:
    """This class is in charge of creating and stopping a game loop. It is useful
    when you want to control a game loop from another thread"""

    def __init__(self, loop_kwargs={}):
        self._loop_kwargs = loop_kwargs

        self._game_loop: GameLoop | None = None

        self._manager_quit = threading.Event()
        self._game_loop_started = threading.Event()

        # Called when an exception happens
        self._exc_handler: Optional[Callable[[None], None]] = None
        # Stores the exception info in case one is reaised
        self.exc_info: OptExcInfo = None

    def set_game_loop(self, game_loop: Loop):
        self._game_loop = game_loop

        self._game_loop_started.set()

    def stop(self):
        self._game_loop_started.clear()
        if self._game_loop is not None:
            self._game_loop.quit()

    def quit(self):
        self._manager_quit.set()
        if self._game_loop is not None:
            self._game_loop.quit()
        self._game_loop_started.set()

    def run(self):
        if self._manager_quit.is_set():
            raise RuntimeError(
                "You cannot call 'run()' after you have called 'quit()'"
            )

        try:
            while not self._manager_quit.is_set():
                self._game_loop_started.wait()
                if self._manager_quit.is_set():
                    break
                self._game_loop.run(**self._loop_kwargs)
        except Exception:
            import sys

            self.exc_info = sys.exc_info()
            if self._exc_handler is not None:
                self._exc_handler()

    def add_exc_handler(self, exc_handler: Callable[[None], None]):
        """Sets the handler that will be called when there is an exception in the loop"""

        self._exc_handler = exc_handler
