from __future__ import annotations

from dataclasses import dataclass
from typing import Coroutine
import types
import logging
import time

from .priority_queue import PriorityQueue

logger = logging.getLogger(__name__)

LoopCoroutine = Coroutine[float, None, None]


@dataclass
class WaitTask:
    coro: Coroutine[float, None, None]
    until: float

    @classmethod
    def from_sleep_time(cls, coro: LoopCoroutine, seconds: float):
        return cls(coro, time.monotonic() + seconds)


@types.coroutine
def sleep(seconds: float) -> LoopCoroutine:
    yield float(seconds)


async def next_update() -> LoopCoroutine:
    await sleep(0)


# TODO: add support for chaining coroutines.
class Scheduler:
    def __init__(self):
        self._tasks = PriorityQueue(keys=(lambda task: task.until,))

    def add_task(self, task: WaitTask):
        logger.debug("Coroutine %s has been scheduled", task.coro)
        self._tasks.put(task)

    def step(self):
        if len(self._tasks) == 0:
            return

        next_tasks = self._get_next_tasks()
        for task in next_tasks:
            try:
                sleep_time = task.coro.send(None)
                if not isinstance(sleep_time, float):
                    raise ValueError(
                        f"Awaited on incorrect value. Expected 'float' but found '{type(sleep_time)}'"
                    )
                self._tasks.put(WaitTask.from_sleep_time(task.coro, sleep_time))
            except StopIteration:
                logger.debug("Coroutine %s has finished", task.coro)

    def _get_next_tasks(self) -> list[WaitTask]:
        next_tasks = []
        current = time.monotonic()

        while len(self._tasks) > 0 and current > self._tasks.peek().until:
            next_tasks.append(self._tasks.pop())

        return next_tasks
