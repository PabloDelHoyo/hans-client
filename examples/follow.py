from __future__ import annotations

from collections import deque

import numpy as np
from hans import HansPlatform, Agent, AgentManager

from typing import TYPE_CHECKING

from hans.state import StateSnapshot

if TYPE_CHECKING:
    from hans.state import StateSnapshot

NAME = "Follow"

HOST = "127.0.0.1"
API_PORT = 3000
MQTT_PORT = 9001


class Follow(Agent):

    def setup(self, lag, follow_idx=0):
        self.lag = lag
        self.follow_idx = follow_idx

        self.position = np.zeros(2)
        self.queue = deque()
        self.counter = 0

    def update(self, snapshot: StateSnapshot, delta: float):
        self.queue.append((snapshot.other_positions[self.follow_idx], self.counter))

        pos, timestamp = self.queue[0]
        timestamp += self.lag

        if self.counter > timestamp:
            self.position = pos
            self.queue.popleft()

        self.counter += delta

        self.client.send_position(self.position)


def main():
    follow_manager = AgentManager(
        Follow,
        agent_kwargs=dict(
            lag=0.5,
            follow_idx=0
        )
    )

    with HansPlatform(NAME, follow_manager) as platform:
        platform.connect(HOST, API_PORT, MQTT_PORT)
        platform.listen()


if __name__ == "__main__":
    main()