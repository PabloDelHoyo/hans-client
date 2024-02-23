from __future__ import annotations

import numpy as np

from hans import HansPlatform, Agent, AgentManager
import hans.utils

from typing import Optional

NAME = "MoveToPopular"

HOST = "127.0.0.1"
API_HOST = f"http://{HOST}:8080"
MQTT_PORT = 9001


class MoveToMostPopular(Agent):

    def setup(self, speed, max_dist_popular, target_min_dist):
        self.speed = speed
        self.max_dist_popular = max_dist_popular
        self.target_min_dist = target_min_dist

        self.position = np.zeros(2)

    def update(self, delta: float):
        popular_answer = self._get_popular_answer(
            self.snapshot.other_positions, 1
        )
        if popular_answer is None:
            return

        target = self.round.answer_positions[popular_answer]

        if hans.utils.distance(self.position, target) > self.target_min_dist:
            direction = target - self.position
            unit = direction / np.linalg.norm(direction)
            self.position += unit * self.speed * delta

        self.client.send_position(self.position)

    def _get_popular_answer(
        self, positions: np.ndarray, threshold: int
    ) -> Optional[int]:
        dists = hans.utils.distance(
            self.round.answer_positions, np.expand_dims(positions, axis=1)
        )

        # Return the index of the closest answer for each participant
        closest_idxs = dists.argmin(axis=-1)

        # Return a count of the for each index
        values, counts = np.unique(closest_idxs, return_counts=True)

        count_argmax = counts.argmax()

        # We could do say that the most popular answer is the one chosen by the most
        # people or the one chosen by more than half of the participants
        # (i.e (len(self.round.participants) - 1) / 2)

        if counts[count_argmax] <= threshold:
            return None

        dist_popular = dists[(values[count_argmax] == closest_idxs).nonzero()][
            :, values[count_argmax]
        ]

        return (
            values[count_argmax]
            if np.all(dist_popular < self.max_dist_popular)
            else None
        )


def main():
    move_to_popular_manager = AgentManager(
        MoveToMostPopular,
        agent_kwargs=dict(
            speed=150,
            max_dist_popular=230,
            target_min_dist=30
        )
    )

    with HansPlatform(NAME, move_to_popular_manager) as platform:
        platform.connect(API_HOST, HOST, MQTT_PORT)
        platform.listen()


if __name__ == "__main__":
    main()
