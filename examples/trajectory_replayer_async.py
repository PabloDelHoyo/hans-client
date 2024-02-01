from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from hans import HansPlatform, Loop, LoopThread
import hans.trajectories
import hans.coro
from hans.trajectories import Trajectory, TrajectoryGenerator

if TYPE_CHECKING:
    from hans.state import StateSnapshot

NAME = "Trajectory Replayer Async"

HOST = ""
API_PORT = 3000
MQTT_PORT = 9001

# IMPORTANT: Generate two trajectories in the Hans platform and write
# their paths. You can also generate one path and write the same
# path for both constants

FIRST_TRAJECTORY_PATH = "first.txt"
SECOND_TRAJECTORY_PATH = "second.txt"


def get_default_handler():
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    return handler


def configure_logger(name, level, handler=None):
    if not handler:
        handler = get_default_handler()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


configure_logger(
    "hans",
    logging.INFO,
)


class TrajectoryReplayer(Loop):
    def setup(
        self,
        first_trajectory: Trajectory,
        second_trajectory: Trajectory,
        duration: float,
        change_after_seconds: float,
    ):
        """
        first_trajectory: the first trajectory which will be followed
        second_trajectory: the second trajectory which will be followed
        duration: the time it takes replay both trajectories
        change_after_seconds: the amount of time elapsed before the trajectory is changed
        to 'second_trajectory'
        """

        self.position = np.array([230.0, -100.0])

        self.point_generator = TrajectoryGenerator(
            self.round.radius, self.round.answer_positions
        )

        self.point_generator.set_trajectory(
            start=self.position,
            end=np.array([-50, -100]),
            trajectory=first_trajectory,
            time_multiplier=hans.trajectories.get_factor_from_time(
                duration, first_trajectory
            ),
        )

        # Schedules the coroutine so that it is runned after 'change_after_seconds'
        self.start_coroutine(
            self.change_direction(change_after_seconds, second_trajectory, duration),
            change_after_seconds,
        )

        self.first_trajectory = first_trajectory

    def update(self, snapshot: StateSnapshot, delta: float):
        self.position = self.point_generator.step(delta)
        self.client.send_position(self.position)

    async def change_direction(self, change_after, second_trajectory, duration):
        self.point_generator.set_trajectory(
            start=self.position,
            end=np.array([-200, 50]),
            trajectory=second_trajectory,
            time_multiplier=hans.trajectories.get_factor_from_time(
                duration, second_trajectory
            ),
        )

        # Wait half the time we waited before and change the trajectory
        # and the destination
        await hans.coro.sleep(10)

        self.point_generator.set_trajectory(
            start=self.position,
            end=np.array([-50, -100.0]),
            trajectory=self.first_trajectory,
            time_multiplier=hans.trajectories.get_factor_from_time(
                duration, self.first_trajectory
            ),
        )


def main():
    first_trajectory = Trajectory.from_file(FIRST_TRAJECTORY_PATH)
    second_trajectory = Trajectory.from_file(SECOND_TRAJECTORY_PATH)

    trajectory_replayer_loop = LoopThread(
        TrajectoryReplayer,
        loop_kwargs=dict(
            first_trajectory=first_trajectory,
            second_trajectory=second_trajectory,
            duration=4,
            change_after_seconds=5,
        ),
    )

    with HansPlatform(NAME, trajectory_replayer_loop) as platform:
        platform.connect(HOST)
        platform.listen()


if __name__ == "__main__":
    main()
