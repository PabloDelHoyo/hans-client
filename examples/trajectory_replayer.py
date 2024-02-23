from __future__ import annotations

import logging

import numpy as np

from hans import HansPlatform, Agent, AgentManager
from hans.trajectories import Trajectory, TrajectoryGenerator


NAME = "Trajectory Replayer"

HOST = "127.0.0.1"
API_HOST = f"http://{HOST}:8080"
MQTT_PORT = 9001

# IMPORTANT: Generate two trajectories in the Hans platform and write
# their paths. You can also generate one path and write the same
# path for both constants

FIRST_TRAJECTORY_PATH = "path/to/first_trajectory.txt"
SECOND_TRAJECTORY_PATH = "path/to/second_trajectory.txt"


def get_default_handler():
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


class TrajectoryReplayer(Agent):

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

        self.point_generator = TrajectoryGenerator(
            self.round.radius, self.round.answer_positions
        )

        self.position = np.array([230.0, -100.0])

        self.point_generator.set_trajectory(
            start=self.position,
            end=np.array([-50, -100]),
            trajectory=first_trajectory,
            duration=duration,
        )

        self.duration = duration
        self.second_trajectory = second_trajectory
        self.change_after_seconds = change_after_seconds

        self.counter = 0
        self.has_changed = False

    def update(self, delta: float):
        if not self.has_changed and self.counter > self.change_after_seconds:
            self.point_generator.set_trajectory(
                start=self.position,
                end=np.array([200, 50]),
                trajectory=self.second_trajectory,
                duration=self.duration,
            )
            self.has_changed = True
            self.counter = 0

        self.position = self.point_generator.step(delta)
        self.counter += delta

        self.client.send_position(self.position)


def main():
    first_trajectory = Trajectory.from_file(FIRST_TRAJECTORY_PATH)
    second_trajectory = Trajectory.from_file(SECOND_TRAJECTORY_PATH)

    trajectory_replayer_manager = AgentManager(
        TrajectoryReplayer,
        agent_kwargs=dict(
            first_trajectory=first_trajectory,
            second_trajectory=second_trajectory,
            duration=5,
            change_after_seconds=7,
        ),
    )

    with HansPlatform(NAME, trajectory_replayer_manager) as platform:
        platform.connect(API_HOST, HOST, MQTT_PORT)
        platform.listen()


if __name__ == "__main__":
    main()
