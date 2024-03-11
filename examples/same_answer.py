import os
import random

import numpy as np

from hans import HansPlatform, AgentManager
from hans.leader import Leader
from hans.follower import Follower
from hans.trajectories import TrajectoryGenerator, Trajectory

HOST = ""
API_HOST = f"http://{HOST}:8080"
MQTT_PORT = 9001

def choose_random_trajectory_gen(path):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")
    
    trajectories_path = [
        os.path.join(path, filename) for filename in os.listdir(path)
    ]

    def choose_random_trajectory():
        chosen_trajectory_path = random.choice(trajectories_path)
        return Trajectory.from_file(chosen_trajectory_path)

    return choose_random_trajectory

class SameAnswerLeader(Leader):
    
    def setup(self, min_change, max_change):
        wait_time = random.uniform(min_change, max_change)
        target_answer = random.randrange(0, len(self.round.answer_positions))
        self.start_coroutine(self.send_change(target_answer), wait_time)
    
    async def send_change(self, target_answer: float):
        self.broadcast(str(target_answer))


# TODO: Pass hardcoded values to setup

class SameAnswerFollower(Follower):
    
    def setup(self, choose_random_trajectory):
        self.position = np.zeros(2)
        self.point_generator = TrajectoryGenerator(
            self.round.radius,
            self.round.answer_positions
        )
        self.choose_random_trajectory = choose_random_trajectory

        self.point_generator.set_trajectory(
            start=self.position,
            end=self.round.answer_positions[self.random_answer()],
            trajectory=self.choose_random_trajectory(),
            duration=random.uniform(3.5, 7)
        )
        
        # Time the agent is stopped at an answer
        self.rest_time = random.uniform(1, 2.5)
        self.is_random_mode = True

        self.change_trajectory_time = 0
        self.global_time = 0
    
    def update(self, delta: float):
        if self.is_random_mode:
            self.run_random_mode(delta)

        self.global_time += delta
        self.position = self.point_generator.step(delta)
        self.client.send_position(self.position)
    
    def run_random_mode(self, delta: float):
        if self.change_trajectory_time >= self.point_generator.replayer_duration() + self.rest_time:
            self.reset_trajectory(self.random_answer())
            self.rest_time = random.uniform(1, 2.5)
            self.change_trajectory_time = 0
        
        self.change_trajectory_time += delta
    
    def reset_trajectory(self, target_answer, duration=None):
        self.point_generator.set_trajectory(
            start=self.position,
            end=self.round.answer_positions[target_answer],
            trajectory=self.choose_random_trajectory(),
            duration=duration if duration else random.uniform(3.5, 7)
        )

    def on_message_receive(self, data: str):
        target_answer = str(data)
        remaining_time = self.round.duration - self.global_time
        
        # TODO: consider the case where the lower bound is greater than the upper bound
        replayer_duration = random.uniform(3, remaining_time - 0.5)
        self.reset_trajectory(target_answer, replayer_duration)
    
    def random_answer(self):
        return random.randrange(0, len(self.round.answer_positions))


if __name__ == "__main__":
    manager = AgentManager(
        SameAnswerFollower,
        agent_kwargs=dict(
            choose_random_trajectory=choose_random_trajectory_gen("trajectories")
        )
    )
    with HansPlatform("Random Walker", manager) as platform:
        platform.connect(API_HOST, HOST, MQTT_PORT)
        platform.listen()