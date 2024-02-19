from __future__ import annotations

import logging

import numpy as np

from hans import HansPlatform, Agent, AgentManager

NAME = "Oscillator"

HOST = "127.0.0.1"
API_PORT = 3000
MQTT_PORT = 9001

def configure_logger(level, formatter, handler=logging.StreamHandler()):
    handler.setFormatter(formatter)

    logger = logging.getLogger("hans")
    logger.setLevel(level)
    logger.addHandler(handler)

# If you want to log info messages, write
# logging.INFO
configure_logger(
    logging.INFO,
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

class Oscillator(Agent):

    def setup(self, max_radius, radius_speed, period):
        self.max_radius = max_radius
        self.radius_speed = radius_speed
        self.angular_velocity = 2 * np.pi / period

        self.radius = 0
        self.angle = 0

    def update(self, delta: float):
        if self.radius <= 0 or self.radius >= self.max_radius:
            self.radius = np.clip(self.radius, 0, self.max_radius)
            self.radius_speed *= -1

        self.radius += self.radius_speed * delta
        self.angle += self.angular_velocity * delta

        position = self.radius * np.array([np.cos(self.angle), np.sin(self.angle)])

        self.client.send_position(position)


def main():
    oscillator_manager = AgentManager(
        Oscillator,
        agent_kwargs=dict(
            max_radius=350,
            radius_speed=100,
            period=4
        )
    )

    with HansPlatform(NAME, oscillator_manager) as platform:
        platform.connect(HOST, API_PORT, MQTT_PORT)
        platform.listen()


if __name__ == "__main__":
    main()
