from __future__ import annotations

import numpy as np

from hans import HansPlatform, Loop, LoopThread

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hans.state import StateSnapshot

NAME = "Oscillator"

HOST = "52.214.15.30"
API_PORT = 3000
MQTT_PORT = 9001

class Oscillator(Loop):
    
    def setup(self, max_radius, radius_speed, period):
        self.max_radius = max_radius
        self.radius_speed = radius_speed
        self.angular_velocity = 2 * np.pi / period

        self.radius = 0
        self.angle = 0

    def update(self, snapshot: StateSnapshot, delta: float):
        if self.radius <= 0 or self.radius >= self.max_radius:
            self.radius = np.clip(self.radius, 0, self.max_radius)
            self.radius_speed *= -1

        self.radius += self.radius_speed * delta
        self.angle += self.angular_velocity * delta
    
    def render(self, sync_ratio: float):
        position = self.radius * np.array([np.cos(self.angle), np.sin(self.angle)])
        self.client.send_position(position)


def main():
    oscillator_thread = LoopThread(loop_cls=Oscillator, loop_kwargs={
        "max_radius": 350,
        "radius_speed": 100,
        "period": 4
    })

    with HansPlatform(NAME, oscillator_thread) as platform:
        platform.connect(HOST, API_PORT, MQTT_PORT)
        platform.listen()


if __name__ == "__main__":
   main()