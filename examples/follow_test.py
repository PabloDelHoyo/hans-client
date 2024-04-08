from __future__ import annotations

from hans.follower import Follower, FollowerManager
from hans import HansPlatform

NAME = "Follower"

HOST = "52.214.15.30"
API_HOST = f"http://{HOST}:8080"
MQTT_PORT = 9001

class MyFollower(Follower):

    def setup(self):
        print("ha comenzado")
        print("Participantes desde follower", self.round.participants)
        self.start_coroutine(self.my_coro(), 2)
    
    def on_message_receive(self, data: str):
        print(f"Se ha recibido del lider: {data}")
    
    async def my_coro(self):
        print("hola")
        self.send_msg("Hola a todos")

follower_manager = FollowerManager(
    MyFollower
)

with HansPlatform(NAME, follower_manager) as platform:
    platform.connect(API_HOST, HOST, MQTT_PORT)
    platform.listen()