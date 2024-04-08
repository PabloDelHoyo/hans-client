import logging

from hans.leader import Leader, LeaderManager


def configure_logger(level, formatter, handler=logging.StreamHandler()):
    handler.setFormatter(formatter)

    logger = logging.getLogger("hans")
    logger.setLevel(level)
    logger.addHandler(handler)

class MyLeader(Leader):
    def setup(self):
        print("Setup has been called")
        print("Participantes desde leader ", self.round.participants)
        self.start_coroutine(self.my_coro(), 2)
    
    def update(self, delta: float):
        pass
        # print(self.positions)

    def close(self):
        print("Se ha terminado")
    
    def on_message_received(self, agent_name: str, data: str):
        print(f"{agent_name}: {data}")
    
    async def my_coro(self):
        self.broadcast("Hello there")


manager = LeaderManager(MyLeader)
manager.bind()
manager.start()