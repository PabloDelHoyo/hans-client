from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Round
    from .client import HansClient


class ThreadLoopManager:
    """Contains the method definitions of the loop manager which HansClient controls"""

    def start_session(self, round: Round, hans_client: HansClient):
        """Called when a session starts"""

    def start_thread(
        self, agent_name: str,
        exc_handler: Callable[[None], None] | None = None
    ):
        """Starts the thread where the loop will be executed
            - agent_name: The name that will be shown in admin view in the web app
            - exc_handler: called when an exception is raised inside the loop
        """

    def quit(self):
        """Stops the game loop"""

    def finish_session(self):
        """Called when the session stops"""

    def is_thread_alive(self):
        """Returns true if the executing thread is alive"""

    @property
    def exc_info(self):
        """Returns the information associated to an exception raised inside a thread, if any"""
