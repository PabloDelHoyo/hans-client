from __future__ import annotations

from typing import TYPE_CHECKING

import threading

import numpy as np

if TYPE_CHECKING:
    from .position_codec import PositionCodec


class StateSnapshot:
    def __init__(self, state: dict[int, np.ndarray], client_id: int):
        self._state = state
        self._array_state = list(state.values())
        self._client_id = client_id

    def position_by_id(self, participant_id: int) -> np.ndarray:
        return self._state[participant_id]

    @property
    def all_positions(self) -> np.ndarray:
        return self._array_state

    @property
    def other_positions(self) -> np.ndarray:
        return [position for participant_id, position in self._state.items() if participant_id != self._client_id]


class State:
    """Class to update the global state given individual delta updates. In a real
    game, we would poll the I/O system to get the latest network messages"""

    def __init__(
        self, pcodec: PositionCodec, participant_ids: list[int], client_id: int
    ):
        self._client_id = client_id
        self._pcodec = pcodec

        # All participants start at the postition (0, 0)
        self._state = {
            participant_id: np.zeros(2) for participant_id in participant_ids
        }

        # We don't want the state to be updated when it is being copy in "get_state"
        self._lock = threading.Lock()

    def update(self, participant_id: int, position: np.ndarray):
        with self._lock:
            self._state[participant_id] = self._pcodec.decode(position)

    def get_snapshot(self) -> StateSnapshot:
        """Copy the current state. This is safe to call when there are multiple threads"""

        with self._lock:
            return StateSnapshot(
                {
                    participant_id: position.copy()
                    for participant_id, position in self._state.items()
                },
                self._client_id,
            )
