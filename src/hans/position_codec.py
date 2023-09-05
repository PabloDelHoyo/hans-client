import numpy as np

from . import utils


class PositionCodec:
    """Useful to transform the position from and to the format which the
    server requires"""

    def __init__(self, answer_points: np.ndarray):
        self.answer_points = answer_points

    def encode(self, point: np.ndarray) -> np.ndarray:
        """Transforms the (2, ) numpy array into another numpy array in the format
        required by the hans platform"""

        distance_to_answers = utils.distance(self.answer_points, point)
        closest_indices = np.argsort(distance_to_answers)[:2]
        new_basis = self.answer_points[closest_indices]
        point_new_basis = np.linalg.solve(new_basis.T, point)

        encoded_position = np.zeros(self.answer_points.shape[0])
        encoded_position[closest_indices] = point_new_basis

        return encoded_position

    def decode(self, encoded_position: np.ndarray) -> np.ndarray:
        """Decodes the format sent by the hans platform"""

        return self.answer_points.T @ encoded_position
