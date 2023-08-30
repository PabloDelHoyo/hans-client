import numpy as np


class PositionCodec:
    """Useful to transform the position from and to the format which the
    server requires"""

    def __init__(self, num_answers: int, radius=430):
        self.radius = radius

        # Keep in mind that the basis vector for the y axis points downwards. Therefore,
        # in this case, the first response will be drawn upwards (at pi / 2)
        angles = np.linspace(
            -np.pi / 2, -np.pi / 2 + 2 * np.pi, num_answers, endpoint=False
        )

        # We truncate because that is what is done in the original
        # (num_answers, 2)
        self.answer_points = np.trunc(
            radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)
        )

    def distance_to_answers(self, point: np.ndarray) -> np.ndarray:
        """Calculates the distance of point to every answer.
        Point must be a (2, ) shape numpy array"""

        diff_sq = (self.answer_points - point) ** 2
        return np.sqrt(diff_sq.sum(axis=1))

    def encode(self, point: np.ndarray) -> np.ndarray:
        """Transforms the (2, ) numpy array into another numpy array in the format
        required by the hans platform"""

        closest_indices = np.argsort(self.distance_to_answers(point))[:2]
        new_basis = self.answer_points[closest_indices]
        point_new_basis = np.linalg.solve(new_basis.T, point)

        encoded_position = np.zeros(self.answer_points.shape[0])
        encoded_position[closest_indices] = point_new_basis

        return encoded_position

    def decode(self, encoded_position: np.array) -> np.ndarray:
        """Decodes the format sent by the hans platform"""

        return self.answer_points.T @ encoded_position
