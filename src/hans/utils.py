from __future__ import annotations

import numpy as np


def calculate_answer_points(num_answers: int, radius: float):
    # Keep in mind that the basis vector for the y axis points downwards. Therefore,
    # in this case, the first response will be drawn upwards (at pi / 2)

    angles = np.linspace(
        -np.pi / 2, -np.pi / 2 + 2 * np.pi, num_answers, endpoint=False
    )

    # We truncate because that is what is done in the original
    # (num_answers, 2)
    return np.trunc(
        radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    )

def distance_squared(v1: np.ndarray, v2: np.ndarray, axis: int = -1):
    return ((v1 - v2) ** 2).sum(axis=axis)


def distance(v1: np.ndarray, v2: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(distance_squared(v1, v2, axis))

def rotation_matrix(angle: float):
    """Returns the matrix which rotates 2D by 'angle' radians"""
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def rotate(point: np.ndarray, angle: float):
    """Rotates a vector 'angle' radians"""
    return rotation_matrix(angle) @ point