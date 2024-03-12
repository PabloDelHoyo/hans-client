from __future__ import annotations

from typing import Protocol
from dataclasses import dataclass

import numpy as np

from . import utils


def lerp(v0: np.ndarray, v1: np.ndarray, t: float):
    return (1 - t) * v0 + t * v1


def calculate_sector(point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Returns the indices of the vertices which delimit the sector where the point is located"""
    distance_to_answers = utils.distance(vertices, point)
    closest_indices = np.argsort(distance_to_answers)

    closest_idx = closest_indices[0]
    next_idx = (closest_idx + 1) % len(vertices)
    previous_idx = (closest_idx - 1) % len(vertices)
    if distance_to_answers[next_idx] < distance_to_answers[previous_idx]:
        second_closest_idx = next_idx
    else:
        second_closest_idx = previous_idx

    return np.array([closest_idx, second_closest_idx])


def get_factor_from_time(time: float, trajectory: Trajectory):
    """Calculates the factor that must be given to Replayer (and as a consequence
    to TrajectoryGenerator) so that it takes 'time' seconds to replay 'trajectory'"""
    return trajectory.original_duration() / time


@dataclass
class TrajectoryPoint:
    # elapsed seconds since the first position
    timestamp: float

    # Normalized vector so that, in case the radius is changed, the trajectory
    # of the data is still useful
    norm_position: np.ndarray

    @classmethod
    def from_row(cls, row):
        row = row.split(",")
        return cls(float(row[0]), np.array(list(map(float, row[1:]))))


@dataclass
class Trajectory:
    points: list[TrajectoryPoint]
    original_target: int

    @classmethod
    def from_file(cls, file_path: str) -> Trajectory:
        with open(file_path) as f:
            original_target, trajectories_rows = f.read().strip().split("\n\n")

            original_target = int(original_target)
            trajectories_rows = trajectories_rows.split("\n")

        return cls(
            [
                TrajectoryPoint.from_row(trajectory_row)
                for trajectory_row in trajectories_rows
            ],
            original_target,
        )

    def original_duration(self) -> float:
        """Return the time it took to record this trajectory. It is named this way in order
        to avoid confusion with the duration it takes for a Replayer to replay a trajectory.
        """

        return self.points[-1].timestamp - self.points[0].timestamp


class PointTransformUpdater(Protocol):
    """Represent the transformation applied to a PointTransform"""

    def update(self, transform: PointTransform, delta: float):
        ...


class PointTransform:
    """Represents the transformation applied to a point"""

    def __init__(
        self,
        angle: float,
        actual_vertices: np.ndarray,
        center_pos: np.ndarray,
        radius: float,
        new_vertices: np.ndarray | None = None,
    ):
        """
        angle: angle by which the space is rotated
        actual_vertices: a (N, 2) with the coordinates of the vertices. Since we are dealing
            with an hexagon, N=6
        center_pos: the origin of the transformed polygon (hexagon)
        radius: the radius of the polygon (hexagon)
        new_vertices: the position of the vertex after the sector of the point is known
        """
        self.center_pos = center_pos
        self.radius = radius
        self.actual_vertices = actual_vertices
        self.angle = angle

        self.new_vertices = (
            actual_vertices.copy() if new_vertices is None else new_vertices
        )

    @classmethod
    def from_vertex_target(
        cls,
        target_idx: int,
        original_target: int,
        actual_vertices: np.ndarray,
        center_pos: np.ndarray,
        radius: float,
        new_vertices: np.ndarray | None = None,
    ) -> PointTransform:
        sector_angle = 2 * np.pi / len(actual_vertices)
        angle = (target_idx - original_target) * sector_angle
        return cls(angle, actual_vertices, center_pos, radius, new_vertices)

    def __call__(self, point: np.ndarray) -> np.ndarray:
        # Rotate the original point
        rotated_point = self.radius * utils.rotate(point, self.angle)

        # Get the vectors which form the sector
        closest_answers = calculate_sector(rotated_point, self.actual_vertices)
        closest_answer_points = self.actual_vertices[closest_answers]

        # Apply the affine transformation
        proportions = np.linalg.solve(closest_answer_points.T, rotated_point)
        new_basis = self.new_vertices[closest_answers] - self.center_pos
        transformed_point = new_basis.T @ proportions + self.center_pos

        return transformed_point


class Replayer:
    """Recreates a given trajectory"""

    def __init__(
        self,
        trajectory: Trajectory,
        transform: PointTransform,
        time_multiplier: float = 1,
    ):
        """
        trajectory: the trajectory which will be applied
        transform: the transformation that has to be applied to the point before being
            return
        time_multiplier: the speed at which the trajectory will be replayed. For example,
            if factor = 2, then the trajectory will be replayed twice as fast.
        """
        self.transform = transform
        self._time_multiplier = time_multiplier

        self._trajectory = trajectory

        self._idx = 0
        self._elapsed_time = 0

    def step(self, delta: float) -> np.ndarray:
        """Returns the appropiate point from the trajectory. Linear interpolation
        is applied to avoid stuttering movement in case the speed at which the trajectory
        is replayed is low"""

        while (
            not self.has_finished()
            and self._elapsed_time >= self._next_point().timestamp
        ):
            self._advance()

        if self.has_finished():
            return self.transform(self._current_point().norm_position)

        current_timestamp = self._current_point().timestamp
        duration = self._next_point().timestamp - current_timestamp

        # NOTE: in the beginning this is negative but I think this is not a problem
        # becuase lerp will interpolate outside the interval. CHECK that statement
        time_spent = self._elapsed_time - current_timestamp

        self._elapsed_time += delta * self._time_multiplier

        return lerp(
            self.transform(self._current_point().norm_position),
            self.transform(self._next_point().norm_position),
            time_spent / duration,
        )

    def duration(self) -> float:
        return self._trajectory.points[-1].timestamp / self._time_multiplier

    def has_finished(self):
        return self._idx == len(self._trajectory.points) - 1

    def _current_point(self):
        return self._trajectory.points[self._idx]

    def _next_point(self):
        return self._trajectory.points[self._idx + 1]

    def _advance(self):
        self._idx += 1


class MoveCenterTowardsOrigin:
    """Updates a PointTransform so that its new center moves at a constant speed
    to the true origin"""

    def __init__(self, speed: float):
        self.speed = speed

    def update(self, transform: PointTransform, delta: float):
        # TODO: since delta is not constant, we may have problems with this
        # if statement because the stop condition is not deterministic. However,
        # since in a sensible game loop delta is upper bound, we have guarantees
        # on that condition

        # TODO: refactor this into a function
        mag = np.linalg.norm(transform.center_pos)
        if mag > self.speed * delta:
            direction = -transform.center_pos / mag
            transform.center_pos += direction * self.speed * delta
        else:
            transform.center_pos = np.zeros(2)


class MoveVertexTowardsTarget:
    """Updates a PointTransform so that one of new_vertices moves at a constant speed
    towards a target point"""

    def __init__(self, speed: float, target: np.ndarray, moving_vertex_idx: int):
        self.speed = speed
        self.target = target

        # The vertex which will be moved towards the target
        self.moving_vertex_idx = moving_vertex_idx

    def update(self, transform: PointTransform, delta: float):
        disp = self.target - transform.new_vertices[self.moving_vertex_idx]

        # TODO: refactor this into a function
        mag = np.linalg.norm(disp)

        # TODO: it seems that applying this transformation is worse than not applying it
        # at all. The main problem is in those trajectories where you stop on a point for
        # for a certain period of time.
        # The following line effectively removes the effect of it.
        transform.new_vertices[self.moving_vertex_idx] = self.target

        if mag > self.speed * delta:
            direction = disp / mag
            transform.new_vertices[self.moving_vertex_idx] += (
                direction * self.speed * delta
            )
        else:
            transform.new_vertices[self.moving_vertex_idx] = self.target


@dataclass
class Duration:
    time: float


@dataclass
class TimeMultiplier:
    multiplier: float


DurationOption = Duration | TimeMultiplier


class TrajectoryGenerator:
    """This class is in charge of generating a trajectory between
    two arbitrary points inside the hexagon based on recorded trajectories
    which start in the origin and end up in one vertex"""

    def __init__(self, radius: float, vertices_pos: np.ndarray):
        """
        radius: the radius of the polygon. It is not obtained directly from 'vertices_pos'
            because, since truncation is applied, each vertex is not at the same distance
            from the origin so depending on the chosen vertex, the radius will be different.

        vertices_pos: A (N, 2) numpy array containing containing the vertices of the
        polygon. Since we are dealing with an hexagon, N=6
        """

        self._radius = radius
        self._vertices_pos = vertices_pos

        self._replayer: Replayer | None = None

        # The transformations applied to PositionTransform
        self._transform_updaters: list[PointTransformUpdater] = []

    def set_trajectory(
        self,
        start: np.ndarray,
        end: np.ndarray,
        trajectory: Trajectory,
        duration: DurationOption | float = 1,
        origin_speed_multiplier=1,
        target_speed_multiplier=1,
    ):
        """
        start: the point where the trajectory begins
        end: the point where the trajectory ends
        trajectory: the original trajectory which will be transformed to create a trajectory
            from 'start' to 'end'
        duration: if a numerical value is passed (int or float), then that value is
            the time TrajectoryGenerator will take to replay those trajectories. It is also
            possible to pass a time multiplier. For example, if the time multiplier is 2, then the
            trajectory will be replayed twice as fast. In order to pass a time multiplier, that
            value must be wrapped in a TimeMultiplier (i.e duration=TimeMultiplier(2))
        origin_speed_multiplier: factor by which the speed passed to MoveCenterTowardsOrigin
            is multiplied. Specifically, the speed is equal to
            origin_speed_multiplier * s / trajectory.original_duration()
            where s is the distance from start and the origin (0, 0)
        target_speed_multiplier: factor by which the speed passed to MoveVertexTowardsTarget
            is multiplied. specifically, the speed is equal to
            target_speed_multiplier * s / trajectory.original_duration()
            where s is the distance from the closest vertex to the target point and the
            target point. THIS CURRENTLY HAS NO EFFECT
        """
        closest_vertex = calculate_sector(end, self._vertices_pos)[0]
        transform = PointTransform.from_vertex_target(
            closest_vertex,
            trajectory.original_target,
            self._vertices_pos,
            start,
            self._radius,
        )

        match duration:
            case Duration(time):
                time_multiplier = get_factor_from_time(time, trajectory)
            case TimeMultiplier(multiplier):
                time_multiplier = multiplier
            case time if isinstance(time, (int, float)):
                time_multiplier = get_factor_from_time(time, trajectory)
            case invalid:
                raise ValueError(
                    f"{invalid} of type {type(invalid)} is not a valid value for the duration"
                )

        self._replayer = Replayer(trajectory, transform, time_multiplier)

        towards_origin_speed = (
            origin_speed_multiplier * np.linalg.norm(start) / self._replayer.duration()
        )

        towards_target_speed = (
            target_speed_multiplier
            * np.linalg.norm(end - closest_vertex)
            / self._replayer.duration()
        )

        self._transform_updaters = [
            MoveCenterTowardsOrigin(towards_origin_speed),
            MoveVertexTowardsTarget(towards_target_speed, end, closest_vertex),
        ]

    def step(self, delta: float) -> np.ndarray:
        point = self._replayer.step(delta)
        for transform_updater in self._transform_updaters:
            transform_updater.update(self._replayer.transform, delta)

        return point

    def replayer_duration(self) -> float:
        """Returns the duration the TrajectoryGenerator is going to take to replay
        the trajectory after considering the time multiplier which is passed to the Replayer
        """

        return self._replayer.duration()

    def current_trajectory(self) -> Trajectory:
        return self._replayer._trajectory

    def has_finished(self) -> bool:
        return self._replayer is not None and self._replayer.has_finished()
