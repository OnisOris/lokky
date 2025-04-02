import numpy as np
from numpy.typing import NDArray
from numpy import cos, sin


def saturation(vector: np.ndarray, max_value: float) -> np.ndarray:
    """
    Limit the vector by the maximum magnitude.

    :param vector: The vector to be limited.
    :type vector: np.ndarray

    :param max_value: The maximum allowed magnitude.
    :type max_value: float

    :return: A vector with its magnitude limited to max_value.
    :rtype: np.ndarray
    """
    norm = np.linalg.norm(vector)
    if norm > max_value:
        return (vector / norm) * max_value
    return vector


def limit_acceleration(
    current_velocity: np.ndarray,
    target_velocity: np.ndarray,
    max_acceleration: float,
) -> np.ndarray:
    """
    Limit the maximum acceleration.

    This method works with vectors of any dimension.

    :param current_velocity: Current velocity vector.
    :type current_velocity: np.ndarray

    :param target_velocity: Target velocity vector.
    :type target_velocity: np.ndarray

    :param max_acceleration: Maximum allowed acceleration.
    :type max_acceleration: float

    :return: The new velocity vector after limiting the change.
    :rtype: np.ndarray
    """
    change = target_velocity - current_velocity
    norm = np.linalg.norm(change)
    if norm > max_acceleration:
        change = change / norm * max_acceleration
    return current_velocity + change


def find_points_in_radius(
    center_sphere: NDArray, points: NDArray, radius: float = 1.0
) -> NDArray:
    """
    Find points within a sphere.

    :param center_sphere: The center of the sphere (1 x 3).
    :type center_sphere: NDArray

    :param points: Array of points in space.
    :type points: NDArray

    :param radius: The search radius.
    :type radius: float

    :return: An array of points that lie within the sphere.
    :rtype: NDArray
    """
    tx, ty, tz = center_sphere
    radius_sq = radius**2
    result = np.zeros(3)
    for x, y, z in points:
        dx = x - tx
        dy = y - ty
        dz = z - tz
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq <= radius_sq:
            result = np.vstack([result, [x, y, z]])
    return result[1:]


def rot_v(vector: np.ndarray, angle: float, axis: np.ndarray) -> np.ndarray:
    """
    Rotate the input vectors around an arbitrary axis.

    Positive rotation is defined as clockwise when looking along the axis toward the observer.

    :param vector: The vectors to rotate (n x 3).
    :type vector: np.ndarray

    :param angle: The angle of rotation in radians.
    :type angle: float

    :param axis: The axis vector around which the rotation occurs.
    :type axis: np.ndarray

    :return: The rotated vectors.
    :rtype: np.ndarray
    """
    axis = normalization(axis, 1)
    x, y, z = axis
    c = cos(angle)
    s = sin(angle)
    t = 1 - c
    rotate = np.array(
        [
            [t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z**2 + c],
        ]
    )
    rot_vector = np.dot(vector, rotate)
    return rot_vector


def check_point_in_radius(
    center_sphere: NDArray, point: NDArray, radius: float = 1.0
) -> tuple[bool, float, NDArray]:
    """
    Check if a point is within a sphere and return distance and vector.

    :param center_sphere: The center of the sphere (1 x 3).
    :type center_sphere: NDArray

    :param point: The point to check in space.
    :type point: NDArray

    :param radius: The search radius.
    :type radius: float

    :return: A tuple containing a boolean indicating if the point is within the sphere,
             the distance to the point, and the vector from the center to the point.
    :rtype: tuple(bool, float, NDArray)
    """
    vector: NDArray = point - center_sphere
    dist = np.linalg.norm(vector)
    return bool(dist <= radius), float(dist), vector


def normalization(vector: np.ndarray, length: float = 1.0) -> np.ndarray:
    """
    Return the normalized vector with a specified length.

    :param vector: The input vector.
    :type vector: np.ndarray

    :param length: The desired length of the normalized vector.
    :type length: float

    :return: The normalized vector scaled to the specified length.
    :rtype: np.ndarray
    """
    return np.array(vector) / np.linalg.norm(vector) * length


class SSolver:
    """
    SSolver (Swarm Solver) - A solver class for swarm behavior.

    System of the form:
        x_dot = A*x + b*u
        y = C*x

    where x = [x, y, z, vx, vy, vz] for each object,
    and A is a matrix in R^(n x 6).
    When running on a drone, x[0] corresponds to the drone's state vector.
    """

    def __init__(self, params: dict):
        """
        Initialize the parameters.

        :param params: Dictionary of parameters.
        :type params: dict
        """
        self.kd = None
        self.ki = None
        self.kp = None
        self.max_acceleration = None
        self.max_speed = None
        self.safety_radius = None
        self.noise_weight = None
        self.unstable_weight = None
        self.repulsion_weight = None
        self.alignment_weight = None
        self.attraction_weight = None
        self.cohesion_weight = None
        self.unstable_radius = None
        self.params = params
        if self.params is None:
            self.params = {
                "kp": 1,
                "ki": 0,
                "kd": 1,
                "attraction_weight": 1.0,
                "cohesion_weight": 1.0,
                "alignment_weight": 1.0,
                "repulsion_weight": 4.0,
                "unstable_weight": 1.0,
                "noise_weight": 1.0,
                "safety_radius": 1.0,
                "max_acceleration": 1,
                "max_speed": 0.4,
            }
        self.read_params(params)
        # Variables for storing previous values
        self.previous_error = None
        self.integral = np.zeros_like(self.kp, dtype=np.float64)

    def read_params(self, params: dict) -> None:
        """
        Read and set the parameters from a dictionary.

        :param params: Dictionary of parameters.
        :type params: dict
        """
        self.params = params
        self.attraction_weight: float = params["attraction_weight"]
        self.cohesion_weight: float = params["cohesion_weight"]
        self.alignment_weight: float = params["alignment_weight"]
        self.repulsion_weight: float = params["repulsion_weight"]
        self.unstable_weight: float = params["unstable_weight"]
        self.noise_weight: float = params["noise_weight"]
        self.safety_radius: float = params["safety_radius"]
        self.max_speed: float = params["max_speed"]
        self.max_acceleration: float = params["max_acceleration"]
        self.unstable_radius: float = params["unstable_radius"]
        self.kp = np.array(params["kp"])
        self.ki = np.array(params["ki"])
        self.kd = np.array(params["kd"])

    def solve(
        self, state_matrix: NDArray, target_matrix: NDArray, dt: float
    ) -> NDArray:
        """
        Solve for the control velocity for each object based on the state matrix.

        :param state_matrix: State matrix of the plant (n x 6).
        :type state_matrix: NDArray

        :param target_matrix: Target state matrix (n x 6).
        :type target_matrix: NDArray

        :param dt: Time step.
        :type dt: float

        :return: The control velocity matrix (n x 3).
        :rtype: NDArray
        """
        error = target_matrix - state_matrix
        # Proportional term
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        # Differential term (error difference)
        if dt == 0.0:
            derivative = 0
        elif self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = np.zeros_like(error)
        d_term = self.kd * derivative
        # Save the current error for the next calculation
        self.previous_error = error
        vda = self.compute_velocity_direction_all(state_matrix, error)
        # Final control signal
        control_signal = saturation(
            p_term[:, 0:3] + i_term[:, 0:3] + d_term[:, 0:3], 1
        )
        control_signal += vda
        return control_signal

    def compute_velocity_direction_all(
        self, state_matrix: NDArray, error_matrix: NDArray
    ) -> NDArray:
        """
        Compute the control velocity for all state vectors.

        :param state_matrix: State matrix (n x 6).
        :type state_matrix: NDArray

        :param error_matrix: Matrix of errors (n x 6).
        :type error_matrix: NDArray

        :return: Control velocities for all objects (n x 3).
        :rtype: NDArray
        """
        control_velocity_matrix = np.zeros(3)
        for index, _ in enumerate(state_matrix):
            control_velocity_matrix = np.vstack(
                [
                    control_velocity_matrix,
                    self.compute_velocity_direction(
                        index, state_matrix, error_matrix
                    ),
                ]
            )
        return control_velocity_matrix[1:]

    def compute_velocity_direction(
        self,
        index_current_state_vector: int,
        state_matrix: NDArray,
        error_matrix: NDArray,
    ) -> NDArray:
        """
        Compute the control velocity for a single state vector.

        :param index_current_state_vector: Index of the current state vector.
        :type index_current_state_vector: int

        :param state_matrix: State matrix (n x 6).
        :type state_matrix: NDArray

        :param error_matrix: Matrix of errors (n x 6).
        :type error_matrix: NDArray

        :return: The new velocity vector (3,).
        :rtype: NDArray
        """
        repulsion_force = np.zeros(3)
        unstable_vector = np.zeros(3)
        current_state = state_matrix[index_current_state_vector]
        current_error_norm = np.linalg.norm(
            error_matrix[index_current_state_vector][0:3]
        )
        state_matrix_filtered = self.state_filter(
            current_state, state_matrix, self.safety_radius * 3
        )

        points_around = find_points_in_radius(
            current_state[0:3],
            state_matrix_filtered[:, 0:3],
            self.unstable_radius,
        )

        if len(points_around) > 0:
            mean_vector = np.mean(points_around - current_state[0:3], axis=0)
            mean_norm = np.linalg.norm(mean_vector)
            if mean_norm > 1e-6:
                mean_dir = mean_vector / mean_norm
                axis = np.cross(mean_dir, np.array([0, 0, 1]))
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-6:
                    axis = np.array(
                        [1, 0, 0]
                    )
                else:
                    axis = normalization(axis)
                projections = np.dot(points_around - current_state[0:3], axis)
                count_positive = np.sum(projections > 0)
                count_negative = len(projections) - count_positive
                angle = (
                    np.pi / 2
                    if count_positive < count_negative
                    else -np.pi / 2
                )
        for j, other_state in enumerate(state_matrix_filtered):
            if j == index_current_state_vector:
                continue
            check_in_sphere, dist_to_object, vector_to_object = (
                check_point_in_radius(
                    current_state[0:3], other_state[0:3], self.safety_radius
                )
            )
            vector_to_object_norm = normalization(vector_to_object, 1)
            if check_in_sphere:
                repulsion_force -= vector_to_object_norm / (
                    (dist_to_object + 1 - self.safety_radius) ** 2
                )

            if self.term_count_unstable_vector(
                dist_to_object, current_error_norm, current_state[3:6]
            ):
                if len(points_around) > 0 and mean_norm > 1e-6:
                    rotated_vector = rot_v(
                        vector_to_object_norm * 0.3, angle, axis
                    )
                    unstable_vector += rotated_vector

        new_velocity = (
            current_state[3:6]
            + self.repulsion_weight * repulsion_force
            + self.unstable_weight * unstable_vector
        )
        new_velocity = limit_acceleration(
            current_state[3:6],
            new_velocity,
            max_acceleration=self.max_acceleration,
        )
        return saturation(new_velocity, self.max_speed)


    def state_filter(
        self, current_state, state_matrix: NDArray, filter_radius: float
    ) -> NDArray:
        state_matrix_filtered = np.zeros(state_matrix.shape[1])
        for other_state in state_matrix:
            if check_point_in_radius(
                current_state[0:3], other_state[0:3], filter_radius
            ):
                state_matrix_filtered = np.vstack(
                    [state_matrix_filtered, other_state]
                )
        return state_matrix_filtered[1:]

    def term_count_unstable_vector(
        self, dist_to_other_drone: float, error: float, speed: NDArray
    ) -> bool:
        """
        Determine whether to add an unstable vector component based on distance, error, and speed.

        :param dist_to_other_drone: Distance to the other drone.
        :type dist_to_other_drone: float

        :param error: The error magnitude.
        :type error: float

        :param speed: The speed vector.
        :type speed: NDArray

        :return: True if the conditions for an unstable vector are met, False otherwise.
        :rtype: bool
        """
        return (
            dist_to_other_drone < self.safety_radius + 0.1
            and np.allclose(np.linalg.norm(speed), 0, atol=0.1)
            and error > self.safety_radius + 0.2
        )
