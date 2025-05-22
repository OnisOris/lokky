import builtins
from typing import Annotated, Any

import numpy as np
from numpy import cos, sin
from numpy.typing import NDArray

Array3 = Annotated[NDArray[Any], (3,)]


def saturation(vector: NDArray, max_value: float) -> NDArray:
    """
    Limit the vector magnitude to a maximum value.

    :param vector: Input vector.
    :param max_value: Maximum allowed magnitude.
    :return: Vector limited to max_value.
    """
    norm = np.linalg.norm(vector)
    if norm > max_value:
        return vector / norm * max_value
    return vector


def limit_acceleration(
    current_velocity: NDArray,
    target_velocity: NDArray,
    max_acceleration: float,
    dt: float,
) -> NDArray:
    """
    Limit the change in velocity (acceleration) to a maximum value.

    :param current_velocity: Current velocity vector.
    :param target_velocity: Desired target velocity vector.
    :param max_acceleration: Maximum allowed acceleration.
    :param dt: Time step
    :return: Updated velocity vector limited by max_acceleration.
    """
    change = target_velocity - current_velocity
    norm = np.linalg.norm(change)
    if norm > max_acceleration * dt:
        change = change / norm * max_acceleration
    return current_velocity + change


def find_points_in_radius(
    center_sphere: NDArray, points: NDArray, radius: float = 1.0
) -> NDArray:
    """
    Vectorized search to find points within a sphere.

    :param center_sphere: Center of the sphere.
    :param points: Array of points.
    :param radius: Sphere radius.
    :return: Points within the sphere.
    """
    diff = points - center_sphere
    distances_sq = np.sum(diff**2, axis=1)
    mask = distances_sq <= radius**2
    return points[mask]


def rot_v(vector: NDArray, angle: float, axis: NDArray) -> NDArray:
    """
    Rotate a vector (or set of vectors) around an arbitrary axis.

    Positive rotation is defined as clockwise when looking along the axis toward the observer.

    :param vector: Vector(s) to rotate.
    :param angle: Rotation angle in radians.
    :param axis: Axis of rotation.
    :return: Rotated vector(s).
    """
    axis = normalization(axis, 1)
    x, y, z = axis
    c = cos(angle)
    s = sin(angle)
    t = 1 - c
    # Rodrigues' rotation matrix
    rotate = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )
    return np.dot(vector, rotate)


def check_point_in_radius(
    center_sphere: Array3, point: Array3, radius: float = 1.0
) -> tuple[np.bool[builtins.bool], float, Array3]:
    """
    Check if a point is within a sphere and calculate the distance and vector from the center.

    :param center_sphere: Center of the sphere.
    :param point: The point to check.
    :param radius: Sphere radius.
    :return: Tuple containing a boolean (is inside), distance, and the vector from the center to the point.
    """
    vector = point - center_sphere
    dist = np.linalg.norm(vector)
    return dist <= radius, float(dist), vector


def normalization(vector: NDArray, length: float = 1.0) -> NDArray:
    """
    Normalize the vector to a specified length.

    :param vector: Input vector.
    :param length: Desired length after normalization.
    :return: Normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return np.zeros_like(vector)
    return vector / norm * length


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

    def __init__(self, params: dict, count_of_objects: int = 5):
        """
        Initialize the parameters.

        :param params: Dictionary of parameters.
        :param count_of_objects: count of objects for solving
        """
        self.params = None
        self.count_of_objects: int = count_of_objects
        self.kd: NDArray = np.ones((self.count_of_objects, 6))
        self.ki: NDArray = np.zeros((self.count_of_objects, 6))
        self.kp: NDArray = np.ones((self.count_of_objects, 6))
        self.current_velocity_weight: float = 0.0
        self.max_acceleration: float = 1.0
        self.max_speed: float = 0.4
        self.safety_radius: float = 1.0
        self.noise_weight: float = 0.0
        self.unstable_weight: float = 1.0
        self.repulsion_weight: float = 4.0
        self.alignment_weight: float = 1.0
        self.attraction_weight: float = 1.0
        self.cohesion_weight: float = 1.0
        self.unstable_radius: float = 1.0
        self.current_velocity_weght: float = 1.0
        self.border_x: NDArray = np.array([-5, 5])
        self.border_y: NDArray = np.array([-5, 5])
        self.border_z: NDArray = np.array([0, 4])
        if params is not None:
            self.read_params(params)
        # Variables for storing previous error and integral term for PID control
        self.previous_error: NDArray = np.zeros((self.count_of_objects, 6))
        self.integral: NDArray = np.zeros_like(self.kp, dtype=np.float64)

    def read_params(self, params: dict) -> None:
        """
        Read and assign parameters from a dictionary.

        :param params: Dictionary of parameters.
        """
        self.params = params
        self.attraction_weight: float = params["attraction_weight"]
        self.cohesion_weight: float = params["cohesion_weight"]
        self.alignment_weight: float = params["alignment_weight"]
        self.current_velocity_weight = params["current_velocity_weight"]
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

    def solve_for_one(
        self, state_matrix: NDArray, target_position: NDArray, dt: float
    ) -> NDArray:
        """
        Compute the control for one object

        :param state_matrix: State of plants, state_matrix[0] - self position
        :param target_position: Target point for plant
        :param dt: Time step
        :return: NDArray
        """
        error = np.array([target_position - state_matrix[0]])
        # Compute PID terms
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        if dt == 0.0:
            derivative = np.zeros_like(error)
        else:
            derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        self.previous_error = error

        # Compute additional velocity direction based on swarm behavior
        vda = self.compute_velocity_direction(
            index_current_state_vector=0,
            state_matrix=state_matrix,
            error_matrix=error,
            dt=dt,
        )
        # Final control signal (limited to a maximum magnitude)
        control_signal = saturation(
            p_term[:, :3] + i_term[:, :3] + d_term[:, :3], 1
        )
        control_signal += vda
        return control_signal

    def solve_for_all(
        self, state_matrix: NDArray, target_matrix: NDArray, dt: float
    ) -> NDArray:
        """
        Compute the control velocity for each object.

        :param state_matrix: Current state matrix (n x 6).
        :param target_matrix: Desired target state matrix (n x 6).
        :param dt: Time step.
        :return: Control velocity matrix (n x 3).
        """
        error = target_matrix - state_matrix
        # Compute PID terms
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        if dt == 0.0:
            derivative = 0
        elif self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = np.zeros_like(error)
        d_term = self.kd * derivative
        self.previous_error = error

        # Compute additional velocity direction based on swarm behavior
        vda = self.compute_velocity_direction_all(
            state_matrix=state_matrix, error_matrix=error, dt=dt
        )
        # Final control signal (limited to a maximum magnitude)
        control_signal = saturation(
            p_term[:, :3] + i_term[:, :3] + d_term[:, :3], 1
        )
        control_signal += vda
        return control_signal

    def compute_velocity_direction_all(
        self, state_matrix: NDArray, error_matrix: NDArray, dt: float
    ) -> NDArray:
        """
        Compute the corrective velocity vectors for all objects.

        :param state_matrix: Current state matrix (n x 6).
        :param error_matrix: Error matrix (n x 6).
        :param dt: Time step
        :return: Corrective velocity matrix (n x 3).
        """
        n = state_matrix.shape[0]
        control_velocity_matrix = np.empty((n, 3), dtype=np.float64)
        for index in range(n):
            control_velocity_matrix[index, :] = (
                self.compute_velocity_direction(
                    index, state_matrix, error_matrix, dt=dt
                )
            )
        return control_velocity_matrix

    def compute_velocity_direction(
        self,
        index_current_state_vector: int,
        state_matrix: NDArray,
        error_matrix: NDArray,
        dt: float,
    ) -> NDArray:
        """
        Compute the corrective velocity vector for a single object considering repulsion and unstable corrections.

        :param index_current_state_vector: Index of the current state vector.
        :param state_matrix: Current state matrix (n x 6).
        :param error_matrix: Error matrix (n x 6).
        :param dt: Time step
        :return: New velocity vector (3,).
        """
        current_state = state_matrix[index_current_state_vector]
        current_error_norm = np.linalg.norm(
            error_matrix[index_current_state_vector][:3]
        )

        # Filter states: take only objects within safety_radius * 3
        state_matrix_filtered = self.state_filter(
            current_state, state_matrix, self.safety_radius * 3
        )

        # Find points within unstable_radius for distribution analysis
        points_around = find_points_in_radius(
            current_state[:3],
            state_matrix_filtered[:, :3],
            self.unstable_radius,
        )
        angle = 0
        axis = np.array([0, 0, 1])
        if points_around.size > 0:
            mean_vector = np.mean(points_around - current_state[:3], axis=0)
            mean_norm = np.linalg.norm(mean_vector)
            if mean_norm > 1e-6:
                mean_dir = mean_vector / mean_norm
                axis = np.cross(mean_dir, np.array([0, 0, 1]))
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-6:
                    axis = np.array([1, 0, 0])
                else:
                    axis = normalization(axis)
                projections = np.dot(points_around - current_state[:3], axis)
                count_positive = np.sum(projections > 0)
                count_negative = projections.size - count_positive
                angle = (
                    np.pi / 2
                    if count_positive < count_negative
                    else -np.pi / 2
                )

        # Vectorized calculation of differences in positions
        diff = state_matrix_filtered[:, :3] - current_state[:3]
        distances = np.linalg.norm(diff, axis=1)
        # Exclude the current object (distance near zero)
        mask = distances > 1e-6
        diff = diff[mask]
        distances = distances[mask]
        # If no neighbors, return current velocity
        if diff.shape[0] == 0:
            return current_state[3:6]

        # Normalize differences while avoiding division by zero
        diff_normalized = diff / np.maximum(distances[:, None], 1e-6)

        # Compute repulsion force:
        # For each neighbor: -diff_normalized / ((distance + 1 - safety_radius)^2)
        factors = 1.0 / ((distances + 1 - self.safety_radius) ** 2)
        repulsion_force = -np.sum(diff_normalized * factors[:, None], axis=0)

        # Compute unstable vector if conditions are met
        unstable_vector = np.zeros(3)
        # Global conditions: the object's speed is near zero and error exceeds safety_radius + 0.2
        if (
            np.allclose(np.linalg.norm(current_state[3:6]), 0, atol=0.1)
            and current_error_norm > self.safety_radius
        ):
            # For neighbors within safety_radius + 0.1, apply a rotation
            mask_unstable = distances < self.safety_radius
            if np.any(mask_unstable):
                unstable_components = diff_normalized[mask_unstable] * 0.3
                # Apply rotation to each vector
                rotated_vectors = rot_v(unstable_components, angle, axis)
                unstable_vector = np.sum(rotated_vectors, axis=0)

        # Compute the new velocity vector
        new_velocity = (
            current_state[3:6] * self.current_velocity_weight
            + self.repulsion_weight * repulsion_force
            + self.unstable_weight * unstable_vector
        )
        new_velocity = limit_acceleration(
            current_state[3:6], new_velocity, self.max_acceleration, dt=dt
        )
        return saturation(new_velocity, self.max_speed)

    def state_filter(
        self,
        current_state: NDArray,
        state_matrix: NDArray,
        filter_radius: float,
    ) -> NDArray:
        """
        Vectorized filtering: returns states that are within a specified radius of current_state.

        :param current_state: The current state vector.
        :param state_matrix: Matrix of states (n x 6).
        :param filter_radius: Filtering radius.
        :return: Filtered state matrix.
        """
        diff = state_matrix[:, :3] - current_state[:3]
        distances_sq = np.sum(diff**2, axis=1)
        mask = distances_sq <= filter_radius**2
        return state_matrix[mask]

    def term_count_unstable_vector(
        self, dist_to_other_drone: float, error_norm: float, speed: NDArray
    ) -> bool:
        """
        Determine whether to add an unstable component.

        (Kept for backward compatibility; conditions are now applied in a vectorized manner in compute_velocity_direction)

        :param dist_to_other_drone: Distance to the other drone.
        :param error: Error magnitude.
        :param speed: Speed vector.
        :return: Boolean indicating if unstable vector should be added.
        """
        return (
            dist_to_other_drone < self.safety_radius
            and np.allclose(np.linalg.norm(speed), 0, atol=0.1)
            and error_norm > self.safety_radius
        )
