import numpy as np
from lokky.pionmath import SSolver


def test_pionmath():
    params = {
        "kp": np.eye(2, 6),
        "ki": np.zeros((2, 6)),
        "kd": np.eye(2, 6) * 0.1,
        "attraction_weight": 0.1,
        "cohesion_weight": 0.1,
        "alignment_weight": 0.1,
        "repulsion_weight": 5.0,
        "unstable_weight": 0.5,
        "noise_weight": 0.1,
        "safety_radius": 1.5,
        "max_acceleration": 0.3,
        "max_speed": 0.5,
        "unstable_radius": 2
    }
    solver = SSolver(params)
    drone1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    drone2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target_matrix = np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_matrix = np.array([drone1, drone2])
    new_velocities = solver.solve(state_matrix, target_matrix, 0.1)
    assert new_velocities.shape == (2, 3)


if __name__ == "__main__":
    test_pionmath()
