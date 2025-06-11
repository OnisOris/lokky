![lokky-analyze plot](https://github.com/OnisOris/lokky/blob/main/img/img.gif?raw=true)

# lokky
Swarm algorithms module

# Installation

## Base install

```
pip install git+https://github.com/OnisOris/lokky
```

```
pip install lokky
```

## Install with visualisation

```
pip install "lokky[plotting]"
```

# Examples

```python
import numpy as np
from lokky.pionmath import SSolver
n_points = 1
params = {
    "kp": np.ones((n_points, 6)),
    "ki": np.zeros((n_points, 6)),
    "kd": np.ones((n_points, 6)) * 0,
    "attraction_weight": 1.0,
    "cohesion_weight": 1.0,
    "alignment_weight": 1.0,
    "repulsion_weight": 1.0,
    "unstable_weight": 1.0,
    "noise_weight": 1.0,
    "current_velocity_weight": 0.0,
    "safety_radius": 1.0,
    "max_acceleration": 1.0,
    "max_speed": 1.0,
    "unstable_radius": 2,
}
solver = SSolver(params)
drone1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
drone2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
target_matrix = np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
state_matrix = np.array([drone1, drone2])
new_velocities = solver.solve_for_all(state_matrix, target_matrix, 0.1)
```

# Testing script

After installation (*lokky[plotting]*) run scripts:

```
lokky-analyze --n 10
```
