import matplotlib.pyplot as plt
import numpy as np
from lokky.pionmath import SSolver  # import the solver class from the lokky module


def main():
    # Number of points
    n_points = 10

    # Set up parameters for SSolver
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
        "safety_radius": 1.0,
        "max_acceleration": 1.0,
        "max_speed": 1.0,
        "unstable_radius": 2,
    }

    solver = SSolver(params)

    # Generate random positions and velocities (with z=0)
    np.random.seed(42)
    positions = np.random.rand(n_points, 3) * np.array([10, 10, 0])
    velocities = np.zeros((n_points, 3))
    velocities[:, 2] = 0

    # Form the state_matrix: first 3 columns are positions, next 3 are velocities
    state_matrix = np.hstack([positions, velocities])

    # Form the target_matrix: displaced positions and slightly modified velocities (with z=0)
    target_positions = positions + np.random.rand(n_points, 3) * 1.0
    target_positions[:, 2] = 0
    target_velocities = np.zeros((n_points, 3))
    target_velocities[:, 2] = 0
    target_matrix = np.hstack([target_positions, target_velocities])

    # Set the time step
    dt = 0.1

    # Calculate the initial control signals
    control_signals = solver.solve_for_all(state_matrix, target_matrix, dt)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display state positions and target positions
    state_scatter = ax.scatter(
        positions[:, 0], positions[:, 1], color="black", label="State Positions"
    )
    target_scatter = ax.scatter(
        target_positions[:, 0],
        target_positions[:, 1],
        color="blue",
        marker="x",
        label="Target Positions",
    )

    # Lists to store arrows
    error_arrows = []
    ctrl_arrows = []

    # Function to draw arrows
    def draw_arrows():
        # Remove old arrows
        for arrow in error_arrows + ctrl_arrows:
            arrow.remove()
        error_arrows.clear()
        ctrl_arrows.clear()

        for i in range(n_points):
            # Error arrow (from state to target)
            error_vec = target_positions[i, :2] - positions[i, :2]
            ea = ax.arrow(
                positions[i, 0],
                positions[i, 1],
                error_vec[0],
                error_vec[1],
                color="orange",
                width=0.05,
                head_width=0.3,
                alpha=0.6,
                linestyle="dashed",
            )
            error_arrows.append(ea)

            # Control signal arrow
            ctrl = control_signals[i, :2]
            ca = ax.arrow(
                positions[i, 0],
                positions[i, 1],
                ctrl[0],
                ctrl[1],
                color="red",
                width=0.05,
                head_width=0.3,
                alpha=0.8,
            )
            ctrl_arrows.append(ca)
        fig.canvas.draw_idle()

    # Draw the initial arrows
    draw_arrows()

    # Set up axis limits, labels, title, legend, and grid
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Visualization of Control Signals (solve)")
    ax.legend()
    ax.grid(True)

    # Variable to store the index of the selected point
    selected_index = None

    # Event handler for mouse button press
    def on_press(event):
        nonlocal selected_index
        if event.inaxes != ax:
            return
        # Calculate distances from the click to each point
        distances = np.hypot(positions[:, 0] - event.xdata, positions[:, 1] - event.ydata)
        # If the closest point is within the threshold, select it
        if distances.min() < 0.5:
            selected_index = np.argmin(distances)

    # Event handler for mouse movement
    def on_motion(event):
        nonlocal selected_index, positions, state_matrix, control_signals
        if selected_index is None or event.inaxes != ax:
            return
        # Update the position of the selected point
        positions[selected_index, 0] = event.xdata
        positions[selected_index, 1] = event.ydata
        state_matrix[selected_index, :2] = event.xdata, event.ydata

        # Recalculate control signals
        control_signals = solver.solve_for_all(state_matrix, target_matrix, dt)

        # Update the state scatter plot
        state_scatter.set_offsets(positions[:, :2])

        # Redraw the arrows
        draw_arrows()

    # Event handler for mouse button release
    def on_release(event):
        nonlocal selected_index
        selected_index = None

    # Connect event handlers to the figure canvas
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    plt.show()


if __name__ == "__main__":
    main()
