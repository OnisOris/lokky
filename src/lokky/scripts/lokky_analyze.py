import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import Circle
from lokky.pionmath import SSolver  # import the SSolver class from the lokky package

def main():
    # Number of points
    n_points = 10

    # Parameters for SSolver (safety_radius is used for drawing the safety zones)
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
    safety_radius = params["safety_radius"]
    solver = SSolver(params)

    # Generate random positions and velocities
    np.random.seed(42)
    positions = np.random.rand(n_points, 3) * np.array([10, 10, 0])
    velocities = np.zeros((n_points, 3))

    # Form the state_matrix (positions + velocities)
    state_matrix = np.hstack([positions, velocities])

    # Form the target_matrix (target positions with an offset)
    target_positions = positions + np.random.rand(n_points, 3) * 1.0
    target_matrix = np.hstack([target_positions, np.zeros((n_points, 3))])

    # Time step
    dt = 0.1

    # Compute control signals
    control_signals = solver.solve(state_matrix, target_matrix, dt)

    # Create a figure with 4 subplots: Top, Side, Front, and 3D view
    fig = plt.figure(figsize=(12, 10))
    ax_top = fig.add_subplot(221)      # Top view (XY)
    ax_front = fig.add_subplot(222)    # Front view (YZ)
    ax_side = fig.add_subplot(223)     # Side view (XZ)
    ax_3d = fig.add_subplot(224, projection="3d")  # 3D view

    # Disable 3D navigation
    ax_3d.set_navigate(False)
    ax_3d.mouse_init = lambda: None

    # Set up 2D axes
    ax_top.set_title("Top view (XY)")
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Y")
    ax_top.set_xlim(-2, 12)
    ax_top.set_ylim(-2, 12)
    ax_top.grid(True)

    ax_side.set_title("Side view (XZ)")
    ax_side.set_xlabel("X")
    ax_side.set_ylabel("Z")
    ax_side.set_xlim(-2, 12)
    ax_side.set_ylim(-2, 12)
    ax_side.grid(True)

    ax_front.set_title("Front view (YZ)")
    ax_front.set_xlabel("Y")
    ax_front.set_ylabel("Z")
    ax_front.set_xlim(-2, 12)
    ax_front.set_ylim(-2, 12)
    ax_front.grid(True)

    # Set up 3D axes
    ax_3d.set_title("3D view")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_xlim(-2, 12)
    ax_3d.set_ylim(-2, 12)
    ax_3d.set_zlim(-2, 12)
    ax_3d.grid(True)
    ax_3d.view_init(elev=30, azim=-60)

    # Create scatter plots for all views
    state_scatter_top = ax_top.scatter(positions[:, 0], positions[:, 1],
                                       color="black", label="State")
    target_scatter_top = ax_top.scatter(target_positions[:, 0],
                                        target_positions[:, 1],
                                        color="blue", marker="x", label="Target")

    state_scatter_side = ax_side.scatter(positions[:, 0], positions[:, 2],
                                         color="black", label="State")
    target_scatter_side = ax_side.scatter(target_positions[:, 0],
                                          target_positions[:, 2],
                                          color="blue", marker="x", label="Target")

    state_scatter_front = ax_front.scatter(positions[:, 1], positions[:, 2],
                                           color="black", label="State")
    target_scatter_front = ax_front.scatter(target_positions[:, 1],
                                             target_positions[:, 2],
                                             color="blue", marker="x", label="Target")

    state_scatter_3d = ax_3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                     color="black", label="State")
    target_scatter_3d = ax_3d.scatter(target_positions[:, 0], target_positions[:, 1],
                                      target_positions[:, 2],
                                      color="blue", marker="x", label="Target")

    # Lists for arrows in 2D views
    error_arrows_top, ctrl_arrows_top = [], []
    error_arrows_side, ctrl_arrows_side = [], []
    error_arrows_front, ctrl_arrows_front = [], []

    # Lists for quiver arrows in the 3D view
    error_quivers_3d, ctrl_quivers_3d = [], []

    # Lists for safety zones (patches for 2D and surfaces for 3D)
    safety_circles_top, safety_circles_side, safety_circles_front = [], [], []
    safety_spheres_3d = []

    # Local variables for the selected point and active view
    selected_index = None
    selected_view = None  # can be "top", "side", "front", or "3d"
    last_y = None       # for vertical offsets in 3D

    # Function to update safety zones (draws a semi-transparent circle/sphere around each point)
    def update_safety_zones():
        nonlocal safety_circles_top, safety_circles_side, safety_circles_front, safety_spheres_3d
        # Remove existing safety zones (2D circles)
        for patch in safety_circles_top:
            patch.remove()
        for patch in safety_circles_side:
            patch.remove()
        for patch in safety_circles_front:
            patch.remove()
        safety_circles_top.clear()
        safety_circles_side.clear()
        safety_circles_front.clear()
        # Remove existing safety zones (3D spheres)
        for surf in safety_spheres_3d:
            surf.remove()
        safety_spheres_3d.clear()
        # For each point, create circles in the 2D views and a sphere in the 3D view
        for i in range(n_points):
            # For selected points, use a highlighted color and thicker edge
            if selected_index == i:
                face_color = "red"
                edge_color = "darkred"
                lw = 2
            else:
                face_color = "green"
                edge_color = "none"
                lw = 1
            # Top view (XY)
            circle_top = Circle((positions[i, 0], positions[i, 1]),
                                safety_radius,
                                facecolor=face_color,
                                edgecolor=edge_color,
                                alpha=0.3,
                                lw=lw)
            ax_top.add_patch(circle_top)
            safety_circles_top.append(circle_top)
            # Side view (XZ)
            circle_side = Circle((positions[i, 0], positions[i, 2]),
                                 safety_radius,
                                 facecolor=face_color,
                                 edgecolor=edge_color,
                                 alpha=0.3,
                                 lw=lw)
            ax_side.add_patch(circle_side)
            safety_circles_side.append(circle_side)
            # Front view (YZ)
            circle_front = Circle((positions[i, 1], positions[i, 2]),
                                  safety_radius,
                                  facecolor=face_color,
                                  edgecolor=edge_color,
                                  alpha=0.3,
                                  lw=lw)
            ax_front.add_patch(circle_front)
            safety_circles_front.append(circle_front)
            # 3D view: draw a sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = positions[i, 0] + safety_radius * np.outer(np.cos(u), np.sin(v))
            y = positions[i, 1] + safety_radius * np.outer(np.sin(u), np.sin(v))
            z = positions[i, 2] + safety_radius * np.outer(np.ones_like(u), np.cos(v))
            # For selected points, use a different color
            color_3d = "red" if selected_index == i else "green"
            sphere = ax_3d.plot_surface(x, y, z, color=color_3d, alpha=0.2, shade=False)
            safety_spheres_3d.append(sphere)
        fig.canvas.draw_idle()

    # Functions to draw arrows and update scatter plots (as before)
    def draw_arrows_2d():
        nonlocal error_arrows_top, ctrl_arrows_top, error_arrows_side, ctrl_arrows_side, error_arrows_front, ctrl_arrows_front
        # Remove old arrows
        for arr in error_arrows_top + ctrl_arrows_top:
            arr.remove()
        for arr in error_arrows_side + ctrl_arrows_side:
            arr.remove()
        for arr in error_arrows_front + ctrl_arrows_front:
            arr.remove()
        error_arrows_top.clear()
        ctrl_arrows_top.clear()
        error_arrows_side.clear()
        ctrl_arrows_side.clear()
        error_arrows_front.clear()
        ctrl_arrows_front.clear()

        for i in range(n_points):
            err = target_positions[i] - positions[i]
            ctrl = control_signals[i, :3]
            # Top view (XY)
            a = ax_top.arrow(positions[i, 0], positions[i, 1],
                             err[0], err[1],
                             color="orange", width=0.05, head_width=0.3,
                             alpha=0.6, linestyle="dashed")
            error_arrows_top.append(a)
            a = ax_top.arrow(positions[i, 0], positions[i, 1],
                             ctrl[0], ctrl[1],
                             color="red", width=0.05, head_width=0.3,
                             alpha=0.8)
            ctrl_arrows_top.append(a)
            # Side view (XZ)
            a = ax_side.arrow(positions[i, 0], positions[i, 2],
                              err[0], err[2],
                              color="orange", width=0.05, head_width=0.3,
                              alpha=0.6, linestyle="dashed")
            error_arrows_side.append(a)
            a = ax_side.arrow(positions[i, 0], positions[i, 2],
                              ctrl[0], ctrl[2],
                              color="red", width=0.05, head_width=0.3,
                              alpha=0.8)
            ctrl_arrows_side.append(a)
            # Front view (YZ)
            a = ax_front.arrow(positions[i, 1], positions[i, 2],
                               err[1], err[2],
                               color="orange", width=0.05, head_width=0.3,
                               alpha=0.6, linestyle="dashed")
            error_arrows_front.append(a)
            a = ax_front.arrow(positions[i, 1], positions[i, 2],
                               ctrl[1], ctrl[2],
                               color="red", width=0.05, head_width=0.3,
                               alpha=0.8)
            ctrl_arrows_front.append(a)
        fig.canvas.draw_idle()

    def draw_arrows_3d():
        nonlocal error_quivers_3d, ctrl_quivers_3d
        for q in error_quivers_3d + ctrl_quivers_3d:
            q.remove()
        error_quivers_3d.clear()
        ctrl_quivers_3d.clear()
        for i in range(n_points):
            err = target_positions[i] - positions[i]
            ctrl = control_signals[i, :3]
            q = ax_3d.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                             err[0], err[1], err[2],
                             color="orange", arrow_length_ratio=0.1)
            error_quivers_3d.append(q)
            q = ax_3d.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                             ctrl[0], ctrl[1], ctrl[2],
                             color="red", arrow_length_ratio=0.1)
            ctrl_quivers_3d.append(q)
        fig.canvas.draw_idle()

    def update_scatter():
        state_scatter_top.set_offsets(positions[:, [0, 1]])
        target_scatter_top.set_offsets(target_positions[:, [0, 1]])

        state_scatter_side.set_offsets(positions[:, [0, 2]])
        target_scatter_side.set_offsets(target_positions[:, [0, 2]])

        state_scatter_front.set_offsets(positions[:, [1, 2]])
        target_scatter_front.set_offsets(target_positions[:, [1, 2]])

        # For 3D scatter, update data manually:
        state_scatter_3d._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        target_scatter_3d._offsets3d = (target_positions[:, 0], target_positions[:, 1], target_positions[:, 2])
        fig.canvas.draw_idle()

    # Helper functions for 3D projection
    def get_2d_coords(ax, x, y, z):
        """Projects 3D coordinates to 2D screen coordinates."""
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        return ax.transData.transform((x2, y2))

    def get_3d_from_2d(ax, x_pixel, y_pixel, z_fixed):
        """Approximate conversion of 2D coordinates to 3D (on a plane with fixed z)."""
        inv = ax.transData.inverted()
        x2d, y2d = inv.transform((x_pixel, y_pixel))
        return x2d, y2d

    # Mouse event handlers
    def on_press(event):
        nonlocal selected_index, selected_view, last_y
        if event.inaxes not in [ax_top, ax_side, ax_front, ax_3d]:
            return
        click = np.array([event.xdata, event.ydata])
        distances = []
        if event.inaxes == ax_top:
            for pos in positions:
                distances.append(np.hypot(pos[0] - click[0], pos[1] - click[1]))
            current_view = "top"
        elif event.inaxes == ax_side:
            for pos in positions:
                distances.append(np.hypot(pos[0] - click[0], pos[2] - click[1]))
            current_view = "side"
        elif event.inaxes == ax_front:
            for pos in positions:
                distances.append(np.hypot(pos[1] - click[0], pos[2] - click[1]))
            current_view = "front"
        elif event.inaxes == ax_3d:
            click_screen = np.array([event.x, event.y])
            for pos in positions:
                proj = get_2d_coords(ax_3d, pos[0], pos[1], pos[2])
                distances.append(np.hypot(proj[0] - click_screen[0], proj[1] - click_screen[1]))
            current_view = "3d"
        distances = np.array(distances)
        threshold = 15 if current_view == "3d" else 0.5
        if distances.min() < threshold:
            selected_index = int(np.argmin(distances))
            selected_view = current_view
            last_y = event.y
            update_safety_zones()

    def on_motion(event):
        nonlocal selected_index, positions, state_matrix, control_signals, last_y
        if selected_index is None or event.inaxes is None:
            return

        if event.inaxes == ax_top and selected_view == "top":
            positions[selected_index, 0] = event.xdata
            positions[selected_index, 1] = event.ydata
            state_matrix[selected_index, :2] = [event.xdata, event.ydata]
        elif event.inaxes == ax_side and selected_view == "side":
            positions[selected_index, 0] = event.xdata
            positions[selected_index, 2] = event.ydata
            state_matrix[selected_index, 0] = event.xdata
            state_matrix[selected_index, 2] = event.ydata
        elif event.inaxes == ax_front and selected_view == "front":
            positions[selected_index, 1] = event.xdata
            positions[selected_index, 2] = event.ydata
            state_matrix[selected_index, 1] = event.xdata
            state_matrix[selected_index, 2] = event.ydata
        elif event.inaxes == ax_3d and selected_view == "3d":
            if event.key is not None and ("shift" in event.key.lower() or "control" in event.key.lower()):
                if last_y is not None and event.y is not None:
                    dy = event.y - last_y
                    scale = 0.05
                    modifier = 1 if "shift" in event.key.lower() else -1
                    positions[selected_index, 2] += dy * scale * modifier
                    state_matrix[selected_index, 2] = positions[selected_index, 2]
            else:
                new_x, new_y = get_3d_from_2d(ax_3d, event.x, event.y, positions[selected_index, 2])
                positions[selected_index, 0] = new_x
                positions[selected_index, 1] = new_y
                state_matrix[selected_index, :2] = [new_x, new_y]
        control_signals[:] = solver.solve(state_matrix, target_matrix, dt)
        update_scatter()
        draw_arrows_2d()
        draw_arrows_3d()
        last_y = event.y
        update_safety_zones()

    def on_release(event):
        nonlocal selected_index, selected_view
        selected_index = None
        selected_view = None
        update_safety_zones()

    # Connect event handlers to the axes
    for ax in [ax_top, ax_side, ax_front, ax_3d]:
        ax.figure.canvas.mpl_connect("button_press_event", on_press)
        ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)
        ax.figure.canvas.mpl_connect("button_release_event", on_release)

    update_safety_zones()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
