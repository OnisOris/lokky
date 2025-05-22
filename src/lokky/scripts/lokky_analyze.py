import argparse

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle, Rectangle
except ImportError:
    plt = None
    print(
        "Warning: matplotlib is not installed. Plotting functions will be disabled."
    )

from mpl_toolkits.mplot3d import proj3d

from lokky.pionmath import SSolver


def draw_boundary_2d(ax, x_min, x_max, y_min, y_max, color="black"):
    rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        edgecolor=color,
        linestyle="--",
    )
    ax.add_patch(rect)


def draw_boundary_3d(
    ax, x_min, x_max, y_min, y_max, z_min, z_max, color="black"
):
    corners = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ]
    )
    lines = [
        [corners[0], corners[1]],
        [corners[1], corners[2]],
        [corners[2], corners[3]],
        [corners[3], corners[0]],
        [corners[4], corners[5]],
        [corners[5], corners[6]],
        [corners[6], corners[7]],
        [corners[7], corners[4]],
        [corners[0], corners[4]],
        [corners[1], corners[5]],
        [corners[2], corners[6]],
        [corners[3], corners[7]],
    ]
    for line in lines:
        xs, ys, zs = zip(*line)
        ax.plot(xs, ys, zs, color=color, linestyle="--", linewidth=1)


def main():
    parser = argparse.ArgumentParser(
        description="Enter number of simulated points"
    )
    parser.add_argument("--n", default=4, help="Number of points")
    args = parser.parse_args()
    n_points = int(args.n)

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
    safety_radius = params["safety_radius"]
    solver = SSolver(params, count_of_objects=n_points)

    np.random.seed(42)
    positions = np.random.rand(n_points, 3) * np.array([10, 10, 0])
    velocities = np.zeros((n_points, 3))
    state_matrix = np.hstack([positions, velocities])
    target_positions = positions + np.random.rand(n_points, 3) * 1.0
    target_matrix = np.hstack([target_positions, np.zeros((n_points, 3))])
    dt = 0.1

    control_signals = solver.solve_for_all(state_matrix, target_matrix, dt)

    x_min, x_max = -2, 12
    y_min, y_max = -2, 12
    z_min, z_max = -2, 12

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_front = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")

    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except Exception:
        try:
            manager.window.state("zoomed")
        except Exception:
            pass

    ax_top.set_title("Top View (XY)")
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Y")
    ax_top.grid(True)
    ax_top.set_xlim(x_min, x_max)
    ax_top.set_ylim(y_min, y_max)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.autoscale(enable=False)

    ax_side.set_title("Side View (XZ)")
    ax_side.set_xlabel("X")
    ax_side.set_ylabel("Z")
    ax_side.grid(True)
    ax_side.set_xlim(x_min, x_max)
    ax_side.set_ylim(z_min, z_max)
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.autoscale(enable=False)

    ax_front.set_title("Front View (YZ)")
    ax_front.set_xlabel("Y")
    ax_front.set_ylabel("Z")
    ax_front.grid(True)
    ax_front.set_xlim(y_min, y_max)
    ax_front.set_ylim(z_min, z_max)
    ax_front.set_aspect("equal", adjustable="box")
    ax_front.autoscale(enable=False)

    ax_3d.set_title("3D View")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_xlim3d(x_min, x_max)
    ax_3d.set_ylim3d(y_min, y_max)
    ax_3d.set_zlim3d(z_min, z_max)
    ax_3d.grid(True)
    ax_3d.view_init(elev=30, azim=-60)
    ax_3d.set_navigate(False)
    ax_3d.mouse_init = lambda: None

    state_scatter_top = ax_top.scatter(
        positions[:, 0], positions[:, 1], color="black", label="State"
    )
    target_scatter_top = ax_top.scatter(
        target_positions[:, 0],
        target_positions[:, 1],
        color="blue",
        marker="x",
        label="Target",
    )
    state_scatter_side = ax_side.scatter(
        positions[:, 0], positions[:, 2], color="black", label="State"
    )
    target_scatter_side = ax_side.scatter(
        target_positions[:, 0],
        target_positions[:, 2],
        color="blue",
        marker="x",
        label="Target",
    )
    state_scatter_front = ax_front.scatter(
        positions[:, 1], positions[:, 2], color="black", label="State"
    )
    target_scatter_front = ax_front.scatter(
        target_positions[:, 1],
        target_positions[:, 2],
        color="blue",
        marker="x",
        label="Target",
    )
    state_scatter_3d = ax_3d.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="black",
        label="State",
    )
    target_scatter_3d = ax_3d.scatter(
        target_positions[:, 0],
        target_positions[:, 1],
        target_positions[:, 2],
        color="blue",
        marker="x",
        label="Target",
    )

    error_arrows_top, ctrl_arrows_top = [], []
    error_arrows_side, ctrl_arrows_side = [], []
    error_arrows_front, ctrl_arrows_front = [], []
    error_quivers_3d, ctrl_quivers_3d = [], []
    safety_circles_top, safety_circles_side, safety_circles_front = [], [], []
    safety_spheres_3d = []
    selected_index = None
    selected_view = None
    last_y = None

    def update_safety_zones():
        nonlocal \
            safety_circles_top, \
            safety_circles_side, \
            safety_circles_front, \
            safety_spheres_3d
        for patch in safety_circles_top:
            patch.remove()
        for patch in safety_circles_side:
            patch.remove()
        for patch in safety_circles_front:
            patch.remove()
        safety_circles_top.clear()
        safety_circles_side.clear()
        safety_circles_front.clear()
        for surf in safety_spheres_3d:
            surf.remove()
        safety_spheres_3d.clear()
        for i in range(n_points):
            if selected_index == i:
                face_color = "red"
                edge_color = "darkred"
                lw = 2
            else:
                face_color = "green"
                edge_color = "none"
                lw = 1
            circle_top = Circle(
                (positions[i, 0], positions[i, 1]),
                safety_radius,
                facecolor=face_color,
                edgecolor=edge_color,
                alpha=0.3,
                lw=lw,
            )
            ax_top.add_patch(circle_top)
            safety_circles_top.append(circle_top)
            circle_side = Circle(
                (positions[i, 0], positions[i, 2]),
                safety_radius,
                facecolor=face_color,
                edgecolor=edge_color,
                alpha=0.3,
                lw=lw,
            )
            ax_side.add_patch(circle_side)
            safety_circles_side.append(circle_side)
            circle_front = Circle(
                (positions[i, 1], positions[i, 2]),
                safety_radius,
                facecolor=face_color,
                edgecolor=edge_color,
                alpha=0.3,
                lw=lw,
            )
            ax_front.add_patch(circle_front)
            safety_circles_front.append(circle_front)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = positions[i, 0] + safety_radius * np.outer(
                np.cos(u), np.sin(v)
            )
            y = positions[i, 1] + safety_radius * np.outer(
                np.sin(u), np.sin(v)
            )
            z = positions[i, 2] + safety_radius * np.outer(
                np.ones_like(u), np.cos(v)
            )
            color_3d = "red" if selected_index == i else "green"
            sphere = ax_3d.plot_surface(
                x, y, z, color=color_3d, alpha=0.2, shade=False
            )
            safety_spheres_3d.append(sphere)
        fig.canvas.draw_idle()

    def draw_arrows_2d():
        nonlocal \
            error_arrows_top, \
            ctrl_arrows_top, \
            error_arrows_side, \
            ctrl_arrows_side, \
            error_arrows_front, \
            ctrl_arrows_front
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
            a = ax_top.arrow(
                positions[i, 0],
                positions[i, 1],
                err[0],
                err[1],
                color="orange",
                width=0.05,
                head_width=0.3,
                alpha=0.6,
                linestyle="dashed",
            )
            error_arrows_top.append(a)
            a = ax_top.arrow(
                positions[i, 0],
                positions[i, 1],
                ctrl[0],
                ctrl[1],
                color="red",
                width=0.05,
                head_width=0.3,
                alpha=0.8,
            )
            ctrl_arrows_top.append(a)
            a = ax_side.arrow(
                positions[i, 0],
                positions[i, 2],
                err[0],
                err[2],
                color="orange",
                width=0.05,
                head_width=0.3,
                alpha=0.6,
                linestyle="dashed",
            )
            error_arrows_side.append(a)
            a = ax_side.arrow(
                positions[i, 0],
                positions[i, 2],
                ctrl[0],
                ctrl[2],
                color="red",
                width=0.05,
                head_width=0.3,
                alpha=0.8,
            )
            ctrl_arrows_side.append(a)
            a = ax_front.arrow(
                positions[i, 1],
                positions[i, 2],
                err[1],
                err[2],
                color="orange",
                width=0.05,
                head_width=0.3,
                alpha=0.6,
                linestyle="dashed",
            )
            error_arrows_front.append(a)
            a = ax_front.arrow(
                positions[i, 1],
                positions[i, 2],
                ctrl[1],
                ctrl[2],
                color="red",
                width=0.05,
                head_width=0.3,
                alpha=0.8,
            )
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
            q = ax_3d.quiver(
                positions[i, 0],
                positions[i, 1],
                positions[i, 2],
                err[0],
                err[1],
                err[2],
                color="orange",
                arrow_length_ratio=0.1,
            )
            error_quivers_3d.append(q)
            q = ax_3d.quiver(
                positions[i, 0],
                positions[i, 1],
                positions[i, 2],
                ctrl[0],
                ctrl[1],
                ctrl[2],
                color="red",
                arrow_length_ratio=0.1,
            )
            ctrl_quivers_3d.append(q)
        fig.canvas.draw_idle()

    def update_scatter():
        state_scatter_top.set_offsets(positions[:, [0, 1]])
        target_scatter_top.set_offsets(target_positions[:, [0, 1]])
        state_scatter_side.set_offsets(positions[:, [0, 2]])
        target_scatter_side.set_offsets(target_positions[:, [0, 2]])
        state_scatter_front.set_offsets(positions[:, [1, 2]])
        target_scatter_front.set_offsets(target_positions[:, [1, 2]])
        state_scatter_3d._offsets3d = (
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )
        target_scatter_3d._offsets3d = (
            target_positions[:, 0],
            target_positions[:, 1],
            target_positions[:, 2],
        )
        fig.canvas.draw_idle()

    def get_2d_coords(ax, x, y, z):
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        return ax.transData.transform((x2, y2))

    def get_3d_from_2d(ax, x_pixel, y_pixel, z_fixed):
        inv = ax.transData.inverted()
        x2d, y2d = inv.transform((x_pixel, y_pixel))
        return x2d, y2d

    def on_press(event):
        nonlocal selected_index, selected_view, last_y
        if event.inaxes not in [ax_top, ax_side, ax_front, ax_3d]:
            return
        click = np.array([event.xdata, event.ydata])
        distances = []
        if event.inaxes == ax_top:
            for pos in positions:
                distances.append(
                    np.hypot(pos[0] - click[0], pos[1] - click[1])
                )
            current_view = "top"
        elif event.inaxes == ax_side:
            for pos in positions:
                distances.append(
                    np.hypot(pos[0] - click[0], pos[2] - click[1])
                )
            current_view = "side"
        elif event.inaxes == ax_front:
            for pos in positions:
                distances.append(
                    np.hypot(pos[1] - click[0], pos[2] - click[1])
                )
            current_view = "front"
        elif event.inaxes == ax_3d:
            click_screen = np.array([event.x, event.y])
            for pos in positions:
                proj = get_2d_coords(ax_3d, pos[0], pos[1], pos[2])
                distances.append(
                    np.hypot(
                        proj[0] - click_screen[0], proj[1] - click_screen[1]
                    )
                )
            current_view = "3d"
        distances = np.array(distances)
        threshold = 15 if current_view == "3d" else 0.5
        if distances.min() < threshold:
            selected_index = int(np.argmin(distances))
            selected_view = current_view
            last_y = event.y
            update_safety_zones()

    def on_motion(event):
        nonlocal \
            selected_index, \
            positions, \
            state_matrix, \
            control_signals, \
            last_y
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
            if event.key is not None and (
                "shift" in event.key.lower() or "control" in event.key.lower()
            ):
                if last_y is not None and event.y is not None:
                    dy = event.y - last_y
                    scale = 0.05
                    modifier = 1 if "shift" in event.key.lower() else -1
                    positions[selected_index, 2] += dy * scale * modifier
                    state_matrix[selected_index, 2] = positions[
                        selected_index, 2
                    ]
            else:
                new_x, new_y = get_3d_from_2d(
                    ax_3d, event.x, event.y, positions[selected_index, 2]
                )
                positions[selected_index, 0] = new_x
                positions[selected_index, 1] = new_y
                state_matrix[selected_index, :2] = [new_x, new_y]
        control_signals[:] = solver.solve_for_all(
            state_matrix, target_matrix, dt
        )
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

    for ax in [ax_top, ax_side, ax_front, ax_3d]:
        ax.figure.canvas.mpl_connect("button_press_event", on_press)
        ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)
        ax.figure.canvas.mpl_connect("button_release_event", on_release)

    update_safety_zones()

    draw_boundary_2d(ax_top, x_min, x_max, y_min, y_max)
    draw_boundary_2d(ax_side, x_min, x_max, z_min, z_max)
    draw_boundary_2d(ax_front, y_min, y_max, z_min, z_max)
    draw_boundary_3d(ax_3d, x_min, x_max, y_min, y_max, z_min, z_max)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
