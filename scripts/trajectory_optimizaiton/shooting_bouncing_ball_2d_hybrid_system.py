import argparse

import matplotlib.pyplot as plt
import numpy as np

from hybrid_tools.basic_hybrid_systems import bouncing_ball_2d
from hybrid_tools.shooting_method_optimizer import (
    ShootingMethodOptimizer,
    ShootingMethodOptimizerConfig,
    ShootingMethodOptimizerResult,
)


def plot_optimization_results(
    result: ShootingMethodOptimizerResult,
    target: np.ndarray,
    title: str = "Trajectory Optimization",
    save_path: str = "shooting_ball_hybrid_system.png",
    show_all_iterations: bool = True,
):
    """
    Plot optimization trajectories.

    Args:
        result: ShootingMethodOptimizerResult from optimizer
        target: Target position [x, y]
        title: Plot title
        save_path: Path to save figure
        show_all_iterations: If True, show all iteration trajectories
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if show_all_iterations and len(result.trajectory_history) > 0:
        n_iters = len(result.trajectory_history)
        for i, states in enumerate(result.trajectory_history):
            alpha_val = min(0.2 + 0.8 * (i + 1) / n_iters, 1.0)
            color = plt.cm.viridis(i / n_iters)
            label = f"Iter {i}" if i == 0 or i == n_iters - 1 else None
            ax.plot(states[:, 0], states[:, 1], alpha=alpha_val, color=color, label=label)
    else:
        ax.plot(
            result.final_trajectory[:, 0],
            result.final_trajectory[:, 1],
            "b-",
            linewidth=2,
            label="Final trajectory",
        )

    ax.scatter(*target, color="red", marker="X", s=200, label="Target", zorder=5)
    ax.scatter(
        result.final_trajectory[0, 0],
        result.final_trajectory[0, 1],
        color="green",
        marker="o",
        s=100,
        label="Start",
        zorder=5,
    )
    ax.scatter(
        result.final_trajectory[-1, 0],
        result.final_trajectory[-1, 1],
        color="blue",
        marker="s",
        s=100,
        label="End",
        zorder=5,
    )

    ax.set_title(f"{title}\nConverged: {result.converged}, Iterations: {result.iterations}")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to: {save_path}")


def main():
    """Example: 2D bouncing ball trajectory optimization."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="2D bouncing ball trajectory optimization using shooting method"
    )
    parser.add_argument(
        "--init-velocity",
        type=float,
        default=8.0,
        help="Initial vertical velocity guess (m/s) (default: 8.0)",
    )
    parser.add_argument(
        "--ground-height",
        type=float,
        default=0.0,
        help="Ground height for bouncing (m) (default: 0.0). Make it a very negative number if you want a system to not look hybrid.",
    )
    args = parser.parse_args()

    # Setup hybrid system
    ground_height = args.ground_height
    hybrid_system = bouncing_ball_2d()
    for key in hybrid_system.noises.keys():
        hybrid_system.noises[key].W *= 0  # No process noise
    parameters = np.array([0.8, 9.81, ground_height])  # [e, g, h]

    # Create optimizer
    optimizer = ShootingMethodOptimizer(
        hybrid_system=hybrid_system,
        init_mode="I",
        parameters=parameters,
        dt=0.005,
        T=5.0,
    )

    # Define optimization problem
    # State: [qx, qy, qx_dot, qy_dot]
    # Optimize initial velocities (indices 2, 3) to reach target position (indices 0, 1)
    init_state_template = np.array([0.0, 1.0, 0.0, 0.0])  # Start at (0, 1)
    control_indices = [2, 3]  # Optimize qx_dot, qy_dot
    output_indices = [0, 1]  # Match qx, qy to target
    target = np.array([4.0, 1.0])  # Target position
    inputs = np.array([0.0, 0.0])  # No control during flight
    initial_guess = np.array([0.8, args.init_velocity])  # Initial velocity guess

    # Configure optimizer
    config = ShootingMethodOptimizerConfig(
        max_iters=50,
        error_tol=0.005,
        input_state_tol=1e-5,
        initial_lambda=1e-4,
        lambda_increase=10.0,
        lambda_decrease=0.1,
        line_search_max_iters=15,
        line_search_backoff=0.5,
        verbose=True,
    )

    # Run optimization
    result = optimizer.optimize(
        init_state_template=init_state_template.copy(),
        control_indices=control_indices,
        initial_guess=initial_guess,
        target_state=target,
        output_indices=output_indices,
        inputs=inputs,
        config=config,
    )

    # Plot results
    plot_optimization_results(
        result,
        target,
        title="Trajectory Optimization using 2D Bouncing Ball Hybrid System",
        show_all_iterations=True,
    )

    print(f"\n{'='*50}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*50}")
    print(
        f"Final velocities: vx0={result.optimal_controls[0]:.4f} m/s, "
        f"vy0={result.optimal_controls[1]:.4f} m/s"
    )
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error: {result.final_error:.6f} m")

    # Verify final position
    final_pos = result.final_trajectory[-1, :2]
    print(f"Final position: ({final_pos[0]:.4f}, {final_pos[1]:.4f})")
    print(f"Target position: ({target[0]:.4f}, {target[1]:.4f})")


if __name__ == "__main__":
    main()
