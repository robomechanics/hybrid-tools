import argparse

import matplotlib.pyplot as plt
import numpy as np

from hybrid_tools.basic_hybrid_systems import bouncing_ball_2d
from hybrid_tools.shooting_method_optimizer import (
    ShootingMethodOptimizer,
    ShootingMethodOptimizerConfig,
    ShootingMethodOptimizerResult,
)


def count_bounces(modes: list[str]) -> int:
    """Count the number of bounces (I->J transitions) in a trajectory."""
    if not modes:
        return 0
    bounces = 0
    for i in range(1, len(modes)):
        if modes[i - 1] == "I" and modes[i] == "J":
            bounces += 1
    return bounces


def plot_velocity_sweep_results(
    results: list[tuple[float, ShootingMethodOptimizerResult]],
    target: np.ndarray,
    title: str = "Velocity Sweep Trajectory Optimization",
    save_path: str = "velocity_sweep_trajectories.png",
):
    """
    Plot all trajectories from velocity sweep on one plot.

    Args:
        results: List of tuples (initial_velocity, result)
        target: Target position [x, y]
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    converged_count = 0
    failed_count = 0

    # Count bounces for all converged trajectories to determine color mapping
    bounce_counts = []
    for _, result in results:
        if result.converged and result.modes:
            bounce_counts.append(count_bounces(result.modes))

    # Determine color mapping range
    if bounce_counts:
        min_bounces = min(bounce_counts)
        max_bounces = max(bounce_counts)
        bounce_range = max(max_bounces - min_bounces, 1)  # Avoid division by zero
    else:
        min_bounces = 0
        max_bounces = 0
        bounce_range = 1

    # Track which bounce counts we've seen for legend
    bounce_labels = {}

    # Plot all trajectories
    for _, result in results:
        if result.converged:
            # Successful trajectories: color by number of bounces
            if result.modes:
                n_bounces = count_bounces(result.modes)
                # Map to colormap (viridis: purple=few bounces, yellow=many bounces)
                color_val = (n_bounces - min_bounces) / bounce_range
                color = plt.cm.viridis(color_val)

                # Create label for legend (only once per bounce count)
                if n_bounces not in bounce_labels:
                    bounce_labels[n_bounces] = f"{n_bounces} bounce" + (
                        "s" if n_bounces != 1 else ""
                    )
                    label = bounce_labels[n_bounces]
                else:
                    label = None
            else:
                color = "blue"
                label = "Converged (no mode data)" if converged_count == 0 else None

            alpha = 0.4
            linewidth = 1.5
            converged_count += 1
        else:
            # Failed trajectories: black and more transparent
            color = "black"
            alpha = 0.15
            linewidth = 1.0
            failed_count += 1
            label = "Failed" if failed_count == 1 else None

        ax.plot(
            result.final_trajectory[:, 0],
            result.final_trajectory[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )

    # Plot target
    ax.scatter(
        *target,
        color="red",
        marker="X",
        s=300,
        label="Target",
        zorder=10,
        edgecolors="black",
        linewidths=2,
    )

    # Plot start position (should be same for all)
    start_pos = results[0][1].final_trajectory[0, :2]
    ax.scatter(
        start_pos[0],
        start_pos[1],
        color="green",
        marker="o",
        s=150,
        label="Start",
        zorder=10,
        edgecolors="black",
        linewidths=2,
    )

    ax.set_title(
        f"{title}\n"
        f"Converged: {converged_count}/{len(results)}, "
        f"Failed: {failed_count}/{len(results)}"
    )
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")


def main():
    """Example: 2D bouncing ball trajectory optimization with velocity sweep."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="2D bouncing ball trajectory optimization with velocity sweep"
    )
    parser.add_argument(
        "--vel-min",
        type=float,
        default=-20.0,
        help="Minimum initial vertical velocity (m/s) (default: -20.0)",
    )
    parser.add_argument(
        "--vel-max",
        type=float,
        default=20.0,
        help="Maximum initial vertical velocity (m/s) (default: 20.0)",
    )
    parser.add_argument(
        "--vel-steps",
        type=int,
        default=21,
        help="Number of velocity steps (default: 41)",
    )
    parser.add_argument(
        "--ground-height",
        type=float,
        default=0.0,
        help="Ground height for bouncing (m) (default: 0.0)",
    )
    parser.add_argument(
        "--target-x",
        type=float,
        default=4.0,
        help="Target x position (m) (default: 4.0)",
    )
    parser.add_argument(
        "--target-y",
        type=float,
        default=1.0,
        help="Target y position (m) (default: 1.0)",
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
    target = np.array([args.target_x, args.target_y])  # Target position
    inputs = np.array([0.0, 0.0])  # No control during flight

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
        verbose=False,  # Disable verbose for sweep
    )

    # Sweep initial velocities
    velocities = np.linspace(args.vel_min, args.vel_max, args.vel_steps)
    results = []

    print(f"\n{'='*60}")
    print("VELOCITY SWEEP OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Sweeping initial vertical velocity from {args.vel_min} to {args.vel_max} m/s")
    print(f"Number of steps: {args.vel_steps}")
    print(f"Target position: ({target[0]:.2f}, {target[1]:.2f})")
    print(f"{'='*60}\n")

    for i, vy_init in enumerate(velocities):
        # Initial guess: horizontal velocity = 0.8, vertical velocity = sweep value
        initial_guess = np.array([0.8, vy_init])

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

        results.append((vy_init, result))

        # Print progress
        status = "✓" if result.converged else "✗"
        print(
            f"[{i+1:3d}/{args.vel_steps}] vy_init={vy_init:6.2f} m/s: "
            f"{status} (error={result.final_error:.4f} m, iters={result.iterations})"
        )

    # Summary statistics
    converged_results = [r for _, r in results if r.converged]
    failed_results = [r for _, r in results if not r.converged]

    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {len(results)}")
    print(f"Converged: {len(converged_results)} ({100*len(converged_results)/len(results):.1f}%)")
    print(f"Failed: {len(failed_results)} ({100*len(failed_results)/len(results):.1f}%)")

    if converged_results:
        avg_error = np.mean([r.final_error for r in converged_results])
        avg_iters = np.mean([r.iterations for r in converged_results])
        print(f"Average error (converged): {avg_error:.6f} m")
        print(f"Average iterations (converged): {avg_iters:.1f}")

    # Plot results
    plot_velocity_sweep_results(
        results,
        target,
        title="Velocity Sweep: 2D Bouncing Ball Trajectory Optimization",
        save_path="shooting_ball_hybrid_system_velocity_sweep.png",
    )


if __name__ == "__main__":
    main()
