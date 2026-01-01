import matplotlib.pyplot as plt
import numpy as np

from hybrid_tools import SKF, HybridSimulator
from hybrid_tools.basic_hybrid_systems import bouncing_ball_2d

GROUND_HEIGHT = 0.0

""" Define dynamics and resets. """
hybrid_system = bouncing_ball_2d()

""" Draw mode transition graph. """
try:
    hybrid_system.draw_mode_graph("bouncing_ball_2d_mode_graph.png")
except ImportError as e:
    print(f"Skipping mode graph generation: {e}")

""" Initialize states and covariance. """
# Start ball at position (0, 2.5) with initial horizontal velocity 1.5 m/s and zero vertical velocity
n_states = 4
mean_init_state = np.array([0.0, 2.5, 1.5, 0.0])  # [qx, qy, qx_dot, qy_dot]
mean_init_cov = 0.1 * np.eye(n_states)
init_mode = "I"  # Modes are {I, J}

""" Define timesteps. """
dt = 0.005

""" Define parameters. """
# Parameters: [e (coefficient of restitution), g (gravity)]
# e = 0.8 means ball retains 80% of velocity after bounce
# g = 9.81 m/s^2 (standard gravity)
parameters = np.array([0.8, 9.81, GROUND_HEIGHT])

""" Initialize filter. """
skf = SKF(
    init_state=mean_init_state,
    init_mode=init_mode,
    init_cov=mean_init_cov,
    dt=dt,
    parameters=parameters,
    hybrid_system=hybrid_system,
)

""" Initialize simulator. """
actual_init_state = np.random.multivariate_normal(mean_init_state, mean_init_cov)
hybrid_simulator = HybridSimulator(
    init_state=actual_init_state,
    init_mode=init_mode,
    dt=dt,
    parameters=parameters,
    hybrid_system=hybrid_system,
)

n_simulate_timesteps = 500
timesteps = np.arange(0.0, n_simulate_timesteps * dt, dt)
measurements = np.zeros((n_simulate_timesteps - 1, n_states))
actual_states = np.zeros((n_simulate_timesteps, n_states))
filtered_states = np.zeros((n_simulate_timesteps, n_states))

actual_states[0, :] = hybrid_simulator.get_state()
filtered_states[0, :] = mean_init_state

mode_list = []
actual_mode_list = []

zero_input = np.array([0.0, 0.0])
posterior_update_frequency = 10
for time_idx in range(1, n_simulate_timesteps):
    hybrid_simulator.simulate_timestep(0, zero_input)
    actual_states[time_idx, :] = hybrid_simulator.get_state()
    measurements[time_idx - 1, :] = hybrid_simulator.get_measurement(measurement_noise_flag=True)
    skf.predict(timesteps[time_idx], zero_input)
    if time_idx % posterior_update_frequency == 0:
        _, _ = skf.update(timesteps[time_idx], zero_input, measurements[time_idx - 1, :])
    filtered_states[time_idx, :] = skf.current_state
    mode_list.append(hybrid_simulator.current_mode)

# 2D Trajectory plot
plt.figure(figsize=(12, 6))
plt.plot(actual_states[:, 0], actual_states[:, 1], "k-", label="Actual trajectory", linewidth=2)
plt.plot(
    measurements[:, 0], measurements[:, 1], "r.", label="Measurements", markersize=2, alpha=0.5
)
plt.plot(
    filtered_states[:, 0], filtered_states[:, 1], "b--", label="Filtered trajectory", linewidth=1.5
)
plt.axhline(y=GROUND_HEIGHT, color="brown", linestyle="-", linewidth=2, label="Ground")
plt.xlabel("Horizontal Position (m)", fontsize=12)
plt.ylabel("Vertical Position (m)", fontsize=12)
plt.title("2D Bouncing Ball Trajectory", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.savefig("bouncing_ball_2d_trajectory.png", dpi=150)
plt.close()

# Time series plots with mode highlighting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Highlight mode regions
mode_colors = {"I": "lightblue", "J": "lightcoral"}
mode_labels = {"I": "Mode I", "J": "Mode J"}
current_mode = mode_list[0] if mode_list else init_mode
start_idx = 1
labeled_modes = set()  # Track which modes have been labeled

for idx in range(1, len(mode_list)):
    if mode_list[idx] != current_mode or idx == len(mode_list) - 1:
        end_idx = idx if mode_list[idx] != current_mode else idx + 1
        # Add label only if this mode hasn't been labeled yet
        label = mode_labels[current_mode] if current_mode not in labeled_modes else None
        if label:
            labeled_modes.add(current_mode)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvspan(
                timesteps[start_idx],
                timesteps[end_idx],
                alpha=0.3,
                color=mode_colors[current_mode],
                label=label,
            )
        current_mode = mode_list[idx]
        start_idx = idx

# Plot horizontal position
ax1.plot(timesteps, actual_states[:, 0], "k-", label="Actual", linewidth=1.5)
ax1.plot(timesteps[:-1], measurements[:, 0], "r.", label="Measurements", markersize=2)
ax1.plot(timesteps, filtered_states[:, 0], "b--", label="Filtered", linewidth=1.5)
ax1.set_ylabel("Horizontal Position (m)", fontsize=12)
ax1.set_title("2D Bouncing Ball Time Series", fontsize=14, fontweight="bold")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Plot vertical position
ax2.plot(timesteps, actual_states[:, 1], "k-", label="Actual", linewidth=1.5)
ax2.plot(timesteps[:-1], measurements[:, 1], "r.", label="Measurements", markersize=2)
ax2.plot(timesteps, filtered_states[:, 1], "b--", label="Filtered", linewidth=1.5)
ax2.set_ylabel("Vertical Position (m)", fontsize=12)
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# Plot horizontal velocity
ax3.plot(timesteps, actual_states[:, 2], "k-", label="Actual", linewidth=1.5)
ax3.plot(timesteps[:-1], measurements[:, 2], "r.", label="Measurements", markersize=2)
ax3.plot(timesteps, filtered_states[:, 2], "b--", label="Filtered", linewidth=1.5)
ax3.set_xlabel("Time (s)", fontsize=12)
ax3.set_ylabel("Horizontal Velocity (m/s)", fontsize=12)
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)

# Plot vertical velocity
ax4.plot(timesteps, actual_states[:, 3], "k-", label="Actual", linewidth=1.5)
ax4.plot(timesteps[:-1], measurements[:, 3], "r.", label="Measurements", markersize=2)
ax4.plot(timesteps, filtered_states[:, 3], "b--", label="Filtered", linewidth=1.5)
ax4.set_xlabel("Time (s)", fontsize=12)
ax4.set_ylabel("Vertical Velocity (m/s)", fontsize=12)
ax4.legend(loc="upper right")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("bouncing_ball_2d_timeseries.png", dpi=150)
plt.close()

print("2D Bouncing Ball simulation complete!")
print(
    f"Initial state: qx={actual_init_state[0]:.2f}m, qy={actual_init_state[1]:.2f}m, "
    f"vx={actual_init_state[2]:.2f}m/s, vy={actual_init_state[3]:.2f}m/s"
)
print(
    f"Final state: qx={actual_states[-1, 0]:.2f}m, qy={actual_states[-1, 1]:.2f}m, "
    f"vx={actual_states[-1, 2]:.2f}m/s, vy={actual_states[-1, 3]:.2f}m/s"
)
print(
    f"Number of bounces: {sum(1 for i in range(1, len(mode_list)) if mode_list[i] != mode_list[i-1])}"
)
