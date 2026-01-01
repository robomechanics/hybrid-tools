import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.matrices import Matrix

from hybrid_tools import (
    SKF,
    HybridDynamicalSystem,
    HybridSimulator,
    ModeDynamics,
    ModeGuard,
    ModeNoise,
    ModeReset,
    create_dynamics,
    create_guards,
    create_resets,
)


def symbolic_dynamics():
    """
    Returns (Tuple[Dict, Dict]): dynamic functions in a nested dict and reset functions in a nested dict.
    TODO: FILL IN WITH BOUNCING BALL. Modes are {'up','down'}. e is coefficient of resititution.
    """
    q, q_dot, e, g, u, dt = sp.symbols("q q_dot e g u dt")

    """ Define the states and inputs. """
    inputs = Matrix([u])
    states = Matrix([q, q_dot])

    """ FILL IN EVERYTHING ELSE BELOW HERE!! """
    """ Defining the dynamics of the system. """
    # Both modes have same dynamics: free fall under gravity with boosters
    # q_dot' = q_dot, q_dot_dot = -g
    fI = Matrix([q_dot, -g + u])
    fJ = Matrix([q_dot, -g + u])

    """ Define the measurements of the system. """
    # Measure both position and velocity
    yI = Matrix([q, q_dot])
    yJ = Matrix([q, q_dot])

    """ Discretize the dynamics using euler integration. """
    fI_disc = states + fI * dt
    fJ_disc = states + fJ * dt

    """ Take the jacobian with respect to states and inputs. """
    AI_disc = fI_disc.jacobian(states)
    AJ_disc = fJ_disc.jacobian(states)

    BI_disc = fI_disc.jacobian(inputs)
    BJ_disc = fJ_disc.jacobian(inputs)

    """ Take the jacobian of the measurements with respect to the states. """
    CI = yI.jacobian(states)
    CJ = yJ.jacobian(states)

    """ Define resets. """
    # When ball hits ground, position stays the same, velocity reverses with energy loss
    # r = [0, -e * q_dot]
    rIJ = Matrix([q, -e * q_dot])
    rJI = Matrix([q, q_dot])

    """ Take the jacobian of resets with resepct to states. """
    RIJ = rIJ.jacobian(states)
    RJI = rJI.jacobian(states)

    """ Define guards. """
    # Guard triggers when ball hits ground: q <= 0
    gIJ = Matrix([q])
    gJI = Matrix([q_dot])

    """ Take the jacobian of resets with resepct to guards. """
    GIJ = gIJ.jacobian(states)
    GJI = gJI.jacobian(states)

    """ Define the parameters of the system. """
    # Parameters: coefficient of restitution (e) and gravity (g)
    parameters = Matrix([e, g])

    rIJ_func = sp.lambdify((states, inputs, dt, parameters), rIJ)
    RIJ_func = sp.lambdify((states, inputs, dt, parameters), RIJ)

    rJI_func = sp.lambdify((states, inputs, dt, parameters), rJI)
    RJI_func = sp.lambdify((states, inputs, dt, parameters), RJI)

    gIJ_func = sp.lambdify((states, inputs, dt, parameters), gIJ)
    GIJ_func = sp.lambdify((states, inputs, dt, parameters), GIJ)

    gJI_func = sp.lambdify((states, inputs, dt, parameters), gJI)
    GJI_func = sp.lambdify((states, inputs, dt, parameters), GJI)

    fI_func = sp.lambdify((states, inputs, dt, parameters), fI)
    AI_disc_func = sp.lambdify((states, inputs, dt, parameters), AI_disc)
    BI_disc_func = sp.lambdify((states, inputs, dt, parameters), BI_disc)

    fJ_func = sp.lambdify((states, inputs, dt, parameters), fJ)
    AJ_disc_func = sp.lambdify((states, inputs, dt, parameters), AJ_disc)
    BJ_disc_func = sp.lambdify((states, inputs, dt, parameters), BJ_disc)

    yI_func = sp.lambdify((states, parameters), yI)
    CI_func = sp.lambdify((states, parameters), CI)

    yJ_func = sp.lambdify((states, parameters), yJ)
    CJ_func = sp.lambdify((states, parameters), CJ)

    dynamics = create_dynamics(
        [
            (
                "I",
                ModeDynamics(
                    f_cont=fI_func,
                    A_disc=AI_disc_func,
                    B_disc=BI_disc_func,
                    y=yI_func,
                    C=CI_func,
                ),
            ),
            (
                "J",
                ModeDynamics(
                    f_cont=fJ_func,
                    A_disc=AJ_disc_func,
                    B_disc=BJ_disc_func,
                    y=yJ_func,
                    C=CJ_func,
                ),
            ),
        ]
    )

    resets = create_resets(
        [
            ("I", "J", ModeReset(r=rIJ_func, R=RIJ_func)),
            ("J", "I", ModeReset(r=rJI_func, R=RJI_func)),
        ]
    )

    guards = create_guards(
        [
            ("I", "J", ModeGuard(g=gIJ_func, G=GIJ_func)),
            ("J", "I", ModeGuard(g=gJI_func, G=GJI_func)),
        ]
    )

    # Define noise matrices
    n_states = 2
    n_inputs = 1
    W_global = 0.01 * np.eye(n_inputs)
    V_global = 0.01 * np.eye(n_states)
    noises = {
        "I": ModeNoise(W=W_global, V=V_global),
        "J": ModeNoise(W=W_global, V=V_global),
    }

    return HybridDynamicalSystem(
        dynamics=dynamics,
        resets=resets,
        guards=guards,
        noises=noises,
    )


""" Define dynamics and resets. """
hybrid_system = symbolic_dynamics()

""" Initialize states and covariance. """
# Start ball at height 2.5m with zero velocity
n_states = 2
mean_init_state = np.array([2.5, 0])
mean_init_cov = 0.1 * np.eye(n_states)
init_mode = "I"  # Modes are {I, J}

""" Define timesteps. """
dt = 0.005

""" Define parameters. """
# Parameters: [e (coefficient of restitution), g (gravity)]
# e = 0.8 means ball retains 80% of velocity after bounce
# g = 9.81 m/s^2 (standard gravity)
parameters = np.array([0.8, 9.81])

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

zero_input = np.array([0.0])
# Larger steps between posterior updates lead to less mode mismatches.
# TODO: we can apply a rule where if the residual is large
# - If we recently transitioned, we should check what happened if we didn't.
# - If we haven't transitioned, but are in some neighborhood of guards, we should check what happens if we simulate more in time until the guard is hit.
posterior_update_frequency = 10
for time_idx in range(1, n_simulate_timesteps):
    hybrid_simulator.simulate_timestep(0, np.array([0]))
    actual_states[time_idx, :] = hybrid_simulator.get_state()
    measurements[time_idx - 1, :] = hybrid_simulator.get_measurement(measurement_noise_flag=True)
    skf.predict(timesteps[time_idx], zero_input)
    if time_idx % posterior_update_frequency == 0:
        _, _ = skf.update(timesteps[time_idx], zero_input, measurements[time_idx - 1, :])
    filtered_states[time_idx, :] = skf.current_state
    mode_list.append(hybrid_simulator.current_mode)

# Phase plot
plt.figure(figsize=(10, 6))
plt.plot(actual_states[:, 0], actual_states[:, 1], "k-", label="Actual states")
plt.plot(measurements[:, 0], measurements[:, 1], "r.", label="Measurements", markersize=2)
plt.plot(filtered_states[:, 0], filtered_states[:, 1], "b--", label="Filtered states")
plt.xlabel("Position (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Bouncing Ball Phase Plot")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("bouncing_ball.png")
plt.close()

# Time series plots with mode highlighting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

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
        ax1.axvspan(
            timesteps[start_idx],
            timesteps[end_idx],
            alpha=0.3,
            color=mode_colors[current_mode],
            label=label,
        )
        ax2.axvspan(
            timesteps[start_idx],
            timesteps[end_idx],
            alpha=0.3,
            color=mode_colors[current_mode],
            label=label,
        )
        current_mode = mode_list[idx]
        start_idx = idx

# Plot position
ax1.plot(timesteps, actual_states[:, 0], "k-", label="Actual", linewidth=1.5)
ax1.plot(timesteps[:-1], measurements[:, 0], "r.", label="Measurements", markersize=2)
ax1.plot(timesteps, filtered_states[:, 0], "b--", label="Filtered", linewidth=1.5)
ax1.set_ylabel("Position (m)", fontsize=12)
ax1.set_title("Bouncing Ball Time Series", fontsize=14, fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Plot velocity
ax2.plot(timesteps, actual_states[:, 1], "k-", label="Actual", linewidth=1.5)
ax2.plot(timesteps[:-1], measurements[:, 1], "r.", label="Measurements", markersize=2)
ax2.plot(timesteps, filtered_states[:, 1], "b--", label="Filtered", linewidth=1.5)
ax2.set_xlabel("Time (s)", fontsize=12)
ax2.set_ylabel("Velocity (m/s)", fontsize=12)
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("bouncing_ball_timeseries.png", dpi=150)
plt.close()
# plt.show()
