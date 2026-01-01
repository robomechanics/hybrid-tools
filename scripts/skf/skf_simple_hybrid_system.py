import matplotlib.pyplot as plt
import numpy as np

from hybrid_tools import SKF, HybridSimulator
from hybrid_tools.basic_hybrid_systems import simple_system

""" Define dynamics and resets. """
hybrid_system = simple_system()

""" Initialize states and covariance. """
n_states = 2
mean_init_state = np.array([-2.5, 0])
mean_init_cov = 0.1 * np.eye(n_states)
init_mode = "I"  # Modes are {I, J}

""" Define timesteps. """
dt = 0.1

""" Define parameters. """
parameters = np.array([])

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

n_simulate_timesteps = 50
timesteps = np.arange(0.0, n_simulate_timesteps * dt, dt)
measurements = np.zeros((n_simulate_timesteps - 1, n_states))
actual_states = np.zeros((n_simulate_timesteps, n_states))
filtered_states = np.zeros((n_simulate_timesteps, n_states))

actual_states[0, :] = hybrid_simulator.get_state()
filtered_states[0, :] = mean_init_state

zero_input = np.array([0.0, 0.0])
for time_idx in range(1, n_simulate_timesteps):
    hybrid_simulator.simulate_timestep(0, zero_input)
    actual_states[time_idx, :] = hybrid_simulator.get_state()
    measurements[time_idx - 1, :] = hybrid_simulator.get_measurement(measurement_noise_flag=True)
    skf.predict(timesteps[time_idx], zero_input)
    filtered_states[time_idx, :], current_cov = skf.update(
        timesteps[time_idx], zero_input, measurements[time_idx - 1, :]
    )

plt.plot(actual_states[:, 0], actual_states[:, 1], "k-", label="Actual states")
plt.plot(measurements[:, 0], measurements[:, 1], "r.", label="Measurements")
plt.plot(filtered_states[:, 0], filtered_states[:, 1], "b--", label="Filtered states")
plt.legend()
plt.show()
