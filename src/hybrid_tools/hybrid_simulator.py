from typing import Callable, Dict

import numpy as np
from scipy.integrate import solve_ivp

from .hybrid_helper_functions import (
    solve_ivp_dynamics_func,
    solve_ivp_extract_hybrid_events,
    solve_ivp_guard_funcs,
)


class HybridSimulator:
    """Simulator for hybrid dynamical systems with discrete mode transitions.

    Simulates continuous dynamics within modes and handles discrete jumps
    at mode boundaries defined by guard conditions and reset maps.
    """

    def __init__(
        self,
        init_state: np.ndarray,
        init_mode: str,
        dt: float,
        noise_matrices: Dict[str, Dict[str, np.ndarray]],
        dynamics: Dict[str, Dict[str, Callable]],
        resets: Dict[str, Dict[str, Dict[str, Callable]]],
        guards: Dict[str, Dict[str, Callable]],
        parameters: np.ndarray,
    ) -> None:
        """Initialize the hybrid system simulator.

        Args:
            init_state: Initial state vector.
            init_mode: Initial discrete mode identifier.
            dt: Time step for simulation.
            noise_matrices: Process ('W') and measurement ('V') noise covariance matrices per mode.
            dynamics: Continuous dynamics and measurement functions per mode.
            resets: Reset maps for state transitions between modes.
            guards: Guard functions defining mode transition conditions.
            parameters: Additional system parameters.
        """
        self._current_state = init_state
        self._current_mode = init_mode
        self._dt = dt
        self._dynamics_dict = dynamics
        self._resets_dict = resets
        self._guards_dict = guards
        self._parameters = parameters
        self._noise_matrices = noise_matrices
        self._n_states = np.shape(self._current_state)[0]

    def simulate_timestep(self, current_time: float, inputs: np.ndarray) -> None:
        """Simulate the hybrid system for one time step.

        Integrates continuous dynamics and handles mode transitions when
        guard conditions are triggered.

        Args:
            current_time: Current simulation time.
            inputs: Control input vector.
        """
        end_time = current_time + self._dt

        # Get noise parameters
        process_gaussian_noise = {
            "mean": np.zeros(
                self._noise_matrices[self._current_mode]["W"].shape[0]
            ),  # TODO: This should be member variable
            "cov": self._noise_matrices[self._current_mode]["W"],
        }
        # Integrate for dt
        current_dynamics = solve_ivp_dynamics_func(
            self._dynamics_dict,
            self._current_mode,
            inputs,
            self._dt,
            self._parameters,
            process_gaussian_noise=process_gaussian_noise,
        )
        current_guards, possible_modes = solve_ivp_guard_funcs(
            self._guards_dict, self._current_mode, inputs, self._dt, self._parameters
        )

        sol = solve_ivp(
            current_dynamics,
            [current_time, end_time],
            self._current_state,
            events=current_guards,
        )
        current_state = np.zeros(self._n_states)

        # If we hit guard, apply reset
        (
            hybrid_event_state,
            hybrid_event_time,
            new_mode,
        ) = solve_ivp_extract_hybrid_events(sol, possible_modes)

        while new_mode is not None:
            # Apply reset
            current_state = self._resets_dict[self._current_mode][new_mode]["r"](
                hybrid_event_state, inputs, self._dt, self._parameters
            ).reshape(np.shape(hybrid_event_state))

            # Update guard and simulate
            self._current_mode = new_mode
            current_dynamics = solve_ivp_dynamics_func(
                self._dynamics_dict,
                self._current_mode,
                inputs,
                self._dt,
                self._parameters,
            )
            current_guards, possible_modes = solve_ivp_guard_funcs(
                self._guards_dict,
                self._current_mode,
                inputs,
                self._dt,
                self._parameters,
            )
            sol = solve_ivp(
                current_dynamics,
                [hybrid_event_time, end_time],
                current_state,
                events=current_guards,
            )
            (
                hybrid_event_state,
                hybrid_event_time,
                new_mode,
            ) = solve_ivp_extract_hybrid_events(sol, possible_modes)

        # Once no more hybrid events, grab the terminal states
        for idx in range(len(sol.y)):
            # Grab the state at the last timestep
            current_state[idx] = sol.y[idx][-1]

        self._current_state = current_state

    def get_measurement(self, measurement_noise_flag: bool = False) -> np.ndarray:
        """Get measurement from current state.

        Args:
            measurement_noise_flag: If True, add measurement noise to output.

        Returns:
            Measurement vector, optionally with added noise.
        """
        measurement = self._dynamics_dict[self._current_mode]["y"](
            self._current_state,
            self._parameters,
        ).flatten()
        if not measurement_noise_flag:
            return measurement
        else:
            measurement_gaussian_noise = {
                "mean": np.zeros(np.shape(measurement)[0]),
                "cov": self._noise_matrices[self._current_mode]["V"],
            }
            return measurement + np.random.multivariate_normal(
                measurement_gaussian_noise["mean"], measurement_gaussian_noise["cov"]
            )

    def get_state(self) -> np.ndarray:
        """Get a copy of the current state vector.

        Returns:
            Copy of current state vector.
        """
        return self._current_state.copy()

    @property
    def current_mode(self) -> str:
        """Current discrete mode of the system."""
        return self._current_mode
