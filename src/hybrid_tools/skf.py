from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .hybrid_helper_functions import (
    compute_saltation_matrix,
    solve_ivp_dynamics_func,
    solve_ivp_extract_hybrid_events,
    solve_ivp_guard_funcs,
)
from .types import HybridDynamicalSystem


class SKF:
    """Saltation Kalman Filter for hybrid dynamical systems.

    Implements state estimation for systems with discrete mode transitions,
    handling both continuous dynamics within modes and discontinuous jumps
    at mode boundaries using saltation matrices.
    """

    def __init__(
        self,
        init_state: np.ndarray,
        init_mode: str,
        init_cov: np.ndarray,
        dt: float,
        parameters: np.ndarray,
        hybrid_system: HybridDynamicalSystem,
    ) -> None:
        """Initialize the Saltation Kalman Filter.

        Args:
            init_state: Initial state vector.
            init_mode: Initial discrete mode identifier.
            init_cov: Initial state covariance matrix.
            dt: Time step for discrete updates.
            parameters: Additional system parameters.
            hybrid_system: Complete hybrid system specification including noise matrices.
        """
        self._hybrid_system = hybrid_system
        self._current_state = init_state
        self._current_cov = init_cov
        self._current_mode = init_mode
        self._dt = dt
        self._parameters = parameters
        self._n_states = np.shape(self._current_state)[0]

    @property
    def current_mode(self) -> str:
        """Current discrete mode of the system."""
        return self._current_mode

    def _propagate_covariance_using_dynamics(
        self,
        current_time: float,
        current_state: np.ndarray,
        inputs: np.ndarray,
        current_mode: str,
        dt: float,
    ) -> None:
        """Forward propagate covariance using linearized dynamics and process noise.

        Args:
            current_time: Current simulation time.
            current_state: Current state vector.
            inputs: Control input vector.
            current_mode: Current discrete mode.
            dt: Time step for propagation.
        """
        dynamics_cov = self._hybrid_system.dynamics[current_mode].A_disc(
            current_state, inputs, dt, self._parameters
        )
        input_jacobian = self._hybrid_system.dynamics[current_mode].B_disc(
            current_state, inputs, dt, self._parameters
        )
        process_cov = input_jacobian @ self._hybrid_system.noises[current_mode].W @ input_jacobian.T
        self._current_cov = dynamics_cov @ self._current_cov @ dynamics_cov.T + process_cov

    def predict(self, current_time: float, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform prediction step (prior update) of the Kalman filter.

        Integrates continuous dynamics and handles mode transitions via guards
        and resets. Updates state and covariance through saltation matrices at
        discrete transitions.

        Args:
            current_time: Current simulation time.
            inputs: Control input vector.

        Returns:
            Tuple of (predicted_state, predicted_covariance).
        """
        end_time = current_time + self._dt

        # Integrate for dt
        current_dynamics = solve_ivp_dynamics_func(
            self._hybrid_system.dynamics, self._current_mode, inputs, self._dt, self._parameters
        )
        current_guards, possible_modes = solve_ivp_guard_funcs(
            self._hybrid_system.guards, self._current_mode, inputs, self._dt, self._parameters
        )

        current_start_state = self._current_state.copy()
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
            current_state = (
                self._hybrid_system.resets[self._current_mode][new_mode]
                .r(hybrid_event_state, inputs, self._dt, self._parameters)
                .reshape(np.shape(hybrid_event_state))
            )

            self._propagate_covariance_using_dynamics(
                current_time=current_time,
                current_state=current_start_state,
                inputs=inputs,
                current_mode=self._current_mode,
                dt=sol.t[-1] - sol.t[0],
            )

            salt = compute_saltation_matrix(
                pre_event_state=hybrid_event_state,
                inputs=inputs,
                dt=self._dt,
                parameters=self._parameters,
                pre_mode=self._current_mode,
                post_mode=new_mode,
                hybrid_system=self._hybrid_system,
                post_event_state=current_state,
            )
            self._current_cov = salt @ self._current_cov @ salt.T

            # Update guard and simulate
            self._current_mode = new_mode
            current_dynamics = solve_ivp_dynamics_func(
                self._hybrid_system.dynamics,
                self._current_mode,
                inputs,
                self._dt,
                self._parameters,
            )
            current_guards, possible_modes = solve_ivp_guard_funcs(
                self._hybrid_system.guards,
                self._current_mode,
                inputs,
                self._dt,
                self._parameters,
            )
            current_start_state = current_state.copy()
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

        # Propagate the rest of the covariance
        self._propagate_covariance_using_dynamics(
            current_time=current_time,
            current_state=current_start_state,
            inputs=inputs,
            current_mode=self._current_mode,
            dt=sol.t[-1] - sol.t[0],
        )

        self._current_state = current_state
        return self._current_state, self._current_cov

    def update(
        self, current_time: float, current_input: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform measurement update step (posterior update) of the Kalman filter.

        Incorporates new measurements to refine state estimate. Checks for mode
        transitions triggered by the updated state and applies resets with
        saltation matrices if necessary.

        Args:
            current_time: Current simulation time.
            current_input: Control input vector.
            measurement: Measurement vector.

        Returns:
            Tuple of (updated_state, updated_covariance).
        """
        C = self._hybrid_system.dynamics[self._current_mode].C(
            self._current_state,
            self._parameters,
        )
        V = self._hybrid_system.noises[self._current_mode].V
        K = self._current_cov @ C.T @ np.linalg.inv(C @ self._current_cov @ C.T + V)

        # Measurement update
        measurement_est = (
            self._hybrid_system.dynamics[self._current_mode]
            .y(
                self._current_state,
                self._parameters,
            )
            .flatten()
        )
        C = self._hybrid_system.dynamics[self._current_mode].C(
            self._current_state,
            self._parameters,
        )
        residual = measurement - measurement_est
        self._current_state = self._current_state + K @ residual
        self._current_cov = self._current_cov - K @ C @ self._current_cov

        # Check guard conditions. If any guard has been reached, apply hybrid posterior update
        current_guards, possible_modes = solve_ivp_guard_funcs(
            self._hybrid_system.guards,
            self._current_mode,
            current_input,
            self._dt,
            self._parameters,
        )
        for guard_idx in range(len(current_guards)):
            # TODO: Add in time component
            if current_guards[guard_idx](current_time, self._current_state) < 0:
                new_mode = possible_modes[guard_idx]
                # Apply reset
                new_state = (
                    self._hybrid_system.resets[self._current_mode][new_mode]
                    .r(self._current_state, current_input, self._dt, self._parameters)
                    .reshape(np.shape(self._current_state))
                )

                # Apply covariance updates: dynamics and saltation matrix
                salt = compute_saltation_matrix(
                    pre_event_state=self._current_state,
                    inputs=current_input,
                    dt=self._dt,
                    parameters=self._parameters,
                    pre_mode=self._current_mode,
                    post_mode=new_mode,
                    hybrid_system=self._hybrid_system,
                    post_event_state=new_state,
                )
                self._current_state = new_state
                self._current_cov = salt @ self._current_cov @ salt.T
                break

        return self._current_state, self._current_cov

    @property
    def current_state(self) -> np.ndarray:
        """Current state estimate."""
        return self._current_state

    @property
    def current_cov(self) -> np.ndarray:
        """Current state covariance estimate."""
        return self._current_cov
