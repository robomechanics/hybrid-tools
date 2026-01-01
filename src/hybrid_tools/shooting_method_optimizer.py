from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from hybrid_tools import HybridSimulator
from hybrid_tools.hybrid_helper_functions import (
    compute_saltation_matrix,
)
from hybrid_tools.types import HybridDynamicalSystem


@dataclass
class ShootingMethodOptimizerConfig:
    """Configuration parameters for ShootingMethodOptimizer."""

    # Convergence criteria
    max_iters: int = 50
    error_tol: float = 0.05
    input_state_tol: float = 1e-3

    # Levenberg-Marquardt parameters
    initial_lambda: float = 1e-4
    lambda_increase: float = 10.0
    lambda_decrease: float = 0.1
    min_lambda: float = 1e-10
    max_lambda: float = 1e6

    # Line search parameters
    line_search_max_iters: int = 15
    line_search_backoff: float = 0.5

    # Verbosity
    verbose: bool = True


@dataclass
class ShootingMethodOptimizerResult:
    """Results from optimization."""

    optimal_controls: np.ndarray
    final_trajectory: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    trajectory_history: List[np.ndarray] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)


class ShootingMethodOptimizer:
    """
    Trajectory optimization using shooting method with adjoint-based gradients.

    Uses Levenberg-Marquardt algorithm to optimize initial conditions
    (e.g., velocities) to reach a target state using a hybrid dynamical system.
    """

    def __init__(
        self,
        hybrid_system: HybridDynamicalSystem,
        init_mode: str,
        parameters: np.ndarray,
        dt: float = 0.005,
        T: float = 5.0,
    ):
        """
        Initialize the shooting method optimizer.

        Args:
            hybrid_system: Hybrid dynamical system to simulate
            init_mode: Initial mode for the system
            parameters: System parameters
            dt: Time step for simulation
            T: Total simulation time
        """
        self.hybrid_system = hybrid_system
        self.init_mode = init_mode
        self.parameters = parameters
        self.dt = dt
        self.T = T
        self.n_steps = int(T / dt)

    def forward_pass(
        self,
        init_state: np.ndarray,
        inputs: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Simulate forward trajectory.

        Args:
            init_state: Initial state vector
            inputs: Control inputs (constant throughout trajectory)

        Returns:
            states: Trajectory of states (n_steps+1, n_states)
            modes: List of modes at each timestep
        """
        simulator = HybridSimulator(
            init_state=init_state.copy(),
            init_mode=self.init_mode,
            dt=self.dt,
            parameters=self.parameters.copy(),
            hybrid_system=self.hybrid_system,
        )

        n_states = len(init_state)
        states = np.zeros((self.n_steps + 1, n_states))
        states[0] = init_state.copy()
        modes = [self.init_mode]

        for k in range(self.n_steps):
            current_time = k * self.dt
            simulator.simulate_timestep(current_time, inputs.copy())
            states[k + 1] = simulator.get_state().copy()
            modes.append(simulator.current_mode)

        return states, modes

    def compute_gradient(
        self,
        states: np.ndarray,
        inputs: np.ndarray,
        output_indices: List[int],
        control_indices: List[int],
        modes: List[str],
    ) -> np.ndarray:
        """
        Compute gradient using adjoint method.

        Efficiently propagates all adjoint vectors simultaneously in a single backward pass.

        Args:
            states: Forward trajectory states
            inputs: Control inputs used in forward pass
            output_indices: Indices of states to compute sensitivity for
            control_indices: Indices of initial state to compute gradient w.r.t.
            modes: List of modes at each timestep

        Returns:
            J: Jacobian matrix (len(output_indices), len(control_indices))
        """
        n_states = states.shape[1]
        n_outputs = len(output_indices)

        # Initialize adjoint matrix: each column is an adjoint for a different output
        Lambda = np.zeros((n_states, n_outputs))
        for i, out_idx in enumerate(output_indices):
            Lambda[out_idx, i] = 1.0

        # Single backward pass propagating all adjoints simultaneously
        for k in range(self.n_steps - 1, -1, -1):
            state = states[k]
            mode = modes[k]

            # Get linearized dynamics
            A_disc = self.hybrid_system.dynamics[mode].A_disc(
                state, inputs, self.dt, self.parameters
            )

            # If there is a mode change, we apply the saltation matrix.
            if k < self.n_steps - 1 and modes[k + 1] != modes[k]:
                # We will use the approximation from III C that the hybrid event
                # occurs at the end of the timestep https://arxiv.org/pdf/2207.04591
                # TODO: we should create a hybridstate struct which contains all the
                # mode info to not have to use this approximation.
                salt = compute_saltation_matrix(
                    pre_event_state=states[k],
                    inputs=inputs,
                    dt=self.dt,
                    parameters=self.parameters,
                    pre_mode=modes[k],
                    post_mode=modes[k + 1],
                    hybrid_system=self.hybrid_system,
                )

                # Apply the saltation matrix at the end of the timestep
                A_disc = salt @ A_disc

            # Adjoint propagation: Λ_k = A^T Λ_{k+1}
            Lambda = A_disc.T @ Lambda

        # Extract Jacobian: rows are outputs, columns are controls
        J = Lambda[control_indices, :].T

        return J

    def compute_lm_step(
        self,
        J: np.ndarray,
        error: np.ndarray,
        lambda_reg: float,
    ) -> np.ndarray:
        """
        Compute Levenberg-Marquardt step: (J^T J + λD) step = J^T error.

        Args:
            J: Jacobian matrix
            error: Current error vector
            lambda_reg: Regularization parameter

        Returns:
            step: Computed step direction
        """
        JtJ = J.T @ J
        Jte = J.T @ error

        # Use diagonal scaling (Marquardt's modification)
        diag_elements = np.maximum(np.diag(JtJ), 1e-10)
        D = np.diag(diag_elements)

        # Solve regularized system
        JtJ_reg = JtJ + lambda_reg * D

        try:
            step = np.linalg.solve(JtJ_reg, Jte)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse as fallback
            step = np.linalg.pinv(JtJ_reg) @ Jte

        return step

    def optimize(
        self,
        init_state_template: np.ndarray,
        control_indices: List[int],
        initial_guess: np.ndarray,
        target_state: np.ndarray,
        output_indices: List[int],
        inputs: np.ndarray,
        config: Optional[ShootingMethodOptimizerConfig] = None,
    ) -> ShootingMethodOptimizerResult:
        """
        Optimize initial conditions to reach target state.

        Args:
            init_state_template: Template for initial state (values at control_indices will be optimized)
            control_indices: Indices in state vector to optimize
            initial_guess: Initial guess for control values
            target_state: Target values for output_indices
            output_indices: Indices of state to match to target
            inputs: Control inputs during trajectory
            config: Optimizer configuration (uses defaults if None)

        Returns:
            ShootingMethodOptimizerResult containing optimal controls, trajectory, and convergence info
        """
        if config is None:
            config = ShootingMethodOptimizerConfig()

        u = initial_guess.copy()
        lambda_reg = config.initial_lambda
        trajectory_history = []

        if config.verbose:
            print(f"{'Iter':<5} | {'Error':<10} | {'λ':<10} | {'Controls'}")
            print("-" * 60)

        for i in range(config.max_iters):
            # Create initial state with current control values
            init_state = init_state_template.copy()
            init_state[control_indices] = u

            # Forward pass
            states, modes = self.forward_pass(init_state, inputs)
            trajectory_history.append(states.copy())

            # Compute error
            final_output = states[-1, output_indices]
            error = final_output - target_state
            err_norm = np.linalg.norm(error)

            if config.verbose:
                controls_str = ", ".join([f"{val:.4f}" for val in u])
                print(f"{i:<5} | {err_norm:<10.4f} | {lambda_reg:<10.2e} | {controls_str}")

            # Check convergence
            if err_norm < config.error_tol:
                if config.verbose:
                    print(f"\nConverged in {i} iterations!")
                return ShootingMethodOptimizerResult(
                    optimal_controls=u,
                    final_trajectory=states,
                    converged=True,
                    iterations=i,
                    final_error=err_norm,
                    trajectory_history=trajectory_history,
                    modes=modes,
                )

            # Compute gradient
            J = self.compute_gradient(states, inputs, output_indices, control_indices, modes)

            # Compute Levenberg-Marquardt step
            step = self.compute_lm_step(J, error, lambda_reg)

            # Try step and adjust regularization
            u_new, lambda_reg = self._adaptive_step(
                u,
                step,
                err_norm,
                init_state_template,
                control_indices,
                target_state,
                output_indices,
                inputs,
                lambda_reg,
                config,
            )

            # Check progress
            u_change = np.linalg.norm(u_new - u)
            if u_change < config.input_state_tol and i > 0:
                if config.verbose:
                    print(f"\nStopped: control change < {config.input_state_tol}")
                return ShootingMethodOptimizerResult(
                    optimal_controls=u_new,
                    final_trajectory=states,
                    converged=err_norm < config.error_tol,
                    iterations=i,
                    final_error=err_norm,
                    trajectory_history=trajectory_history,
                    modes=modes,
                )

            u = u_new

        if config.verbose:
            print(f"\nReached max iterations ({config.max_iters})")

        return ShootingMethodOptimizerResult(
            optimal_controls=u,
            final_trajectory=states,
            converged=False,
            iterations=config.max_iters,
            final_error=err_norm,
            trajectory_history=trajectory_history,
            modes=modes,
        )

    def _adaptive_step(
        self,
        u: np.ndarray,
        step: np.ndarray,
        current_error: float,
        init_state_template: np.ndarray,
        control_indices: List[int],
        target_state: np.ndarray,
        output_indices: List[int],
        inputs: np.ndarray,
        lambda_reg: float,
        config: ShootingMethodOptimizerConfig,
    ) -> Tuple[np.ndarray, float]:
        """
        Adaptive step with Levenberg-Marquardt regularization adjustment.

        Returns:
            u_new: Updated control values
            lambda_new: Updated regularization parameter
        """
        # Try full step
        u_test = u - step
        init_state_test = init_state_template.copy()
        init_state_test[control_indices] = u_test
        states_test, _ = self.forward_pass(init_state_test, inputs)
        err_test = np.linalg.norm(states_test[-1, output_indices] - target_state)

        if err_test < current_error:
            # Good step - decrease λ (more aggressive)
            lambda_new = max(lambda_reg * config.lambda_decrease, config.min_lambda)
            return u_test, lambda_new

        # Bad step - increase λ and try smaller steps
        current_step = step.copy()
        current_lambda = lambda_reg

        for _ in range(config.line_search_max_iters):
            current_lambda = min(current_lambda * config.lambda_increase, config.max_lambda)
            current_step = current_step * config.line_search_backoff
            u_test = u - current_step

            init_state_test = init_state_template.copy()
            init_state_test[control_indices] = u_test
            states_test, _ = self.forward_pass(init_state_test, inputs)
            err_test = np.linalg.norm(states_test[-1, output_indices] - target_state)

            if err_test < current_error:
                return u_test, current_lambda

        # No improvement found, keep current position but increase λ
        return u, current_lambda
