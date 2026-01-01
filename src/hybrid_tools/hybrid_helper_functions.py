from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def solve_ivp_dynamics_func(
    dynamics_dict: Dict[str, Dict[str, Callable]],
    mode: str,
    inputs: np.ndarray,
    dt: float,
    parameters: np.ndarray,
    process_gaussian_noise: Optional[Dict[str, np.ndarray]] = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Create dynamics function compatible with scipy.integrate.solve_ivp.

    Args:
        dynamics_dict: Dictionary of dynamics functions per mode.
        mode: Current discrete mode identifier.
        inputs: Control input vector.
        dt: Time step.
        parameters: System parameters.
        process_gaussian_noise: Optional dict with 'mean' and 'cov' for process noise.

    Returns:
        Lambda function f(t, states) for use with solve_ivp.

    Raises:
        ValueError: If process noise size doesn't match input size.
    """
    if process_gaussian_noise is not None:
        process_noise = np.random.multivariate_normal(
            process_gaussian_noise["mean"], process_gaussian_noise["cov"]
        )
        if process_noise.size != inputs.size:
            raise ValueError(
                f"Process noise size {process_noise.size} does not match input size {inputs.size}"
            )
        noisy_inputs = inputs + process_noise
        return lambda t, states: dynamics_dict[mode]["f_cont"](
            states, noisy_inputs, dt, parameters
        ).reshape(np.shape(states))
    else:
        return lambda t, states: dynamics_dict[mode]["f_cont"](
            states, inputs, dt, parameters
        ).reshape(np.shape(states))


def solve_ivp_guard_funcs(
    guards_dict: Dict[str, Dict[str, Dict[str, Callable]]],
    mode: str,
    inputs: np.ndarray,
    dt: float,
    parameters: np.ndarray,
) -> Tuple[List[Callable], List[str]]:
    """Create guard functions compatible with scipy.integrate.solve_ivp events.

    Args:
        guards_dict: Dictionary of guard functions per mode and transition.
        mode: Current discrete mode identifier.
        inputs: Control input vector.
        dt: Time step.
        parameters: System parameters.

    Returns:
        Tuple of (guard_functions, possible_target_modes).
    """
    guards: List[Callable] = []
    new_modes: List[str] = []
    if mode in guards_dict:
        for key, val in guards_dict[mode].items():
            # Fix closure issue by using default argument to capture val
            # Flatten the result to return a scalar for solve_ivp
            def guard(t: float, states: np.ndarray, v: Dict[str, Callable] = val) -> float:
                return v["g"](states, inputs, dt, parameters).flatten()[0]

            guards.append(guard)
            new_modes.append(key)
    return guards, new_modes


def solve_ivp_extract_hybrid_events(
    sol, possible_modes: List[str]
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
    """Extract hybrid events from scipy.integrate.solve_ivp solution.

    Args:
        sol: Solution object from solve_ivp.
        possible_modes: List of possible target modes corresponding to guards.

    Returns:
        Tuple of (event_state, event_time, new_mode) or (None, None, None) if no event.
    """
    for idx in range(len(possible_modes)):
        # Assume we cannot activate multiple guards at once
        if sol.t_events[idx].size > 0:
            return (
                sol.y_events[idx].flatten(),
                sol.t_events[idx][0],
                possible_modes[idx],
            )  # Flatten creates a copy. TODO: change to reshape
    return None, None, None


def compute_saltation_matrix(
    pre_event_state: np.ndarray,
    inputs: np.ndarray,
    dt: float,
    parameters: np.ndarray,
    pre_mode: str,
    post_mode: str,
    dynamics_dict: Dict[str, Dict[str, Callable]],
    resets_dict: Dict[str, Dict[str, Dict[str, Callable]]],
    guards_dict: Dict[str, Dict[str, Dict[str, Callable]]],
    post_event_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute saltation matrix for hybrid system mode transition.

    The saltation matrix captures the discontinuous jump in state covariance
    at mode boundaries, accounting for the reset map and guard surface geometry.

    Args:
        pre_event_state: State vector before mode transition.
        inputs: Control input vector.
        dt: Time step.
        parameters: System parameters.
        pre_mode: Mode before transition.
        post_mode: Mode after transition.
        dynamics_dict: Dictionary of dynamics functions per mode.
        resets_dict: Dictionary of reset maps per transition.
        guards_dict: Dictionary of guard functions per transition.
        post_event_state: Optional state after reset. Computed if not provided.

    Returns:
        Saltation matrix for covariance propagation through mode transition.

    Note:
        TODO: Add in time dependencies.
    """
    if post_event_state is None:
        # Compute reset if post event state is not given
        post_event_state = resets_dict[pre_mode][post_mode]["r"](
            pre_event_state, inputs, dt, parameters
        ).reshape(np.shape(pre_event_state))

    DxR = resets_dict[pre_mode][post_mode]["R"](pre_event_state, inputs, dt, parameters)
    DxG = guards_dict[pre_mode][post_mode]["G"](pre_event_state, inputs, dt, parameters)
    f_pre = dynamics_dict[pre_mode]["f_cont"](pre_event_state, inputs, dt, parameters).reshape(
        np.shape(pre_event_state)
    )
    f_post = dynamics_dict[post_mode]["f_cont"](post_event_state, inputs, dt, parameters).reshape(
        np.shape(pre_event_state)
    )

    salt = DxR + np.outer((f_post - DxR @ f_pre), DxG) / (DxG @ f_pre)
    return salt
