"""Type definitions for hybrid dynamical systems.

This module provides dataclass definitions for organizing hybrid system components
including dynamics, resets, and guards.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from graphviz import Digraph


@dataclass
class ModeDynamics:
    """Dynamics functions for a single mode.

    Attributes:
        f_cont: Continuous dynamics function
        A_disc: Discrete state Jacobian (linearization)
        B_disc: Discrete input Jacobian
        y: Measurement function
        C: Measurement Jacobian
    """

    f_cont: Callable
    A_disc: Callable
    B_disc: Callable
    y: Callable
    C: Callable


@dataclass
class ModeReset:
    """Reset map for state transitions between modes.

    Attributes:
        r: Reset function that maps pre-transition state to post-transition state
        R: Reset Jacobian (linearization of reset map)
    """

    r: Callable
    R: Callable


@dataclass
class ModeGuard:
    """Guard condition for triggering mode transitions.

    Attributes:
        g: Guard function that defines transition condition (triggers when g <= 0)
        G: Guard Jacobian (gradient of guard function)
    """

    g: Callable
    G: Callable


@dataclass
class ModeNoise:
    """Noise covariance matrices for a single mode.

    Attributes:
        W: Process noise covariance matrix
        V: Measurement noise covariance matrix
    """

    W: "np.ndarray"  # Forward reference for numpy
    V: "np.ndarray"


# Type aliases for nested dictionary structures
Dynamics = dict[str, ModeDynamics]
Resets = dict[str, dict[str, ModeReset]]
Guards = dict[str, dict[str, ModeGuard]]
Noises = dict[str, ModeNoise]


@dataclass
class HybridDynamicalSystem:
    """Complete specification of a hybrid dynamical system.

    Bundles together the dynamics, resets, guards, and noise specifications
    that define a hybrid system with discrete mode transitions.

    Attributes:
        dynamics: Dynamics functions for each mode
        resets: Reset maps for state transitions between modes
        guards: Guard conditions for triggering mode transitions
        noises: Process and measurement noise covariance matrices per mode
    """

    dynamics: Dynamics
    resets: Resets
    guards: Guards
    noises: Noises

    def __post_init__(self) -> None:
        """Validate the hybrid system structure after initialization.

        Ensures that:
        1. All modes in guards have corresponding dynamics
        2. All modes in resets have corresponding dynamics
        3. Guards and resets define the same transitions
        4. All modes have noise specifications

        Raises:
            ValueError: If validation fails
        """
        # Get all modes from dynamics
        dynamics_modes = set(self.dynamics.keys())

        # Validate guards
        for source_mode, targets in self.guards.items():
            if source_mode not in dynamics_modes:
                raise ValueError(
                    f"Guard source mode '{source_mode}' not found in dynamics. "
                    f"Available modes: {dynamics_modes}"
                )
            for target_mode in targets.keys():
                if target_mode not in dynamics_modes:
                    raise ValueError(
                        f"Guard target mode '{target_mode}' (from '{source_mode}') "
                        f"not found in dynamics. Available modes: {dynamics_modes}"
                    )

        # Validate resets
        for source_mode, targets in self.resets.items():
            if source_mode not in dynamics_modes:
                raise ValueError(
                    f"Reset source mode '{source_mode}' not found in dynamics. "
                    f"Available modes: {dynamics_modes}"
                )
            for target_mode in targets.keys():
                if target_mode not in dynamics_modes:
                    raise ValueError(
                        f"Reset target mode '{target_mode}' (from '{source_mode}') "
                        f"not found in dynamics. Available modes: {dynamics_modes}"
                    )

        # Validate that guards and resets define the same transitions
        guard_transitions = {
            (src, tgt) for src, targets in self.guards.items() for tgt in targets.keys()
        }
        reset_transitions = {
            (src, tgt) for src, targets in self.resets.items() for tgt in targets.keys()
        }

        if guard_transitions != reset_transitions:
            missing_in_resets = guard_transitions - reset_transitions
            missing_in_guards = reset_transitions - guard_transitions

            error_msg = "Mismatch between guards and resets transitions:\n"
            if missing_in_resets:
                error_msg += f"  Guards define but resets missing: {missing_in_resets}\n"
            if missing_in_guards:
                error_msg += f"  Resets define but guards missing: {missing_in_guards}"
            raise ValueError(error_msg)

        # Validate that all modes have noise specifications
        noise_modes = set(self.noises.keys())
        if dynamics_modes != noise_modes:
            missing_noise = dynamics_modes - noise_modes
            extra_noise = noise_modes - dynamics_modes

            error_msg = "Mismatch between dynamics and noise specifications:\n"
            if missing_noise:
                error_msg += f"  Modes missing noise specs: {missing_noise}\n"
            if extra_noise:
                error_msg += f"  Extra noise specs for non-existent modes: {extra_noise}"
            raise ValueError(error_msg)

    def draw_mode_graph(self, filename: str = "mode_graph.png", format: str = "png") -> None:
        """Draw a directed graph showing mode transitions.

        Creates a visualization of the hybrid system's mode structure, showing
        all modes as nodes and guard-triggered transitions as directed edges.

        Args:
            filename: Output filename for the graph image
            format: Output format ('png', 'pdf', 'svg', etc.)

        Example:
            >>> hybrid_system.draw_mode_graph("my_system.png")
        """
        # Create directed graph
        dot = Digraph(comment="Hybrid System Mode Graph")
        dot.attr(rankdir="LR")  # Left to right layout

        # Add all modes as nodes
        all_modes = set(self.dynamics.keys())
        for mode in all_modes:
            dot.node(mode, mode, shape="circle", style="filled", fillcolor="lightblue")

        # Add transitions as edges based on guards
        for source_mode, targets in self.guards.items():
            for target_mode in targets.keys():
                dot.edge(source_mode, target_mode, label=f"{source_mode}→{target_mode}")

        # Render the graph
        output_path = filename.rsplit(".", 1)[0]  # Remove extension if present
        dot.render(output_path, format=format, cleanup=True)
        print(f"Mode graph saved to {output_path}.{format}")


def create_dynamics(mode_dynamics_list: list[tuple[str, ModeDynamics]]) -> Dynamics:
    """Create a Dynamics dictionary from a list of (mode_name, ModeDynamics) tuples.

    Args:
        mode_dynamics_list: List of tuples containing (mode_name, ModeDynamics)

    Returns:
        Dynamics dictionary mapping mode names to ModeDynamics objects

    Example:
        >>> dynamics = create_dynamics([
        ...     ("I", ModeDynamics(f_cont=fI, A_disc=AI, B_disc=BI, y=yI, C=CI)),
        ...     ("J", ModeDynamics(f_cont=fJ, A_disc=AJ, B_disc=BJ, y=yJ, C=CJ)),
        ... ])
    """
    return dict(mode_dynamics_list)


def create_resets(mode_transitions: list[tuple[str, str, ModeReset]]) -> Resets:
    """Create a Resets dictionary from a list of (source_mode, target_mode, ModeReset) tuples.

    Args:
        mode_transitions: List of tuples containing (source_mode, target_mode, ModeReset)

    Returns:
        Resets dictionary with nested structure: source_mode -> target_mode -> ModeReset

    Example:
        >>> resets = create_resets([
        ...     ("I", "J", ModeReset(r=rIJ, R=RIJ)),
        ...     ("J", "I", ModeReset(r=rJI, R=RJI)),
        ... ])
    """
    result: Resets = {}
    for source_mode, target_mode, reset in mode_transitions:
        if source_mode not in result:
            result[source_mode] = {}
        result[source_mode][target_mode] = reset
    return result


def create_guards(mode_transitions: list[tuple[str, str, ModeGuard]]) -> Guards:
    """Create a Guards dictionary from a list of (source_mode, target_mode, ModeGuard) tuples.

    Args:
        mode_transitions: List of tuples containing (source_mode, target_mode, ModeGuard)

    Returns:
        Guards dictionary with nested structure: source_mode -> target_mode -> ModeGuard

    Example:
        >>> guards = create_guards([
        ...     ("I", "J", ModeGuard(g=gIJ, G=GIJ)),
        ...     ("J", "I", ModeGuard(g=gJI, G=GJI)),
        ... ])
    """
    result: Guards = {}
    for source_mode, target_mode, guard in mode_transitions:
        if source_mode not in result:
            result[source_mode] = {}
        result[source_mode][target_mode] = guard
    return result
