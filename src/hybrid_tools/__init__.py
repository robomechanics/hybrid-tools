"""
Hybrid Tools - Salted Kalman Filter for Hybrid Dynamical Systems

This package provides tools for state estimation in hybrid dynamical systems,
including the Salted Kalman Filter (SKF) implementation.

References:
    Kong, N. J., Payne, J. J., Council, G., & Johnson, A. M. (2021).
    The Salted Kalman Filter: Kalman Filtering on Hybrid Dynamical Systems.
    Automatica, 131, 109752.
"""

from .hybrid_helper_functions import (
    compute_saltation_matrix,
    solve_ivp_dynamics_func,
    solve_ivp_extract_hybrid_events,
    solve_ivp_guard_funcs,
)
from .hybrid_simulator import HybridSimulator
from .skf import SKF

__version__ = "0.1.0"
__all__ = [
    "SKF",
    "HybridSimulator",
    "solve_ivp_dynamics_func",
    "solve_ivp_guard_funcs",
    "solve_ivp_extract_hybrid_events",
    "compute_saltation_matrix",
]
