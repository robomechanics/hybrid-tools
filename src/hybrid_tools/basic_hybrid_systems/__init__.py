"""Basic Hybrid Systems.

This module provides pre-defined hybrid dynamical systems that can be imported
and used in scripts and applications.

Available systems:
- bouncing_ball: 1D bouncing ball with gravity
- bouncing_ball_2d: 2D bouncing ball with gravity
- simple_system: Simple 2-state hybrid system
"""

from hybrid_tools.basic_hybrid_systems.bouncing_ball import symbolic_dynamics as bouncing_ball
from hybrid_tools.basic_hybrid_systems.bouncing_ball_2d import (
    symbolic_dynamics as bouncing_ball_2d,
)
from hybrid_tools.basic_hybrid_systems.simple_system import symbolic_dynamics as simple_system

__all__ = ["bouncing_ball", "bouncing_ball_2d", "simple_system"]
