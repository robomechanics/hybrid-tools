"""Simple Hybrid System.

This module defines a simple hybrid dynamical system with two modes.
States: [x1, x2]
Modes: {'I', 'J'} representing different phases of motion
Parameters: None
"""

import numpy as np
import sympy as sp
from sympy.matrices import Matrix

from hybrid_tools import (
    HybridDynamicalSystem,
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
    Returns HybridDynamicalSystem: Simple hybrid system with two modes.

    States: [x1, x2]
    Modes: {'I', 'J'}
    """
    x1, x2, u1, u2, dt = sp.symbols("x1 x2 u1 u2 dt")

    """ Define the states and inputs. """
    inputs = Matrix([u1, u2])  # note inputs also specify how process noise can be introduced.
    states = Matrix([x1, x2])

    """ Defining the dynamics of the system. """
    fI = Matrix([1 + u1, -1 + u2])
    fJ = Matrix([1 + u1, 1 + u2])

    """ Define the measurements of the system. """
    yI = Matrix([x1, x2])
    yJ = Matrix([x1, x2])

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
    rIJ = Matrix([x1, x2])

    """ Take the jacobian of resets with respect to states. """
    RIJ = rIJ.jacobian(states)

    """ Define guards. """
    gIJ = Matrix([-x1])

    """ Take the jacobian of guards with respect to states. """
    GIJ = gIJ.jacobian(states)

    """ Define the parameters of the system. """
    parameters = Matrix([])

    rIJ_func = sp.lambdify((states, inputs, dt, parameters), rIJ)
    RIJ_func = sp.lambdify((states, inputs, dt, parameters), RIJ)

    gIJ_func = sp.lambdify((states, inputs, dt, parameters), gIJ)
    GIJ_func = sp.lambdify((states, inputs, dt, parameters), GIJ)

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
        ]
    )

    guards = create_guards(
        [
            ("I", "J", ModeGuard(g=gIJ_func, G=GIJ_func)),
        ]
    )

    # Define noise matrices
    n_states = 2
    W_global = 0.01 * np.eye(n_states)
    V_global = 0.025 * np.eye(n_states)
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
