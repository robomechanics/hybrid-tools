"""1D Bouncing Ball Hybrid System.

This module defines a 1D bouncing ball hybrid dynamical system with two modes.
States: [q, q_dot] - position and velocity
Modes: {'I', 'J'} representing different phases of motion
Parameters: [e (coefficient of restitution), g (gravity)]
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
    Returns HybridDynamicalSystem: 1D Bouncing Ball hybrid system.

    Bouncing Ball: Modes are {'I','J'}. e is coefficient of restitution.
    States: [q, q_dot] - position and velocity
    """
    q, q_dot, e, g, u, dt = sp.symbols("q q_dot e g u dt")

    """ Define the states and inputs. """
    inputs = Matrix([u])
    states = Matrix([q, q_dot])

    """ Defining the dynamics of the system. """
    # Both modes have same dynamics: free fall under gravity with boosters
    # q_dot' = q_dot, q_dot_dot = -g + u
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
    # r = [q, -e * q_dot]
    rIJ = Matrix([q, -e * q_dot])
    rJI = Matrix([q, q_dot])

    """ Take the jacobian of resets with respect to states. """
    RIJ = rIJ.jacobian(states)
    RJI = rJI.jacobian(states)

    """ Define guards. """
    # Guard triggers when ball hits ground: q <= 0
    gIJ = Matrix([q])
    gJI = Matrix([q_dot])

    """ Take the jacobian of guards with respect to states. """
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
