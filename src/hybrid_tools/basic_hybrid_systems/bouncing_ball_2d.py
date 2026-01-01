"""2D Bouncing Ball Hybrid System.

This module defines a 2D bouncing ball hybrid dynamical system with two modes.
States: [qx, qy, qx_dot, qy_dot] - horizontal pos, vertical pos, horizontal vel, vertical vel
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
    Returns HybridDynamicalSystem: 2D Bouncing Ball hybrid system.

    2D Bouncing Ball: Modes are {'I','J'}. e is coefficient of restitution.
    States: [qx, qy, qx_dot, qy_dot] - horizontal pos, vertical pos, horizontal vel, vertical vel
    """
    qx, qy, qx_dot, qy_dot, e, g, ux, uy, dt = sp.symbols("qx qy qx_dot qy_dot e g ux uy dt")

    """ Define the states and inputs. """
    inputs = Matrix([ux, uy])
    states = Matrix([qx, qy, qx_dot, qy_dot])

    """ Defining the dynamics of the system. """
    # Both modes have same dynamics:
    # Horizontal: constant velocity (no gravity)
    # Vertical: free fall under gravity with boosters
    # qx_dot' = qx_dot, qx_dot_dot = ux (horizontal control)
    # qy_dot' = qy_dot, qy_dot_dot = -g + uy (vertical with gravity and control)
    fI = Matrix([qx_dot, qy_dot, ux, -g + uy])
    fJ = Matrix([qx_dot, qy_dot, ux, -g + uy])

    """ Define the measurements of the system. """
    # Measure all positions and velocities
    yI = Matrix([qx, qy, qx_dot, qy_dot])
    yJ = Matrix([qx, qy, qx_dot, qy_dot])

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
    # When ball hits ground (qy <= 0):
    # - Horizontal position and velocity stay the same
    # - Vertical position stays at 0
    # - Vertical velocity reverses with energy loss
    rIJ = Matrix([qx, qy, qx_dot, -e * qy_dot])
    rJI = Matrix([qx, qy, qx_dot, qy_dot])

    """ Take the jacobian of resets with respect to states. """
    RIJ = rIJ.jacobian(states)
    RJI = rJI.jacobian(states)

    """ Define guards. """
    # Guard triggers when ball hits ground: qy <= 0
    gIJ = Matrix([qy])
    gJI = Matrix([qy_dot])

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
    n_states = 4
    n_inputs = 2
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
