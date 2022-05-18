import numpy as np
from itertools import count

from opentorsion.shaft_element import Shaft
from opentorsion.disk_element import Disk
from opentorsion.gear_element import Gear
from opentorsion.assembly import Assembly
from opentorsion.plots import Plots


def back_to_back_testbench():
    """
    Example of a powertrain with shaft, lumped mass and gear elements.
    """

    ## Powertrain component parameters such as stiffness (K) and moment of inertia (J).

    J_IM = 0.196  # Induction motor inertia (kgm^2)
    J_SRM = 0.575  # Synchronous reluctance motor inertia (kgm^2)
    Ig = 0  # Gear inertia
    J_coupling = 17e-3 + 17e-3 + 37e-3 * (0.55 - 2 * 0.128)  # Coupling inertia (kgm^2)
    K_coupling = 1 / (1 / 41300 + 1 / 41300)  # Coupling stiffness (Nm/rad)

    ## A list for each powertrain element type
    shafts = []
    disks = []
    gears = []

    ## Powertrain element objects are initiated and appended to corresponding list.

    disks.append(
        Disk(0, J_SRM)
    )  # A motor as a lumped mass element, inputs: node number and inertia

    ## Shaft element inputs: left node number, right node number, length, outer diameter.
    ## Alternatively, material parameters such as stiffness and and inertia can be inputted instead.
    shafts.append(Shaft(0, 1, None, None, k=K_coupling, I=J_coupling))  # Coupling

    ## Roll with varying dimensions as shaft elements
    shafts.append(Shaft(1, 2, 185, 100))
    shafts.append(Shaft(2, 3, 335, 119))
    shafts.append(Shaft(3, 4, 72, 125))
    shafts.append(Shaft(4, 5, 150, 320))
    shafts.append(Shaft(5, 6, 3600, 320, idl=287))
    shafts.append(Shaft(6, 7, 150, 320))
    shafts.append(Shaft(7, 8, 72, 125))
    shafts.append(Shaft(8, 9, 335, 119))
    shafts.append(Shaft(9, 10, 185, 100))
    ##

    shafts.append(Shaft(10, 11, None, None, k=K_coupling, I=J_coupling))  # Coupling

    ## Gear element, inputs: node number, gear inertia and gear ratio
    gears.append(gear1 := Gear(11, Ig, 1))  # Gear pinion
    gears.append(Gear(12, Ig, 1.95, parent=gear1))  # Gear
    ##

    disks.append(Disk(12, J_IM))  # Motor

    ## Refer to shaft_element.py, disk_element.py and gear_element.py documentation for more details.

    ## The shaft, disk (optional) and gear element (optional) lists are inputted to an assembly object.
    assembly = Assembly(shafts, disk_elements=disks, gear_elements=gears)

    return assembly


if __name__ == "__main__":

    assembly = back_to_back_testbench()  ## An assembly of a powertrain

    omegas_damped, eigenfrequencies, damping_ratios = assembly.modal_analysis()

    print(eigenfrequencies.round(3))
