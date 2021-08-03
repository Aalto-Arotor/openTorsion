from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.sparse import linalg as las

from opentorsion.disk_element import Disk
from opentorsion.shaft_element import Shaft
from opentorsion.gear_element import Gear
from opentorsion.induction_motor import Induction_motor
from opentorsion.assembly import Assembly
from opentorsion.plots import Plots


"""Example cases"""


def ex_1():
    """
    Two disks connected with a shaft.
    Eigenfrequencies and a campbell diagram.
    """
    shafts, disks = [], []

    Ip = 400  # Disk inertia (kgm²)
    L = 2.5e3  # Shaft length (mm)
    D = 0.2e3  # Shaft outer diameter (mm)

    node = 0
    disks.append(Disk(node, Ip))
    shafts.append(Shaft(node, node + 1, L, D))
    node += 1
    disks.append(Disk(node, Ip))

    assembly = Assembly(shaft_elements=shafts, disk_elements=disks)

    # Print the eigenfrequencies of the powertrain
    o_d, eigenfrequencies, d_r = assembly.modal_analysis()
    print(eigenfrequencies.round(3))

    # Campbell diagram
    plot_tools = Plots(assembly)
    plot_tools.campbell_diagram(frequency_range=10)

    return print("Example 1, complete")


def ex_2():
    """Powertrain with shafts, disks and gears"""
    # Induction motor inertia
    J_IM = 0.196  # kgm^2
    # Synchronous reluctance motor inertia
    J_SRM = 0.575  # kgm^2
    # Gear inertia
    Ig = 0.007  # kgm^2

    # Inertias
    J_hub1 = 17e-3  # kgm^2
    J_hub2 = 17e-3  # kgm^2
    J_tube = 37e-3 * (0.55 - 2 * 0.128)  # kgm^2
    J_coupling = J_hub1 + J_hub2 + J_tube  # kgm^2
    # Stiffnesses
    K_insert1 = 41300  # Nm/rad
    K_insert2 = 41300  # Nm/rad
    K_tube = 389000 * (0.55 - 2 * 0.128)  # Nm/rad
    K_coupling = 1 / (1 / K_insert1 + 1 / K_tube)  # Nm/rad

    shafts, disks, gears = [], [], []

    rho = 7850  # Material density
    G = 81e9  # Shear modulus

    disks.append(Disk(0, J_IM))  # Motor represented as a mass
    gears.append(gear1 := Gear(0, Ig, 1.95))  # Gear
    gears.append(Gear(1, Ig, 1, parent=gear1))  # Gear

    shafts.append(Shaft(1, 2, None, None, k=40050, I=0.045))  # Coupling

    # Roll with varying diameters
    shafts.append(Shaft(2, 3, 185, 100))
    shafts.append(Shaft(3, 4, 335, 119))
    shafts.append(Shaft(4, 5, 72, 125))
    shafts.append(Shaft(5, 6, 150, 320))
    shafts.append(Shaft(6, 7, 3600, 320, idl=287))
    shafts.append(Shaft(7, 8, 150, 320))
    shafts.append(Shaft(8, 9, 72, 125))
    shafts.append(Shaft(9, 10, 335, 119))
    shafts.append(Shaft(10, 11, 185, 100))
    ##

    shafts.append(Shaft(11, 12, None, None, k=180e3, I=15e-4))  # Torque transducer
    shafts.append(Shaft(12, 13, None, None, k=40050, I=0.045))  # Coupling
    disks.append(Disk(13, J_SRM))  # Motor represented as a mass

    assembly = Assembly(shaft_elements=shafts, disk_elements=disks, gear_elements=gears)

    plot_tools = Plots(assembly)
    plot_tools.figure_2D()  # A 2D representation of the powertrain
    plot_tools.figure_eigenmodes()  # Plot the eigenmodes of the powertrain

    return print("Example 2, complete")


def ex_3(linear_parameters=True, noload=True):
    """Induction motor and two disks connected with a shaft"""
    if noload:
        if linear_parameters:
            parameters_nonlinear = (
                np.array([23.457, 19.480, 30.470, 30.030, 28.904]) * 1e-3
            )
            parameters_linear = (
                np.array([23.486, 18.900, 13.119, 12.981, 11.963]) * 1e-3
            )
        else:
            parameters_nonlinear = (
                np.array([23.457, 19.480, 30.470, 30.030, 28.904]) * 1e-3
            )
            parameters_linear = (
                np.array([23.457, 19.480, 30.470, 30.030, 28.904]) * 1e-3
            )

    else:
        if linear_parameters:
            parameters_nonlinear = (
                np.array([24.492, 19.450, 29.386, 29.004, 27.921]) * 1e-3
            )
            parameters_linear = (
                np.array([23.342, 18.668, 12.686, 12.507, 11.579]) * 1e-3
            )

        else:
            parameters_nonlinear = (
                np.array([24.492, 19.450, 29.386, 29.004, 27.921]) * 1e-3
            )
            parameters_linear = (
                np.array([24.492, 19.450, 29.386, 29.004, 27.921]) * 1e-3
            )

    f = 60
    p = 4
    speed = 895.3
    voltage = 4000
    motor_holopainen = Induction_motor(
        0,
        speed,
        f,
        p,
        voltage=voltage,
        small_signal=True,
        circuit_parameters_nonlinear=parameters_nonlinear,
        circuit_parameters_linear=parameters_linear,
    )

    shafts = [Shaft(0, 1, 0, 0, k=9.775e6)]
    disks = [Disk(0, 211.4), Disk(1, 8.3)]
    assembly = Assembly(shafts, disk_elements=disks, motor_elements=[motor_holopainen])

    damped, freqs, ratios = assembly.modal_analysis()

    for f in damped / (np.pi * 2):
        print(f)

    return print("Example 3, complete")


def ex_4():
    """
    Planetary gear:
    Input shaft attached to sun gear and ring gear, output shaft attached to carrier and planet gears.

    Three cases:
    1. Ring gear stationary, sun and carrier moving. Output speed positive.
    2. Sun gear stationary, ring and carrier moving. Output speed 2. > 1.
    3. Carrier stationary, sun and ring moving. Output speed negative.
    """

    pgs = 5  # number of planet gears
    gears = []
    case = 0

    if case == 0:

        sun_gear = Gear(n, I=0, R=6)
        gears.append(sun_gear)
        n += 1

        planet_gear = Gear(n, I=0, R=3, parent=sun_gear)
        gears.append(planet_gear)
        n += 1

        ring_gear = Gear(n, I=0, R=9, parent=planet_gear)
        gears.append(ring_gear)

        return n, gears

    if case == 1:

        sun_gear = Gear(n, I=0, R=56)
        gears.append(sun_gear)
        n += 1

        carrier = Gear(n, I=0, R=95)  # R_c = R_s + R_p
        gears.append(carrier)
        n += 1

        for i in range(pgs):
            planet_gear = Gear(n, I=0, R=39, parent=sun_gear, parent2=carrier)
            gears.append(planet_gear)
            n += 1
        n -= 1

        return carrier.node, gears

    if case == 2:

        ring_gear = Gear(n, I=0, R=12)
        gears.append(ring_gear)
        n += 1

        carrier = Gear(n, I=0, R=9)  # R_c = R_s + R_p
        gears.append(carrier)
        n += 1

        for i in range(pgs):
            planet_gear = Gear(n, I=0, R=-3, parent=carrier, parent2=ring_gear)
            gears.append(planet_gear)
            n += 1
        n -= 1

        return carrier.node, gears

    if case == 3:

        sun_gear = Gear(n, I=0, R=56)
        gears.append(sun_gear)
        n += 1

        ring_gear = Gear(n + pgs, I=0, R=134)
        gears.append(ring_gear)
        # n += 1

        for i in range(pgs):
            planet_gear = Gear(n, I=0, R=39, parent=sun_gear, parent2=ring_gear)
            gears.append(planet_gear)
            n += 1
        # n -= 1

        gears.pop(1)
        ring_gear = Gear(n, I=0, R=134, parent=sun_gear)
        gears.append(ring_gear)

    return print("Example 4, complete")


def ex_5():
    """Time-domain analysis of a powertrain"""
    return


if __name__ == "__main__":
    """Comment/uncomment to run examples"""
    # ex_1()
    ex_2()
    # ex_3()
    # ex_4()
    # ex_5()
