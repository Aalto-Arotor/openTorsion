import numpy as np
import unittest
import matplotlib.pyplot as plt

from opentorsion.shaft_element import Shaft
from opentorsion.disk_element import Disk
from opentorsion.gear_element import Gear
from opentorsion.assembly import Assembly


class Test(unittest.TestCase):
    """Unittests for the FEM"""

    # def test_induction_motor(self):
    #     f = 60
    #     p = 4
    #     omega = 895.3
    #     k = 9.775e6
    #     J1 = 211.4
    #     J2 = 8.3
    #     motor = induction_motor(0, 1, omega, f, p, J1, J2, k)

    #     L, R = motor.L(), motor.R()

    #     np.set_printoptions(suppress=True)

    #     drivetrain = Rotor([Shaft(0,1, 0, 0, k=9.775e6, I=0), Shaft(1, 2, 20, 5)], disk_elements=[Disk(0, 211.4), Disk(1, 8.3)])

    #     A, B = motor.state_space(drivetrain)

    #     lam, vec = LA.eig(R, L)
    #     print(np.abs(lam/(2*np.pi)).round(3))

    #     lam, vec = LA.eig(A, B)
    #     print(np.abs(lam/(2*np.pi)).round(3))

    # def test_hauptmann(self):
    #     nl = 0
    #     nr = 1
    #     rated_speed = 895.3
    #     omega = rated_speed*2.0*np.pi/60.0
    #     rated_torque = 39.73e3
    #     break_torque = 82.86e3
    #     freq = 60.0
    #     pole_pairs = 4.0
    #     J1 = 211.4
    #     J2 = 8.3
    #     k = 9.775e6

    #     motor = Motor_hauptmann(nl, nr, omega, rated_torque, break_torque, freq, pole_pairs, J1, J2, k)

    #     assembly = Rotor(None, motor_elements=[motor])

    #     omegas_damped, freqs, damping_ratios = assembly.modal_analysis()

    #     self.assertEqual(1, 1, "Hauptmann gives the wrong result")

    def test_shaft(self):
        J1 = 211.4
        J2 = 8.3
        k = 9.775e6

        shaft = Shaft(0, 1, 0, 0, k=k)
        disks = [Disk(0, J1), Disk(1, J2)]

        assembly = Assembly([shaft], disk_elements=disks)

        omegas_damped, freqs, damping_ratios = assembly.modal_analysis()

        self.assertEqual(
            freqs.round(1).tolist(), [0, 176.1], "shaft gives the wrong result"
        )

    def test_friswell_pagex(self):
        correct = [0.0, 425.4, 634.1, 3247.2]
        Im = 400e-6
        Ir = 700e-6
        Ig = 25e-6

        k_m = 10000
        k_R = 5000

        disks = [Disk(0, Im), Disk(3, Ir), Disk(4, Ir)]
        shafts = [
            Shaft(0, 1, 0, 0, k=k_m),
            Shaft(2, 3, 0, 0, k=k_R),
            Shaft(1, 4, 0, 0, k=k_R),
        ]
        gear1 = Gear(1, Ig, 1)
        gear2 = Gear(2, Ig, 1, parent=gear1)
        gears = [gear1, gear2]

        rotor = Assembly(shafts, disk_elements=disks, gear_elements=gears)

        _, freqs, _ = rotor.modal_analysis()

        self.assertEqual(freqs.round(1).tolist(), correct, "geared system incorrect")

    def test_friswell_ex971(self):
        correct = [0, 81.21, 141.21, 378.95, 536.36]
        Ip1to3 = 3.5e-3
        Ip4 = 0.15
        Ip5 = 0.05

        k1 = 12e3
        k2 = k1
        k3 = 14e3
        k4 = 10e3

        shafts = []
        disks = []
        n = 0
        disks.append(Disk(n, Ip1to3))
        shafts.append(Shaft(n, n + 1, 0, 0, k=k1))
        n += 1

        disks.append(Disk(n, Ip1to3))
        shafts.append(Shaft(n, n + 1, 0, 0, k=k2))
        n += 1

        disks.append(Disk(n, Ip1to3))
        shafts.append(Shaft(n, n + 1, 0, 0, k=k3))
        n += 1

        disks.append(Disk(n, Ip4))
        shafts.append(Shaft(n, n + 1, 0, 0, k=k4))
        n += 1

        disks.append(Disk(n, Ip5))

        assembly = Assembly(shafts, disk_elements=disks)

        a, freqs, b = assembly.modal_analysis()

        self.assertEqual(freqs.round(2).tolist(), correct, "geared system incorrect")

        # def friswell_9_6_3(self):
        correct = [0, 10.304, 20.479, 24.423]
        Ip0 = 3200
        Ip1 = 200
        Ip2 = 800
        Ip3 = 450
        L1 = 2.5e3
        L2 = 3.5e3
        D1 = 0.2e3
        D2 = 0.25e3
        R1 = 0.5e3
        R2 = 0.8e3
        G = 68e9

        disks = []
        shafts = []
        gears = []

        disks.append(Disk(0, Ip0))
        disks.append(Disk(5, Ip3))
        disks.append(Disk(6, Ip3))

        shafts.append(Shaft(0, 1, L1, D1, G=G, rho=0))
        shafts.append(Shaft(3, 5, L2, D2, G=G, rho=0))
        shafts.append(Shaft(4, 6, L2, D2, G=G, rho=0))

        gear1 = Gear(1, 200, 0.5)
        gear2 = Gear(2, 200, 0.5, parent=gear1)
        gear3 = Gear(3, 800, 0.8, parent=gear1)
        gear4 = Gear(4, 800, 0.8, parent=gear2)
        gears.append(gear1)
        gears.append(gear2)
        gears.append(gear3)
        gears.append(gear4)

        assembly = Assembly(shafts, disk_elements=disks, gear_elements=gears)

        # print(( assembly.K() * 1e-6).round(3))
        # print(assembly.E())
        # print(assembly.T(assembly.E()))
        a, freqs, b = assembly.modal_analysis()

        # print(freqs)
        self.assertEqual(freqs.round(3).tolist(), correct, "geared system incorrect")

    def test_friswell_09_09(self):
        L = 800e3
        m = 10
        mt = 200
        od = 0.78e3
        idl = 0.65e3
        G = 75e9
        rho = 8000

        freqs = [[0], [0], [0]]
        for j, ne in enumerate([5, 8, 50]):
            le = L / ne
            disks = []
            shafts = []
            for i in range(0, ne):
                shafts.append(Shaft(i, i + 1, le, od, idl=idl, G=G, rho=rho))

            disks.append(Disk(0, mt))
            disks.append(Disk(ne, m))
            assembly = Assembly(shafts, disk_elements=disks)
            a, freq, b = assembly.modal_analysis()
            freqs[j] = freq.round(4)[0:5].tolist()

        correct = [
            [0.0000, 1.9417, 4.0719, 6.5437, 9.1547],
            [0.0000, 1.9225, 3.9187, 6.0625, 8.4185],
            [0.0000, 1.9106, 3.8232, 5.7394, 7.6613],
        ]

        self.assertEqual(freqs, correct, "Shaft discretization not correct")


if __name__ == "__main__":
    unittest.main()
