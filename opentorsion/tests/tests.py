import numpy as np
import unittest
import matplotlib.pyplot as plt
import scipy.linalg as LA

from opentorsion.shaft_element import Shaft
from opentorsion.disk_element import Disk
from opentorsion.gear_element import Gear
from opentorsion.assembly import Assembly
from opentorsion.excitation import SystemExcitation
from opentorsion.plots import Plots


class Test(unittest.TestCase):
    """
    Unittests for openTorsion
    """

    def test_shaft(self):
        J1 = 211.4
        J2 = 8.3
        k = 9.775e6

        shaft = Shaft(0, 1, 0, 0, k=k)
        disks = [Disk(0, J1), Disk(1, J2)]

        assembly = Assembly([shaft], disk_elements=disks)

        omegas_damped, freqs, damping_ratios = assembly.modal_analysis()

        self.assertEqual(
            freqs.round(1).tolist(),
            [0, 0, 176.1, 176.1],
            "shaft gives the wrong result",
        )

    def test_friswell_pagex(self):
        correct = [0.0, 0.0, 425.4, 425.4, 634.1, 634.1, 3247.2, 3247.2]
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
        correct = [0, 0, 81.21, 81.21, 141.21, 141.21, 378.95, 378.95, 536.36, 536.36]
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
        correct = [0, 0, 10.304, 10.304, 20.479, 20.479, 24.423, 24.423]
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

        a, freqs, b = assembly.modal_analysis()

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
            freqs[j] = freq.round(4)[0:10].tolist()

        correct = [
            [
                0.0000,
                0.0000,
                1.9417,
                1.9417,
                4.0719,
                4.0719,
                6.5437,
                6.5437,
                9.1547,
                9.1547,
            ],
            [
                0.0000,
                0.0000,
                1.9225,
                1.9225,
                3.9187,
                3.9187,
                6.0625,
                6.0625,
                8.4185,
                8.4185,
            ],
            [
                0.0000,
                0.0000,
                1.9106,
                1.9106,
                3.8232,
                3.8232,
                5.7394,
                5.7394,
                7.6613,
                7.6613,
            ],
        ]

        self.assertEqual(freqs, correct, "Shaft discretization not correct")

    def test_friswell_09_06(self):
        ## Here state matrix composition and eigenmode calculation is identical to assembly,
        ## however they are done manually here as the stiffness matrix in this example had to be modified
        k1 = 10e6
        k2 = 2.5e6
        k3 = k2
        k4 = k2
        k5 = k2
        k6 = k2
        k7 = 40e6
        k8 = k7
        k9 = 3e6
        k10 = k1

        m1 = 90
        m2 = 140
        m3 = m2
        m4 = m2
        m5 = m2
        m6 = m1
        m7 = 350
        m8 = 500
        m9 = 22000

        shafts, disks = [], []
        n = 0
        disks.append(Disk(n, I=m1))
        shafts.append(Shaft(n, n + 1, None, None, k=k2))
        n += 1
        disks.append(Disk(n, I=m2))
        shafts.append(Shaft(n, n + 1, None, None, k=k3))
        n += 1
        disks.append(Disk(n, I=m3))
        shafts.append(Shaft(n, n + 1, None, None, k=k4))
        n += 1
        disks.append(Disk(n, I=m4))
        shafts.append(Shaft(n, n + 1, None, None, k=k5))
        n += 1
        disks.append(Disk(n, I=m5))
        shafts.append(Shaft(n, n + 1, None, None, k=k6))
        n += 1
        disks.append(Disk(n, I=m6))
        shafts.append(Shaft(n, n + 1, None, None, k=k7))
        n += 1
        disks.append(Disk(n, I=m7))
        shafts.append(Shaft(n, n + 1, None, None, k=k8))
        n += 1
        disks.append(Disk(n, I=m8))
        shafts.append(Shaft(n, n + 1, None, None, k=k9))
        n += 1
        disks.append(Disk(n, I=m9))

        assembly = Assembly(shafts, disk_elements=disks)
        M, K = assembly.M(), assembly.K()
        K[0, 0] += k1
        K[5, 5] += k10
        Z = np.zeros(M.shape, dtype=np.float64)

        A = np.vstack([np.hstack([Z, K]), np.hstack([-M, Z])])
        B = np.vstack([np.hstack([M, Z]), np.hstack([Z, M])])

        lam, vec = assembly._eig(A, B)
        lam = lam[::2]
        vec = vec[: int(vec.shape[0] / 2)]
        vec = vec[:, ::2]

        inds = np.argsort(np.abs(lam))
        sorted_vec = np.zeros(vec.shape)
        for i, v in enumerate(inds):
            sorted_vec[:, i] = vec[:, v]

        correct_vec = np.array(
            [
                [
                    6.60605199481177e-5,
                    0.00667406247011805,
                    -0.00451698937983853,
                    0.0122100078859736,
                    0.0137040069782762,
                    0.00956129679649614,
                    0.103042990675201,
                    -4.54419853685659e-6,
                    -2.51310389680045e-9,
                ],
                [
                    0.000330077639141967,
                    0.0320211322617543,
                    -0.0205225369398110,
                    0.0506773033773183,
                    0.0463814794663998,
                    0.0258629404003599,
                    -0.0175470307647990,
                    8.24220088009509e-6,
                    5.04281185022623e-8,
                ],
                [
                    0.000592346256462216,
                    0.0472988420882423,
                    -0.0219519455363917,
                    0.0221751991784241,
                    -0.0374962957739650,
                    -0.0501676592843986,
                    0.00298794179246030,
                    -6.63323141738523e-5,
                    -1.86290692935124e-6,
                ],
                [
                    0.000851477070985716,
                    0.0477029658395100,
                    -0.00778997721920759,
                    -0.0356311433682274,
                    -0.0271470197620900,
                    0.0529032836459798,
                    -0.000508118203721655,
                    0.000562164074952234,
                    6.88615997175939e-5,
                ],
                [
                    0.00110609740349712,
                    0.0331064228274443,
                    0.0119048251136658,
                    -0.0463513916947389,
                    0.0514218915409095,
                    -0.0328936585423667,
                    8.24492991471238e-5,
                    -0.00476784120131347,
                    -0.00254544227606329,
                ],
                [
                    0.00135485846790594,
                    0.00809923955809763,
                    0.0231442212898544,
                    0.00418112343958315,
                    0.000769152819570238,
                    -0.00125827338620105,
                    9.90422170641066e-6,
                    0.0404375594726567,
                    0.0940912846852475,
                ],
                [
                    0.00170883228914918,
                    0.00845877027092123,
                    0.0289722758348375,
                    0.00816268781687740,
                    -0.00228201460479100,
                    0.000584856450504386,
                    4.64572960193113e-6,
                    0.0361514836562086,
                    -0.0237522688507098,
                ],
                [
                    0.00206139171945407,
                    0.00840268481139112,
                    0.0315850852333137,
                    0.0104588012300431,
                    -0.00443714557289372,
                    0.00210174058558432,
                    -6.45089912032581e-6,
                    -0.0280060906386327,
                    0.00311362987148320,
                ],
                [
                    0.00672968503001360,
                    -0.000209129883290887,
                    -0.000343282659273933,
                    -6.07886503115980e-5,
                    1.35246108824622e-5,
                    -4.50526327439023e-6,
                    6.13082758290678e-9,
                    2.01919434383870e-5,
                    -6.09911575562020e-7,
                ],
            ]
        )

        normalized_correct_vec = np.zeros(correct_vec.shape)
        for i in range(len(inds)):
            normalized_correct_vec[:, i] = correct_vec[:, i] / (
                LA.norm(correct_vec[:, i])
            )

        normalized_correct_vec = normalized_correct_vec.round(2)
        sorted_vec = sorted_vec.round(2)

        normalized_correct_vec = abs(normalized_correct_vec)
        sorted_vec = abs(sorted_vec)
        correct_eigenmodes = normalized_correct_vec.tolist()
        eigenmodes = sorted_vec.tolist()

        self.assertEqual(
            eigenmodes, correct_eigenmodes, "Eigenmode calculation not correct"
        )

    def test_mass_matrix(self):
        correct_M = np.array([[10, 0], [0, 10]])
        shafts = []
        disks = []
        disks.append(Disk(0, I=10))
        shafts.append(Shaft(0, 1, None, None, k=428400, I=0))
        disks.append(Disk(1, I=10))

        assembly = Assembly(shafts, disk_elements=disks)
        M = assembly.M()

        mass_values, correct = [], []
        correct = correct_M.tolist()
        mass_values = M.tolist()

        self.assertEqual(mass_values, correct, "Mass matrix not correct")

    def test_stiffness_matrix(self):
        correct_K = np.array([[428400, -428400], [-428400, 428400]])
        shafts = []
        disks = []
        disks.append(Disk(0, I=10))
        shafts.append(Shaft(0, 1, None, None, k=428400, I=0))
        disks.append(Disk(1, I=10))

        assembly = Assembly(shafts, disk_elements=disks)
        K = assembly.K()

        stiffness_values, correct = [], []
        correct = correct_K.tolist()
        stiffness_values = K.tolist()

        self.assertEqual(stiffness_values, correct, "Stiffness matrix not correct")

    def test_stiffness_matrix_2(self):
        correct_K = np.array(
            [[428400, -428400, 0], [-428400, 856800, -428400], [0, -428400, 428400]]
        )
        shafts = []
        disks = []
        disks.append(Disk(0, I=10))
        shafts.append(Shaft(0, 1, None, None, k=428400, I=0))
        disks.append(Disk(1, I=10))
        shafts.append(Shaft(1, 2, None, None, k=428400, I=0))

        assembly = Assembly(shafts, disk_elements=disks)
        K = assembly.K()

        stiffness_values, correct = [], []
        correct = correct_K.tolist()
        stiffness_values = K.tolist()

        self.assertEqual(stiffness_values, correct, "Stiffness matrix not correct")

    def test_modal_damping_matrix(self):
        shafts = []
        disks = []
        disks.append(Disk(0, I=10))
        shafts.append(Shaft(0, 1, None, None, k=428400, I=0))
        disks.append(Disk(1, I=10))

        assembly = Assembly(shafts, disk_elements=disks)
        assembly.xi = 0.02

        ## D. Inman, in Encyclopedia of Vibration, 2001, Critical Damping in Lumped Parameter Models
        M = assembly.M()
        M_inv = LA.inv(M)
        K = assembly.K()
        M_K_M = LA.sqrtm(M_inv) @ K @ LA.sqrtm(M_inv)
        correct_C = 2 * 0.02 * (LA.sqrtm(M) @ LA.sqrtm(M_K_M) @ LA.sqrtm(M))
        correct_C = correct_C.round(4)

        C = assembly.C_modal(assembly.M(), assembly.K())
        C = C.round(4)

        damping_values, correct = [], []
        correct = correct_C.tolist()
        damping_values = C.tolist()

        self.assertEqual(damping_values, correct, "Modal damping matrix not correct")

    def test_frequency_domain_excitation_matrix_shape(self):
        correct_U_shape = (4, 9)

        dofs = 4
        omegas = np.arange(1, 10, 1)
        U = SystemExcitation(dofs, omegas)
        U_shape = U.excitation_amplitudes().shape

        self.assertEqual(U_shape, correct_U_shape, "Excitation matrix not correct")

    def test_vibratory_torque(self):
        omegas = [
            377.07005102,
            565.60507654,
            754.14010205,
            942.67512756,
            1131.21015307,
            1319.74517859,
            1508.2802041,
        ]
        amplitudes = [
            5218.9072902,
            51899.13360809,
            6958.5430536,
            9857.9359926,
            84962.89738629,
            5218.9072902,
            3189.3322329,
        ]
        correct_T_v = np.array(
            [
                [
                    1.49729531e2,
                    8.16086180e2,
                    9.06632152e1,
                    2.15202352e2,
                    1.07777439e3,
                    1.68374089e1,
                    5.71958086,
                ],
                [
                    3.26530199e2,
                    3.96288796e3,
                    7.90700574e2,
                    3.03291995e3,
                    2.24215649e4,
                    4.39597289e2,
                    1.50103728e2,
                ],
                [
                    1.83532558e2,
                    3.18463695e3,
                    7.03339886e2,
                    2.82005601e3,
                    2.13497396e4,
                    4.24731919e2,
                    1.46465771e2,
                ],
            ]
        )
        correct_T_e = np.array(
            [
                [
                    473.70624465,
                    4753.84505304,
                    878.73058195,
                    3246.09226993,
                    23493.93453882,
                    454.60020436,
                    153.86833128,
                ],
                [
                    508.12735443,
                    7143.30388221,
                    1493.84368878,
                    5852.89044489,
                    43771.15840575,
                    864.29304103,
                    296.53665642,
                ],
            ]
        )

        k1 = 3.67e8
        k2 = 5.496e9
        J1 = 1e7
        J2 = 5770
        J3 = 97030

        shafts, disks = [], []
        disks.append(Disk(0, J1))
        shafts.append(Shaft(0, 1, None, None, k=k1, I=0))
        disks.append(Disk(1, J2))
        shafts.append(Shaft(1, 2, None, None, k=k2, I=0))
        disks.append(Disk(2, J3))

        assembly = Assembly(shafts, disk_elements=disks)

        M, K = assembly.M(), assembly.K()
        assembly.xi = 0.02
        C = assembly.C_modal(M, K)
        A, B = assembly.state_matrix(C)
        lam, vec = LA.eig(A, B)

        U = SystemExcitation(assembly.dofs, omegas)
        U.add_harmonic(2, amplitudes)

        X, tanphi = assembly.ss_response(M, C, K, U.excitation_amplitudes(), omegas)

        T_v, T_e = assembly.vibratory_torque(M, C, K, U.excitation_amplitudes(), omegas)

        T_e = T_e.round(5)
        correct_T_e = correct_T_e.round(5)
        shaft_torques = T_e.tolist()
        correct_shaft_torques = correct_T_e.tolist()

        self.assertEqual(
            shaft_torques, correct_shaft_torques, "Vibratory shaft torque wrong"
        )

    # def test_frequency_domain_excitation_matrix_sweep(self):
    #     correct_U = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0, 10.0, 133.75, 275.5, 381.25, 505.0, 628.75, 752.5, 1000.0, 1000.0]])

    #     dofs = 4
    #     omegas = np.arange(1, 100, 10)
    #     U = SystemExcitation(dofs, omegas)
    #     U.add_sweep(3, 1000)

    #     correct = correct_U.tolist()
    #     excitation_values = U.excitation_amplitudes().tolist()
    #     print(correct)
    #     print(excitation_values)

    #     self.assertEqual(excitation_values, correct, "Excitation matrix not correct")

    # def test_time_domain_excitation_matrix(self):
    #     correct_U = np.array([])

    #     self.assertEqual(excitation_values, correct, "Excitation matrix not correct")

    # def test_time_domain_response(self):

    #     return


if __name__ == "__main__":
    unittest.main()
