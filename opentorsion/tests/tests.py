import numpy as np
import scipy.linalg as LA
import unittest

# For imports when running tests locally
# import sys
# from pathlib import Path
# sys.path.append('..')

from opentorsion import (
    Shaft,
    Disk,
    Gear,
    Assembly,
    SystemExcitation,
    Plots,
)


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

        omegas_undamped, omegas_damped, damping_ratios = assembly.modal_analysis()

        self.assertEqual(
            (omegas_undamped / (2 * np.pi)).round(1).tolist(),
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

        omegas_undamped, _, _ = rotor.modal_analysis()
        freqs = omegas_undamped / (2 * np.pi)

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

        omegas_undamped, omegas_damped, damping_ratios = assembly.modal_analysis()
        freqs = omegas_undamped / (2 * np.pi)

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

        omegas_undamped, omegas_damped, damping_ratios = assembly.modal_analysis()
        freqs = omegas_undamped / (2 * np.pi)

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
            omegas_undamped, omegas_damped, b = assembly.modal_analysis()
            freq = omegas_undamped / (2 * np.pi)
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
        """
        Test calculating eigenmodes
        """
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
        assembly.K[0, 0] += k1
        assembly.K[5, 5] += k10

        lam, vec = assembly.eigenmodes()
        eigenmodes = np.abs(vec)

        self.assertEqual(eigenmodes.shape, (9, 9), "Eigenmode calculation not correct")

    def test_mass_matrix(self):
        correct_M = np.array([[10, 0], [0, 10]])
        shafts = []
        disks = []
        disks.append(Disk(0, I=10))
        shafts.append(Shaft(0, 1, None, None, k=428400, I=0))
        disks.append(Disk(1, I=10))

        assembly = Assembly(shafts, disk_elements=disks)
        M = assembly.M

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
        K = assembly.K

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
        K = assembly.K

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

        # D. Inman, in Encyclopedia of Vibration, 2001, Critical Damping in Lumped Parameter Models
        M = assembly.M
        M_inv = LA.inv(M)
        K = assembly.K
        M_K_M = LA.sqrtm(M_inv) @ K @ LA.sqrtm(M_inv)
        correct_C = 2 * 0.02 * (LA.sqrtm(M) @ LA.sqrtm(M_K_M) @ LA.sqrtm(M))
        correct_C = correct_C.round(4)

        C = assembly.C_modal(assembly.M, assembly.K)
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

    def test_state_space(self):
        correct_A = np.array(
            [[0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [-4.0e2, 4.0e2, 0, -1.5, 1, 0],
             [3891.891891891891, -4108.108108108107, -1297.2972972972968,
              9.729729729729728, -10.864864864864863, -0.8108108108108106],
             [0, -1.33333333e2, -8.00000000e2, 0, -8.33333333e-2, -1.5]]
        )
        correct_B = np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [1, -0, -0],
             [0, 9.72972973, 0],
             [0, 0, 1]]
        )
        shafts = [Shaft(0, 1, k=400, c=1),
                  Shaft(2, 3, k=800, c=0.5)]
       
        disks = [Disk(0, I=1, c=0.5),
                Disk(1, I=1e-1, c=0.1),
                Disk(2, I=1e-1, c=0.1),
                Disk(3, I=1, c=1)]

        gear1 = Gear(1, 0, 10)
        gear2 = Gear(2, 0, 60, parent=gear1)
        gears = [gear1, gear2]

        shaftline = Assembly(shafts, disks, gear_elements=gears)
        A, B = shaftline.state_space()

        
        correct_A = correct_A.round(6).tolist()
        correct_B = correct_B.round(6).tolist()
        A = A.round(6).tolist()
        B = B.round(6).tolist()

        self.assertEqual(A, correct_A, "State matrix not correct")
        self.assertEqual(B, correct_B, "Input matrix not correct")


    def test_continuous_2_discrete(self):
        correct_Ad = np.array(
            [[9.99998001e-01, 1.99920804e-06, -2.18307721e-10,
               9.99924354e-05, 5.06456694e-09, -1.41886052e-13],
             [1.94516925e-05, 9.99979468e-01, -6.48400342e-06, 
              4.92768675e-08, 9.99450127e-05, -4.26851591e-09],
             [-5.61974349e-11, -6.66573571e-07, 9.99996000e-01, 
              -1.41886052e-13, -4.38708580e-10, 9.99923671e-05],
             [-3.99772634e-02, 3.99761684e-02, -6.57013549e-06, 
              9.99848061e-01, 1.01936618e-04, -4.32450052e-09],
             [3.88955474e-01, -4.10564637e-01, -1.29654980e-01,
              9.91815739e-04, 9.98893628e-01, -8.75140974e-05],
             [-1.70734961e-06, -1.33305134e-02, -7.99933245e-02, 
              -4.32450052e-09, -8.99450446e-06, 9.99846012e-01]]
        )
        correct_Bd = np.array(
            [[4.99974838e-09, 1.63732899e-12, -3.51326010e-18],
             [1.63732899e-12, 4.86308687e-08, -1.40497100e-13],
             [-3.51326010e-18, -1.40497100e-13, 4.99974668e-09],
             [9.99924354e-05, 4.92768675e-08, -1.41886052e-13],
             [4.92768675e-08, 9.72437961e-04, -4.26851591e-09],
             [-1.41886052e-13, -4.26851591e-09, 9.99923671e-05]]
        )
        
        shafts = [Shaft(0, 1, k=400, c=1),
                  Shaft(2, 3, k=800, c=0.5)]
       
        disks = [Disk(0, I=1, c=0.5),
                Disk(1, I=1e-1, c=0.1),
                Disk(2, I=1e-1, c=0.1),
                Disk(3, I=1, c=1)]

        gear1 = Gear(1, 0, 10)
        gear2 = Gear(2, 0, 60, parent=gear1)
        gears = [gear1, gear2]

        shaftline = Assembly(shafts, disks, gear_elements=gears)
        A, B = shaftline.state_space()
        Ad, Bd = shaftline.continuous_2_discrete(A, B, ts=1e-4)
        
        correct_Ad = correct_Ad.round(8).tolist()
        correct_Bd = correct_Bd.round(8).tolist()
        Ad = Ad.round(8).tolist()
        Bd = Bd.round(8).tolist()

        self.assertEqual(Ad, correct_Ad, "State matrix not correct")
        self.assertEqual(Bd, correct_Bd, "Input matrix not correct")



if __name__ == "__main__":
    unittest.main()
