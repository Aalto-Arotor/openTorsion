""" 
This script reproduces the results of "Dynamic Torque Analysis of a Wind Turbine Drive
Train Including a Direct-Driven Permanent-Magnet Generator" (https://doi.org/10.1109/TIE.2010.2087301)
Case A described in the article is considered and the implementation follows the derivations in the article
instead of the standard derivations used in OpenTorsion.
"""


import opentorsion as ot
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA


def ss_coefficients(M, C, K, amplitudes, omegas):
    """
    The coefficient vectors a and b of the steady-state responses of the system.
    This is equation (15) in the article
    """
    amplitudes = np.array(amplitudes)
    Z = np.zeros_like(amplitudes)
    a = np.zeros_like(amplitudes)
    b = np.zeros_like(amplitudes)
    U = np.vstack([amplitudes, Z])

    # setting excitation to zero at very low frequencies
    for i, omega in enumerate(omegas):
        AA = np.vstack(
            [
                np.hstack([K - (omega ** 2 * M), -omega * C]),
                np.hstack([omega * C, K - (omega * omega * M)]),
            ]
        )
        ab = LA.inv(AA) @ np.array([U[:, i]]).T
        a_i, b_i = np.split(ab, 2)

        a[:, i] = a_i.T
        b[:, i] = b_i.T

    return a, b


def vibratory_torque(M, C, K, amplitudes, omegas):
    """
    Elemental vibratory torque
    Equations (20) - (22) in the article
    """

    aa, bb = ss_coefficients(M, C, K, amplitudes, omegas)

    T_v = np.zeros(amplitudes.shape)
    T_e = np.zeros(M.shape[0]-1)

    T_vs, T_vc = np.zeros(T_v.shape), np.zeros(T_v.shape)
    for i, omega in enumerate(omegas):
        a, b = aa[:, i], bb[:, i]
        T_vs[:, i] += K @ a - (omega * C) @ b
        T_vc[:, i] += K @ b + (omega * C) @ a

    # Vibratory torque at nodes
    for i in range(T_vs.shape[1]):
        T_v[:, i] = np.array([np.sqrt(T_vs[:, i] ** 2 + T_vc[:, i] ** 2)])

    # Vibratory torque at elements (first sum the nodal values)
    T_vss = np.sum(T_vs, axis=1)
    T_vcc = np.sum(T_vc, axis=1)
    for j in range(T_e.shape[0]):
        T_e[j] = np.sqrt(
            np.power((T_vss[j + 1] - T_vss[j]), 2)
            + np.power((T_vcc[j + 1] - T_vcc[j]), 2)
        )

    return T_v, T_e


def get_assembly():
    # Numerical values
    k1 , k2 = 3.7e8, 5.5e9
    I1 , I2 , I3 = 1.0e7, 5.8e3, 97e3

    # Elements are initiated and added to corresponding list
    shafts, disks = [], []
    shafts.append(ot.Shaft (0, 1, k=k1 , I=0))
    shafts.append(ot.Shaft (1, 2, k=k2 , I=0))

    disks.append(ot.Disk (0, I1))
    disks.append(ot.Disk (1, I2))
    disks.append(ot.Disk (2, I3))

    # An assembly is initiated with the lists of elements
    assembly = ot. Assembly (shaft_elements=shafts, disk_elements = disks)

    # Creating a modal damping matrix
    M, K = assembly.M, assembly.K # Mass and stiffness matrices
    C = assembly.C_modal(M, K, xi=0.02) # Damping matrix

    return M, C, K


def generator_torque(rpm):
    """Generator torque as a function of rotor rotating speed."""
    rated_T = 2.9e6
    if rpm < 4 : return 0
    if rpm < 15 : return (0.375 * rated_T/11) * rpm

    return (rated_T * 15 / rpm ) if rpm < 22 else 0


def get_windmill_excitation(rpm):
    """
    Cogging torque and torque ripple as harmonic excitation. (Table III in the article)
    """
    vs = np.array([4, 6, 8, 10, 12, 14, 16])
    rated_T, amplitudes = 2.9e6, generator_torque(rpm) * np.array([0.0018, 0.0179, 0.0024, 0.0034, 0.0117, 0.0018, 0.0011])
    amplitudes[4] += rated_T * 0.0176
    return 2*np.pi*vs*rpm, amplitudes, np.zeros_like(amplitudes)


def calculation_sopanen(n_steps=1000):
    res = np.zeros((2, n_steps))
    M, C, K = get_assembly()
    for i, rpm in enumerate(np.linspace(0.1, 25, n_steps)):
        omegas, amplitudes, phases = get_windmill_excitation(rpm)
        exc = np.zeros([3, len(amplitudes)])
        exc[2] = amplitudes
        T_v, T_e = vibratory_torque(M, C, K, exc, omegas)
        res[:, i] = T_e

    return res
    
def compare_to_data_extracted_from_article():
    # Data extracted from the plots in original publication
    # "Dynamic Torque Analysis of a Wind Turbine Drive Train Including a Direct-Driven Permanent-Magnet Generator"
    element_1 = np.array([[0.1338, 53.476], [0.3070, 65.613], [0.4173, 79.772], [0.4960, 98.988], [0.5433, 120.22], [0.5748, 144.500], [0.6062, 163.716], [0.6377, 183.944], [0.6535, 204.171], [0.6771, 222.376], [0.7716, 960.176], [0.8582, 221.365], [0.8661, 202.149], [0.8740, 181.921], [0.8976, 160.682], [0.9055, 137.420], [0.9291, 103.034], [0.9685, 68.647], [1.1102, 46.396], [1.2992, 31.226], [1.5826, 21.112], [1.8976, 16.055], [2.2440, 12.010], [2.6062, 8.975], [3.1102, 8.975], [3.6929, 6.953], [4.2598, 8.975], [4.9999, 7.964], [5.6141, 8.975], [6.3070, 6.953], [6.8267, 6.953], [7.3779, 13.021], [8.1653, 13.021], [8.8582, 12.010], [9.4409, 13.021], [10.1338, 14.032], [10.9842, 14.032], [11.7559, 19.089], [12.4645, 25.158], [12.9212, 36.283], [13.2519, 54.487], [13.3937, 69.658], [13.5039, 80.783], [13.6456, 91.908], [13.8661, 76.738], [13.9921, 65.613], [14.1338, 51.453], [14.2440, 44.374], [14.4803, 36.283], [14.6692, 32.237], [15.2992, 28.192], [15.5196, 24.146], [15.9133, 26.169], [16.3858, 30.214], [16.6692, 24.146], [17.0944, 19.089], [17.4566, 17.067], [17.9606, 16.055], [18.4488, 13.021], [19.0157, 14.032], [19.5354, 14.032], [20.0393, 14.032], [20.5433, 17.067], [20.9527, 14.032], [21.2834, 12.010], [21.7244, 10.998], [22.2125, 5.9418], [22.9055, 6.9532], [23.7244, 4.9304], [24.4015, 3.9190], [24.8267, 3.9190]])
    element_2 = np.array([[0.149, 53.476], [0.322, 66.624], [0.417, 81.795], [0.480, 101.011], [0.511, 118.204], [0.543, 137.420], [0.574, 157.648], [0.606, 176.864], [0.629, 196.080], [0.653, 214.285], [0.669, 231.479], [0.763, 963.716], [0.850, 231.479], [0.866, 211.251], [0.889, 181.921], [0.889, 145.512], [0.937, 107.079], [0.984, 66.624], [1.157, 45.385], [1.267, 18.078], [1.047, 39.317], [1.645, 2.907], [2.007, 0.884], [2.700, 0.884], [3.519, 0.884], [4.291, 0.884], [4.984, 1.896], [7.440, 14.032], [8.228, 14.032], [9.000, 19.089], [9.803, 20.101], [10.464, 22.123], [11.236, 26.169], [11.881, 37.294], [12.574, 45.385], [12.905, 59.544], [13.283, 100.000], [13.362, 119.216], [13.472, 145.512], [13.551, 165.739], [13.677, 176.864], [13.771, 157.648], [13.897, 139.443], [13.960, 118.204], [14.086, 97.977], [14.259, 78.761], [14.464, 65.613], [14.779, 49.431], [15.078, 42.351], [15.204, 56.510], [15.503, 51.453], [16.039, 51.453], [16.448, 51.453], [16.811, 43.362], [17.204, 36.283], [17.724, 31.226], [18.370, 25.158], [19.047, 24.146], [19.850, 24.146], [20.259, 27.180], [20.716, 28.192], [21.614, 21.112], [22.228, 5.941], [23.173, 3.919], [24.259, 1.896], [24.748, 2.907]])

    n_steps = 1000
    res = calculation_sopanen(n_steps=n_steps)

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(np.linspace(0.1, 25, n_steps), res[0]/1000, c="b", label="Calculated")
    ax1.scatter(element_1[:, 0], element_1[:, 1], c="r", label="Sopanen & al.")
    ax1.legend()

    ax2.plot(np.linspace(0.1, 25, n_steps), res[1]/1000, c="b", label="Calculated")
    ax2.scatter(element_2[:, 0], element_2[:, 1], c="r", label="Sopanen & al.")
    ax2.legend()
    ax1.set_xlabel("Rotating speed (rpm)")
    ax2.set_xlabel("Rotating speed (rpm)")

    ax1.set_ylabel("Vibratory torque (kNm)")
    ax2.set_ylabel("Vibratory torque (kNm)")
    plt.show()


if __name__ == "__main__":
    compare_to_data_extracted_from_article()

