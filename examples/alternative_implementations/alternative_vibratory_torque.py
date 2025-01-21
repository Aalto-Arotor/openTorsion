import opentorsion as ot
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA


def ss_coefficients(M, C, K, amplitudes, omegas):
    """The coefficient vectors a and b of the steady-state responses of the system"""
    # M, K = self.M(), self.K()
    # N = np.array([amplitudes[:,0]]).T.shape
    if type(amplitudes) is np.ndarray:
        Z = np.zeros(amplitudes.shape)
        a, b = np.zeros(amplitudes.shape), np.zeros(amplitudes.shape)
    else:
        Z = np.zeros(np.array([amplitudes]).shape)
        a, b = np.zeros(np.array([amplitudes]).shape), np.zeros(
            np.array([amplitudes]).shape
        )

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
    """Elemental vibratory torque"""

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

    print(T_vs.shape)
    print(T_vc.shape)
    # Vibratory torque between nodes
    T_vss = np.sum(T_vs, axis=1)
    T_vcc = np.sum(T_vc, axis=1)
    for j in range(T_e.shape[0]):
        T_e[j] = np.sqrt(
            np.power((T_vss[j + 1] - T_vss[j]), 2)
            + np.power((T_vcc[j + 1] - T_vcc[j]), 2)
        )

    return T_v, T_e

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

# Eigenfrequencies of the powertrain
w_undamped, w_damped, d_r = assembly.modal_analysis()
# Initiate OpenTorsion plotting tools
plot_tools = ot.Plots(assembly)

# Plot eigenmodes of the powertrain
# plot_tools.plot_eigenmodes (modes =3)

# Plot Campbell diagram of the powertrain
# plot_tools.plot_campbell()

# Creating a modal damping matrix
M, K = assembly.M, assembly.K # Mass and stiffness matrices
C = assembly.C_modal(M, K, xi=0.02) # Damping matrix

# The excitation depends on the rotational speed of the system.
# The response is calculated at each rotational speed.
# Total response is the sum of these responses.
def generator_torque(rpm):
    """Generator torque as a function of rotor rotating speed."""
    rated_T = 2.9e6
    if rpm < 4 : return 0
    if rpm < 15 : return (0.375 * rated_T/11) * rpm

    return (rated_T * 15 / rpm ) if rpm < 22 else 0


def get_windmill_excitation(rpm):
    """ 18 Cogging torque and torque ripple as harmonic excitation.
    (Table III from https://doi.org/10.1109/TIE .2010.2087301) """
    vs = np.array([4, 6, 8, 10, 12, 14, 16])
    rated_T, amplitudes = 2.9e6, generator_torque(rpm) * np.array([0.0018, 0.0179, 0.0024, 0.0034, 0.0117, 0.0018, 0.0011])
    amplitudes[4] += rated_T * 0.0176
    return 2*np.pi*vs*rpm, amplitudes, np.zeros_like(amplitudes)


def calculation_sopanen():
    n_steps = 5000
    res = np.zeros((2, n_steps))
    for i, rpm in enumerate(np.linspace(0.1, 25, n_steps)):
        omegas, amplitudes, phases = get_windmill_excitation(rpm)
        exc = np.zeros([3, len(amplitudes)])
        exc[2] = amplitudes
        T_v, T_e = vibratory_torque(M, C, K, exc, omegas)
        res[:, i] = T_e

    plt.plot(np.linspace(0.1, 25, n_steps), res[0]/1000, c="b")
    plt.plot(np.linspace(0.1, 25, n_steps), res[1]/1000, c="r")
    

if __name__ == "__main__":
    plt.figure(1)
    calculation_sopanen()
    plt.show()