"""
This script includes an OpenTorsion implementation of Example 8.1 presented in [1],
which describes a vibratory torque analysis for a marine steam turbine propulsion system.

[1] Gunter, E.J. and Chen, W.J., 2001. Introduction to dynamics of rotor-bearing systems, pp. 388-392.
"""

import opentorsion as ot
import numpy as np
import matplotlib.pyplot as plt


def get_dyrobes_response():
    """Response data digitized from Figure 8.10-5 [1], p. 392"""
    file_path = "./dyrobes_data/dyrobes_response.csv"
    curve = np.loadtxt(file_path, delimiter=";")
    return curve


def propeller_torque(rpm_p):
    """Propeller torque"""
    rpm_o = 85
    T_o = 22.24e6 * 0.11298
    T_p = T_o * (rpm_p/rpm_o)**2
    return T_p


def get_propeller_excitation(rpm):
    """Propeller excitation"""
    amplitudes = np.array([0.1]) * propeller_torque(rpm)
    return np.array([2*np.pi*5*rpm/60]), amplitudes, np.zeros_like(amplitudes)


def branched_drivetrain():
    # values for unit conversion
    inertia_to_si = 0.11298
    stiffness_to_si = 0.11298
    damping_to_si = 0.11298

    # Numerical values
    I_propeller = 2.454E06*inertia_to_si
    I_gear = 0.826E06*inertia_to_si
    I_high_pressure_intermediate_gear = 2.723E04*inertia_to_si
    I_high_pressure_turbine = 2.612E02*inertia_to_si
    I_low_pressure_intermediate_gear = 1.283E04*inertia_to_si
    I_low_pressure_turbine = 1.509E04*inertia_to_si

    k_propeller_shaft = 826E06*stiffness_to_si
    k_high_pressure_intermediate_shaft = 24.17E06*stiffness_to_si
    k_high_pressure_turbine_shaft = 14.26E06*stiffness_to_si
    k_low_pressure_intermediate_shaft = 203.94E06*stiffness_to_si
    k_low_pressure_turbine_shaft = 30.51E06*stiffness_to_si

    d_propeller = 3.86E06*damping_to_si
    d_high_pressure_turbine = 42.738*damping_to_si
    d_low_pressure_turbine = 108.77*damping_to_si

    # Elements are initiated and added to corresponding list
    shafts, disks = [], []

    # Propeller branch
    disks.append(ot.Disk (0, I_propeller, c=d_propeller))
    shafts.append(ot.Shaft(0, 1, k=k_propeller_shaft, I=0))
    gear_1 = ot.Gear(1, I=I_gear, R=9.4094)

    # low pressure turbine branch
    gear_2 = ot.Gear(2, I=0, R=1, parent=gear_1)
    shafts.append(ot.Shaft(2, 3, k=k_low_pressure_intermediate_shaft, I=0))
    gear_3 = ot.Gear(3, I=I_low_pressure_intermediate_gear, R=40.0424)
    gear_4 = ot.Gear(4, I=0, R=9.4094, parent=gear_3)
    shafts.append(ot.Shaft(4, 5, k=k_low_pressure_turbine_shaft, I=0))
    disks.append(ot.Disk (5, I_low_pressure_turbine, c=d_low_pressure_turbine))

    # High pressure turbine branch
    gear_5 = ot.Gear(6, I=0, R=1, parent=gear_1)
    shafts.append(ot.Shaft(6, 7, k=k_high_pressure_intermediate_shaft, I=0))
    gear_6 = ot.Gear(7, I=I_high_pressure_intermediate_gear, R=78.2365)
    gear_7 = ot.Gear(8, I=0, R=9.4094, parent=gear_6)
    shafts.append(ot.Shaft(8, 9, k=k_high_pressure_turbine_shaft, I=0))
    disks.append(ot.Disk (9, I_high_pressure_turbine, c=d_high_pressure_turbine))

    gears = [gear_1, gear_2, gear_3, gear_4, gear_5, gear_6, gear_7]
    # An assembly is initiated with the lists of elements
    assembly = ot.Assembly(shaft_elements=shafts, disk_elements = disks, gear_elements=gears)

    # Eigenfrequencies of the powertrain
    w_undamped, w_damped, d_r = assembly.modal_analysis()

    print("Eigenfrequencies should be: 177.7, 220.2 and 1282.6 (Hz)")
    freqs = np.round(np.sort(np.abs(w_damped))/(2*np.pi)*60, 1)[::2]
    print(f"Eigenfrequencies are: {freqs[1]}, {freqs[2]} and {freqs[3]} (Hz)")

    return assembly


def run_forced_response_analysis():
    # Create assembly
    assembly = branched_drivetrain()

    # Create a modal damping matrix
    M, K = assembly.M, assembly.K # Mass and stiffness matrices
    C = assembly.C_modal(M, K, xi=0.00398) + assembly.C # Damping matrix
    dofs = M.shape[0] # number of degrees of freedom

    operating_speed_range = (0.1, 100) # rpm
    n_steps = 5000 # number of calculation steps
    VT_sum_result = np.zeros((5, n_steps))

    # Calculate forced response in operating speed range (0.1-100 rpm)
    for i, rpm in enumerate(np.linspace(*operating_speed_range, n_steps)):
        # Defining the excitation
        omegas, amplitudes, phases = get_propeller_excitation(rpm)
        excitation = ot.PeriodicExcitation(dofs, omegas)
        excitation.add_sines(0, omegas, amplitudes, phases)

        # Compute response for this rpm
        _, T_vib_sum = assembly.vibratory_torque(excitation, C=C)
        VT_sum_result[:, i] = np.ravel(T_vib_sum)


    dyrobes_response = get_dyrobes_response()
    lbin_to_Nm = 0.11298
    plt.plot(np.linspace(0.1, 100, n_steps), VT_sum_result[0]/1000, c='C0', label="OpenTorsion")
    plt.scatter(dyrobes_response[:, 0], dyrobes_response[:, 1]*lbin_to_Nm/1000, c='C1', marker='x', label="Digitized data")
    plt.legend()
    plt.xlabel("Rotational speed (rpm)")
    plt.ylabel("Vibratory torque (kNm)")
    plt.show()


if __name__ == "__main__":
    run_forced_response_analysis()
