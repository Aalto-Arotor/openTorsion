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
    dyrobes_response = np.array([[14.482758620689651, 69148.93617021292], [19.04640272456364, 132978.72340425476], [21.021711366538952, 180851.06382978708], [24.563644103873983, 324468.0851063831], [26.947637292464876, 468085.10638297815], [28.10557684120902, 595744.6808510637], [29.127288207747974, 739361.7021276588], [30.080885483184325, 898936.1702127652], [30.762026394210295, 1053191.4893617015], [31.170710940825877, 1218085.106382979], [31.64750957854406, 1367021.2765957443], [32.192422307364836, 1574468.0851063826], [32.8054491272882, 1856382.978723404], [33.07790549169859, 2095744.6808510632], [33.282247765006375, 2345744.680851063], [33.62281822051936, 2585106.382978723], [33.89527458492975, 2803191.4893617015], [34.09961685823754, 3042553.191489361], [34.37207322264793, 3319148.9361702125], [34.712643678160916, 3585106.382978723], [34.9851000425713, 3824468.0851063826], [35.46189868028948, 4148936.170212765], [35.1213282247765, 3984042.5531914886], [36.14303959131545, 3984042.5531914886], [36.27926777352064, 3776595.7446808508], [36.48361004682843, 3585106.382978723], [36.82418050234141, 3319148.9361702125], [37.0966368667518, 3122340.4255319145], [37.300979140059596, 2877659.5744680846], [37.70966368667518, 2563829.787234042], [38.186462324393354, 2297872.3404255314], [38.45891868880374, 2117021.2765957443], [39.00383141762451, 1851063.8297872338], [39.68497232865048, 1643617.0212765955], [40.63856960408685, 1436170.2127659568], [41.796509152830986, 1218085.106382979], [43.70370370370369, 1021276.5957446806], [45.20221370796083, 893617.0212765951], [46.90506598552576, 797872.3404255314], [48.335461898680286, 744680.8510638298], [50.0383141762452, 696808.5106382975], [51.87739463601532, 659574.4680851055], [53.716475095785434, 622340.4255319145], [56.6453810131971, 569148.9361702129], [58.893146019582794, 553191.4893617015], [61.209025117071086, 521276.5957446806], [64.00170285227756, 500000], [67.88420604512558, 473404.2553191483], [72.58407833120475, 454787.23404255323], [82.32439335887611, 425531.9148936169], [87.50106428267347, 414893.6170212766], [91.45168156662407, 409574.46808510646], [97.10515112813962, 409574.46808510646], [75.7173265219242, 457446.80851063784], [78.78246062154108, 425531.9148936169]])
    return dyrobes_response


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

    plot_tools = ot.Plots(assembly)
    plot_tools.plot_assembly()

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
