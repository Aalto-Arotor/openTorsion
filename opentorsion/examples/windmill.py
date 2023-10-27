import sys
sys.path.append("/Users/sampo/repos/openTorsion")

import opentorsion as ot


def windmill_assembly():
    """
    A shaft-line Finite Element Model of a windmill based on model presented
    in https://doi.org/10.1109/TIE.2010.2087301. A mechanical shaft-line FEM
    consists of shaft elements, disk elements/lumped masses (optional) and gear
    elements (optional). Elements are initiated by calling Shaft(), Disk() or
    Gear(). The elements take as inputs the element node number and element
    parameter values (refer to shaft_element.py, disk_element.py and
    gear_element.py documentation).
    """
    k1 = 3.67e8  # Turbine shaft stiffness
    k2 = 5.496e9  # Rotor stiffness
    J1 = 1e7  # Turbine inertia
    J2 = 5770  # Rotor inner inertia
    J3 = 97030  # Rotor outer inertia

    # Elements are initiated and added to corresponding list
    shafts, disks = [], []
    disks.append(ot.Disk(0, J1))
    shafts.append(ot.Shaft(0, 1, None, None, k=k1, I=0))
    disks.append(ot.Disk(1, J2))
    shafts.append(ot.Shaft(1, 2, None, None, k=k2, I=0))
    disks.append(ot.Disk(2, J3))

    # An assembly is initiated with the lists of powertrain elements
    assembly = ot.Assembly(shafts, disk_elements=disks)

    return assembly


if __name__ == "__main__":
    assembly = windmill_assembly()

    # Calculation of the eigenfrequencies of the powertrain
    omegas_undamped, omegas_damped, damping_ratios = assembly.modal_analysis()

    # Print eigenfrequencies.
    # The list contains each eigenfrequency twice: e.g. eigenfrequencies = [1st, 1st, 2nd, 2nd, 3rd, 3rd, ...]
    print("Eigenfrequencies: ", omegas_undamped.round(3))

    # Initiate plotting tools calling Plots(assembly)
    plot_tools = ot.Plots(assembly)

    # Plot eigenmodes, input number of eigenmodes
    plot_tools.plot_assembly()
    plot_tools.plot_eigenmodes(modes=3)
    plot_tools.plot_campbell(frequency_range_rpm=[0, 300], num_modes=2)
