Tutorial - Modal analysis and Campbell diagram
===================================

.. code:: bash

    import opentorsion as ot

    # An example assembly
    # Creating 4 shaft elements using stiffness values
    # Syntax: ot.Shaft(node 1, node 2, Length [mm], outer diameter [mm], stiffness [Nm/rad])
    shaft1 = ot.Shaft(0, 1, L=None, odl=None, k=25e+6)
    shaft2 = ot.Shaft(1, 2, L=None, odl=None, k=25e+6)
    shaft3 = ot.Shaft(2, 3, L=None, odl=None, k=25e+6)
    shaft4 = ot.Shaft(3, 4, L=None, odl=None, k=25e+6)

    # Creating 5 disk elements
    # Syntax: ot.Disk(node, inertia [kgm^2])
    disk1 = ot.Disk(0, I=100)
    disk2 = ot.Disk(1, I=10)
    disk3 = ot.Disk(2, I=50)
    disk4 = ot.Disk(3, I=10)
    disk5 = ot.Disk(4, I=80)

    # Adding the elements to lists corresponding to an element type
    shafts = [shaft1, shaft2, shaft3, shaft4]
    disks = [disk1, disk2, disk3, disk4, disk5]

    # Syntax: ot.Assembly(shaft_elements, disk_elements)
    assembly = ot.Assembly(shaft_elements=shafts, disk_elements=disks)

    # initialize OpenTorsion plotting tools
    plot_tools = ot.Plots(assembly)

    # Calculation of the system's eigenfrequencies
    omegas_undamped, omegas_damped, damping_ratios = assembly.modal_analysis()
    # Print eigenfrequencies.
    # The list contains each eigenfrequency twice, i.e., eigenfrequencies = [1st, 1st, 2nd, 2nd, 3rd, 3rd, ...]
    print("Eigenfrequencies [rad/s]: ", omegas_undamped.round(3))

    # Plot eigenmodes, takes as parameter the number of eigenmodes to be plotted
    plot_tools.plot_eigenmodes(modes=3)

    # Campbell plot takes as parameter
    # - the rotational frequency range [rpm]
    # - number of eigenfrequencies to be plotted
    # - number of harmonics to be plotted
    # - operating speed range
    plot_tools.plot_campbell(
        frequency_range_rpm=[0, 5000],
        num_modes=3,
        harmonics=[1, 2, 3],
        operating_speeds_rpm=[3600]
    )

.. figure:: figs/mode_example.svg
   :width: 80%
   :align: center
   :alt: Eigenmode plot.
   :target: .

.. figure:: figs/campbell_example.svg
   :width: 80%
   :align: center
   :alt: Campbell diagram.
   :target: .
