Tutorial - Analyses
===================
Torsional vibration analyses available in OpenTorsion include modal analysis, Campbell diagram, forced response and time-stepping simulation for transient response. An OpenTorsion assembly is reqired to run analyses.

Modal analysis and Campbell diagram
-----------------------------------

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
   :width: 100%
   :align: center
   :alt: Eigenmode plot.
   :target: .

.. figure:: figs/campbell_example.svg
   :width: 100%
   :align: center
   :alt: Campbell diagram.
   :target: .

Forced response
---------------
Forced response example. Calculating forced response requires an assembly and excitation.

.. code:: bash

    import numpy as np
    import matplotlib.pyplot as plt
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

    harmonics = np.array([1, 2, 3, 4])  # excitation harmonics, multiples of rotational frequency
    amplitudes = [200, 50, 5, 2]  # excitation amplitudes, corresponding to harmonics
    # defining an excitation matrix: a rotational speed dependent excitation is applied to node 0
    # rows correspond to assembly nodes, columns correspond to excitation frequencies
    excitation = np.zeros([assembly.dofs, len(amplitudes)])
    excitation[0] = amplitudes

    w = 3600*(2*np.pi)/60  # base rotational frequency
    t = np.linspace(0, (2*np.pi)/w, 200)  # time, used for plotting
    omegas = w*harmonics  # array of excitation frequencies

    # steady-state response
    q_res, w_res = assembly.ss_response(excitation, omegas)

    # angle difference between two consecutive nodes
    q_difference = (q_res.T[:, 1:] - q_res.T[:, :-1]).T

    # initiate 4 subplots for the 4 shafts
    fig, axes = plt.subplots(4, 1, figsize=(8, 8))

    # Shaft stiffness values are used to calculate the torque from the angle differences
    shaft_stiffness = [25e+6, 25e+6, 25e+6, 25e+6]

    # Loop over the 4 shafts to plot the response for each of them
    for n in range(4):
        shaft_response = q_difference[n]
        sum_wave = np.zeros_like(t)
        # Loop over the harmonic components and cumulate the result
        for i, (response_component, harmonic) in enumerate(zip(shaft_response, harmonics)):
            # Get the waveform of each response component
            this_wave = np.real(response_component*np.exp(1.0j*harmonic*w*t))

            # Cumulate the sum wave
            sum_wave += this_wave

            # Plot the individual component in newton meters
            axes[n].plot(t, this_wave*shaft_stiffness[n], '--', c='gray')

        # Plot the sum excitation signal in newton meters
        axes[n].plot(t, sum_wave*shaft_stiffness[n], c='red')

        axes[n].set_title(f'Torque at shaft {n+1}')
        axes[n].set_xlabel('Time (s)')
        axes[n].set_ylabel('Torque (Nm)')
    plt.tight_layout()
    plt.show()

.. figure:: figs/forced_response.svg
   :width: 100%
   :align: center
   :alt: Torque at shafts 1, 2, 3 and 4.
   :target: .

Transient response
------------------
Time-stepping simulation example. Calculating transient response requires an assembly and excitation.

.. code:: bash

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import dlsim

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

    # Defining and impulse excitation
    dt = 0.002
    t = np.arange(0, 0.500, dt)
    impulse = np.zeros((len(t), assembly.dofs))
    ramp = np.arange(0, 2000, int(2000/8))
    impulse[22:30,0] = ramp
    impulse[30,0] = 2000
    impulse[31:39,0] = ramp[::-1]

    plt.plot(t, impulse[:,0], c='black')
    plt.title("Impulse excitation")
    plt.xlabel("Times (s)")
    plt.ylabel("Torque (Nm)")
    plt.show()

    # Discrete-time LTI state-space model
    A, B = assembly.state_space()
    Ad, Bd = assembly.continuous_2_discrete(A, B, dt)
    C = np.eye(A.shape[1])
    D = np.zeros((C.shape[0], B.shape[1]))
    sys = (Ad, Bd, C, D, dt)

    # scipy.signal.dlsim used for time-step simulation
    tout, yout, xout = dlsim(sys, impulse, t)
    # simulation result is nodal rotational angles and speeds
    angles, speeds = np.split(yout, 2, axis=1)

    # initiate 4 subplots for the 4 shafts
    fig, axes = plt.subplots(4, 1, figsize=(8, 8))

    # Shaft stiffness values are used to calculate the torque from the angle differences
    shaft_stiffness = [25e+6, 25e+6, 25e+6, 25e+6]

    # Loop over the 4 shafts to plot the response for each of them
    for n in range(4):
        # Plot the shaft response in newton meters
        axes[n].plot(t, shaft_stiffness[n]*(angles[:,(n+1)]-angles[:,n]), c='red')

        axes[n].set_title(f'Torque at shaft {n+1}')
        axes[n].set_xlabel('Time (s)')
        axes[n].set_ylabel('Torque (Nm)')
    plt.tight_layout()
    plt.show()

.. figure:: figs/impulse.svg
   :width: 100%
   :align: center
   :alt: Impulse excitation.
   :target: .

.. figure:: figs/transient_response.svg
   :width: 100%
   :align: center
   :alt: Torque at shafts 1, 2, 3 and 4.
   :target: .
