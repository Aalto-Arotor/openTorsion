Transient response
==================
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
