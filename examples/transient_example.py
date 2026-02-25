import opentorsion as ot
import numpy as np
import matplotlib.pyplot as plt

# MODEL PARAMETERS
I1, k1, c1 = 0.5, 5000, 10
I2, k2, c2 = 0.1, 500, 0.8
I3, k3, c3, d3 = 0.5, 1000, 5, 0.2
I4, d4 = 0.1, 0.2
I5, d5 = 1, 5
z1, z2 = 10, 80  # Number of teeth in gear elements


def drivetrain_assembly():
    """
    Creates an assembly of a drivetrain.

    Returns
    -------
    openTorsion Assembly class instance
        Drivetrain assembly

    """
    # Creating shaft elements
    # Syntax is: ot.Shaft(node 1, node 2, Length [mm], outer diameter [mm], stiffness [Nm/rad], damping)
    shaft1 = ot.Shaft(0, 1, L=None, odl=None, k=k1, c=c1)
    shaft2 = ot.Shaft(1, 2, L=None, odl=None, k=k2, c=c2)
    shaft3 = ot.Shaft(3, 4, L=None, odl=None, k=k3, c=c3)

    shafts = [shaft1, shaft2, shaft3]

    # Creating disk elements
    # Syntax is: ot.Disk(node, Inertia [kgm^2], damping)
    disk1 = ot.Disk(0, I=I1)
    disk2 = ot.Disk(1, I=I2)
    disk3 = ot.Disk(2, I=I3, c=d3)
    disk4 = ot.Disk(3, I=I4, c=d4)
    disk5 = ot.Disk(4, I=I5, c=d5)

    disks = [disk1, disk2, disk3, disk4, disk5]

    # Creating gear elements with a gear ratio of 80 / 10 = 8
    # Syntax is: ot.Gear(node, Inertia [kgm^2], radius/teeth, parent)
    gear1 = ot.Gear(2, 0, z1)
    gear2 = ot.Gear(3, 0, z2, parent=gear1)
    gears = [gear1, gear2]

    # Creating an assembly of the elements
    drivetrain = ot.Assembly(shafts, disks, gear_elements=gears)
    return drivetrain


if __name__ == "__main__":
    t = np.linspace(0, 1, 1000)

    drivetrain = drivetrain_assembly()
    excitation = ot.TransientExcitation(4, t)
    u = np.zeros_like(t)
    u += 1
    excitation.add_transient(0, u)

    (
        torques,
        speeds,
        t_arr,
    ) = drivetrain.dsim(excitation)

    plt.figure(figsize=(5, 4))

    for i, T in enumerate(torques):
        plt.plot(t_arr, T, label=f"Shaft {i+1}")

    plt.title("Step response")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.tight_layout()
    plt.show()
