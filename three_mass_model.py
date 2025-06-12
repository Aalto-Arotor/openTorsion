import numpy as np

import opentorsion as ot

# Shafts parameters
k1, c1 = 50000, 0

# Disks parameters
I1, I2, I3 = 0.6, 0.1, 0.5

# Gear parameters
k_gear, c_gear = 80000, 20  # Modify gear mesh damping here
r1, r2 = (10, 35)


def ThreeMassModel():
    """Create a three-mass model with two elastic gears."""

    # Creating shaft elements
    # Syntax is: ot.Shaft(node 1, node 2, Length [mm], outer diameter [mm], stiffness [Nm/rad], damping[Nms/rad])  # noqa: E501
    shaft1 = ot.Shaft(0, 1, L=None, odl=None, k=k1, c=c1)
    shafts = [shaft1]

    # Creating disk elements
    # Syntax is: ot.Disk(node, Inertia [kgm^2], damping[Nms/rad])
    disk1 = ot.Disk(0, I=I1)
    disk2 = ot.Disk(1, I=I2)
    disk3 = ot.Disk(2, I=I3)
    disks = [disk1, disk2, disk3]

    # Creating elastic gear elements
    # Syntax is ot.ElasticGear(node, Inertia [kgm^2], radius/teeth, stiffness [Nm/rad], damping[Nms/rad], parent)  # noqa: E501
    # Note: stiffness and damping should only be added to child gears, not to parent gears.  # noqa: E501
    gear1 = ot.ElasticGear(1, 0, r1)
    gear2 = ot.ElasticGear(2, 0, r2, k=k_gear, c=c_gear, parent=gear1)
    gears = [gear1, gear2]

    # Creating the assembly
    # Syntax is ot.Assembly(shaft_elements, disk_elements, gear_elements, elastic_gear_elements)  # noqa: E501
    assembly = ot.Assembly(
        shaft_elements=shafts, disk_elements=disks, elastic_gear_elements=gears
    )
    return assembly


if __name__ == "__main__":
    # Create the three-mass model
    model = ThreeMassModel()

    # Perform modal analysis
    wn, vec, d = model.modal_analysis()

    # Print the natural frequencies
    print(f"{'Mode':<5} {'fn (cpm)':<15}")
    for i, f in enumerate(wn):
        # Convert to CPM
        freq = f / (2 * np.pi) * 60
        print(f"{i+1:<5} {freq:<15.5f}")
