import numpy as np
import opentorsion as ot

# MODEL PARAMETERS
# Shafts
k1 = 4.1e6
k2 = 3.3e5
k3 = 1.4e5
k4_gear = 1.5e8
k5 = 3.1e5
k6 = 7.4e5
k8 = 1.3e6
k9 = 8.7e5
k11 = 1.5e6

# Inertias
I1 = 41
I2 = 1.5
I3 = 0.6
I4 = 34
I5 = 0.8
I6 = 3.5
I7 = 1.2
I8 = 0.09
I9 = 1.2
I10 = 0.8
I11 = 0.4
I12 = 0.05

# Gear ratios
r5 = 1/8
r8 = 1/3
r11 = 1/20

def model():
    # Shaft elements
    shaft1 = ot.Shaft(0, 1, k=k1)
    shaft2 = ot.Shaft(1, 2, k=k2)
    shaft3 = ot.Shaft(2, 3, k=k3)
    shaft5 = ot.Shaft(4, 5, k=k5)
    shaft6 = ot.Shaft(5, 6, k=k6)
    shaft8 = ot.Shaft(7, 8, k=k8)
    shaft9 = ot.Shaft(8, 9, k=k9)
    shaft11 = ot.Shaft(10, 11, k=k11)
    shafts = [shaft1, shaft2, shaft3, shaft5, shaft6, shaft8, shaft9, shaft11]

    # Disk elements
    disk1 = ot.Disk(0, I=I1)
    disk2 = ot.Disk(1, I=I2)
    disk3 = ot.Disk(2, I=I3)
    disk4 = ot.Disk(3, I=I4)
    disk5 = ot.Disk(4, I=I5)
    disk6 = ot.Disk(5, I=I6)
    disk7 = ot.Disk(6, I=I7)
    disk8 = ot.Disk(7, I=I8)
    disk9 = ot.Disk(8, I=I9)
    disk10 = ot.Disk(9, I=I10)
    disk11 = ot.Disk(10, I=I11)
    disk12 = ot.Disk(11, I=I12)
    disks = [disk1, disk2, disk3, disk4, disk5, disk6, disk7, disk8, disk9, disk10, disk11, disk12]

    # Elastic gear elements
    gear4 = ot.ElasticGear(3, I=0, R=1)
    gear5 = ot.ElasticGear(5, I=0, R=r5, k=k4_gear, parent=gear4)
    gear8 = ot.ElasticGear(8, I=0, R=r8, k=k4_gear, parent=gear4)
    gear11 = ot.ElasticGear(10, I=0, R=r11, k=k4_gear, parent=gear4)
    gears = [gear4, gear5, gear8, gear11]

    # Return ot assembly
    return ot.Assembly(shafts, disks, elastic_gear_elements=gears)


if __name__ == "__main__":
    # Get assembly
    machine_train = model()

    # Perform undamped modal analysis and sort eigen frequencies
    wn, _, = machine_train.undamped_modal_analysis()
    wn = sorted(wn, key=np.abs)

    # Print natural frequencies of machine train
    print('Natural frequencies')
    for i, f in enumerate(np.real(wn[1:])):
        print(f"{i+1}: {np.sqrt(f)/(2*np.pi)*60}")