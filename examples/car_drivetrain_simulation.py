import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import dlsim
import opentorsion as ot


def drivetrain_4mass():
    Jv = 3100
    Jc = 50
    Jg = 100
    Jm1 = 300
    Jp = 110
    Jm2 = 172

    ks = 23470
    k1 = 80000*6.4423
    k2 = 80000*9.994
    Cs = 150
    C1 = 150
    C2 = 150

    shafts, disks, gears = [], [], []
    disks.append(ot.Disk(0, I=Jv))
    shafts.append(ot.Shaft(0, 1, None, None, I=0, k=ks, c=Cs))
    gear_c = ot.Gear(1, I=Jv, R=1, parent=None)
    gears.append(gear_c)
    gear_g = ot.Gear(2, I=Jg, R=2, parent=gear_c)
    gears.append(gear_g)
    shafts.append(ot.Shaft(2, 3, None, None, I=0, k=k1, c=C1))
    disks.append(ot.Disk(3, I=Jm1))
    gear_p = ot.Gear(4, I=Jp, R=2, parent=gear_c)
    gears.append(gear_p)
    shafts.append(ot.Shaft(4, 5, None, None, I=0, k=k2, c=C2))
    disks.append(ot.Disk(5, I=Jm2))

    assembly = ot.Assembly(shaft_elements=shafts, disk_elements=disks, gear_elements=gears)
    wn, wd, r = assembly.modal_analysis()
    print(wn)

    return assembly


if __name__ == "__main__":
    dt = 1e-3
    sim_time = np.arange(0, 10+dt, dt)
    assembly = drivetrain_4mass()
    A, B = assembly.state_space()
    Ad, Bd = assembly.continuous_2_discrete(A, B, dt)
    C = np.eye(Ad.shape[1])
    D = np.zeros((C.shape[0], B.shape[1]))

    # apply excitations to both motors
    U = np.zeros((Bd.shape[1], len(sim_time)))
    U[-3,:] = np.ones(len(sim_time)) * 1100
    U[-1,:] = np.ones(len(sim_time)) * 650

    tout, yout, _ = dlsim((Ad, Bd, C, D, dt), U.T, t=sim_time)
    angle, speed = np.split(yout, 2, axis=1)

    k1 = 80000*6.4423
    plt.subplot(211)
    plt.plot(tout, k1*(angle[:,3]-angle[:,2]), 'b')
    plt.ylabel("Torque (Nm)")
    plt.subplot(212)
    plt.plot(tout, -speed[:,2], 'b')
    plt.ylabel("Speed (rad/s)")
    plt.xlabel("Time (s)")
    plt.show()
