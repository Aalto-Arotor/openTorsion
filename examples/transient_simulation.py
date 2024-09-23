""" Simulates the transient torque response in a lumped element model. The model is
simulated using discerete-time state-space form. The shaft torque is calculated by
multilpying the rotational stiffness of the shaft with the difference in angular
displacement of its disks. """
import matplotlib.pylab as plt
import numpy as np
import opentorsion as ot


# MODEL PARAMETERS
I1, k1, c1 = 0.5, 5000, 10
I2, k2, c2 = 0.1, 500, 0.8
I3, k3, c3, d3 = 0.5, 1000, 5, 0.2
I4, d4 = 0.1, 0.2
I5, d5 = 1, 5
z1, z2 = 10, 80   # Number of teeth in gear elements


class TransientExcitation:
    """
    This class is used for creating impulse and step excitations.

    Attributes
    ----------
    ts : float
        Time step size
    t_excite : float
        Time instance for applying the excitation
    magnitude : float
        Excitation magnitude
    """

    def __init__(self, ts=0, t_excite=0, magnitude=0):
        """
        Parameters
        ----------
        ts : float
            Time step size
        t_excite : float
            Time instance for applying the excitation
        magnitude : float
            Excitation magnitude
        """

        self.ts = ts
        self.excite = t_excite
        self.magnitude = magnitude
        self.impulse = 0

    def step_next(self, t):
        """
        Calculates the next step excitation.

        Parameters
        ----------
        t : float
            Current time step

        Returns
        -------
        float
            Torque magnitude of the next step excitation
        """

        if t >= self.excite:
            return self.magnitude
        return 0

    def impulse_next(self, t):
        """
        Calculates the next impulse excitation.

        Parameters
        ----------
        t : float
            Current time step

        Returns
        -------
        float
            Torque magnitude of the next excitation
        """

        width = 0.1
        if self.excite <= t <= self.excite + width:
            self.impulse += self.magnitude * (self.ts / width)
        elif self.excite + width <= t <= self.excite + 2 * width:
            self.impulse -= self.magnitude * (self.ts / width)

        return self.impulse


class PI():

  def __init__(self, Kp, Ki, dt, setpoint, limit):

    self.Kp = Kp
    self.Ki = Ki
    self.dt = dt
    self.setpoint = setpoint
    self.limit = limit
    self.integral_error = 0

  def next_step(self, x):

    error = self.setpoint - x
    self.integral_error += error*self.Ki*self.dt
    out = self.integral_error + error*self.Kp

    if self.integral_error > self.limit:
        self.integral_error = self.limit

    if out > self.limit:
        return self.limit

    return out


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


def shaft_torque(states, k_list, idx_list, ratio_list):
    """ Calculates the shaft torque between the given indices.

    Parameters
    ----------
    states : ndarray
        Matrix with the calculated states
    k_list : list
        List with stiffnesses
    idx_list : list
        List with indices
    ratio_list : list
        List with gear ratios

    Returns
    -------
    Calculated shaft torque
    """
    torques = []
    for i, k in enumerate(k_list):
        if ratio_list[i] >= 1:
            T = k * (np.abs(states[:, idx_list[i]]) / ratio_list[i] - np.abs(states[:, idx_list[i] + 1]))
        else:
            T = k * (np.abs(states[:, idx_list[i]]) - np.abs(states[:, idx_list[i] + 1]) * ratio_list[i])
        torques.append(T)

    return torques


def plot_rpm(t, rpm, target):
    """ Plots the angular velocity of a model.

    Parameters
    ----------
    t : array
        Time vector
    rpm : array
        Velocity vector
    target : float
        Target velocity
    """
    plt.figure(figsize=(6, 4))
    plt.plot(t, rpm, label='Model velocity')
    plt.axhline(target, color='g', linestyle='--', label='Target velocity')

    plt.title('Angular velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (RPM)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_torque(t, torques):
    """ Plots the shaft torques in a model.

    Parameters
    ----------
    t : array
        Time vector
    torques : array
        Torque vector
    """
    plt.figure(figsize=(7,4))
    for i, torque in enumerate(torques):
        plt.plot(t, torque, label=f'Shaft {i+1}')

    plt.title('Simulated torque response')
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def transient_simulation():
    """ Creating model assembly and calculating the discrete state and input matrices. """
    drivetrain = drivetrain_assembly()
    A, B = drivetrain.state_space()
    ts = 0.001  # Time step
    # Syntax is: self.continuous_2_discrete(state matrix, input matrix, time step)
    Ad, Bd = drivetrain.continuous_2_discrete(A, B, ts=0.001)

    """ The model can be controlled with a simple PI controller. A PI controller calculates the error, i.e. the
    diffrence between the desired model output and the real model output also referred to as a negative feedback
    loop. This error is passed to the proportional and intergal part of the controller. In this part, the signal
    is turned into a more suitable input to the model. We use the controller PI defined above. """
    # Parameters
    Kp = 3
    Ki = 3
    target = 200 # RPM
    limit = 20   # Torque (Nm)
    # Syntax is: ot.PI(Proportional gain, Integral gain, Time step [s], Target velocity [RPM], Limit [Nm])
    controller = PI(Kp, Ki, ts, target, limit)

    """ Transient excitations for the model can be created using the TransientExcitations class. The instance
    can currently be used to simulate step and impulse excitations. The step and impulse excitations are obtained
    by calling step_next and impulse_next. They both take the current time as parameter and returns the excitation
    torque. """
    # Parameters
    t_excite = 3    # Time (s)
    magnitude = 30  # Torque (Nm)
    # Syntax is: TransientExcitation(Time step [s], Time for applying excitation [s], Magnitude [Nm])
    excitations = TransientExcitation(ts, t_excite, magnitude)

    """ Calculating the states of the model for a step excitation. The next state is calculated by adding the product
    of the discrete state matrix and the state vector with the product of the discrete input matrix and the input vector.
    The input torque at the first index is obtained from the controller and the excitation torque at the last index is
    obtained from the excitation instance. """
    # Defining necessary variables
    t_end = 6                                             # Simulation time
    x0 = np.zeros(2 * drivetrain.M.shape[0])              # Initial state
    u0 = np.zeros(drivetrain.M.shape[0])                  # Input vector
    iterations = np.linspace(0, t_end, int(t_end/ts))     # Iterations based on simulation time and time step
    rpm = 60 / (2 * np.pi)                                # Conversion from rad/s to RPM
    states_step = []
    rpms = []

    # Calculating all the states
    for i in iterations:
        u0[0] = controller.next_step(x0[drivetrain.M.shape[0]] * rpm)
        u0[-1] = excitations.step_next(i)
        x0 = Ad @ x0 + Bd @ u0
        states_step.append(x0)
        rpms.append(x0[drivetrain.M.shape[0]] * rpm)
    states_step = np.array(states_step)
    rpms = np.array(rpms)

    """ Calculating torque responses. """
    i = z2 / z1   # Gear ratio
    # Syntax is: shaft_torque(states matrix, stiffnesses list, indices list, ratios list)
    torques_step = shaft_torque(states_step, [k1, k2, k3], [0, 1, 2], [1, 1, i])

    """ Plotting model velocity and shaft torques. """
    # Syntax is: plot_rmp(time vector, velocity vector, target velocity)
    plot_rpm(iterations, rpms, target)
    # Syntax is: plot_torque(time vector, torque vector)
    plot_torque(iterations, torques_step)


if __name__ == "__main__":

    """ Simulate the transient torque response in a model. """
    transient_simulation()
