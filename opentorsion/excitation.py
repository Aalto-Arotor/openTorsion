import numpy as np


class PeriodicExcitation:
    """
    This class is for building system excitation matrices.

    Attributes
    ----------
    dofs : int
        Number of degrees of freedom of the system
    omegas : ndarray
        Array of excitation frequencies
    U : ndarray
        Array of excitation amplitudes
    """

    def __init__(self, n_dofs, omegas):
        """
        Parameters
        ----------
        n_dofs : int
            Number of degrees of freedom of the system
        """

        if n_dofs <= 0 or len(omegas) <= 0:
            raise ValueError("Number of dofs and components must be positive")
        self.n_dofs = n_dofs
        self.n_components  = len(omegas)
        # Angular frequencies of the excitations (rad/s)
        self.omegas = omegas
        # Excitation matrix of shape (n_dofs, n_excitation_components)
        self.U = np.zeros([self.n_dofs, self.n_components], dtype=complex)


    def add_sines(self, node, angular_frequency, amplitude, phase):
        """
        Adds a sinusoidal excitation based on the omegas and amplitudes of the
        excitation this method should extensively check if all of the
        excitations have same size of omegas

        Parameters
        ----------
        node : int
            Node number where excitation is inputted
        angular_frequency: ndarray
            Angular_frequencies of the excitations [rad/s]
        amplitudes : ndarray
            Amplitudes corresponding to the frequencies [Nm]
        """
        angular_frequency, amplitude, phase = np.array(angular_frequency), np.array(amplitude), np.array(phase)
        if self.n_dofs < node or node < 0:
            raise ValueError(f"Input dof: {node} outside the number of dofs of the system: {self.n_dofs}")

        if len(angular_frequency) != len(amplitude) or len(angular_frequency) != len(phase):
            raise ValueError(f"Length of the angular frequency vector {len(angular_frequency)} differs from the length of the amplitude vector: {len(amplitude)} or phase vector: {len(phase)}")

        for i, (a, p) in enumerate(zip(amplitude, phase)):
            self.U[node, i] += a*np.exp(1j*p)

        return

    def excitation_matrix(self):
        """
        Excitation amplitudes for steady-state and vibratory torque analysis

        Returns
        -------
        ndarray
            The excitation amplitude matrix
        """

        return self.U


class TransientExcitation:
    """
    This class is for creating transient excitations used in time stepping simulations.
    """

    def __init__(self, n_dofs, times):
        """
        Initializes an excitation matrix used in time-stepping simulation.
        The excitation torque values are known for all time steps.

        Parameters
        ----------
        n_dof : int
            Number of degrees of freedom of the system
        times : ndarray
            Simulation time steps
        """
        self.n_dofs = n_dofs
        self.times = times
        self.U = np.zeros((n_dofs, len(self.times)))

    def add_transient(self, node, torques):
        """
        Parameters
        ----------
        node : int
            Node number where excitation is inputted
        torques : ndarray
            Excitation torque values corresponding to time steps used in simulation
        """
        if self.U is None:
            raise ValueError(f"Excitation matrix U: {self.U} has not been initialized using ot.TransientExcitation.init_U")

        if self.n_dofs < node or node < 0:
            raise ValueError(f"Input dof: {node} outside the number of dofs of the system: {self.n_dofs}")

        if len(self.times) != len(torques):
            raise ValueError(f"Length of the time vector {len(self.times)} differs from the length of the torque vector: {len(torques)}")

        self.U[node, :] += torques

        return

    def excitation_matrix(self):
        """
        Returns
        -------
        ndarray
            Excitation matrix used in time stepping simulation.
        """
        return self.U
