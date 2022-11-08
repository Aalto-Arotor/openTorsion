import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.sparse import linalg as las
from scipy.signal import lti
from scipy.signal import lsim


class SystemExcitation:
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

    def __init__(self, dofs, omegas, shape=None, harmonic=True, transient=False):
        """
        Parameters
        ----------
        dofs : int
            Number of degrees of freedom of the system
        omegas : ndarray
            Array of excitation frequencies
        shape : tuple, optional
            Time domain excitation matrix shape
        harmonic : bool, optional
            If True, excitation is harmonic
        transient : bool
            If True, excitation is transient
        """

        if dofs is None:
            print("Zero DOF system")
            return None

        if omegas[0] < 10:
            print("WARNGING: RESPONSES UNDER 10 rad/s are unstable")

        self.dofs = dofs

        self.omegas = omegas

        if harmonic:
            self.U = np.zeros([self.dofs, len(omegas)])

        elif transient:
            self.U = np.zeros(shape)
            pass  # TODO

        else:
            self.U = None

        return

    def time_domain_excitation(self, nodes, data):
        """
        Excitation matrix for transient analysis.
        The function takes as input the node numbers where excitation data is inputted and the time domain excitation data. The excitation data must be listed in the same order as the listed nodes. The excitation data arrays must be of equal length.

        Parameters
        ----------
        nodes: list, int
            List of nodes where excitation data is inputted
        data: list, ndarray
            Excitation amplitudes as a list of (1 x n) shaped numpy arrays

        Returns
        -------
        ndarray
            Excitation matrix in time domain, containing excitation amplitudes for each time step
        """

        if len(nodes) < 1:
            raise ValueError("No nodes were defined for excitation input.")
        elif len(data) < 1:
            raise ValueError("No excitation data was given.")
        else:
            for data_array in data:
                if data_array.shape != data[0].shape:
                    raise ValueError(
                        "Excitation data contains arrays of different size."
                    )

            # TODO: self.dofs may be too large due to gears
            excitation_array = np.zeros((self.dofs, data[0].shape[0]))

            for i, node in enumerate(nodes):
                print(node)
                excitation_array[node] += data[i]
            self.U = excitation_array

        return excitation_array.T

    def excitation_frequencies(self, interval):
        """
        Excitation frequencies for steady-state and vibratory torque analysis

        Parameters
        ----------
        interval : list
            Lowest and highest excitation frequency values

        Returns
        -------
        ndarray
            Excitation frequencies evenly spaced over the specified interval
        """

        return np.linspace(interval[0], interval[-1])

    def add_sweep(self, node, amplitude):
        """
        Adds a sweep excitation with the given uniform amplitude to the given node

        Parameters
        ----------
        node : int
            Node number where excitation is inputted
        amplitude : ndarray
            Harmonic excitation amplitudes
        """
        amplitudes = np.ones(self.omegas.shape) * amplitude
        self.add_harmonic(node, amplitudes)

        return

    def add_harmonic(self, node, amplitudes):
        """
        Adds a harmonic excitaiton based on the omegas and amplitudes of the excitation this method should extensively check if all of the excitations have same size of omegas

        Parameters
        ----------
        node : int
            Node number where excitation is inputted
        amplitudes : ndarray
            Harmonic excitation amplitudes
        """

        if self.U is None:
            return "Error"  # TODO

        if len(amplitudes) != len(self.omegas):
            return "Error"  # TODO

        self.U[node] += amplitudes

        return

    def excitation_amplitudes(self):
        """
        Excitation amplitudes for steady-state and vibratory torque analysis

        Returns
        -------
        ndarray
            The excitation amplitude matrix
        """

        return self.U
