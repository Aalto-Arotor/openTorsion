import numpy as np


class Disk:
    """A disk object

    Arguments:
    ----------
    node: int
        Nodal position of the disk in the global coordinates
    I: float
        Moment of inertia of the disk [kgm^2]
    c: float
        Damping coefficient of the disk [Nms/rad]
    """

    def __init__(self, node, I, c=0):
        self.node = node
        self.I = I
        self.damping = c

    def __repr__(self):
        """String representation of a disk element"""
        return (str(self.n), str(self.m), str(self.I), str(self.I))

    def M(self):
        """Mass Matrix of 1 DOF disk element

        Returns
        -------
        M: ndarray
            Mass matrix of the disk element
        """
        I = self.I
        M = np.array([I], dtype=np.float64)

        return M

    def C(self):
        """Damping matrix of a disk element

        Returns
        -------
        C: ndarray
            Damping matrix of disk element
        """

        return np.array([self.damping], dtype=np.float64) * self.damping

    def K(self):
        """Stiffness matrix of a disk element

        Returns
        -------
        K: ndarray
            Stiffness matrix of the disk element
        """

        return np.zeros((1), dtype=np.float64)
