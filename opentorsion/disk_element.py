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
        Viscous damping coefficient of the disk [Nms/rad]
    k: float
        Equivalent stiffness coefficient of the disk [Nm/rad]
    """

    def __init__(self, node, I, c=0, k=0):
        self.node = node
        self.I = I
        self.k = k
        self.c = c

    def __repr__(self):
        """
        String representation of a disk element
        """

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

        return np.array([self.c], dtype=np.float64) * self.c

    def K(self):
        """Stiffness matrix of a disk element

        Returns
        -------
        K: ndarray
            Stiffness matrix of the disk element
        """
        k = self.k
        K = np.array([k], dtype=np.float64)

        return K

    def __str__(self):
        return 'Disk, pos: ' + str(self.node) + ' I:'  + str(self.I) + ' c: ' + str(self.c) + ' k: ' + str(self.k)
