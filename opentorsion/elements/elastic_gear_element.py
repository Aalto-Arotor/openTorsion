# flake8: noqa
import numpy as np


class ElasticGear:
    """An elastic gear object
    Gears consist of two parts, parent gear and child gear.
    One gear can have multiple children, but only one parent.
    Either radius or teeth count can be used, as long as the
    the use is constant. Stiffness should be added
    to all gears, except for parent gears.

    Arguments:
    ----------
    node: int
        Nodal position of the gear in the global coordinates
    I: float
        Moment of inertia of the gear [kgm^2]
    R: float
        Radius of the gear [mm]
    k: float
        Stiffness of gear [Nm/rad]

    Keyword arguments:
    ------------------
    Parent: Gear
        openTorsion Gear instance of the connected parent gear
    """

    def __init__(self, node, I, R, k=None, c=None, parent=None):

        self.node = node
        self.I = I
        self.R = R
        self.k = k
        self.c = c
        self.parent = parent

        if parent is None:
            self.stages = None
        else:
            self.stages = []
            self.stages.append([[parent.node, parent.R], [self.node, self.R]])

    def M(self):
        """Mass Matrix of two 1 DOF gear elements.

        Returns
        -------
        M: ndarray
            Mass matrix of the gear elements
        """

        I = self.I
        M = np.array([[I]], dtype=np.float64)

        return M

    def K(self):
        """Stiffness matrix of a gear element. Gear mesh stiffness is assumed constant.

        Returns
        -------
        K: ndarray
            Stiffness matrix of elastic gear element
        """

        k = self.k

        # Initialize matrix
        K = np.array([[1, -1], [-1, 1]], dtype=np.float64) * k

        # Multiply first row and first column with R of parent
        R_P = -1
        K[0] *= R_P
        K[0][0] *= R_P
        K[1][0] *= R_P

        # Multiply second row and second column with R of child
        R = self.R / self.parent.R
        K[1] *= R
        K[0][1] *= R
        K[1][1] *= R

        return K

    def C(self):
        """Damping matrix of a gear element. Gears are assumed to have no damping.

        Returns
        -------
        C: ndarray
            Damping matrix of the gear element
        """

        c = self.c

        # Initialize matrix
        C = np.array([[1, -1], [-1, 1]], dtype=np.float64) * c

        # Multiply first row and first column with R of parent
        R_P = -1
        C[0] *= R_P
        C[0][0] *= R_P
        C[1][0] *= R_P

        # Multiply second row and second column with R of child
        R = self.R / self.parent.R
        C[1] *= R
        C[0][1] *= R
        C[1][1] *= R

        return C
