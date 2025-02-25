import numpy as np


class ElasticGear:
    """A elastic gear object
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

    def __init__(self, node, I, R, k=None, parent=None):

        self.node = node
        self.I = I
        self.R = R
        self.k = k
        self.parent = parent


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
        K[0] *= self.parent.R
        K[0][0] *= self.parent.R
        K[1][0] *= self.parent.R

        # Multiply second row and second column with R of child
        R = self.R
        K[1] *= R
        K[0][1] *= R
        K[1][1] *= R

        return K

    def C(self):
        """Damping matrix of a gear element. Gears are assumed to have no damping.

        Returns
        -------
        M: ndarray
            Damping matrix of the gear element
        """
        
        C = np.zeros((1))

        return C