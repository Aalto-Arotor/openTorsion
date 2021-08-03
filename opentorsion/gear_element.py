import numpy as np


class Gear:
    """A gear object
    Gears consist of two parts, parent gear and child gear.
    One gear can have multiple children, but only one parent.
    Either radius or teeth count can be used, as long as the
    the use is constant.

    Arguments:
    ---------_
    node: int
        Nodal position of the gear in the global coordinates
    I: float
        Moment of inertia of the gear [kgm^2]
    R: float
        Radius of the gear [mm]

    Keyword arguments:
    ------------------
    Parent: Int
        Node value of the parent gear element
    """

    def __init__(self, node, I, R, parent=None, parent2=None):

        self.node = node
        self.I = I
        self.R = R

        if parent is None:
            self.stages = None
        elif parent2 is None:
            self.stages = []
            self.stages.append([[parent.node, parent.R], [self.node, self.R]])
        else:
            self.stages = []
            self.stages.append(
                [
                    [parent.node, parent.R],
                    [parent2.node, parent2.R],
                    [self.node, self.R],
                ]
            )

    def M(self):
        """Mass Matrix of 1 DOF gear element

        Returns
        -------
        M: ndarray
            Mass matrix of the gear element
        """

        I = self.I
        M = np.array([I])

        return M

    def K(self):
        """Stiffness matrix of a gear element. Gears are assumed to be rigid

        Returns
        -------
        M: ndarray
            Mass matrix of the gear element
        """
        K = np.zeros((1))

        return K

    def C(self):
        """Damping matrix of a gear element. Gears are assumed to have no damping

        Returns
        -------
        M: ndarray
            Mass matrix of the gear element
        """
        C = np.zeros((1))

        return C
