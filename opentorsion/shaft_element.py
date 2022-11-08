import numpy as np


class Shaft:
    """A 2-degree of freedom shaft object
    A shaft with a constant circular cross-section can be defined using length and diameter values. Other types of shafts are defined using stiffness and moment of inertia values. Giving either the stiffness or inertia value overrides all stiffness and inertia calculations.

    Arguments:
    ----------
    nl: int
        Nodal position of the left end of the element
    nr: int
        Nodal position of the right end of the element
    L: float
        Length of the shaft element [mm]
    odl: float
        Outer diameter of the shaft [mm]

    Keyword arguments:
    ------------------
    idl: float
        Inner diameter of the shaft [mm]
    G: float
        Shear modulus of the material [Pa]
    E: float
        Young's modulus of the material [Pa]
    rho: float
        Density of the material [kg/m^3]
    k: float
        Stiffness of the shaft [Nm/rad]
    I: float
        Moment of inertia of the shaft [kgm^2]
    c: float
        Damping coefficient of the shaft [Nms/rad]
    """

    def __init__(
        self, nl, nr, L, odl, idl=0, G=80e9, E=200e9, rho=8000, k=None, I=0.0, c=0.0
    ):

        if k is None:
            self.L = L * 1e-3
            self.idl = float(idl) * 1e-3
            self.odl = float(odl) * 1e-3

            # Calculate polar of inertia
            A = np.pi * ((self.odl ** 4) - (self.idl ** 4))
            J = np.pi * ((self.odl ** 4) - (self.idl ** 4)) / 32

            # Calculate mass moment of inertia
            self.mass = rho * J * self.L / 6

            # Calculate torsional stiffness
            self.k = G * J / self.L

        else:
            self.k = k
            self.idl = None
            self.odl = None
            self.mass = I

        self.nl = nl
        self.nr = nr
        self.c = float(c)

    def M(self):
        """Mass matrix of a shaft element"""

        A = np.array([[2, 1], [1, 2]], dtype=np.float64)

        M = A * self.mass

        return M

    def K(self):
        """Stiffness matrix of a shaft element"""

        K = np.array([[1, -1], [-1, 1]], dtype=np.float64) * self.k

        return K

    def C(self):
        """Damping matrix of a shaft element"""

        C = np.array([[1, -1], [-1, 1]], dtype=np.float64) * self.c

        return C
    
    def __str__(self):
        return 'Shaft, nl: ' + str(self.nl) + ' nr: '  + str(self.nr) + ' L: ' + str(self.L) + ' odl: ' + str(self.odl) + ' k: ' + str(self.k) + ' I: ' + str(self.I) + ' c: ' + str(self.c)
