from copy import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.sparse import linalg as las
from scipy.signal import lti
from scipy.signal import lsim

from opentorsion.disk_element import Disk
from opentorsion.shaft_element import Shaft
from opentorsion.gear_element import Gear

# from opentorsion.induction_motor import Induction_motor
from opentorsion.errors import DOF_mismatch_error


class Assembly:
    """Powertrain assembly"""

    def __init__(
        self,
        shaft_elements,
        disk_elements=None,
        gear_elements=None,
        motor_elements=None,
    ):

        ## Initiate shaft elements
        if shaft_elements is None:
            raise DOF_mismatch_error("Shaft elements == None")
            self.shaft_elements = None
        else:
            self.shaft_elements = [
                copy(shaft_element) for shaft_element in shaft_elements
            ]

        ## Initiate gear elements
        if gear_elements is None:
            self.gear_elements = None
        else:
            self.gear_elements = [copy(gear_element) for gear_element in gear_elements]

        ## Initiate motor elements
        if motor_elements is None:
            self.motor_elements = None
        else:
            self.motor_elements = [
                copy(motor_element) for motor_element in motor_elements
            ]

        self.disk_elements = disk_elements

        self.dofs = self._check_dof()

    def __repr__(self):
        pass

    def __str__(self):
        return f"rotor"

    def M(self):
        """Assembles the mass matrix"""

        M = np.zeros((self.dofs, self.dofs))

        if self.shaft_elements is not None:
            for element in self.shaft_elements:
                dofs = np.array([element.nl, element.nr])
                M[np.ix_(dofs, dofs)] += element.M()

        if self.disk_elements is not None:
            for element in self.disk_elements:
                M[element.node, element.node] += element.M()

        # if self.motor_elements is not None:
        #     for element in self.motor_elements:
        #         dof = np.array([element.nl, element.nr])
        #         M[np.ix_(dof, dof)] += element.M()

        if self.gear_elements is not None:
            for element in self.gear_elements:
                M[element.node, element.node] += element.M()

            # Build transformation matrix
            E = self.E()
            transform = self.T(E)
            # Calculate transformed mass matrix
            M = np.dot(np.dot(transform.T, M), transform)

        return M

    def K(self):
        """Assembles the stiffness matrix"""
        K = np.zeros((self.dofs, self.dofs))

        if self.shaft_elements is not None:
            for element in self.shaft_elements:
                dofs = np.array([element.nl, element.nr])
                K[np.ix_(dofs, dofs)] += element.K()

        # if self.motor_elements is not None:
        #     for element in self.motor_elements:
        #         dofs = np.array([element.nl, element.nr])
        #         K[np.ix_(dofs, dofs)] += element.K()

        if self.gear_elements is not None:
            # Build transformation matrix
            E = self.E()
            transform = self.T(E)
            # Calculate transformed mass matrix
            K = np.dot(np.dot(transform.T, K), transform)

        # print(K)
        return K

    def C(self):
        """Assembles the damping matrix"""
        C = np.zeros((self.dofs, self.dofs))

        if self.shaft_elements is not None:
            for element in self.shaft_elements:
                dof = np.array([element.nl, element.nr])
                C[np.ix_(dof, dof)] += element.C()

        # if self.motor_elements is not None:
        #     for element in self.motor_elements:
        #         dof = np.array([element.nl, element.nr])
        #         C[np.ix_(dof, dof)] += element.C()

        if self.disk_elements is not None:
            for element in self.disk_elements:
                C[element.node, element.node] += element.C()

        if self.gear_elements is not None:
            for element in self.gear_elements:
                C[element.node, element.node] += element.C()

            # Build transformation matrix
            E = self.E()
            transform = self.T(E)
            # Calculate transformed mass matrix
            C = np.dot(np.dot(transform.T, C), transform)

        return C

    def E(self):
        """Assembles the gear constraint matrix"""

        stages = []
        for gear in self.gear_elements:
            if gear.stages is not None:
                stages += gear.stages

        E = np.zeros([self.dofs, len(stages)])
        for i, stage in enumerate(stages):
            E[stage[0][0]][i] += stage[0][1]
            E[stage[1][0]][i] += stage[1][1]
            try:
                E[stage[2][0]][i] += stage[2][1]
            except:
                pass

        return E

    def state_matrix(self):
        """Assembles the state-space matrices"""

        M, K, C = self.M(), self.K(), self.C()
        Z = np.zeros(M.shape, dtype=np.float64)

        if self.motor_elements is not None:
            motor = self.motor_elements[0]

            if motor.small_signal:  # Different versions for linear and nonlinear models
                R, L = motor.R_linear(), motor.L_linear()
            else:
                R, L = motor.R(), motor.L()

            A = np.zeros((self.dofs * 2 + 4, self.dofs * 2 + 4))
            B = np.zeros(A.shape)

            dof = np.array([0, 1, 2, 3, 4])

            A[np.ix_(dof, dof)] += R
            B[np.ix_(dof, dof)] += L

            K_m = np.vstack([np.hstack([C, K]), np.hstack([-M, Z])])

            M_m = np.vstack([np.hstack([M, Z]), np.hstack([Z, M])])

            dof = np.array(range(4, self.dofs * 2 + 4))
            A[np.ix_(dof, dof)] += K_m
            B[np.ix_(dof, dof)] += M_m

        else:
            A = np.vstack([np.hstack([C, K]), np.hstack([-M, Z])])

            B = np.vstack([np.hstack([M, Z]), np.hstack([Z, M])])

            # Solved versions
            # A = np.vstack([
            #     np.hstack([LA.solve(-M, C), LA.solve(-M, K)]),
            #     np.hstack([I, Z]) # ])
            # B = np.vstack([M_inv, Z])

        # np.set_printoptions(suppress=True)
        # print(A)

        return A, B

    def modal_analysis(self):
        """Calculates the eigenvalues and eigenfrequencies of the assembly"""
        A, B = self.state_matrix()
        lam, vec = self._eig(A, B)

        # Sort and delete complex conjugates

        omegas = np.sort(np.absolute(lam))
        omegas_damped = np.sort(np.abs(np.imag(lam)))
        freqs = omegas / (2 * np.pi)
        damping_ratios = -np.real(lam) / (np.absolute(lam))

        return omegas_damped, freqs, damping_ratios

    def _eig(self, A, B):
        """Solves the eigenvalues of the state space matrix using ARPACK"""
        lam, vec = LA.eig(A, B)

        return lam, vec

    def _check_dof(self):
        """Returns the number of degrees of freedom in the model"""
        nodes = set()
        if self.shaft_elements is not None:
            for element in self.shaft_elements:
                nodes.add(element.nl)
                nodes.add(element.nr)

        if self.disk_elements is not None:
            for element in self.disk_elements:
                nodes.add(element.node)

        if self.gear_elements is not None:
            for element in self.gear_elements:
                nodes.add(element.node)

        if self.motor_elements is not None:
            for element in self.motor_elements:
                nodes.add(element.n)

        return max(nodes) + 1

    def T(self, E):
        """Method for determining gear constraint transformation matrix"""
        r, c = E.shape
        T = np.eye(r)
        for i in range(c):
            E_i = np.dot(T.T, E)

            # (1) Set T_i = I(n+1) (The identity matrix of dimension (n_i + 1))
            T_i = np.eye(r)

            # (2) Define k as the position of the entry having the largest absolute value in the ith column of E_i-1
            k = np.argmax(np.abs(E_i[:, i]))

            # (3) Replace row k of T_i with the transpose of column i from E_(i-1)
            T_i[k] = E_i[:, i]

            # (4) Divide this row by the negative of its kth element
            T_i[k] = T_i[k] / (-1 * T_i[k][k])

            # (5) Strike out column k from the matrix
            T_i = np.delete(T_i, k, axis=1)
            T = np.dot(T, T_i)

            r -= 1

        return T

    def U(self, u1, u2):
        """Input matrix of the state-space model"""
        # u1 at node '0', u2 at node 'n'

        if np.array([u2]).all() == None:
            u2 = np.zeros((1, np.size(u1)))
            u2 = u2[0]

        return np.vstack([[u1], np.zeros((self.M().shape[1] - 2, np.size(u1))), [u2]]).T

    def time_domain(self, t_in, u1, u2=None, U=None, system=None, x_in=None):
        """Time-domain analysis of the powertrain"""

        if system == None:
            system = self.system()

        if U == None:
            U = self.U(u1, u2)

        tout, yout, xout = lsim(system, U, t_in, X0=x_in)

        torques, omegas, thetas = self.torque(yout)

        return tout, torques, omegas, thetas

    def system(self):
        """System model in the ltis-format"""
        M, K, C = self.M(), self.K(), self.C()

        Z = np.zeros(M.shape, dtype=np.float64)
        I = np.eye(M.shape[0])
        M_inv = LA.inv(M)

        A = np.vstack([np.hstack([-M_inv @ C, -M_inv @ K]), np.hstack([I, Z])])

        B = np.vstack([M_inv, Z])

        C, D = np.eye(B.shape[0]), np.zeros(B.shape)

        return lti(A, B, C, D)

    def torque(self, yout):
        """Calculate torque between every node"""

        omegas, thetas = np.hsplit(yout, 2)

        k_val = np.abs(np.diag(self.K(), 1))
        K = np.diag(k_val, 1)
        K -= np.vstack(
            [
                np.hstack([np.diag(k_val), np.transpose([np.zeros(K.shape[1] - 1)])]),
                [np.zeros(K.shape[0])],
            ]
        )
        K = K[:-1, :]

        if self.gear_elements is not None:
            i = 0
            for element in self.gear_elements:
                if element.stages is not None:
                    print(element.stages)
                    K[(element.stages[0][0][0] - i)] = [
                        np.abs(element.stages[0][0][1] / element.stages[0][1][1]) * x
                        if x < 0
                        else x
                        for x in K[element.stages[0][0][0] - i]
                    ]
                    i += 1

        torques = [np.dot(K, np.abs(x)) for x in thetas]

        return torques, omegas, thetas
