from copy import copy
import numpy as np
from scipy import linalg as LA

from opentorsion.utils import DOF_mismatch_error


class Assembly:
    """
    This class assembles the multi-degree of freedom system matrices, includes
    functions for modal analysis and response analysis.

    Attributes
    ----------
    shaft_elements : list
        List containing the shaft elements
    disk_elements : list
        List containing the disk elements
    gear_elements : list
        List containing the gear elements
    """

    def __init__(
        self,
        shaft_elements,
        disk_elements=None,
        gear_elements=None,
    ):
        """
        Parameters
        ----------
        shaft_elements: list
            List containing the shaft elements
        disk_elements : list
            List containing the disk elements
        gear_elements : list
            List containing the gear elements
        """
        if shaft_elements is None:
            raise DOF_mismatch_error("Shaft elements == None")
            self.shaft_elements = None
        else:
            self.shaft_elements = [
                copy(shaft_element) for shaft_element in shaft_elements
            ]

        if gear_elements is None:
            self.gear_elements = None
        else:
            self.gear_elements = [copy(gear_element) for gear_element in gear_elements]

        self.disk_elements = disk_elements

        self.dofs = self.check_dof()

        self.M = self.assemble_M()
        self.C = self.assemble_C()
        self.K = self.assemble_K()

    def assemble_M(self):
        """
        Assembles the mass matrix

        Returns
        -------
        ndarray
            The mass matrix
        """

        M = np.zeros((self.dofs, self.dofs))

        if self.shaft_elements is not None:
            for element in self.shaft_elements:
                dofs = np.array([element.nl, element.nr])
                M[np.ix_(dofs, dofs)] += element.M()

        if self.disk_elements is not None:
            for element in self.disk_elements:
                M[element.node, element.node] += element.M()

        if self.gear_elements is not None:
            for element in self.gear_elements:
                M[element.node, element.node] += element.M()

            # Build transformation matrix
            E = self.E()
            transform = self.T(E)
            # Calculate transformed mass matrix
            M = np.dot(np.dot(transform.T, M), transform)

        return M

    def assemble_K(self):
        """
        Assembles the stiffness matrix

        Returns
        -------
        ndarray
            The stiffness matrix
        """

        K = np.zeros((self.dofs, self.dofs))

        if self.shaft_elements is not None:
            for element in self.shaft_elements:
                dofs = np.array([element.nl, element.nr])
                K[np.ix_(dofs, dofs)] += element.K()

        if self.disk_elements is not None:
            for element in self.disk_elements:
                K[element.node, element.node] += element.K()

        if self.gear_elements is not None:
            # Build transformation matrix
            E = self.E()
            transform = self.T(E)
            # Calculate transformed mass matrix
            K = np.dot(np.dot(transform.T, K), transform)

        return K

    def nongearK(self):
        """
        Assembles the stiffness matrix when gears are not considered
        """

        K = np.zeros((self.dofs, self.dofs))

        if self.shaft_elements is not None:
            for element in self.shaft_elements:
                dofs = np.array([element.nl, element.nr])
                K[np.ix_(dofs, dofs)] += element.K()

        return K

    def assemble_C(self):
        """
        Assembles the damping matrix

        Returns
        -------
        ndarray
            The damping matrix assembled with component specific damping coefficients
        """

        C = np.zeros((self.dofs, self.dofs))

        if self.shaft_elements is not None:
            for element in self.shaft_elements:
                dof = np.array([element.nl, element.nr])
                C[np.ix_(dof, dof)] += element.C()

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
        """
        Assembles the gear constraint matrix

        Returns
        -------
        ndarray
            The gear constraint matrix
        """

        stages = []
        for gear in self.gear_elements:
            if gear.stages is not None:
                stages += gear.stages

        E = np.zeros([self.dofs, len(stages)])
        for i, stage in enumerate(stages):
            E[stage[0][0]][i] += stage[0][1]
            E[stage[1][0]][i] += stage[1][1]

        return E

    def T(self, E):
        """
        Method for determining gear constraint transformation matrix

        Parameters
        ----------
        E : ndarray
            The gear constraint matrix

        Returns
        -------
        ndarray
            The gear constraint transformation matrix
        """

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

    def state_matrix(self, C=None):
        """
        Assembles the state matrices for eigenvalue calculation

        Parameters
        ----------
        C : ndarray, optional
            Damping matrix

        Returns
        -------
        ndarray
            The state matrix
        ndarray
            The input matrix
        """

        if C is None:
            C = self.C
        M, K = self.M, self.K
        Z = np.zeros(M.shape, dtype=np.float64)

        A = np.vstack([np.hstack([C, K]), np.hstack([-M, Z])])
        B = np.vstack([np.hstack([M, Z]), np.hstack([Z, M])])

        return A, B

    def C_modal(self, M, K, xi=0.02):
        """
        Full damping matrix for mechanical system obtained from modal damping matrix

        Parameters
        ----------
        M : ndarray
            Assembly mass matrix, included for debugging reasons
        K : ndarray
            Assembly stiffness matrix, included for debugging reasons
        xi : float, optional
            Modal damping factor, default is 0

        Returns
        -------
        ndarray
            The full damping matrix
        """
        omegas, phi = LA.eig(K, M)
        omegas = np.absolute(omegas)

        # Modal mass matrix is calculated using the eigenvectors
        M_modal = phi.T @ M @ phi

        # Nondiagonal elements are removed to prevent floating point error and inverse square root is applied
        M_modal_inv = LA.fractional_matrix_power(np.diag(np.diag(M_modal)), -0.5)

        # The mode shape matrix is normalized by multiplying with the inverse modal matrix
        phi_norm = phi @ M_modal_inv

        # The diagonal modal damping matrix is achieved by applying the modal damping
        C_modal_elements = 2 * xi * np.sqrt(omegas)
        C_modal_diag = np.diag(C_modal_elements)

        # The final damping matrix
        C = LA.inv(phi_norm.T) @ C_modal_diag @ LA.inv(phi_norm)

        return C

    def ss_response(self, excitations, omegas, C=None):
        '''
        Calculation of the steady-state torsional response.

        Parameters
        ----------
        excitations: complex ndarray
            A numpy array containing the excitationse each row corresponds to
            one node, and each column corresponds to one frequency.
        omegas: ndarray
            Angular frequencies of the excitations
        C: ndarray, optional
            Damping matrix, if not given, uses the default damping matrix

        Returns
        -------
        complex ndarray
            Displacement response
        complex ndarray
            Speed response
        '''
        if C is None:
            C = self.C
        N = self.M.shape[0]
        q_matrix = np.zeros((N, len(omegas)), dtype='complex128')
        w_matrix = np.zeros((N, len(omegas)), dtype='complex128')

        for i, w in enumerate(omegas):
            receptance = np.linalg.inv(-self.M*w**2 + w*1.0j*C + self.K)
            q = receptance @ excitations[:, i]
            w = q * 1.0j * w
            q_matrix[:, i] = q.ravel()
            w_matrix[:, i] = w.ravel()

        return q_matrix, w_matrix

    def vibratory_torque(self, excitations, omegas, k_shafts, C=None):
        """
        Vibratory torque calculation at one rotating speed.

        Parameters
        ----------
        excitations: complex ndarray
            A numpy array containing the excitationse each row corresponds to
            one node, and each column corresponds to one frequency.
        omegas: ndarray
            Angular frequencies of the excitations
        k_shafts: ndarray
            Stiffness values of the shafts
        C: ndarray, optional
            Damping matrix, if not given, uses the default damping matrix

        Returns
        -------
        ndarray
            Vibratory torque at each shaft resulting from each excitation
        """
        if C is None:
            C = self.C
        q_res, _ = self.ss_response(excitations, omegas, C=C)
        q_difference = (q_res.T[:, 1:] - q_res.T[:, :-1]).T
        T_vib = q_difference * k_shafts[:, None]

        return T_vib

    def undamped_modal_analysis(self):
        """
        Calculates the undamped eigenvalues and eigenvectors of the assembly

        Returns
        -------
        complex ndarray
            The eigenvalues of the undamped assembly
        complex ndarray
            The eigenvectors of the undamped assembly
        """

        lam, vec = LA.eig(self.K, self.M)

        return lam, vec

    def modal_analysis(self, C=None):
        """
        Calculates the eigenvalues and eigenfrequencies of the assembly

        Returns
        -------
        ndarray
            The undamped eigenfrequencies in rad/s
        ndarray
            The damped eigenfrequencies in rad/s
        ndarray
            The damping ratios
        """
        if C is None:
            C = self.C

        M, K = self.M, self.K
        N = M.shape[0]
        A = np.vstack([
            np.hstack([np.zeros((N, N)), np.eye(N)]),
            np.hstack([-np.linalg.inv(M) @ K, -np.linalg.inv(M)@C])
        ])

        evals, evecs = np.linalg.eig(A)
        evals = sorted(evals, key=np.abs)
        wn = np.abs(evals)
        wd = np.imag(evals)
        damping_ratios = -np.real(evals) / np.abs(evals)

        return wn, wd, damping_ratios

    def eigenmodes(self):
        """
        Solve system eigenmodes

        Returns
        -------
        complex ndarray
            Eigenmode array, columns correspond to modes left to right starting
            from zero
        """
        A, B = self.state_matrix()
        lam, vec = LA.eig(A, B)

        lam = lam[::2]
        vec = vec[: int(vec.shape[0] / 2)]
        vec = vec[:, ::2]

        inds = np.argsort(np.abs(lam))
        eigenmodes = np.zeros(vec.shape, dtype="complex128")
        for i, v in enumerate(inds):
            eigenmodes[:, i] = vec[:, v]

        return lam, eigenmodes

    def check_dof(self):
        """
        Returns the number of degrees of freedom in the model

        Returns
        -------
        int
            Number of degrees of freedom of the assembly
        """

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

        return max(nodes) + 1

    def state_space(self, C=None):
        '''
        State space matrices of the second order system.

        Parameters
        ----------
        C: ndarray, optional
            Damping matrix, if not given, uses the default damping matrix

        Returns
        -------
        A_sys: ndarray
            State matrix A
        B_sys: ndarray
            Input matrix B
        '''
        if C is None:
            C = self.C
        M, K = self.M, self.K
        Z = np.zeros(M.shape)
        I_mat = np.eye(M.shape[0])
        M_inv = LA.inv(M)
        A_sys = np.vstack([
            np.hstack([Z, I_mat]),
            np.hstack([-M_inv @ K, -M_inv @ C])
        ])
        B_sys = np.vstack([Z, M_inv])

        return A_sys, B_sys

    def continuous_2_discrete(self, A, B, ts):
        """
        C2D computes a discrete-time model of a system (A_c,B_c) with sample time
        ts. The function returns matrices Ad, Bd of the discrete-time system.

        Parameters
        -------
        A: ndarray
            Continuous system state matrix A
        B: ndarray
            Continuous system input matrix B
        ts: float
            Sample time for the discrete system

        Returns
        -------
        Ad: ndarray
            Discrete system state matrix A
        Bd: ndarray
            Discrete system input matrix B
        """
        m, n = A.shape
        nb = B.shape[1]
        s = np.concatenate([A, B], axis=1)
        s = np.concatenate([s, np.zeros((nb, n+nb))], axis=0)
        S = LA.expm(s*ts)
        Ad = S[0:n, 0:n]
        Bd = S[0:n, n:n+nb+1]

        return Ad, Bd
