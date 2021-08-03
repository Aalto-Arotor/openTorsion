from copy import copy
import matplotlib

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.sparse import linalg as las
from scipy.signal import lti
from scipy.signal import lsim

from disk_element import Disk
from shaft_element import Shaft
from gear_element import Gear
# from induction_motor2 import Induction_motor
from errors import DOF_mismatch_error


class Rotor():
    '''Powertrain assembly'''
    def __init__(self,
                 shaft_elements,
                 disk_elements=None,
                 gear_elements=None,
                 motor_elements=None):

        ## Initiate shaft elements
        if shaft_elements is None:
            raise DOF_mismatch_error('Shaft elements == None')
            self.shaft_elements = None
        else:
            self.shaft_elements = [copy(shaft_element) for shaft_element in shaft_elements]

        ## Initiate gear elements
        if gear_elements is None:
            self.gear_elements = None
        else:
            self.gear_elements = [copy(gear_element) for gear_element in gear_elements]

        ## Initiate motor elements
        if motor_elements is None:
            self.motor_elements = None
        else:
            self.motor_elements = [copy(motor_element) for motor_element in motor_elements]

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

            if motor.small_signal: # Different versions for linear and nonlinear models
                R, L = motor.R_linear(), motor.L_linear()
            else:
                R, L = motor.R(), motor.L()

            # print(R)
            A = np.zeros((self.dofs*2+4, self.dofs*2+4))
            B = np.zeros(A.shape)

            dof = np.array([0,1,2,3,4])

            A[np.ix_(dof, dof)] += R
            B[np.ix_(dof, dof)] += L

            K_m = np.vstack([
                np.hstack([C, K]),
                np.hstack([-M, Z])
            ])

            M_m = np.vstack([
                np.hstack([M, Z]),
                np.hstack([Z, M])
            ])

            dof = np.array(range(4, self.dofs*2+4))
            A[np.ix_(dof, dof)] += K_m
            B[np.ix_(dof, dof)] += M_m

        else:
            A = np.vstack([
                np.hstack([C, K]),
                np.hstack([-M, Z])
            ])

            B = np.vstack([
                np.hstack([M, Z]),
                np.hstack([Z, M])
            ])

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
        freqs = omegas/(2*np.pi)
        damping_ratios = -np.real(lam)/(np.absolute(lam))

        return omegas_damped, freqs, damping_ratios

    def _eig(self, A, B):
        """Solves the eigenvalues of the state space matrix using ARPACK"""
        lam, vec = LA.eig(A, B)

        return lam, vec

    def _check_dof(self):
        '''Returns the number of degrees of freedom in the model'''
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

        return max(nodes)+1

    def T(self, E):
        ''' Method for determining gear constraint transformation matrix'''
        r, c = E.shape
        T = np.eye(r)
        for i in range(c):
            E_i = np.dot(T.T, E)

            # (1) Set T_i = I(n+1) (The identity matrix of dimension (n_i + 1))
            T_i = np.eye(r)

            # (2) Define k as the position of the entry having the largest absolute value in the ith column of E_i-1
            k = np.argmax(np.abs(E_i[:,i]))

            # (3) Replace row k of T_i with the transpose of column i from E_(i-1)
            T_i[k] = E_i[:,i]

            # (4) Divide this row by the negative of its kth element
            T_i[k] = T_i[k]/(-1*T_i[k][k])

            # (5) Strike out column k from the matrix
            T_i = np.delete(T_i, k, axis=1)
            T = np.dot(T, T_i)

            r -= 1

        return T

    def U(self, u1, u2):
        '''Input matrix of the state-space model '''
        # u1 at node '0', u2 at node 'n'

        if np.array([u2]).all() == None:
            u2 = np.zeros((1, np.size(u1)))

        return np.vstack([[u1], np.zeros((self.M().shape[1]-2, np.size(u1))), [u2]]).T

    def time_domain(self, t_in, u1, u2=None, U=None, system=None, x_in=None):
        '''Time-domain analysis of the powertrain'''

        if system == None:
            system = self.system()

        if U == None:
            U = self.U(u1, u2)

        tout, yout, xout = lsim(system, U, t_in, X0=x_in)

        torques, omegas, thetas = self.torque(yout)

        return tout, torques, omegas, thetas

    def system(self):
        '''System model in the ltis-format'''
        M, K, C  = self.M(), self.K(), self.C()

        Z = np.zeros(M.shape, dtype=np.float64)
        I = np.eye(M.shape[0])
        M_inv = LA.inv(M)

        A = np.vstack([
            np.hstack([-M_inv @ C, -M_inv @ K]),
            np.hstack([         I,          Z])
        ])

        B = np.vstack([M_inv, Z])

        C, D = np.eye(B.shape[0]), np.zeros(B.shape)

        return lti(A, B, C, D)

    def torque(self, yout):
        '''Calculate torque between every node'''

        omegas, thetas = np.hsplit(yout, 2)

        k_val = np.diag(self.K(), 1)*(-1)
        K = np.diag(k_val, 1)
        K -= np.vstack([
            np.hstack([np.diag(k_val), np.transpose([np.zeros(K.shape[1]-1)])]),
            [np.zeros(K.shape[0])]
        ])
        K = K[:-1, :]

        if self.gear_elements is not None:
            i = 0
            for element in self.gear_elements:
                if element.stages is not None:
                    K[(element.stages[0][0][0]-i)] = [
                        np.abs(element.stages[0][0][1]/element.stages[0][1][1])*x
                        if x < 0 else x for x in K[element.stages[0][0][0]-i]
                    ]
                    i += 1

        torques = [np.dot(K, x) for x in thetas]

        return torques, omegas, thetas

# if __name__ == '__main__':

#     #Induction motor inertia
#     J_IM=0.196
#     # Synchronous reluctance motor inertia
#     J_SRM=0.575

#     M_c=30                                  #kg
#     J_hub1=17e-3                            #kgm^2
#     J_hub2=17e-3                            #kgm^2
#     J_tube=37e-3*(0.55-2*0.128)             #kgm^2
#     J_coupling=J_hub1+J_hub2+J_tube         #kgm^2
#     K_insert1=41300                        #Nm/rad
#     K_insert2=41300                        #Nm/rad
#     K_tube=389000*(0.55-2*0.128)            #Nm/rad
#     K_coupling=1/(1/K_insert1+1/K_tube) #Nm/rad

#     shafts = []
#     disks = []
#     gears = []
#     motors = []

#     rho = 7850
#     G = 81e9
#     J_rotor = 13094740e-6

#     Ig = 0

#     n = 0 # edellisen elementin viimeinen arvo

#     f = 50
#     p = 1
#     omega = 2957
#     voltage = 400
#     operating = [0.024910, 0.018530, 0.016077, 0.015782, 0.014726]
#     motor_arotor = Induction_motor(0, 1, omega, f, p, voltage=voltage, circuit_parameters=operating)

#     motors.append(Induction_motor())

#     disks.append(Disk(0, J_IM)) # Moottori
#     gears.append(gear1 := Gear(0, Ig, 1.95)) #vaihde
#     gears.append(Gear(1, Ig, 1, parent=gear1)) # vaihde

#     shafts.append(Shaft(1, 2, None,None, k=K_coupling, I=J_coupling)) # Kytkin

#     # TELA
#     shafts.append(Shaft(2, 3, 185, 100))
#     shafts.append(Shaft(3, 4, 335, 119))
#     shafts.append(Shaft(4, 5, 72, 125))
#     shafts.append(Shaft(5, 6, 150, 320))
#     shafts.append(Shaft(6, 7, 3600, 320, idl=287))
#     shafts.append(Shaft(7, 8, 150, 320))
#     shafts.append(Shaft(8, 9, 72, 125))
#     shafts.append(Shaft(9, 10, 335, 119))
#     shafts.append(Shaft(10, 11, 185, 100))
#     ##

#     shafts.append(Shaft(11, 12, None, None, k=180e3, I=15e-4)) # Momenttianturi
#     # shafts.append(Shaft(12, 13, None,None, k=K_coupling, I=J_coupling)) # Kytkin
#     shafts.append(Shaft(12, 13, None,None, k=K_coupling, I=J_coupling)) # Kytkin
#     disks.append(Disk(13, J_SRM)) # Toinen moottori

#     # shafts.append(Shaft(10, 11, None,None, k=K_coupling, I=J_coupling)) # kytkin
#     # gears.append(gear1 := Gear(11, Ig, 1)) #vaihde
#     # gears.append(Gear(12, Ig, 1.95, parent=gear1)) # vaihde
#     # disks.append(Disk(13, J_IM)) # Toinen moottori

#     assembly = Rotor(shafts, disk_elements=disks, gear_elements=gears)

#     assembly.modal_analysis()
