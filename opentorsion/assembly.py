from copy import copy
import matplotlib

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.sparse import linalg as las

from opentorsion.disk_element import Disk
from opentorsion.shaft_element import Shaft
from opentorsion.gear_element import Gear
from opentorsion.errors import DOF_mismatch_error


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

        if self.motor_elements is not None:
            for element in self.motor_elements:
                dof = np.array([element.nl, element.nr])
                M[np.ix_(dof, dof)] += element.M()

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

        if self.motor_elements is not None:
            for element in self.motor_elements:
                dofs = np.array([element.nl, element.nr])
                K[np.ix_(dofs, dofs)] += element.K()

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

        if self.motor_elements is not None:
            for element in self.motor_elements:
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
        """Assembles the gear constraint matrix"""

        stages = []
        for gear in self.gear_elements:
            if gear.stages is not None:
                stages += gear.stages

        E = np.zeros([self.dofs, len(stages)])
        for i, stage in enumerate(stages):
            E[stage[0][0]][i] += stage[0][1]
            E[stage[1][0]][i] += stage[1][1]

        return E

    def state_matrix(self):
        """Assembles the state space matrix"""
        # S = np.dot(self.K(), LA.inv(self.M()))

        M, K, C = self.M(), self.K(), self.C()

        Z = np.zeros(M.shape, dtype=np.float64)

        A = np.vstack([
            np.hstack([C, K]),
            np.hstack([-M, Z])
        ])

        B = np.vstack([
            np.hstack([M, Z]),
            np.hstack([Z, M])
        ])

        # A = np.vstack(
             # np.hstack([la.solve(-self.M(), self.K(frequency)), la.solve(-self.M(), (self.C(frequency))])])

        return A, B

    def modal_analysis(self):
        """Calculates the eigenvalues and eigenfrequencies of the assembly"""
        A, B = self.state_matrix()
        lam, vec = self._eig(A, B)

        # Sort and delete complex conjugates
        lam = np.sort(lam)
        lam = np.delete(lam, [i*2+1 for i in range(len(lam)//2)])

        omegas = np.sort(np.absolute(lam))
        omegas_damped = np.sort(np.imag(lam))
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
                nodes.add(element.nl)
                nodes.add(element.nr)

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

    def torque(self, yout):
        '''Incomplete'''
        thetas, omegas = np.hsplit(yout, 2)
        k_values = []
        torques = []
        for element in self.shaft_elements:
            k_values.append(element.k)

        j = 0
        for theta in thetas:
            torques.append([])
            for i in range(len(k_values)):
                torques[j].append((theta[i+1]-theta[i])*k_values[i])
            j += 1

        return torques, thetas

    def campbell_diagram(self, num_modes=4, frequency_range=100):
        omegas_damped, freqs, damping_ratios = self.modal_analysis()
        freqs = freqs[:num_modes]
        print(freqs)
        self.plot_campbell(frequency_range, freqs)

        return

    def plot_campbell(self, frequency_range, modes, excitations=[1, 2, 3, 4]):
        '''Plots the campbell diagram of the powertrain'''
        # fig = plt.figure()
        for i in modes:
            plt.plot([0, frequency_range], [i, i])

        for i in excitations:
            plt.plot([0, frequency_range], [0, i*frequency_range])

        plt.xlim([0, frequency_range])
        #plt.ylim([0, max(modes)+10])
        plt.show()

        return


    def figure_2D(self):
        '''Creates a 2D-figure of the powertrain'''

        fig, ax = plt.subplots(nrows=1, ncols=1)
        x_axis, y_axis = ax.get_xaxis(), ax.get_yaxis()
        x_axis.set_visible(False)
        y_axis.set_visible(False)
        shaft_mass, disk_mass, nodes_sl, nodes_sr, nodes_d, nodes_g, disk_on_gear = [], [], [], [], [], [], []

        '''Appends nodes to lists'''
        if self.shaft_elements is not None:
            for element in self.shaft_elements:
                shaft_mass.append(element.mass)
                nodes_sl.append(element.nl)
                nodes_sr.append(element.nr)
            if nodes_sr != []:
                nodes_sl.append(nodes_sr[-1]) # nodes_sl contains all nodes

        if self.disk_elements is not None:
            for element_d in self.disk_elements:
                disk_mass.append(element_d.I)
                nodes_d.append(element_d.node)

        if self.gear_elements is not None:
            for element_g in self.gear_elements:
                nodes_g.append(element_g.node)

        '''Axis limits and starting point'''
        max_nodes = len(nodes_sl)+1+len(nodes_g)/2
        x0 = 0.5 # figure start coordinates (x0, y0)
        y0 = 1/(3*len(nodes_g)+1)
        l = 1/(2*x0) # length of a shaft/gear element
        h = 0.05*l # height of a shaft/gear element
        plt.xlim(right=max_nodes)
        plt.ylim(top=y0+l/2, bottom=y0-l/2)
        k = len(nodes_g)*h # to level shaft and gear figures right
        i = 0
        m = 0.01*l # for node number text coordinate

        '''Draws gear elements and possible disk elements'''
        for node in nodes_g:
            if node in nodes_d:
                polygon_edges = [(x0+node*l, y0+k),
                                 (x0+node*l+0.2*l, y0+k+0.05*l),
                                 (x0+node*l+0.2*l, y0+k+0.1*l),
                                 (x0+node*l-0.2*l, y0+k+0.1*l),
                                 (x0+node*l-0.2*l, y0+k+0.05*l)] # coordinates for upper disk polygon

                neg_polygon_edges = [(x0+node*l, y0+k-2*h),
                                     (x0+node*l+0.2*l, y0+k-0.05*l-2*h),
                                     (x0+node*l+0.2*l, y0+k-0.1*l-2*h),
                                     (x0+node*l-0.2*l, y0+k-0.1*l-2*h),
                                     (x0+node*l-0.2*l, y0+k-0.05*l-2*h)] # coordinates for lower disk polygon

                if node in nodes_d:
                    ax.add_patch(matplotlib.patches.Polygon(polygon_edges, ec='black', fc='black'))
                    # disk figure upper part
                    ax.add_patch(matplotlib.patches.Polygon(neg_polygon_edges, ec='black', fc='black'))
                    # disk figure lower part

            if i % 2 == 0:
                plt.text(x0+node*l+m*4, y0+k+m, str(node)) # add node number as text
                ax.add_patch(matplotlib.patches.Rectangle((x0+node*l, y0+k), l, -h, ec='black', fc='green'))
                # gear figure upper part
                k -= h
                ax.add_patch(matplotlib.patches.Rectangle((x0+node*l, y0+k), l, -h, ec='black', fc='green'))
                # gear figure lower part
            i += 1
        k = len(nodes_g)*h

        '''Draws shaft and possible disk elements'''
        for node in nodes_sl:
            if node in nodes_g:
                k -= h
            plt.text(x0+node*l+m*4, y0+k+m, str(node)) # add node number as text

            if node != nodes_sl[-1]:
                ax.add_patch(matplotlib.patches.Rectangle((x0+node*l, y0+k), l, -h, ec='black')) # shaft figure

            polygon_edges = [(x0+node*l, y0+k),
                             (x0+node*l+0.2*l, y0+k+0.05*l),
                             (x0+node*l+0.2*l, y0+k+0.1*l),
                             (x0+node*l-0.2*l, y0+k+0.1*l),
                             (x0+node*l-0.2*l, y0+k+0.05*l)] # coordinates for upper disk polygon

            neg_polygon_edges = [(x0+node*l, y0+k-h),
                                 (x0+node*l+0.2*l, y0+k-0.05*l-h),
                                 (x0+node*l+0.2*l, y0+k-0.1*l-h),
                                 (x0+node*l-0.2*l, y0+k-0.1*l-h),
                                 (x0+node*l-0.2*l, y0+k-0.05*l-h)] # coordinates for lower disk polygon
            if node in nodes_d:
                ax.add_patch(matplotlib.patches.Polygon(polygon_edges, ec='black', fc='black'))
                # disk figure upper part
                ax.add_patch(matplotlib.patches.Polygon(neg_polygon_edges, ec='black', fc='black'))
                # disk figure lower part
        plt.show()


    def figure_eigenmodes(self, modes=4):
        '''Plots eigenmodes of the powertrain'''
        fig_modes, axs = plt.subplots(4, 1, sharex=True)
        plt.ylim(-1.1, 1.1)

        A, B = self.state_matrix()
        lam, vec = self._eig(A, B)
        inds = np.argsort(np.abs(lam))

        nodes_g = []

        if self.gear_elements is not None:
            for element_g in self.gear_elements:
                nodes_g.append(element_g.node)
            nodes_g = nodes_g[1::2]

        s = []

        for mode in range(modes):
            mode *= 2
            plot_vec = []

            this_mode = vec[:, inds[mode]]
            this_mode = np.abs(this_mode[-this_mode.size//2:])

            # Do not normalize rigid body mode
            if mode <= 1:
                normalized_mode = this_mode

            else:
                normalized_mode = (this_mode-np.min(this_mode))/(np.max(this_mode)-np.min(this_mode))

            s = np.arange(1, normalized_mode.size+1)

            former_node = 0
            f = 0

            for node in nodes_g:
                axs[mode//2].plot(s[former_node:node-f],
                                  np.real(normalized_mode[former_node:node-f]), color='blue')
                former_node = node - f
                f += 1

            axs[mode//2].plot(s[former_node:],
                              np.real(normalized_mode[former_node:]), color='blue')
            axs[mode//2].scatter(s, np.real(normalized_mode), color='blue')
            axs[mode//2].text(0.6, -0.9, 'Mode {:d}: {:.2f} Hz'.format(mode//2,
                                                                       np.abs(lam[inds[mode]])/(2*np.pi)))

            axs[mode//2].set_ylim(-1.1, 1.1)

        plt.xticks(s)
        plt.xlabel('Node number')
        plt.ylabel('Relative displacement', loc='center')
        # plt.ylabel('                         Relative displacement', loc='bottom')

        # plt.savefig('konsbergmodes.pdf')
        plt.show()

if __name__ == '__main__':

    # Rotor.plot_campbell('this', 50, [30, 40, 50], [1, 2, 3])

    assembly = Rotor([Shaft(0,1, 1, 30)], disk_elements=[Disk(0, 4), Disk(1, 3)])

    '''assembly.campbell_diagram()'''
