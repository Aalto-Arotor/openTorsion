import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import lti
from scipy.signal import lsim
from scipy import linalg as LA


class Plots():
    def __init__(self, assembly):
        self.assembly = assembly

    def state_space(self, t_in, u1, u2=None):
        '''u1 & u2 are 1 x m shape excitation data arrays'''
        M, K, C  = self.assembly.M(), self.assembly.K(), self.assembly.C()

        Z = np.zeros(M.shape, dtype=np.float64)
        r1, c1 = M.shape
        I = np.eye(r1)

        A1 = LA.inv(-M) @ C
        A2 = LA.inv(-M) @ K

        A = np.vstack([
            np.hstack([A1, A2]),
            np.hstack([I, Z])
        ])

        M_inv = LA.inv(M)

        B = np.vstack([
            np.hstack([M_inv]),
            np.hstack([Z])
        ])

        r2, c2 = B.shape
        C = np.eye(r2)
        D = np.zeros(B.shape)

        system = lti(A, B, C, D)

        r3 = len(u1)
        c3 = c2-2
        if u2 == None:
            u2 = np.zeros((1, r3))
        U_zero = np.zeros((c2-2, r3))
        U = np.vstack([[u1], U_zero, [u2]])
        U = np.transpose(U) # Excitation data array,
        # u1 & u2 are located so that they affect the ends of the powertrain

        tout, yout, xout = lsim(system, U, t_in)

        torques, omegas, thetas = self.torque(yout)

        return tout, torques, omegas, thetas

    def torque(self, yout):
        '''Calculate torque between every node'''
        omegas, thetas = np.hsplit(yout, 2)
        thetas = np.abs(thetas)
        k_values, torques, row, nodes, ratios = [], [], [], [], []

        for element in self.assembly.shaft_elements:
            k_values.append(element.k)

        if self.assembly.gear_elements is not None:
            for element in self.assembly.gear_elements:
                if element.stages is not None:
                    ratios.append([element.stages[0][0][1], element.stages[0][1][1]])
                    nodes.append([element.stages[0][0][0], element.stages[0][1][0]])

        for theta in thetas:
            row = []
            deg = 0
            for i in range(len(k_values)):
                ratio = 1
                for j in range(len(nodes)):
                    if i == nodes[j][0]-deg:
                        ratio = np.abs(ratios[j][1])
                        deg += 1
                        break
                    else:
                        continue
                row.append((theta[i+1]-theta[i]/ratio)*k_values[i])
            torques.append(row)

        return torques, omegas, thetas

    def campbell_diagram(self, num_modes=4, frequency_range=100):
        omegas_damped, freqs, damping_ratios = self.assembly.modal_analysis()
        freqs = freqs[:num_modes]

        self.plot_campbell(frequency_range, freqs)

        return

    def plot_campbell(self, frequency_range, modes, excitations=[1, 2, 3, 4]):
        '''Plots the campbell diagram of the powertrain'''
        # fig = plt.figure()
        for i in modes:
            plt.plot([0, frequency_range], [i, i], color='black')

        for i in excitations:
            plt.plot([0, frequency_range], [0, i*frequency_range], color='C0')

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
        if self.assembly.shaft_elements is not None:
            for element in self.assembly.shaft_elements:
                shaft_mass.append(element.mass)
                nodes_sl.append(element.nl)
                nodes_sr.append(element.nr)
            if nodes_sr != []:
                nodes_sl.append(nodes_sr[-1]) # nodes_sl contains all nodes

        if self.assembly.disk_elements is not None:
            for element_d in self.assembly.disk_elements:
                disk_mass.append(element_d.I)
                nodes_d.append(element_d.node)

        if self.assembly.gear_elements is not None:
            for element_g in self.assembly.gear_elements:
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
        m = 0.1*l # for node number text coordinate

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
                plt.text(x0+node*l+m, y0+k+m*0.1, str(node)) # add node number as text
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
            plt.text(x0+node*l+m, y0+k+m*0.1, str(node)) # add node number as text

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

        return

    def figure_eigenmodes(self, modes=4):
        '''Plots eigenmodes of the powertrain'''
        fig_modes, axs = plt.subplots(4, 1, sharex=True)
        plt.ylim(-1.1, 1.1)

        A, B = self.assembly.state_matrix()
        lam, vec = self.assembly._eig(A, B)
        inds = np.argsort(np.abs(lam))

        nodes_g = []

        if self.assembly.gear_elements is not None:
            for element_g in self.assembly.gear_elements:
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

        plt.show()

        return
