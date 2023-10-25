import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import lti
from scipy.signal import lsim
from scipy import linalg as LA


class Plots:
    """
    This class includes plotting functions

    Attributes
    ----------
    assembly : openTorsion Assembly class instance
    """

    def __init__(self, assembly):
        """
        Parameters
        ----------
        assembly : openTorsion Assembly class instance
        """

        self.assembly = assembly

    def campbell_diagram(self, num_modes=10, frequency_range=100):
        """
        Creates a Campbell diagram

        Parameters
        ----------
        num_modes : int, optional
            Number of modes to be plotted, default is 10
        frequency_range : int, optional
            Analysis frequency range, default is 100 Hz
        """

        omegas_damped, freqs, damping_ratios = self.assembly.modal_analysis()
        freqs = freqs[:num_modes]
        freqs = freqs[::2]

        self.plot_campbell(frequency_range, freqs)

        return

    def plot_campbell(self, frequency_range, modes, excitations=[1, 2, 3, 4]):
        """
        Plots the Campbell diagram

        Parameters
        ----------
        frequency_range : int
            Analysis frequency range
        modes : int
            Number of modes to be plotted
        excitations : list, optional
            List containing the numbers of harmonics, default is 1 through 4
        """

        plt.rcParams.update(
            {"text.usetex": False, "font.serif": ["Computer Modern Roman"]}
        )
        legend, freq_num = [], [1, 2, 3, 4, 5]
        for i, v in enumerate(modes):
            plt.plot([0, frequency_range], [v, v], color="black")
            legend.append("$f_{i}$={v} Hz".format(i=freq_num[i], v=v.round(2)))
            plt.text(
                (i * 0.1 + 0.3) * frequency_range,
                v + 0.07 * frequency_range,
                "$f_{i}$".format(i=freq_num[i]),
            )
        plt.legend(legend)

        for i, v in enumerate(excitations):
            plt.plot([0, frequency_range], [0, v * (frequency_range + 50)], color="C0")
            plt.text(
                0.90 * frequency_range,
                0.95 * (frequency_range + 50) * v,
                "{v}x".format(v=v),
            )

        plt.xlim([0, frequency_range])
        plt.xlabel("Excitation frequency (Hz)")
        plt.ylabel("Natural Frequency (Hz)")
        plt.show()

        return
    
    def plot_eigenmodes(self, modes=5):
        """
        Updated eigenmode plot. Geared systems not supported.
        """
        eigenmodes = self.assembly.eigenmodes()
        phases = np.angle(eigenmodes)
        nodes = np.arange(0, self.assembly.dofs)

        fig_modes, axs = plt.subplots(modes, 1, sharex=True)

        for i in range(modes):
            # eigenvector corresponding to mode i
            eigenvector = eigenmodes[:,i]
            # find node with largest displacement
            max_disp = np.argmax(np.abs(eigenvector))
            # the system is rotated so that the imaginary component is zero at the node with max. displacement
            eigenvector_rotated = eigenvector * np.exp(-1.0j*phases[max_disp,i])
            # plot eigenvector
            self.plot_on_ax(self.assembly,axs[i],alpha=0.2)
            axs[i].plot(nodes, np.real(eigenvector_rotated)/np.sqrt(np.sum(np.real(eigenvector_rotated)**2)),color='red')
            #axs[i].plot(nodes, -np.real(eigenvector_rotated)/np.sqrt(np.sum(np.real(eigenvector_rotated)**2)),'--',color='red',alpha=0.6)
            axs[i].plot([nodes,nodes],[np.abs(eigenvector_rotated),-np.abs(eigenvector_rotated)],'--',color='black')
            axs[i].set_ylim([-1.1,1.1])
        plt.show()

    def figure_eigenmodes(self, modes=5):
        """
        Plots the eigenmodes of the powertrain

        Parameters
        ----------
        modes : int, optional
            Number of modes to be plotted, default is 5
        """

        fig_modes, axs = plt.subplots(modes, 1, sharex=True)
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
            this_mode = np.abs(this_mode[-this_mode.size // 2 :])

            # Do not normalize rigid body mode
            normalized_mode = this_mode / LA.norm(this_mode)

            s = np.arange(1, normalized_mode.size + 1)

            # At the moment, there is no discontinuity at gear nodes in the plot
            axs[mode // 2].plot(s, np.real(normalized_mode), color="C0")
            axs[mode // 2].scatter(s, np.real(normalized_mode), color="C0")
            axs[mode // 2].text(
                0.6,
                -0.9,
                "Mode {:d}: {:.2f} Hz".format(
                    mode // 2, np.abs(lam[inds[mode]]) / (2 * np.pi)
                ),
            )

            axs[mode // 2].set_ylim(-1.1, 1.1)

        plt.rcParams.update(
            {"text.usetex": False, "font.serif": ["Computer Modern Roman"]}
        )
        plt.xticks(s)
        plt.xlabel("Node number")
        # plt.ylabel('Relative displacement', loc='center')
        plt.ylabel("                         Relative displacement", loc="bottom")

        plt.show()

        return

    def torque_response_plot(self, omegas, T, show_plot=False, save=False):
        """
        Plots forced response amplitude as a function of rotational speed.

        Parameters:
        -----------
        omegas : ndarray
            Drivetrain rotational speed in rad/s
        T : ndarray
            Drivetrain response amplitudes in Nm
        show_plot : bool, optional
            If True, plot is shown
        save : bool
            If True, plot is saved to a pdf, show_plot must be False if save == True
        """

        c = np.pi / 30

        ax1 = plt.subplot(211)
        ax1.plot(omegas, T[0] * 1 / 1000, label="Shaft 1")
        ax1.legend()
        plt.ylabel("Amplitude (kNm)", loc="center")
        plt.grid()

        ax2 = plt.subplot(212)
        ax2.plot(omegas, T[1] * (1 / 1000), label="Shaft 2")
        ax2.legend()
        plt.ylabel("Amplitude (kNm)", loc="center")
        plt.xlabel("$\omega$ (RPM)")
        plt.grid()

        if show_plot:
            plt.show()
        if save:
            plt.savefig("response.pdf")

    def plot_assembly(self, assembly=None):
        """
        Plots the given assembly as disk and spring elements

        Parameters:
        -----------
        assembly : openTorsion Assembly class instance
        """
        assembly = self.assembly
        fig, ax = plt.subplots(figsize=(5,4))
        self.plot_on_ax(assembly, ax)

        ax.set_xticks(np.arange(0, assembly._check_dof(), step=1))
        ax.set_xlabel('node')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()
        plt.show()
        return


    def plot_on_ax(self, assembly, ax, alpha=1):

        """
        Plots disk and spring elements

        Parameters:
        -----------
        assembly : openTorsion Assembly class instance
            
        ax : matplotlib Axes class instance
            The Axes where the elements are plotted
        alpha : float, optional
            Adjust the opacity of the plotted elements
        """

        disks = assembly.disk_elements
        shafts = assembly.shaft_elements

        max_I_disk = max(disks, key=lambda disk: disk.I)
        min_I_disk = min(disks, key=lambda disk: disk.I)
        max_I_value = max_I_disk.I
        min_I_value = min_I_disk.I

        disk_max, disk_min = 2, 0.5
        width = 0.5

        num_segments = 6 # number of lines in a spring
        amplitude = 0.1  # spring "height"

        # plot springs connecting the disk elements
        for i, shaft in enumerate(shafts):
            if i < len(shafts):
                x1, y1 = shaft.nl+width/2, 0
                x2, y2 = shaft.nr-width/2, 0
                
                x_values = np.linspace(x1, x2, num_segments)
                y_values = np.linspace(y1, y2, num_segments)

                for i in range(0, num_segments):
                    if i % 2 == 0:
                        y_values[i] += amplitude
                    else:
                        y_values[i] -= amplitude

                for i in range(num_segments - 1):
                    ax.plot(x_values[i:i+2], y_values[i:i+2], color='k', alpha=alpha)

        # plot disk elements
        for i, disk in enumerate(disks):
            node = disk.node
            height = (disk.I - min_I_value) / (max_I_value - min_I_value) * (disk_max - disk_min) + disk_min
            pos = (node-width/2, -height/2)

            ax.add_patch(matplotlib.patches.Rectangle(pos, width, height, fill=True, edgecolor='black', facecolor='darkgrey', linewidth=1.5, alpha=alpha))
        return