import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
import matplotlib.patches as patches


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

    def plot_campbell(self,
                      frequency_range_rpm=[0, 1000],
                      num_modes=5,
                      harmonics=[1, 2, 3, 4],
                      operating_speeds_rpm=[]):
        """
        Plots the Campbell diagram

        Parameters
        ----------
        frequency_range : list, optional
            Analysis frequency range, default is 0 to 100 Hz
        num_modes : int, optional
            Number of modes to be plotted
        harmonics : list, optional
                List containing the harmonic multipliers
            """
        fig, ax = plt.subplots()

        # Operating speeds
        for i, operating_speed_rpm in enumerate(operating_speeds_rpm):
            ax.plot([operating_speed_rpm, operating_speed_rpm],
                    [0, harmonics[-1] * (frequency_range_rpm[1] + 50)/60],
                    "--",
                    color="red")
            rectangle = patches.Rectangle(
                (operating_speed_rpm*0.9, 0),
                operating_speed_rpm*0.2,
                harmonics[-1] * (frequency_range_rpm[1] + 50)/60,
                color='blue',
                alpha=0.2)
            ax.add_patch(rectangle)

        harmonics = sorted(harmonics)

        undamped_nf, damped_nf, damping_ratios = self.assembly.modal_analysis()
        freqs = undamped_nf[::2]/(2*np.pi)
        freqs = freqs[:num_modes]

        # Natural frequencies
        for i, freq in enumerate(freqs):
            ax.plot(frequency_range_rpm, [freq, freq], color="black",
                    label=f"$f_{i}$={freq.round(2)} Hz")
            ax.text(1.01*frequency_range_rpm[1], freq, f"$f_{i}$")

        # Diagonal lines
        for i, harmonic in enumerate(harmonics):
            ax.plot(frequency_range_rpm,
                    [0, harmonic * (frequency_range_rpm[1] + 50)/60],
                    color="blue")
            ax.text(
                0.90 * frequency_range_rpm[1],
                0.95 * (frequency_range_rpm[1] + 50) * harmonic / 60,
                f"{harmonic}x"
            )
        ax.legend(loc='upper left')
        ax.set_xlim(frequency_range_rpm)
        ax.set_xlabel("Excitation frequency (rpm)")
        ax.set_ylabel("Natural Frequency (Hz)")
        plt.show()

        return

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
