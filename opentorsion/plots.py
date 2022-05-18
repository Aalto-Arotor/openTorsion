import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import lti
from scipy.signal import lsim
from scipy import linalg as LA


class Plots:
    """This class contains plot and figure tools"""

    def __init__(self, assembly):
        self.assembly = assembly

    def campbell_diagram(self, num_modes=10, frequency_range=100):
        """Campbell diagram of the powertrain"""
        omegas_damped, freqs, damping_ratios = self.assembly.modal_analysis()
        freqs = freqs[:num_modes]
        freqs = freqs[::2]

        self.plot_campbell(frequency_range, freqs)

        return

    def plot_campbell(self, frequency_range, modes, excitations=[1, 2, 3, 4]):
        """Plots the campbell diagram"""
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

    def figure_eigenmodes(self, modes=5, l=0):
        """Plots the eigenmodes of the powertrain"""
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
            if mode <= 1:
                normalized_mode = this_mode

            else:
                normalized_mode = (this_mode - np.min(this_mode)) / (
                    np.max(this_mode) - np.min(this_mode)
                )

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
