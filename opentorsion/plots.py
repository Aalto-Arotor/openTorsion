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


class Fig_2D:
    """Creates a 2D-figure of the powertrain"""

    def __init__(self, assembly):
        self.assembly = assembly
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.x0, self.y0 = 1, 0
        self.l = 1  # block_length
        self.h = 0.1  # block height
        self.gear_nodes = []

    def draw_gears(self):
        gear_stages = []

        for element in self.assembly.gear_elements:
            if element.stages is not None:
                self.gear_nodes.append(element.node)
                gear_stages.append(element.stages)

        for i, node in enumerate(gear_stages):
            plt.text(self.x0 + node[0][0][0], self.y0 - i * self.h, str(node[0][0][0]))
            plt.text(
                self.x0 + node[0][0][0] + self.h,
                self.y0 - self.h * (1 / 2 + i),
                "Ratio: " + str(abs(node[0][0][1])) + ":" + str(abs(node[0][1][1])),
                color="red",
            )

            self.ax.add_patch(
                patch.Rectangle(
                    (self.x0 + node[0][0][0], self.y0 - i * self.h),
                    self.l,
                    -self.h,
                    ec="black",
                    fc="green",
                )
            )
            self.ax.add_patch(
                patch.Rectangle(
                    (self.x0 + node[0][0][0], self.y0 - self.h * (1 + i)),
                    self.l,
                    -self.h,
                    ec="black",
                    fc="green",
                )
            )

    def draw_shafts(self):
        shaft_nodes = []

        for element in self.assembly.shaft_elements:
            shaft_nodes.append(element.nl)
        shaft_nodes.append(shaft_nodes[-1] + 1)
        print(shaft_nodes)

        a = 0  # to level shafts accroding to gears
        for node in shaft_nodes:
            if node in self.gear_nodes:
                a += self.h
            if node != shaft_nodes[-1]:
                plt.text(
                    self.x0 + node, self.y0 - a, str(node)
                )  # add node number as text
                self.ax.add_patch(
                    patch.Rectangle(
                        (self.x0 + node, self.y0 - a), self.l, -self.h, ec="black"
                    )
                )
            else:
                plt.text(self.x0 + node, self.y0 - a, str(node))

    def draw_disks(self):
        a = self.h * self.l
        b = 0

        for element in self.assembly.disk_elements:
            x = self.x0 + element.node

            for gear_node in self.gear_nodes:
                print("gear node:", gear_node)
                print("element node:", element.node)
                if element.node > gear_node:
                    b += 1

            c = b * self.h
            d = self.h + c

            polygon_edges = [
                (x, self.y0 - c),
                (x + 2 * a, self.y0 + a / 2 - c),
                (x + 2 * a, self.y0 + a - c),
                (x - 2 * a, self.y0 + a - c),
                (x - 2 * a, self.y0 + a / 2 - c),
            ]

            negative_edges = [
                (x, self.y0 - d),
                (x + 2 * a, self.y0 - a / 2 - d),
                (x + 2 * a, self.y0 - a - d),
                (x - 2 * a, self.y0 - a - d),
                (x - 2 * a, self.y0 - a / 2 - d),
            ]

            self.ax.add_patch(patch.Polygon(polygon_edges, ec="black", fc="black"))
            self.ax.add_patch(patch.Polygon(negative_edges, ec="black", fc="black"))

    def draw_figure(self):
        x_axis, y_axis = self.ax.get_xaxis(), self.ax.get_yaxis()
        x_axis.set_visible(False)
        y_axis.set_visible(False)

        if self.assembly.gear_elements is not None:
            self.draw_gears()

        if self.assembly.shaft_elements is not None:
            self.draw_shafts()

        if self.assembly.disk_elements is not None:
            self.draw_disks()

        plt.rcParams.update(
            {"text.usetex": False, "font.serif": ["Computer Modern Roman"]}
        )

        plt.xlim(right=self.x0 + int(self.assembly._check_dof()))
        plt.ylim(top=self.y0 + self.l / 2, bottom=self.y0 - self.l)
        plt.show()
