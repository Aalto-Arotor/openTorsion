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

    def figure_2D(self):
        """Creates a 2D-figure of the powertrain"""
        # TODO: if shaft masses are identical, v==nan --> figure broken
        # TODO: disks should be drawn last as coordinates may not be correct due to varying shaft size
        # TODO: correct shaft placement when amount of consecutive gears is > 2

        fig, ax = plt.subplots(nrows=1, ncols=1)
        x_axis, y_axis = ax.get_xaxis(), ax.get_yaxis()
        x_axis.set_visible(False)
        y_axis.set_visible(True)
        shaft_mass, disk_mass, nodes_sl, nodes_sr, nodes_d, nodes_g, disk_on_gear = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        plt.rcParams.update(
            {"text.usetex": False, "font.serif": ["Computer Modern Roman"]}
        )

        # Appends nodes to lists
        if self.assembly.shaft_elements is not None:
            for element in self.assembly.shaft_elements:
                shaft_mass.append(element.mass)
                nodes_sl.append(element.nl)
                nodes_sr.append(element.nr)
            if nodes_sr != []:
                nodes_sl.append(nodes_sr[-1])  # nodes_sl contains all nodes

        if self.assembly.disk_elements is not None:
            for element_d in self.assembly.disk_elements:
                disk_mass.append(element_d.I)
                nodes_d.append(element_d.node)

        if self.assembly.gear_elements is not None:
            for element_g in self.assembly.gear_elements:
                nodes_g.append(element_g.node)

        # Axis limits and starting point
        max_nodes = len(nodes_sl) + 1 + len(nodes_g) / 2
        x0 = 0.5  # figure start coordinates (x0, y0)
        y0 = 1 / (3 * len(nodes_g) + 1)
        l = 1 / (2 * x0)  # length of a shaft/gear element
        h = 0.05 * l  # height of a shaft/gear element
        plt.xlim(right=max_nodes)
        plt.ylim(top=y0 + l / 2, bottom=y0 - l / 2)
        k = len(nodes_g) * h  # to level shaft and gear figures right
        m = 0.1 * l  # for node number text coordinate
        t = 0  # for gear block effect on x coordinate
        last_node = -1

        # Draws gear elements and possible disk elements
        for node in nodes_g:
            if node in nodes_d:
                polygon_edges = [
                    (x0 + node * l, y0 + k),
                    (x0 + node * l + 0.2 * l, y0 + k + 0.05 * l),
                    (x0 + node * l + 0.2 * l, y0 + k + 0.1 * l),
                    (x0 + node * l - 0.2 * l, y0 + k + 0.1 * l),
                    (x0 + node * l - 0.2 * l, y0 + k + 0.05 * l),
                ]  # coordinates for upper disk polygon

                neg_polygon_edges = [
                    (x0 + node * l, y0 + k - 2 * h),
                    (x0 + node * l + 0.2 * l, y0 + k - 0.05 * l - 2 * h),
                    (x0 + node * l + 0.2 * l, y0 + k - 0.1 * l - 2 * h),
                    (x0 + node * l - 0.2 * l, y0 + k - 0.1 * l - 2 * h),
                    (x0 + node * l - 0.2 * l, y0 + k - 0.05 * l - 2 * h),
                ]  # coordinates for lower disk polygon

                if node in nodes_d:
                    ax.add_patch(
                        matplotlib.patches.Polygon(
                            polygon_edges, ec="black", fc="black"
                        )
                    )
                    # disk figure upper part
                    ax.add_patch(
                        matplotlib.patches.Polygon(
                            neg_polygon_edges, ec="black", fc="black"
                        )
                    )
                    # disk figure lower part

            if node == last_node + 1:
                t += 1
            else:
                plt.text(
                    x0 - t + node * l + m, y0 + k + m * 0.1, str(node)
                )  # add node number as text
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x0 - t + node * l, y0 + k), l, -h, ec="black", fc="green"
                )
            )
            k -= h
            last_node = node

        k = len(nodes_g) * h

        # Draws shaft and possible disk elements
        for i, node in enumerate(nodes_sl):
            if node in nodes_g:
                k -= h

            if i != len(nodes_sl) - 1:
                v = (
                    h
                    / 2
                    * (shaft_mass[i] - np.min(shaft_mass))
                    / (np.max(shaft_mass - np.min(shaft_mass)))
                )

            plt.text(
                x0 + node * l + m, y0 + k + m * 0.1 + v / 2, str(node)
            )  # add node number as text

            if node != nodes_sl[-1]:
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (x0 + node * l, y0 + k + v / 2), l, -h - v, ec="black"
                    )
                )  # shaft figure

            upper_polygon = [
                (x0 + node * l, y0 + k + v),
                (x0 + node * l + 0.2 * l, y0 + k + 0.05 * l + v),
                (x0 + node * l + 0.2 * l, y0 + k + 0.1 * l + v),
                (x0 + node * l - 0.2 * l, y0 + k + 0.1 * l + v),
                (x0 + node * l - 0.2 * l, y0 + k + 0.05 * l + v),
            ]  # coordinates for upper disk polygon

            lower_polygon = [
                (x0 + node * l, y0 + k - h - v),
                (x0 + node * l + 0.2 * l, y0 + k - 0.05 * l - h - v),
                (x0 + node * l + 0.2 * l, y0 + k - 0.1 * l - h - v),
                (x0 + node * l - 0.2 * l, y0 + k - 0.1 * l - h - v),
                (x0 + node * l - 0.2 * l, y0 + k - 0.05 * l - h - v),
            ]  # coordinates for lower disk polygon
            if node in nodes_d:
                ax.add_patch(
                    matplotlib.patches.Polygon(upper_polygon, ec="black", fc="black")
                )
                # disk figure upper part
                ax.add_patch(
                    matplotlib.patches.Polygon(lower_polygon, ec="black", fc="black")
                )
                # disk figure lower part
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
            print(this_mode)

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


class fig_2D:
    def init(self, assembly):
        self.assembly = assembly

    def draw_gear(self):
        pass

    def draw_shaft(self):
        pass

    def draw_disk(self):
        pass
