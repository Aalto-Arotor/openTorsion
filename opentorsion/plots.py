import numpy as np
import matplotlib.pyplot as plt
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

    def plot_campbell(
        self,
        frequency_range_rpm=[0, 1000],
        num_modes=5,
        harmonics=[1, 2, 3, 4],
        harmonic_labels=[],
        operating_speeds_rpm=[],
    ):
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
        harmonic_labels : list, optional
                List containing labels for the harmonic lines
        """
        fig, ax = plt.subplots()

        # Operating speeds
        for i, operating_speed_rpm in enumerate(operating_speeds_rpm):
            ax.plot(
                [operating_speed_rpm, operating_speed_rpm],
                [0, harmonics[-1] * (frequency_range_rpm[1] + 50) / 60],
                "--",
                color="red",
            )
            rectangle = patches.Rectangle(
                (operating_speed_rpm * 0.9, 0),
                operating_speed_rpm * 0.2,
                harmonics[-1] * (frequency_range_rpm[1] + 50) / 60,
                color="blue",
                alpha=0.2,
            )
            ax.add_patch(rectangle)

        harmonics = sorted(harmonics)

        undamped_nf, damped_nf, damping_ratios = self.assembly.modal_analysis()
        freqs = undamped_nf[::2] / (2 * np.pi)
        freqs = freqs[1:num_modes]

        # Natural frequencies
        for i, freq in enumerate(freqs):
            ax.plot(
                frequency_range_rpm,
                [freq, freq],
                color="black",
                label=f"$f_{i+1}$={freq.round(2)} Hz",
            )
            ax.text(1.01 * frequency_range_rpm[1], freq, f"$f_{i+1}$")

        # Diagonal lines
        for i, harmonic in enumerate(harmonics):
            ax.plot(
                frequency_range_rpm,
                [0, harmonic * (frequency_range_rpm[1]) / 60],
                color="blue",
            )
            if harmonic_labels:
                ax.text(
                    0.90 * frequency_range_rpm[1],
                    0.95 * harmonic * (frequency_range_rpm[1]) / 60,
                    harmonic_labels[i],
                    bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0')
                )
            else:
                ax.text(
                    0.90 * frequency_range_rpm[1],
                    0.95 * harmonic * (frequency_range_rpm[1]) / 60,
                    f"{harmonic}x",
                )
        ax.legend(loc="upper left")
        ax.set_xlim(frequency_range_rpm)
        ax.set_xlabel("Excitation frequency (rpm)")
        ax.set_ylabel("Natural Frequency (Hz)")
        plt.show()
        return

    def plot_eigenmodes(self, modes=5):
        """
        Updated eigenmode plot. Geared systems not supported.
        The eigenvectors are plotted over the assembly schematic, and the
        trajectories are plotted with dashed lines. Each plotted eigenvector is
        rotated so that the node with maximum abs displacement has phase of 0

        Parameters
        ----------
        modes : int
            Number of eigenodes to be plotted
        """
        if self.assembly.gear_elements is not None:
            raise NotImplementedError("Support for geared assemblies not implemented")
        if self.assembly.dofs < modes:
            modes = self.assembly.dofs
        lam, eigenmodes = self.assembly.eigenmodes()
        phases = np.angle(eigenmodes)
        nodes = np.arange(0, self.assembly.dofs)

        fig_modes, axs = plt.subplots(modes, 1, sharex=True)

        for i in range(modes):
            eigenvector = eigenmodes[:, i]
            max_disp = np.argmax(np.abs(eigenvector))
            eigenvector_rotated = eigenvector * np.exp(-1.0j * phases[max_disp, i])
            self.plot_on_ax(self.assembly, axs[i], lighter=True)
            axs[i].plot(
                nodes,
                np.real(eigenvector_rotated)
                / np.sqrt(np.sum(np.real(eigenvector_rotated) ** 2)),
                color="red",
            )
            axs[i].plot(
                [nodes, nodes],
                [np.abs(eigenvector_rotated), -np.abs(eigenvector_rotated)],
                "--",
                color="black",
            )
            axs[i].set_ylim([-1.1, 1.1])
        plt.show()

    def torque_response_plot(self, omegas, T, show_plot=False):
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
        """

        ax1 = plt.subplot(211)
        ax1.plot(omegas, T[0] * 1 / 1000, label="Shaft 1")
        ax1.legend()
        plt.ylabel("Amplitude (kNm)", loc="center")
        plt.grid()

        ax2 = plt.subplot(212)
        ax2.plot(omegas, T[1] * (1 / 1000), label="Shaft 2")
        ax2.legend()
        plt.ylabel("Amplitude (kNm)", loc="center")
        plt.xlabel(r"$\omega$ (RPM)")
        plt.grid()

        if show_plot:
            plt.show()

    def plot_assembly(self):
        """
        Plots the given assembly as disk and spring elements
        """
        fig, ax = plt.subplots(figsize=(5, 4))
        self.plot_on_ax(self.assembly, ax)
        ax.set_xticks(np.arange(0, self.assembly.dofs, step=1))
        ax.set_xlabel("node")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()
        plt.show()
        return

    def plot_on_ax(self, assembly, ax, lighter=False):
        """
        Plots disk and spring elements

        Parameters:
        -----------
        assembly : openTorsion Assembly class instance

        ax : matplotlib Axes class instance
            The Axes where the elements are plotted
        lighter : boolean, optional
            Change the opacity of the plotted elements
        """
        disks = assembly.disk_elements
        shafts = assembly.shaft_elements
        if assembly.gear_elements is not None:
            gears = assembly.gear_elements
        else:
            gears = []
        max_I_disk = max(disks, key=lambda disk: disk.I)
        min_I_disk = min(disks, key=lambda disk: disk.I)
        max_I_value = max_I_disk.I
        min_I_value = min_I_disk.I
        disk_max, disk_min = 1.8, 0.5
        width = 0.5
        num_segments = 8  # number of lines in a spring
        amplitude = 0.1  # spring "height"
        if lighter:
            gear_face_color = 'red'
            disk_face_color = 'lightgrey'
            disk_edge_color = 'darkgrey'
            spring_color = 'darkgrey'
            dashpot_color = 'darkgrey'
        else:
            gear_face_color = 'red'
            disk_face_color = 'darkgrey'
            disk_edge_color = 'black'
            spring_color = 'black'
            dashpot_color = 'black'
        lw = 1.5
        disk_lw = lw
        spring_lw = lw
        dashpot_lw = lw

        def draw_spring(shaft, y_pos):
            left  = shaft.nl
            right = shaft.nr
            if shaft.c != 0:
                draw_dashpot(((left + right) / 2, -2 * y_pos + amplitude * 1.5), 2 * amplitude, right - left - width)
                y_pos += amplitude*1.5/2
            x1, y1 = left  + width / 2, -2 * y_pos
            x2, y2 = right - width / 2, -2 * y_pos
            seg_len = (x2-x1) / num_segments # length of a spring segment
            x_values = np.linspace(x1 + 1.5 * seg_len, x2 - 1.5 * seg_len, num_segments - 2)
            x_values = np.insert(x_values,  0, x1)
            x_values = np.insert(x_values,  1, x1 + seg_len)
            x_values = np.append(x_values, x2 - seg_len)
            x_values = np.append(x_values, x2)
            y_values = np.linspace(y1, y2, num_segments+2)
            for i in range(2, len(y_values)-2):
                if i % 2 == 0:
                    y_values[i] += amplitude
                else:
                    y_values[i] -= amplitude
            ax.plot(
                x_values, y_values, color=spring_color, linewidth=spring_lw
            )

        def draw_disk(disk, i, color="darkgrey"):
            if max_I_value == min_I_value:
                height = disk_max
            else:
                height = disk_min + (disk.I - min_I_value) * (disk_max - disk_min) / (
                    max_I_value - min_I_value) 
            pos = (disk.node - width / 2, -height / 2 - 2 * i)
            ax.add_patch(
                patches.Rectangle(
                    pos,
                    width,
                    height,
                    fill=True,
                    edgecolor=disk_edge_color,
                    facecolor=color,
                    linewidth=disk_lw)
            )

        def draw_dashpot(center, height, width):
            ax.plot([center[0] - width / 2, center[0] - width / 4],
                    [center[1], center[1]],
                    color=dashpot_color, lw=dashpot_lw)
            ax.plot([center[0] - width / 4, center[0] - width / 4],
                    [center[1] - height / 2, center[1] + height / 2],
                    color=dashpot_color, lw=dashpot_lw)
            ax.plot([center[0] - width / 4, center[0] + width / 4],
                    [center[1] + height / 2, center[1] + height / 2],
                    color=dashpot_color, lw=dashpot_lw)
            ax.plot([center[0] - width / 4, center[0] + width / 4],
                    [center[1] - height / 2, center[1] - height / 2],
                    color=dashpot_color, lw=dashpot_lw)
            ax.plot([center[0], center[0]],
                    [center[1] - height / 2, center[1] + height / 2],
                    color=dashpot_color, lw=dashpot_lw)
            ax.plot([center[0], center[0] + width / 2],
                    [center[1], center[1]],
                    color=dashpot_color, lw=dashpot_lw)

        gear_pos = {}
        for gear in gears:
            gear_pos[gear.node] = [gear, [gear.node, 0]]
        disk_pos = {}
        for disk in disks:
            disk_pos[disk.node] = disk

        prev_nr = 0
        y_height = 0
        for i, shaft in enumerate(shafts):
            if shaft.nl == prev_nr:
                draw_spring(shaft, y_height)
                prev_nr = shaft.nr
            else:
                if shaft.nl-1 in gear_pos:
                    draw_disk(gear_pos[shaft.nl-1][0], y_height, gear_face_color)
                    gear_pos[shaft.nl-1][1] = [shaft.nl-1, -2 * y_height]
                elif shaft.nl-1 in disk_pos:
                    draw_disk(disk_pos[shaft.nl-1], y_height, disk_face_color)
                y_height += 1
                draw_spring(shaft, y_height)
                prev_nr = shaft.nr
            if shaft.nl in gear_pos:
                draw_disk(gear_pos[shaft.nl][0], y_height, gear_face_color)
                gear_pos[shaft.nl][1] = [shaft.nl, -2 * y_height]
            elif shaft.nl in disk_pos:
                draw_disk(disk_pos[shaft.nl], y_height, disk_face_color)
        shaft = shafts[-1]
        if shaft.nr in gear_pos:
            draw_disk(gear_pos[shaft.nr][0], y_height, gear_face_color)
            gear_pos[shaft.nr][1] = [shaft.nr, -2 * y_height]
        elif shaft.nr in disk_pos:
            draw_disk(disk_pos[shaft.nr], y_height, disk_face_color)
        # draw dashedlines connecting gear to parent gear
        for node, gear_and_pos in gear_pos.items():
            gear = gear_and_pos[0]
            pos = gear_and_pos[1]
            if gear.stages is None:
                pass
            else:
                plt.plot(
                    [pos[0], pos[0], gear.stages[0][0][0], gear.stages[0][0][0]],
                    [
                        pos[1],
                        pos[1] + 1,
                        pos[1] + 1,
                        gear_pos[gear.stages[0][0][0]][1][1],
                    ],
                    "k--",
                    zorder=-1)
        return
