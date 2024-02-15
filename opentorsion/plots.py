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
        lam, eigenmodes = self.assembly.eigenmodes()
        phases = np.angle(eigenmodes)
        nodes = np.arange(0, self.assembly.dofs)

        fig_modes, axs = plt.subplots(modes, 1, sharex=True)

        for i in range(modes):
            eigenvector = eigenmodes[:, i]
            max_disp = np.argmax(np.abs(eigenvector))
            eigenvector_rotated = eigenvector * np.exp(-1.0j*phases[max_disp, i])
            self.plot_on_ax(self.assembly,axs[i], alpha=0.2)
            axs[i].plot(nodes, np.real(eigenvector_rotated)/np.sqrt(np.sum(np.real(eigenvector_rotated)**2)),color='red')
            axs[i].plot([nodes,nodes],[np.abs(eigenvector_rotated),-np.abs(eigenvector_rotated)],'--',color='black')
            axs[i].set_ylim([-1.1,1.1])
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
        Note: doesn't work with assemblies that have gears
        """
        fig, ax = plt.subplots(figsize=(5, 4))
        self.plot_on_ax(self.assembly, ax)
        ax.set_xticks(np.arange(0, self.assembly.dofs, step=1))
        ax.set_xlabel('node')
        # ax.set_yticks([])
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
        gears = assembly.gear_elements
        max_I_disk = max(disks, key=lambda disk: disk.I)
        min_I_disk = min(disks, key=lambda disk: disk.I)
        max_I_value = max_I_disk.I
        min_I_value = min_I_disk.I
        disk_max, disk_min = 2, 0.5
        width = 0.5
        num_segments = 6  # number of lines in a spring
        amplitude = 0.1  # spring "height"

        def draw_spring(left, right, y_pos):
            x1, y1 = left+width/2, -2*y_pos
            x2, y2 = right-width/2, -2*y_pos
            x_values = np.linspace(x1, x2, num_segments)
            y_values = np.linspace(y1, y2, num_segments)
            for i in range(0, num_segments):
                if i % 2 == 0:
                    y_values[i] += amplitude
                else:
                    y_values[i] -= amplitude
            for i in range(num_segments - 1):
                ax.plot(x_values[i:i+2], y_values[i:i+2], color='k', alpha=alpha)

        def draw_disk(disk, pos, i, color='darkgrey'):
            node = pos
            height = (disk.I - min_I_value)/(max_I_value - min_I_value)*(disk_max - disk_min) + disk_min
            pos = (node-width/2, -height/2-2*i)
            ax.add_patch(patches.Rectangle(pos, width, height, fill=True, edgecolor='black', facecolor=color, linewidth=1.5, alpha=alpha))

        def draw_gear(gear, pos, i):
            node = pos
            height = (gear.I - min_I_value)/(max_I_value - min_I_value)*(disk_max - disk_min) + disk_min
            pos = (node-width/2, -height/2-2*(i))
            ax.add_patch(patches.Rectangle(pos, width, height, fill=True, edgecolor='black', facecolor='red', linewidth=1.5, alpha=alpha))

        for i,shaft in enumerate(shafts):
            print(f'list index: {i}, shaft nodes: {shaft.nl}, {shaft.nr}')

        gear_pos = {}
        for gear in gears:
            gear_pos[gear.node] = [gear, [gear.node, 0]]
        disk_pos = {}
        for disk in disks:
            disk_pos[disk.node] = disk
        print(f'gear positions: {gear_pos}')
        # print(f'gear positions: {disk_pos}')
        parent_gear = None
        prev_nr = 0
        y_height = 0
        for i, shaft in enumerate(shafts):
            if shaft.nl == prev_nr:
                draw_spring(shaft.nl, shaft.nr, y_height)
                prev_nr = shaft.nr
            else:
                y_height+=1
                draw_spring(shaft.nl, shaft.nr, y_height)
                prev_nr = shaft.nr
                
            if shaft.nl in gear_pos:
                draw_disk(gear_pos[shaft.nl][0], shaft.nl, y_height, 'red')
                gear_pos[shaft.nl][1] = [shaft.nl, -2*y_height]
            else:
                draw_disk(disk_pos[shaft.nl], shaft.nl, y_height)

            if shaft.nr in gear_pos:
                draw_disk(gear_pos[shaft.nr][0], shaft.nr, y_height, 'red')
                gear_pos[shaft.nr][1] = [shaft.nr, -2*y_height]
            else:
                draw_disk(disk_pos[shaft.nr], shaft.nr, y_height)

        # draw dashedlines
        print(f'updated gear positions: {gear_pos}')
        for node, gear_and_pos in gear_pos.items():
            gear = gear_and_pos[0]
            pos = gear_and_pos[1]
            if gear.stages is None:
                pass
            else:
                print([pos[0],gear.stages[0][0][0]])
                print(gear_pos[gear.stages[0][0][0]][1][1])
                plt.plot([pos[0],gear.stages[0][0][0],gear.stages[0][0][0]],[pos[1],pos[1],gear_pos[gear.stages[0][0][0]][1][1]],'k--', zorder=-1)




        # for i, shaft in enumerate(shafts):
        #     if shaft.nl in gear_pos:
        #         for gear in gears:
        #             if shaft.nl == gear.node:
        #                 if gear.stages == None:
        #                     draw_spring(shaft.nl, shaft.nr, 0)
        #                     draw_disk(gear, gear.node, 0, 'red')
        #                     parent_gear = gear
        #                 else:
        #                     parent_pos = gear.stages[0][0][0]
        #                     draw_spring(parent_pos, parent_pos+1, g)
        #                     draw_disk(gear, parent_pos, g, 'red')
        #                     level.append(shaft.nr)
        #                     g+=1
        #     elif shaft.nl in level:
        #         level.append(shaft.nr)
        #     else:
        #         draw_spring(shaft.nl, shaft.nr, 0)
        # print(f'level: {level}')
        # i=0
        # j=1
        # prev_level = level[0]
        # for disk in disks:
        #     if disk.node in level:
        #         if len(level) >= 2:
        #             print(f'lasku: {level[0]}-{prev_level}={level[0]-prev_level}')
        #             print(f'level: {level}')
        #             if level[0]-prev_level > 1:
        #                 j+=1
        #                 i=0
        #             draw_spring(parent_gear.node+1+i, parent_gear.node+2+i, j)
        #         draw_disk(disk, parent_gear.node+1+i, j)
        #         prev_level = level[0]
        #         level.pop(0)
        #         i+=1
        #     else:
        #         draw_disk(disk, disk.node, 0)

        # for i, disk in enumerate(disks):
        #     draw_disk(disk, disk.node)
        
        # for i, gear in enumerate(gears):
        #     node = gear.node
        #     height = (gear.I - min_I_value)/(max_I_value - min_I_value)*(disk_max - disk_min) + disk_min
        #     pos = (node-width/2, -height/2-2*(i))
        #     ax.add_patch(patches.Rectangle(pos, width, height, fill=True, edgecolor='black', facecolor='red', linewidth=1.5, alpha=alpha))
        
        return
