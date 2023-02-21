"""
Tools for animating simulation data.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
import discretisedfield as df
import re
from MicromagneticAnalysisTools import Read
from MicromagneticAnalysisTools import Plot
from MicromagneticAnalysisTools import Calculate
plt.rcdefaults()  # Reset stylesheet imported by discretisedfield


def oneDQuantity(quantityArray, timeArray, yLabel, quantityTextValue, quantityTextFormat="{:.2f}", out_name="Movie.mp4", initialMinValue=0, initialMaxValue=0, fps=25):
    """ Outputs an animation of a generic quantity (e.g. Nsk, energy, current) over time. The input arrays are to be given as file names, e.g. Nsk.npy, Time.npy. """

    t = timeArray * 1e9

    fig, ax = plt.subplots()

    thePlot, = ax.plot(t[0], quantityArray[0], animated=True)

    plt.xlim(0, 0)
    plt.xlabel(r"$t$ (ns)")
    plt.ylabel(yLabel)

    minValue = initialMinValue
    maxValue = initialMaxValue

    quantityText = ax.text(0, maxValue, "")
    timeText = ax.text(0.05 * t[0], maxValue, "")

    def init():
        pass

    def updateAnim(i):

        print("Animating value", i, "of", len(quantityArray), end='\r')

        nonlocal maxValue
        nonlocal minValue

        try:
            if np.min(quantityArray[:i]) < minValue:
                minValue = np.min(quantityArray[:i])
            if np.max(quantityArray[:i]) > maxValue:
                maxValue = np.max(quantityArray[:i])

            quantityText.set_text(
                quantityTextValue + " = " + quantityTextFormat.format(quantityArray[i]))

            if maxValue == 0:  # Stop the texts being placed on top of one another if the energy is zero
                quantity_text_y = 0.02
                time_text_y = 0.03
            
            else:
                quantity_text_y = 0.3 * maxValue
                time_text_y = 0.5 * maxValue

            text_x = 0.05 * t[i]

            quantityText.set_position((text_x, quantity_text_y))
            timeText.set_position((text_x, time_text_y))

            timeText.set_text('t = ' + '{:.2f}'.format(t[i]) + ' ns')

        except ValueError:
            pass

        ax.set_xlim(t[0], t[i])
        ax.set_ylim(minValue, 1.1*maxValue)
        thePlot.set_data(t[:i+1], quantityArray[:i+1])
        ax.get_xaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: '{:.1f}'.format(x)))

    dataIterator = iter(range(len(quantityArray)))

    print("Producing animation", out_name)

    anim = animation.FuncAnimation(fig, updateAnim, dataIterator, init_func=init,
                                   blit=False, save_count=len(quantityArray))
    plt.tight_layout()

    anim.save(out_name, fps=fps, writer='ffmpeg')


def NskAnimation(NskArray, timeArray):
    oneDQuantity(NskArray, timeArray, r"$N_{\mathrm{sk}}$", r"$N_{\mathrm{sk}}$",
                 out_name="Nsk.mp4", initialMinValue=-1.1, initialMaxValue=1.1)


def currentAnimation(currentArray, timeArray, component):
    oneDQuantity(currentArray, timeArray, "$j_" + component + "$ (A m$^{-2}$)", "$j_" + component + "$",
                 quantityTextFormat="{:.2e}", out_name="Current" + component + ".mp4", initialMinValue=0, initialMaxValue=1)


def energyAnimation(energyArray, timeArray):
    oneDQuantity(energyArray, timeArray, "$E$ (J)", "$E$ (J)",
                 quantityTextFormat="{:.2e}", out_name="Energy.mp4", initialMinValue=energyArray[0], initialMaxValue=0.9 * energyArray[0])

class MagnetizationAnimator:

    def __init__(self,
    plot_type,
    directory,
    fig = None,
    ax = None,
    z_index = 0,
    com_array_file = None,
    component=None,
    show_time=False,
    time_units=None,
    plot_impurity=None,
    plot_pinning=None,
    show_component=False,
    interpolation=None,
    limits=None,
    length_units=None,
    step=1,
    quiver_colour=[1., 1., 1.],
    start_file = None,
    end_file = None,
    rectangle_fracs = None,
    out_name = None):
        self.plot_type = plot_type
        self.directory = directory
        self.fig = fig
        self.ax = ax
        self.z_index = z_index
        self.com_array_file = com_array_file
        self.component = component
        self.show_time = show_time
        self.time_units = time_units
        self.plot_impurity = plot_impurity
        self.plot_pinning = plot_pinning
        self.show_component = show_component
        self.interpolation = interpolation
        self.limits = limits
        self.length_units  = length_units
        self.step = step
        self.quiver_colour = quiver_colour
        self.start_file = start_file
        self.end_file = end_file
        self.rectangle_fracs = rectangle_fracs
        self.out_name = out_name

        self.files_to_scan = Read.getFilesToScan(self.directory, self.start_file, self.end_file)
        assert len(self.files_to_scan) != 0

        self.Lx, self.Ly = Read.sampleExtent(self.directory)
        self.m_array = df.Field.fromfile(self.directory + '/' + self.files_to_scan[0]).array
        if self.limits == None: self.limits = [0., self.Lx, 0., self.Ly]
        self.limits_indices = self._get_limits_indices()

        # Offset the centre of mass array index if we do not start the animation at the beginning of the simulation
        self.corrected_idx = int(re.findall(r'\d+', start_file)[0]) if self.start_file else 0

        if fig == None:
            if ax == None:
                self.fig, self.ax = plt.subplots()
            else:
                raise ValueError('If supplying an axis, also need to supply a figure.')

        # Ensure the centre of mass marker colour is easily visible with the colour scheme
        self.marker_colour = 'green' if 'magnetization' in self.plot_type else 'red'

        self.plotter = Plot.MagnetizationPlotter(self.plot_type, self.directory, self.files_to_scan[0],
        ax=self.ax, z_index=self.z_index, component=self.component, plot_impurity=self.plot_impurity,
        plot_pinning=self.plot_pinning, show_component=self.show_component, interpolation=self.interpolation,
        limits=self.limits, length_units=self.length_units, step=self.step)

        self.magnetization_plot = self.plotter.plot()

        if self.com_array_file:
            self.com_array = np.load(self.com_array_file)
            self.com_marker = Circle(
                (self.com_array[0, 0], self.com_array[0, 1]), 10, color=self.marker_colour, alpha=0.8)
            self.ax.add_patch(self.com_marker) 

        if self.show_time:

            file_text_x = 0.5 * (self.limits[1] - self.limits[0]) + self.limits[0]
            file_text_y = 1.1 * (self.limits[3] - self.limits[2]) + self.limits[2]
            time_text_x = self.limits[0]
            time_text_y = 1.1 * (self.limits[3] - self.limits[2]) + self.limits[2]

            print(file_text_x)

            if length_units:
                file_text_x /= 1e9*length_units
                file_text_y /= 1e9*length_units
                time_text_x /= 1e9*length_units
                time_text_y /= 1e9*length_units

            print(file_text_x)

            self.file_text = self.ax.text(file_text_x, file_text_y, "")
            self.time_text = self.ax.text(time_text_x, time_text_y, "")

    def _get_limits_indices(self):

        try:
            start_x_idx = int(np.round((self.limits[0] / self.Lx) * self.m_array.shape[0]))
            end_x_idx = int(np.round((self.limits[1] / self.Lx) * self.m_array.shape[0]))
            start_y_idx = int(np.round((self.limits[2] / self.Ly) * self.m_array.shape[1]))
            end_y_idx  = int(np.round((self.limits[3] / self.Ly) * self.m_array.shape[1]))
        
        except TypeError:
            start_x_idx = 0
            end_x_idx = self.m_array.shape[0]
            start_y_idx = 0
            end_y_idx = self.m_array.shape[1]

        return [start_x_idx, end_x_idx, start_y_idx, end_y_idx]

    def _get_limits_indices(self):

        try:
            start_x_idx = int(np.round((self.limits[0] / self.Lx) * self.m_array.shape[0]))
            end_x_idx = int(np.round((self.limits[1] / self.Lx) * self.m_array.shape[0]))
            start_y_idx = int(np.round((self.limits[2] / self.Ly) * self.m_array.shape[1]))
            end_y_idx  = int(np.round((self.limits[3] / self.Ly) * self.m_array.shape[1]))
        
        except TypeError:
            start_x_idx = 0
            end_x_idx = self.m_array.shape[0]
            start_y_idx = 0
            end_y_idx = self.m_array.shape[1]

        return [start_x_idx, end_x_idx, start_y_idx, end_y_idx]

    def _update_magnetization_array(self, full_file):

        magnetization_array = df.Field.fromfile(full_file).array[:, :, self.z_index, :]

        magnetization_array = magnetization_array[self.limits_indices[0]: self.limits_indices[1],
            self.limits_indices[2]: self.limits_indices[3]]
            
        plot_array = Plot.vecToRGB(magnetization_array).transpose(1, 0, 2)
        self.magnetization_plot.set_array(plot_array)

    def _update_magnetization_single_component_array(self, full_file):

        magnetization_array = df.Field.fromfile(full_file).array

        magnetization_array = magnetization_array[self.limits_indices[0]: self.limits_indices[1],
            self.limits_indices[2]: self.limits_indices[3]]

        if self.component == 'x':
            new_plot_array = magnetization_array[:, :, self.z_index, 0]

        elif self.component == 'y':
            new_plot_array = magnetization_array[:, :, self.z_index, 1]

        elif self.component == 'z':
            new_plot_array = magnetization_array[:, :, self.z_index, 2]

        self.magnetization_plot.set_array(new_plot_array.T)

    def _update_skyrmion_density_array(self, full_file):

        magnetization_array = df.Field.fromfile(full_file).array[:, :, self.z_index]

        magnetization_array = magnetization_array[self.limits_indices[0]: self.limits_indices[1],
            self.limits_indices[2]: self.limits_indices[3]]

        magnetization_array = magnetization_array.reshape(magnetization_array.shape[0],
            magnetization_array.shape[1], 3)
        
        dx, dy = Read.sampleDiscretisation(self.directory)[:2]
        skyrmion_density_array = Calculate.skyrmionNumberDensity(magnetization_array, dx, dy, self.length_units).T
        self.magnetization_plot.set_array(skyrmion_density_array)

    def _update_quiver_array(self, full_file):

        # Load magnetization array and cut depending on user-defined limits
        magnetization_array = df.Field.fromfile(full_file).array[:, :, self.z_index, :]
        magnetization_array = magnetization_array[self.limits_indices[0]: self.limits_indices[1],
            self.limits_indices[2]: self.limits_indices[3]]

        # Remove the z-axis as only plotting in 2D plane
        magnetization_array = magnetization_array.reshape(
                magnetization_array.shape[0], magnetization_array.shape[1], 3)
        
        # Skip points specified by the step parameter
        magnetization_array = magnetization_array[::self.step, ::self.step, :]

        # Get in-plane magnetization value to normalise arrows
        in_plane_magnitude = np.sqrt(magnetization_array[:, :, 0]**2 + magnetization_array[:, :, 1]**2).T
        
        # Update arrow directions 
        self.magnetization_plot.set_UVC(magnetization_array[:, :, 0].T / in_plane_magnitude, magnetization_array[:, :, 1].T / in_plane_magnitude)

        # Update arrow colours
        colour_array = self.plotter._get_quiver_colour_array(magnetization_array)
        self.magnetization_plot.set_facecolors(colour_array)

    def _update_marker_position(self, i):

        com_idx = i + self.corrected_idx
        marker_x, marker_y = self.com_array[com_idx]

        # For when the skrymion goes over the bounary with PBCs
        while marker_x >= self.Lx:
            marker_x -= self.Lx

        while marker_x < 0:
            marker_x += self.Lx

        while marker_y >= self.Ly:
            marker_y -= self.Ly

        while marker_y < 0:
            marker_y += self.Ly

        self.com_marker.center = [marker_x, marker_y]

    def animate(self):

        def update_anim(i):

            print("Producing animation frame", i, "of", len(self.files_to_scan), end='\r')

            full_file = self.directory + '/' + self.files_to_scan[i]

            if self.show_time:
                self.file_text.set_text('File: ' + self.files_to_scan[i])
                self.time_text.set_text('$t$ = ' + "{:.2f}".format(Read.fileTime(full_file) * 1e9) + " ns")

            if self.plot_type == 'magnetization':
                self._update_magnetization_array(full_file)

            elif self.plot_type == 'magnetization_single_component':
                self._update_magnetization_single_component_array(full_file)

            elif self.plot_type == 'skyrmion_density':
                self._update_skyrmion_density_array(full_file)

            elif self.plot_type == 'quiver':
                self._update_quiver_array(full_file)

            if self.com_array_file: self._update_marker_position(i)

        anim = animation.FuncAnimation(
                self.fig, update_anim, iter(range(len(self.files_to_scan))), blit=False, save_count=len(self.files_to_scan))         

        if self.out_name is None:

            if self.plot_type == 'magnetization':
                if self.start_file or self.limits:
                    # After so much time generating the animation for the full simulation, don't want to overwrite it when looking at part of the simulation
                    self.out_name = 'mag_part.mp4'

                else:
                    self.out_name = 'mag.mp4'

            elif self.plot_type == 'magnetization_single_component':
                if self.start_file or self.limits:
                    # After so much time generating the animation for the full simulation, don't want to overwrite it when looking at part of the simulation
                    self.out_name = 'm' + self.component + '_part' + '.mp4'

                else:
                    self.out_name = 'm' + self.component + '.mp4'

            elif self.plot_type == 'skyrmion_density':
                if self.start_file or self.limits:
                    # After so much time generating the animation for the full simulation, don't want to overwrite it when looking at part of the simulation
                    self.out_name = 'SkDensityPart.mp4'

                else:
                    self.out_name = 'SkDensity.mp4'

            elif self.plot_type == 'quiver':
                if self.start_file or self.limits:
                    # After so much time generating the animation for the full simulation, don't want to overwrite it when looking at part of the simulation
                    self.out_name = 'QuiverPart.mp4'

                else:
                    self.out_name = 'Quiver.mp4'

        anim.save(self.out_name, fps=25, writer='ffmpeg')
