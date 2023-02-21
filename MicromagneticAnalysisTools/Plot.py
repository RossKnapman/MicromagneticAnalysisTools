"""
Tools for plotting simulation data and associated quantities.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import discretisedfield as df
from MicromagneticAnalysisTools import Read
from MicromagneticAnalysisTools import Calculate
import logging
plt.rcdefaults()  # Reset stylesheet imported by discretisedfield

# Stop annoying warnings from matplotlib (may need to remove this for debugging)
logging.getLogger('matplotlib').setLevel(level=logging.ERROR)


def getImpurityArray(directory, impurityColour, zIndex):
    """
    Returns an array used to plot the impurity on a colour plot.

    Args:
        directory (str): The directory in which the simulation data is stored.
        impurityColour (ndarray): Colour of the impurity, which should be a NumPy array of the form [R, G, B, Alpha].
        zIndex (int): The index along the z-axis for which the skyrmion centre should be calculated (for a 3D sample). For a 2D sample, should be 0.
    
    """

    impurityFile = directory + '/' + 'K.ovf'
    K = df.Field.fromfile(impurityFile).array[:, :, zIndex, :]
    impurityArray = np.abs(
        K - K[K.shape[0]-1, K.shape[1]-1]).reshape(K.shape[0], K.shape[1])
    truthArray = np.zeros(
        shape=(impurityArray.shape[0], impurityArray.shape[1], 4))
    truthArray[impurityArray > 0.5] = impurityColour

    return truthArray


def getPinningArray(directory):
    """
    Returns an array used to plot the pinned region on a colour plot.

    Args:
        directory (str): The directory in which the simulation data is stored.

    Returns:
        Array of regions that are pinned (used to plot pinned region on colour plot with grey overlay).
    
    """

    pinningFile = directory + '/' + 'FrozenSpins.ovf'
    frozen = df.Field.fromfile(pinningFile).array[:, :, 0, 0]
    pinningArray = np.zeros(shape=(frozen.shape[0], frozen.shape[1], 4))
    pinningArray[frozen == 1.] = np.array(
        [0, 0, 0, 0.5])  # Where frozen, overlay grey

    return pinningArray


def vecToRGB(m):
    """
    Allows for HLS colour plot of magnetization without having to manually loop pixel-by-pixel.
    Essentially a vectorised version of colorsys.hls_to_rgb.

    Args:
        directory (str): The directory in which the simulation data is stored.

    Returns:
        An array of shape `(Nx, Ny, 3)`, where the final axis is contains the [R, G, B] values.
    
    """

    m[np.where(m == -0.)] = 0.  # Change -0. to 0.

    ONE_THIRD = 1.0/3.0
    ONE_SIXTH = 1.0/6.0
    TWO_THIRD = 2.0/3

    def _v(m1, m2, hue):

        hue = hue % 1.

        outValue = np.zeros(m1.shape, dtype=float)

        whereHueLessOneSixth = np.where(hue < ONE_SIXTH)
        outValue[whereHueLessOneSixth] = m1[whereHueLessOneSixth] + \
            (m2[whereHueLessOneSixth] - m1[whereHueLessOneSixth]) * \
            hue[whereHueLessOneSixth] * 6.

        whereHueLessHalf = np.where((hue < 0.5) & (hue >= ONE_SIXTH))
        outValue[whereHueLessHalf] = m2[whereHueLessHalf]

        whereHueLessTwoThird = np.where((hue < TWO_THIRD) & (hue >= 0.5))
        outValue[whereHueLessTwoThird] = m1[whereHueLessTwoThird] + \
            (m2[whereHueLessTwoThird] - m1[whereHueLessTwoThird]) * \
            (TWO_THIRD - hue[whereHueLessTwoThird]) * 6.0

        remainingPositions = np.where(hue >= TWO_THIRD)
        outValue[remainingPositions] = m1[remainingPositions]

        return outValue

    s = np.linalg.norm(m, axis=2)
    l = 0.5 * m[:, :, 2] + 0.5
    h = np.arctan2(m[:, :, 1], m[:, :, 0]) / (2 * np.pi)

    h[np.where(h > 1)] -= 1
    h[np.where(h < 0)] += 1

    rgbArray = np.zeros(m.shape, dtype=float)

    wheresIsZero = np.where(s == 0.)

    try:
        rgbArray[wheresIsZero] = np.array(
            [float(l[wheresIsZero]), float(l[wheresIsZero]), float(l[wheresIsZero])])

    except TypeError:  # No such points found
        pass

    m2 = np.zeros((rgbArray.shape[0], rgbArray.shape[1]), dtype=float)

    wherelIsLessThanHalf = np.where(l <= 0.5)
    m2[wherelIsLessThanHalf] = l[wherelIsLessThanHalf] * \
        (1.0 + s[wherelIsLessThanHalf])

    wherelIsMoreThanHalf = np.where(l > 0.5)
    m2[wherelIsMoreThanHalf] = l[wherelIsMoreThanHalf] + \
        s[wherelIsMoreThanHalf] - l[wherelIsMoreThanHalf] * s[wherelIsMoreThanHalf]

    m1 = 2.0 * l - m2

    rgbArray[:, :, 0] = _v(m1, m2, h+ONE_THIRD)
    rgbArray[:, :, 1] = _v(m1, m2, h)
    rgbArray[:, :, 2] = _v(m1, m2, h-ONE_THIRD)

    return rgbArray


class MagnetizationPlotter:
    """Class for dealing with plotting magnetization texture.

    Attributes:
        plot_type (str): The style of the plot, which can be "magnetization" which plots a HSL colour plot, "magnetization_single_component"
            which plots a single component specified by the `component` attribute, "skyrmion_density", which plots skyrmion density, or
            "quiver", which shows arrows.
        directory (str): The directory in which the simulation data is stored.
        plot_file (str): The file to be plotted (e.g. "m000231.ovf").
        ax (matplotlib.axis.Axis): The Matplotlib axis on which to plot.
        z_index (int, optional): The index along the z-axis for which the skyrmion centre should be calculated (for a 3D sample).
        component (str, optional): For the case that `plot_type` is `magnetization_single_component`, this speficied the component.
        plot_impurity (bool, optional): Whether or not to plot an impurity (e.g. region of modified uniaxial anisotropy).
        plot_pinning (bool, optional): Whether or not to plot a region in which the spins are frozen (shown in semi-transparent grey).
        interpolation (str, optional): Interpolation method used for the colour plots (passed to `matplotlib.pyplot.imshow`).
        limits (List[int], optional): Limits in which to plot (in nm) of the form `[x_min, x_max, y_min, y_max]`.
        length_units(float64, optional): Amount by which to scale discretisation (which rescales number density).
        max_skyrmion_density(float64, optional): The maximum skyrmion number density to be plotted (corresponds to vmax and -vmin in `imshow`).
        step (int, optional): For the quiver plot, how many cells should be skipped between points.
        quiver_colour (List[float64], optional): List of form `[R, G, B]` for colour of arrows.
    
    """

    def __init__(self,
    plot_type,
    directory,
    plot_file,
    ax=None,
    z_index=0,
    component=None,
    plot_impurity=False,
    plot_pinning=False,
    show_component=False,
    interpolation=None,
    limits=None,
    length_units=None,
    max_skyrmion_density=None,
    step=1,
    quiver_colour=[0, 0, 0]):
        self.plot_type = plot_type
        self.directory = directory
        self.plot_file = plot_file
        self.ax = ax
        self.z_index = z_index
        self.component = component
        self.plot_impurity = plot_impurity
        self.plot_pinning = plot_pinning
        self.show_component = show_component
        self.interpolation = interpolation
        self.limits = limits
        self.length_units = length_units
        self.max_skyrmion_density = max_skyrmion_density
        self.step = step 
        self.quiver_colour = quiver_colour

        self.Lx, self.Ly = Read.sampleExtent(self.directory)
        self.m_array = df.Field.fromfile(self.directory + '/' + self.plot_file).array

        if self.limits == None: self.limits = [0., self.Lx, 0., self.Ly]
        self.limits_indices = self._get_limits_indices()
        
        self.m_array = self.m_array[self.limits_indices[0]: self.limits_indices[1],
            self.limits_indices[2]: self.limits_indices[3]]

        if self.plot_type == 'magnetization_single_component' and self.component == None:
            raise ValueError('Must specify component x, y, or z.')

        if self.ax == None: self.ax = plt.subplots()[1]

        # If length units are supplied, we want to rescale the axis extent
        if self.length_units:
            self.limits = list(map(lambda x: 1e-9 * x / length_units, self.limits))

    def _get_limits_indices(self):

        start_x_idx = int(np.round((self.limits[0] / self.Lx) * self.m_array.shape[0]))
        end_x_idx = int(np.round((self.limits[1] / self.Lx) * self.m_array.shape[0]))
        start_y_idx = int(np.round((self.limits[2] / self.Ly) * self.m_array.shape[1]))
        end_y_idx  = int(np.round((self.limits[3] / self.Ly) * self.m_array.shape[1]))
        
        return [start_x_idx, end_x_idx, start_y_idx, end_y_idx]

    def _plot_magnetization(self):

        magnetization_array = self.m_array[:, :, self.z_index, :]
        plot_array = vecToRGB(magnetization_array).transpose(1, 0, 2)
        self.out_plot = self.ax.imshow(plot_array, animated=True, origin='lower',
            interpolation=self.interpolation, extent=self.limits)

    def _plot_magnetization_single_component(self):

        if self.component == 'x':
            plot_array = self.m_array[:, :, self.z_index, 0]

        elif self.component == 'y':
            plot_array = self.m_array[:, :, self.z_index, 1]

        elif self.component == 'z':
            plot_array = self.m_array[:, :, self.z_index, 2]

        self.out_plot = self.ax.imshow(plot_array.T, animated=True, vmin=-1, vmax=1, origin='lower',
            cmap='RdBu_r', interpolation=self.interpolation, extent=self.limits)

        if self.show_component:
            self.ax.text(self.Lx + 5, 0, "$m_" + self.component + "$")

    def _plot_skyrmion_density(self):

        magnetization_array = self.m_array[:, :, self.z_index].reshape\
            (self.m_array.shape[0], self.m_array.shape[1], 3)

        dx, dy = Read.sampleDiscretisation(self.directory)[:2]

        skyrmion_density_array = Calculate.skyrmionNumberDensity(magnetization_array, dx, dy, self.length_units).T

        if self.max_skyrmion_density:
            colour_map_max = self.max_skyrmion_density

        else:
            colour_map_max = np.max(np.abs(skyrmion_density_array))

        self.out_plot = self.ax.imshow(skyrmion_density_array, animated=True, vmin=-1*colour_map_max, vmax=colour_map_max,
            origin='lower', cmap='PRGn', interpolation=self.interpolation, extent=self.limits)

    def _plot_quiver(self):

        # Note that the arrays are "flipped" here due to how the axes in quiver() are defined
        x = np.linspace(self.limits[2], self.limits[3], self.m_array.shape[1])
        y = np.linspace(self.limits[0], self.limits[1], self.m_array.shape[0])
        X, Y = np.meshgrid(x, y)

        X = X[::self.step, ::self.step]
        Y = Y[::self.step, ::self.step]

        # Skip points specified by the step parameter
        magnetization_array = self.m_array[::self.step, ::self.step, self.z_index, :]

        # Get in-plane magnetization value to normalise arrows
        in_plane_magnitude = np.sqrt(magnetization_array[:, :, 0]**2 + magnetization_array[:, :, 1]**2).T

        colour_array = self._get_quiver_colour_array(magnetization_array)
        self.out_plot = self.ax.quiver(Y.T, X.T, magnetization_array[:, :, 0].T / in_plane_magnitude,
            magnetization_array[:, :, 1].T / in_plane_magnitude, color=colour_array, units='xy', scale_units='xy', pivot='mid',
            headwidth=6, headlength=10, headaxislength=10, linewidth=5)

    def _plot_impurity(self):
        impurity_array = np.flip(getImpurityArray(self.directory, np.array([0, 1, 0, 0.2]), self.z_index)[
self.limits_indices[0]: self.limits_indices[1], self.limits_indices[2]: self.limits_indices[3]].transpose(1, 0, 2), axis=0)
        self.ax.imshow(impurity_array, extent=self.limits)

    def _plot_pinning(self):
        pinning_array = np.flip(getPinningArray(self.directory)[
self.limits_indices[0]: self.limits_indices[1], self.limits_indices[2]: self.limits_indices[3]].transpose(1, 0, 2), axis=0)
        self.ax.imshow(pinning_array, extent=self.limits)

    def plot(self):
        """Plots the magnetization plot.

        Returns:
            The magnetization plot.
        
        """

        if self.plot_type == 'magnetization':
            self._plot_magnetization()

        elif self.plot_type == 'magnetization_single_component':
            self._plot_magnetization_single_component()

        elif self.plot_type == 'skyrmion_density':
            self._plot_skyrmion_density()

        elif self.plot_type == 'quiver':
            self._plot_quiver()

        else:
            raise ValueError('Plot type must be one of: magnetization, magnetization_single_component, \
                skyrmion_density, quiver.')

        if self.plot_impurity: self._plot_impurity()
        if self.plot_pinning: self._plot_pinning()

        self.ax.set_xlabel(r'$x$ (nm)' if not self.length_units else r'$x$ (dimensionless)')
        self.ax.set_ylabel(r'$y$ (nm)' if not self.length_units else r'$y$ (dimensionless)')
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.limits[0], self.limits[1])
        self.ax.set_ylim(self.limits[2], self.limits[3])

        return self.out_plot

    def _get_quiver_colour_array(self, magnetization_array):  # Get the colour array for the quiver plots

        colour_array = np.ones((magnetization_array.shape[0], magnetization_array.shape[1], 4))  # Array of [r, g, b, alpha]

        # Set the arrow colour to the user-defined colour
        colour_array[:, :, :3] = self.quiver_colour

        # Get the alpha value based on in-plane magnetization component (perfectly transparent for completely out-of-plane and vice-versa)
        colour_array[:, :, 3] = np.sqrt(magnetization_array[:, :, 0]**2 + magnetization_array[:, :, 1]**2)

        # Reshape to fit matplotlib quiver convention
        colour_array = colour_array.transpose(1, 0, 2)

        # Should already be between 0 and 1, but sometimes we get e.g. 1.0000001, which causes a ValueError from matplotlib
        colour_array[np.where(colour_array > 1.)] = 1.

        return np.reshape(colour_array, (magnetization_array.shape[0] * magnetization_array.shape[1], 4))


def plotSpeedOverSimulations(directories, currentComponent, speedComponent, COMFileName='COM.npy', ax=None, presentationMethod='legend', showStopRamp=False, timeMultiplier=None, speedMultiplier=None, currentMultiplier=None, colorbarLabel=None):
    """
    For a given directory, plot the speeds of the skyrmions in the respective sub-directories.

    Args:
        directory (str): The directory in which the simulation data is stored.
        currentcomponent (str): The component of the current to be read. Must be "x", "y", or "z".
        currentcomponent (str): The component of the speed to be calculated. Must be "x", "y", or "z".
        COMFileName (str, optional): The filename of the NumPy array containing skyrmion centre positions (calculated using e.g. skyrmion COMArray()).
        ax (matplotlib.axis.Axis): The Matplotlib axis on which to plot.
        presentationMethod (str, optional): The method used to distinguish between different quantites being plotted; either "legend" or "colormap".
            For example, say we want to plot skyrmion speed as a function of current for varies damping constants 0.01, 0.02, ..., 0.10. We could
            either have a legend showing the colour for each constant ("legend"), or make the line for 0.01 light and increasingly dark until 0.10
            ("colormap").
        showStopRamp (bool): Whether or not to draw a vertical line where the current stops ramping.
        timeMultiplier (float): Quantity by which to multiply the times array in seconds (e.g. to convert to dimensionless units).
        energyMultiplier (float): Quantity by which to multiply the energy array in Joules (e.g. to convert to dimensionless units).
        speedMultiplier (float): Quantity by which to multiply the energy array in m/s (e.g. to convert to dimensionless units).
        currentMultiplier (float): Quantity by which to multiply the current array in A (e.g. to convert to dimensionless units),
            if the quantity being varied for different plots is current
        colorBarLabel (str): The label of the colour bar.
    
    """

    if ax is None:
        ax = plt.subplots()[1]

    if currentMultiplier:
        directoriesNumerical = [
            currentMultiplier * float(directory.split('/')[-1]) for directory in directories]

    else:
        directoriesNumerical = [float(directory.split('/')[-1])
                                for directory in directories]

    if presentationMethod == 'colormap':
        cmap = matplotlib.cm.get_cmap('Reds')
        norm = matplotlib.colors.Normalize(vmin=np.min(
            directoriesNumerical), vmax=np.max(directoriesNumerical))

    for directory in directories:

        times = Read.simulationTimeArray(directory + '/Data/')
        COM = np.load(directory + '/' + COMFileName)
        speeds = Calculate.speedAgainstTime(times, COM, speedComponent)
        if timeMultiplier:
            times *= timeMultiplier
        if speedMultiplier:
            speeds *= speedMultiplier

        # TODO Change this name; the quantity does not necessarily need to be current
        current = directory.split('/')[-1]
        if currentMultiplier:
            current = str(currentMultiplier * float(current))

        if presentationMethod == 'legend':
            ax.plot(times * 1e9, speeds, label=current)

        elif presentationMethod == 'colormap':
            ax.plot(times * 1e9, speeds, color=cmap(norm(float(current))))

        else:
            raise ValueError(
                'Method of presenting data must be \'legend\' or \'colormap\'.')

    if showStopRamp:
        stopRampIdx = Calculate.getRampIdx(
            directories[0] + '/Data/', currentComponent)
        ax.axvline(times[stopRampIdx] * 1e9, color='black')

    if presentationMethod == 'legend':
        ax.legend()

    elif presentationMethod == 'colormap':
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(sm, cax=cax)
        cb.ax.set_ylabel(colorbarLabel, rotation=270, labelpad=30, fontsize=16)
