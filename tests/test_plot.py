from MicromagneticAnalysisTools import Plot
from MicromagneticAnalysisTools import Calculate
import discretisedfield as df
from colorsys import hls_to_rgb
import numpy as np
import pytest


def test_magnetizationQuiver():
    magnetization_plotter = Plot.MagnetizationPlotter('quiveronly', 'tests/data', 'm000000.ovf')
    magnetization_plotter.plot()


def test_magnetizationPlot():
    magnetization_plotter = Plot.MagnetizationPlotter('magnetization_single_component', 'tests/data', 'm000000.ovf', component='z')
    magnetization_plotter.plot()


def test_magnetizationPlot_with_limits():
    magnetization_plotter = Plot.MagnetizationPlotter('magnetization_single_component', 'tests/data', 'm000000.ovf', component='x', limits=(10, 900, 10, 300))
    magnetization_plotter.plot()


def test_magnetizationPlot_with_frozen_spins():
    magnetization_plotter = Plot.MagnetizationPlotter('magnetization_single_component', 'tests/data', 'm000000.ovf', component='z', plot_pinning=True)
    magnetization_plotter.plot()


def test_colour_plot_invalid_type():
    with pytest.raises(ValueError):
        magnetization_plotter = Plot.MagnetizationPlotter('some invalid plot type', 'tests/data', 'm000000.ovf', component='y')
        magnetization_plotter.plot()


def test_magnetizationPlotHSL():
    magnetization_plotter = Plot.MagnetizationPlotter('magnetization', 'tests/data', 'm000000.ovf')
    magnetization_plotter.plot()
    

def test_skyrmion_density_plot():
    skyrmion_density_plotter = Plot.MagnetizationPlotter('skyrmion_density', 'tests/data', 'm000000.ovf')
    skyrmion_density_plotter.plot()


def test_show_component():
    magnetization_plotter = Plot.MagnetizationPlotter('magnetization_single_component', 'tests/data', 'm000000.ovf', component='z', show_component=True)
    magnetization_plotter.plot()


def test_vecToRGB():
    """ Test that the vectorised vecToRGB function gives the same result as for colorsys.hls_to_rgb. """

    def vecToHSL(vec):

        s = np.linalg.norm(vec)
        l = 0.5 * vec[2] + 0.5
        h = np.arctan2(vec[1], vec[0]) / (2 * np.pi)

        while h < 0:
            h += 1

        while h > 1:
            h -= 1

        return hls_to_rgb(h, l, s)

    m = df.Field.fromfile('tests/data/m000000.ovf').array[:, :, 0, :]
    rgbVectorised = Plot.vecToRGB(m)

    rgbArray = np.zeros_like(m, dtype=float)

    for i in range(rgbArray.shape[0]):
        for j in range(rgbArray.shape[1]):
            rgbArray[i, j] = vecToHSL(m[i, j])

    assert np.array_equal(rgbArray, rgbVectorised)

    # This array has caused problems
    m = np.array([[[-0.005441933153905671, 6.66444601810444e-19, 0.9999851925721442]]])
    assert np.array_equal(Plot.vecToRGB(m)[0][0], np.array(vecToHSL(m[0, 0])))
