from src import Plot
from src import Calculate
import numpy as np
import pytest


def test_magnetizationQuiver():
    magnetization_plotter = Plot.MagnetizationPlotter('quiver', 'tests/data', 'm000000.ovf')
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