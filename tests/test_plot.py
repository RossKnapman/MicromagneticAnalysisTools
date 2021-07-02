from src import Plot
from src import Calculate
import numpy as np


def test_magnetizationQuiver():
    Plot.magnetizationQuiver('tests/data', 'm000000.ovf')


def test_magnetizationPlot():
    Plot.magnetizationPlot('tests/data', 'z', 'm000000.ovf')


def test_magnetizationPlot_with_limits():
    Plot.magnetizationPlot('tests/data', 'z', 'm000000.ovf', showFromX=10, showToX=900, showFromY=10, showToY=300)


def test_magnetizationPlot_with_frozen_spins():
    Plot.magnetizationPlot('tests/data', 'z', 'm000000.ovf', plotPinning=True)


def test_magnetizationPlotHSL():
    Plot.magnetizationPlotHSL('tests/data', 'z', 'm000000.ovf')
    

def test_magnetizationPlotHSL_with_limits():
    Plot.magnetizationPlotHSL('tests/data', 'z', 'm000000.ovf', showFromX=10, showToX=900, showFromY=10, showToY=300)


def test_magnetizationPlotHSL_with_frozen_spins():
    Plot.magnetizationPlotHSL('tests/data', 'z', 'm000000.ovf', plotPinning=True)
