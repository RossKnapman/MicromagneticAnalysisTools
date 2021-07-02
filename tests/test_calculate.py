import pytest
from src import Calculate
import numpy as np


def test_skyrmion_number_trivial():
    """ Ensure that a collinear state has a topological charge of zero. """

    array_size = 100
    mArray = np.zeros((array_size, array_size, 3), dtype=np.float64)
    mArray[:, :, 0] = 0.
    mArray[:, :, 1] = 0.
    mArray[:, :, 2] = 1.
    assert np.abs(Calculate.skyrmionNumber(mArray)) < 1e-3


@pytest.mark.parametrize('m', [1, 2, 0, -1])
def test_skyrmion_number_nontrivial(m):

    """ Ensure that a texture that is a skyrmion has a skyrmion number of one, given a vorticity m. """

    array_size = 100
    side_length = 20
    R = 5  # Skyrmion radius
    w = 2  # Skyrmion domain wall width
    x = y = np.linspace(-side_length, side_length, array_size)
    X, Y = np.meshgrid(x, y)

    rho = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    Theta = 2 * np.arctan2(np.sinh(R/w), np.sinh(rho/w))
    Phi = m * phi

    mx = np.cos(Phi) * np.sin(Theta)
    my = np.sin(Phi) * np.sin(Theta)
    mz = np.cos(Theta)

    mArray = np.zeros((len(X), len(Y), 3), dtype=np.float64)
    mArray[:, :, 0] = mx
    mArray[:, :, 1] = my
    mArray[:, :, 2] = mz

    assert np.abs(Calculate.skyrmionNumber(mArray) - m) < 0.05
