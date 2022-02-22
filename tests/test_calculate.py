import pytest
from MicromagneticAnalysisTools import Calculate
import numpy as np
import os


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

def test_hopf_index():

    """ Load in sample hopfion texture and ensure the Hopf index is approximately 1. """

    with np.load("tests/data/Hopfion.npz") as compressed:
        m = compressed['m']

    assert np.abs(1. - Calculate.HopfIdx(m)) < 0.05


def test_helicity_calculation():

    """ Test helicity calculation using various sample skyrmion ovf files. (All have helicity 0 unless stated otherwise in file name.) """

    tol = 1e-2  # Tolerance for helicity calculation
    helicity_test_dir = 'tests/data/helicity_tests/'
    
    for filename in os.listdir(helicity_test_dir):

        # Deal with where the helicity is explicitly given in file name
        if filename == 'helicity_0.ovf':
            assert np.abs(Calculate.skyrmion_helicity(helicity_test_dir, filename)) < tol
        
        elif filename == 'helicity_pi_2.ovf':
            assert (np.abs(Calculate.skyrmion_helicity(helicity_test_dir, filename)) - np.pi/2) < tol

        elif filename == 'helicity_pi.ovf':
            assert (np.abs(Calculate.skyrmion_helicity(helicity_test_dir, filename)) - np.pi) < tol

        elif filename == 'helicity_3_pi_2.ovf':
            assert (np.abs(Calculate.skyrmion_helicity(helicity_test_dir, filename)) - 3*np.pi/2) < tol
        
        else:
            assert np.abs(Calculate.skyrmion_helicity(helicity_test_dir, filename)) < tol
