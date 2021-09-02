from MicromagneticAnalysisTools.Plot import MagnetizationPlotter
from MicromagneticAnalysisTools import Animate
from MicromagneticAnalysisTools import Calculate
from MicromagneticAnalysisTools import Read
import numpy as np
import os
import pytest
import matplotlib.pyplot as plt
import discretisedfield as df


@pytest.fixture(autouse=True)
def cleanup():
    yield
    for item in os.listdir(os.getcwd()):
        if item.endswith('.mp4') or item.endswith('.npy'):
            print('removing', item)
            os.remove(item)


def test_NskAnimation():
    """ Test 1D animated skyrmion density over time plot. """
    Nsk_array = Calculate.NskArray('tests/data')
    time_array = Read.simulationTimeArray('tests/data', loadMethod='files')
    Animate.NskAnimation(Nsk_array, time_array)


def test_currentAnimation():
    current_array = Read.simulationCurrentArray('tests/data', 'x')
    time_array = Read.simulationTimeArray('tests/data', loadMethod='table')
    Animate.currentAnimation(current_array, time_array, 'x')


def test_applying_limits():
    """ Try showing a smaller region of the system, and make sure that the array is actually being cut. """
    original_array = df.Field.fromfile('tests/data/m000000.ovf').array
    animator = Animate.MagnetizationAnimator('magnetization', 'tests/data', limits=(10, 900, 10, 300))
    animator.animate()
    cut_array = animator.colour_plot.get_array()
    # The original array has desretisation cell size 1nm in x
    assert cut_array.shape[1] == 890

def test_magnetization_component_animation():
    """ Test animating e.g. m_z. """
    animator = Animate.MagnetizationAnimator('magnetization_single_component', 'tests/data', component='z')
    animator.animate()


def test_magnetization_component_animation_with_com_tracking():
    """ Test showing the COM in the animation. """
    com_array = Calculate.skyrmionCOMArrayCoMoving('tests/data')
    np.save('COM', com_array)
    animator = Animate.MagnetizationAnimator('magnetization_single_component', 'tests/data', component='x', com_array_file='COM.npy')
    animator.animate()
    

def test_magnetization_animation():
    """ Test animation magnetzation HSL colour plot. """
    animator = Animate.MagnetizationAnimator('magnetization', 'tests/data')
    animator.animate()


def test_magnetization_animation_with_frozen_spins():
    animator = Animate.MagnetizationAnimator('magnetization', 'tests/data', plot_pinning=True)
    animator.animate()


def test_magnetization_animation_with_quiver():
    """ Test magnetization animation with quiver plot. """
    animator = Animate.MagnetizationAnimator('magnetization', 'tests/data', quiver=True)
    animator.animate()


def test_skyrmion_density_animation():
    """ Test animation skyrmion density. """
    animator = Animate.MagnetizationAnimator('skyrmion_density', 'tests/data')
    animator.animate()


def test_colour_plot_invalid_type():
    """ Ensure error thrown when the user passes something other than magnetization, magnetization_single_component, skyrmion_density. """
    with pytest.raises(ValueError):
        animator = Animate.MagnetizationAnimator('something invalid', 'tests/data')
        animator.animate()


def test_supplying_own_ax():
    """ Test the case where the user supplies their own axes that they created. """
    fig, ax = plt.subplots()
    animator = Animate.MagnetizationAnimator('magnetization', 'tests/data', fig=fig, ax=ax)
    animator.animate()


def test_trying_to_supply_ax_without_figure():
    """ Should raise an exception if the user tries to supply matplotlib axes without also supplying a figure. """
    with pytest.raises(ValueError):
        ax = plt.subplots()[1]
        animator = Animate.MagnetizationAnimator('magnetization', 'tests/data', ax=ax)
        animator.animate()