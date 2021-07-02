from src import Animate
from src import Calculate
from src import Read
import numpy as np


def test_NskAnimation():
    """ Test 1D animated skyrmion density over time plot. """
    Nsk_array = Calculate.NskArray('tests/data')
    time_array = Read.simulationTimeArray('tests/data', loadMethod='files')
    print(Nsk_array.shape)
    print(time_array.shape)
    Animate.NskAnimation(Nsk_array, time_array)


def test_currentAnimation():
    current_array = Read.simulationCurrentArray('tests/data', 'x')
    time_array = Read.simulationTimeArray('tests/data', loadMethod='table')
    Animate.currentAnimation(current_array, time_array, 'x')


def test_magnetization_component_animation():
    """ Test animating e.g. m_z. """
    Animate.colourAnimation('tests/data', 'magnetization', component='z')


def test_magnetization_component_animation_with_com_tracking():
    """ Test showing the COM in the animation. """
    com_array = Calculate.skyrmionCOMArrayCoMoving('tests/data')
    np.save('COM', com_array)
    Animate.colourAnimation('tests/data', 'magnetization', component='z', COMArray='COM.npy')
    

def test_magnetization_HSL_animation():
    """ Test animation magnetzation HSL colour plot. """
    Animate.colourAnimation('tests/data', 'magnetizationHSL')


def test_skyrmion_density_animation():
    """ Test animation skyrmion density. """
    Animate.colourAnimation('tests/data', 'skyrmionDensity')