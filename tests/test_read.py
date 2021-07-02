from src import Read
import pytest

def test_getInitialFile():
    Read.getInitialFile('tests/data')

def test_simulationTimeArray():
    Read.simulationTimeArray('tests/data')


# def test_simulationCurrentArray():


# def test_simulationEnergyArray():


# def test_fileTime():


# def test_getFilesToScan():


# def test_sampleDiscretisation():


# def test_sampleExtent():


# def test_initialFileArray():


# def test_loadFile():