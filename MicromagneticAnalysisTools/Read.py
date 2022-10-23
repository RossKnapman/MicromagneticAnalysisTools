"""
Tools for reading in simulation files.
"""

import os
import pandas as pd
import discretisedfield as df
import numpy as np
import warnings

# Note: The "directory" argument is usually /path/to/Data


def getInitialFile(directory):
    """ Get the name of the intial file of the simulation (as I call them different things: Relaxed.ovf, and m000000.ovf). """

    if os.path.isfile(directory + "/Relaxed.ovf"):
        return directory + "/Relaxed.ovf"

    elif os.path.isfile(directory + "/m000000.ovf"):
        return directory + "/m000000.ovf"

    else:

        # Just get any file in the directory and use that if no Relaxed.ovf or m000000.ovf; they will probably all have the same extent etc. anyway
        for file in os.listdir(directory):
            if file.endswith('.ovf'):
                initialFile = file
                break

        try:
            warnings.warn(
                'Using random file from directory as no initial file found.')
            return directory + '/' + initialFile

        except UnboundLocalError:
            raise FileNotFoundError('No ovf files found.')


def simulationTimeArray(directory, loadMethod="table", startFile=None, endFile=None):
    """ Reads the time of a simulation and outputs an array. We can either do this by loading the table output by the MuMax, or we can obtain the timestamp of each file in the simulation. """

    if startFile:
        loadMethod = "files"

    if loadMethod == "table":  # This method is much faster

        df = pd.read_csv(directory + '/table.txt', delimiter='\t')
        timeArray = df['# t (s)'].to_numpy()

    elif loadMethod == "files":

        filesToScan = getFilesToScan(directory, startFile, endFile)
        timeArray = np.zeros(len(filesToScan))

        for i in range(len(filesToScan)):
            print("Getting time of file", i, "of", len(filesToScan), end="\r")
            timeArray[i] = fileTime(directory + '/' + filesToScan[i])

    else:
        raise Exception('loadMethod should be "table" or "files".')

    return timeArray


def simulationCurrentArray(directory, component):
    """ Reads the current of a simulation and outputs an array. Component can be x, y, z. """

    df = pd.read_csv(directory + '/table.txt', delimiter='\t')
    # Negative as the electron charge is negative
    currentArray = -df['J' + component + ' (A/m2)'].to_numpy()

    return currentArray


def simulationEnergyArray(directory):
    """ Read the energy fo a simulation and outputs an array. """

    df = pd.read_csv(directory + '/table.txt', delimiter='\t')
    energyArray = df['E_total (J)'].to_numpy()

    return energyArray


def fileTime(file):
    """ Reads the total simulation time given the name of a .ovf file. """

    with open(file, "rb") as ovffile:
        f = ovffile.read()
        lines = f.split(b"\n")

        mdatalines = filter(lambda s: s.startswith(bytes("#", "utf-8")), lines)
        for line in mdatalines:
            if b"Total simulation time" in line:
                return float(line.split()[5])


def getFilesToScan(directory, startFile=None, endFile=None):
    """ Outputs a sorted list of .ovf files which are to be parsed. """

    filesToScan = []

    for file in os.listdir(directory):
        if file.endswith(".ovf") and file != "Relaxed.ovf" and file != "K.ovf" and file != "FrozenSpins.ovf":
            filesToScan.append(file)

    filesToScan = sorted(filesToScan)

    if startFile and endFile:

        startIdx = filesToScan.index(startFile)
        endIdx = filesToScan.index(endFile)

        return filesToScan[startIdx:endIdx + 1]

    else:

        return filesToScan


def sampleDiscretisation(directory):
    """ Get spatial extent of the sample in nm. """

    initialFile = getInitialFile(directory)

    with open(initialFile, 'rb') as f:
        for i in range(26):
            f.readline()
            if i == 22:  #  Loop through the header of the file
                dx = float(str(f.readline()).split(
                    'stepsize: ')[1].split('\\n')[0])
                dy = float(str(f.readline()).split(
                    'stepsize: ')[1].split('\\n')[0])
                dz = float(str(f.readline()).split(
                    'stepsize: ')[1].split('\\n')[0])

    return dx, dy, dz


def sampleExtent(directory):

    dx, dy = sampleDiscretisation(directory)[:2]
    mInit = initialFileArray(directory)

    Lx = dx * mInit.shape[0] * 1e9
    Ly = dy * mInit.shape[1] * 1e9
    # Really should generalise to also output Lz

    return Lx, Ly


def initialFileArray(directory):
    """ Get the initial .ovf file of the simulation as a numpy array. """

    initialFile = getInitialFile(directory)
    mInit = df.Field.fromfile(initialFile).array

    if mInit.shape[2] == 1:  # If thin film, the z axis is redundant
        mInit = mInit.reshape(mInit.shape[0], mInit.shape[1], 3)

    return mInit

