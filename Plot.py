import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import discretisedfield as df

from .Read import *
from .Calculate import *


def lengthToIdx(length, systemSize, array):

    """ Convert a length (in e.g. metres) to array index, given the system's size systemSize (e.g. Lx, the x-length of the system in nanometres). """
    return int(np.round(length))


def getImpurityArray(directory, impurityColour, zIndex):

        """ Returns an array used to plot the impurity on a colour plot. The colour is of the form of a numpy array [R, G, B, alpha]. """

        impurityFile = directory + '/' + 'K.ovf'
        K = df.Field.fromfile(impurityFile).array[:, :, zIndex, :]
        impurityArray = np.abs(K - K[K.shape[0]-1, K.shape[1]-1]).reshape(K.shape[0], K.shape[1])
        truthArray = np.zeros(shape=(impurityArray.shape[0], impurityArray.shape[1], 4))
        truthArray[impurityArray > 0.5] = impurityColour 

        return truthArray


def getPinningArray(directory):

        """ Returns an array used to plot the pinned region on a colour plot. """

        pinningFile = directory + '/' + 'FrozenSpins.ovf'
        frozen = df.Field.fromfile(pinningFile).array[:, :, 0, 0]
        pinningArray = np.zeros(shape=(frozen.shape[0], frozen.shape[1], 4))
        pinningArray[frozen == 1.] = np.array([0, 0, 0, 0.5])  # Where frozen, overlay grey

        return pinningArray


def magnetizationQuiver(directory, theFile, zIndex=0, ax=None, step=1, quiverColour='black', **kwargs):

    """ Produce a quiver plot of the in-plane component of magnetization. """

    if ax is None:
        fig, ax = plt.subplots()

    fullFile = directory + '/' + theFile

    Lx, Ly = sampleExtent(directory)

    # If supplied, cut the plot to show only a certain subset, according to the desired size in metres
    showFromX = kwargs.get('showFromX', None)
    showToX = kwargs.get('showToX', None)
    showFromY = kwargs.get('showFromY', None)
    showToY = kwargs.get('showToY', None)

    fullFile = directory + '/' + theFile
    mArray = df.Field.fromfile(fullFile).array[:, :, zIndex, :]
    mArray = mArray.reshape(mArray.shape[0], mArray.shape[1], 3)

    if showFromX:

        startXIdx = int(np.round((showFromX / Lx) * mArray.shape[0]))
        endXIdx = int(np.round((showToX / Lx) * mArray.shape[0]))
        startYIdx = int(np.round((showFromY / Ly) * mArray.shape[1]))
        endYIdx = int(np.round((showToY / Ly) * mArray.shape[1]))

        mArray = mArray[startXIdx:endXIdx, startYIdx:endYIdx]
        x = np.linspace(showFromY, showToY, mArray.shape[1])
        y = np.linspace(showFromX, showToX, mArray.shape[0])

    else:
      
        x = np.linspace(0, Lx, mArray.shape[0])
        y = np.linspace(0, Ly, mArray.shape[1])

    X, Y = np.meshgrid(x, y)
    X = X[::step, ::step]
    Y = Y[::step, ::step]

    mArray = mArray[::step, ::step, :]
    alphaArray = np.ones((mArray.shape[0], mArray.shape[1], 4))
    alphaArray[:, :, 3] = np.sqrt(mArray[:, :, 0]**2 + mArray[:, :, 1]**2)
    alphaArray = alphaArray / np.max(alphaArray)
    alphaArray = alphaArray.transpose(1, 0, 2)
    alphaArray = np.interp(alphaArray, (0.4, np.max(alphaArray)), (0, 1))

    thePlot = ax.quiver(Y.transpose(), X.transpose(), mArray[:, :, 0].transpose(), mArray[:, :, 1].transpose(), color=np.reshape(alphaArray, (mArray.shape[0] * mArray.shape[1], 4)), units='xy', scale_units='xy', pivot='mid', headwidth=6, headlength=10, headaxislength = 10, linewidth=10)
    # thePlot = ax.quiver(Y[::step, ::step].transpose(), X[::step, ::step].transpose(), (mArray[:, :, 0] / np.sqrt(mArray[:, :, 0]**2 + mArray[:, :, 1]**2)).transpose(), (mArray[:, :, 1] / np.sqrt(mArray[:, :, 0]**2 + mArray[:, :, 1]**2)).transpose(), color=np.reshape(alphaArray, (mArray.shape[0] * mArray.shape[1], 4)), scale=50, pivot='mid')

    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")
  
    return thePlot


def vecToRGB(m):

    """ Vectorised version of colorsys.hls_to_rgb """

    ONE_THIRD = 1.0/3.0
    ONE_SIXTH = 1.0/6.0
    TWO_THIRD = 2.0/3

    def _v(m1, m2, hue):

        hue = hue % 1.
        
        outValue = np.zeros_like(m1, dtype=float)

        whereHueLessOneSixth = np.where(hue < ONE_SIXTH)
        outValue[whereHueLessOneSixth] = m1[whereHueLessOneSixth] + (m2[whereHueLessOneSixth] - m1[whereHueLessOneSixth]) * hue[whereHueLessOneSixth] * 6.

        whereHueLessHalf = np.where((hue < 0.5) & (hue >= ONE_SIXTH))
        outValue[whereHueLessHalf] = m2[whereHueLessHalf]

        whereHueLessTwoThird = np.where((hue < TWO_THIRD) & (hue > 0.5))
        outValue[whereHueLessTwoThird] = m1[whereHueLessTwoThird] + (m2[whereHueLessTwoThird] - m1[whereHueLessTwoThird]) * (TWO_THIRD - hue[whereHueLessTwoThird]) * 6.0

        remainingPositions = np.where(hue >= TWO_THIRD)
        outValue[remainingPositions] = m1[remainingPositions]
        
        return outValue


    s = np.linalg.norm(m, axis=2)
    l = 0.5 * m[:, :, 2] + 0.5
    h = np.arctan2(m[:, :, 1], m[:, :, 0]) / (2 * np.pi)

    h[np.where(h > 1)] -= 1
    h[np.where(h < 0)] += 1


    rgbArray = np.zeros_like(m, dtype=float)

    wheresIsZero = np.where(s == 0.)

    try:
        rgbArray[wheresIsZero] = np.array([float(l[wheresIsZero]), float(l[wheresIsZero]), float(l[wheresIsZero])])

    except TypeError:  # No such points found
        pass

    m2 = np.zeros((rgbArray.shape[0], rgbArray.shape[1]), dtype=float)

    wherelIsLessThanHalf = np.where(l <= 0.5)
    m2[wherelIsLessThanHalf] = l[wherelIsLessThanHalf] * (1.0 + s[wherelIsLessThanHalf])

    wherelIsMoreThanHalf = np.where(l > 0.5)
    m2[wherelIsMoreThanHalf] = l[wherelIsMoreThanHalf] + s[wherelIsMoreThanHalf] - l[wherelIsMoreThanHalf] * s[wherelIsMoreThanHalf]

    m1 = 2.0 * l - m2

    rgbArray[:, :, 0] = _v(m1, m2, h+ONE_THIRD)
    rgbArray[:, :, 1] = _v(m1, m2, h)
    rgbArray[:, :, 2] = _v(m1, m2, h-ONE_THIRD)

    return rgbArray


def magnetizationPlotHSL(directory, component, theFile, zIndex=0, plotImpurity=False, plotPinning=False, showComponent=True, interpolation=None, ax = None, **kwargs):

    """ Produces a colour plot (imshow) of the magnetization with component x, y, or z, corresponding to mx, my, mz. """

    if ax == None:
        fig, ax = plt.subplots()

    fullFile = directory + '/' + theFile

    Lx, Ly = sampleExtent(directory)

    # If supplied, cut the plot to show only a certain subset, according to the desired size in metres
    showFromX = kwargs.get('showFromX', None)
    showToX = kwargs.get('showToX', None)
    showFromY = kwargs.get('showFromY', None)
    showToY = kwargs.get('showToY', None)

    mArray = df.Field.fromfile(directory + '/' + theFile).array[:, :, zIndex, :]
    rgbArray = vecToRGB(mArray)


    if showFromX:

        startXIdx = int(np.round((showFromX / Lx) * mArray.shape[0]))
        endXIdx = int(np.round((showToX / Lx) * mArray.shape[0]))
        startYIdx = int(np.round((showFromY / Ly) * mArray.shape[1]))
        endYIdx = int(np.round((showToY / Ly) * mArray.shape[1]))

        rgbArray = rgbArray[startXIdx:endXIdx, startYIdx:endYIdx]
        thePlot = ax.imshow(rgbArray.transpose(1, 0, 2), animated=True, origin='lower', interpolation=interpolation, extent=(showFromX, showToX, showFromY, showToY))


        # I have no bloody clue why the impurity and pinning arrays need to be flipped in y to match with the magnetization array but they do
        if plotImpurity:
            impurityArray = np.flip(getImpurityArray(directory, np.array([0, 0, 0, 0.1]), zIndex)[startXIdx:endXIdx, startYIdx:endYIdx].transpose(1, 0, 2), axis=0)
            ax.imshow(impurityArray, extent=(showFromX, showToX, showFromY, showToY))

        if plotPinning:
            pinningArray = np.flip(getPinningArray(directory)[startXIdx:endXIdx, startYIdx:endYIdx].transpose(1, 0, 2), axis=0)
            ax.imshow(pinningArray, extent=(showFromX, showToX, showFromY, showToY))

    else:

        thePlot = ax.imshow(rgbArray.transpose(1, 0, 2), animated=True, origin='lower', interpolation=interpolation, extent=(0, Lx, 0, Ly))
        if plotImpurity:
            impurityArray = np.flip(getImpurityArray(directory, np.array([0, 1, 0, 0.2]), zIndex).transpose(1, 0, 2), axis=0)
            ax.imshow(impurityArray, extent=(0, Lx, 0, Ly))

        if plotPinning:
            pinningArray = np.flip(getPinningArray(directory).transpose(1, 0, 2), axis=0)
            ax.imshow(pinningArray, extent=(0, Lx, 0, Ly))

    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")
    if showComponent: ax.text(Lx + 5, 0, "$m_" + component + "$")
    
    return thePlot


def magnetizationPlot(directory, component, theFile, zIndex=0, plotImpurity=False, plotPinning=False, showComponent=True, interpolation=None, ax = None, **kwargs):

    """ Produces a colour plot (imshow) of the magnetization with component x, y, or z, corresponding to mx, my, mz. """

    if ax == None:
        fig, ax = plt.subplots()

    fullFile = directory + '/' + theFile

    Lx, Ly = sampleExtent(directory)

    # If supplied, cut the plot to show only a certain subset, according to the desired size in metres
    showFromX = kwargs.get('showFromX', None)
    showToX = kwargs.get('showToX', None)
    showFromY = kwargs.get('showFromY', None)
    showToY = kwargs.get('showToY', None)

    mArray = loadFile(fullFile, component, zIndex)

    if showFromX:

        startXIdx = int(np.round((showFromX / Lx) * mArray.shape[0]))
        endXIdx = int(np.round((showToX / Lx) * mArray.shape[0]))
        startYIdx = int(np.round((showFromY / Ly) * mArray.shape[1]))
        endYIdx = int(np.round((showToY / Ly) * mArray.shape[1]))

        mArray = mArray[startXIdx:endXIdx, startYIdx:endYIdx]
        thePlot = ax.imshow(mArray.transpose(), animated=True, vmin=-1, vmax=1, origin="lower", cmap="RdBu_r", interpolation=interpolation, extent=(showFromX, showToX, showFromY, showToY))

        # I have no bloody clue why the impurity and pinning arrays need to be flipped in y to match with the magnetization array but they do
        if plotImpurity:
            impurityArray = np.flip(getImpurityArray(directory, np.array([0, 1, 0, 0.2]), zIndex)[startXIdx:endXIdx, startYIdx:endYIdx].transpose(1, 0, 2), axis=0)
            ax.imshow(impurityArray, extent=(showFromX, showToX, showFromY, showToY))

        if plotPinning:
            pinningArray = np.flip(getPinningArray(directory)[startXIdx:endXIdx, startYIdx:endYIdx].transpose(1, 0, 2), axis=0)
            ax.imshow(pinningArray, extent=(showFromX, showToX, showFromY, showToY))

    else:

        thePlot = ax.imshow(mArray.transpose(), animated=True, vmin=-1, vmax=1, origin="lower", cmap="RdBu_r", interpolation=interpolation, extent=(0, Lx, 0, Ly))
        if plotImpurity:
            impurityArray = np.flip(getImpurityArray(directory, np.array([0, 1, 0, 0.2]), zIndex).transpose(1, 0, 2), axis=0)
            ax.imshow(impurityArray, extent=(0, Lx, 0, Ly))

        if plotPinning:
            pinningArray = np.flip(getPinningArray(directory).transpose(1, 0, 2), axis=0)
            ax.imshow(pinningArray, extent=(0, Lx, 0, Ly))

    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")
    if showComponent: ax.text(Lx + 5, 0, "$m_" + component + "$")
    
    return thePlot


def skyrmionDensityPlot(directory, theFile, zIndex=0, plotImpurity=False, plotPinning=False, interpolation=None, ax=None, maxSkyrmionDensity=None, lengthUnits=None, **kwargs):

    """ Return the plot of the skyrmion number density. """

    if ax == None:
        fig, ax = plt.subplots()

    fullFile = directory + '/' + theFile

    Lx, Ly = sampleExtent(directory)
    dx, dy, dz = sampleDiscretisation(directory)

    # If supplied, cut the plot to show only a certain subset, according to the desired size in metres
    showFromX = kwargs.get('showFromX', None)
    showToX = kwargs.get('showToX', None)
    showFromY = kwargs.get('showFromY', None)
    showToY = kwargs.get('showToY', None)

    mArray = df.Field.fromfile(fullFile).array[:, :, zIndex]
    mArray = mArray.reshape(mArray.shape[0], mArray.shape[1], 3)
    rhoSkArray = skyrmionNumberDensity(mArray, dx, dy, lengthUnits)

    if maxSkyrmionDensity:
        colourMapMax = maxSkyrmionDensity

    else:
        colourMapMax = np.max(np.abs(rhoSkArray))  # For normalisation of the colour map such that white corresponds to zero skyrmion density

    print(np.max(np.abs(rhoSkArray)))

    if showFromX:

        startXIdx = int(np.round((showFromX / Lx) * mArray.shape[0]))
        endXIdx = int(np.round((showToX / Lx) * mArray.shape[0]))
        startYIdx = int(np.round((showFromY / Ly) * mArray.shape[1]))
        endYIdx = int(np.round((showToY / Ly) * mArray.shape[1]))

        rhoSkArray = rhoSkArray[startXIdx:endXIdx, startYIdx:endYIdx]
        thePlot = ax.imshow(rhoSkArray.transpose(), animated=True, vmin=-colourMapMax, vmax=colourMapMax, origin="lower", cmap="PRGn", interpolation=interpolation, extent=(showFromX, showToX, showFromY, showToY))

        # I have no bloody clue why the impurity and pinning arrays need to be flipped in y to match with the magnetization array but they do
        if plotImpurity:
            impurityArray = np.flip(getImpurityArray(directory, np.array([1, 0, 0, 0.2]), zIndex)[startXIdx:endXIdx, startYIdx:endYIdx].transpose(1, 0, 2), axis=0)
            ax.imshow(impurityArray, extent=(showFromX, showToX, showFromY, showToY))

        if plotPinning:
            pinningArray = np.flip(getPinningArray(directory)[startXIdx:endXIdx, startYIdx:endYIdx].transpose(1, 0, 2), axis=0)
            ax.imshow(pinningArray, extent=(showFromX, showToX, showFromY, showToY))

    else:

        thePlot = ax.imshow(rhoSkArray.transpose(), animated=True, vmin=-colourMapMax, vmax=colourMapMax, origin="lower", cmap="PRGn", interpolation=interpolation, extent=(0, Lx, 0, Ly))
        if plotImpurity:
            impurityArray = np.flip(getImpurityArray(directory, np.array([1, 0, 0, 0.2]), zIndex).transpose(1, 0, 2), axis=0)
            ax.imshow(impurityArray, extent=(0, Lx, 0, Ly))

        if plotPinning:
            pinningArray = np.flip(getPinningArray(directory).transpose(1, 0, 2), axis=0)
            ax.imshow(pinningArray, extent=(0, Lx, 0, Ly))

    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")

    return thePlot


def plotSpeedOverSimulations(directories, currentComponent, speedComponent, COMFileName='COM.npy', ax=None, presentationMethod='legend', showStopRamp=False, timeMultiplier=None, speedMultiplier=None, currentMultiplier=None, colorbarLabel=None):

    """ For a given directory, plot the speeds of the skyrmions in the respective sub-directories. """

    if ax is None:
        print('Failed')
        fig, ax = plt.subplots()

    if currentMultiplier:
        directoriesNumerical = [currentMultiplier * float(directory.split('/')[-1]) for directory in directories]

    else:
        directoriesNumerical = [float(directory.split('/')[-1]) for directory in directories]

    if presentationMethod == 'colormap':
        cmap = matplotlib.cm.get_cmap('Reds')
        norm = matplotlib.colors.Normalize(vmin=np.min(directoriesNumerical), vmax=np.max(directoriesNumerical))

    for directory in directories:

        times = simulationTimeArray(directory + '/Data/')
        COM = np.load(directory + '/' + COMFileName)
        speeds = speedAgainstTime(times, COM, speedComponent)
        if timeMultiplier: times *= timeMultiplier
        if speedMultiplier: speeds *= speedMultiplier

        current = directory.split('/')[-1]
        if currentMultiplier: current = str(currentMultiplier * float(current))

        if presentationMethod == 'legend':
            plot = ax.plot(times * 1e9, speeds, label=current)

        elif presentationMethod == 'colormap':
            ax.plot(times * 1e9, speeds, color=cmap(norm(float(current))))

        else:
            raise ValueError('Method of presenting data must be \'legend\' or \'colormap\'.')

    if showStopRamp:
        stopRampIdx = getRampIdx(directories[0] + '/Data/', currentComponent)
        ax.axvline(times[stopRampIdx] * 1e9, color='black')

    if presentationMethod == 'legend':
        ax.legend()

    elif presentationMethod == 'colormap':
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(sm, cax=cax)
        cb.ax.set_ylabel(colorbarLabel, rotation=270, labelpad=30, fontsize=16)
