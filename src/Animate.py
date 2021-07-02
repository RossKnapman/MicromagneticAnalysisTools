import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
import discretisedfield as df
import re
from src import Read
from src import Plot
from src import Calculate


def oneDQuantity(quantityArray, timeArray, yLabel, quantityTextValue, quantityTextFormat="{:.2f}", outName="Movie.mp4", initialMinValue=0, initialMaxValue=0):
    """ Outputs an animation of a generic quantity (e.g. Nsk, energy, current) over time. The input arrays are to be given as file names, e.g. Nsk.npy, Time.npy. """

    t = timeArray * 1e9

    fig, ax = plt.subplots()

    thePlot, = ax.plot(t[0], quantityArray[0], animated=True)

    plt.xlim(0, 0)
    plt.xlabel(r"$t$ (ns)")
    plt.ylabel(yLabel)

    minValue = initialMinValue
    maxValue = initialMaxValue

    quantityText = ax.text(0, maxValue, "")
    timeText = ax.text(0.1 * np.max(timeArray), maxValue, "")

    def init():
        pass

    def updateAnim(i):

        print("Animating value", i, "of", len(quantityArray), end='\r')

        nonlocal maxValue
        nonlocal minValue

        try:
            if np.min(quantityArray[:i]) < minValue:
                minValue = np.min(quantityArray[:i])
            if np.max(quantityArray[:i]) > maxValue:
                maxValue = np.max(quantityArray[:i])

            quantityText.set_text(
                quantityTextValue + " = " + quantityTextFormat.format(quantityArray[i]))
            quantityText.set_position((0.05 * t[i], 0.3 * maxValue))
            timeText.set_text('t = ' + '{:.2f}'.format(t[i]) + ' ns')
            timeText.set_position((0.05 * t[i], 0.5 * maxValue))

        except ValueError:
            pass

        ax.set_xlim(t[0], t[i])
        ax.set_ylim(minValue, 1.1*maxValue)
        thePlot.set_data(t[:i+1], quantityArray[:i+1])
        ax.get_xaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: '{:.1f}'.format(x)))

    dataIterator = iter(range(len(quantityArray)))

    print("Producing animation", outName)

    anim = animation.FuncAnimation(fig, updateAnim, dataIterator, init_func=init,
                                   interval=25, blit=False, save_count=len(quantityArray))
    plt.tight_layout()

    anim.save(outName, fps=25, writer='ffmpeg')


def NskAnimation(NskArray, timeArray):
    oneDQuantity(NskArray, timeArray, r"$N_{\mathrm{sk}}$", r"$N_{\mathrm{sk}}$",
                 outName="Nsk.mp4", initialMinValue=-1.1, initialMaxValue=1.1)


def currentAnimation(currentArray, timeArray, component):
    oneDQuantity(currentArray, timeArray, "$j_" + component + "$ (A m$^{-2}$)", "$j_" + component + "$",
                 quantityTextFormat="{:.2e}", outName="Current" + component + ".mp4", initialMinValue=0, initialMaxValue=1)


def energyAnimation(energyArray, timeArray):
    oneDQuantity(energyArray, timeArray, "$E$ (J)", "$E$",
                 quantityTextFormat="{:.2e}", outName="Energy.mp4", initialMinValue=energyArray[0], initialMaxValue=0.9 * energyArray[0])


def colourAnimation(directory, plotType, zIndex=0, COMArray=None, plotImpurity=False, plotPinning=False, showComponent=True, quiver=False, step=1, showFromX=None, showToX=None, showFromY=None, showToY=None, edgeCutXFrac=None, edgeCutYFrac=None,
                    startXFrac=None, endXFrac=None, startYFrac=None, endYFrac=None, startFile=None, endFile=None, lengthUnits=None, **kwargs):
    """ Create an animation of a 2D scalar field. """

    # A lot of this code could probably be merged with the skyrmion density animation plot to make it more modular

    filesToScan = Read.getFilesToScan(directory, startFile, endFile)
    assert (len(filesToScan) != 0)

    component = kwargs.get('component')
    if plotType == 'magnetization':
        assert component == 'x' or component == 'y' or component == 'z'

    fig, ax = plt.subplots()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Correct centre of mass array start is startFile is not the first file
    if startFile:
        correctedIdx = int(re.findall(r'\d+', startFile)[0])

    if plotType == 'magnetization':
        thePlot = Plot.magnetizationPlot(directory, component, filesToScan[0], zIndex, plotImpurity, plotPinning,
                                         showComponent, ax=ax, showFromX=showFromX, showToX=showToX, showFromY=showFromY, showToY=showToY)
        comColour = 'green'

    elif plotType == 'magnetizationHSL':
        showComponent = False
        thePlot = Plot.magnetizationPlotHSL(directory, component, filesToScan[0], zIndex, plotImpurity, plotPinning,
                                            showComponent, ax=ax, showFromX=showFromX, showToX=showToX, showFromY=showFromY, showToY=showToY)
        comColour = 'green'

    elif plotType == 'skyrmionDensity':
        thePlot = Plot.skyrmionDensityPlot(directory, filesToScan[0], zIndex, plotImpurity, plotPinning,
                                           ax=ax, showFromX=showFromX, showToX=showToX, showFromY=showFromY, showToY=showToY)
        comColour = 'red'

    if quiver:
        quiverPlot = Plot.magnetizationQuiver(
            directory, filesToScan[0], ax=ax, step=step, showFromX=showFromX, showToX=showToX, showFromY=showFromY, showToY=showToY)

    if COMArray:
        comArray = np.load(COMArray)
        comTracer = Circle(
            (comArray[0, 0], comArray[0, 1]), 10, color=comColour, alpha=0.8)
        ax.add_patch(comTracer)

    Lx, Ly = Read.sampleExtent(directory)

    if showFromX:
        fileText = ax.text(0.5 * (showToX - showFromX) + showFromX,
                           1.1 * (showToY - showFromY) + showFromY, "")
        timeText = ax.text(
            showFromX, 1.1 * (showToY - showFromY) + showFromY, "")

    else:
        fileText = ax.text(0.5*Lx, 1.1*Ly, "")
        timeText = ax.text(0, 1.1*Ly, "")

    if edgeCutXFrac and edgeCutYFrac:
        startXFrac = edgeCutXFrac
        endXFrac = 1. - edgeCutXFrac
        startYFrac = edgeCutYFrac
        endYFrac = 1. - edgeCutYFrac

    if startXFrac and endXFrac and startYFrac and endYFrac:
        rect = Rectangle((startXFrac * Lx, startYFrac * Ly), (endXFrac - startXFrac)
                         * Lx, (endYFrac - startYFrac) * Ly, fill=False, color='lime', lw=2)
        ax.add_patch(rect)

    def init():
        pass

    def updateAnim(i):

        print("Producing animation frame", i, "of", len(filesToScan), end='\r')

        fullFile = directory + '/' + filesToScan[i]

        fileText.set_text('File: ' + filesToScan[i])
        timeText.set_text(
            '$t$ = ' + "{:.2f}".format(Read.fileTime(fullFile) * 1e9) + " ns")

        if plotType == 'magnetization':
            mArray = Read.loadFile(fullFile, component, zIndex)

        elif plotType == 'magnetizationHSL':
            mArray = df.Field.fromfile(fullFile).array[:, :, zIndex, :]

        elif plotType == 'skyrmionDensity':
            mArray = df.Field.fromfile(fullFile).array[:, :, zIndex]
            mArray = mArray.reshape(mArray.shape[0], mArray.shape[1], 3)

        if showFromX:
            # Should probably be in a separate function as various things use this
            startXIdx = int(np.round((showFromX / Lx) * mArray.shape[0]))
            endXIdx = int(np.round((showToX / Lx) * mArray.shape[0]))
            startYIdx = int(np.round((showFromY / Ly) * mArray.shape[1]))
            endYIdx = int(np.round((showToY / Ly) * mArray.shape[1]))
            mArray = mArray[startXIdx:endXIdx, startYIdx:endYIdx]

        if quiver:
            mArrayQuiver = df.Field.fromfile(fullFile).array[:, :, zIndex, :]
            mArrayQuiver = mArrayQuiver.reshape(
                mArrayQuiver.shape[0], mArrayQuiver.shape[1], 3)

            if showFromX:
                mArrayQuiver = mArrayQuiver[startXIdx:endXIdx,
                                            startYIdx:endYIdx]

            quiverPlot.set_UVC(mArrayQuiver[::step, ::step, 0].transpose(
            ), mArrayQuiver[::step, ::step, 1].transpose())

        if plotType == 'magnetization':
            thePlot.set_array(mArray.transpose())

        elif plotType == 'magnetizationHSL':
            rgbArray = Plot.vecToRGB(mArray)
            print(rgbArray.transpose(1, 0, 2).shape)
            thePlot.set_array(rgbArray.transpose(1, 0, 2))

        elif plotType == 'skyrmionDensity':
            dx, dy = Read.sampleDiscretisation(directory)[:2]
            thePlot.set_array(Calculate.skyrmionNumberDensity(
                mArray, dx, dy, lengthUnits).transpose())

        comIdx = i

        if COMArray:

            if startFile:
                comIdx = i + correctedIdx

            tracerX, tracerY = comArray[comIdx]

            # For when the skyrmion goes over the boundary
            while tracerX >= Lx:
                tracerX -= Lx

            while tracerX < 0:
                tracerX += Lx

            while tracerY >= Ly:
                tracerY -= Ly

            while tracerY < 0:
                tracerY += Ly

            comTracer.center = [tracerX, tracerY]

    dataIterator = iter(range(len(filesToScan)))

    anim = animation.FuncAnimation(
        fig, updateAnim, dataIterator, init_func=init, interval=25, blit=False, save_count=len(filesToScan))

    if plotType == 'magnetization':
        if startFile or showFromX:
            # After so much time generating the animation for the full simulation, don't want to overwrite it when looking at part of the simulation
            outName = "m" + component + "Part" + ".mp4"

        else:
            outName = "m" + component + ".mp4"

    elif plotType == 'magnetizationHSL':
        if startFile or showFromX:
            # After so much time generating the animation for the full simulation, don't want to overwrite it when looking at part of the simulation
            outName = "mHSLPart.mp4"

        else:
            outName = "mHSL.mp4"

    elif plotType == 'skyrmionDensity':
        if startFile or showFromX:
            # After so much time generating the animation for the full simulation, don't want to overwrite it when looking at part of the simulation
            outName = "SkDensityPart.mp4"

        else:
            outName = "SkDensity.mp4"

    anim.save(outName, fps=25, writer='ffmpeg')
