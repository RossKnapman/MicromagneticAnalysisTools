import numpy as np
import discretisedfield as df
import findiff

from MicromagneticAnalysisTools import Read


def skyrmionNumber(m):
    """ Get the skyrmion number given a magnetisation array of shape (Nx, Ny, 1, 3) for a given z.
    This vectorised calculation is ~1000 faster than two for loops for getting the derivatives. """

    mdx = np.diff(m, axis=0)[:, :-1]
    mdy = np.diff(m, axis=1)[:-1, :]
    mReduced = m[:-1, :-1, :]
    #  Element-wise dot product
    toSum = np.einsum('ijk,ijk->ij', mReduced, np.cross(mdx, mdy))
    return (1 / (4 * np.pi)) * np.sum(toSum)


def skyrmionNumberDensity(m, dx, dy, lengthUnits=None):
    """ Get the skyrmion number given a magnetisation array of shape (Nx, Ny, 1, 3) for a given z.
    This vectorised calculation is ~1000 faster than two for loops for getting the derivatives. """

    # args: axis, lattice constant, derivative order, accuracy
    d_dx = findiff.FinDiff(0, 1, 1, acc=8)
    # args: axis, lattice constant, derivative order, accuracy
    d_dy = findiff.FinDiff(1, 1, 1, acc=8)

    if lengthUnits:
        dx /= lengthUnits
        dy /= lengthUnits

    mdx = d_dx(m) / dx
    mdy = d_dy(m) / dy

    skDensityArray = np.einsum('ijk,ijk->ij', m, np.cross(mdx, mdy))
    return skDensityArray


def skyrmionCOM(directory, inFile, edgeCutXFrac, edgeCutYFrac, dx, dy, zIndex):

    Lx, Ly = Read.sampleExtent(directory)

    edgeCutX = int(np.round(edgeCutXFrac * Lx))
    edgeCutY = int(np.round(edgeCutYFrac * Ly))

    m = df.Field.fromfile(directory + '/' + inFile).array[:, :, zIndex]
    m = m[edgeCutX:m.shape[0]-edgeCutX, edgeCutY:m.shape[1]-edgeCutY]

    # Define operators
    # args: axis, lattice constant, derivative order, accuracy
    d_dx = findiff.FinDiff(0, 1, 1, acc=8)
    # args: axis, lattice constant, derivative order, accuracy
    d_dy = findiff.FinDiff(1, 1, 1, acc=8)

    # Apply them to m
    mdx = d_dx(m)
    mdy = d_dy(m)

    rhoSk = np.einsum('ijk,ijk->ij', m, np.cross(mdx, mdy))
    Nsk = np.sum(rhoSk)  # Out from the "usual" Nsk by 4pi

    xy = np.indices((m.shape[0], m.shape[1])).transpose(1, 2, 0)
    comArray = np.einsum('ijk,ij->ijk', xy, rhoSk)

    return np.array([(np.sum(comArray[:, :, 0] / Nsk) + edgeCutX) * dx * 1e9, (np.sum(comArray[:, :, 1] / Nsk) + edgeCutY) * dy * 1e9])


def skyrmionCOMArray(directory, edgeCutXFrac=0.1, edgeCutYFrac=0.1, zIndex=0):
    """ Gets an array of the x, y position of the skyrmion centre of mass over the course of the simulation. """

    filesToScan = Read.getFilesToScan(directory)
    dx, dy = Read.sampleDiscretisation(directory)[:2]
    COM = np.zeros((len(filesToScan), 2))

    for i in range(len(filesToScan)):
        print('Calculating skyrmion position', i,
              'of', len(filesToScan) - 1, end='\r')
        COM[i] = skyrmionCOM(directory, filesToScan[i],
                             edgeCutXFrac, edgeCutYFrac, dx, dy, zIndex)

    return COM


def skyrmionCOMCoMoving(directory, inFile, dx, dy, zIndex, guessX, guessY, boxSize=None):
    """ Calculate the skyrmion's position by converting to a co-moving frame, which allows seamlessness with PBCs. """

    Lx = Read.sampleExtent(directory)[0]

    m = df.Field.fromfile(directory + '/' + inFile).array[:, :, zIndex]

    # The center of the grid is:
    centerX = int(np.round(0.5 * m.shape[0]))
    centerY = int(np.round(0.5 * m.shape[1]))

    sX = int(np.round(centerX-guessX))
    sY = int(np.round(centerY-guessY))
    m = np.roll(m, [sX, sY], axis=(0, 1))
    xy = np.indices((m.shape[0], m.shape[1])).transpose(1, 2, 0)

    if boxSize:  # The size of the box around the guessed COM in nm to reduce computational resources rather than integration over entire system. This does not really seem to give much of a performance boost.
        # Convert size to array indices
        boxSizeIdx = int(np.round(m.shape[0] * boxSize / Lx))

        leftIdx = int(np.round(m.shape[0]/2 - boxSizeIdx))
        rightIdx = int(np.round(m.shape[0]/2 + boxSizeIdx))
        bottomIdx = int(np.round(m.shape[1]/2 - boxSizeIdx))
        topIdx = int(np.round(m.shape[1]/2 + boxSizeIdx))

        m = m[leftIdx:rightIdx, bottomIdx:topIdx]
        xy = xy[leftIdx:rightIdx, bottomIdx:topIdx]

    # Define operators
    # args: axis, lattice constant, derivative order, accuracy
    d_dx = findiff.FinDiff(0, 1, 1, acc=8)
    # args: axis, lattice constant, derivative order, accuracy
    d_dy = findiff.FinDiff(1, 1, 1, acc=8)

    # Apply them to m
    mdx = d_dx(m)
    mdy = d_dy(m)

    rhoSk = np.einsum('ijk,ijk->ij', m, np.cross(mdx, mdy))
    Nsk = np.sum(rhoSk)

    comArray = np.einsum('ijk,ij->ijk', xy, rhoSk)

    COMX = np.sum(comArray[:, :, 0]) / Nsk
    COMY = np.sum(comArray[:, :, 1]) / Nsk

    return np.array([(COMX-sX) * dx * 1e9, (COMY-sY) * dy * 1e9])


def skyrmionCOMArrayCoMoving(directory, zIndex=0, boxSize=None, startFile=None, endFile=None):

    filesToScan = Read.getFilesToScan(
        directory, startFile=startFile, endFile=endFile)
    dx, dy = Read.sampleDiscretisation(directory)[:2]
    COM = np.zeros((len(filesToScan), 2))
    initialFile = filesToScan[0].split('/')[-1]
    guessX, guessY = skyrmionCOM(directory, initialFile, 0, 0, dx, dy, zIndex)

    for i in range(len(filesToScan)):
        print('Calculating skyrmion position', i,
              'of', len(filesToScan) - 1, end='\r')
        COM[i] = skyrmionCOMCoMoving(
            directory, filesToScan[i], dx, dy, zIndex, guessX, guessY, boxSize)
        guessX, guessY = COM[i]

    return COM


def getRampIdx(directory, component):
    """ Get the index of the point in the simulation (i.e. 0 -> m000000.ovf, 1-> m000001.ovf, ...) where the current reaches a constant. Note that we assume the current to be monotonically increasing. """

    currentArray = Read.simulationCurrentArray(directory, component)

    return np.searchsorted(currentArray, 0.999999999*np.max(currentArray))


def speedAgainstTime(times, COM, component):
    """ Return the speed of the skyrmion as an array. """

    if component == 'x':
        return np.gradient(COM[:, 0] * 1e-9, times)

    elif component == 'y':
        return np.gradient(COM[:, 1] * 1e-9, times)

    else:
        raise ValueError('Component must by "x" or "y".')


def getMeanSpeed(times, COM, rampIdx, component):
    """ Get the mean x-component of the skyrmion velocity over time. Only consider values obtained after ramping has finished. """

    return np.average(speedAgainstTime(times[rampIdx:], COM[rampIdx:], component))


def NskArray(directory, zIndex=0, startFile=None, endFile=None, calculateFromX=None, calculateToX=None, calculateFromY=None, calculateToY=None, startXFrac=0, endXFrac=1, startYFrac=0, endYFrac=1):
    """ Get the Nsk of the simulation over time. The arguments define the area of the sample over which the Nsk array should be calculated. Can either specify in terms of units (i.e. nanometres),
    of in terms of the fractional length of the system. """

    filesToScan = Read.getFilesToScan(directory, startFile, endFile)

    Nsk = np.zeros(len(filesToScan))

    for i in range(len(filesToScan)):

        print("Calculating Nsk", i, "of", len(filesToScan) - 1, end='\r')

        m = df.Field.fromfile(
            directory + '/' + filesToScan[i]).array[:, :, zIndex]

        if calculateFromX:
            Lx, Ly = Read.sampleExtent(directory)
            startXFrac = calculateFromX / Lx
            endXFrac = calculateToX / Lx
            startYFrac = calculateFromY / Ly
            endYFrac = calculateToY / Ly

        startX = int(np.round(startXFrac * m.shape[0]))
        endX = int(np.round(endXFrac * m.shape[0]))
        startY = int(np.round(startYFrac * m.shape[1]))
        endY = int(np.round(endYFrac * m.shape[1]))

        m = m[startX:endX, startY:endY]
        Nsk[i] = skyrmionNumber(m)

    return Nsk


def HopfIdx(m, acc=8):

    """ Calculate the Hopf index of a magnetization vector field of shape (x_length, y_length, z_length, 3). """
    
    d_dx = findiff.FinDiff(0, 1, 1, acc=acc)  # args: axis, lattice constant, derivative order, accuracy
    d_dy = findiff.FinDiff(1, 1, 1, acc=acc)
    d_dz = findiff.FinDiff(2, 1, 1, acc=acc)

    mdx = d_dx(m)
    mdy = d_dy(m)
    mdz = d_dz(m)

    Fx = 2 * np.einsum('ijkl,ijkl->ijk', m, np.cross(mdy, mdz))
    Fy = 2 * np.einsum('ijkl,ijkl->ijk', m, np.cross(mdz, mdx))
    Fz = 2 * np.einsum('ijkl,ijkl->ijk', m, np.cross(mdx, mdy))

    F = np.zeros((Fx.shape[0], Fx.shape[1], Fx.shape[2], 3))
    F[:, :, :, 0] = Fx
    F[:, :, :, 1] = Fy
    F[:, :, :, 2] = Fz

    A = np.zeros((Fz.shape[0], Fz.shape[1], Fz.shape[2], 3))
    A[:, :, :, 0] = -np.cumsum(Fz, axis=1)  # Cumulative sum along y-axis
    A[:, :, :, 2] = np.cumsum(Fx, axis=1)

    dotProduct = np.einsum('ijkl,ijkl->ijk', F, A)
    hopfIdx = -np.sum(dotProduct) / (8 * np.pi)**2
    return(hopfIdx)


def Hopf_density(m, acc=8):
    
    """ Calculate the Hopf index density of a magnetization vector field of shape (x_length, y_length, z_length, 3). """

    d_dx = findiff.FinDiff(0, 1, 1, acc=acc)  # args: axis, lattice constant, derivative order, accuracy
    d_dy = findiff.FinDiff(1, 1, 1, acc=acc)
    d_dz = findiff.FinDiff(2, 1, 1, acc=acc)

    mdx = d_dx(m)
    mdy = d_dy(m)
    mdz = d_dz(m)

    Fx = 2 * np.einsum('ijkl,ijkl->ijk', m, np.cross(mdy, mdz))
    Fy = 2 * np.einsum('ijkl,ijkl->ijk', m, np.cross(mdz, mdx))
    Fz = 2 * np.einsum('ijkl,ijkl->ijk', m, np.cross(mdx, mdy))

    F = np.zeros((Fx.shape[0], Fx.shape[1], Fx.shape[2], 3))
    F[:, :, :, 0] = Fx
    F[:, :, :, 1] = Fy
    F[:, :, :, 2] = Fz

    A = np.zeros((Fz.shape[0], Fz.shape[1], Fz.shape[2], 3))
    A[:, :, :, 0] = -np.cumsum(Fz, axis=1)  # Cumulative sum along y-axis
    A[:, :, :, 2] = np.cumsum(Fx, axis=1)

    dotProduct = np.einsum('ijkl,ijkl->ijk', F, A)
    return dotProduct