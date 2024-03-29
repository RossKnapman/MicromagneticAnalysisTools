"""
Tools for calculating quantities from simulation data.
"""

import numpy as np
import discretisedfield as df
import findiff

from MicromagneticAnalysisTools import Read


def skyrmionNumber(m):
    """Calculate the skyrmion number for a magnetization array of shape `(Nx, Ny, 1, 3)`, for a given z.

    Args:
        m (ndarray): Array of shape `(Nx, Ny, 1, 3)`.

    Returns:
        The skyrmion number of the texture in the array.

    """

    mdx = np.diff(m, axis=0)[:, :-1]
    mdy = np.diff(m, axis=1)[:-1, :]
    mReduced = m[:-1, :-1, :]
    #  Element-wise dot product
    toSum = np.einsum('ijk,ijk->ij', mReduced, np.cross(mdx, mdy))
    return (1 / (4 * np.pi)) * np.sum(toSum)


def skyrmionNumberDensity(m, dx, dy, lengthUnits=None):
    """Calculate the skyrmion number density of a magnetic texture contained in an array of shape `(Nx, Ny, 1, 3)`

    Args:
        m (ndarray): Array of shape `(Nx, Ny, 1, 3)`.
        dx (float64): The simulation cell size in the x-dimension, in nm.
        dy (float64): The simulation cell size in the y-dimension, in nm.
        lengthUnits(float64, optional): Amount by which to scale discretisation (which rescales number density).

    Returns:
        Two-dimensional array of skyrmion number densities over the x-y plane.

    """

    # args: axis, lattice constant, derivative order, accuracy
    d_dx = findiff.FinDiff(0, 1, 1, acc=4)
    # args: axis, lattice constant, derivative order, accuracy
    d_dy = findiff.FinDiff(1, 1, 1, acc=4)

    if lengthUnits:
        dx /= lengthUnits
        dy /= lengthUnits

    mdx = d_dx(m) / dx
    mdy = d_dy(m) / dy

    skDensityArray = np.einsum('ijk,ijk->ij', m, np.cross(mdx, mdy))
    return skDensityArray


def skyrmionCOM(directory, inFile, dx, dy, zIndex=0, edgeCutXFrac=0.1, edgeCutYFrac=0.1):
    """Calculate the centre of the skyrmion by obtaining the "centre of mass" of the skyrmion number density.

    Args:
        directory (str): The directory in which the simulation data is stored.
        inFile (str): The filename for which to calculate the skyrmion's centre.
        dx (float64): The simulation cell size in the x-dimension, in nm.
        dy (float64): The simulation cell size in the y-dimension, in nm.
        zIndex (int, optional): The index along the z-axis for which the skyrmion centre should be calculated (for a 3D sample).
        edgeCutXFrac (float64, optional): The fraction of the system along the x-axis that should be cut off at the edges (thus allowing edge effects to be excluded).
        edgeCutYFrac (float64, optional): Same as above but for the y-axis.

    Returns:
        A two-element array of the form `[x-coordinate of centre of skyrmion, y-coordinate of centre of skyrmion]`.

    """

    Lx, Ly = Read.sampleExtent(directory)

    edgeCutX = int(np.round(edgeCutXFrac * Lx))
    edgeCutY = int(np.round(edgeCutYFrac * Ly))

    m = df.Field.fromfile(directory + '/' + inFile).array[:, :, zIndex]
    m = m[edgeCutX:m.shape[0]-edgeCutX, edgeCutY:m.shape[1]-edgeCutY]

    # Define operators
    # args: axis, lattice constant, derivative order, accuracy
    d_dx = findiff.FinDiff(0, 1, 1, acc=4)
    # args: axis, lattice constant, derivative order, accuracy
    d_dy = findiff.FinDiff(1, 1, 1, acc=4)

    # Apply them to m
    mdx = d_dx(m)
    mdy = d_dy(m)

    rhoSk = np.einsum('ijk,ijk->ij', m, np.cross(mdx, mdy))
    Nsk = np.sum(rhoSk)  # Out from the "usual" Nsk by 4pi

    xy = np.indices((m.shape[0], m.shape[1])).transpose(1, 2, 0)
    comArray = np.einsum('ijk,ij->ijk', xy, rhoSk)

    return np.array([(np.sum(comArray[:, :, 0] / Nsk) + edgeCutX) * dx * 1e9, (np.sum(comArray[:, :, 1] / Nsk) + edgeCutY) * dy * 1e9])


def skyrmionCOMArray(directory, edgeCutXFrac=0.1, edgeCutYFrac=0.1, zIndex=0):
    """Get an array of the centre of mass of the skyrmion over the entire simulation.

    Args:
        directory (str): The directory in which the simulation data is stored.
        edgeCutXFrac (float64, optional): The fraction of the system along the x-axis that should be cut off at the edges (thus allowing edge effects to be excluded).
        edgeCutYFrac (float64, optional): Same as above but for the y-axis.
        zIndex (int, optional): The index along the z-axis for which the skyrmion centre should be calculated (for a 3D sample).

    Returns:
        Array of centres of mass over the entire simulation, of the shape `(Number of files, 2)`.

    """

    filesToScan = Read.getFilesToScan(directory)
    dx, dy = Read.sampleDiscretisation(directory)[:2]
    COM = np.zeros((len(filesToScan), 2))

    for i in range(len(filesToScan)):
        print('Calculating skyrmion position', i,
              'of', len(filesToScan) - 1, end='\r')
        COM[i] = skyrmionCOM(directory, filesToScan[i],
            dx, dy, zIndex, edgeCutXFrac, edgeCutYFrac)

    return COM


def skyrmionCOMCoMoving(directory, inFile, dx, dy, zIndex, guessX, guessY, boxSize=None):
    """Calculate the skyrmion's position by converting to a co-moving frame, which allows seamlessness with PBCs. 

    Args:
        directory (str): The directory in which the simulation data is stored.
        inFile (str): The filename for which to calculate the skyrmion's centre.
        dx (float64): The simulation cell size in the x-dimension, in nm.
        dy (float64): The simulation cell size in the y-dimension, in nm.
        zIndex (int, optional): The index along the z-axis for which the skyrmion centre should be calculated (for a 3D sample).
        guessX (float64): Initial guess of the skyrmion's x-position, in cells.
        guessY (float64): Initial guess of the skyrmion's y-position, in cells.
        boxSize (float64, optional): Size of the box around the guess centre of mass in nm, to avoid integrating over the entire system.

    Returns:
        A two-element array of the form `[x-coordinate of centre of skyrmion, y-coordinate of centre of skyrmion]`.

    """

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
    """Get an array of the centre of mass of the skyrmion over the entire simulation=, using a co-moving frame, allowing seamlessness when using PBCs.

    Args:
        directory (str): The directory in which the simulation data is stored.
        zIndex (int, optional): The index along the z-axis for which the skyrmion centre should be calculated (for a 3D sample).
        boxSize (float64, optional): Size of the box around the guess centre of mass in nm, to avoid integrating over the entire system.
        startFile (str, optional): The starting file for which the helicity should be calculated.
        endFile (str, optional): The ending file for which the helicity should be calculated.

    Returns:
        Array of centres of mass over the entire simulation, of the shape `(Number of files, 2)`.

    """
    filesToScan = Read.getFilesToScan(
        directory, startFile=startFile, endFile=endFile)
    dx, dy = Read.sampleDiscretisation(directory)[:2]
    COM = np.zeros((len(filesToScan), 2))
    initialFile = filesToScan[0].split('/')[-1]
    guessX, guessY = skyrmionCOM(directory, initialFile, dx, dy, zIndex=zIndex, edgeCutXFrac=0., edgeCutYFrac=0.)

    for i in range(len(filesToScan)):
        print('Calculating skyrmion position', i,
              'of', len(filesToScan) - 1, end='\r')
        COM[i] = skyrmionCOMCoMoving(
            directory, filesToScan[i], dx, dy, zIndex, guessX, guessY, boxSize)
        guessX, guessY = COM[i]

    return COM


def getRampIdx(directory, component):
    """Get the index of the point in the simulation (e.g. 0 -> m000000.ovf, 1-> m000001.ovf, ...) where the current reaches a constant.
    Note that we assume the current to be monotonically increasing.

    Args:
        directory (str): The directory in which the simulation data is stored.
        component (str): The direction in which the current is running; can be x, y, or z.

    Returns:
        The index at which the current stops ramping.

    """

    currentArray = Read.simulationCurrentArray(directory, component)
    return np.searchsorted(currentArray, 0.999999999*np.max(currentArray))


def speedAgainstTime(times, COM, component):
    """Return the speed of the skyrmion as an array.

    Args:
        times (ndarray): Times array of simulation, calculated using :py:func:`MicromagneticAnalysisTools.Read.simulationTimeArray()`.
        COM (ndarray): The array of centre positions of the skyrmion, calculated using :py:func:`MicromagneticAnalysisTools.Calculate.skyrmionCOMArray` or :py:func:`MicromagneticAnalysisTools.Calculate.skyrmionCOMArrayCoMoving`.
        component (str): The component for which the speed should be measured (x or y).

    Returns:
        An array of speed against time in the specified direction.

    """

    if component == 'x':
        return np.gradient(COM[:, 0] * 1e-9, times)

    elif component == 'y':
        return np.gradient(COM[:, 1] * 1e-9, times)

    else:
        raise ValueError('Component must by "x" or "y".')


def getMeanSpeed(times, COM, rampIdx, component):
    """Get the mean x-component of the skyrmion velocity over time. Only consider values obtained after ramping has finished.

    Args:
        times (ndarray): Times array of simulation, calculated using :py:func:`MicromagneticAnalysisTools.Read.simulationTimeArray()`.
        COM (ndarray): The array of centre positions of the skyrmion, calculated using :py:func:`MicromagneticAnalysisTools.Calculate.skyrmionCOMArray` or :py:func:`MicromagneticAnalysisTools.Calculate.skyrmionCOMArrayCoMoving`.
        rampIdx (int): Index at which the current ramping stops, calculated usign :py:func:`MicromagneticAnalysisTools.Calculate.getRampIdx`.
        component (str): The component for which the speed should be measured (x or y).

    Returns:
        The average speed along the specified direction over the course of the simulation.

    """

    return np.average(speedAgainstTime(times[rampIdx:], COM[rampIdx:], component))


def NskArray(directory, zIndex=0, startFile=None, endFile=None, calculateFromX=None, calculateToX=None, calculateFromY=None, calculateToY=None, startXFrac=0, endXFrac=1, startYFrac=0, endYFrac=1):
    """Get the Nsk of the simulation over time. The arguments define the area of the sample over which the Nsk array should be calculated. Can either specify in terms of units (i.e. nanometres),
    of in terms of the fractional length of the system.

    Args:
        directory (str): The directory in which the simulation data is stored.
        zIndex (int, optional): The index along the z-axis for which the skyrmion centre should be calculated (for a 3D sample).
        startFile (str, optional): The simulation file from which the counting should start (i.e. skipping over files at the beginning).
        endFile (str, optional): The simulation at which the counting shold end.
        calculateFromX (float, optional): The value of x in nm from which the skyrmion number calculation should start (e.g. cutting off edge effects).
        calcualteToX(float, optional): The value of x in nm to which the skyrmion number calculation should start (e.g. cutting off edge effects).
        calculateFromY (float, optional): As above, but in the y-direction.
        calculateToY (float, optional): As above, but in the y-direction.
        startXFrac (float, optional): Like above, but with fractions rather than nm.
        endXFrac (float, optional): Like above, but with fractions rather than nm.
        startYFrac (float, optional): Like above, but with fractions rather than nm.
        startYFrac (float, optional): Like above, but with fractions rather than nm.

    Returns:
        A 1D array of the skyrmion number over the course of the simulation, for each file.

    """

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
    """Calculate the Hopf index of a magnetization vector field of shape `(Nx, Ny, Nz, 3)`.

    Args:
        m (ndarray): Magnetization array of the shape `(Nx, Ny, Nz, 3)`.
        acc (int, optional): Order of the accuracy for which the derivatives should be calculated (see `findiff documentation <https://findiff.readthedocs.io/en/latest/index.html>`_).

    Returns:
        The calculated Hopf index of the texture.
    
    """
    
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
    """Calculate the Hopf index density of a magnetization vector field of shape `(Nx, Ny, Nz, 3)`.

    Args:
        m (ndarray): Magnetization array of the shape `(Nx, Ny, Nz, 3)`.
        acc (int, optional): Order of the accuracy for which the derivatives should be calculated (see `findiff documentation <https://findiff.readthedocs.io/en/latest/index.html>`_).

    Returns:
        Array of Hopf index density of the system, of the shape `(Nx, Ny, Nz)`.
    
    """

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


def skyrmion_helicity(directory, filename):

    """Calculate the skyrmion helicity from an ovf file.
    The calculation works by sweeping down the y-axis, and getting the points to the left and right of the
    centre, which are closest to m_z = 0 (where we define the radius of the skyrmion). We then take the mean
    helicity of all of these points on the radius. We assume that the system is a thin film, i.e. that the
    system only has one layer along the z-axis.

    IMPORTANT: We assume a single skyrmion in a collinear background!
    
    Args:
        directory (str): The directory containing the ovf file for which the helicity should be calculated.
        filename (str): The ovf file containing the skyrmion for which the helicity should be calculated.

    Returns:
        The calculated helicity (between -pi and pi).

    """

    # Load the magnetization array
    m = df.Field.fromfile(directory + '/' + filename).array

    # Read the cell size
    dx, dy, dz = Read.sampleDiscretisation(directory)

    # Calculate the centre of mass of the skyrmion
    com = skyrmionCOM(directory, filename, dx, dy, edgeCutXFrac=0., edgeCutYFrac=0.)

    # Get the indices of the cells closest to the centre of mass
    central_x_idx = int(np.round(com[0] / (dx*1e9)))
    central_y_idx = int(np.round(com[1] / (dy*1e9)))

    # Get the core polarization of the skyrmion
    polarization = -1 if m[central_x_idx, central_y_idx, 0, 2] < 0 else 1

    # List to store points on the radius (as indices of array)
    radius_points = []

    # Loop through y-values, getting x-values that are on the skyrmion boundary (thanks to Robin Msiska for the inspration)
    for y_idx in range(m.shape[1]):
    
        # Check that there are parts of this line that are actually within the skyrmion radius (defined by m_z = 0)
        if np.any(np.sign(m[:, y_idx, 0, 2]) == np.sign(polarization)):

            # Get points to left and right of centre of skyrmion, on its radius
            radius_points.append([np.argmin(np.abs(m[:central_x_idx, y_idx, 0, 2])), y_idx])
            radius_points.append([central_x_idx + np.argmin(np.abs(m[central_x_idx:, y_idx, 0, 2])), y_idx])

    # Get helicities for each point
    helicities = np.zeros(len(radius_points), dtype=float)

    for i in range(len(helicities)):

        x_idx = radius_points[i][0]
        y_idx = radius_points[i][1]

        # Get x- and y-position of point in nm
        x = x_idx * dx * 1e9
        y = y_idx * dy * 1e9

        # Get displacements from centre of skyrmion
        x_displacement = x - com[0]
        y_displacement = y - com[1]

        # Calculate polar coordinate in plane
        plane_angle = np.arctan2(y_displacement, x_displacement)
        
        # Calculate polar coordinate of in-plane components of magnetization
        phi = np.arctan2(m[x_idx, y_idx, 0, 1], m[x_idx, y_idx, 0, 0])

        # Save helicity values
        helicities[i] = phi - plane_angle

    # Deal with helicities that are outside of the range [-pi, pi]
    for i in range(len(helicities)):
        while helicities[i] > np.pi:
            helicities[i] -= 2*np.pi            
        while helicities[i] < -np.pi:
            helicities[i] += 2*np.pi

    # Below, deal with the case that the helicity is close to pi, such that some values have
    # helicity ≈ -pi and other have helicity ≈ pi

    # If absolute value of helicity is close to pi, and there is a mixture of positive and negative helicities
    if np.abs(np.mean(np.abs(helicities)) - np.pi) < np.pi/4 and np.any(helicities > 0) and np.any(helicities < 0):

        # Instead get angle w.r.t. pi direction (-x), to avoid discontinous point at helicity = pi or -pi
        anglesToMinusX = np.zeros_like(helicities, dtype=float)

        for i in range(len(anglesToMinusX)):

            if helicities[i] >= 0:
                anglesToMinusX[i] = np.pi - helicities[i]
            
            else:
                anglesToMinusX[i] = -np.pi - helicities[i]
                
        averageAngleToMinusX = np.mean(anglesToMinusX)

        if averageAngleToMinusX >= 0:
            return np.pi - averageAngleToMinusX
        
        else:
            return -np.pi - averageAngleToMinusX

    else:
        return np.average(helicities)


def skyrmion_helicity_array(directory, startFile=None, endFile=None):

    """Get an array of skyrmion helicities for a given simulation directory.
    Note that this only works for a single skyrmion in a collinear background.

    Args:
        directory (str): The directory containing the ovf file for which the helicity should be calculated.
        startFile (str, optional): The starting file for which the helicity should be calculated.
        endFile (str, optional): The ending file for which the helicity should be calculated.

    Returns:
        Array of calculated helicities.

    """

    filesToScan = Read.getFilesToScan(directory, startFile, endFile)

    helicities = np.zeros(len(filesToScan), dtype=float)

    for i in range(len(filesToScan)):

        print("Calculating helicity", i, "of", len(filesToScan) - 1, end='\r')
        helicities[i] = skyrmion_helicity(directory, filesToScan[i])

    return helicities
