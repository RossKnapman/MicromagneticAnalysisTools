from json import load
import numpy as np
from numba import cuda
from cmath import exp, pi
from math import ceil
import discretisedfield as df
from MicromagneticAnalysisTools import Read
import random


def getTime(file):

    """
    Reads an OVF file to obtain its time.

    Args:
        file (str): The input filename.

    Returns:
        The total simulation time of the file.

    """

    with open(file, "rb") as ovffile:

        for line in ovffile:
            if b'Total simulation time' in line:
                return float(line.split()[5])


def read_part_file(t_index, magnetization_array, directory, z_index, files_to_scan, dimensions):

    """
    Reads a slice of a file with a given z-index at time index i. I wrote this function
    instead of using discretisedfield.Field.fromfile as this allows reading only a slice
    of the file, rather reading the full 3D file and only afterwards slicing it, so loads
    faster.

    Args:
        if i < len(files_to_scan):  # The last thread may have fewer than batch_size entries to process
        t_idx (int): The time index of the file to be read.
        magnetization_array (ndarray): Array of dimensions ((Number of times) x Lx x Ly x 3) to be filled by this function.
        directory (str): The directory in which the simulation files are stored.
        z_index (int): The index of the z-slice we want to calculate.
        files_to_scan (List[str]): List of the filenames over the different times.
        dimensions (List[int]): x- and y-dimensions of the file, i.e. [Lx, Ly].

    """
    
    print('Reading file', t_index, 'of', len(magnetization_array), end='\r')
    fileName = directory + '/' + files_to_scan[t_index]

    Lx, Ly = dimensions

    with open (fileName, "rb") as f:

        # Ensure OVF 2.0 (as I hard-coded for OVF 2.0 only; for more flexibility, see discretisedfield _fromovf() function)
        assert b'OVF 2.0' in next(f)

        # Skip over header
        for line in f:
            line = line.decode("utf-8")

            if line.startswith("# Begin: Data"):
                break

        # Skip test value (that is to test byte order; see https://math.nist.gov/oommf/doc/userguide12b4/userguide/OVF_2.0_format.html)
        f.read(4)

        # Read z-slice (OVF file is stored such that z increments slowest, hence we just need to specify offset and count)
        magnetization_array[t_index, :, :, :] = np.fromfile(
                        f, count=int(Lx*Ly*3), dtype=f"<f", offset=z_index*3*Lx*Ly*np.dtype(np.float32).itemsize
                    ).reshape((Lx, Ly, 3)).transpose(1, 0, 2)


@cuda.jit
def calculate_time_sums(diffArray, timesArray, frequenciesArray, resultArray):

    """ The CUDA kernel called by Psr_GPU. Called by each thread on the GPU. """

    # Absolute position of current thread (x- and y-axes correspond to those of magnetic system; z-axis corresponds to frequencies)
    i, j, k = cuda.grid(3)

    # Only calculate if thread index is within system we are calculating
    if i < diffArray.shape[1] and j < diffArray.shape[2] and k < len(frequenciesArray):

        componentSum = 0.
        for component_idx in range(3):  # Sum over components x, y, z

            timeSum = 0.
            for time_idx in range(diffArray.shape[0]):  # Sum over times
                timeSum += diffArray[time_idx, i, j, component_idx] * exp(-2*pi*1j * frequenciesArray[k] * timesArray[time_idx])

            componentSum += abs(timeSum)**2

        resultArray[k, i, j] = componentSum


def Psr_GPU(frequencies, directory, relaxedStateFile, blocksize, z_index=0, load_method='part_file'):

    """
    Calculate the spatially-resolved power spectral density (P_sr) for a magnetic texture, accelerated on the GPU
    (in my experience, ~10x faster than CPU-based calculation optimised with NumPy). For a 3D texture, this function
    should be called for each 2D slice along the z-axis and the results added together.

    Args:
        frequencies (ndarray): Array of frequencies for which P_sr should be calculated.
        directory (str): Directory in which simulation OVF files are stored.
        relaxedStateFile (str): The OVF file containing the initial, relaxed state (with file file path).
        blocksize (int): The calculation will be processed using (blocksize x blocksize) threads per block.
        z_index (int): The z-index of slice of the system we want to calculate P_sr for.
        load_method (str): The method used for loading the files, either discretisedfield, which can handle
            various versions of OVF and does some basic testing, or part_file, which does not do any testing
            etc., but only loads part of the file (rather than the entire thing then slicing it), so is faster
            for 3D systems.

    """

    relaxedStateArray = df.Field.fromfile(relaxedStateFile).array[:, :, z_index, :]

    Lx = relaxedStateArray.shape[0]
    Ly = relaxedStateArray.shape[1]
    N  = Lx*Ly

    files_to_scan = Read.getFilesToScan(directory)[:10]
    timesArray = np.zeros(len(files_to_scan))

    magnetization_over_time = np.zeros((len(timesArray), Lx, Ly, 3), dtype=float)

    if load_method == 'discretisedfield':

        for file_idx in range(len(files_to_scan)):
            print("Loading file", file_idx+1, "of", len(files_to_scan), end='\r')
            magnetization_over_time[file_idx, :, :, :] = df.Field.fromfile(directory + '/' + files_to_scan[file_idx]).array[:, :, z_index, :]
            timesArray[file_idx] = getTime(directory + '/' + files_to_scan[file_idx])

    elif load_method == 'part_file':

        for file_idx in range(len(files_to_scan)):
            read_part_file(file_idx, magnetization_over_time, directory, z_index, files_to_scan, [Lx, Ly])
            timesArray[file_idx] = getTime(directory + '/' + files_to_scan[file_idx])

    else:
        raise ValueError('load_method must either be discretisedfield (uses the external discretisedfield \
module to load the file) or part_file (reads only slice of OVF file).')

    # All loaded vectors should have a norm of 1 (otherwise wasn't loaded correctly)
    assert np.all(np.isclose(np.linalg.norm(magnetization_over_time, axis=3), 1))

    diffArray = magnetization_over_time - relaxedStateArray

    blockdim = (blocksize, blocksize)  # Dimensions of blocks (in threads)
    griddim  = (ceil(Lx/blocksize), ceil(Ly/blocksize), len(frequencies))  # Dimensions of grid (in blocks)

    # Copy magnetization difference, times, and frequencies arrays to GPU
    diffArray_global_mem        = cuda.to_device(diffArray)
    timesArray_global_mem       = cuda.to_device(timesArray)
    frequenciesArray_global_mem = cuda.to_device(frequencies)

    # Allocate memory in GPU to store result, of dimensions (number of frequencies, Lx, Ly))
    resultArray_global_mem = cuda.device_array((len(frequencies), diffArray.shape[1], diffArray.shape[2]))

    print('\nCalculating Psr on GPU...')

    # Call function to calculate on GPU
    calculate_time_sums[griddim, blockdim](diffArray_global_mem, timesArray_global_mem, frequenciesArray_global_mem, resultArray_global_mem)

    # Copy result array back from GPU to host
    resultArray = resultArray_global_mem.copy_to_host()

    return np.sum(resultArray, axis=(1, 2)) / N


if __name__ == '__main__':

    blocksize = 16  # There are (blocksize x blocksize) threads per block (playing around I found 16 to be best for 128x128 sample)
    z = random.randint(0, 127)  #  Pick a random z-index (for which file is hopefully not already loaded into CPU cache) for testing speed
    directory = "/scratch/ttausend/skyrmion_timed/skyrmion_excitation/tube_skyrmion/outputs/Out_of_plane"
    relaxedStateFile = "/scratch/ttausend/skyrmion_timed/skyrmion_excitation/tube_skyrmion/Initial_config/Skyrmiontube_relaxed.ovf"
    frequencies = np.arange(0, 50, 0.01) * 1e9

    # Test that my loading method produces the same results as discretisedfield
    Psrpart = Psr_GPU(frequencies, directory, relaxedStateFile, blocksize, z_index=z, load_method='part_file')
    Psrdf = Psr_GPU(frequencies, directory, relaxedStateFile, blocksize, z_index=z, load_method='discretisedfield')

    print(np.all(Psrpart == Psrdf))  # Should be True
