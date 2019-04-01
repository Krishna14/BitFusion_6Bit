import numpy as np

def cprint(file, func, message):
    print(str(file)+" <- "+str(func)+" "+str(message))

def bindigits(n, bits):
    s = bin(n & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)

def getNameString(unit, row, col):
    return unit+"_"+str(row)+"_"+str(col)

def pad(array, reference_shape, offsets):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """

    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

def align_num_to_byte(data):
    # assumed that after accumulating from a single column
    # max output = 255 * 255 * 16 < 2^20 bits
    temp_data = data
    data_byte_array = []
    data_byte_array.append(temp_data & 0xff)
    temp_data = temp_data >> 8

    data_byte_array.append(temp_data & 0xff)
    temp_data = temp_data >> 8

    data_byte_array.append(temp_data & 0xff)

    # more than this not allowed
    temp_data = temp_data >> 8
    assert temp_data == 0, 'utils.py - align_num_to_byte - data greater than 24 bits'
    return data_byte_array

def twos_complement(n, bits=8):
    return (1 << bits) - n