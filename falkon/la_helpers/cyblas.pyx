# cython: language_level=3
import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

cimport scipy.linalg.cython_lapack

"""
Compilation instructions

$ cython -a cyblas.pyx
$ gcc -shared -pthread -fopenmp -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing \
      -I$CONDA_ENV_ROOT/include/python3.7m \
      -I$CONDA_ENV_ROOT/lib/python3.7/site-packages/numpy/core/include \
      -o cyblas.so cyblas.c
"""

class BlasError(Exception):
    pass


ctypedef fused floating:
    float
    double


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void _fill_diagonal(floating[:,:] matrix, floating[:] diagonal) nogil:
    """Equivalent to the `np.fill_diagonal` function"""
    cdef int i = 0
    for i in range(matrix.shape[0]):
        matrix[i,i] = diagonal[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void _copy_diagonal(floating[:,:] matrix, floating[:] diagonal) nogil:
    """Equivalent to `diagonal = np.diagonal(matrix).copy()`"""
    cdef int i = 0
    cdef int shape = diagonal.shape[0]
    for i in range(shape):
        diagonal[i] = matrix[i,i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def vec_mul_triang(floating[:,:] array,
                   floating[:] multiplier,
                   bint upper,
                   int side):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    if cols != rows:
        raise ValueError("Input matrix to vec_mul_triang must be square.")
    if cols != multiplier.shape[0]:
        raise ValueError("Multiplier shape mismatch. Expected %d found %d" %
                         (cols, multiplier.shape[0]))

    cdef floating mul
    cdef int i, j

    if array.is_f_contig(): # Column-contiguous
        if upper and side == 1:  # upper=1, side=1
            for j in prange(cols, nogil=True, schedule='guided', chunksize=max(cols//1000, 30)):
                mul = multiplier[j]
                for i in range(j + 1):
                    array[i, j] *= mul
        elif upper and side == 0:  # upper=1, side=0
            for j in prange(cols, nogil=True, schedule='guided', chunksize=max(cols//1000, 30)):
                for i in range(j + 1):
                    mul = multiplier[i]
                    array[i, j] *= mul
        elif side == 1:  # upper=0, side=1
            for j in prange(cols, nogil=True, schedule='guided', chunksize=max(cols//1000, 30)):
                mul = multiplier[j]
                for i in range(j, rows):
                    array[i, j] *= mul
        else: # upper=0, side=0
            for j in prange(cols, nogil=True, schedule='guided', chunksize=max(cols//1000, 30)):
                for i in range(j, rows):
                    mul = multiplier[i]
                    array[i, j] *= mul
    elif array.is_c_contig():
        if upper and side == 1:  # upper=1, side=1
            for i in prange(rows, nogil=True, schedule='guided', chunksize=max(rows//1000, 30)):
                for j in range(i, cols):
                    mul = multiplier[j]
                    array[i, j] *= mul
        elif upper and side == 0:  # upper=1, side=0
            for i in prange(rows, nogil=True, schedule='guided', chunksize=max(rows//1000, 30)):
                mul = multiplier[i]
                for j in range(i, cols):
                    array[i, j] *= mul
        elif side == 1:  # upper=0, side=1
            for i in prange(rows, nogil=True, schedule='guided', chunksize=max(rows//1000, 30)):
                for j in range(i + 1):
                    mul = multiplier[j]
                    array[i, j] *= mul
        else:  # upper=0, side=0
            for i in prange(rows, nogil=True, schedule='guided', chunksize=max(rows//1000, 30)):
                mul = multiplier[i]
                for j in range(i + 1):
                    array[i, j] *= mul
    else:
        raise ValueError("Matrix is not memory-contiguous")

    return array.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int _mul_triang(floating[:,:] array,
                     floating[:] diag,
                     bint f_contig,
                     bint upper,
                     bint preserve_diag,
                     floating multiplier) nogil:
    cdef int size = array.shape[1]
    cdef int info = 0
    cdef char arr_type
    if f_contig:
        arr_type = b'U' if upper else b'L'
    else:
        arr_type = b'L' if upper else b'U'

    # KL, KU are not used
    cdef int KL = 0
    cdef int KU = 0
    cdef floating CFROM = 1
    cdef floating CTO = multiplier

    if preserve_diag:
        _copy_diagonal(array, diag)
    if floating is double:
        scipy.linalg.cython_lapack.dlascl(&arr_type, &KL, &KU, &CFROM, &CTO, &size, &size, &array[0,0], &size, &info)
    elif floating is float:
        scipy.linalg.cython_lapack.slascl(&arr_type, &KL, &KU, &CFROM, &CTO, &size, &size, &array[0,0], &size, &info)
    if preserve_diag:
        _fill_diagonal(array, diag)

    return info


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mul_triang(floating[:,:] array,
               bint upper,
               bint preserve_diag,
               floating multiplier):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    cdef int info = 0
    cdef bint f_contig = array.is_f_contig()
    cdef floating[:] diag

    if cols != rows:
        raise ValueError("Input matrix to lascl must be square.")
    if not f_contig and not array.is_c_contig():
        raise ValueError("Array is not contiguous.")

    if preserve_diag:
        diag = np.empty_like(array, shape=rows)
    else:
        diag = None
    with nogil:
        info = _mul_triang(array, diag, f_contig, upper, preserve_diag, multiplier)
    if info != 0:
        raise BlasError("LAPACK lascl failed with status %s" % (str(info)))

    return array.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def potrf(floating[:,:] array, bint upper, bint clean, bint overwrite):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    cdef int info1 = 0
    cdef int info2 = 0
    if cols != rows:
        raise ValueError("Input matrix to potrf must be square.")

    cdef floating[:] temp_diag_buf = np.empty_like(array, shape=rows)
    cdef char arr_type
    cdef bint f_contig = array.is_f_contig()
    if f_contig:
        arr_type = b'U' if upper else b'L'
    elif array.is_c_contig():
        arr_type = b'L' if upper else b'U'
    else:
        raise ValueError("Array is not contiguous.")

    # Copy array if necessary
    if not overwrite:
        array = np.copy(array, order="A")

    # Run Cholesky Factorization
    with nogil:
        if floating is double:
            scipy.linalg.cython_lapack.dpotrf(&arr_type, &rows, &array[0, 0], &rows, &info1)
        elif floating is float:
            scipy.linalg.cython_lapack.spotrf(&arr_type, &rows, &array[0, 0], &rows, &info1)
        # Clean non-factorized part of the matrix
        if clean:
            info2 = _mul_triang(array, temp_diag_buf, f_contig, not upper, True, 0.0)

    if info1 != 0:
        raise BlasError(
            "LAPACK potrf failed with status %s. Params: uplo %s , rows %d" %
            (str(info1), str(arr_type), rows))
    if info2 != 0:
        raise BlasError("LAPACK lascl failed with status %s." % (str(info2)))

    return array.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def copy_triang(floating[:,:] array, bint upper):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    if cols != rows:
        raise ValueError("Input array to copy_triang must be square.")

    cdef bint fin_upper, transpose
    fin_upper = upper
    transpose = False
    if array.is_f_contig():
        upper = not upper
        array = array.T
        transpose = True

    cdef int i, j
    with nogil:
        if upper:
            for i in prange(cols, nogil=False, schedule='guided', chunksize=max(rows//1000, 1)):
                for j in range(0, i):
                    array[i, j] = array[j, i]
        else:
            for i in prange(cols - 1, -1, -1, nogil=False, schedule='guided', chunksize=max(rows//1000, 1)):
                for j in range(i+1, rows):
                    array[i, j] = array[j, i]

    if transpose:
        return array.T.base
    return array.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def add_symmetrize(floating[:,:] array):
    cdef int rows = array.shape[0]
    cdef int cols = array.shape[1]
    if cols != rows:
        raise ValueError("Input array to copy_triang must be square.")

    cdef bint transpose = False
    if array.is_f_contig():
        array = array.T
        transpose = True

    cdef int i, j
    cdef floating temp
    for i in prange(cols, nogil=True, schedule='guided', chunksize=max(rows//1000, 1)):
        for j in range(0, i):
            temp = array[i, j]
            array[i, j] += array[j, i]
            array[j, i] += temp
        array[i, i] *= 2

    if transpose:
        return array.T.base
    return array.base

