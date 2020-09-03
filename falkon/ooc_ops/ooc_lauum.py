import math
import threading
from typing import List, Optional

from falkon.cuda import initialization
from falkon.utils import devices, PropagatingThread
from falkon.utils.tensor_helpers import copy_same_stride
from falkon.utils.helpers import sizeof_dtype
from falkon.options import FalkonOptions, LauumOptions
from .ooc_utils import calc_block_sizes3
from .parallel_lauum import par_lauum_f_lower, par_lauum_c_lower, BlockAlloc
from ..utils.tensor_helpers import is_f_contig, is_contig

__all__ = ("gpu_lauum",)


def _parallel_lauum_runner(A, write_opposite: bool, opt: LauumOptions, gpu_info):
    # Choose target:
    if is_f_contig(A):
        target = par_lauum_f_lower
    elif is_contig(A):
        target = par_lauum_c_lower
    else:
        raise NotImplementedError("Parallel LAUUM is only implemented for contiguous matrices")

    N = A.shape[0]
    dt = A.dtype
    dts = sizeof_dtype(dt)
    if A.is_cuda:
        gpu_info = [g for g in gpu_info if g.Id == A.device.index]
        avail_ram = gpu_info[0].actual_free_mem / dts
        if target.__name__ == "par_lauum_f_lower":
            # Each GPU should hold in memory two additional blocks (2*B^2 <= M)
            max_block_size = int(math.floor(math.sqrt(avail_ram / 2)))
        else:
            # Same RAM requirements as the non-CUDA version
            max_block_size = int(math.floor((-2 * N + math.sqrt(4 * N**2 + 8 * avail_ram)) / 4))
        if max_block_size < 1:
            raise RuntimeError(
                "Cannot run parallel LAUUM with minimum "
                "available memory of %.2fMB" % (avail_ram * dts / 2**20))
        # All computations on the same device (where data is stored). No multi-GPU support!
        block_sizes = calc_block_sizes3(max_block_size, 1, N)
    else:
        avail_ram = min([g.actual_free_mem for g in gpu_info]) / dts
        # Each GPU should be able to hold in memory 2 block columns
        # Plus two blocks (=> quadratic equation 2B^2 + 2BN - M <= 0
        max_block_size = int(math.floor((-2 * N + math.sqrt(4 * N**2 + 8 * avail_ram)) / 4))
        if max_block_size < 1:
            raise RuntimeError(
                "Cannot run parallel LAUUM with minimum "
                "available memory of %.2fMB" % (avail_ram * dts / 2**20))

        block_sizes = calc_block_sizes3(max_block_size, len(gpu_info), N)

    # Create BlockAlloc objects describing the subdivision of input
    block_allocations: List[BlockAlloc] = []
    cur_n = 0
    for bs in block_sizes:
        block_allocations.append(BlockAlloc(start=cur_n, end=cur_n + bs, length=bs))
        cur_n += bs

    num_gpus = len(gpu_info)
    if num_gpus < 1:
        raise ValueError("Parallel LAUUM can only run when a GPU is available.")
    barrier = threading.Barrier(num_gpus, timeout=1000)
    threads = []
    for g in gpu_info:
        gid_allocs = [i for i in range(len(block_allocations)) if i % num_gpus == g.Id]
        cublas_handle = initialization.cublas_handle(g.Id)
        if cublas_handle is None:
            raise RuntimeError("CUBLAS must be initialized "
                               "on device %d before running parallel LAUUM." % (g.Id))
        t = PropagatingThread(target=target, name="GPU-%d" % (g.Id), args=(
            A, block_allocations, gid_allocs, barrier, g.Id, cublas_handle, write_opposite))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return A


def gpu_lauum(A, upper, overwrite=True, write_opposite=False, opt: Optional[FalkonOptions] = None):
    """
    Parameters
    -----------
    A : torch.Tensor [N, N]
        2D positive-definite matrix that will be factorized as
        A = U.T @ U (if `upper` is True) or A = L @ L.T if `upper`
        is False.
    overwrite : bool
        Whether to overwrite matrix A or to output the result in a new
        buffer.
    """
    if opt is None:
        opt = FalkonOptions()
    if not overwrite:
        A = copy_same_stride(A, pin_memory=True)
    gpu_info = [v for k, v in devices.get_device_info(opt).items() if k >= 0]
    for g in gpu_info:
        g.actual_free_mem = min((g.free_memory - 300 * 2 ** 20) * 0.95,
                                opt.max_gpu_mem * 0.95)

    # Parallel can only do lower C or F-contiguous arrays
    # By transposing as necessary, it is able to run with every combination of inputs.
    transposed = False
    # noinspection PyUnresolvedReferences
    if upper:
        A = A.T
        transposed = True

    # The parallel runner chooses based on the contiguity pattern of the inputs.
    _parallel_lauum_runner(A, write_opposite, opt, gpu_info)

    if transposed:
        A = A.T
    return A
