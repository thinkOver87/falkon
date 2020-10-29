""" Handles various initialization routines for GPU """
import torch

import falkon.cuda.cusolver_gpu as cusolver
from falkon.cuda.cublas_gpu import cublasCreate
from falkon.options import BaseOptions
from falkon.utils.devices import get_device_info

__all__ = ("init", "cublas_handle", "cusolver_handle")


def _normalize_device(device):
    # Normalize input device
    if isinstance(device, torch.device):
        if device.type != 'cuda':
            raise RuntimeError(
                "CuBLAS handle only exists on CUDA devices. Given CPU device %s" % (device))
        device = device.index
    elif device is not None:
        device = int(device)
        if device < 0:
            raise RuntimeError(
                "Device index must be greater or equal than 0. Given index was %d" % (device))
    else:
        # Device is None
        device = torch.cuda.current_device()

    return device


_cublas_handles = {}
_cusolver_handles = {}


def cublas_handle(device=None):
    device = _normalize_device(device)
    global _cublas_handles
    try:
        return _cublas_handles[device]
    except KeyError:
        raise RuntimeError(
            "Device %d is not initialized properly. CuBLAS handle missing." % (device))


def cusolver_handle(device=None):
    device = _normalize_device(device)
    global _cusolver_handles
    try:
        return _cusolver_handles[device]
    except KeyError:
        raise RuntimeError(
            "Device %d is not initialized properly. CuSOLVER handle missing." % (device))


def init(opt: BaseOptions):
    if opt.use_cpu:
        return

    device_ids = [k for k in get_device_info(opt).keys() if k >= 0]

    global _cublas_handles
    global _cusolver_handles
    for i in device_ids:
        with torch.cuda.device(i):
            # CuBLAS handle
            if _cublas_handles.get(i, None) is None:
                handle = cublasCreate()
                _cublas_handles[i] = handle
            # CuSOLVER (Dense) handle
            if _cusolver_handles.get(i, None) is None:
                handle = cusolver.cusolverDnCreate()
                _cusolver_handles[i] = handle
