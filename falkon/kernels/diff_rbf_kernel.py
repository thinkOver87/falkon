import functools
import warnings
from typing import Union, Optional

import torch

from falkon import FalkonOptions
from falkon.kernels import GaussianKernel
from falkon.kernels.tiling_red import TilingGenred
from falkon.mmv_ops.keops import _decide_backend, _keops_dtype
from falkon.utils.helpers import check_same_device


class DiffGaussianKernel(GaussianKernel):
    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        super().__init__(sigma, opt)

    @staticmethod
    def _get_sigma_kt(sigma: torch.Tensor):
        return sigma, "single"

    def _sigma2gamma(self, sigma: torch.Tensor):
        return sigma

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt: FalkonOptions):
        formula = 'Exp(SqDist(x1 / g, x2 / g) * IntInv(-2)) * v'
        aliases = [
            'x1 = Vi(%d)' % (X1.shape[1]),
            'x2 = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'g = Pm(%d)' % (self.gamma.shape[0]),
        ]
        other_vars = [self.gamma.to(device=X1.device, dtype=X1.dtype)]

        # Choose backend
        N, D = X1.shape
        backend = _decide_backend(opt, D)
        dtype = _keops_dtype(X1.dtype)
        device = X1.device

        if not check_same_device(X1, X2, v, out, *other_vars):
            raise RuntimeError("All input tensors must be on the same device.")
        if (device.type == 'cuda') and (not backend.startswith("GPU")):
            warnings.warn("KeOps backend was chosen to be CPU, but GPU input tensors found. "
                          "Defaulting to 'GPU_1D' backend. To force usage of the CPU backend, "
                          "please pass CPU tensors; to avoid this warning if the GPU backend is "
                          "desired, check your options (i.e. set 'use_cpu=False').")
            backend = "GPU_1D"

        func = TilingGenred(formula, aliases, reduction_op='Sum', axis=1, dtype=dtype,
                            dtype_acc="auto", sum_scheme="auto", opt=opt)
        return func(X1, X2, v, *other_vars, out=out, backend=backend)

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        return self._keops_mmv_impl

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)

    def _prepare(self, X1, X2):
        gamma = self.gamma.to(device=X1.device, dtype=X1.dtype)
        return super()._prepare(X1.div(gamma), X2.div(gamma))

    def _apply(self, X1, X2, out):
        gamma = self.gamma.to(device=X1.device, dtype=X1.dtype)
        if X1.shape[0] == gamma.shape[0]:
            X1 = X1.div(gamma)
            X2 = (X2.T.div(gamma)).T
        else:
            X1 = X1.div(gamma)
            X2 = (X2.T.div(gamma)).T
        return super()._apply(X1, X2, out)

    def _transform(self, A) -> torch.Tensor:
        A.mul_(-0.5)
        A.exp_()
        return A

    def __repr__(self):
        return f"DiffGaussianKernel(sigma={self.sigma})"

    def __str__(self):
        return f"DiffGaussian kernel<{self.sigma}>"
