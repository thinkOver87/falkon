import dataclasses
import time

import numpy as np
import torch

import falkon
from falkon.options import FalkonOptions
from falkon.center_selection import UniformSelector, FixedSelector
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.hypergrad.common import AbsHypergradModel
from falkon.hypergrad.hypergrad import compute_hypergrad
from falkon.optim import FalkonConjugateGradient, ConjugateGradient



class FalkonHO(AbsHypergradModel):
    def __init__(self, M, maxiter, Xtr, Ytr, Xts, Yts, opt):
        super().__init__()
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Xts = Xts
        self.Yts = Yts

        self.M = M
        self.maxiter = maxiter
        self.opt = opt

        centers = UniformSelector(np.random.default_rng(10)).select(self.Xtr, None, M)
        self.center_selection = FixedSelector(centers)

    def inner_opt(self, params, hparams):
        """This is NOT DIFFERENTIABLE"""
        alpha = params[0]
        penalty, sigma = hparams

        kernel = DiffGaussianKernel(sigma.detach(), self.opt)
        def sq_err(y_true, y_pred):
            return torch.mean((y_true - y_pred)**2)
        self.flk = self.flk_class(
            kernel,
            torch.exp(-penalty.detach()).item(),
            self.M,
            center_selection=self.center_selection,
            maxiter=self.maxiter,
            seed=129,
            options=self.opt)
        self.flk.fit(self.Xtr, self.Ytr)
        return [self.flk.alpha_]

    def val_loss(self, params, hparams):
        alpha = params[0]
        penalty, sigma = hparams
        kernel = DiffGaussianKernel(sigma, self.opt)
        ny_points = self.flk.ny_points_
        preds = kernel.mmv(self.Xts, ny_points, alpha)
        return torch.mean((preds - self.Yts) ** 2)

    def val_loss_grads(self, params, hparams):
        o_loss = self.val_loss(params, hparams)
        return (
            torch.autograd.grad(o_loss, params, allow_unused=True, create_graph=False,
                                retain_graph=True),
            torch.autograd.grad(o_loss, hparams, allow_unused=True, create_graph=False,
                                retain_graph=False)
        )

    def param_derivative(self, params, hparams):
        """Derivative of the training loss, with respect to the parameters"""
        alpha = params[0]
        penalty, sigma = hparams
        N = self.Xtr.shape[0]
        ny_points = self.flk.ny_points_

        kernel = DiffGaussianKernel(sigma, self.opt)

        # 2/N * (K_MN(K_NM @ alpha - Y)) + 2*lambda*(K_MM @ alpha)
        out = (kernel.mmv(ny_points, self.Xtr, kernel.mmv(self.Xtr, ny_points, alpha, opt=self.opt) - self.Ytr, opt=self.opt) +
                torch.exp(-penalty) * N * kernel.mmv(ny_points, ny_points, alpha, opt=self.opt))
        return [out]

    def mixed_vector_product(self, hparams, first_derivative, vector):
        return torch.autograd.grad(first_derivative, hparams, grad_outputs=vector,
                                   allow_unused=True)

    def hessian_vector_product(self, params, first_derivative, vector):
        N = self.Xtr.shape[0]
        ny_points = self.flk.ny_points_
        kernel = self.flk.kernel
        penalty = torch.tensor(self.flk.penalty)
        vector = vector[0]

        out = ((1/N) * kernel.mmv(ny_points, self.Xtr, kernel.mmv(self.Xtr, ny_points, vector, opt=self.opt), opt=self.opt) + \
                torch.exp(-penalty) * kernel.mmv(ny_points, ny_points, vector, opt=self.opt))
        return [out]

    def solve_hessian(self, params, hparams, vector, max_iter, cg_tol):
        penalty = torch.exp(-torch.tensor(self.flk.penalty))

        opt = dataclasses.replace(self.opt, cg_tolerance=cg_tol)
        vector = vector[0].detach()
        N = self.Xtr.shape[0]

        cg = ConjugateGradient(opt)
        kernel = self.flk.kernel
        precond = self.flk.precond

        start_time = time.time()
        B = precond.apply_t(vector / N)
        def mmv(sol):
            v = precond.invA(sol)
            cc = kernel.dmmv(self.Xtr, self.flk.ny_points_, precond.invT(v), None)
            return precond.invAt(precond.invTt(cc / N) + penalty * v)
        d = cg.solve(X0=None, B=B, mmv=mmv, max_iter=max_iter)
        c = precond.apply(d)
        elapsed = time.time() - start_time

        return [c], max_iter, elapsed / max_iter

    def to(self, device):
        self.Xtr = self.Xtr.to(device)
        self.Ytr = self.Ytr.to(device)
        self.Xts = self.Xts.to(device)
        self.Yts = self.Yts.to(device)
        return self

    @property
    def flk_class(self):
        if self.Xtr.device.type == 'cuda':
            return falkon.InCoreFalkon
        return falkon.Falkon

    @property
    def model(self):
        return self.flk



def run_falkon_hypergrad(data,
                         falkon_M,
                         falkon_maxiter,
                         falkon_opt,
                         outer_lr,
                         outer_steps,
                         hessian_cg_steps,
                         hessian_cg_tol,
                         callback,
                         debug):
    import time
    Xtr, Ytr, Xts, Yts = data['Xtr'], data['Ytr'], data['Xts'], data['Yts']

    n, d = Xtr.size()
    t = Ytr.size(1)
    dt, dev = Xtr.dtype, Xtr.device

    hparams = [
        torch.tensor(12, requires_grad=True, dtype=dt, device=dev),  # Penalty
        torch.tensor([1] * d, requires_grad=True, dtype=dt, device=dev),  # Sigma
    ]
    params = [torch.zeros(falkon_M, t, requires_grad=True, dtype=dt, device=dev)]

    outer_opt = torch.optim.Adam(lr=outer_lr, params=hparams)
    flk_helper = FalkonHO(falkon_M, falkon_maxiter, Xtr, Ytr, Xts, Yts, falkon_opt)

    hparam_history = []
    val_loss_history = []
    hgrad_history = []
    for o_step in range(outer_steps):
        # Run inner loop to get alpha_*
        i_start = time.time()
        params = flk_helper.inner_opt(params, hparams)
        if debug:
            if callback is not None:
                callback(flk_helper.model)
            print()
        inner_opt_t = time.time()

        outer_opt.zero_grad()
        hgrad_out = compute_hypergrad(params, hparams, model=flk_helper,
                                      cg_steps=hessian_cg_steps, cg_tol=hessian_cg_tol, set_grad=True,
                                      timings=debug)
        outer_opt.step()
        hparams[0].data.clamp_(min=1e-10)
        hparams[1].data.clamp_(min=1e-10)
        i_end = time.time()
        if debug:
            print("[%4d/%4d] - time %.2fs (inner-opt %.2fs, outer-opt %.2fs)" % (o_step, outer_steps, i_end - i_start, inner_opt_t - i_start, i_end - inner_opt_t))
            #print("GRADIENT", hgrad_out[1])
            print("NEW HPARAMS", hparams)
            print("NEW VAL LOSS", hgrad_out[0])
        hparam_history.append([h.detach().clone() for h in hparams])
        val_loss_history.append(hgrad_out[0])
        hgrad_history.append([g.detach() for g in hgrad_out[1]])

    return hparam_history, val_loss_history, hgrad_history, flk_helper.model
