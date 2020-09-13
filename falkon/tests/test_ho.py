import abc
from functools import partial
from typing import Tuple, Callable, List, Iterable, Sequence

import numpy as np
import torch

from pykeops.torch.generic.generic_red import GenredAutograd, Genred
from sklearn.kernel_ridge import KernelRidge

from falkon.tests.cg_torch import cg
from falkon.tests import cg_torch

import falkon
from falkon import FalkonOptions
from falkon.optim import ConjugateGradient
from falkon.center_selection import UniformSelector, FixedSelector
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.kernels.tiling_red import TilingGenred
from falkon.hypergrad.hypergrad import compute_hypergrad
from falkon.hypergrad.falkon_ho import FalkonHO
from falkon.hypergrad.krr_ho import KRR

""" TEST HO FOR KRR """


def gen_data(n, d, seed=2, test_amount=50, dtype=torch.float32):
    torch.manual_seed(seed)

    w_oracle = torch.randn(d, dtype=dtype)
    X = torch.randn(n, d, dtype=dtype)
    Y = X @ w_oracle + 0.8 * torch.randn(n)
    Y = (Y > 0.).to(dtype=dtype)  # binary classification output
    Y = Y.reshape(-1, 1)
    X = X.to(dtype=dtype)

    Xtr = X[:n - test_amount]
    Xts = X[n - test_amount:]
    Ytr = Y[:n - test_amount]
    Yts = Y[n - test_amount:]

    return Xtr, Ytr, Xts, Yts


def test_krr_ho():
    # Generate some data
    n = 100
    d = 3
    inner_lr = 1e-4
    outer_lr = 1
    inner_steps = 100
    outer_steps = 200
    hessian_cg_steps = 30
    hessian_cg_tol = 1e-4
    Xtr, Ytr, Xts, Yts = gen_data(n, d)

    hparams = [
        torch.tensor(1e-1, requires_grad=True, dtype=torch.float32),  # Penalty
        torch.tensor(0.2, requires_grad=True, dtype=torch.float32),  # Sigma
    ]
    outer_opt = torch.optim.SGD(lr=outer_lr, params=hparams)
    krr_helper = KRR(inner_lr, Xtr, Ytr, Xts, Yts)

    for o_step in range(outer_steps):
        # Run inner loop to get alpha_*
        params = [torch.zeros(Xtr.shape[0], 1, requires_grad=True, dtype=torch.float32)]
        for i_step in range(inner_steps):
            params = krr_helper.inner_fp_map(params, hparams)

        outer_opt.zero_grad()
        # hgrad_out = my_hypergrad(params, hparams, val_loss=krr_helper.val_loss,
        #                          fp_map=krr_helper.inner_fp_map,
        #                          cg_steps=hessian_cg_steps, cg_tol=hessian_cg_tol, set_grad=True)
        hgrad_out = my_hypergrad(params, hparams, krr=krr_helper,
                                 cg_steps=hessian_cg_steps, cg_tol=hessian_cg_tol, set_grad=True)
        outer_opt.step()
        hparams[0].data.clamp_(min=1e-10)
        hparams[1].data.clamp_(min=1e-10)
        print("VAL LOSS", hgrad_out['val_loss'])
        print("GRADIENT", hgrad_out['h_grads'])
        print("NEW HPARAMS", hparams)
        print()


def test_flk_ho():
    # FURTHER IMPROVEMENTS:
    # - Restart falkon from existing alpha.
    # - profile different parts of the code. Check if a manual gradient computation (an easy one, e.g. one of the val_loss_gradients) is faster than autograd.
    # - derive the computational complexity of one outer iteration
    #
    # Generate some data
    n = 1000
    d = 3
    M = 100
    outer_lr = 0.6
    outer_steps = 200
    hessian_cg_steps = 20
    hessian_cg_tol = 1e-4
    Xtr, Ytr, Xts, Yts = gen_data(n, d, test_amount=100, dtype=torch.float32)
    Xtr[:,0] *= 1
    Xts[:,0] *= 1

    hparams = [
        torch.tensor(1e-1, requires_grad=True, dtype=Xtr.dtype),  # Penalty
        torch.tensor([2, 2, 2], requires_grad=True, dtype=Xtr.dtype),  # Sigma
    ]
    outer_opt = torch.optim.Adam(lr=outer_lr, params=hparams)
    flk_helper = FalkonHO(M, Xtr, Ytr, Xts, Yts)

    hparam_history = []
    val_loss_history = []
    hgrad_history = []
    import time

    params = [torch.zeros(M, 1, requires_grad=True, dtype=torch.float32)]
    for o_step in range(outer_steps):
        # Run inner loop to get alpha_*
        i_start = time.time()
        params = flk_helper.inner_opt(params, hparams)
        inner_opt_t = time.time()

        outer_opt.zero_grad()
        hgrad_out = compute_hypergrad(params, hparams, model=flk_helper,
                                 cg_steps=hessian_cg_steps, cg_tol=hessian_cg_tol, set_grad=True,
                                 timings=True)
        outer_opt.step()
        hparams[0].data.clamp_(min=1e-10)
        hparams[1].data.clamp_(min=1e-10)
        i_end = time.time()
        print("Iteration took %.2fs (inner-opt %.2fs, outer-opt %.2fs)" % (i_end - i_start, inner_opt_t - i_start, i_end - inner_opt_t))
        print("GRADIENT", hgrad_out[1])
        print("NEW HPARAMS", hparams)
        print("NEW VAL LOSS", hgrad_out[0])
        print()
        hparam_history.append([h.detach().clone() for h in hparams])
        val_loss_history.append(hgrad_out[0])
        hgrad_history.append([g.detach() for g in hgrad_out[1]])

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(range(len(val_loss_history)), val_loss_history)
    # ax.set_xlabel("Iteration")
    # ax.set_ylabel("Validation loss")

    # fig, ax = plt.subplots()
    # for i in range(len(hparam_history[0])):
    #     ax.plot(range(len(hparam_history)), [h[i] for h in hparam_history])
    # ax.set_xlabel("Iteration")
    # ax.set_ylabel("Hyperparams")
    plt.show()



""" TEST KEOPS AUTOGRAD """
def test_keops_autograd():
    # Generate some data
    n = 1000
    d = 3
    Xtr, Ytr, Xts, Yts = gen_data(n, d, dtype=torch.float64)
    v = torch.rand_like(Ytr)
    dt = torch.float64
    Xtr = Xtr.to(dtype=dt)
    Ytr = Ytr.to(dtype=dt)
    v = v.to(dtype=dt)

    # Run with keops autodiff
    formula = 'Exp(g * SqDist(x1, x2)) * v'
    aliases = [
        'x1 = Vi(%d)' % (Xtr.shape[1]),
        'x2 = Vj(%d)' % (Xtr.shape[1]),
        'v = Vj(%d)' % (Ytr.shape[1]),
        'g = Pm(1)'
    ]
    other_vars = [torch.tensor([0.1], dtype=Xtr.dtype, requires_grad=True)]
    backend = "CPU"
    kv = torch.empty_like(Ytr)#.requires_grad_()
    from falkon.mmv_ops.keops import _keops_dtype
    my_routine = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=_keops_dtype(dt))
    vars = [Xtr, Xtr, Ytr] + other_vars
    o_kv = my_routine(*vars, out=kv, backend=backend)
    grad = torch.autograd.grad(o_kv, other_vars, grad_outputs=v)
    print("KeOps Gradient", grad)

    # Run with normal torch autodiff
    K = torch.norm(Xtr, p=2, dim=1, keepdim=True).pow_(2) + \
        torch.norm(Xtr, p=2, dim=1, keepdim=True).pow_(2).T - \
            2 * (Xtr @ Xtr.T)
    K = K * other_vars[0]
    K = torch.exp(K)
    kv = K @ Ytr
    grad = torch.autograd.grad(kv, other_vars, grad_outputs=v)
    print("PyTorch Autodiff Gradient", grad)

    # Last test
    routine2 = TilingGenred(formula, aliases, reduction_op='Sum', axis=1, dtype=_keops_dtype(dt))
    vars = [Xtr, Xtr, Ytr] + other_vars
    kv = torch.empty_like(Ytr)
    o_kv = routine2(*vars, out=kv, backend=backend)
    grad = torch.autograd.grad(o_kv, other_vars, grad_outputs=v)
    print("My KeOps Gradient", grad)



""" OLD """


class FALKON_FP_MAP(torch.nn.Module):
    def __init__(self, nypoints, x_train, y_train, inner_lr):
        super().__init__()
        self.nypoints = nypoints
        self.M = nypoints.shape[0]
        self.N = x_train.shape[0]
        self.X = x_train
        self.Y = y_train
        self.inner_lr = inner_lr

    def run_map(self, params, hparams):
        penalty, sigma = hparams
        # Calculate preconditioner
        K_MM = self.kernel(self.nypoints, self.nypoints, sigma)
        T = torch.cholesky(K_MM, upper=True)
        A = torch.cholesky(
            (1 / self.M) * (T @ T.T) + penalty * torch.eye(self.M, dtype=self.X.dtype), upper=True)
        # Create kernel
        K_NM = self.kernel(self.X, self.nypoints, sigma)

        # Define preconditioned CG map
        def prec_mv(v):
            v = v[0]
            vv = torch.triangular_solve(v, A, transpose=False, upper=True)[0]
            return [torch.triangular_solve(
                torch.triangular_solve(
                    K_NM.T @ (K_NM @ torch.triangular_solve(vv, T)[0]),
                    T, transpose=True)[0] + penalty * self.N * vv, A, transpose=True)[0]]

        rhs = torch.triangular_solve(
            torch.triangular_solve(K_NM.T @ self.Y, T, transpose=True)[0], A, transpose=True)[0]
        betas = cg_torch.cg(prec_mv, [rhs], params, max_iter=1)[0]
        alphas = torch.triangular_solve(torch.triangular_solve(betas, A)[0], T)[0]
        return [alphas]

    def run_gd_map(self, params, hparams):
        penalty, sigma = hparams
        alpha = params[0]
        loss = torch.mean((alpha.T @ self.kernel(self.X, self.nypoints, sigma).T - self.Y) ** 2) + \
               penalty * alpha.T @ self.kernel(self.nypoints, self.nypoints, sigma) @ alpha
        return [alpha - self.inner_lr * torch.autograd.grad(loss, params, create_graph=True)[0]]

    def kernel(self, X1, X2, sigma):
        x_i = X1.unsqueeze(1)
        y_j = X2.unsqueeze(0)
        xmy = ((x_i - y_j) ** 2).sum(2)
        return torch.exp(xmy / (- 2 * sigma ** 2))
        # D = torch.norm(X1, p=2, dim=1, keepdim=True).pow_(2) + \
        #     torch.norm(X2, p=2, dim=1, keepdim=True).pow_(2).T - \
        #     2 * (X1 @ X2.T)
        # D = D / (-2 * sigma ** 2)
        # return torch.exp(D)

    def predict(self, params, hparams, X):
        penalty, sigma = hparams
        k = self.kernel(X, self.nypoints, sigma)
        return k @ params[0]


def test_fp_map():
    # Generate some data
    torch.manual_seed(2)
    n = 200
    d = 10

    w_oracle = torch.randn(d)
    X = torch.randn(n, d)
    Y = X @ w_oracle + 0.2 * torch.randn(n)
    Y = (Y > 0.).to(dtype=torch.float64)  # binary classification output
    Y = Y.reshape(-1, 1)
    X = X.to(dtype=torch.float64)

    Xtr = X[:n - 50]
    Xts = X[n - 50:]
    Ytr = Y[:n - 50]
    Yts = Y[n - 50:]

    sigma_init = torch.tensor(5., dtype=torch.float64, requires_grad=False)
    lambda_init = torch.tensor(0.00001, dtype=torch.float64, requires_grad=False)

    cg_fp_map = falkon_fp_map(Xtr[:10], Xtr, Ytr, FalkonOptions(keops_active="no"))

    prev_beta = torch.zeros(10, 1, dtype=torch.float64, requires_grad=False)
    all_params = [[torch.zeros(10, 1, dtype=torch.float64, requires_grad=False)]]
    cg_fp_map.set_hparams(hparams=[lambda_init, sigma_init])
    for i in range(10):
        new_params, prev_beta = cg_fp_map.run_map(params=prev_beta)
        preds = cg_fp_map.predict(new_params, Xts)
        err = torch.mean((preds - Yts) ** 2)
        print("ERROR", err)
        all_params.append(new_params)


def test_nkrr_ho_mixed_cg_gd():
    # Generate some data
    n = 2000
    d = 10
    M = 10
    Xtr, Ytr, Xts, Yts = gen_data(n, d)
    hparams = [torch.tensor(0.1, requires_grad=True, dtype=torch.float32),
               torch.tensor(3.0, requires_grad=True, dtype=torch.float32)]

    cg_map = falkon_fp_map(Xtr[:M].clone().detach(), Xtr, Ytr, FalkonOptions(keops_active="no"))
    gd_map = krr_gd_map(Xtr[:M].clone().detach(), Xtr, Ytr, 0.01, FalkonOptions(keops_active="no"))
    outer_opt = torch.optim.SGD(lr=1., momentum=0, params=hparams)

    def val_loss(params, hparams):
        preds = gd_map.predict(params, hparams, Xts)
        return torch.mean((preds - Yts) ** 2)

    outer_steps = 50
    for o_step in range(outer_steps):
        cg_map.set_hparams([hp.detach() for hp in hparams])
        opt_params = cg_map.run_inner_prob()
        loss = val_loss(opt_params, hparams)
        print("VAL LOSS", loss)
        print()
        print("Parameters at iteration %d: " % (o_step))
        print(opt_params)

        outer_opt.zero_grad()
        cg_torch.cg_hypergrad(opt_params, hparams, 10, gd_map.run_map, val_loss,
                              set_grad=True, stochastic=False)

        print("HYPER GRADIENTS", hparams[0].grad, hparams[1].grad)
        outer_opt.step()
        hparams[0].data.clamp_(min=1e-10)
        hparams[1].data.clamp_(min=1e-10)
        print("NEW HPARAMS", hparams)


def test_nkrr_ho_gd():
    # Generate some data
    n = 2000
    d = 10
    M = 10
    Xtr, Ytr, Xts, Yts = gen_data(n, d)

    hparams = [torch.tensor(0.1, requires_grad=True, dtype=torch.float32),
               torch.tensor(3.0, requires_grad=True, dtype=torch.float32)]

    gd_map = krr_gd_map(Xtr[:M].clone().detach(), Xtr, Ytr, 0.01, FalkonOptions(keops_active="no"))
    outer_opt = torch.optim.SGD(lr=1., momentum=0, params=hparams)

    def val_loss(params, hparams):
        preds = gd_map.predict(params, hparams, Xts)
        return torch.mean((preds - Yts) ** 2)

    outer_steps = 5
    inner_steps = 50
    for o_step in range(outer_steps):
        params = [torch.zeros(M, 1, dtype=torch.float32, requires_grad=True)]
        for t in range(inner_steps):
            params = gd_map.run_map(params=params, hparams=hparams)
        print("Parameters at iteration %d" % (o_step))
        print(params)

        loss = val_loss(params, hparams)
        print("VAL LOSS", loss)
        print()

        outer_opt.zero_grad()
        cg_torch.cg_hypergrad(params, hparams, 10, gd_map.run_map, val_loss,
                              set_grad=True, stochastic=False)

        print("HYPER GRADIENTS", hparams[0].grad, hparams[1].grad)
        outer_opt.step()
        hparams[0].data.clamp_(min=1e-10)
        hparams[1].data.clamp_(min=1e-10)
        print("NEW HPARAMS", hparams)


def run_inner(f, X, Y):
    f.fit(X, Y)
    return f


def solve_for_v(flk: falkon.Falkon, Xtr, Ytr, Xts, Yts, max_iter, opt):
    prec = flk.precond

    # LHS solver
    def mmv(sol):
        v = prec.invA(sol)
        cc = flk.kernel.dmmv(Xtr, flk.ny_points_, prec.invT(v), None, opt=opt)
        return prec.invAt(prec.invTt(cc) + flk.penalty * v)

    # RHS
    B = flk.kernel.dmmv(Xtr, flk.ny_points_, flk.alpha_, None, opt=opt) - \
        flk.kernel.mmv(flk.ny_points_, Xts, Yts, opt=opt) + \
        flk.kernel.mmv(flk.ny_points_, flk.ny_points_, flk.alpha_, opt=opt)

    optimizer = ConjugateGradient(opt.get_conjgrad_options())
    return optimizer.solve(None, B, mmv, max_iter, None)


def ker_grad_mmv(X, Y, v, kernel, opt):
    """
    If gamma
    """
    # Need the squared distances between X, Y (D_{ij} = ||x_i - y_j||^2
    # Then (D*K_{NM}) @ y

    K_nm = kernel(X, Y, opt=opt)
    D = torch.norm(X, p=2, dim=1, keepdim=True).pow_(2) + \
        torch.norm(Y, p=2, dim=1, keepdim=True).pow_(2).T - \
        2 * (X @ Y.T)
    return (D * K_nm) @ v


def run_outer(Xtr, Ytr, Xval, Yval, flk, opt):
    # First step: Solve the inner problem
    flk.fit(Xtr, Ytr)
    preds = flk.predict(Xval)
    err = torch.mean((preds - Yval) ** 2)
    print("ERROR: %.5f" % err)
    ker = flk.kernel
    oa = flk.alpha_  # Optimal alpha (for the inner problem)

    # Second step: Compute v using CG
    v = solve_for_v(flk, Xtr, Ytr, Xval, Yval, 10, opt=opt)

    # Third step compute the hyperparam gradients
    # i) mixed gradient of the inner problem
    i0 = ker.mmv(flk.ny_points_, flk.ny_points_, oa, opt=opt)
    i_lambda_full = (- i0).T @ v
    i1 = 2 * ker_grad_mmv(flk.ny_points_, Xtr, ker.mmv(Xtr, flk.ny_points_, oa, opt=opt), ker,
                          opt=opt)
    i2 = ker_grad_mmv(flk.ny_points_, Xtr, Ytr, ker, opt=opt)
    i3 = flk.penalty * ker_grad_mmv(flk.ny_points_, flk.ny_points_, oa, ker, opt=opt)
    i_gamma_full = (- i1 + i2 - i3).T @ v

    # ii) gradient of the validation loss wrt hparams
    ii_lambda_full = oa.T @ ker.mmv(flk.ny_points_, flk.ny_points_, oa, opt=opt)
    ii1 = 2 * oa.T @ ker_grad_mmv(flk.ny_points_, Xval, ker.mmv(Xval, flk.ny_points_, oa, opt=opt),
                                  ker, opt=opt)
    ii2 = 2 * oa.T @ ker_grad_mmv(flk.ny_points_, Xval, Yval, ker, opt=opt)
    ii3 = flk.penalty * oa.T @ ker_grad_mmv(flk.ny_points_, flk.ny_points_, oa, ker, opt=opt)
    ii_gamma_full = ii1 - ii2 + ii3

    # Combine i) and ii)
    full_grad = (
        i_gamma_full + ii_gamma_full,
        i_lambda_full + ii_lambda_full
    )
    return full_grad


def test_rnd():
    # Generate some data
    n = 200
    d = 10
    X = torch.randn(n, d)
    Y = (X[:, 0] ** 2 + X[:, 1] ** 3 + 3 - X[:, 3] + X[:, 4] + X[:, 5] ** 2 - X[:, 6] ** 4).reshape(
        -1, 1)

    Xtr = X[:n - 50]
    Xts = X[n - 50:]
    Ytr = Y[:n - 50]
    Yts = Y[n - 50:]

    sigma_init = 4
    lambda_init = 0.001

    sig, lam = sigma_init, lambda_init

    for i in range(10):
        ker = falkon.kernels.GaussianKernel(sig)
        opt = FalkonOptions(keops_active="no")
        flk: falkon.Falkon = falkon.Falkon(ker, penalty=lam, M=1000, options=opt)
        print()
        grads = run_outer(Xtr, Ytr, Xts, Yts, flk, opt=opt)
        print(grads)

        sig = sig - 0.01 * grads[0].item()
        lam = lam - 0.01 * grads[1].item()

        print(sig, lam)


def test_with_falkon():
    # Generate some data
    torch.manual_seed(2)
    n = 200
    d = 10

    w_oracle = torch.randn(d)
    X = torch.randn(n, d)
    Y = X @ w_oracle + 1 * torch.randn(n)
    Y = (Y > 0.).to(dtype=torch.float64)  # binary classification output
    Y = Y.reshape(-1, 1)
    X = X.to(dtype=torch.float64)

    Xtr = X[:n - 50]
    Xts = X[n - 50:]
    Ytr = Y[:n - 50]
    Yts = Y[n - 50:]

    def e_fn(labels, preds):
        return torch.mean((labels - preds) ** 2)

    ker = falkon.kernels.GaussianKernel(5)
    flk = falkon.Falkon(ker, 1e-10, 50, maxiter=10, error_every=1, error_fn=e_fn,
                        options=FalkonOptions(cg_tolerance=1e-13))
    flk.fit(Xtr, Ytr, Xts, Yts)
