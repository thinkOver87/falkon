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


class AbstractHypergradModule(abc.ABC):
    @abc.abstractmethod
    def val_loss(self, params, hparams):
        pass

    @abc.abstractmethod
    def param_derivative(self, params, hparams):
        pass

    @abc.abstractmethod
    def hessian_vector_product(self, params, first_derivative, vector):
        pass

    @abc.abstractmethod
    def mixed_vector_product(self, hparams, first_derivative, vector):
        pass

    @abc.abstractmethod
    def val_loss_grads(self, params, hparams):
        pass


class KRR(AbstractHypergradModule):
    def __init__(self, lr, Xtr, Ytr, Xts, Yts):
        super().__init__()
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Xts = Xts
        self.Yts = Yts
        self.lr = lr

    def inner_fp_map(self, params, hparams):
        alpha = params[0]
        penalty, sigma = hparams
        N = alpha.shape[0]
        K = self.kernel(self.Xtr, self.Xtr, sigma)

        # loss = torch.mean((K @ alpha - self.Ytr)**2) + penalty * alpha.T @ K @ alpha
        # update = torch.autograd.grad(loss, params)[0]
        Kalpha = K @ alpha
        update = (2 / N) * K @ (Kalpha - self.Ytr) + 2 * penalty * Kalpha
        return [alpha - self.lr * update]

    def val_loss(self, params, hparams):
        alpha = params[0]
        penalty, sigma = hparams

        Kts = self.kernel(self.Xts, self.Xtr, sigma)
        preds = Kts @ alpha
        return torch.mean((preds - self.Yts) ** 2)

    def tr_loss(self, params, hparams):
        alpha = params[0]
        penalty, sigma = hparams

        Ktr = self.kernel(self.Xtr, self.Xtr, sigma)
        preds = Ktr @ alpha
        return torch.mean((preds - self.Ytr) ** 2)

    def kernel(self, X1, X2, sigma):
        D = torch.norm(X1, p=2, dim=1, keepdim=True).pow_(2) + \
            torch.norm(X2, p=2, dim=1, keepdim=True).pow_(2).T - \
            2 * (X1 @ X2.T)
        D = D / (-2 * sigma ** 2)
        return torch.exp(D)

    def param_derivative(self, params, hparams):
        """Derivative of the training loss, with respect to the parameters"""
        alpha = params[0]
        penalty, sigma = hparams
        N = alpha.shape[0]
        K = self.kernel(self.Xtr, self.Xtr, sigma)

        # loss = torch.mean((K @ alpha - self.Ytr)**2) + penalty * alpha.T @ K @ alpha
        # update = torch.autograd.grad(loss, params)[0]
        Kalpha = K @ alpha
        update = (2 / N) * K @ (Kalpha - self.Ytr) + 2 * penalty * Kalpha
        return update
        # return [alpha - self.lr * update]

    def hessian_vector_product(self, params, first_derivative, vector):
        """
        Calculates the Hessian of the training loss (with respect to the parameters), multiplied
        by an arbitrary vector using the magic of autodiff.

        Parameters
        ----------
        params
        hparams
        first_derivative
            The derivative of the training loss, wrt params
        vector
            The vector to multiply the Hessian by
        """
        hvp = torch.autograd.grad(first_derivative, params, grad_outputs=vector, retain_graph=True)
        return hvp

    def mixed_vector_product(self, hparams, first_derivative, vector):
        """
        Calculates the mixed-derivative of the training loss, with respect to the parameters
        first (that derivative should be in the `first_derivative` parameter), and then
        with respect to the hyper-parameters; all multiplied by an arbitrary vector.

        Parameters
        ----------
        params
        hparams
        first_derivative
            The derivative of the training loss, wrt params
        vector
            The vector to multiply the derivative by
        """
        return torch.autograd.grad(first_derivative, hparams, grad_outputs=vector,
                                   allow_unused=True)

    def val_loss_grads(self, params, hparams):
        o_loss = self.val_loss(params, hparams)
        return (
            torch.autograd.grad(o_loss, params, allow_unused=True, create_graph=True,
                                retain_graph=True),
            torch.autograd.grad(o_loss, hparams, allow_unused=True, create_graph=True,
                                retain_graph=True)
        )


class FalkonHO(AbstractHypergradModule):
    def __init__(self, M, Xtr, Ytr, Xts, Yts):
        super().__init__()
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Xts = Xts
        self.Yts = Yts

        self.M = M
        self.opt = FalkonOptions(use_cpu=True)

        centers = UniformSelector(np.random.default_rng(10)).select(self.Xtr, None, M)
        self.center_selection = FixedSelector(centers)
        self.maxiter = 10

    def inner_opt(self, params, hparams):
        """This is NOT DIFFERENTIABLE"""
        alpha = params[0]
        penalty, sigma = hparams

        kernel = DiffGaussianKernel(sigma.detach(), self.opt)
        def sq_err(y_true, y_pred):
            return torch.mean((y_true - y_pred)**2)
        self.flk = falkon.Falkon(
            kernel,
            torch.exp(-penalty.detach()).item(),
            self.M,
            center_selection=self.center_selection,
            maxiter=self.maxiter,
            seed=129,
            options=self.opt)
        self.flk.fit(self.Xtr, self.Ytr, alpha=alpha.detach())
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
            torch.autograd.grad(o_loss, params, allow_unused=True, create_graph=True,
                                retain_graph=True),
            torch.autograd.grad(o_loss, hparams, allow_unused=True, create_graph=True,
                                retain_graph=True)
        )

    def param_derivative(self, params, hparams):
        """Derivative of the training loss, with respect to the parameters"""
        alpha = params[0]
        penalty, sigma = hparams
        N = self.Xtr.shape[0]
        ny_points = self.flk.ny_points_

        kernel = DiffGaussianKernel(sigma, self.opt)

        # 2/N * (K_MN(K_NM @ alpha - Y)) + 2*lambda*(K_MM @ alpha)
        out = 2 * ((1/N) * kernel.mmv(
            ny_points, self.Xtr, kernel.mmv(self.Xtr, ny_points, alpha, opt=self.opt) - self.Ytr, opt=self.opt) +
              torch.exp(-penalty) * kernel.mmv(ny_points, ny_points, alpha, opt=self.opt))
        return out

    def mixed_vector_product(self, hparams, first_derivative, vector):
        return torch.autograd.grad(first_derivative, hparams, grad_outputs=vector,
                                   allow_unused=True)

    def hessian_vector_product(self, params, first_derivative, vector):
        hvp = torch.autograd.grad(first_derivative, params, grad_outputs=vector,
                                  retain_graph=True)
        return hvp


def my_hypergrad(params: Sequence[torch.Tensor],
                 hparams: Sequence[torch.Tensor],
                 krr: AbstractHypergradModule,
                 cg_steps: int,
                 cg_tol: float = 1e-4,
                 set_grad: bool = True
                 ):
    params = [w.detach().requires_grad_(True) for w in params]
    grad_outer_params, grad_outer_hparams = krr.val_loss_grads(params, hparams)

    # Define a function which calculates the Hessian-vector product of the inner-loss
    first_diff = krr.param_derivative(params, hparams)
    # w_mapped = fp_map(params, hparams)
    # def hvp(vec):
    #     Jfp = torch.autograd.grad(w_mapped, params, grad_outputs=vec, retain_graph=True)
    #     return Jfp
    # Calculate the Hessian multiplied by the outer-gradient wrt alpha
    hvp = partial(krr.hessian_vector_product, params, first_diff)
    vs = cg(hvp, grad_outer_params, max_iter=cg_steps, epsilon=cg_tol)
    # Multiply the mixed inner gradient by `vs`
    grads = krr.mixed_vector_product(hparams, first_derivative=first_diff, vector=vs)
    # grads = torch.autograd.grad(first_diff, hparams, grad_outputs=vs, allow_unused=True)

    final_grads = []
    for ohp, g in zip(grad_outer_hparams, grads):
        if ohp is not None:
            final_grads.append(ohp - g)
        else:
            final_grads.append(-g)

    if set_grad:
        update_tensor_grads(hparams, final_grads)

    return {'val_loss': krr.val_loss(params, hparams), 'h_grads': final_grads}


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


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

    for o_step in range(outer_steps):
        # Run inner loop to get alpha_*
        params = [torch.zeros(M, 1, requires_grad=True, dtype=torch.float32)]
        params = flk_helper.inner_opt(params, hparams)

        outer_opt.zero_grad()
        hgrad_out = my_hypergrad(params, hparams, krr=flk_helper,
                                 cg_steps=hessian_cg_steps, cg_tol=hessian_cg_tol, set_grad=True)
        outer_opt.step()
        hparams[0].data.clamp_(min=1e-10)
        hparams[1].data.clamp_(min=1e-10)
        print("VAL LOSS", hgrad_out['val_loss'])
        print("GRADIENT", hgrad_out['h_grads'])
        print("NEW HPARAMS", hparams)
        print()
        hparam_history.append([h.detach().clone() for h in hparams])
        val_loss_history.append(hgrad_out['val_loss'])
        hgrad_history.append([g.detach() for g in hgrad_out['h_grads']])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(range(len(val_loss_history)), val_loss_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation loss")

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
