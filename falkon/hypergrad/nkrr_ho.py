import time
import itertools

import numpy as np
import torch
import torch.nn as nn

import falkon
from falkon.kernels.diff_rbf_kernel import DiffGaussianKernel
from falkon.center_selection import UniformSelector, FixedSelector, CenterSelector
from falkon.hypergrad.leverage_scores import subs_deff_simple


class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size, shuffle=False, drop_last=False, cuda=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.num_points = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.cuda = cuda

        n_batches, remainder = divmod(self.num_points, self.batch_size)
        if remainder > 0 and not drop_last:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.num_points)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        try:
            if self.i >= self.n_batches:  # This should handle drop_last correctly
                raise StopIteration()
        except AttributeError:
            raise RuntimeError("Make sure you make the tensor data-loader an iterator before iterating over it!")

        if self.indices is not None:
            indices = self.indices[self.i * self.batch_size : (self.i + 1) * self.batch_size]
            batch = tuple(t[indices] for t in self.tensors)
        else:
            batch = tuple(t[self.i * self.batch_size : (self.i + 1) * self.batch_size] for t in self.tensors)
        if self.cuda:
            batch = tuple(t.cuda() for t in batch)
        self.i += 1
        return batch

    def __len__(self):
        return self.n_batches


class NKRR(nn.Module):
    def __init__(self, sigma_init, penalty_init, centers_init, opt):
        super().__init__()
        penalty = nn.Parameter(torch.tensor(penalty_init, requires_grad=True))
        self.register_parameter('penalty', penalty)
        sigma = nn.Parameter(torch.tensor(sigma_init, requires_grad=True))
        self.register_parameter('sigma', sigma)
        centers = nn.Parameter(centers_init.requires_grad_())
        self.register_parameter('centers', centers)
        alpha = nn.Parameter(torch.zeros(centers_init.shape[0], 1, requires_grad=True))
        self.register_parameter('alpha', alpha)

        self.opt = opt

    def forward(self, X, Y):
        """
        l = 1/N ||K_{NM} @ a - Y|| + lambda * alpha.T @ K_{MM} @ alpha
        """
        k = DiffGaussianKernel(self.sigma, self.opt)

        preds = self.predict(X)
        loss = torch.mean((preds - Y)**2)
        reg = torch.exp(-self.penalty) * (self.alpha.T @ (k.mmv(self.centers, self.centers, self.alpha)))

        return (loss + reg), preds

    def predict(self, X):
        k = DiffGaussianKernel(self.sigma, self.opt)
        return k.mmv(X, self.centers, self.alpha)


class FLK_NKRR(nn.Module):
    def __init__(self, sigma_init, penalty_init, centers_init, opt, regularizer):
        super().__init__()
        penalty = nn.Parameter(torch.tensor(penalty_init, requires_grad=True))
        self.register_parameter('penalty', penalty)
        sigma = nn.Parameter(torch.tensor(sigma_init, requires_grad=True))
        self.register_parameter('sigma', sigma)
        centers = nn.Parameter(centers_init.requires_grad_())
        self.register_parameter('centers', centers)

        self.f_alpha = torch.zeros(centers_init.shape[0], 1, requires_grad=False)
        self.register_buffer('alpha', self.f_alpha)
        self.f_alpha_pc = torch.zeros(centers_init.shape[0], 1, requires_grad=False)
        self.register_buffer('alpha_pc', self.f_alpha_pc)

        self.opt = opt
        self.flk_maxiter = 10
        self.regularizer = regularizer

    def forward(self, X, Y):
        """
        l = 1/N ||K_{NM} @ a - Y|| + lambda * alpha.T @ K_{MM} @ alpha
        """
        k = DiffGaussianKernel(self.sigma, self.opt)

        preds = self.predict(X)
        loss = torch.mean((preds - Y)**2)
        pen = torch.exp(-self.penalty)
        if self.regularizer == "deff":
            #d_eff = subs_deff_simple(k, penalty=pen, X=X, J=self.centers)
            d_eff = subs_deff_simple(k, penalty=pen, X=self.centers, J=self.centers)
            #d_eff = subs_deff_simple(k, penalty=pen, X=X, J=X[:self.centers.shape[0]])
            reg = d_eff / X.shape[0]
        elif self.regularizer == "tikhonov":
            # This is the normal RKHS norm of the function
            reg = pen * (self.alpha.T @ (k.mmv(self.centers, self.centers, self.alpha)))
        else:
            raise ValueError("Regularizer %s not implemented" % (self.regularizer))

        return (loss + reg), preds

    def adapt_alpha(self, X, Y):
        k = DiffGaussianKernel(self.sigma.detach(), self.opt)
        if X.is_cuda:
            fcls = falkon.InCoreFalkon
        else:
            fcls = falkon.Falkon

        model = fcls(k,
                     torch.exp(-self.penalty).item(),
                     M=self.centers.shape[0],
                     center_selection=FixedSelector(self.centers.detach()),
                     maxiter=self.flk_maxiter,
                     options=self.opt)
        model.fit(X, Y, warm_start=self.alpha_pc)

        self.alpha = model.alpha_.detach()
        self.alpha_pc = model.beta_.detach()

    def predict(self, X):
        k = DiffGaussianKernel(self.sigma, self.opt)
        preds = k.mmv(X, self.centers, self.alpha)
        return preds


def nkrr_ho(Xtr, Ytr,
            Xts, Yts,
            num_epochs: int,
            sigma_type: str,
            sigma_init: float,
            penalty_init: float,
            falkon_centers: CenterSelector,
            falkon_M: int,
            hp_lr: float,
            p_lr: float,
            batch_size: int,
            cuda: bool,
            loss_every: int,
            err_fn,
            opt,
           ):
    # Choose start value for sigma
    if sigma_type == 'single':
        start_sigma = [sigma_init]
    elif sigma_type == 'diag':
        start_sigma = [sigma_init] * d
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    model = NKRR(
        start_sigma,
        penalty_init,
        falkon_centers.select(Xtr, Y=None, M=falkon_M),
        opt,
    )
    if cuda:
        model = model.cuda()

    opt_p = torch.optim.Adam([
        {"params": [model.alpha], "lr": p_lr},
    ])
    opt_hp = torch.optim.Adam([
        {"params": [model.sigma, model.penalty, model.centers], "lr": hp_lr},
    ])

    train_loader = FastTensorDataLoader(Xtr, Ytr, batch_size=batch_size, shuffle=True, drop_last=False, cuda=cuda)
    test_loader = FastTensorDataLoader(Xts, Yts, batch_size=batch_size, shuffle=False, drop_last=False, cuda=cuda)

    for epoch in range(num_epochs):
        train_loader = iter(train_loader)
        model.train()
        e_start = time.time()

        running_error = 0
        samples_processed = 0
        try:
            for i in itertools.count(0):
                b_tr_x, b_tr_y = next(train_loader)
                samples_processed += b_tr_x.shape[0]

                opt_p.zero_grad()
                opt_hp.zero_grad()
                loss, preds = model(b_tr_x, b_tr_y)
                loss.backward()
                opt_p.step()
                opt_hp.step()

                err, err_name = err_fn(b_tr_y.detach().cpu(), preds.detach().cpu())
                running_error += err * b_tr_x.shape[0]
                if i % loss_every == (loss_every - 1):
                    print(f"step {i} - {err_name} {running_error / samples_processed}")
        except StopIteration:
            t_elapsed = time.time() - e_start  # Stop the time

            model.eval()
            test_loader = iter(test_loader)
            test_preds = []
            try:
                while True:
                    b_ts_x, _ = next(test_loader)
                    test_preds.append(model.predict(b_ts_x))
            except StopIteration:
                test_preds = torch.cat(test_preds)
                test_err, err_name = err_fn(Yts.detach().cpu(), test_preds.detach().cpu())
            print(f"Epoch {epoch} ({t_elapsed:5.2f}s) - Tr {err_name} = {running_error / samples_processed:6.5f} , Ts {err_name} = {test_err:6.5f} -- Sigma {model.sigma.item():.3f} - Penalty {np.exp(-model.penalty.item()):e}")


def flk_nkrr_ho(Xtr, Ytr,
                Xts, Yts,
                num_epochs: int,
                sigma_type: str,
                sigma_init: float,
                penalty_init: float,
                falkon_centers: CenterSelector,
                falkon_M: int,
                hp_lr: float,
                p_lr: float,  # Only for signature compatibility
                batch_size: int,
                cuda: bool,
                loss_every: int,
                err_fn,
                opt,
                regularizer: str,
               ):
    print("Starting Falkon-NKRR-HO optimization.")
    print(f"{num_epochs} epochs - {sigma_type} sigma ({sigma_init}) - penalty ({penalty_init}) - {falkon_M} centers. HP-LR={hp_lr} - batch {batch_size}")
    print(f"{regularizer} regularizer")
    # Choose start value for sigma
    if sigma_type == 'single':
        start_sigma = [sigma_init]
    elif sigma_type == 'diag':
        start_sigma = [sigma_init] * d
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    model = FLK_NKRR(
        start_sigma,
        penalty_init,
        falkon_centers.select(Xtr, Y=None, M=falkon_M),
        opt,
        regularizer,
    )
    if cuda:
        model = model.cuda()

    opt_hp = torch.optim.Adam([
        {"params": [model.sigma, model.penalty, model.centers], "lr": hp_lr},
    ])

    train_loader = FastTensorDataLoader(Xtr, Ytr, batch_size=batch_size, shuffle=True, drop_last=False, cuda=cuda)
    test_loader = FastTensorDataLoader(Xts, Yts, batch_size=batch_size, shuffle=False, drop_last=False, cuda=cuda)

    for epoch in range(num_epochs):
        train_loader = iter(train_loader)
        model.train()
        e_start = time.time()

        running_error = 0
        samples_processed = 0
        try:
            for i in itertools.count(0):
                b_tr_x, b_tr_y = next(train_loader)
                samples_processed += b_tr_x.shape[0]

                opt_hp.zero_grad()
                loss, preds = model(b_tr_x, b_tr_y)
                loss.backward()
                print("centers has NaN gradients", torch.isnan(model.centers.grad).sum().item())
                print("sigma has NaN gradients", torch.isnan(model.sigma.grad).sum().item())

                # Change w
                model.adapt_alpha(b_tr_x, b_tr_y)
                # Change theta
                opt_hp.step()

                preds = model.predict(b_tr_x)  # Redo predictions to check adapted model

                err, err_name = err_fn(b_tr_y.detach().cpu(), preds.detach().cpu())
                running_error += err * preds.shape[0]
                if i % loss_every == (loss_every - 1):
                    print(f"step {i} - {err_name} {running_error / samples_processed}")
                    running_error = 0
                    samples_processed = 0

        except StopIteration:
            t_elapsed = time.time() - e_start  # Stop the time

            model.eval()
            test_loader = iter(test_loader)
            test_preds = []
            try:
                while True:
                    b_ts_x, _ = next(test_loader)
                    test_preds.append(model.predict(b_ts_x))
            except StopIteration:
                test_preds = torch.cat(test_preds)
                test_err, err_name = err_fn(Yts.detach().cpu(), test_preds.detach().cpu())
            print(f"Epoch {epoch} ({t_elapsed:5.2f}s) - Tr {err_name} = {running_error / samples_processed:6.5f} , Ts {err_name} = {test_err:6.5f} -- Sigma {model.sigma.item():.3f} - Penalty {np.exp(-model.penalty.item()):e}")


def flk_nkrr_ho_val(Xtr, Ytr,
                    Xts, Yts,
                    num_epochs: int,
                    sigma_type: str,
                    sigma_init: float,
                    penalty_init: float,
                    falkon_centers: CenterSelector,
                    falkon_M: int,
                    hp_lr: float,
                    p_lr: float,  # Only for signature compatibility
                    batch_size: int,
                    cuda: bool,
                    loss_every: int,
                    err_fn,
                    opt,
                   ):
    print("Starting Falkon-NKRR-HO optimization.")
    print(f"{num_epochs} epochs - {sigma_type} sigma ({sigma_init}) - penalty ({penalty_init}) - {falkon_M} centers. HP-LR={hp_lr} - batch {batch_size}")
    # Choose start value for sigma
    if sigma_type == 'single':
        start_sigma = [sigma_init]
    elif sigma_type == 'diag':
        start_sigma = [sigma_init] * d
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    model = FLK_NKRR(
        start_sigma,
        penalty_init,
        falkon_centers.select(Xtr, Y=None, M=falkon_M),
        opt,
    )
    if cuda:
        model = model.cuda()

    opt_hp = torch.optim.Adam([
        {"params": [model.sigma, model.penalty, model.centers], "lr": hp_lr},
    ])

    n_tr_samples = int(Xtr.shape[0] * 1.0)
    train_loader = FastTensorDataLoader(Xtr[:n_tr_samples], Ytr[:n_tr_samples], batch_size=batch_size, shuffle=True, drop_last=False, cuda=cuda)
    val_loader = FastTensorDataLoader(Xtr[n_tr_samples:], Ytr[n_tr_samples:], batch_size=batch_size, shuffle=True, drop_last=False, cuda=cuda)
    test_loader = FastTensorDataLoader(Xts, Yts, batch_size=batch_size, shuffle=False, drop_last=False, cuda=cuda)

    for epoch in range(num_epochs):
        train_loader = iter(train_loader)
        val_loader = iter(val_loader)
        model.train()
        e_start = time.time()

        running_error = 0
        samples_processed = 0
        try:
            for i in itertools.count(0):
                b_tr_x, b_tr_y = next(train_loader)
                b_vl_x, b_vl_y = next(val_loader)
                samples_processed += b_vl_x.shape[0]

                # Inner opt with train
                # Outer (hp) opt with validation
                opt_hp.zero_grad()
                loss, preds = model(b_vl_x, b_vl_y)
                loss.backward()
                opt_hp.step()
                model.adapt_alpha(b_tr_x, b_tr_y)

                preds = model.predict(b_vl_x)  # Redo predictions to check adapted model

                err, err_name = err_fn(b_vl_y.detach().cpu(), preds.detach().cpu())
                running_error += err * preds.shape[0]
                if i % loss_every == (loss_every - 1):
                    print(f"step {i} - {err_name} {running_error / samples_processed}")
        except StopIteration:
            t_elapsed = time.time() - e_start  # Stop the time

            model.eval()
            test_loader = iter(test_loader)
            test_preds = []
            try:
                while True:
                    b_ts_x, _ = next(test_loader)
                    test_preds.append(model.predict(b_ts_x))
            except StopIteration:
                test_preds = torch.cat(test_preds)
                test_err, err_name = err_fn(Yts.detach().cpu(), test_preds.detach().cpu())
            print(f"Epoch {epoch} ({t_elapsed:5.2f}s) - Tr {err_name} = {running_error / samples_processed:6.5f} , Ts {err_name} = {test_err:6.5f} -- Sigma {model.sigma.item():.3f} - Penalty {np.exp(-model.penalty.item()):e}")

