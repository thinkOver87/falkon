"""from https://github.com/lrjconan/RBP/blob/9c6e68d1a7e61b1f4c06414fae04aeb43c8527cb/utils/model_helper.py"""
from typing import List, Callable

import torch


def cg(Ax, b, x0=None, max_iter=100, epsilon=1.0e-5):
    """ Conjugate Gradient
      Args:
        Ax: function, takes list of tensors as input
        b: list of tensors
      Returns:
        x_star: list of tensors
    """
    if x0 is None:
        x_last = [torch.zeros_like(bb) for bb in b]
        r_last = [torch.zeros_like(bb).copy_(bb) for bb in b]
    else:
        x_last = x0
        mmvs = Ax(x0)
        r_last = [bb - mmmvs for (bb, mmmvs) in zip(b, mmvs)]
    p_last = [torch.zeros_like(rr).copy_(rr) for rr in r_last]
    for ii in range(max_iter):
        Ap = Ax(p_last)
        Ap_vec = cat_list_to_tensor(Ap)
        p_last_vec = cat_list_to_tensor(p_last)
        r_last_vec = cat_list_to_tensor(r_last)
        rTr = torch.sum(r_last_vec * r_last_vec)
        pAp = torch.sum(p_last_vec * Ap_vec)
        alpha = rTr / pAp

        x = [xx + alpha * pp for xx, pp in zip(x_last, p_last)]
        r = [rr - alpha * pp for rr, pp in zip(r_last, Ap)]
        r_vec = cat_list_to_tensor(r)

        if float(torch.norm(r_vec)) < epsilon:
            break

        beta = torch.sum(r_vec * r_vec) / rTr
        p = [rr + beta * pp for rr, pp in zip(r, p_last)]

        x_last = x
        p_last = p
        r_last = r

    return x_last


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


def cg_hypergrad(params: List[torch.Tensor],
                 hparams: List[torch.Tensor],
                 K: int,
                 fp_map: Callable[[List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]],
                 outer_loss: Callable[[List[torch.Tensor], List[torch.Tensor]], torch.Tensor],
                 tol=1e-10,
                 set_grad=True,
                 stochastic=False) -> List[torch.Tensor]:
    """
     Computes the hypergradient by applying K steps of the conjugate gradient method (CG).
     It can end earlier when tol is reached.
     Args:
         params: the output of the inner solver procedure.
         hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
         K: the maximum number of conjugate gradient iterations
         fp_map: the fixed point map which defines the inner problem
         outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
         tol: end the method earlier when the norm of the residual is less than tol
         set_grad: if True set t.grad to the hypergradient for every t in hparams
         stochastic: set this to True when fp_map is not a deterministic function of its inputs
     Returns:
         the list of hypergradients for each element in hparams
     """
    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, hparams)

    def dfp_map_dw(xs):
        if stochastic:
            w_mapped_in = fp_map(params, hparams)
            Jfp_mapTv = torch.autograd.grad(w_mapped_in, params, grad_outputs=xs,
                                            retain_graph=False)
        else:
            Jfp_mapTv = torch.autograd.grad(w_mapped, params, grad_outputs=xs, retain_graph=True)
        return [v - j for v, j in zip(xs, Jfp_mapTv)]

    vs = cg(dfp_map_dw, grad_outer_w, max_iter=K, epsilon=tol)  # K steps of conjugate gradient

    if stochastic:
        w_mapped = fp_map(params, hparams)

    grads = torch.autograd.grad(w_mapped, hparams, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))
